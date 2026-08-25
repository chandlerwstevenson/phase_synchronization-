"""Open-loop N-node phase sync over configurable measurement graphs.

Generalizes the mesh harness (hybrid_calibration/mesh.py run_dfpc_mesh)
to an arbitrary edge list with per-edge measurement type:

  "two"  a dedicated two-way exchange on that edge: the half-difference
         of the two directions cancels the propagation phase, giving an
         unbiased (mod pi) measurement of the relative oscillator phase
  "one"  a single one-way capture: the measured phase is the relative
         oscillator phase PLUS the unknown propagation phase of that
         edge's channel (no separation possible from this edge alone)

Master-free open loop: every edge runs its own 2-state Kalman filter on
its own measurements and its endpoints apply degree-weighted symmetric
corrections (the classical consensus law, held fixed across topologies
so graph comparisons are not confounded by the control law). Node 0 is
a bookkeeping gauge reference only.

Trap avoidance (this project retracted four artifact findings):
  - per-node initial frequency offsets are RANDOMIZED per seed
    (uniform, no arithmetic grid - the CFO-grid aliasing trap)
  - run lengths cover >= 4 cycles of the sparsest service cadence
  - the length-blind calibration cache is not used at all here
  - every campaign records seeds >= 3 with spread

PREDICTIONS, stated before the campaigns were run (see also
RESULTS_topology.md, written in the same order):

  Exp 1 (identifiability): all-two-way trees and rings hold bounded
    relative phases. One-way-only graphs: frequency still locks (a
    static channel adds no frequency), but each pair's phase settles at
    MINUS its channel phase - stable radian-scale biases in a static
    channel; under environment motion those biases track the channel,
    i.e. the array's internal phase wanders with the environment. The
    directive's coarser prediction "drifts" is refined to: "channel-
    valued bias; drifts if and only if the environment drifts." Adding
    a single two-way edge fixes only that edge's pair; distant pairs
    stay biased.
  Exp 2 (accuracy vs topology): steady phase-error variance between
    nodes i and j should track the graph's effective resistance
    between i and j (chain end-to-end worst, complete graph best).
  Exp 3 (scaling): chain end-to-end deviation std ~ sqrt(hops)
    (series resistance), ring antipodal ~ sqrt(N)/2. Consensus
    convergence time grows ~ N^2 on chains (spectral gap), so large-N
    cells may not reach steady state in bounded runs - reported
    honestly if so, as a known property, not a discovery.
  Exp 4 (branch states, two-way ring, adverse acquisition, branch
    check disabled): steady states have each edge near 0 or pi with an
    EVEN number of pi-edges around the cycle (physical consistency:
    edge relative phases sum to 0 mod 2pi); multiple distinct states
    across seeds.

Usage:
    python openloop_topology_study.py --part smoke
    python openloop_topology_study.py --part exp1
    python openloop_topology_study.py --part exp2
    python openloop_topology_study.py --part exp3 [--big]
    python openloop_topology_study.py --part exp4
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sionna.phy import config as sionna_config

from ota_sync.coherent import _pick_half_phase
from ota_sync.core import (
    REAL_DTYPE,
    Oscillator,
    PhaseFrequencyEKF,
    resolve_device,
    wrap_phase,
)
from ota_sync.network import MAX_LINK_SNR_DB, place_stations
from ota_sync.sdr import (
    SDRRadioLink,
    SDRSimulationConfig,
    SDRSynchronizer,
    _FlickerFrequencyNoise,
    _measurement_covariance,
    _quantize_correction,
    make_sync_preamble,
)

HERE = Path(__file__).resolve().parent
CACHE = HERE / "openloop_topology_cache.json"


# ---------------------------------------------------------------------
# topologies
# ---------------------------------------------------------------------

def chain_edges(n, kind="two"):
    return [(i, i + 1, kind) for i in range(n - 1)]


def ring_edges(n, kind="two"):
    return chain_edges(n, kind) + [(n - 1, 0, kind)]


def star_edges(n, kind="two"):
    return [(0, i, kind) for i in range(1, n)]


def complete_edges(n, kind="two"):
    return [(i, j, kind) for i in range(n) for j in range(i + 1, n)]


def random_connected_edges(n, extra, seed, kind="two"):
    """Random spanning tree plus `extra` random chords."""

    rng = np.random.default_rng(seed)
    nodes = list(rng.permutation(n))
    edges = set()
    for k in range(1, n):
        attach = nodes[rng.integers(0, k)]
        edges.add((min(nodes[k], attach), max(nodes[k], attach)))
    while len(edges) < n - 1 + extra:
        i, j = rng.integers(0, n, 2)
        if i != j:
            edges.add((min(i, j), max(i, j)))
    return [(p, q, kind) for p, q in sorted(edges)]


def bfs_subtrees(n, edge_spec):
    """BFS spanning tree from node 0: returns (tree_edge_set oriented
    parent->child, subtree_nodes[child]) and the set of chord edges.
    Used by the branch check: flipping child's whole subtree preserves
    every other TREE edge's relative phase (chords crossing the cut
    shift by pi - physically real, monitored, not checkable mod pi)."""

    adjacency = {i: [] for i in range(n)}
    for p, q, _ in edge_spec:
        adjacency[p].append(q)
        adjacency[q].append(p)
    parent = {0: None}
    order = [0]
    queue = [0]
    while queue:
        node = queue.pop(0)
        for nxt in adjacency[node]:
            if nxt not in parent:
                parent[nxt] = node
                order.append(nxt)
                queue.append(nxt)
    tree_pairs = set()
    for child, par in parent.items():
        if par is not None:
            tree_pairs.add((min(par, child), max(par, child)))
    subtree = {}
    for child in parent:
        if parent[child] is None:
            continue
        members = {child}
        frontier = [child]
        while frontier:
            node = frontier.pop()
            for nxt in adjacency[node]:
                if parent.get(nxt) == node and nxt not in members:
                    members.add(nxt)
                    frontier.append(nxt)
        subtree[child] = members
    return tree_pairs, parent, subtree


def effective_resistance(n, edges):
    """Pairwise effective resistance of the unweighted graph."""

    lap = np.zeros((n, n))
    for p, q, _ in edges:
        lap[p, p] += 1.0
        lap[q, q] += 1.0
        lap[p, q] -= 1.0
        lap[q, p] -= 1.0
    pinv = np.linalg.pinv(lap)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            out[i, j] = pinv[i, i] + pinv[j, j] - 2.0 * pinv[i, j]
    return out


# ---------------------------------------------------------------------
# circular statistics
# ---------------------------------------------------------------------

def circ_mean_std(values: torch.Tensor) -> tuple[float, float]:
    phasors = torch.exp(1j * values.to(torch.complex128))
    mean_ph = torch.mean(phasors)
    resultant = torch.abs(mean_ph).item()
    mean = float(torch.angle(mean_ph).item())
    if resultant <= 1e-12:
        return mean, float("inf")
    std = math.sqrt(max(-2.0 * math.log(min(resultant, 1.0)), 0.0))
    return mean, std


# ---------------------------------------------------------------------
# the open-loop graph run
# ---------------------------------------------------------------------

def run_openloop_graph(
    settings: SDRSimulationConfig,
    num_nodes: int,
    edge_spec: list[tuple[int, int, str]],
    *,
    budget_edges_per_interval: int | None = None,
    acquisition_intervals: int = 10,
    branch_check: bool = True,
    init_phase_span: float = 0.5,
    init_cfo_span_hz: float = 1500.0,
    radius_m: float = 500.0,
    path_loss_exponent: float = 2.7,
    reference_distance_m: float = 500.0,
):
    """Run the graph; return per-node traces and per-edge residuals.

    Returns dict with:
      node_traces  (num_nodes, intervals) wrap(theta_i - theta_0)
      edge_means   list of steady circular mean per edge
      detect_rate  fraction of serviced captures detected
      wall_s       wall-clock seconds
    """

    t0 = time.time()
    device = resolve_device(settings.device)
    torch.manual_seed(settings.seed)
    sionna_config.seed = settings.seed
    generator = torch.Generator(device=device)
    generator.manual_seed(settings.seed + 1)
    init_rng = np.random.default_rng(settings.seed + 977)

    positions = place_stations(num_nodes, radius_m, settings.seed)
    dt = settings.sync_interval
    dt_samples = int(round(dt * settings.sample_rate))

    frequency_process_std = 2.0 * math.pi * settings.frequency_process_std_hz
    interval_covariance = torch.diag(
        torch.tensor(
            [settings.phase_process_std_rad**2, frequency_process_std**2],
            dtype=REAL_DTYPE,
            device=device,
        )
    )
    # Randomized (non-grid) initial conditions - the aliasing trap.
    oscillators = []
    for index in range(num_nodes):
        phase0 = float(init_rng.uniform(-init_phase_span, init_phase_span))
        cfo0 = float(init_rng.uniform(-init_cfo_span_hz, init_cfo_span_hz))
        oscillators.append(
            Oscillator(
                phase0,
                2.0 * math.pi * cfo0,
                dt,
                interval_covariance,
                device,
                generator,
            )
        )
    flickers = [
        _FlickerFrequencyNoise(
            settings.flicker_frequency_std_hz,
            dt,
            settings.num_iterations * settings.sync_interval,
            device,
            generator,
        )
        for _ in range(num_nodes)
    ]
    flicker_previous = [
        torch.zeros((), dtype=REAL_DTYPE, device=device)
        for _ in range(num_nodes)
    ]

    degree = {i: 0 for i in range(num_nodes)}
    for p, q, _ in edge_spec:
        degree[p] += 1
        degree[q] += 1

    preamble = make_sync_preamble(settings, device)
    white_fm_interval = settings.phase_noise_std_rad**2 * dt_samples
    edges = []
    for p, q, kind in edge_spec:
        distance = max(float(np.linalg.norm(positions[p] - positions[q])), 1.0)
        snr_db = min(
            settings.snr_db
            - 10.0 * path_loss_exponent
            * math.log10(distance / reference_distance_m),
            MAX_LINK_SNR_DB,
        )
        edge_settings = replace(settings, snr_db=snr_db)
        link_fwd = SDRRadioLink(edge_settings, preamble, device, generator)
        link_rev = SDRRadioLink(
            edge_settings, preamble, device, generator, mirror_of=link_fwd
        )
        oneway_noise = _measurement_covariance(edge_settings, preamble, device)
        measurement_noise = (
            0.5 * oneway_noise if kind == "two" else oneway_noise
        )
        ekf = PhaseFrequencyEKF(
            dt,
            2.0 * interval_covariance
            + torch.diag(
                torch.tensor(
                    [
                        white_fm_interval,
                        2.0 * flickers[0].innovation_variance,
                    ],
                    dtype=REAL_DTYPE,
                    device=device,
                )
            ),
            measurement_noise,
            device,
            initial_covariance=torch.diag(
                torch.tensor(
                    [math.pi**2, (2.0 * math.pi * 50e3) ** 2],
                    dtype=REAL_DTYPE,
                    device=device,
                )
            ),
        )
        edges.append(
            {
                "pair": (p, q),
                "kind": kind,
                "snr_db": snr_db,
                "link_fwd": link_fwd,
                "link_rev": link_rev,
                "synchronizer": SDRSynchronizer(edge_settings, preamble),
                "ekf": ekf,
                "acquired": False,
                "pending": {},
                "loaded": False,
            }
        )

    node_corr_freq = [
        torch.zeros((), dtype=REAL_DTYPE, device=device)
        for _ in range(num_nodes)
    ]
    chain_bias = math.radians(settings.twoway_chain_asymmetry_deg)
    per_node_walk = settings.phase_noise_std_rad / math.sqrt(2.0)
    capture_samples = edges[0]["link_fwd"].input_length + edges[0]["link_fwd"].l_tot - 1
    tree_pairs, _, subtree_of = bfs_subtrees(num_nodes, edge_spec)
    flip_count = 0
    realign_count = 0

    node_rows: list[list[torch.Tensor]] = [[] for _ in range(num_nodes)]
    freq_rows: list[list[torch.Tensor]] = [[] for _ in range(num_nodes)]
    serviced_count = 0
    detected_count = 0
    rr_pointer = 0

    for iteration in range(settings.num_iterations):
        for index in range(num_nodes):
            oscillators[index].step()
            now = flickers[index].step()
            oscillators[index].state[1] = (
                oscillators[index].state[1] + (now - flicker_previous[index])
            )
            flicker_previous[index] = now

        # Load due corrections (symmetric degree-weighted, side channel).
        node_delta = [
            torch.zeros(2, dtype=REAL_DTYPE, device=device)
            for _ in range(num_nodes)
        ]
        any_loaded = False
        node_incident = [0] * num_nodes
        for edge in edges:
            due = edge["pending"].pop(iteration, None)
            if due is None:
                continue
            p, q = edge["pair"]
            # Half/half strength per correction; a node receiving
            # several corrections in one interval applies their MEAN
            # (normalization below) - the standard consensus rule.
            # Plain summation overshoots high-degree hubs (star hub:
            # 7 simultaneous half-corrections, x3.5 overshoot,
            # measured unstable in pre-campaign validation).
            node_delta[p] = node_delta[p] - due / 2.0
            node_delta[q] = node_delta[q] + due / 2.0
            node_incident[p] += 1
            node_incident[q] += 1
            edge["loaded"] = True
            any_loaded = True
        for index in range(num_nodes):
            if node_incident[index] > 1:
                node_delta[index] = node_delta[index] / node_incident[index]
        if any_loaded:
            for index in range(num_nodes):
                if torch.any(node_delta[index] != 0.0):
                    oscillators[index].apply_correction(node_delta[index])
                    node_corr_freq[index] = (
                        node_corr_freq[index] + node_delta[index][1]
                    )
            for edge in edges:
                p, q = edge["pair"]
                relative = node_delta[p] - node_delta[q]
                if torch.any(relative != 0.0):
                    edge["ekf"].reset_after_correction(-relative)

        # Periodic 1-bit branch check on two-way TREE edges, gated
        # until after acquisition (frequency must settle first, or the
        # check fires on transient spin and injects pi shifts). Flips
        # move the child's whole BFS subtree so every other tree edge's
        # relative phase is preserved (the mesh.py lesson). Chord edges
        # are not checkable this way; their branch state is monitored,
        # not corrected (that residual ambiguity is exp 4's subject).
        if (
            branch_check
            and iteration >= acquisition_intervals
            and iteration % 4 == 0
        ):
            for edge in edges:
                if edge["kind"] != "two" or not edge["loaded"]:
                    continue
                p, q = edge["pair"]
                if (min(p, q), max(p, q)) not in tree_pairs:
                    continue
                # Fire only on the silent-wrong-lock signature: the
                # filter is confident near zero while the truth is
                # anti-phase. Flipping an honest excursion the filter
                # already knows about would CREATE a pi error.
                filter_believes_locked = (
                    abs(wrap_phase(edge["ekf"].state[0]).item()) < 0.5
                )
                if filter_believes_locked and torch.cos(
                    oscillators[p].state[0] - oscillators[q].state[0]
                ) < -0.6:
                    child = q if q in subtree_of and p not in subtree_of.get(
                        q, set()
                    ) else p
                    flip = torch.tensor(
                        [math.pi, 0.0], dtype=REAL_DTYPE, device=device
                    )
                    for member in subtree_of.get(child, {child}):
                        oscillators[member].apply_correction(flip)
                    flip_count += 1
                    # Flip-storm escape: mod-pi phase sampled every T is
                    # invariant under frequency errors of k/(2T), so a
                    # marginal acquisition can lock a frequency ALIAS in
                    # which the true phase advances ~pi per interval and
                    # the check fires repeatedly. Two flips of the same
                    # edge within 4 intervals => discard the filter and
                    # re-acquire (what a real modem does). Counted as
                    # "realign" events, reported.
                    last = edge.get("last_flip", -10)
                    if iteration - last <= 4:
                        edge["acquired"] = False
                        edge["ekf"].state = torch.zeros(
                            2, dtype=REAL_DTYPE, device=device
                        )
                        realign_count += 1
                    edge["last_flip"] = iteration

        # Which edges get serviced this interval.
        if (
            budget_edges_per_interval is None
            or iteration < acquisition_intervals
        ):
            serviced = list(range(len(edges)))
        else:
            serviced = [
                (rr_pointer + k) % len(edges)
                for k in range(budget_edges_per_interval)
            ]
            rr_pointer = (
                rr_pointer + budget_edges_per_interval
            ) % len(edges)

        for edge_index in serviced:
            edge = edges[edge_index]
            p, q = edge["pair"]
            physical_p = oscillators[p].state[1] - node_corr_freq[p]
            physical_q = oscillators[q].state[1] - node_corr_freq[q]
            if settings.sample_clock_offset_ppm is not None:
                sfo = settings.sample_clock_offset_ppm
            else:
                sfo = float(
                    (physical_q - physical_p).item()
                    / (2.0 * math.pi * settings.carrier_frequency_hz)
                    * 1e6
                )
            capture_fwd = edge["link_fwd"].capture(
                oscillators[p], oscillators[q], iteration, sfo
            )
            oscillators[p].state[0] = wrap_phase(
                oscillators[p].state[0] + capture_fwd.lo_walk_end
            )
            forward = edge["synchronizer"].estimate(capture_fwd.samples)
            if edge["kind"] == "two":
                capture_rev = edge["link_rev"].capture(
                    oscillators[q], oscillators[p], iteration, -sfo
                )
                oscillators[q].state[0] = wrap_phase(
                    oscillators[q].state[0] + capture_rev.lo_walk_end
                )
                reverse = edge["synchronizer"].estimate(capture_rev.samples)
                detected = forward.detected and reverse.detected
            else:
                detected = bool(forward.detected)
            serviced_count += 1
            detected_count += int(detected)
            if not detected:
                continue
            ekf = edge["ekf"]
            ekf.predict()
            if edge["kind"] == "two":
                frequency_obs = (forward.frequency - reverse.frequency) / 2.0
                # TDD-turnaround compensation (the production star does
                # this; the mesh lineage omitted it): the pair phase
                # advances 2*pi*f_rel*tau between the two directions,
                # biasing the half-difference by pi*f_rel*tau — 2.4 rad
                # at 750 Hz offset, enough to corrupt the branch.
                combined_half = wrap_phase(
                    wrap_phase(forward.phase - reverse.phase) / 2.0
                    + chain_bias
                    - frequency_obs * settings.tdd_turnaround_s / 2.0
                )
                if not edge["acquired"]:
                    theta_obs = _pick_half_phase(
                        combined_half, torch.zeros_like(combined_half)
                    )
                    ekf.state = torch.stack((theta_obs, frequency_obs))
                    ekf.covariance = torch.diag(
                        torch.stack(
                            (
                                ekf.measurement_covariance[0, 0],
                                ekf.measurement_covariance[2, 2],
                            )
                        )
                    )
                    edge["acquired"] = True
                else:
                    theta_obs = _pick_half_phase(
                        combined_half, wrap_phase(ekf.state[0])
                    )
                    ekf.update(
                        torch.stack(
                            (
                                torch.cos(theta_obs),
                                torch.sin(theta_obs),
                                frequency_obs,
                            )
                        )
                    )
            else:
                theta_obs = wrap_phase(forward.phase)
                frequency_obs = forward.frequency
                if not edge["acquired"]:
                    ekf.state = torch.stack(
                        (theta_obs, frequency_obs)
                    )
                    ekf.covariance = torch.diag(
                        torch.stack(
                            (
                                ekf.measurement_covariance[0, 0],
                                ekf.measurement_covariance[2, 2],
                            )
                        )
                    )
                    edge["acquired"] = True
                else:
                    ekf.update(
                        torch.stack(
                            (
                                torch.cos(theta_obs),
                                torch.sin(theta_obs),
                                frequency_obs,
                            )
                        )
                    )
            edge.setdefault("debug_obs", []).append(
                (
                    iteration,
                    float(theta_obs.item()),
                    float(
                        wrap_phase(
                            oscillators[edge["pair"][0]].state[0]
                            - oscillators[edge["pair"][1]].state[0]
                        ).item()
                    ),
                )
            )
            # Correction issuance: full-rate while the edge settles
            # (residual frequency must shrink fast or the pair phase
            # slews across the mod-pi fold between corrections and the
            # branch migrates); parity-staggered (Gauss-Seidel) once
            # settled. Sparse budgets are already staggered.
            edge["settle_count"] = edge.get("settle_count", 0) + 1
            stagger = (
                budget_edges_per_interval is None
                and edge["settle_count"] > 12
            )
            if (not stagger) or (iteration + 1) % 2 == edge_index % 2:
                predicted = ekf.transition @ ekf.state
                edge["pending"][iteration + 1] = _quantize_correction(
                    predicted, settings
                )

        # Dead-time walk.
        used = capture_samples * (2 if edges else 0)
        remainder = max(0, dt_samples - used)
        if per_node_walk > 0.0 and remainder > 0:
            for index in range(num_nodes):
                oscillators[index].state[0] = wrap_phase(
                    oscillators[index].state[0]
                    + torch.randn(
                        (), dtype=REAL_DTYPE, device=device,
                        generator=generator,
                    )
                    * per_node_walk
                    * math.sqrt(remainder)
                )

        for index in range(num_nodes):
            node_rows[index].append(
                wrap_phase(
                    oscillators[index].state[0] - oscillators[0].state[0]
                ).clone()
            )
            freq_rows[index].append(
                (
                    (oscillators[index].state[1] - oscillators[0].state[1])
                    / (2.0 * math.pi)
                ).clone()
            )

    node_traces = torch.stack(
        [torch.stack(row).detach().cpu() for row in node_rows]
    )
    freq_traces = torch.stack(
        [torch.stack(row).detach().cpu() for row in freq_rows]
    )
    edge_means = []
    steady = slice(settings.num_iterations // 2, settings.num_iterations)
    for edge in edges:
        p, q = edge["pair"]
        diff = wrap_phase(node_traces[p, steady] - node_traces[q, steady])
        mean, std = circ_mean_std(diff)
        edge_means.append({"pair": [p, q], "kind": edge["kind"],
                           "mean": mean, "std": std})
    return {
        "node_traces": node_traces,
        "freq_traces": freq_traces,
        "edge_means": edge_means,
        "detect_rate": detected_count / max(serviced_count, 1),
        "flips": flip_count,
        "realigns": realign_count,
        "debug_obs": [edge.get("debug_obs", []) for edge in edges],
        "wall_s": time.time() - t0,
    }


# ---------------------------------------------------------------------
# campaign helpers
# ---------------------------------------------------------------------

def pair_stats(node_traces: torch.Tensor, window: slice):
    """Circular (mean, std) for every node pair over the window."""

    n = node_traces.shape[0]
    stats = {}
    for i in range(n):
        for j in range(i + 1, n):
            diff = wrap_phase(node_traces[i, window] - node_traces[j, window])
            stats[f"{i}-{j}"] = circ_mean_std(diff)
    return stats


def window_halves(num_iterations: int):
    steady_start = num_iterations // 2
    mid = (steady_start + num_iterations) // 2
    return slice(steady_start, mid), slice(mid, num_iterations)


def load_cache():
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return {}


def save_cache(data):
    CACHE.write_text(json.dumps(data, indent=1))


def base_settings(seed, iterations, speed=0.0, profile=None):
    """TCXO-class oscillator noise by default (realistic small-cell
    hardware, via the repo's own datasheet-anchored profiles). The
    legacy SDR-class noise chases at 0.2-0.4 rad on interior edges,
    which buries topology structure under control jitter."""

    from ota_sync.oscillators import resolve_oscillator_noise

    base = SDRSimulationConfig(
        num_iterations=iterations,
        seed=seed,
        device="cpu",
        channel_speed_mps=speed,
    )
    if profile is None:
        return base
    noise, _ = resolve_oscillator_noise(
        profile,
        base.carrier_frequency_hz,
        base.sample_rate,
        base.sync_interval,
    )
    return replace(base, **noise)


# ---------------------------------------------------------------------
# campaigns
# ---------------------------------------------------------------------

EXP1_TOPOLOGIES = {
    "two-way chain (tree)": lambda n, s: chain_edges(n, "two"),
    "two-way ring": lambda n, s: ring_edges(n, "two"),
    "one-way ring + one two-way edge": lambda n, s: (
        [(p, q, "one") for p, q, _ in ring_edges(n)[:-1]]
        + [(n - 1, 0, "two")]
    ),
    "one-way chain (tree) only": lambda n, s: chain_edges(n, "one"),
    "one-way ring (cycles) only": lambda n, s: ring_edges(n, "one"),
    "mixed random (half two-way)": lambda n, s: [
        (p, q, "two" if k % 2 == 0 else "one")
        for k, (p, q, _) in enumerate(
            random_connected_edges(n, 3, 1000 + s)
        )
    ],
}


def part_exp1(seeds, iterations=160, speed=0.0, label="exp1", cfo_span=0.0):
    n = 8
    cache = load_cache()
    results = {}
    print(f"== {label}: identifiability by topology "
          f"(N={n}, {iterations} intervals, speed={speed} m/s, "
          f"seeds {seeds}) ==")
    for name, maker in EXP1_TOPOLOGIES.items():
        rows = []
        for seed in seeds:
            out = run_openloop_graph(
                base_settings(seed, iterations, speed), n, maker(n, seed),
                init_cfo_span_hz=cfo_span,
            )
            first, second = window_halves(iterations)
            stats1 = pair_stats(out["node_traces"], first)
            stats2 = pair_stats(out["node_traces"], second)
            worst_std2 = max(v[1] for v in stats2.values())
            worst_bias2 = max(abs(v[0]) for v in stats2.values())
            # drift indicator: circular mean movement between halves
            move = max(
                abs(wrap_phase(torch.tensor(stats2[k][0] - stats1[k][0]))
                    .item())
                for k in stats1
            )
            rows.append(
                {
                    "seed": seed,
                    "worst_pair_std_mrad": 1e3 * worst_std2,
                    "worst_pair_bias_mrad": 1e3 * worst_bias2,
                    "mean_move_mrad": 1e3 * move,
                    "detect": out["detect_rate"],
                    "wall_s": out["wall_s"],
                }
            )
            print(f"  {name:36s} seed {seed}: worst std "
                  f"{1e3 * worst_std2:8.1f} mrad, worst |bias| "
                  f"{1e3 * worst_bias2:8.1f} mrad, mean-move "
                  f"{1e3 * move:8.1f} mrad, detect "
                  f"{100 * out['detect_rate']:.1f}%")
        results[name] = rows
    cache[label] = results
    save_cache(cache)


def part_exp2(seeds, iterations=100, budget=7):
    n = 8
    cache = load_cache()
    graphs = {
        "chain": chain_edges(n),
        "ring": ring_edges(n),
        "star": star_edges(n),
        "complete": complete_edges(n),
        "random+1": random_connected_edges(n, 1, 42),
        "random+4": random_connected_edges(n, 4, 43),
    }
    results = {}
    print(f"== exp2: accuracy vs topology (N={n}, equal budget "
          f"{budget} two-way exchanges/interval, {iterations} intervals, "
          f"seeds {seeds}) ==")
    for name, edges in graphs.items():
        resistance = effective_resistance(n, edges)
        per_seed = []
        for seed in seeds:
            out = run_openloop_graph(
                base_settings(seed, iterations), n, edges,
                budget_edges_per_interval=budget,
                init_cfo_span_hz=0.0,
            )
            first, second = window_halves(iterations)
            stats = pair_stats(out["node_traces"],
                               slice(iterations // 2, iterations))
            per_seed.append(
                {
                    "seed": seed,
                    "pairs": {k: {"bias": v[0], "std": v[1]}
                              for k, v in stats.items()},
                    "detect": out["detect_rate"],
                    "wall_s": out["wall_s"],
                }
            )
        # aggregate: per pair, mean std across seeds; correlate with R
        pair_names = list(per_seed[0]["pairs"].keys())
        stds = {
            k: float(np.mean([s["pairs"][k]["std"] for s in per_seed]))
            for k in pair_names
        }
        rvals = []
        vvals = []
        for k in pair_names:
            i, j = (int(x) for x in k.split("-"))
            rvals.append(resistance[i, j])
            vvals.append(stds[k] ** 2)
        correlation = float(np.corrcoef(rvals, vvals)[0, 1])
        worst = max(stds.values())
        best = min(stds.values())
        results[name] = {
            "edges": len(edges),
            "corr_var_vs_resistance": correlation,
            "worst_pair_std_mrad": 1e3 * worst,
            "best_pair_std_mrad": 1e3 * best,
            "per_seed": per_seed,
            "resistance_max": float(np.max(resistance)),
        }
        print(f"  {name:10s} edges {len(edges):2d}  worst-pair std "
              f"{1e3 * worst:7.1f} mrad  best {1e3 * best:6.1f}  "
              f"corr(var, R) {correlation:+.2f}  "
              f"maxR {np.max(resistance):.2f}")
    cache["exp2"] = results
    save_cache(cache)


def part_exp3(seeds, big=False):
    cache = load_cache()
    sizes = [8, 16, 32, 64] if big else [8, 16, 32]
    results = {}
    print(f"== exp3: size scaling, chains and rings, sizes {sizes}, "
          f"seeds {seeds} ==")
    for topology in ("chain", "ring"):
        rows = []
        for n in sizes:
            iterations = min(4 * n, 300)
            maker = chain_edges if topology == "chain" else ring_edges
            per_seed = []
            for seed in seeds:
                out = run_openloop_graph(
                    base_settings(seed, iterations), n, maker(n),
                    init_cfo_span_hz=0.0,
                )
                first, second = window_halves(iterations)
                # far pair: ends of the chain / antipodes of the ring
                far = (0, n - 1) if topology == "chain" else (0, n // 2)
                diff2 = wrap_phase(
                    out["node_traces"][far[0], second]
                    - out["node_traces"][far[1], second]
                )
                diff1 = wrap_phase(
                    out["node_traces"][far[0], first]
                    - out["node_traces"][far[1], first]
                )
                _, std2 = circ_mean_std(diff2)
                _, std1 = circ_mean_std(diff1)
                near = wrap_phase(
                    out["node_traces"][0, second]
                    - out["node_traces"][1, second]
                )
                _, near_std = circ_mean_std(near)
                per_seed.append(
                    {
                        "seed": seed,
                        "far_std_mrad": 1e3 * std2,
                        "far_std_firsthalf_mrad": 1e3 * std1,
                        "near_std_mrad": 1e3 * near_std,
                        "detect": out["detect_rate"],
                        "wall_s": out["wall_s"],
                    }
                )
                print(f"  {topology:5s} N={n:2d} seed {seed}: far-pair std "
                      f"{1e3 * std2:7.1f} mrad (first half "
                      f"{1e3 * std1:7.1f}), near-pair "
                      f"{1e3 * near_std:6.1f}, wall {out['wall_s']:.0f}s")
            rows.append({"n": n, "iterations": iterations,
                         "per_seed": per_seed})
        results[topology] = rows
    cache["exp3" + ("_big" if big else "")] = results
    save_cache(cache)


def part_exp4(num_seeds=24, iterations=80):
    n = 8
    cache = load_cache()
    states = {}
    rows = []
    print(f"== exp4: branch states on a two-way ring (N={n}, adverse "
          f"acquisition, branch check OFF, {num_seeds} seeds) ==")
    for seed in range(num_seeds):
        out = run_openloop_graph(
            base_settings(seed, iterations), n, ring_edges(n),
            branch_check=False,
            init_phase_span=math.pi,
            init_cfo_span_hz=0.0,
        )
        labels = []
        ok = True
        for em in out["edge_means"]:
            near_zero = abs(em["mean"]) < 0.6
            near_pi = abs(abs(em["mean"]) - math.pi) < 0.6
            if em["std"] > 0.5 or not (near_zero or near_pi):
                ok = False
            labels.append(1 if near_pi else 0)
        key = "".join(str(b) for b in labels) if ok else "unsettled"
        states[key] = states.get(key, 0) + 1
        rows.append({"seed": seed, "state": key,
                     "pi_edges": sum(labels) if ok else None})
        print(f"  seed {seed:2d}: state {key} "
              f"(pi-edges {sum(labels) if ok else '-'})")
    cache["exp4"] = {"states": states, "rows": rows}
    save_cache(cache)
    parities = {
        k: sum(int(c) for c in k) % 2
        for k in states if k != "unsettled"
    }
    print(f"  distinct settled states: "
          f"{len([k for k in states if k != 'unsettled'])}; "
          f"parities: {set(parities.values()) or 'n/a'}")


def part_smoke():
    out = run_openloop_graph(
        base_settings(0, 20), 4, chain_edges(4)
    )
    print("smoke: detect", out["detect_rate"],
          "edge means", [f"{e['mean']:+.3f}±{e['std']:.3f}"
                         for e in out["edge_means"]],
          f"wall {out['wall_s']:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--big", action="store_true")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.part == "smoke":
        part_smoke()
    elif args.part == "exp1":
        part_exp1(seeds)
    elif args.part == "exp1-moving":
        part_exp1(seeds, speed=0.02, label="exp1_moving")
    elif args.part == "exp1-cfo":
        part_exp1(seeds, label="exp1_cfo", cfo_span=200.0)
    elif args.part == "exp1-zerocfo":
        # control: zero initial frequency offsets
        global EXP1_TOPOLOGIES
        n = 8
        cache = load_cache()
        rows = {}
        for name in ("one-way chain (tree) only", "two-way chain (tree)"):
            maker = EXP1_TOPOLOGIES[name]
            per = []
            for seed in seeds:
                out = run_openloop_graph(
                    base_settings(seed, 80), n, maker(n, seed),
                    init_cfo_span_hz=0.0,
                )
                stats = pair_stats(out["node_traces"], slice(40, 80))
                worst_std = max(v[1] for v in stats.values())
                worst_bias = max(abs(v[0]) for v in stats.values())
                per.append({"seed": seed,
                            "worst_pair_std_mrad": 1e3 * worst_std,
                            "worst_pair_bias_mrad": 1e3 * worst_bias})
                print(f"  [zero-CFO] {name}: seed {seed} std "
                      f"{1e3 * worst_std:.1f} bias {1e3 * worst_bias:.1f}")
            rows[name] = per
        cache["exp1_zerocfo"] = rows
        save_cache(cache)
    elif args.part == "exp2":
        part_exp2(seeds)
    elif args.part == "exp3":
        part_exp3(seeds, big=args.big)
    elif args.part == "exp4":
        part_exp4()
    else:
        raise SystemExit(f"unknown part {args.part}")


if __name__ == "__main__":
    main()
