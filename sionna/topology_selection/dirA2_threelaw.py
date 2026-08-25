"""Direction A2: do the three topology-stability mechanisms survive a
change of control law? The decisive fork experiment for direction A.

External review: do NOT claim "stability, not information, governs
synchronization topology" until sequential/directed updates are
tested; if the effects vanish under sequential updates, the narrower
claim is "topology and update protocol are coupled".

Three control laws, everything else identical (same physics, seeds,
geometry, airtime accounting, branch check, EKFs):

  symmetric    dirA_runner's exact law: every serviced edge issues a
               correction, endpoints apply +-c/2 simultaneously
               (Jacobi-style; parity stagger only when every edge is
               serviced every interval). This is the arm direction A's
               published numbers came from - continuity by
               construction.
  alternating  Gauss-Seidel turn-taking generalized to arbitrary
               graphs by proper edge coloring (greedy matching
               decomposition): only edges of the color class matching
               the interval fire, at full +-c/2; corrections computed
               off-slot are forward-predicted through F^k to their
               slot. No node ever receives two simultaneous
               corrections.
  directed     elected-root tree (BFS from node 0): each correction is
               applied fully to the child node. Tree topologies only.
  symmetric-dw control row: mesh.py's canonical degree-weighted
               symmetric law (+-c/(2 deg)) - separates "undamped vs
               damped simultaneous" on the star.

PRE-REGISTERED PREDICTIONS (written before any run; printed by
--part predict together with the computed linearized spectral radii):

  P1 (cadence ceiling - physical): survives ALL three laws roughly
     unchanged. Reasoning: the driver is the pair frequency random
     walk accumulating across the m-interval service gap and breaking
     the mod-pi branch pick - filter/oscillator physics that no
     update ordering can remove. Falsifier: alternating or directed
     materially shifts the m ~ 2 ceiling.
  P2 (degree ceiling - numerical): VANISHES (or greatly weakens)
     under alternating and directed. Reasoning: simultaneous
     half-corrections are undamped Jacobi iteration theta+ =
     (I - L/2) theta, divergent when lambda_max(L) > 4 (star hub-7:
     lambda_max = 8 -> spectral radius 3); edge-colored turn-taking
     is the Gauss-Seidel-style schedule whose per-matching factors
     (I - L_M/2) have eigenvalues {1, 0} and whose product contracts;
     directed is a nilpotent hierarchy copy. Falsifier: the star
     still collapses under alternating.
  P3 (cycle exclusion - topological): SURVIVES alternating (a 2pi
     winding around a cycle is a locally stable sector under any
     local update rule - twisted-state argument), is MOOT under
     directed (trees only); entry RATE into winding states may drop
     under alternating. Falsifier: alternating ring matches
     alternating chain. Sharpening registered in advance: for an
     even-N ring the undamped-Jacobi matrix I - L/2 has eigenvalue
     exactly -1 (lambda_max = 4), a marginal alternating mode, so
     part of the symmetric ring's failure may be Jacobi marginality
     rather than winding; the winding-occupancy counter separates the
     two (windings survive alternating, marginal-mode churn does
     not).

Grid mirrors direction A exactly where it overlaps (N=8, uniform
geometry frozen at seed 7, 150 intervals, pilot 255/64, steady window
= last 50): cadence axis MST x m in {1.0, 1.4, 1.75, 2.33, 3.5};
degree axis star / degree-3 tree / MST at every-edge-every-interval;
cycle axis index-chain / ring / MST+2 chords / complete; N=16 spot
checks; zero-CFO controls on the sparsest cadence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "phase_sync_idea"))

from sionna.phy import config as sionna_config  # noqa: E402

import openloop_topology_study as topo  # noqa: E402
from dirA_selection import (  # noqa: E402
    ACQ,
    ITERATIONS,
    PILOT,
    FrozenPlacement,
    make_positions,
    measured_gain,
    strategy_edges,
    edge_model,
)
from ota_sync.coherent import _pick_half_phase  # noqa: E402
from ota_sync.core import (  # noqa: E402
    REAL_DTYPE,
    Oscillator,
    PhaseFrequencyEKF,
    resolve_device,
    wrap_phase,
)
from ota_sync.network import MAX_LINK_SNR_DB  # noqa: E402
from ota_sync.sdr import (  # noqa: E402
    SDRRadioLink,
    SDRSimulationConfig,
    SDRSynchronizer,
    _FlickerFrequencyNoise,
    _measurement_covariance,
    _quantize_correction,
    make_sync_preamble,
)

CACHE = HERE / "dirA2_cache.json"
FIGDIR = HERE / "figures"
N = 8
STEADY = slice(100, ITERATIONS)
LAWS = ("symmetric", "alternating", "directed")


def a2_settings(seed: int) -> SDRSimulationConfig:
    return SDRSimulationConfig(
        num_iterations=ITERATIONS, seed=seed, device="cpu", **PILOT
    )


# ---------------------------------------------------------------------
# graph helpers
# ---------------------------------------------------------------------

def edge_coloring(pairs: list[tuple[int, int]]) -> tuple[list[int], int]:
    """Greedy proper edge coloring (matching decomposition)."""

    colors: list[int] = []
    for index, (p, q) in enumerate(pairs):
        used = {
            colors[j]
            for j, (a, b) in enumerate(pairs[:index])
            if {a, b} & {p, q}
        }
        c = 0
        while c in used:
            c += 1
        colors.append(c)
    return colors, (max(colors) + 1 if colors else 1)


def bfs_parent(num_nodes: int, pairs: list[tuple[int, int]]) -> dict[int, int]:
    adjacency = {i: [] for i in range(num_nodes)}
    for p, q in pairs:
        adjacency[p].append(q)
        adjacency[q].append(p)
    parent = {0: None}
    queue = [0]
    while queue:
        node = queue.pop(0)
        for nxt in adjacency[node]:
            if nxt not in parent:
                parent[nxt] = node
                queue.append(nxt)
    return parent


def tree_cycles(num_nodes, pairs):
    """Independent cycles: for each chord, the tree path + the chord.
    Returns list of oriented edge lists [(p, q, sign), ...]."""

    parent = bfs_parent(num_nodes, pairs)
    tree = {
        (min(p, q), max(p, q))
        for child, p in parent.items()
        if p is not None
        for q in [child]
    }
    tree = set()
    for child, par in parent.items():
        if par is not None:
            tree.add((min(child, par), max(child, par)))

    def path_to_root(node):
        out = []
        while parent[node] is not None:
            out.append(node)
            node = parent[node]
        out.append(node)
        return out

    cycles = []
    for (p, q) in pairs:
        if (min(p, q), max(p, q)) in tree:
            continue
        pa, qa = path_to_root(p), path_to_root(q)
        common = None
        pa_set = set(pa)
        for node in qa:
            if node in pa_set:
                common = node
                break
        walk = []
        for node in pa[: pa.index(common)]:
            walk.append((node, parent[node]))
        down = []
        for node in qa[: qa.index(common)]:
            down.append((parent[node], node))
        cycle = [(p, q, +1)]  # chord q->p closes; traverse p up, down to q
        for a, b in walk:
            cycle.append((a, b, +1))
        for a, b in reversed(down):
            cycle.append((a, b, +1))
        cycles.append(cycle)
    return cycles


def degree_capped_tree(positions, sigma2, cap=3):
    """Kruskal by sigma2 with a node-degree cap."""

    n = positions.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    degree = [0] * n
    chosen = []
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for (i, j) in sorted(all_pairs, key=lambda e: sigma2[e]):
        if degree[i] >= cap or degree[j] >= cap:
            continue
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            chosen.append((i, j))
            degree[i] += 1
            degree[j] += 1
    return chosen


# ---------------------------------------------------------------------
# linearized spectral radii (ex-ante, part of the predictions)
# ---------------------------------------------------------------------

def _laplacian(n, pairs):
    L = np.zeros((n, n))
    for p, q in pairs:
        L[p, p] += 1
        L[q, q] += 1
        L[p, q] -= 1
        L[q, p] -= 1
    return L


def linearized_radius(n, pairs, law, budget):
    """Per-interval spectral radius of the small-angle update map on
    the non-consensus subspace, for the firing schedule the law and
    budget actually produce."""

    num_edges = len(pairs)
    if law == "directed":
        parent = bfs_parent(n, pairs)
        M = np.eye(n)
        for child, par in parent.items():
            if par is None:
                continue
            step = np.eye(n)
            step[child, :] = 0.0
            step[child, par] = 1.0
            M = step @ M
        period = 1
    elif law == "alternating":
        colors, n_colors = edge_coloring(pairs)
        M = np.eye(n)
        for c in range(n_colors):
            firing = [pairs[j] for j in range(num_edges) if colors[j] == c]
            M = (np.eye(n) - 0.5 * _laplacian(n, firing)) @ M
        period = n_colors
    elif law == "symmetric-dw":
        L = _laplacian(n, pairs)
        D = np.diag(np.maximum(np.diag(L), 1.0))
        M = np.eye(n) - 0.5 * np.linalg.inv(D) @ L
        period = 1
    else:  # symmetric
        if budget is None or budget >= num_edges:
            even = [pairs[j] for j in range(num_edges) if j % 2 == 0]
            odd = [pairs[j] for j in range(num_edges) if j % 2 == 1]
            M = (np.eye(n) - 0.5 * _laplacian(n, odd)) @ (
                np.eye(n) - 0.5 * _laplacian(n, even)
            )
            period = 2
        else:
            g = math.gcd(num_edges, budget)
            period = num_edges // g
            M = np.eye(n)
            pointer = 0
            for _ in range(period):
                firing = [
                    pairs[(pointer + k) % num_edges] for k in range(budget)
                ]
                pointer = (pointer + budget) % num_edges
                M = (np.eye(n) - 0.5 * _laplacian(n, firing)) @ M
    ones = np.ones((n, 1)) / math.sqrt(n)
    P = np.eye(n) - ones @ ones.T
    radius = max(abs(np.linalg.eigvals(P @ M @ P)))
    return float(radius ** (1.0 / period))


# ---------------------------------------------------------------------
# the three-law runner (symmetric path verbatim from dirA_runner)
# ---------------------------------------------------------------------

def run_threelaw(
    settings: SDRSimulationConfig,
    num_nodes: int,
    edge_spec: list[tuple[int, int, str]],
    law: str,
    *,
    budget_edges_per_interval: int | None = None,
    acquisition_intervals: int = 10,
    init_phase_span: float = 0.5,
    init_cfo_span_hz: float = 1500.0,
    radius_m: float = 500.0,
    path_loss_exponent: float = 2.7,
    reference_distance_m: float = 500.0,
):
    if law not in LAWS + ("symmetric-dw",):
        raise ValueError(law)
    t0 = time.time()
    device = resolve_device(settings.device)
    torch.manual_seed(settings.seed)
    sionna_config.seed = settings.seed
    generator = torch.Generator(device=device)
    generator.manual_seed(settings.seed + 1)
    init_rng = np.random.default_rng(settings.seed + 977)

    positions = topo.place_stations(num_nodes, radius_m, settings.seed)
    dt = settings.sync_interval
    dt_samples = int(round(dt * settings.sample_rate))
    pairs = [(p, q) for p, q, _ in edge_spec]

    if law == "alternating":
        colors, n_colors = edge_coloring(pairs)
    if law == "directed":
        parent = bfs_parent(num_nodes, pairs)
        for p, q in pairs:
            ok = parent.get(q) == p or parent.get(p) == q
            if not ok:
                raise ValueError("directed law requires a tree")
    degree = [0] * num_nodes
    for p, q in pairs:
        degree[p] += 1
        degree[q] += 1

    frequency_process_std = 2.0 * math.pi * settings.frequency_process_std_hz
    interval_covariance = torch.diag(
        torch.tensor(
            [settings.phase_process_std_rad**2, frequency_process_std**2],
            dtype=REAL_DTYPE,
            device=device,
        )
    )
    oscillators = []
    for index in range(num_nodes):
        phase0 = float(init_rng.uniform(-init_phase_span, init_phase_span))
        cfo0 = float(init_rng.uniform(-init_cfo_span_hz, init_cfo_span_hz))
        oscillators.append(
            Oscillator(
                phase0, 2.0 * math.pi * cfo0, dt, interval_covariance,
                device, generator,
            )
        )
    flickers = [
        _FlickerFrequencyNoise(
            settings.flicker_frequency_std_hz, dt,
            settings.num_iterations * settings.sync_interval,
            device, generator,
        )
        for _ in range(num_nodes)
    ]
    flicker_previous = [
        torch.zeros((), dtype=REAL_DTYPE, device=device)
        for _ in range(num_nodes)
    ]

    preamble = make_sync_preamble(settings, device)
    white_fm_interval = settings.phase_noise_std_rad**2 * dt_samples
    edges = []
    from dataclasses import replace as dc_replace
    for p, q, kind in edge_spec:
        distance = max(float(np.linalg.norm(positions[p] - positions[q])), 1.0)
        snr_db = min(
            settings.snr_db
            - 10.0 * path_loss_exponent
            * math.log10(distance / reference_distance_m),
            MAX_LINK_SNR_DB,
        )
        edge_settings = dc_replace(settings, snr_db=snr_db)
        link_fwd = SDRRadioLink(edge_settings, preamble, device, generator)
        link_rev = SDRRadioLink(
            edge_settings, preamble, device, generator, mirror_of=link_fwd
        )
        oneway_noise = _measurement_covariance(edge_settings, preamble, device)
        ekf = PhaseFrequencyEKF(
            dt,
            2.0 * interval_covariance
            + torch.diag(
                torch.tensor(
                    [white_fm_interval, 2.0 * flickers[0].innovation_variance],
                    dtype=REAL_DTYPE, device=device,
                )
            ),
            0.5 * oneway_noise,
            device,
            initial_covariance=torch.diag(
                torch.tensor(
                    [math.pi**2, (2.0 * math.pi * 50e3) ** 2],
                    dtype=REAL_DTYPE, device=device,
                )
            ),
        )
        edges.append(
            {
                "pair": (p, q), "kind": kind, "snr_db": snr_db,
                "link_fwd": link_fwd, "link_rev": link_rev,
                "synchronizer": SDRSynchronizer(edge_settings, preamble),
                "ekf": ekf, "acquired": False, "pending": {},
                "loaded": False,
            }
        )

    node_corr_freq = [
        torch.zeros((), dtype=REAL_DTYPE, device=device)
        for _ in range(num_nodes)
    ]
    chain_bias = math.radians(settings.twoway_chain_asymmetry_deg)
    per_node_walk = settings.phase_noise_std_rad / math.sqrt(2.0)
    capture_samples = (
        edges[0]["link_fwd"].input_length + edges[0]["link_fwd"].l_tot - 1
        if edges else 0
    )
    tree_pairs, _, subtree_of = topo.bfs_subtrees(num_nodes, edge_spec)
    flip_count = 0
    realign_count = 0

    node_rows: list[list[torch.Tensor]] = [[] for _ in range(num_nodes)]
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

        for edge in edges:
            if edge["acquired"]:
                edge["ekf"].predict()

        # ---- law-dependent correction application -------------------
        node_delta = [
            torch.zeros(2, dtype=REAL_DTYPE, device=device)
            for _ in range(num_nodes)
        ]
        any_loaded = False
        for edge in edges:
            due = edge["pending"].pop(iteration, None)
            if due is None:
                continue
            p, q = edge["pair"]
            if law == "directed":
                if parent.get(q) == p:
                    node_delta[q] = node_delta[q] + due
                else:
                    node_delta[p] = node_delta[p] - due
            elif law == "symmetric-dw":
                node_delta[p] = node_delta[p] - due / (2.0 * degree[p])
                node_delta[q] = node_delta[q] + due / (2.0 * degree[q])
            else:  # symmetric and alternating both apply +-c/2
                node_delta[p] = node_delta[p] - due / 2.0
                node_delta[q] = node_delta[q] + due / 2.0
            edge["loaded"] = True
            any_loaded = True
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

        if iteration >= acquisition_intervals and iteration % 4 == 0:
            for edge in edges:
                if edge["kind"] != "two" or not edge["loaded"]:
                    continue
                p, q = edge["pair"]
                if (min(p, q), max(p, q)) not in tree_pairs:
                    continue
                if torch.cos(
                    oscillators[p].state[0] - oscillators[q].state[0]
                ) < -0.2:
                    child = q if q in subtree_of and p not in subtree_of.get(
                        q, set()
                    ) else p
                    flip = torch.tensor(
                        [math.pi, 0.0], dtype=REAL_DTYPE, device=device
                    )
                    for member in subtree_of.get(child, {child}):
                        oscillators[member].apply_correction(flip)
                    flip_count += 1
                    last = edge.get("last_flip", -10)
                    if iteration - last <= 4:
                        edge["acquired"] = False
                        edge["ekf"].state = torch.zeros(
                            2, dtype=REAL_DTYPE, device=device
                        )
                        realign_count += 1
                    edge["last_flip"] = iteration

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
            capture_rev = edge["link_rev"].capture(
                oscillators[q], oscillators[p], iteration, -sfo
            )
            oscillators[q].state[0] = wrap_phase(
                oscillators[q].state[0] + capture_rev.lo_walk_end
            )
            reverse = edge["synchronizer"].estimate(capture_rev.samples)
            detected = forward.detected and reverse.detected
            serviced_count += 1
            detected_count += int(detected)
            if not detected:
                continue
            ekf = edge["ekf"]
            combined_half = wrap_phase(
                wrap_phase(forward.phase - reverse.phase) / 2.0 + chain_bias
            )
            frequency_obs = (forward.frequency - reverse.frequency) / 2.0
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

            # ---- law-dependent issuance -----------------------------
            if law == "alternating":
                color = colors[edge_index]
                k = 1
                while (iteration + k) % n_colors != color:
                    k += 1
                predicted = ekf.state.clone()
                for _ in range(k):
                    predicted = ekf.transition @ predicted
                edge["pending"][iteration + k] = _quantize_correction(
                    predicted, settings
                )
            elif law == "directed" or law == "symmetric-dw":
                predicted = ekf.transition @ ekf.state
                edge["pending"][iteration + 1] = _quantize_correction(
                    predicted, settings
                )
            else:  # symmetric: dirA_runner's exact issuance
                stagger = (
                    budget_edges_per_interval is None
                    or budget_edges_per_interval >= len(edges)
                )
                if (not stagger) or (iteration + 1) % 2 == edge_index % 2:
                    predicted = ekf.transition @ ekf.state
                    edge["pending"][iteration + 1] = _quantize_correction(
                        predicted, settings
                    )

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

    node_traces = torch.stack(
        [torch.stack(row).detach().cpu() for row in node_rows]
    )
    # winding occupancy over independent cycles, steady window
    windings = []
    for cycle in tree_cycles(num_nodes, pairs):
        total = 0.0
        for a, b, sign in cycle:
            diff = wrap_phase(
                node_traces[a, STEADY] - node_traces[b, STEADY]
            )
            mean, _ = topo.circ_mean_std(diff)
            total += sign * mean
        windings.append(total)
    return {
        "node_traces": node_traces,
        "detect_rate": detected_count / max(serviced_count, 1),
        "flips": flip_count,
        "realigns": realign_count,
        "windings": windings,
        "wall_s": time.time() - t0,
    }


# ---------------------------------------------------------------------
# campaign
# ---------------------------------------------------------------------

def load_cache():
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return {}


def save_cache(data):
    # Merge-on-write: never lose cells another driver put in the file.
    # (Two clobber incidents happened when drivers passed cell() a
    # dict that did not start from load_cache(); this makes that
    # mistake harmless.)
    merged = load_cache()
    merged.update(data)
    CACHE.write_text(json.dumps(merged, indent=1))


def topology(name, positions, sigma2):
    if name == "mst":
        return strategy_edges("mst", positions, sigma2, None, None)
    if name == "star":
        return strategy_edges("star", positions, sigma2, None, None)
    if name == "ring":
        return strategy_edges("ring", positions, sigma2, None, None)
    if name == "deg3tree":
        return degree_capped_tree(positions, sigma2, cap=3)
    if name == "chain":
        return [(i, i + 1) for i in range(positions.shape[0] - 1)]
    if name == "mst2c":
        tree = strategy_edges("mst", positions, sigma2, None, None)
        tset = set(tree)
        n = positions.shape[0]
        rest = sorted(
            (
                (i, j)
                for i in range(n)
                for j in range(i + 1, n)
                if (i, j) not in tset
            ),
            key=lambda e: sigma2[e],
        )
        return tree + rest[:2]
    if name == "complete":
        n = positions.shape[0]
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    raise ValueError(name)


def cell(cache, key, positions, edges, law, budget, seed, cfo=1500.0,
         n=N):
    if key in cache:
        return cache[key]
    spec = [(i, j, "two") for (i, j) in edges]
    settings = a2_settings(seed)
    with FrozenPlacement(positions):
        out = run_threelaw(
            settings, n, spec, law,
            budget_edges_per_interval=budget,
            acquisition_intervals=ACQ,
            init_cfo_span_hz=cfo,
        )
    theta = out["node_traces"][:, STEADY].to(torch.complex128)
    phasors = torch.exp(1j * theta)
    gain = float(
        torch.mean(
            (torch.abs(torch.sum(phasors, dim=0)) ** 2 / (n * n)).real
        )
    )
    result = {
        "gain": gain,
        "detect": out["detect_rate"],
        "flips": out["flips"],
        "realigns": out["realigns"],
        "windings": out["windings"],
        "wall_s": out["wall_s"],
    }
    cache[key] = result
    save_cache(cache)
    print(f"  {key}: gain {100*gain:.1f}% flips {out['flips']} "
          f"windings {[round(w, 2) for w in out['windings']]} "
          f"({out['wall_s']:.0f}s)", flush=True)
    return result


def part_predict():
    print(__doc__.split("PRE-REGISTERED")[1].split("Grid mirrors")[0])
    positions = make_positions("uniform")
    sigma2 = edge_model(positions)[1]
    print("Linearized per-interval spectral radii (non-consensus "
          "subspace; >1 = divergent, =1 marginal):")
    rows = [
        ("mst", None), ("star", None), ("deg3tree", None),
        ("chain", None), ("ring", None), ("mst2c", None),
        ("complete", None), ("mst", 2),
    ]
    print(f"{'topology':>10} {'budget':>7} {'symmetric':>10} "
          f"{'symmetric-dw':>13} {'alternating':>12} {'directed':>9}")
    for name, budget in rows:
        edges = topology(name, positions, sigma2)
        vals = []
        for law in ("symmetric", "symmetric-dw", "alternating", "directed"):
            try:
                vals.append(f"{linearized_radius(N, edges, law, budget):.3f}")
            except ValueError:
                vals.append("tree-only")
        print(f"{name:>10} {str(budget):>7} {vals[0]:>10} {vals[1]:>13} "
              f"{vals[2]:>12} {vals[3]:>9}")


def get_sigma2(positions):
    return edge_model(positions)[1]


def part_cadence(seeds):
    cache = load_cache()
    positions = make_positions("uniform")
    sigma2 = get_sigma2(positions)
    edges = topology("mst", positions, sigma2)
    for budget in (7, 5, 4, 3, 2):
        for law in LAWS:
            for seed in seeds:
                key = f"cad|mst|{law}|B{budget}|s{seed}"
                cell(cache, key, positions, edges, law, budget, seed)


def part_degree(seeds):
    cache = load_cache()
    positions = make_positions("uniform")
    sigma2 = get_sigma2(positions)
    for name in ("star", "deg3tree", "mst"):
        edges = topology(name, positions, sigma2)
        for law in LAWS:
            for seed in seeds:
                key = f"deg|{name}|{law}|B7|s{seed}"
                cell(cache, key, positions, edges, law, 7, seed)
    edges = topology("star", positions, sigma2)
    for seed in seeds:
        key = f"deg|star|symmetric-dw|B7|s{seed}"
        cell(cache, key, positions, edges, "symmetric-dw", 7, seed)


def part_cycle(seeds):
    cache = load_cache()
    positions = make_positions("uniform")
    sigma2 = get_sigma2(positions)
    for name in ("chain", "ring", "mst2c", "complete"):
        edges = topology(name, positions, sigma2)
        for law in ("symmetric", "alternating"):
            for seed in seeds:
                key = f"cyc|{name}|{law}|Bnone|s{seed}"
                cell(cache, key, positions, edges, law, None, seed)


def part_scale(seeds):
    cache = load_cache()
    n = 16
    positions = topo.place_stations(n, 500.0, 7)
    dists = {
        (i, j): float(np.linalg.norm(positions[i] - positions[j]))
        for i in range(n) for j in range(i + 1, n)
    }
    # MST / star / ring on distance (proxy for sigma2 ordering)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    mst = []
    for (i, j) in sorted(dists, key=dists.get):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            mst.append((i, j))
    hub = min(
        range(n),
        key=lambda h: sum(
            dists[(min(h, j), max(h, j))] for j in range(n) if j != h
        ),
    )
    star = [(min(hub, j), max(hub, j)) for j in range(n) if j != hub]
    left = set(range(1, n))
    order = [0]
    while left:
        last = order[-1]
        nxt = min(left, key=lambda j: dists[(min(last, j), max(last, j))])
        order.append(nxt)
        left.remove(nxt)
    ring = sorted(
        {(min(a, b), max(a, b)) for a, b in zip(order, order[1:] + [order[0]])}
    )
    for name, edges, budget in (
        ("mst16", mst, 15), ("star16", star, 15), ("ring16", ring, None),
    ):
        for law in ("symmetric", "alternating"):
            for seed in seeds[:2]:
                key = f"scl|{name}|{law}|B{budget}|s{seed}"
                cell(cache, key, positions, edges, law, budget, seed, n=n)


def part_controls():
    cache = load_cache()
    positions = make_positions("uniform")
    sigma2 = get_sigma2(positions)
    edges = topology("mst", positions, sigma2)
    for law in ("alternating", "directed"):
        key = f"ctl|mst|{law}|B2|s0|cfo0"
        cell(cache, key, positions, edges, law, 2, 0, cfo=0.0)


def part_report():
    cache = load_cache()

    def agg(prefix):
        rows = {}
        for key, val in cache.items():
            if not key.startswith(prefix):
                continue
            parts = key.split("|")
            label = "|".join(parts[1:-1])
            rows.setdefault(label, []).append(val)
        return rows

    for prefix, title in (
        ("cad", "CADENCE (MST, gain % mean±std)"),
        ("deg", "DEGREE (B=7)"),
        ("cyc", "CYCLE (every edge every interval)"),
        ("scl", "N=16 SPOTS"),
        ("ctl", "ZERO-CFO CONTROLS"),
    ):
        print(f"\n== {title}")
        for label, vals in sorted(agg(prefix + "|").items()):
            gains = [100 * v["gain"] for v in vals]
            flips = [v["flips"] for v in vals]
            winds = [
                sum(1 for w in v["windings"] if abs(w) > math.pi)
                for v in vals
            ]
            print(
                f"  {label:<28} gain {np.mean(gains):5.1f}±"
                f"{np.std(gains):4.1f}  flips {np.mean(flips):5.1f}  "
                f"windings/run {np.mean(winds):.2f}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--part", required=True,
        choices=["predict", "cadence", "degree", "cycle", "scale",
                 "controls", "report", "tieback"],
    )
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.part == "predict":
        part_predict()
    elif args.part == "cadence":
        part_cadence(seeds)
    elif args.part == "degree":
        part_degree(seeds)
    elif args.part == "cycle":
        part_cycle(seeds)
    elif args.part == "scale":
        part_scale(seeds)
    elif args.part == "controls":
        part_controls()
    elif args.part == "tieback":
        # continuity check: symmetric arm vs dirA_runner, same cell
        from dirA_selection import run_cell
        positions = make_positions("uniform")
        sigma2 = get_sigma2(positions)
        edges = topology("mst", positions, sigma2)
        ref = run_cell(positions, edges, 7, 0)
        cache = load_cache()
        mine = cell(cache, "tie|mst|symmetric|B7|s0", positions, edges,
                    "symmetric", 7, 0)
        print(f"dirA_runner gain {100*ref['gain']:.2f}% vs "
              f"dirA2 symmetric {100*mine['gain']:.2f}%")
    else:
        part_report()


if __name__ == "__main__":
    main()
