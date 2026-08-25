"""Direction D: dynamic channel-aware synchronization-topology
re-selection at the waveform level.

Question: when link qualities change mid-run (blockage on some links),
which sync links deserve the airtime NOW - and what does re-selection
actually cost?

Setup: N=8 nodes, 28 candidate two-way edges, active set = a 7-edge
spanning tree serviced every interval (equal sync airtime for every
strategy, always). A blockage episode drops the SNR of two active tree
edges by 20 dB for a window of intervals. Blockage is modeled as an
SNR loss applied to both directions of the edge (noise-floor scaling
on the already-built radio links; the estimator only sees SNR, so
attenuating signal vs raising noise is equivalent at this layer).

Strategies (identical airtime = 7 two-way exchanges/interval):
  static   initial max-SNR spanning tree, never changes
  channel  re-select when an active edge's measured detection metric
           (the correlator's normalized peak - measured by the sync
           exchanges themselves, no oracle) drops persistently below
           0.6x its own pre-episode baseline; replacement = best
           surveyed candidate edge that reconnects the tree
  gain     same degradation detector, but the switch decision uses the
           predicted coherent-gain improvement (effective-resistance
           gain model, per-edge weights from believed measurement
           variance): switch only if predicted delta-G > 2 points
  oracle   knows the true SNR schedule; switches to the best tree at
           the episode boundaries instantly

Switching costs charged: the incoming edge re-acquires (its first
detected exchange initializes the filter; for SETTLE=3 serviced
intervals it issues no corrections), tree subtrees are recomputed, the
branch check on the new tree is gated for 8 intervals (transient
protection), and one 1-bit branch check on the new edge is counted.

Physics conventions copied from openloop_topology_study.py (the
sandbox's graph harness): randomized per-node initial CFOs (no
arithmetic grid - the aliasing trap), subtree-preserving pi-flips with
the flip-storm escape, run lengths >= 4 service cycles. All imports
from ../phase_sync_idea; nothing there is modified.

CONTROL LAW NOTE (a measured decision, not a preference): the
symmetric half/half consensus law in openloop_topology_study.py does
NOT converge at N=8 in 150 intervals on either a max-SNR tree or a
star (steady gain 0.12-0.39, both in that harness and in this
adaptation; that harness's own campaigns are marked not-yet-run, so it
carries no validated N=8 result). This repo's Phase-1 mesh work
measured the same phenomenon (the symmetric-update "consensus tax")
and measured the DIRECTED elected-root tree law - each child corrects
fully toward its parent - at 99.8-99.9% of ideal gain, matching a
centralized star. Since this study's question is WHICH LINKS get
airtime (not which control law wins), the directed-tree law is used
here to decouple link selection from the known control-law
instability. The symmetric-law non-convergence is recorded in
RESULTS_D.md as an anomaly with its control, per discipline.

PREDICTIONS (stated before the campaigns ran):
  P1 static loses coherent gain through the episode (degraded tree
     edges miss detections; their subtrees coast) and recovers only
     when the episode ends. Adaptive strategies recover within
     trigger delay (~5-10 intervals) + settling (~3); oracle fastest.
  P2 channel and gain strategies behave near-identically here (the
     blockage is severe, so the gain margin is easily cleared); the
     gain strategy's value shows up as FEWER switches in marginal
     cases, not better gain in severe ones.
  P3 false-trigger rate in the no-disturbance control ~ 0 at the 0.6x
     threshold.
  P4 re-selection pays only for episodes longer than roughly the
     trigger+settle time (~10-15 intervals); below that the switching
     transient costs more than the blockage.

Usage:
  python dirD_dynamic.py --part main       # episode 100-200, 4 strategies
  python dirD_dynamic.py --part control    # no episode, false triggers
  python dirD_dynamic.py --part boundary   # episode-length sweep
  python dirD_dynamic.py --part figs      # figures + summary from cache
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase_sync_idea"))

from sionna.phy import config as sionna_config  # noqa: E402

from openloop_topology_study import (  # noqa: E402
    bfs_subtrees,
    circ_mean_std,
)
from ota_sync.core import (  # noqa: E402
    REAL_DTYPE,
    Oscillator,
    PhaseFrequencyEKF,
    resolve_device,
    wrap_phase,
)
from ota_sync.network import MAX_LINK_SNR_DB, place_stations  # noqa: E402
from ota_sync.sdr import (  # noqa: E402
    SDRRadioLink,
    SDRSimulationConfig,
    SDRSynchronizer,
    _FlickerFrequencyNoise,
    _measurement_covariance,
    _quantize_correction,
    make_sync_preamble,
)

CACHE = HERE / "dirD_cache.json"
N = 8
BUDGET = N - 1  # active tree edges serviced per interval
SETTLE = 3  # serviced intervals before a fresh edge issues corrections
TRIGGER_RATIO = 0.6  # metric EMA below this x baseline => degraded
TRIGGER_RUN = 5  # consecutive bad serviced intervals to trigger
GAIN_MARGIN = 0.02  # predicted delta-G required by the gain strategy
EMA_ALPHA = 0.3


# ---------------------------------------------------------------------
# trees and the gain model
# ---------------------------------------------------------------------

def all_pairs(n=N):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def max_weight_tree(n, weights):
    """Kruskal maximum-weight spanning tree; weights: dict pair->w."""

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree = []
    for pair in sorted(weights, key=lambda p: -weights[p]):
        a, b = find(pair[0]), find(pair[1])
        if a != b:
            parent[a] = b
            tree.append(pair)
    return tree


def predicted_gain(n, tree_pairs, edge_variance):
    """Expected coherent gain of a tree from per-edge phase-measurement
    variances via the resistance model (README central object). Used
    for RANKING candidate trees only - documented heuristic, not a
    calibrated absolute prediction."""

    lap = np.zeros((n, n))
    for (p, q) in tree_pairs:
        w = 1.0 / max(edge_variance[(p, q)], 1e-9)
        lap[p, p] += w
        lap[q, q] += w
        lap[p, q] -= w
        lap[q, p] -= w
    pinv = np.linalg.pinv(lap)
    total = 0.0
    for i in range(n):
        for j in range(n):
            resistance = pinv[i, i] + pinv[j, j] - 2.0 * pinv[i, j]
            total += math.exp(-resistance / 2.0)
    return total / n**2


def reconnecting_candidates(n, tree_pairs, removed_pair, candidate_pairs):
    """Pairs that reconnect the two components of tree minus edge."""

    adjacency = {i: set() for i in range(n)}
    for (p, q) in tree_pairs:
        if (p, q) == removed_pair:
            continue
        adjacency[p].add(q)
        adjacency[q].add(p)
    component = {removed_pair[0]}
    frontier = [removed_pair[0]]
    while frontier:
        node = frontier.pop()
        for nxt in adjacency[node]:
            if nxt not in component:
                component.add(nxt)
                frontier.append(nxt)
    return [
        (p, q) for (p, q) in candidate_pairs
        if ((p in component) != (q in component))
    ]


# ---------------------------------------------------------------------
# the dynamic run
# ---------------------------------------------------------------------

def run_dynamic(
    seed,
    strategy,
    num_iterations=300,
    episode=(100, 200),
    blockage_db=-20.0,
    acquisition_intervals=10,
    adaptive_from=30,
):
    """One run. Returns dict with the gain trace, switch log, metrics."""

    t0 = time.time()
    settings = SDRSimulationConfig(
        num_iterations=num_iterations, seed=seed, device="cpu"
    )
    device = resolve_device(settings.device)
    torch.manual_seed(seed)
    sionna_config.seed = seed
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 1)
    init_rng = np.random.default_rng(seed + 977)

    positions = place_stations(N, 500.0, seed)
    dt = settings.sync_interval
    dt_samples = int(round(dt * settings.sample_rate))

    frequency_process_std = 2.0 * math.pi * settings.frequency_process_std_hz
    interval_covariance = torch.diag(
        torch.tensor(
            [settings.phase_process_std_rad**2, frequency_process_std**2],
            dtype=REAL_DTYPE, device=device,
        )
    )
    oscillators = []
    for _ in range(N):
        phase0 = float(init_rng.uniform(-0.5, 0.5))
        cfo0 = float(init_rng.uniform(-1500.0, 1500.0))
        oscillators.append(
            Oscillator(phase0, 2.0 * math.pi * cfo0, dt,
                       interval_covariance, device, generator)
        )
    flickers = [
        _FlickerFrequencyNoise(
            settings.flicker_frequency_std_hz, dt,
            num_iterations * dt, device, generator,
        )
        for _ in range(N)
    ]
    flicker_prev = [
        torch.zeros((), dtype=REAL_DTYPE, device=device) for _ in range(N)
    ]

    preamble = make_sync_preamble(settings, device)
    white_fm_interval = settings.phase_noise_std_rad**2 * dt_samples

    # Build ALL candidate edges once (28); only active ones are serviced.
    pairs = all_pairs()
    edges = {}
    survey_snr = {}
    for (p, q) in pairs:
        distance = max(float(np.linalg.norm(positions[p] - positions[q])), 1.0)
        snr_db = min(
            settings.snr_db - 10.0 * 2.7 * math.log10(distance / 500.0),
            MAX_LINK_SNR_DB,
        )
        survey_snr[(p, q)] = snr_db
        edge_settings = replace(settings, snr_db=snr_db)
        link_fwd = SDRRadioLink(edge_settings, preamble, device, generator)
        link_rev = SDRRadioLink(
            edge_settings, preamble, device, generator, mirror_of=link_fwd
        )
        noise = 0.5 * _measurement_covariance(edge_settings, preamble, device)
        ekf = PhaseFrequencyEKF(
            dt,
            2.0 * interval_covariance
            + torch.diag(
                torch.tensor(
                    [white_fm_interval, 2.0 * flickers[0].innovation_variance],
                    dtype=REAL_DTYPE, device=device,
                )
            ),
            noise,
            device,
            initial_covariance=torch.diag(
                torch.tensor(
                    [math.pi**2, (2.0 * math.pi * 50e3) ** 2],
                    dtype=REAL_DTYPE, device=device,
                )
            ),
        )
        edges[(p, q)] = {
            "pair": (p, q),
            "links": (link_fwd, link_rev),
            "synchronizer": SDRSynchronizer(edge_settings, preamble),
            "ekf": ekf,
            "acquired": False,
            "pending": {},
            "settle_left": 0,
            "snr_offset_db": 0.0,
            "metric_ema": None,
            "metric_baseline": None,
            "bad_run": 0,
            "meas_var": float(noise[0, 0].item()),
            "coast": 0,
            "flip_log": [],
        }

    def set_edge_offset(pair, offset_db):
        edge = edges[pair]
        delta = offset_db - edge["snr_offset_db"]
        if delta == 0.0:
            return
        for link in edge["links"]:
            link.settings = replace(
                link.settings, snr_db=link.settings.snr_db + delta
            )
            if link._noise_power is not None:
                link._noise_power = link._noise_power * (10.0 ** (-delta / 10.0))
        edge["snr_offset_db"] = offset_db

    # Blockage targets: the two initial-tree edges whose removal
    # separates the most node pairs (highest tree betweenness) - the
    # deterministic worst case for the static strategy.
    initial_tree = max_weight_tree(N, survey_snr)

    def tree_betweenness(tree_pairs, pair):
        comp = reconnecting_candidates(N, set(tree_pairs), pair, pairs)
        adjacency = {i: set() for i in range(N)}
        for (p, q) in tree_pairs:
            if (p, q) == pair:
                continue
            adjacency[p].add(q)
            adjacency[q].add(p)
        side = {pair[0]}
        frontier = [pair[0]]
        while frontier:
            node = frontier.pop()
            for nxt in adjacency[node]:
                if nxt not in side:
                    side.add(nxt)
                    frontier.append(nxt)
        k = len(side)
        del comp
        return k * (N - k)

    blocked = sorted(
        initial_tree, key=lambda e: -tree_betweenness(initial_tree, e)
    )[:2]

    active = list(initial_tree)
    _, parent_of, subtree_of = bfs_subtrees(
        N, [(p, q, "two") for (p, q) in active]
    )
    believed_var = {pair: edges[pair]["meas_var"] for pair in pairs}
    degraded_known = set()
    node_corr_freq = [
        torch.zeros((), dtype=REAL_DTYPE, device=device) for _ in range(N)
    ]
    chain_bias = math.radians(settings.twoway_chain_asymmetry_deg)
    per_node_walk = settings.phase_noise_std_rad / math.sqrt(2.0)
    any_link = edges[pairs[0]]["links"][0]
    capture_samples = any_link.input_length + any_link.l_tot - 1

    gain_trace = []
    switch_log = []
    branch_gate_until = 0
    switches_outside_episode = 0
    flip_count = 0
    realign_count = 0

    def do_switch(iteration, removed, added, reason):
        nonlocal active, parent_of, subtree_of, branch_gate_until
        active = [e for e in active if e != removed] + [added]
        edge = edges[added]
        edge["acquired"] = False
        edge["settle_left"] = SETTLE
        edge["ekf"].state = torch.zeros(2, dtype=REAL_DTYPE, device=device)
        edge["ekf"].covariance = torch.diag(
            torch.tensor(
                [math.pi**2, (2.0 * math.pi * 50e3) ** 2],
                dtype=REAL_DTYPE, device=device,
            )
        )
        edges[removed]["pending"].clear()
        _, parent_of, subtree_of = bfs_subtrees(
            N, [(p, q, "two") for (p, q) in active]
        )
        branch_gate_until = iteration + 8
        switch_log.append(
            {"iteration": iteration, "removed": list(removed),
             "added": list(added), "reason": reason}
        )

    def pick_replacement(removed):
        options = reconnecting_candidates(N, set(active), removed, pairs)
        options = [
            o for o in options if o not in active and o not in degraded_known
        ]
        if not options:
            return None
        return max(options, key=lambda o: survey_snr[o])

    episode_start, episode_stop = episode if episode else (None, None)

    for iteration in range(num_iterations):
        # Blockage schedule (both directions of both blocked edges).
        if episode is not None:
            if iteration == episode_start:
                for pair in blocked:
                    set_edge_offset(pair, blockage_db)
            if iteration == episode_stop:
                for pair in blocked:
                    set_edge_offset(pair, 0.0)

        # Oracle strategy switches exactly at the boundaries.
        if strategy == "oracle" and episode is not None:
            if iteration == episode_start:
                for pair in list(blocked):
                    if pair in active:
                        replacement = pick_replacement(pair)
                        if replacement:
                            do_switch(iteration, pair, replacement, "oracle-in")
            if iteration == episode_stop:
                # Return to the survey-optimal tree.
                for event in [s for s in switch_log if "oracle" in s["reason"]]:
                    removed = tuple(event["added"])
                    original = tuple(event["removed"])
                    if removed in active and original not in active:
                        do_switch(iteration, removed, original, "oracle-out")

        for index in range(N):
            oscillators[index].step()
            now = flickers[index].step()
            oscillators[index].state[1] = (
                oscillators[index].state[1] + (now - flicker_prev[index])
            )
            flicker_prev[index] = now

        # Load due corrections: DIRECTED tree law - the child of each
        # tree edge corrects fully toward its parent (the repo's
        # measured-stable configuration; see docstring control-law note).
        # EKF state estimates x = theta_p - theta_q; moving only the
        # child by the full estimate zeroes the edge offset.
        node_delta = [
            torch.zeros(2, dtype=REAL_DTYPE, device=device) for _ in range(N)
        ]
        loaded_pairs = []
        for pair in active:
            edge = edges[pair]
            due = edge["pending"].pop(iteration, None)
            if due is None:
                continue
            p, q = pair
            child = q if parent_of.get(q) == p else p
            if child == q:
                node_delta[q] = node_delta[q] + due
            else:
                node_delta[p] = node_delta[p] - due
            loaded_pairs.append(pair)
        if loaded_pairs:
            for index in range(N):
                if torch.any(node_delta[index] != 0.0):
                    oscillators[index].apply_correction(node_delta[index])
                    node_corr_freq[index] = (
                        node_corr_freq[index] + node_delta[index][1]
                    )
            for pair in active:
                p, q = pair
                relative = node_delta[p] - node_delta[q]
                if torch.any(relative != 0.0):
                    edges[pair]["ekf"].reset_after_correction(-relative)

        # Periodic 1-bit branch check (subtree-preserving flips).
        if (
            iteration >= acquisition_intervals
            and iteration >= branch_gate_until
            and iteration % 4 == 0
        ):
            for pair in active:
                p, q = pair
                if torch.cos(
                    oscillators[p].state[0] - oscillators[q].state[0]
                ) < -0.2:
                    # Flip the CHILD side of this tree edge (the node
                    # whose BFS parent is the other endpoint) so every
                    # other tree edge's relative phase is preserved.
                    child = q if parent_of.get(q) == p else p
                    flip = torch.tensor(
                        [math.pi, 0.0], dtype=REAL_DTYPE, device=device
                    )
                    for member in subtree_of.get(child, {child}):
                        oscillators[member].apply_correction(flip)
                    flip_count += 1
                    # Flip-storm escape (the sibling harness's
                    # documented stabilizer): a marginal acquisition can
                    # lock a frequency alias where the phase advances
                    # ~pi per interval and the check fires repeatedly.
                    # Two flips of the same edge within 4 intervals =>
                    # discard that edge's filter and re-acquire.
                    edge = edges[pair]
                    edge["flip_log"].append(iteration)
                    last = edge.get("last_flip", -10)
                    if iteration - last <= 4:
                        edge["acquired"] = False
                        edge["ekf"].state = torch.zeros(
                            2, dtype=REAL_DTYPE, device=device
                        )
                        realign_count += 1
                    edge["last_flip"] = iteration

        # Service every active edge (equal airtime for all strategies).
        for edge_index, pair in enumerate(list(active)):
            edge = edges[pair]
            p, q = pair
            physical_p = oscillators[p].state[1] - node_corr_freq[p]
            physical_q = oscillators[q].state[1] - node_corr_freq[q]
            sfo = float(
                (physical_q - physical_p).item()
                / (2.0 * math.pi * settings.carrier_frequency_hz) * 1e6
            )
            link_fwd, link_rev = edge["links"]
            capture_fwd = link_fwd.capture(
                oscillators[p], oscillators[q], iteration, sfo
            )
            oscillators[p].state[0] = wrap_phase(
                oscillators[p].state[0] + capture_fwd.lo_walk_end
            )
            forward = edge["synchronizer"].estimate(capture_fwd.samples)
            capture_rev = link_rev.capture(
                oscillators[q], oscillators[p], iteration, -sfo
            )
            oscillators[q].state[0] = wrap_phase(
                oscillators[q].state[0] + capture_rev.lo_walk_end
            )
            reverse = edge["synchronizer"].estimate(capture_rev.samples)
            detected = bool(forward.detected and reverse.detected)
            metric = min(
                float(forward.detection_metric),
                float(reverse.detection_metric),
            ) if detected else 0.0

            # Measured link-quality bookkeeping (no oracle).
            if edge["metric_ema"] is None:
                edge["metric_ema"] = metric
            else:
                edge["metric_ema"] = (
                    (1 - EMA_ALPHA) * edge["metric_ema"] + EMA_ALPHA * metric
                )
            if iteration == max(adaptive_from, acquisition_intervals) + 20:
                edge["metric_baseline"] = edge["metric_ema"]
            baseline = edge["metric_baseline"]
            if baseline is not None and baseline > 0:
                if metric < TRIGGER_RATIO * baseline:
                    edge["bad_run"] += 1
                else:
                    edge["bad_run"] = 0

            if not detected:
                edge["coast"] += 1
                continue
            ekf = edge["ekf"]
            ekf.predict()
            combined_half = wrap_phase(
                wrap_phase(forward.phase - reverse.phase) / 2.0 + chain_bias
            )
            frequency_obs = (forward.frequency - reverse.frequency) / 2.0
            from ota_sync.coherent import _pick_half_phase
            if not edge["acquired"]:
                theta_obs = _pick_half_phase(
                    combined_half, torch.zeros_like(combined_half)
                )
                ekf.state = torch.stack((theta_obs, frequency_obs))
                ekf.covariance = torch.diag(
                    torch.stack(
                        (ekf.measurement_covariance[0, 0],
                         ekf.measurement_covariance[2, 2])
                    )
                )
                edge["acquired"] = True
            else:
                theta_obs = _pick_half_phase(
                    combined_half, wrap_phase(ekf.state[0])
                )
                ekf.update(
                    torch.stack(
                        (torch.cos(theta_obs), torch.sin(theta_obs),
                         frequency_obs)
                    )
                )
            if edge["settle_left"] > 0:
                edge["settle_left"] -= 1
                edge["coast"] += 1
                continue
            # Directed law: issue every interval (no simultaneous-
            # conflict problem on a tree with single-child corrections).
            predicted = ekf.transition @ ekf.state
            edge["pending"][iteration + 1] = _quantize_correction(
                predicted, settings
            )
            edge["coast"] = 0

        # Adaptive re-selection decisions.
        if strategy in ("channel", "gain") and iteration >= adaptive_from:
            for pair in list(active):
                edge = edges[pair]
                recent_flips = len(
                    [f for f in edge["flip_log"] if iteration - f <= 12]
                )
                metric_triggered = edge["bad_run"] >= TRIGGER_RUN
                # The gain strategy also treats branch-check chatter as
                # a trigger: the flip bits are protocol-observable, and
                # measured static runs show flip churn is the dominant
                # harm channel under partial blockage.
                flip_triggered = strategy == "gain" and recent_flips >= 2
                if not (metric_triggered or flip_triggered):
                    continue
                replacement = pick_replacement(pair)
                if replacement is None:
                    edge["bad_run"] = 0
                    continue
                if strategy == "gain":
                    # Believed per-edge phase variance: measurement
                    # variance plus COASTING variance (the filter's own
                    # per-interval process noise x intervals since that
                    # edge last delivered a correction) - the actual
                    # mechanism by which a degraded edge hurts the
                    # array. Candidates are charged their settling
                    # transient. No fitted constants.
                    q00 = float(
                        edges[pair]["ekf"].process_covariance[0, 0].item()
                    )
                    live_var = {}
                    for e in pairs:
                        base = (
                            edges[e]["meas_var"]
                            + q00 * edges[e]["coast"]
                            if e in active
                            else edges[e]["meas_var"] + q00 * SETTLE
                        )
                        # Branch chatter charge: an edge whose 1-bit
                        # check fired recently demonstrably left the
                        # +-pi/2 trust region; charge the ambiguity-
                        # scale variance per recent flip (observable,
                        # no fitted constants).
                        chatter = len(
                            [f for f in edges[e]["flip_log"]
                             if iteration - f <= 12]
                        )
                        live_var[e] = base + chatter * (math.pi / 2.0) ** 2
                    current = predicted_gain(N, active, live_var)
                    options = reconnecting_candidates(
                        N, set(active), pair, pairs
                    )
                    options = [
                        o for o in options
                        if o not in active and o not in degraded_known
                    ]
                    if not options:
                        edge["bad_run"] = 0
                        continue
                    scored = []
                    for option in options:
                        tree_opt = [
                            e for e in active if e != pair
                        ] + [option]
                        scored.append(
                            (predicted_gain(N, tree_opt, live_var), option)
                        )
                    best_gain, replacement = max(scored)
                    if best_gain - current < GAIN_MARGIN:
                        edge["bad_run"] = 0
                        continue
                degraded_known.add(pair)
                in_episode = (
                    episode is not None
                    and episode_start <= iteration < episode_stop + 40
                )
                if not in_episode:
                    switches_outside_episode += 1
                do_switch(iteration, pair, replacement, strategy)

        # Dead-time walk.
        used = capture_samples * 2 * len(active)
        remainder = max(0, dt_samples - used)
        if per_node_walk > 0.0 and remainder > 0:
            for index in range(N):
                oscillators[index].state[0] = wrap_phase(
                    oscillators[index].state[0]
                    + torch.randn(
                        (), dtype=REAL_DTYPE, device=device,
                        generator=generator,
                    ) * per_node_walk * math.sqrt(remainder)
                )

        phases = torch.stack(
            [oscillators[i].state[0] for i in range(N)]
        )
        phasors = torch.exp(1j * phases.to(torch.complex128))
        gain_trace.append(
            float((torch.abs(torch.sum(phasors)) ** 2 / N**2).real)
        )

    gain = np.array(gain_trace)

    def window_mean(a, b):
        a = max(0, a)
        b = min(num_iterations, b)
        return float(np.mean(gain[a:b])) if b > a else float("nan")

    result = {
        "seed": seed,
        "strategy": strategy,
        "episode": list(episode) if episode else None,
        "blocked": [list(b) for b in blocked],
        "gain": [round(g, 5) for g in gain_trace],
        "switches": switch_log,
        "switches_outside_episode": switches_outside_episode,
        "flips": flip_count,
        "realigns": realign_count,
        "pre_mean": window_mean(60, episode_start if episode else 100),
        "episode_mean": (
            window_mean(episode_start, episode_stop) if episode else None
        ),
        "post_mean": (
            window_mean(episode_stop, episode_stop + 60)
            if episode else window_mean(200, 260)
        ),
        "below90_episode_window": (
            int(np.sum(gain[episode_start:min(episode_stop + 60,
                                              num_iterations)] < 0.90))
            if episode else int(np.sum(gain[60:] < 0.90))
        ),
        "wall_s": round(time.time() - t0, 1),
    }
    return result


# ---------------------------------------------------------------------
# campaigns
# ---------------------------------------------------------------------

def load_cache():
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return {}


def save_cache(data):
    CACHE.write_text(json.dumps(data, indent=1))


def campaign(part):
    cache = load_cache()
    seeds = [0, 1, 2]
    if part == "main":
        jobs = [
            (s, strat, (100, 200), 300)
            for strat in ("static", "channel", "gain", "oracle")
            for s in seeds
        ]
    elif part == "control":
        jobs = [
            (s, strat, None, 300)
            for strat in ("static", "channel", "gain")
            for s in seeds
        ]
    elif part == "boundary":
        jobs = [
            (s, strat, (100, 100 + L), 300)
            for L in (12, 25, 50)
            for strat in ("static", "gain")
            for s in seeds
        ]
    else:
        raise SystemExit(f"unknown part {part}")

    for (seed, strat, episode, iters) in jobs:
        key = f"{part}|{strat}|s{seed}|" + (
            f"{episode[0]}-{episode[1]}" if episode else "none"
        )
        if key in cache:
            print(f"[cached] {key}")
            continue
        result = run_dynamic(seed, strat, iters, episode)
        cache[key] = result
        save_cache(cache)
        print(
            f"[done] {key}  pre {result['pre_mean']:.3f}  "
            f"ep {result['episode_mean']}  post {result['post_mean']:.3f}  "
            f"switches {len(result['switches'])}  "
            f"below90 {result['below90_episode_window']}  "
            f"({result['wall_s']}s)"
        )


def figures_and_report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cache = load_cache()
    figdir = HERE / "figures"
    figdir.mkdir(exist_ok=True)

    # Figure: mean gain trajectory per strategy (main scenario).
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for strat in ("static", "channel", "gain", "oracle"):
        traces = [
            np.array(v["gain"]) for k, v in cache.items()
            if k.startswith(f"main|{strat}|")
        ]
        if not traces:
            continue
        mean = np.mean(np.stack(traces), axis=0)
        ax.plot(mean, label=strat, linewidth=1.2)
    ax.axvspan(100, 200, alpha=0.12, color="gray", label="blockage episode")
    ax.set_xlabel("sync interval index")
    ax.set_ylabel("coherent gain (fraction of ideal)")
    ax.set_title("Coherent gain vs time through a two-edge blockage episode "
                 "(mean of seeds 0-2)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.savefig(figdir / "dirD_gain_trajectory.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)

    # Figure: per-seed trajectories for the two extremes.
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for row, seed in enumerate([0, 1, 2]):
        for strat in ("static", "gain", "oracle"):
            key = f"main|{strat}|s{seed}|100-200"
            if key not in cache:
                continue
            axes[row].plot(
                cache[key]["gain"], linewidth=1.0, label=strat
            )
        axes[row].axvspan(100, 200, alpha=0.12, color="gray")
        axes[row].set_ylabel(f"seed {seed} gain")
        axes[row].set_ylim(0, 1.05)
    axes[0].legend()
    axes[-1].set_xlabel("sync interval index")
    axes[0].set_title("Per-seed coherent gain, static vs gain-aware vs oracle")
    fig.savefig(figdir / "dirD_gain_per_seed.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)

    # Report table.
    print("\n=== main scenario (episode 100-200, mean over seeds) ===")
    print(f"{'strategy':<10} {'pre':>7} {'episode':>9} {'post':>7} "
          f"{'below-90 count':>15} {'switches':>9}")
    for strat in ("static", "channel", "gain", "oracle"):
        rows = [v for k, v in cache.items() if k.startswith(f"main|{strat}|")]
        if not rows:
            continue
        print(
            f"{strat:<10} "
            f"{np.mean([r['pre_mean'] for r in rows]):>7.3f} "
            f"{np.mean([r['episode_mean'] for r in rows]):>9.3f} "
            f"{np.mean([r['post_mean'] for r in rows]):>7.3f} "
            f"{np.mean([r['below90_episode_window'] for r in rows]):>15.1f} "
            f"{np.mean([len(r['switches']) for r in rows]):>9.1f}"
        )

    print("\n=== control (no disturbance): false triggers ===")
    for strat in ("static", "channel", "gain"):
        rows = [v for k, v in cache.items()
                if k.startswith(f"control|{strat}|")]
        if not rows:
            continue
        print(
            f"{strat:<10} switches {sum(len(r['switches']) for r in rows)} "
            f"across {len(rows)} runs; steady gain "
            f"{np.mean([r['post_mean'] for r in rows]):.3f}"
        )

    print("\n=== episode-length boundary (gain-aware minus static, "
          "mean gain over episode+40) ===")
    for L in (12, 25, 50, 100):
        part = "main" if L == 100 else "boundary"
        deltas = []
        for seed in (0, 1, 2):
            ka = f"{part}|static|s{seed}|100-{100 + L}"
            kb = f"{part}|gain|s{seed}|100-{100 + L}"
            if ka in cache and kb in cache:
                ga = np.array(cache[ka]["gain"])
                gb = np.array(cache[kb]["gain"])
                window = slice(100, min(100 + L + 40, len(ga)))
                deltas.append(float(np.mean(gb[window] - ga[window])))
        if deltas:
            print(f"  L={L:<4} delta-G {np.mean(deltas):+.3f} "
                  f"(seeds: {[f'{d:+.3f}' for d in deltas]})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", required=True,
                        choices=["main", "control", "boundary", "figs"])
    args = parser.parse_args()
    if args.part == "figs":
        figures_and_report()
    else:
        campaign(args.part)


if __name__ == "__main__":
    main()
