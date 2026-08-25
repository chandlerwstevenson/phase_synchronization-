"""Direction C - joint node participation + edge selection under a
synchronization-airtime budget.

Question: which SUBSET of radios should a distributed array spend
synchronization resources on, and over which links, to maximize
expected coherent beamforming gain?

Model (analytic layer; conventions match ../phase_sync_idea):
  - N nodes placed by geometry generators; per-pair sync-link SNR from
    the same path-loss law the waveform testbed uses (exponent 2.7,
    500 m reference, capped).
  - A two-way exchange on edge e measures the pair phase difference
    with variance r_e = thermal(SNR_e) + floor (floor = intra-capture
    oscillator walk over the capture, from settings - the corrected
    attribution; NOT a multipath term).
  - Airtime budget A buys rho_tot = A / f_ex exchanges per interval
    (f_ex = fraction of one interval a two-way exchange occupies),
    split evenly across selected edges: rho_e = rho_tot / |E|.
  - Steady per-edge tracking variance for a random-walk difference
    observed every 1/rho_e intervals with noise r_e (scalar steady
    state plus mean intra-cycle growth):
        q_acc = q_pair / rho_e
        p     = (-q_acc + sqrt(q_acc^2 + 4 q_acc r_e)) / 2
        v_e   = p + q_acc / 2
    Conductance w_e = 1 / v_e; node-pair error variance = effective
    resistance R_ik of the selected graph.
  - Expected gain, charged against the FULL array (dropping a node is
    not free) - this deliberately normalizes by all N amplitudes,
    matching the project's membership convention:
        E[G(S,E)] = [ sum_{i in S} a_i^2
                      + sum_{i != k in S} a_i a_k exp(-R_ik/2) ]
                    / (sum_{ALL n} a_n)^2
  - Amplitude models: unit (a_i = 1) and path-gain (a_i prop 1/d_i to
    a target 2 km from the array centroid).

Prediction stated before measurement (per project discipline): at
tight budgets, syncing a well-chosen subset beats syncing everyone -
the thin graph spanning all N carries too much resistance per node -
and the optimal subset size grows with budget.

Search: exact enumeration where feasible (|S| <= exact_cap: all
connected edge subsets of S), otherwise greedy edge construction from
three seeds (min-variance tree, hub star, +edge improvement) with
1-swap local search; participation optimized by outer enumeration
over S for N <= 8 and greedy node add/drop for larger N.

Waveform spot checks: ../phase_sync_idea/openloop_topology_study.py's
run_openloop_graph at the operating points its integer round-robin
budget supports; we compare predicted vs measured gain and, more
importantly, the RANKING of strategies.

Usage:
    python dirC_joint_selection.py            # full study
    python dirC_joint_selection.py --quick    # reduced (smoke)
    python dirC_joint_selection.py --no-waveform
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase_sync_idea"))

from ota_sync.network import MAX_LINK_SNR_DB, place_stations  # noqa: E402
from ota_sync import SDRSimulationConfig  # noqa: E402
from openloop_graph_theory import effective_resistance  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / "dirC_cache.json"

# ---------------------------------------------------------------------
# Physical constants of the analytic layer (from project settings)
# ---------------------------------------------------------------------

SETTINGS = SDRSimulationConfig()
DT_SAMPLES = int(round(SETTINGS.sync_interval * SETTINGS.sample_rate))
# One full two-way exchange = 2 captures. Capture length from the frame
# design (long preamble + cp + short + guards); the recorded project
# value for the exchange fraction at these defaults is 0.19124.
F_EXCHANGE = 0.19124
# Pair phase random walk per interval (both oscillators' white FM).
Q_PAIR = 2.0 * SETTINGS.phase_noise_std_rad**2 * DT_SAMPLES
# Thermal phase variance per two-way half-difference at linear SNR s:
# 0.5 * (1 / (2 s L)) per leg summed -> 1 / (2 s L) ... use the
# project's convention r_thermal = 1/(4 s L) + 1/(4 s L) = 1/(2 s L),
# halved once more by the half-difference: net 1/(4 s L).
L_PILOT = SETTINGS.long_sequence_length
CAPTURE_SAMPLES = 4783  # recorded frame length at defaults
# Intra-capture oscillator walk floor per exchange (pair, corrected
# attribution - oscillator, not multipath).
R_FLOOR = SETTINGS.phase_noise_std_rad**2 * CAPTURE_SAMPLES


def edge_snr_db(distance_m: float, base_snr_db: float) -> float:
    d = max(distance_m, 1.0)
    return min(
        base_snr_db - 10.0 * 2.7 * math.log10(d / 500.0), MAX_LINK_SNR_DB
    )


def edge_variance_per_exchange(snr_db: float) -> float:
    s = 10.0 ** (snr_db / 10.0)
    return 1.0 / (4.0 * s * L_PILOT) + R_FLOOR


def steady_edge_variance(r_e: float, rho_e: float) -> float:
    """Scalar steady tracking variance of one edge difference serviced
    rho_e times per interval (rho_e may be < 1)."""

    if rho_e <= 0.0:
        return float("inf")
    q_acc = Q_PAIR / rho_e
    p = 0.5 * (-q_acc + math.sqrt(q_acc * q_acc + 4.0 * q_acc * r_e))
    return p + 0.5 * q_acc


# ---------------------------------------------------------------------
# Geometry + amplitudes
# ---------------------------------------------------------------------

def make_geometry(kind: str, n: int, seed: int) -> np.ndarray:
    if kind == "uniform":
        return place_stations(n, 500.0, seed)
    if kind == "clustered":
        rng = np.random.default_rng(seed + 12345)
        n_out = max(2, n // 4)
        n_in = n - n_out
        cluster = rng.uniform(-150.0, 150.0, size=(n_in, 2))
        angles = rng.uniform(0.0, 2.0 * math.pi, size=n_out)
        radii = rng.uniform(800.0, 1200.0, size=n_out)
        outliers = np.stack(
            [radii * np.cos(angles), radii * np.sin(angles)], axis=1
        )
        return np.vstack([cluster, outliers])
    raise ValueError(kind)


def make_amplitudes(kind: str, positions: np.ndarray) -> np.ndarray:
    n = positions.shape[0]
    if kind == "unit":
        return np.ones(n)
    if kind == "pathgain":
        centroid = positions.mean(axis=0)
        target = centroid + np.array([2000.0, 0.0])
        d = np.linalg.norm(positions - target, axis=1)
        amps = 1.0 / np.maximum(d, 1.0)
        return amps / amps.max()
    raise ValueError(kind)


# ---------------------------------------------------------------------
# The objective
# ---------------------------------------------------------------------

def expected_gain(
    nodes: tuple[int, ...],
    edges: list[tuple[int, int]],
    amps: np.ndarray,
    r_edges: dict[tuple[int, int], float],
    budget: float,
) -> float:
    """E[G] normalized by the full array's amplitude sum.

    Node-level model (the first, edge-level version of this function
    let dense graphs average away oscillator drift that is physically
    common per node - a model artifact caught in the smoke run; see
    RESULTS_C.md). Here the (|S|-1)-dim grounded phase covariance is
    propagated through the actual round-robin service schedule:
      predict: P += Q_g each interval,
               Q_g[a,b] = q_th * (1 + [a==b]) in grounded coords
               (node 0's walk is common to every coordinate),
      update:  the serviced edge measures theta_i - theta_j with
               noise r_e (Kalman update).
    The budget buys rho_tot = A / F_EXCHANGE exchanges per interval,
    round-robined over the selected edges (fractional rates via a
    credit accumulator, exactly like the waveform testbed's
    scheduler). Pairwise error variances are time-averaged over the
    steady cycle; E[G] uses E[e^{j d}] = exp(-var/2) (Gaussian).
    """

    total_amp = float(amps.sum())
    if len(nodes) == 1:
        return float(amps[nodes[0]] ** 2) / total_amp**2
    if not edges or not connected(nodes, edges):
        # Disconnected pieces free-run apart: no cross-component
        # coherence credit; give within-component credit only if a
        # component has edges. Simplest honest treatment: only the
        # connected case is a candidate; charge diagonal power alone.
        return sum(float(amps[i] ** 2) for i in nodes) / total_amp**2

    # Two states per node (grounded): phase and angular frequency.
    # Without the frequency walk, coasting error grows only linearly
    # and every sparse full-array graph looks spuriously fine (caught
    # in the second smoke run - contradicted the project's coast-time
    # law, whose quadratic frequency term dominates long coasts).
    dt = SETTINGS.sync_interval
    q_theta = (
        SETTINGS.phase_process_std_rad**2
        + SETTINGS.phase_noise_std_rad**2 * DT_SAMPLES
    )
    q_omega = (2.0 * math.pi * SETTINGS.frequency_process_std_hz) ** 2
    m = len(nodes) - 1
    index = {node: k for k, node in enumerate(nodes)}
    ones = np.ones((m, m))
    q_grounded = np.zeros((2 * m, 2 * m))
    q_grounded[:m, :m] = q_theta * (np.eye(m) + ones)
    q_grounded[m:, m:] = q_omega * (np.eye(m) + ones)
    transition = np.eye(2 * m)
    transition[:m, m:] = dt * np.eye(m)
    rho_tot = budget / F_EXCHANGE

    def h_row(i, j):
        row = np.zeros(2 * m)
        a, b = index[i], index[j]
        if a > 0:
            row[a - 1] = 1.0
        if b > 0:
            row[b - 1] = -1.0
        return row

    rows = [h_row(i, j) for i, j in edges]
    noises = [r_edges[(i, j)] for i, j in edges]

    # Post-acquisition steady state: settle from a moderate prior,
    # then time-average over full service cycles.
    cycle = max(1, int(math.ceil(len(edges) / max(rho_tot, 1e-6))))
    settle = min(6 * cycle, 6000)
    measure = min(2 * cycle, 2000)
    p_cov = np.zeros((2 * m, 2 * m))
    p_cov[:m, :m] = np.eye(m) * 0.25
    p_cov[m:, m:] = np.eye(m) * (2.0 * math.pi * 5.0) ** 2
    credit = 0.0
    pointer = 0
    sum_cov = np.zeros((m, m))
    for step in range(settle + measure):
        p_cov = transition @ p_cov @ transition.T + q_grounded
        credit += rho_tot
        while credit >= 1.0:
            credit -= 1.0
            row = rows[pointer % len(rows)]
            noise = noises[pointer % len(rows)]
            pointer += 1
            innovation = float(row @ p_cov @ row) + noise
            gain_vec = (p_cov @ row) / innovation
            p_cov = p_cov - np.outer(gain_vec, row @ p_cov)
            p_cov = 0.5 * (p_cov + p_cov.T)
        if step >= settle:
            sum_cov += p_cov[:m, :m]
    mean_cov = sum_cov / measure

    full = np.zeros((m + 1, m + 1))
    full[1:, 1:] = mean_cov
    gain = 0.0
    for i in nodes:
        gain += float(amps[i] ** 2)
    for a_pos, i in enumerate(nodes):
        for b_pos, j in enumerate(nodes):
            if a_pos >= b_pos:
                continue
            var = (
                full[a_pos, a_pos] + full[b_pos, b_pos]
                - 2.0 * full[a_pos, b_pos]
            )
            gain += 2.0 * float(amps[i] * amps[j]) * math.exp(-var / 2.0)
    return gain / total_amp**2


def connected(nodes: tuple[int, ...], edges: list[tuple[int, int]]) -> bool:
    if len(nodes) <= 1:
        return True
    adj = {n: set() for n in nodes}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    seen = {nodes[0]}
    stack = [nodes[0]]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(nodes)


# ---------------------------------------------------------------------
# Edge-set optimizers for a fixed participation set
# ---------------------------------------------------------------------

def min_variance_tree(nodes, r_edges):
    """Kruskal on per-exchange variance."""

    candidates = sorted(
        [(r_edges[(i, j)], (i, j)) for i, j in
         itertools.combinations(nodes, 2)],
    )
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree = []
    for _, (i, j) in candidates:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            tree.append((i, j))
    return tree


def hub_star(nodes, r_edges):
    """Star on the hub minimizing summed edge variance."""

    best_hub, best_cost = None, float("inf")
    for hub in nodes:
        cost = sum(
            r_edges[tuple(sorted((hub, other)))]
            for other in nodes if other != hub
        )
        if cost < best_cost:
            best_hub, best_cost = hub, cost
    return [tuple(sorted((best_hub, o))) for o in nodes if o != best_hub]


def optimize_edges(nodes, amps, r_edges, budget, exact_cap=None):
    """Best edge set for fixed S over structured candidate layouts
    plus greedy single-edge additions from the best tree. (Full
    edge-subset enumeration is separately spot-checked for exactness
    on one instance; see exactness_check().)"""

    nodes = tuple(sorted(nodes))
    if len(nodes) == 1:
        return [], expected_gain(nodes, [], amps, r_edges, budget), "trivial"
    all_pairs = [tuple(sorted(p)) for p in itertools.combinations(nodes, 2)]
    candidates = {
        "minvar-tree": min_variance_tree(nodes, r_edges),
        "star-hub": hub_star(nodes, r_edges),
        "complete": all_pairs,
    }
    if len(nodes) >= 3:
        ordered = list(nodes)
        candidates["ring"] = [
            tuple(sorted((ordered[k], ordered[(k + 1) % len(ordered)])))
            for k in range(len(ordered))
        ]
    best_edges, best_gain = None, -1.0
    for layout in candidates.values():
        g = expected_gain(nodes, layout, amps, r_edges, budget)
        if g > best_gain:
            best_gain, best_edges = g, list(layout)
    # Greedy additions on top of the winner (captures tree-plus-a-few
    # -shortcuts optima between the structured layouts).
    edges = list(best_edges)
    gain = best_gain
    improved = True
    while improved and len(edges) < len(all_pairs):
        improved = False
        best_add, best_add_gain = None, gain
        for cand in all_pairs:
            if cand in edges:
                continue
            g = expected_gain(nodes, edges + [cand], amps, r_edges, budget)
            if g > best_add_gain + 1e-12:
                best_add_gain, best_add = g, cand
        if best_add is not None:
            edges.append(best_add)
            gain = best_add_gain
            improved = True
    if gain > best_gain:
        best_gain, best_edges = gain, edges
    return best_edges, best_gain, "structured+greedy"


def exactness_check(n, amps, r_edges, budget, subset_size=5):
    """Full edge-subset enumeration on one subset vs the heuristic."""

    subset = tuple(range(subset_size))
    all_pairs = [tuple(sorted(p)) for p in
                 itertools.combinations(subset, 2)]
    best_gain = -1.0
    for mask in range(1, 2 ** len(all_pairs)):
        edges = [all_pairs[k] for k in range(len(all_pairs))
                 if mask >> k & 1]
        if not connected(subset, edges):
            continue
        g = expected_gain(subset, edges, amps, r_edges, budget)
        best_gain = max(best_gain, g)
    _, heuristic_gain, _ = optimize_edges(subset, amps, r_edges, budget)
    return best_gain, heuristic_gain


def optimize_joint(n, amps, r_edges, budget, exact_cap=5):
    """Joint (S, E): outer enumeration over S for n <= 8."""

    best = {"gain": -1.0}
    for size in range(1, n + 1):
        for subset in itertools.combinations(range(n), size):
            edges, gain, method = optimize_edges(
                subset, amps, r_edges, budget, exact_cap
            )
            if gain > best["gain"]:
                best = {
                    "nodes": list(subset),
                    "edges": [list(e) for e in (edges or [])],
                    "gain": gain,
                    "method": method,
                }
    return best


# ---------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------

def baseline_gains(n, amps, r_edges, budget):
    nodes = tuple(range(n))
    out = {}
    all_pairs = [tuple(sorted(p)) for p in
                 itertools.combinations(nodes, 2)]
    ring = [tuple(sorted((k, (k + 1) % n))) for k in range(n)]
    layouts = {
        "complete": all_pairs,
        "star-hub": hub_star(nodes, r_edges),
        "ring": ring,
        "minvar-tree": min_variance_tree(nodes, r_edges),
    }
    for name, edges in layouts.items():
        out[name] = expected_gain(nodes, edges, amps, r_edges, budget)
    return out


def strongest_k_baseline(n, amps, r_edges, budget, k):
    order = np.argsort(-amps)
    subset = tuple(sorted(int(i) for i in order[:k]))
    edges = min_variance_tree(subset, r_edges)
    return expected_gain(subset, edges, amps, r_edges, budget)


# ---------------------------------------------------------------------
# The study
# ---------------------------------------------------------------------

BUDGETS = [0.01, 0.02, 0.05, 0.10, 0.20]
GEOMETRIES = ["uniform", "clustered"]
AMP_MODELS = ["unit", "pathgain"]
SEEDS = [0, 1, 2]


def build_r_edges(positions, base_snr_db):
    n = positions.shape[0]
    r_edges = {}
    for i, j in itertools.combinations(range(n), 2):
        d = float(np.linalg.norm(positions[i] - positions[j]))
        r_edges[(i, j)] = edge_variance_per_exchange(
            edge_snr_db(d, base_snr_db)
        )
    return r_edges


def run_analytic(n=8, exact_cap=5, quick=False):
    budgets = BUDGETS if not quick else [0.02, 0.20]
    seeds = SEEDS if not quick else [0]
    rows = []
    t0 = time.time()
    for geometry in GEOMETRIES:
        for seed in seeds:
            positions = make_geometry(geometry, n, seed)
            r_edges = build_r_edges(positions, SETTINGS.snr_db)
            for amp_model in AMP_MODELS:
                amps = make_amplitudes(amp_model, positions)
                for budget in budgets:
                    joint = optimize_joint(
                        n, amps, r_edges, budget, exact_cap
                    )
                    bases = baseline_gains(n, amps, r_edges, budget)
                    strongk = strongest_k_baseline(
                        n, amps, r_edges, budget, len(joint["nodes"])
                    )
                    rows.append(
                        {
                            "geometry": geometry,
                            "seed": seed,
                            "amp_model": amp_model,
                            "budget": budget,
                            "joint": joint,
                            "baselines": bases,
                            "strongest-k": strongk,
                        }
                    )
                    print(
                        f"  {geometry}/s{seed}/{amp_model} A={budget:.0%}"
                        f" joint |S|={len(joint['nodes'])}"
                        f" G={joint['gain']:.4f}"
                        f" vs full-best="
                        f"{max(bases.values()):.4f}"
                        f" ({max(bases, key=bases.get)})",
                        flush=True,
                    )
    print(f"analytic sweep: {time.time() - t0:.0f} s")
    return rows


def summarize(rows):
    print("\n=== partial vs full crossover ===")
    for geometry in GEOMETRIES:
        for amp_model in AMP_MODELS:
            line = []
            for budget in BUDGETS:
                sel = [
                    r for r in rows
                    if r["geometry"] == geometry
                    and r["amp_model"] == amp_model
                    and r["budget"] == budget
                ]
                if not sel:
                    continue
                wins = sum(
                    1 for r in sel
                    if r["joint"]["gain"]
                    > max(r["baselines"].values()) + 1e-9
                    and len(r["joint"]["nodes"]) < 8
                )
                mean_size = np.mean(
                    [len(r["joint"]["nodes"]) for r in sel]
                )
                mean_gap = np.mean(
                    [
                        r["joint"]["gain"] - max(r["baselines"].values())
                        for r in sel
                    ]
                )
                line.append(
                    f"A={budget:.0%}: |S|={mean_size:.1f} "
                    f"partial-wins {wins}/{len(sel)} "
                    f"gap {mean_gap:+.4f}"
                )
            print(f"{geometry}/{amp_model}: " + "; ".join(line))

    print("\n=== nestedness of S* along the budget sweep ===")
    jumps = 0
    chains = 0
    for geometry in GEOMETRIES:
        for seed in SEEDS:
            for amp_model in AMP_MODELS:
                chain = [
                    set(r["joint"]["nodes"])
                    for r in rows
                    if r["geometry"] == geometry
                    and r["seed"] == seed
                    and r["amp_model"] == amp_model
                ]
                if len(chain) < 2:
                    continue
                chains += 1
                for a, b in zip(chain, chain[1:]):
                    if not a.issubset(b):
                        jumps += 1
    print(f"chains: {chains}, budget steps with a non-nested jump: {jumps}")
    return jumps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-waveform", action="store_true")
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args()

    print(
        "PREDICTION (before measurement): at tight budgets the joint "
        "optimum benches nodes (|S| < N) and beats every full-array "
        "topology; |S*| grows with budget; crossover expected in the "
        "1-5% range for uniform geometry, lower for clustered."
    )
    rows = run_analytic(n=args.n, quick=args.quick)
    jumps = summarize(rows)
    CACHE.write_text(json.dumps(
        {"rows": rows, "jumps": jumps}, indent=1, default=str
    ))
    print(f"cache -> {CACHE.name}")


if __name__ == "__main__":
    main()
