"""Direction A: static airtime-constrained synchronization-topology
selection, compared against the README's six baselines at equal
airtime, at waveform level.

Model (stated before any run; every constant from the sandbox physics):

  Edge quality. A candidate two-way edge (i,j) has sync SNR from the
  same path-loss law the testbed applies internally:
      snr_ij = min(20 dB - 27 log10(d_ij / 500 m), 50 dB),
  and per-exchange half-difference phase variance
      sigma_ij^2 = 0.5 * (one-way measurement covariance [0,0])
  evaluated by the sandbox's _measurement_covariance at that SNR with
  the SHORTENED pilot used throughout (long sequence 255, cyclic
  prefix 64 - the capture-model optimum region; the default 2047-sample
  pilot would make every budget below 19% unreachable).

  Airtime. Every two-way exchange costs 2 captures. A budget of B
  exchanges per interval costs
      A(B) = B * 2 * capture_samples / interval_samples,
  independent of how many edges share it (round-robin): selecting more
  edges dilutes per-edge service rate, it does not change airtime.
  Acquisition (first 10 intervals, all edges serviced) is excluded from
  the accounting, as in the sibling testbed.

  Per-edge steady variance under dilution. An edge serviced every
  m = |E|/B intervals, tracked as a scalar random walk with pair
  process noise q per interval and measurement noise sigma^2, has
  steady post-update variance
      p+ = ( -a + sqrt(a^2 + 4 a sigma^2) ) / 2,   a = m q,
  and reading-time variance averaged over the coast
      v(m) = p+ + q (m+1)/2.
  q = 2 * phase_process_std^2 + phase_noise_std^2 * interval_samples
  (the same terms the testbed's per-edge filter carries).

  Objective. Conductance w_e = 1/v_e(m_e) on selected edges, weighted
  Laplacian L_w, pairwise variance = effective resistance R_ik, and
      E[G(E)] = (1/N^2) * sum_ik exp(-R_ik(E)/2)
  (equal amplitudes; amplitude weighting is direction B's subject).
  Pairs in different components contribute 0. This is a BLUE-style
  approximation of a consensus loop with actuation latency - the
  predicted-vs-measured comparison judges it; the primary claim is
  strategy ORDERING at equal airtime, not calibration.

Strategies (all get identical airtime B):
  complete   all N(N-1)/2 edges
  star       best-hub star (hub minimizing total edge variance)
  ring       nearest-neighbour ring
  mst        minimum-variance spanning tree (with monotone path loss
             this coincides with the max-SNR tree and the geometric
             MST - one row covers all three README names)
  spectral   greedy edge addition maximizing lambda_2 of L_w with rate
             dilution, stop when no improvement
  greedy     greedy edge addition maximizing predicted E[G] with rate
             dilution, stop when no improvement

Discipline: predictions are computed and printed for every cell
BEFORE any waveform run; geometry is frozen across noise seeds (the
testbed's internal placement is monkey-patched, the project's
established outside-in pattern); >=3 seeds; randomized initial
frequency offsets are the testbed's own (no grid).

Usage:
    python dirA_selection.py            # full campaign (cached)
    python dirA_selection.py --quick    # 1 seed, 2 budgets
    python dirA_selection.py --report   # tables/figures from cache
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "phase_sync_idea"))

import torch  # noqa: E402

import openloop_topology_study as topo  # noqa: E402
from ota_sync.core import resolve_device, wrap_phase  # noqa: E402
from ota_sync.sdr import (  # noqa: E402
    SDRSimulationConfig,
    _measurement_covariance,
    make_sync_preamble,
)

HERE = Path(__file__).resolve().parent
CACHE = HERE / "dirA_cache.json"
FIGDIR = HERE / "figures"

N = 8
PILOT = dict(long_sequence_length=255, long_cp_length=64)
BUDGETS = [2, 3, 4, 5, 7]       # exchanges per interval
SEEDS = [0, 1, 2]
ITERATIONS = 150
ACQ = 10
STEADY = slice(100, ITERATIONS)  # last 50 intervals

# Branch-stability factor S(m): measured mean gain of the MST at
# per-edge service interval m (uniform geometry, seeds 0-2, this
# file's own calibration runs, recorded in RESULTS_A.md), divided by
# the resistance-layer prediction (~1.0 there). Gain declines
# GRADUALLY with m - branch crossings are probabilistic per coast and
# cascade - with large across-seed variance in the transition, at
# every pilot length 255..2047 (ruling out per-capture frequency
# noise; the driver is the frequency random walk between services).
# DISCLOSED calibration: one topology, one geometry; every OTHER
# strategy/geometry prediction is then a transfer test of this curve.
STABILITY_M = [1.0, 1.4, 1.75, 2.33, 3.5, 7.0]
STABILITY_S = [0.84, 0.69, 0.48, 0.45, 0.26, 0.22]


def stability(m: float) -> float:
    return float(np.interp(m, STABILITY_M, STABILITY_S))


# ---------------------------------------------------------------------
# geometry (frozen across noise seeds)
# ---------------------------------------------------------------------

def make_positions(kind: str) -> np.ndarray:
    if kind == "uniform":
        return topo.place_stations(N, 500.0, 7)
    if kind == "clustered":
        rng = np.random.default_rng(7)
        centers = np.array([[-400.0, 0.0], [400.0, 0.0]])
        pts = []
        for c in range(2):
            for _ in range(N // 2):
                while True:
                    p = centers[c] + rng.uniform(-80, 80, 2)
                    if all(np.linalg.norm(p - q) > 10 for q in pts):
                        pts.append(p)
                        break
        return np.array(pts)
    raise ValueError(kind)


class FrozenPlacement:
    """Monkey-patch the testbed's internal placement (outside-in)."""

    def __init__(self, positions: np.ndarray):
        self.positions = positions

    def __enter__(self):
        self._saved = topo.place_stations
        topo.place_stations = lambda n, r, s: self.positions
        return self

    def __exit__(self, *exc):
        topo.place_stations = self._saved


# ---------------------------------------------------------------------
# ex-ante edge model
# ---------------------------------------------------------------------

def base_settings(seed: int) -> SDRSimulationConfig:
    return SDRSimulationConfig(
        num_iterations=ITERATIONS, seed=seed, device="cpu", **PILOT
    )


def edge_model(positions: np.ndarray):
    """Per-pair SNR, half-difference phase variance, capture length."""

    device = resolve_device("cpu")
    settings = base_settings(0)
    preamble = make_sync_preamble(settings, device)
    snr = {}
    sigma2 = {}
    for i in range(N):
        for j in range(i + 1, N):
            d = max(float(np.linalg.norm(positions[i] - positions[j])), 1.0)
            s = min(settings.snr_db - 27.0 * math.log10(d / 500.0), 50.0)
            snr[(i, j)] = s
            oneway = _measurement_covariance(
                _replace_snr(settings, s), preamble, device
            )
            sigma2[(i, j)] = 0.5 * float(oneway[0, 0])
    # capture length from a probe link (same construction as testbed)
    from ota_sync.sdr import SDRRadioLink
    gen = torch.Generator(device=device)
    gen.manual_seed(1)
    link = SDRRadioLink(settings, preamble, device, gen)
    capture = link.input_length + link.l_tot - 1
    interval = int(round(settings.sync_interval * settings.sample_rate))
    q = (2.0 * settings.phase_process_std_rad**2
         + settings.phase_noise_std_rad**2 * interval)
    return snr, sigma2, capture, interval, q


def _replace_snr(settings, snr_db):
    from dataclasses import replace
    return replace(settings, snr_db=snr_db)


def steady_variance(m: float, q: float, sigma2: float) -> float:
    a = m * q
    p_plus = 0.5 * (-a + math.sqrt(a * a + 4.0 * a * sigma2))
    return p_plus + q * (m + 1.0) / 2.0


def predicted_gain(edges, budget, q, sigma2):
    """E[G] from the weighted-resistance model; component-aware."""

    if not edges:
        return 1.0 / N  # each node alone: only diagonal terms
    m = len(edges) / budget
    lap = np.zeros((N, N))
    for (i, j) in edges:
        w = 1.0 / steady_variance(m, q, sigma2[(i, j)])
        lap[i, i] += w
        lap[j, j] += w
        lap[i, j] -= w
        lap[j, i] -= w
    # components
    seen = [False] * N
    comps = []
    adj = {i: [] for i in range(N)}
    for (i, j) in edges:
        adj[i].append(j)
        adj[j].append(i)
    for s in range(N):
        if seen[s]:
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(comp)
    total = float(N)  # diagonal i==k terms
    for comp in comps:
        if len(comp) == 1:
            continue
        idx = np.array(comp)
        sub = lap[np.ix_(idx, idx)]
        pinv = np.linalg.pinv(sub)
        for a_, b_ in itertools.combinations(range(len(comp)), 2):
            r = pinv[a_, a_] + pinv[b_, b_] - 2.0 * pinv[a_, b_]
            total += 2.0 * math.exp(-r / 2.0)
    gain_resistance = total / (N * N)
    # Stabilized prediction: resistance layer x measured branch-
    # stability transfer curve; floored at the incoherent value.
    floor = 1.0 / N
    return max(floor, floor + (gain_resistance - floor) * stability(m))


# ---------------------------------------------------------------------
# strategies
# ---------------------------------------------------------------------

ALL_PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]


def strategy_edges(name, positions, sigma2, q, budget):
    if name == "complete":
        return list(ALL_PAIRS)
    if name == "star":
        best, hub = None, 0
        for h in range(N):
            cost = sum(sigma2[(min(h, j), max(h, j))]
                       for j in range(N) if j != h)
            if best is None or cost < best:
                best, hub = cost, h
        return [(min(hub, j), max(hub, j)) for j in range(N) if j != hub]
    if name == "ring":
        left = set(range(1, N))
        order = [0]
        while left:
            last = order[-1]
            nxt = min(left, key=lambda j: np.linalg.norm(
                positions[last] - positions[j]))
            order.append(nxt)
            left.remove(nxt)
        ring = [(min(a, b), max(a, b))
                for a, b in zip(order, order[1:] + [order[0]])]
        return sorted(set(ring))
    if name == "mst":
        parent = list(range(N))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        chosen = []
        for (i, j) in sorted(ALL_PAIRS, key=lambda e: sigma2[e]):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
                chosen.append((i, j))
        return chosen
    if name == "spectral":
        return _greedy(positions, sigma2, q, budget, objective="lambda2")
    if name == "greedy":
        return _greedy(positions, sigma2, q, budget, objective="gain")
    raise ValueError(name)


def _greedy(positions, sigma2, q, budget, objective):
    # Spectral greedy cannot start from empty (lambda_2 stays 0 until
    # the graph is connected, so no first edge ever "improves" it);
    # standard practice is to connect first, then add chords. Both
    # greedies therefore start from the minimum-variance spanning tree
    # when the budget can sustain it, else from empty (gain greedy can
    # build forests; spectral is then marked infeasible by returning
    # the tree anyway and letting the predictor score it).
    tree = strategy_edges("mst", positions, sigma2, q, budget)
    if objective == "lambda2":
        edges = list(tree)
    else:
        # Gain greedy: try both an empty start (can settle on a small
        # stable forest) and a tree start (spanning bias); keep the
        # better predicted set.
        from_empty = _greedy_run([], positions, sigma2, q, budget,
                                 objective)
        from_tree = _greedy_run(list(tree), positions, sigma2, q,
                                budget, objective)
        ge = predicted_gain(from_empty, budget, q, sigma2)
        gt = predicted_gain(from_tree, budget, q, sigma2)
        return from_empty if ge >= gt else from_tree
    return _greedy_run(edges, positions, sigma2, q, budget, objective)


def _greedy_run(edges, positions, sigma2, q, budget, objective):

    def score(es):
        if objective == "gain":
            return predicted_gain(es, budget, q, sigma2)
        if not es:
            return 0.0
        m = len(es) / budget
        lap = np.zeros((N, N))
        for (i, j) in es:
            w = 1.0 / steady_variance(m, q, sigma2[(i, j)])
            lap[i, i] += w
            lap[j, j] += w
            lap[i, j] -= w
            lap[j, i] -= w
        return float(np.sort(np.linalg.eigvalsh(lap))[1])

    current = score(edges)
    while True:
        best_gain, best_edge = 0.0, None
        for e in ALL_PAIRS:
            if e in edges:
                continue
            s = score(edges + [e])
            if s - current > best_gain + 1e-12:
                best_gain, best_edge = s - current, e
        if best_edge is None:
            return edges
        edges.append(best_edge)
        current += best_gain


# ---------------------------------------------------------------------
# submodularity check (empirical)
# ---------------------------------------------------------------------

def submodularity_check(q, sigma2, budget, trials=400, dilution=True):
    rng = np.random.default_rng(3)
    violations = 0
    for _ in range(trials):
        small = [e for e in ALL_PAIRS if rng.random() < 0.3]
        extra = [e for e in ALL_PAIRS
                 if e not in small and rng.random() < 0.4]
        big = small + extra
        rest = [e for e in ALL_PAIRS if e not in big]
        if not rest:
            continue
        e = rest[rng.integers(len(rest))]

        def g(es):
            b = budget if dilution else len(es) if es else 1
            return predicted_gain(es, b, q, sigma2)

        if (g(big + [e]) - g(big)) > (g(small + [e]) - g(small)) + 1e-12:
            violations += 1
    return violations, trials


# ---------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------

def measured_gain(node_traces: torch.Tensor, window: slice) -> float:
    theta = node_traces[:, window].to(torch.complex128)
    phasors = torch.exp(1j * theta)
    g = torch.abs(torch.sum(phasors, dim=0)) ** 2 / (N * N)
    return float(torch.mean(g.real))


def run_cell(positions, edges, budget, seed):
    from dirA_runner import run_openloop_graph_cadence
    if not edges:
        return {"gain": 1.0 / N, "detect": 1.0, "flips": 0, "wall_s": 0.0}
    spec = [(i, j, "two") for (i, j) in edges]
    with FrozenPlacement(positions):
        out = run_openloop_graph_cadence(
            base_settings(seed), N, spec,
            budget_edges_per_interval=budget,
            acquisition_intervals=ACQ,
        )
    return {
        "gain": measured_gain(out["node_traces"], STEADY),
        "detect": out["detect_rate"],
        "flips": out["flips"],
        "realigns": out["realigns"],
        "wall_s": out["wall_s"],
    }


# ---------------------------------------------------------------------
# campaign
# ---------------------------------------------------------------------

STRATEGIES = ["complete", "star", "ring", "mst", "spectral", "greedy"]
GEOMETRIES = ["uniform", "clustered"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    budgets = BUDGETS[:2] if args.quick else BUDGETS
    seeds = SEEDS[:1] if args.quick else SEEDS

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    # ---- predictions first, printed before any waveform run --------
    plans = {}
    for geom in GEOMETRIES:
        positions = make_positions(geom)
        snr, sigma2, capture, interval, q = edge_model(positions)
        s2 = {tuple(k): v for k, v in sigma2.items()} \
            if isinstance(next(iter(sigma2)), tuple) else sigma2
        print(f"\n=== geometry {geom}: pair SNR range "
              f"{min(snr.values()):.1f}..{max(snr.values()):.1f} dB, "
              f"capture {capture} samples, exchange airtime "
              f"{100 * 2 * capture / interval:.2f}%/interval ===")
        for budget in budgets:
            air = 100.0 * budget * 2 * capture / interval
            print(f"-- budget B={budget} ({air:.2f}% airtime) "
                  f"predicted E[G]:")
            for name in STRATEGIES:
                edges = strategy_edges(name, positions, s2, q, budget)
                pred = predicted_gain(edges, budget, q, s2)
                plans[(geom, name, budget)] = (edges, pred, air)
                print(f"   {name:9s} |E|={len(edges):2d} "
                      f"pred G={100 * pred:6.2f}%")
        v_dil, t = submodularity_check(q, s2, 2, dilution=True)
        v_fix, _ = submodularity_check(q, s2, 2, dilution=False)
        print(f"   submodularity violations (B=2): with dilution "
              f"{v_dil}/{t}, fixed-rate {v_fix}/{t}")

    if args.report:
        report(cache, plans, budgets)
        return

    # ---- waveform campaign -----------------------------------------
    t0 = time.time()
    for geom in GEOMETRIES:
        positions = make_positions(geom)
        for budget in budgets:
            for name in STRATEGIES:
                edges, pred, air = plans[(geom, name, budget)]
                for seed in seeds:
                    key = f"{geom}|{name}|{budget}|{seed}"
                    if key in cache:
                        continue
                    cell = run_cell(positions, edges, budget, seed)
                    cell["pred"] = pred
                    cell["airtime_pct"] = air
                    cell["num_edges"] = len(edges)
                    cache[key] = cell
                    CACHE.write_text(json.dumps(cache, indent=1))
                    print(f"  ran {key}: G={100 * cell['gain']:6.2f}% "
                          f"(pred {100 * pred:6.2f}%) detect "
                          f"{100 * cell['detect']:.0f}% "
                          f"[{cell['wall_s']:.1f}s]")
    print(f"campaign wall time {time.time() - t0:.0f}s")
    report(cache, plans, budgets)


def report(cache, plans, budgets):
    print("\n=== strategy x budget: measured gain (mean+-std over seeds), "
          "predicted alongside ===")
    for geom in GEOMETRIES:
        print(f"\n[{geom}]")
        header = "strategy   " + "".join(
            f"   B={b} ({plans[(geom, 'mst', b)][2]:.1f}%)   "
            for b in budgets)
        print(header)
        for name in STRATEGIES:
            row = f"{name:9s}"
            for b in budgets:
                vals = [cache[k]["gain"] for k in cache
                        if k.startswith(f"{geom}|{name}|{b}|")]
                pred = plans[(geom, name, b)][1]
                if vals:
                    row += (f"  {100 * np.mean(vals):5.1f}"
                            f"±{100 * np.std(vals):4.1f}"
                            f" (p{100 * pred:5.1f})")
                else:
                    row += "        --        "
            print(row)

    # figures: plain matplotlib, no in-axes text
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGDIR.mkdir(exist_ok=True)
    for geom in GEOMETRIES:
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
        for name in STRATEGIES:
            xs, ys = [], []
            for b in budgets:
                vals = [cache[k]["gain"] for k in cache
                        if k.startswith(f"{geom}|{name}|{b}|")]
                if vals:
                    xs.append(plans[(geom, name, b)][2])
                    ys.append(100 * np.mean(vals))
            ax.plot(xs, ys, "o-", label=name)
        ax.set_xlabel("synchronization airtime (% of frame)")
        ax.set_ylabel("measured coherent gain (%)")
        ax.set_title(f"Coherent gain vs sync airtime by topology "
                     f"strategy ({geom} geometry, N=8)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGDIR / f"dirA_gain_vs_budget_{geom}.png")
        plt.close(fig)
    print(f"figures written to {FIGDIR}")


if __name__ == "__main__":
    main()
