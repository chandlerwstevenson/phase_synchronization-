"""Direction C - refinement, exactness check, waveform validation,
and the figure. Companion to dirC_joint_selection.py (imports it).

1. Budget refinement {3,4,6,8}% around the observed partial-
   participation window at 5%, tracing |S*|(A).
2. Exactness spot-check: full edge-subset enumeration on a 5-node
   subset vs the structured+greedy heuristic.
3. Waveform spot checks via ../phase_sync_idea's run_openloop_graph.
   The testbed's scheduler services an integer number of edges per
   interval, so the 5% operating point is not directly reachable;
   we validate the MODEL at reachable operating points that span the
   same per-edge coast range (benign 4-interval to harsh 28-interval
   cycles), predictions printed before measurement.
4. Figure: gain vs budget, joint-optimal vs best full-array baseline.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "phase_sync_idea"))

from dirC_joint_selection import (  # noqa: E402
    BUDGETS,
    F_EXCHANGE,
    SETTINGS,
    baseline_gains,
    build_r_edges,
    exactness_check,
    expected_gain,
    make_amplitudes,
    make_geometry,
    min_variance_tree,
    optimize_joint,
)

REFINE_BUDGETS = [0.03, 0.04, 0.06, 0.08]


def part_refine():
    print("=== refinement: |S*| vs budget through the window ===")
    out = {}
    for geometry in ["uniform", "clustered"]:
        positions = make_geometry(geometry, 8, 0)
        r_edges = build_r_edges(positions, SETTINGS.snr_db)
        amps = make_amplitudes("pathgain", positions)
        line = []
        for budget in sorted(REFINE_BUDGETS + BUDGETS):
            joint = optimize_joint(8, amps, r_edges, budget)
            bases = baseline_gains(8, amps, r_edges, budget)
            line.append(
                (budget, len(joint["nodes"]), joint["gain"],
                 max(bases.values()), joint["nodes"])
            )
            print(
                f"  {geometry} A={budget:.0%}: |S*|={len(joint['nodes'])} "
                f"G*={joint['gain']:.4f} full-best={max(bases.values()):.4f}"
                f" S*={joint['nodes']}"
            )
        out[geometry] = line
    return out


def part_exact():
    print("\n=== exactness check (5-node subset, full enumeration) ===")
    positions = make_geometry("uniform", 8, 0)
    r_edges = build_r_edges(positions, SETTINGS.snr_db)
    amps = make_amplitudes("unit", positions)
    for budget in [0.05, 0.20]:
        t0 = time.time()
        exact, heur = exactness_check(8, amps, r_edges, budget)
        print(
            f"  A={budget:.0%}: exact {exact:.5f} vs heuristic "
            f"{heur:.5f} (gap {exact - heur:+.5f}, "
            f"{time.time() - t0:.0f} s)"
        )


def part_waveform(seeds=(0, 1, 2)):
    from ota_sync import SDRSimulationConfig
    from openloop_topology_study import run_openloop_graph

    print("\n=== waveform validation (predictions first) ===")
    positions = make_geometry("uniform", 8, 0)
    r_edges = build_r_edges(positions, SETTINGS.snr_db)
    amps = make_amplitudes("unit", positions)

    # Configs reachable by the integer scheduler (1 edge serviced per
    # interval; per-edge cycle = |E| intervals). Model equivalent:
    # rho_tot = 1 exchange/interval -> budget = F_EXCHANGE.
    hub = min_variance_tree(tuple(range(8)), r_edges)
    star5_nodes = tuple(range(5))
    tree5 = min_variance_tree(star5_nodes, r_edges)
    complete8 = [
        (i, j) for i in range(8) for j in range(i + 1, 8)
    ]
    configs = {
        "tree8 (7-interval cycles)": (tuple(range(8)), hub),
        "tree5 (4-interval cycles)": (star5_nodes, tree5),
        "complete8 (28-interval cycles)": (tuple(range(8)), complete8),
    }

    predictions = {}
    for name, (nodes, edges) in configs.items():
        predictions[name] = expected_gain(
            nodes, [tuple(sorted(e)) for e in edges], amps, r_edges,
            F_EXCHANGE,
        )
        print(f"  PREDICTED {name}: G = {predictions[name]:.4f}")

    results = {}
    for name, (nodes, edges) in configs.items():
        measured = []
        for seed in seeds:
            settings = SDRSimulationConfig(
                num_iterations=100, seed=seed, device="cpu"
            )
            run = run_openloop_graph(
                settings,
                8,
                [(i, j, "two") for i, j in edges],
                budget_edges_per_interval=1,
            )
            traces = run["node_traces"]
            steady = traces[:, 50:]
            phasors = np.exp(1j * np.asarray(steady))
            sel = np.zeros(8)
            sel[list(nodes)] = 1.0
            amp_vec = (amps * sel)[:, None]
            g_t = (
                np.abs((amp_vec * phasors).sum(axis=0)) ** 2
                / amps.sum() ** 2
            )
            measured.append(float(g_t.mean()))
        results[name] = (
            float(np.mean(measured)), float(np.std(measured))
        )
        print(
            f"  MEASURED  {name}: G = {results[name][0]:.4f} "
            f"± {results[name][1]:.4f}  "
            f"(pred {predictions[name]:.4f}, "
            f"ratio {results[name][0] / max(predictions[name], 1e-9):.2f})"
        )
    return predictions, results


def part_figure(refine):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    for ax, geometry in zip(axes, ["uniform", "clustered"]):
        line = refine[geometry]
        budgets = [100 * b for b, *_ in line]
        joint = [g for _, _, g, _, _ in line]
        full = [f for _, _, _, f, _ in line]
        ax.plot(budgets, full, "o-", label="best full-array topology")
        ax.plot(budgets, joint, "s-", label="joint node+edge optimum")
        ax.set_xlabel("synchronization airtime budget (%)")
        ax.set_title(f"{geometry} geometry (path-gain amplitudes)")
        ax.set_xscale("log")
    axes[0].set_ylabel("expected coherent gain")
    axes[0].legend()
    fig.tight_layout()
    out = HERE / "figures" / "dirC_gain_vs_budget.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"\nfigure -> {out.relative_to(HERE)}")


def main():
    refine = part_refine()
    part_exact()
    predictions, measured = part_waveform()
    part_figure(refine)
    (HERE / "dirC_validate_cache.json").write_text(
        json.dumps(
            {
                "refine": {k: [list(map(str, r)) for r in v]
                           for k, v in refine.items()},
                "waveform_pred": predictions,
                "waveform_meas": measured,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
