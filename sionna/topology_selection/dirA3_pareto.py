"""dirA3 - the coherence/airtime Pareto frontier per (topology, protocol).

Reviewer request: make synchronization airtime an explicit secondary
metric. For each (topology, protocol) pair we sweep the exchange
budget B (round-robin edges serviced per interval) and record realized
coherent gain vs steady-state sync airtime = B x (2 x capture / dt)
= B x 4.276% with the A2 pilot (255-sample sequence). Acquisition
(first 10 intervals, all edges serviced) is excluded from the
accounting and identical across cells.

PRE-REGISTERED PREDICTIONS (printed before any run):
  P-i   directed dominates the frontier at every airtime level on star
        and mst (its cadence sensitivity was the shallowest measured:
        96->80% vs 84->38% over m=1->3.5).
  P-ii  within the directed protocol, frontier ordering follows
        resistance: mst ~ star >= chain (chain's end-to-end resistance
        is worst).
  P-iii there is an airtime level at which no bidirectional protocol
        delivers a usable point (gain >= 50%) on any tested topology
        while directed still delivers >= 80% - quantified below as the
        airtime-advantage ratio: (lowest airtime at which the best
        bidirectional pair holds >= 80%) / (lowest airtime at which
        the best directed pair holds >= 80%).

Grid: topology {star, mst, chain} x law {symmetric, alternating,
directed} x budget B {7, 5, 4, 3, 2, 1} x seeds {0, 1, 2}, N = 8,
uniform geometry - reusing dirA2_cache.json cells where the identical
configuration was already run (cad|mst B{7,5,4,3,2}, deg|star B7);
new cells go to dirA3_cache.json only.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "phase_sync_idea"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dirA_selection import (  # noqa: E402
    ACQ,
    FrozenPlacement,
    ITERATIONS,
    edge_model,
    make_positions,
)
from dirA2_threelaw import (  # noqa: E402
    LAWS,
    N,
    STEADY,
    a2_settings,
    run_threelaw,
    topology,
)

A2_CACHE = HERE / "dirA2_cache.json"
A3_CACHE = HERE / "dirA3_cache.json"
FIGDIR = HERE / "paper_figures"

TOPOS = ("star", "mst", "chain")
BUDGETS = (7, 5, 4, 3, 2, 1)
SEEDS = (0, 1, 2)
EXCHANGE_FRACTION = 0.04276  # 2 x 1069 / 50000, computed from the A2 pilot


def airtime(budget: int) -> float:
    return budget * EXCHANGE_FRACTION


def a2_alias(topo_name: str, law: str, budget: int, seed: int) -> str | None:
    if topo_name == "mst" and budget in (7, 5, 4, 3, 2):
        return f"cad|mst|{law}|B{budget}|s{seed}"
    if topo_name == "star" and budget == 7:
        return f"deg|star|{law}|B{budget}|s{seed}"
    return None


def run_cell(a2, a3, topo_name, edges, law, budget, seed):
    alias = a2_alias(topo_name, law, budget, seed)
    if alias and alias in a2:
        return a2[alias]
    key = f"par|{topo_name}|{law}|B{budget}|s{seed}"
    if key in a3:
        return a3[key]
    positions = make_positions("uniform")
    settings = a2_settings(seed)
    spec = [(i, j, "two") for (i, j) in edges]
    with FrozenPlacement(positions):
        out = run_threelaw(
            settings, N, spec, law,
            budget_edges_per_interval=budget,
            acquisition_intervals=ACQ,
        )
    theta = out["node_traces"][:, STEADY].to(torch.complex128)
    phasors = torch.exp(1j * theta)
    gain = float(
        torch.mean(
            (torch.abs(torch.sum(phasors, dim=0)) ** 2 / (N * N)).real
        )
    )
    a3[key] = {
        "gain": gain,
        "detect": out["detect_rate"],
        "flips": out["flips"],
        "wall_s": out["wall_s"],
    }
    A3_CACHE.write_text(json.dumps(a3, indent=1))
    print(f"  {key}: gain {100*gain:.1f}% flips {out['flips']} "
          f"({out['wall_s']:.0f}s)", flush=True)
    return a3[key]


def main() -> None:
    print(__doc__.split("Grid:")[0])
    a2 = json.loads(A2_CACHE.read_text()) if A2_CACHE.exists() else {}
    a3 = json.loads(A3_CACHE.read_text()) if A3_CACHE.exists() else {}
    positions = make_positions("uniform")
    sigma2 = edge_model(positions)[1]

    frontier = {}
    for topo_name in TOPOS:
        edges = topology(topo_name, positions, sigma2)
        for law in LAWS:
            points = []
            for budget in BUDGETS:
                gains = [
                    run_cell(a2, a3, topo_name, edges, law, budget, seed)[
                        "gain"
                    ] * 100.0
                    for seed in SEEDS
                ]
                points.append(
                    (
                        100.0 * airtime(budget),
                        statistics.mean(gains),
                        statistics.stdev(gains) if len(gains) > 1 else 0.0,
                    )
                )
            frontier[(topo_name, law)] = points

    # ---- tables ----------------------------------------------------
    print("\nFrontier (airtime% -> gain% +- std):")
    for (topo_name, law), points in frontier.items():
        row = "  ".join(
            f"{a:.1f}%:{g:.1f}±{s:.1f}" for a, g, s in points
        )
        print(f"  {topo_name:>6} {law:>11}  {row}")

    print("\nEfficiency max gain/airtime (points with gain >= 50%):")
    for (topo_name, law), points in frontier.items():
        usable = [(g / a, a, g) for a, g, s in points if g >= 50.0]
        if usable:
            ratio, a, g = max(usable)
            print(f"  {topo_name:>6} {law:>11}  {ratio:.2f} %gain/%airtime "
                  f"(at {a:.1f}% -> {g:.1f}%)")
        else:
            print(f"  {topo_name:>6} {law:>11}  no usable point")

    print("\nConstrained optimum per airtime cap:")
    for cap in (5.0, 10.0, 20.0):
        best = None
        for (topo_name, law), points in frontier.items():
            for a, g, s in points:
                if a <= cap and (best is None or g > best[0]):
                    best = (g, s, a, topo_name, law)
        g, s, a, topo_name, law = best
        print(f"  T_max {cap:>4.0f}%: {topo_name}/{law} at {a:.1f}% "
              f"-> {g:.1f}±{s:.1f}%")

    threshold = 80.0
    def cheapest(law_filter):
        best = None
        for (topo_name, law), points in frontier.items():
            if not law_filter(law):
                continue
            for a, g, s in points:
                if g >= threshold and (best is None or a < best[0]):
                    best = (a, g, topo_name, law)
        return best

    direct = cheapest(lambda law: law == "directed")
    bidir = cheapest(lambda law: law != "directed")
    print(f"\nCheapest >= {threshold:.0f}% gain: directed {direct}, "
          f"bidirectional {bidir}")
    if direct and bidir:
        print(f"P-iii airtime-advantage ratio: {bidir[0]/direct[0]:.1f}x")

    # ---- figure ----------------------------------------------------
    FIGDIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    styles = {"star": "-", "mst": "--", "chain": ":"}
    colors = {"symmetric": "C0", "alternating": "C1", "directed": "C2"}
    for (topo_name, law), points in frontier.items():
        xs = [a for a, g, s in points]
        ys = [g for a, g, s in points]
        es = [s for a, g, s in points]
        ax.errorbar(
            xs, ys, yerr=es, capsize=2,
            linestyle=styles[topo_name], color=colors[law],
            marker="o", markersize=3,
            label=f"{topo_name}, {law}",
        )
    ax.set_xlabel("synchronization airtime (% of frame, steady state)")
    ax.set_ylabel("coherent gain (%)")
    ax.set_title(
        "Coherent gain vs synchronization airtime by topology and protocol "
        "(N=8, seeds 0-2)"
    )
    ax.legend(fontsize=8, ncol=3)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig3_pareto_frontier.png", dpi=200)
    print(f"\nwrote {FIGDIR / 'fig3_pareto_frontier.png'}")


if __name__ == "__main__":
    main()
