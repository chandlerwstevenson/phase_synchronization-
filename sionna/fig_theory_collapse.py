"""Universality-test figure (coherence_collapse_study.py), plain
matplotlib, fully fresh grid: every cell is re-simulated into a fresh
cache file (no reads of the original study cache), then plotted.

Chunked so the ~10-minute grid can run in its own foreground call and
resume if interrupted (resume is within THIS fresh cache only):

    .venv/bin/python fig_theory_collapse.py --grid
    .venv/bin/python fig_theory_collapse.py --plot

Output: figures/studies/theory_collapse_master_curve.png
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from coherence_collapse_study import grid_cells, isotonic_fit, run_cell

FIGURES_DIR = Path(__file__).resolve().parent / "figures" / "studies"
FRESH_CACHE = FIGURES_DIR / "theory_collapse_fresh_cache.json"
FLEETS = ("custom", "sdr", "tcxo", "ocxo", "mixed")


def load_fresh():
    if FRESH_CACHE.exists():
        with open(FRESH_CACHE) as handle:
            return json.load(handle)
    return []


def run_grid() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    records = load_fresh()
    have = {
        (r["n"], r["fleet"], r["capacity"], r["seed"], r["latency"],
         round(r["budget"], 6))
        for r in records
    }
    cells = grid_cells(quick=False)
    todo = [
        cell for cell in cells
        if (cell[0], cell[1], cell[2], cell[3], cell[4],
            round(cell[5], 6)) not in have
    ]
    print(f"fresh grid: {len(cells)} cells, {len(todo)} to run")
    started = time.time()
    for index, (n, fleet, capacity, seed, latency, budget, iters) in (
        enumerate(todo)
    ):
        rec = run_cell(n, fleet, capacity, seed, latency, budget, iters)
        records.append(rec)
        with open(FRESH_CACHE, "w") as handle:
            json.dump(records, handle)
        if (index + 1) % 25 == 0 or index + 1 == len(todo):
            print(
                f"[{index + 1}/{len(todo)}] "
                f"{(time.time() - started) / 60.0:.1f} min",
                flush=True,
            )


def panel(axis, records, key, label):
    x = np.array([r[key] for r in records])
    y = np.array([r["gain"] for r in records])
    for fleet in FLEETS:
        rows = [r for r in records if r["fleet"] == fleet]
        axis.scatter(
            [r[key] for r in rows], [r["gain"] for r in rows],
            s=12, label=fleet, alpha=0.7, edgecolors="none",
        )
    fitted, r_squared = isotonic_fit(x, y)
    order = np.argsort(x)
    axis.plot(
        x[order], fitted[order], "k-", linewidth=1.4,
        label=f"isotonic fit ($R^2$ = {r_squared:.2f})",
    )
    axis.set_xlabel(label)
    axis.set_xscale("log")
    axis.grid(True)
    return r_squared


def plot() -> None:
    records = load_fresh()
    print(f"{len(records)} fresh runs")

    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    r_rho = panel(
        axes[0], records, "rho_phys",
        "sync supply / demand ratio $\\rho$",
    )
    r_naive = panel(
        axes[1], records, "naive",
        "capacity / (N$-$1)",
    )
    axes[0].set_ylabel("array coherent gain (fraction of perfect)")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[1].legend(loc="lower right", fontsize=8)
    figure.suptitle(
        "Array gain vs supply/demand ratio and vs naive normalization"
    )
    figure.subplots_adjust(top=0.88)
    figure.savefig(
        FIGURES_DIR / "theory_collapse_master_curve.png",
        dpi=200, bbox_inches="tight",
    )
    plt.close(figure)
    print(
        f"saved theory_collapse_master_curve.png "
        f"(R^2 rho={r_rho:.3f}, naive={r_naive:.3f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.grid:
        run_grid()
    if args.plot:
        plot()


if __name__ == "__main__":
    main()
