"""Coast-time law figures (coast_law.py), plain matplotlib, fresh runs.

Runs the full validation grid fresh every time (no cache reads), then
renders:
  figures/studies/theory_coast_predicted_vs_measured.png
  figures/studies/theory_filter_overconfidence.png

    .venv/bin/python fig_theory_coast.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from coast_law import resampling_phase_variance, run_validation_cell

FIGURES_DIR = Path(__file__).resolve().parent / "figures" / "studies"
CLASSES = ("ocxo", "tcxo", "sdr")
BUDGETS = (0.2, 0.314, 0.6)
LATENCIES = (1, 2, 4)
SEEDS = (0, 1, 2)


def run_grid():
    cells = []
    exact, total = 0, 0
    for profile in CLASSES:
        iterations = 600 if profile == "ocxo" else 150
        for budget in BUDGETS:
            for latency in LATENCIES:
                gaps, preds, residuals = [], [], []
                for seed in SEEDS:
                    _, rows = run_validation_cell(
                        profile, budget, latency, seed,
                        iterations=iterations,
                    )
                    for row in rows:
                        preds.append(row["predicted_cycle_intervals"])
                        residuals.extend(row["service_residuals"])
                        for gap in row["measured_gaps"]:
                            gaps.append(gap)
                            total += 1
                            if gap == round(
                                row["predicted_cycle_intervals"]
                            ):
                                exact += 1
                if not gaps:
                    continue
                cells.append(
                    {
                        "class": profile,
                        "budget": budget,
                        "latency": latency,
                        "pred_median": float(np.median(preds)),
                        "gap_median": float(np.median(gaps)),
                        "gap_q1": float(np.percentile(gaps, 25)),
                        "gap_q3": float(np.percentile(gaps, 75)),
                        "resid_rms": float(
                            np.sqrt(np.mean(np.square(residuals)))
                        ),
                    }
                )
                print(
                    f"{profile} B={budget} L={latency}: pred "
                    f"{np.median(preds):.1f} meas {np.median(gaps):.1f}"
                    f" ({len(gaps)} gaps)", flush=True,
                )
    return cells, exact, total


def main() -> None:
    cells, exact, total = run_grid()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with open(FIGURES_DIR / "theory_coast_data.json", "w") as handle:
        json.dump(
            {"cells": cells, "exact": exact, "total": total},
            handle, indent=1,
        )
    print(f"exact {exact}/{total} ({100 * exact / total:.2f}%)")

    # Predicted vs measured coast time, log-log identity axes.
    figure, axis = plt.subplots(figsize=(5.4, 5.0))
    limits = (0.7, 200.0)
    axis.plot(limits, limits, "k--", linewidth=1.0,
              label="prediction = measurement")
    for profile in CLASSES:
        rows = [c for c in cells if c["class"] == profile]
        x = [c["pred_median"] for c in rows]
        y = [c["gap_median"] for c in rows]
        lower = [c["gap_median"] - c["gap_q1"] for c in rows]
        upper = [c["gap_q3"] - c["gap_median"] for c in rows]
        axis.errorbar(
            x, y, yerr=[lower, upper], fmt="o", label=profile,
            capsize=2.5,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel("predicted coast time (sync intervals)")
    axis.set_ylabel("measured coast time (sync intervals)")
    axis.set_title("Predicted vs measured coast time")
    axis.legend(loc="upper left")
    axis.grid(True)
    figure.savefig(
        FIGURES_DIR / "theory_coast_predicted_vs_measured.png",
        dpi=200, bbox_inches="tight",
    )
    plt.close(figure)
    print("saved theory_coast_predicted_vs_measured.png")

    # Residual-to-budget ratio vs budget (tcxo, latency 1) with the
    # no-fit resampling-noise model as a reference curve.
    sigma_r = math.sqrt(resampling_phase_variance())
    rows = sorted(
        (c for c in cells if c["class"] == "tcxo" and c["latency"] == 1),
        key=lambda c: c["budget"],
    )
    budgets_mrad = [1e3 * c["budget"] for c in rows]
    ratios = [c["resid_rms"] / c["budget"] for c in rows]
    grid = np.linspace(0.15, 0.65, 200)
    model = np.sqrt(grid**2 + sigma_r**2) / grid

    figure, axis = plt.subplots(figsize=(5.4, 3.8))
    axis.plot(
        1e3 * grid, model, "k--", linewidth=1.2,
        label=r"$\sqrt{b^2 + \sigma_r^2}\,/\,b$ (resampling model)",
    )
    axis.plot(budgets_mrad, ratios, "o-", label="measured (tcxo, latency 1)")
    axis.set_xlabel("phase budget b (mrad)")
    axis.set_ylabel("residual at service / budget")
    axis.set_ylim(0.95, 1.55)
    axis.set_title("Residual-to-budget ratio vs phase budget")
    axis.legend(loc="upper right")
    axis.grid(True)
    figure.savefig(
        FIGURES_DIR / "theory_filter_overconfidence.png",
        dpi=200, bbox_inches="tight",
    )
    plt.close(figure)
    print("saved theory_filter_overconfidence.png")


if __name__ == "__main__":
    main()
