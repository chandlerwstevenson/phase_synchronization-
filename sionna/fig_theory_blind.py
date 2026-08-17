"""Blind-prediction figures, plain matplotlib, fully fresh runs.

Round 1 is the wall_prediction_study.py log (run it fresh first and
pass the log). Round-2 and N-extension measurements are regenerated
into FRESH run files (the module cache paths are redirected before any
measurement, so the original caches are never read).

    .venv/bin/python wall_prediction_study.py > wall_fresh_plain.log
    .venv/bin/python fig_theory_blind.py --wall-log wall_fresh_plain.log

Outputs:
  figures/studies/theory_blind_scorecard.png
  figures/studies/theory_blind_plateau.png
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import phase_diagram_round2 as round2
import theory_nscaling_study as nscale

FIGURES_DIR = Path(__file__).resolve().parent / "figures" / "studies"
ROUND2_FRESH = str(FIGURES_DIR / "round2_fresh_runs.json")
NSCALE_FRESH = str(FIGURES_DIR / "nscale_fresh_runs.json")


def scored_rows(frozen, runs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        hits, total, rows = round2.score(frozen, runs)
    return hits, total, rows


def category_tallies(rows):
    tallies = {"class": [0, 0], "plateau": [0, 0], "knee": [0, 0]}
    for _, verdicts, _ in rows:
        for name, hit, _ in verdicts:
            tallies[name][0] += int(hit)
            tallies[name][1] += 1
    return tallies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wall-log", type=str, required=True)
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    with open(args.wall_log) as handle:
        wall_text = handle.read()
    wall_verdicts = re.findall(r"\b(HIT|MISS)\b", wall_text)
    wall_hits = sum(1 for v in wall_verdicts if v == "HIT")
    wall_total = len(wall_verdicts)
    print(f"round 1 parsed: {wall_hits}/{wall_total}")

    # Redirect both modules' run caches to fresh files so every
    # measurement below is re-simulated.
    round2.ROUND2_PATH = ROUND2_FRESH
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        frozen_r2 = round2.freeze_predictions()
    runs_r2 = round2.measure_curves(frozen_r2)
    hits_r2, total_r2, rows_r2 = scored_rows(frozen_r2, runs_r2)
    tallies_r2 = category_tallies(rows_r2)
    print(f"round 2 fresh: {hits_r2}/{total_r2}")

    with contextlib.redirect_stdout(buffer):
        frozen_ns = nscale.freeze_nscaling_predictions()
    runs_ns = nscale.measure_nscaling(frozen_ns, path=NSCALE_FRESH)
    hits_ns, total_ns, rows_ns = scored_rows(frozen_ns, runs_ns)
    tallies_ns = category_tallies(rows_ns)
    print(f"N-extension fresh: {hits_ns}/{total_ns}")

    with open(FIGURES_DIR / "theory_blind_data.json", "w") as handle:
        json.dump(
            {
                "round1": [wall_hits, wall_total],
                "round2": [hits_r2, total_r2],
                "round2_categories": tallies_r2,
                "nextension": [hits_ns, total_ns],
                "nextension_categories": tallies_ns,
            },
            handle, indent=1,
        )

    # Scorecard: grouped horizontal bars, fraction correct.
    categories = ["overall", "classification", "plateau", "knee"]
    rounds = [
        (f"round 1 ({wall_hits}/{wall_total} overall)",
         {"overall": (wall_hits, wall_total)}),
        (f"round 2 ({hits_r2}/{total_r2} overall)",
         {
             "overall": (hits_r2, total_r2),
             "classification": tuple(tallies_r2["class"]),
             "plateau": tuple(tallies_r2["plateau"]),
             "knee": tuple(tallies_r2["knee"]),
         }),
        (f"N = 16-20 extrapolation ({hits_ns}/{total_ns} overall)",
         {
             "overall": (hits_ns, total_ns),
             "classification": tuple(tallies_ns["class"]),
             "plateau": tuple(tallies_ns["plateau"]),
             "knee": tuple(tallies_ns["knee"]),
         }),
    ]
    figure, axis = plt.subplots(figsize=(7.4, 3.9))
    bar_height = 0.24
    y_positions = np.arange(len(categories))[::-1]
    for offset, (label, values) in zip(
        (-bar_height, 0.0, bar_height), rounds
    ):
        ys, fractions = [], []
        for index, category in enumerate(categories):
            if category not in values:
                continue
            hit, total = values[category]
            ys.append(y_positions[index] - offset)
            fractions.append(hit / total)
        axis.barh(ys, fractions, height=bar_height * 0.9, label=label)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(categories)
    axis.set_xlim(0, 1.0)
    axis.set_xlabel("blind predictions correct (fraction)")
    axis.set_title("Blind prediction score by round and category")
    axis.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
        fontsize=8,
    )
    axis.grid(True, axis="x")
    figure.savefig(
        FIGURES_DIR / "theory_blind_scorecard.png",
        dpi=200, bbox_inches="tight",
    )
    plt.close(figure)
    print("saved theory_blind_scorecard.png")

    # Predicted vs measured plateau gain.
    def points(rows):
        clean, failure = [], []
        for (curve, _, measured) in rows:
            pred = curve["plateau"]
            meas = measured[max(curve["capacities"])]
            is_failure = (
                curve["fleet"] == "sdr1" or curve["latency"] >= 2
            )
            (failure if is_failure else clean).append((pred, meas))
        return clean, failure

    clean_r2, failure_r2 = points(rows_r2)
    clean_ns, _ = points(rows_ns)

    figure, axis = plt.subplots(figsize=(5.2, 5.0))
    axis.plot([0, 1], [0, 1], "k--", linewidth=1.0,
              label="prediction = measurement")
    axis.scatter(
        *zip(*clean_r2), s=40, marker="o",
        label="clean regime, N = 8-12",
    )
    axis.scatter(
        *zip(*clean_ns), s=40, marker="s",
        label="clean regime, N = 16-20",
    )
    axis.scatter(
        *zip(*failure_r2), s=40, marker="^",
        label="known failure modes",
    )
    axis.set_xlim(0.45, 1.02)
    axis.set_ylim(0.45, 1.02)
    axis.set_xlabel("predicted plateau gain (fraction of perfect)")
    axis.set_ylabel("measured plateau gain (fraction of perfect)")
    axis.set_title("Predicted vs measured plateau gain")
    axis.legend(loc="upper left", fontsize=8)
    axis.grid(True)
    figure.savefig(
        FIGURES_DIR / "theory_blind_plateau.png",
        dpi=200, bbox_inches="tight",
    )
    plt.close(figure)
    print("saved theory_blind_plateau.png")


if __name__ == "__main__":
    main()
