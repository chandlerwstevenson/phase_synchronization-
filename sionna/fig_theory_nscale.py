"""Array-size and noise-floor figures for the theory stack, plain
matplotlib, fresh runs (each part re-simulates; nothing is read from
prior study caches).

Chunked so each part runs in its own foreground call:

    .venv/bin/python fig_theory_nscale.py --part exact
    .venv/bin/python fig_theory_nscale.py --part resampling
    .venv/bin/python fig_theory_nscale.py --part doppler
    .venv/bin/python fig_theory_nscale.py --part plot   # after all three

Outputs:
  figures/studies/theory_exactness_vs_n.png
  figures/studies/theory_resampling_vs_n.png
  figures/studies/theory_doppler_structure.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path(__file__).resolve().parent / "figures" / "studies"
DATA_PATH = FIGURES_DIR / "theory_nscale_data.json"
STATIONS = (6, 10, 14)


def load_data() -> dict:
    if DATA_PATH.exists():
        with open(DATA_PATH) as handle:
            return json.load(handle)
    return {}


def store(key: str, value) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    data[key] = value
    with open(DATA_PATH, "w") as handle:
        json.dump(data, handle, indent=1)


def part_exact() -> None:
    from coast_law import run_validation_cell

    results = []
    for n in STATIONS:
        exact, total = 0, 0
        for profile in ("ocxo", "tcxo"):
            iterations = 600 if profile == "ocxo" else 150
            for budget in (0.2, 0.6):
                for seed in (0, 1, 2):
                    _, rows = run_validation_cell(
                        profile, budget, 1, seed, num_stations=n,
                        iterations=iterations,
                    )
                    for row in rows:
                        for gap in row["measured_gaps"]:
                            total += 1
                            if gap == round(
                                row["predicted_cycle_intervals"]
                            ):
                                exact += 1
        results.append({"n": n, "exact": exact, "total": total})
        print(f"N={n}: {exact}/{total} exact "
              f"({100 * exact / total:.1f}%)", flush=True)
    store("exact", results)


def part_resampling() -> None:
    from doppler_coast_study import (
        bias_structure,
        reciprocity_bias_series,
        run_star_instrumented,
    )
    from ota_sync import SDRSimulationConfig

    results = []
    for n in STATIONS:
        sigmas, d1s = [], []
        for seed in (0, 1):
            settings = SDRSimulationConfig(
                num_iterations=60, seed=seed, device="cpu"
            )
            result, tape = run_star_instrumented(
                settings, num_stations=n, policy="uniform",
                max_exchanges_per_interval=n - 1,
            )
            series = reciprocity_bias_series(result, tape, settings)
            sigma_b, structure = bias_structure(series)
            sigmas.append(sigma_b)
            d1s.append(math.sqrt(structure.get(1, float("nan"))))
        results.append(
            {
                "n": n,
                "sigma_b_mrad": 1e3 * float(
                    np.sqrt(np.mean(np.square(sigmas)))
                ),
                "d1_mrad": 1e3 * float(
                    np.sqrt(np.mean(np.square(d1s)))
                ),
            }
        )
        print(f"N={n}: {results[-1]}", flush=True)
    store("resampling", results)


def part_doppler() -> None:
    from doppler_coast_study import (
        bias_structure,
        reciprocity_bias_series,
        run_star_instrumented,
    )
    from ota_sync import SDRSimulationConfig

    speeds = (0.0, 1.0, 3.0)
    out = []
    for speed in speeds:
        all_series = []
        for seed in (0, 1):
            settings = SDRSimulationConfig(
                num_iterations=60, seed=seed, device="cpu",
                channel_speed_mps=speed,
            )
            result, tape = run_star_instrumented(
                settings, num_stations=6, policy="uniform",
                max_exchanges_per_interval=5,
            )
            all_series.extend(
                reciprocity_bias_series(result, tape, settings)
            )
        sigma_b, structure = bias_structure(all_series)
        out.append(
            {
                "speed": speed,
                "sigma_b": sigma_b,
                "structure": {
                    str(gap): value for gap, value in structure.items()
                },
            }
        )
        print(f"{speed} m/s: sigma_b {1e3 * sigma_b:.0f} mrad, "
              f"D(1) {1e3 * math.sqrt(structure[1]):.0f} mrad",
              flush=True)
    store("doppler", out)


def part_plot() -> None:
    from doppler_coast_study import jakes_structure

    data = load_data()

    # Coast-law exact fraction vs array size.
    rows = data["exact"]
    figure, axis = plt.subplots(figsize=(5.0, 3.4))
    x = [r["n"] for r in rows]
    y = [100 * r["exact"] / r["total"] for r in rows]
    axis.plot(x, y, "o-")
    axis.set_ylim(95, 100.4)
    axis.set_xticks(x)
    axis.set_xlabel("array size N (stations)")
    axis.set_ylabel("coast gaps predicted exactly (%)")
    axis.set_title("Coast-law exact-prediction fraction vs array size")
    axis.grid(True)
    figure.savefig(
        FIGURES_DIR / "theory_exactness_vs_n.png",
        dpi=200, bbox_inches="tight",
    )
    plt.close(figure)
    print("saved theory_exactness_vs_n.png")

    # Resampling noise vs array size.
    rows = data["resampling"]
    figure, axis = plt.subplots(figsize=(5.0, 3.4))
    x = [r["n"] for r in rows]
    axis.plot(
        x, [r["d1_mrad"] for r in rows], "o-",
        label="per-exchange resampling noise",
    )
    axis.plot(
        x, [r["sigma_b_mrad"] for r in rows], "s-",
        label="per-link bias spread",
    )
    axis.set_ylim(0, 160)
    axis.set_xticks(x)
    axis.set_xlabel("array size N (stations)")
    axis.set_ylabel("noise (mrad)")
    axis.set_title("Multipath resampling noise vs array size")
    axis.legend(loc="lower right", fontsize=8)
    axis.grid(True)
    figure.savefig(
        FIGURES_DIR / "theory_resampling_vs_n.png",
        dpi=200, bbox_inches="tight",
    )
    plt.close(figure)
    print("saved theory_resampling_vs_n.png")

    # Reciprocity-bias structure function vs service gap.
    rows = data["doppler"]
    figure, axis = plt.subplots(figsize=(5.6, 3.8))
    interval_s = 0.05
    for row in rows:
        gaps = sorted(int(g) for g in row["structure"])
        axis.plot(
            gaps,
            [1e3 * math.sqrt(row["structure"][str(g)]) for g in gaps],
            "o-", label=f"measured, {row['speed']:g} m/s",
        )
    fastest = rows[-1]
    f_doppler = fastest["speed"] / (299792458.0 / 915e6)
    gaps = sorted(int(g) for g in fastest["structure"])
    model = [
        1e3 * math.sqrt(
            jakes_structure(fastest["sigma_b"], f_doppler, g * interval_s)
        )
        for g in gaps
    ]
    axis.plot(
        gaps, model, "k--",
        label=f"Jakes model, {fastest['speed']:g} m/s",
    )
    axis.set_ylim(0, None)
    axis.set_xticks(gaps)
    axis.set_xlabel("gap between services (sync intervals)")
    axis.set_ylabel("reciprocity-bias wander (mrad)")
    axis.set_title("Reciprocity-bias structure function vs service gap")
    axis.legend(loc="lower right", fontsize=8)
    axis.grid(True)
    figure.savefig(
        FIGURES_DIR / "theory_doppler_structure.png",
        dpi=200, bbox_inches="tight",
    )
    plt.close(figure)
    print("saved theory_doppler_structure.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--part", choices=("exact", "resampling", "doppler", "plot"),
        required=True,
    )
    args = parser.parse_args()
    {
        "exact": part_exact,
        "resampling": part_resampling,
        "doppler": part_doppler,
        "plot": part_plot,
    }[args.part]()


if __name__ == "__main__":
    main()
