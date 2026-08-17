"""Figures for piggyback array-size scaling (fresh runs, plain
default matplotlib).

Produces, in figures/studies/:
  piggyback_airtime_wall_vs_n.png
  piggyback_error_artifact_vs_n.png
  piggyback_aliasing_rootcause.png

Seeds 0-2 at N in {6, 10, 14, 20}; seed 0 at N=30. Cells cache to
fig_cache_clutter_scaling.json so an interrupted run resumes; delete
the cache for a fully fresh pass. Pass --status for progress.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from ota_sync import SDRSimulationConfig
from ota_sync.scheduled import run_scheduled_star
from piggyback_largen_study import run_piggyback_variant

CACHE = Path(__file__).resolve().parent / "fig_cache_clutter_scaling.json"
FIGDIR = Path(__file__).resolve().parent / "figures" / "studies"
ITERATIONS = 60
K = 40
SWEEP = [
    (6, [0, 1, 2]),
    (10, [0, 1, 2]),
    (14, [0, 1, 2]),
    (20, [0, 1, 2]),
]
BONUS_N30 = (30, [0])


def save(figure, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / f"{name}.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(path)


def _load() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return {}


def _store(cache: dict) -> None:
    CACHE.write_text(json.dumps(cache, indent=1))


def _cell(cache: dict, key: str, compute) -> dict:
    if key not in cache:
        cache[key] = compute()
        _store(cache)
        print(f"  computed {key}")
    return cache[key]


def piggyback_cell(n: int, seed: int, variant: str) -> dict:
    settings = SDRSimulationConfig(
        num_iterations=ITERATIONS, seed=seed, device="cpu",
        **({"slave_initial_frequency_hz": 0.0} if variant == "zerocfo"
           else {}),
    )
    kwargs = dict(num_stations=n, anchor_every_intervals=K)
    if variant == "mitigated":
        kwargs["inflate_process"] = True
    result = run_piggyback_variant(settings, **kwargs)
    return {
        "worst": result.star.worst_rms_mrad,
        "stations": result.star.station_rms_mrad,
        "gain": result.star.mean_array_gain,
        "airtime": result.star.piggyback_airtime,
    }


def twoway_cell(n: int, seed: int) -> dict:
    settings = SDRSimulationConfig(
        num_iterations=ITERATIONS, seed=seed, device="cpu"
    )
    star = run_scheduled_star(settings, num_stations=n, policy="scheduled")
    worst = max(
        (v for v in star.station_steady_rms if v == v),
        default=float("nan"),
    )
    return {
        "worst": 1e3 * worst,
        "airtime": star.airtime_used_fraction,
    }


def collect(cache: dict) -> dict:
    for n, seeds in SWEEP + [BONUS_N30]:
        for seed in seeds:
            for variant in ("current", "mitigated", "zerocfo"):
                if n == 30 and variant != "mitigated":
                    continue  # bonus point: fixed variant only
                _cell(
                    cache, f"pig_{variant}_N{n}_s{seed}",
                    lambda n=n, seed=seed, variant=variant: piggyback_cell(
                        n, seed, variant
                    ),
                )
            if n <= 20:
                _cell(
                    cache, f"two_N{n}_s{seed}",
                    lambda n=n, seed=seed: twoway_cell(n, seed),
                )
    return cache


def _mean(cache: dict, prefix: str, seeds: list[int], field: str):
    values = [cache[f"{prefix}_s{s}"][field] for s in seeds]
    tensor = torch.tensor(values, dtype=torch.float64)
    return tensor.mean().item(), (
        tensor.std().item() if len(values) > 1 else 0.0
    )


def fig_airtime_wall(cache: dict) -> None:
    ns = [n for n, _ in SWEEP]
    pig_air = [
        100.0 * _mean(cache, f"pig_current_N{n}", seeds, "airtime")[0]
        for n, seeds in SWEEP
    ]
    two_air = [
        100.0 * _mean(cache, f"two_N{n}", seeds, "airtime")[0]
        for n, seeds in SWEEP
    ]
    n30_air = 100.0 * cache["pig_mitigated_N30_s0"]["airtime"]

    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    axis.plot(
        ns, two_air, marker="o", color="C1", label="two-way baseline"
    )
    axis.plot(
        ns + [30], pig_air + [n30_air], marker="o", color="C0",
        label=f"piggyback (anchors every {K})",
    )
    axis.axhline(
        100.0, color="gray", linestyle=":", label="100% of frame"
    )
    axis.set_xlabel("number of stations N")
    axis.set_ylabel("sync airtime (% of frame)")
    axis.set_xticks(ns + [30])
    axis.set_ylim(0, 110)
    axis.set_title(
        f"Sync airtime vs number of stations "
        f"(K={K}, {ITERATIONS} intervals, seeds 0-2)"
    )
    axis.legend()
    save(figure, "piggyback_airtime_wall_vs_n")


def fig_error_artifact(cache: dict) -> None:
    ns = [n for n, _ in SWEEP]
    series = [
        ("current", "o", "piggyback, original configuration"),
        ("mitigated", "s", "piggyback, filter mitigation"),
        ("zerocfo", "^", "piggyback, zero-offset control"),
    ]
    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    two_worst = [
        _mean(cache, f"two_N{n}", seeds, "worst")[0]
        for n, seeds in SWEEP
    ]
    axis.plot(
        ns, two_worst, marker="d", color="C3",
        label="two-way baseline (worst station)",
    )
    for variant, marker, label in series:
        means, stds = [], []
        for n, seeds in SWEEP:
            m, s = _mean(cache, f"pig_{variant}_N{n}", seeds, "worst")
            means.append(m)
            stds.append(s)
        axis.errorbar(
            ns, means, yerr=stds, marker=marker, capsize=3, label=label
        )
    axis.set_xlabel("number of stations N")
    axis.set_ylabel("worst-station clock error (mrad RMS)")
    axis.set_xticks(ns)
    axis.set_ylim(bottom=0)
    axis.set_title(
        f"Worst-station clock error vs number of stations "
        f"(K={K}, seeds 0-2)"
    )
    axis.legend()
    save(figure, "piggyback_error_artifact_vs_n")


def fig_aliasing(cache: dict) -> None:
    points_x, points_y = [], []
    for n in (10, 14):
        for seed in (0, 1, 2):
            stations = cache[f"pig_current_N{n}_s{seed}"]["stations"]
            for index, rms in enumerate(stations):
                station = index + 1
                cfo = 1500.0 * station / (n - 1)
                distance = min(cfo % 100.0, 100.0 - (cfo % 100.0))
                points_x.append(distance)
                points_y.append(rms)
    figure, axis = plt.subplots(figsize=(6.2, 4.4))
    axis.scatter(points_x, points_y, s=28, alpha=0.75)
    axis.set_xlabel(
        "station frequency offset, distance from 100 Hz grid (Hz)"
    )
    axis.set_ylabel("per-station clock error (mrad RMS)")
    axis.set_ylim(bottom=0)
    axis.set_title(
        "Per-station clock error vs frequency-offset distance from "
        "the\nobservation grid (N=10 and 14, seeds 0-2)"
    )
    save(figure, "piggyback_aliasing_rootcause")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    cache = _load()
    if args.status:
        print(f"{len(cache)} cells cached")
        return
    collect(cache)
    fig_airtime_wall(cache)
    fig_error_artifact(cache)
    fig_aliasing(cache)


if __name__ == "__main__":
    main()
