"""Experiments A and C of the phase_sync_idea test plan.

A: scheme x N scaling. N in {2,4,8,16,32,64}, seeds 0-2, five schemes:
     conv-uniform    dedicated two-way, every interval (classical
                     baseline; physically capacity-capped at ~5
                     exchanges/interval, so it starves past N~6 -
                     that collapse IS the measurement)
     conv-scheduled  dedicated two-way, posterior-scheduled (the
                     strongest conventional baseline)
     opportunistic   piggyback OFDM observations, anchors K=40,
                     process-noise inflation on
     hybrid          same machinery, dense anchors K=5
     no-sync         acquire then coast forever: run_scheduled_star
                     policy="scheduled" with a huge budget - the
                     settling logic forces acquisition service, after
                     which no link ever triggers again

C: the anchor-rate frontier. N=8, opportunistic, K in
   {2,5,10,20,40,80,160,320}, residual vs anchor airtime, with the
   dedicated (scheduled) scheme as a reference point.

Metrics: worst/mean station residual (steady window when the run
reaches steady; otherwise the last-quarter tail - stated per cell in
the cache), mean array gain over the same window, sync airtime as
accounted by each runner. Results cached to experiment_a_c_cache.json;
reruns skip finished cells.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from ota_sync import SDRSimulationConfig
from ota_sync.scheduled import run_scheduled_star
from piggyback_largen_study import run_piggyback_variant

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "experiment_a_c_cache.json")
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

N_LIST = [2, 4, 8, 16, 32, 64]
SEEDS = [0, 1, 2]
K_LIST = [2, 5, 10, 20, 40, 80, 160, 320]


def load_cache() -> dict:
    if os.path.exists(CACHE):
        with open(CACHE) as handle:
            return json.load(handle)
    return {}


def save_cache(cache: dict) -> None:
    with open(CACHE, "w") as handle:
        json.dump(cache, handle, indent=1)


def scheduled_metrics(result) -> dict:
    """Steady-window metrics with an honest tail fallback for runs
    that never reach all-stations steady (starved links)."""

    steady = bool(torch.any(result.steady))
    if steady:
        rms = [v for v in result.station_steady_rms]
        gain = result.mean_array_gain
        window = "steady"
    else:
        intervals = result.residuals.shape[1]
        tail = slice(max(0, intervals - max(1, intervals // 4)), intervals)
        rms = [
            torch.sqrt(torch.mean(row[tail].square())).item()
            for row in result.residuals
        ]
        gain = torch.mean(result.array_gain[tail]).item()
        window = "tail-quarter"
    return {
        "worst_mrad": 1e3 * max(rms),
        "mean_mrad": 1e3 * sum(rms) / len(rms),
        "gain": gain,
        "airtime": result.airtime_used_fraction,
        "demand": result.airtime_uniform_fraction,
        "window": window,
    }


def piggyback_metrics(res) -> dict:
    star = res.star
    return {
        "worst_mrad": star.worst_rms_mrad,
        "mean_mrad": (
            sum(star.station_rms_mrad) / len(star.station_rms_mrad)
        ),
        "gain": star.mean_array_gain,
        "airtime": star.piggyback_airtime,
        "demand": star.piggyback_airtime,
        "window": "valid",
    }


def run_cell(scheme: str, n: int, seed: int, k: int | None = None) -> dict:
    if scheme in ("conv-uniform", "conv-scheduled", "no-sync"):
        intervals = 60
        if scheme == "no-sync":
            # acquisition takes ~8 services/link at <=5 slots/interval
            intervals = max(60, 12 * math.ceil((n - 1) / 5) + 48)
        settings = SDRSimulationConfig(
            num_iterations=intervals, seed=seed, device="cpu"
        )
        if scheme == "conv-uniform":
            result = run_scheduled_star(
                settings, num_stations=n, policy="uniform"
            )
        elif scheme == "conv-scheduled":
            result = run_scheduled_star(
                settings, num_stations=n, policy="scheduled"
            )
        else:
            result = run_scheduled_star(
                settings,
                num_stations=n,
                policy="scheduled",
                budgets_rad=[1e6] * (n - 1),
            )
        return scheduled_metrics(result)

    cadence = k if k is not None else (40 if scheme == "opportunistic" else 5)
    intervals = max(60, 4 * cadence + n)
    settings = SDRSimulationConfig(
        num_iterations=intervals, seed=seed, device="cpu"
    )
    res = run_piggyback_variant(
        settings,
        num_stations=n,
        anchor_every_intervals=cadence,
        inflate_process=True,
    )
    return piggyback_metrics(res)


def part_a(schemes: list[str]) -> None:
    cache = load_cache()
    for scheme in schemes:
        for n in N_LIST:
            for seed in SEEDS:
                key = f"A|{scheme}|{n}|{seed}"
                if key in cache:
                    continue
                cache[key] = run_cell(scheme, n, seed)
                save_cache(cache)
                print(key, {k: round(v, 3) if isinstance(v, float) else v
                            for k, v in cache[key].items()}, flush=True)


def part_c() -> None:
    cache = load_cache()
    for k in K_LIST:
        for seed in SEEDS:
            key = f"C|opportunistic|8|{seed}|K{k}"
            if key in cache:
                continue
            cache[key] = run_cell("opportunistic", 8, seed, k=k)
            save_cache(cache)
            print(key, {kk: round(v, 3) if isinstance(v, float) else v
                        for kk, v in cache[key].items()}, flush=True)
    for seed in SEEDS:
        key = f"C|conv-scheduled|8|{seed}"
        if key in cache:
            continue
        cache[key] = run_cell("conv-scheduled", 8, seed)
        save_cache(cache)
        print(key, cache[key], flush=True)


def agg(cache: dict, prefix: str, field: str) -> tuple[float, float]:
    values = [cell[field] for key, cell in cache.items()
              if key.startswith(prefix)]
    mean = sum(values) / len(values)
    std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
    return mean, std


def figures() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cache = load_cache()
    os.makedirs(FIGDIR, exist_ok=True)
    schemes = ["conv-uniform", "conv-scheduled", "opportunistic",
               "hybrid", "no-sync"]

    fig, axis = plt.subplots(figsize=(7, 5))
    for scheme in schemes:
        xs, ys, es = [], [], []
        for n in N_LIST:
            prefix = f"A|{scheme}|{n}|"
            if not any(key.startswith(prefix) for key in cache):
                continue
            mean, std = agg(cache, prefix, "worst_mrad")
            xs.append(n)
            ys.append(mean)
            es.append(std)
        axis.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=scheme)
    axis.set_yscale("log")
    axis.set_xscale("log", base=2)
    axis.set_xlabel("number of stations N")
    axis.set_ylabel("worst-station phase residual (mrad)")
    axis.set_title("Residual phase error vs array size, by scheme")
    axis.legend()
    fig.savefig(os.path.join(FIGDIR, "figA1_residual_vs_N.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5))
    for scheme in schemes:
        xs, ys = [], []
        for n in N_LIST:
            prefix = f"A|{scheme}|{n}|"
            if not any(key.startswith(prefix) for key in cache):
                continue
            mean, _ = agg(cache, prefix, "airtime")
            xs.append(n)
            ys.append(100.0 * mean)
        axis.plot(xs, ys, marker="o", label=scheme + " (used)")
    xs, ys = [], []
    for n in N_LIST:
        prefix = f"A|conv-uniform|{n}|"
        if any(key.startswith(prefix) for key in cache):
            mean, _ = agg(cache, prefix, "demand")
            xs.append(n)
            ys.append(100.0 * mean)
    axis.plot(xs, ys, linestyle="--", marker="s",
              label="conv-uniform (demand)")
    axis.axhline(100.0, linestyle=":", color="gray", label="frame limit")
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("number of stations N")
    axis.set_ylabel("synchronization airtime (% of frame)")
    axis.set_title("Synchronization airtime vs array size, by scheme")
    axis.legend(fontsize=8)
    fig.savefig(os.path.join(FIGDIR, "figA2_airtime_vs_N.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5))
    xs, ys, es = [], [], []
    for k in K_LIST:
        prefix = f"C|opportunistic|8|"
        keys = [key for key in cache
                if key.startswith(prefix) and key.endswith(f"K{k}")]
        if not keys:
            continue
        air = sum(cache[key]["airtime"] for key in keys) / len(keys)
        worst = [cache[key]["worst_mrad"] for key in keys]
        mean = sum(worst) / len(worst)
        std = (sum((v - mean) ** 2 for v in worst) / len(worst)) ** 0.5
        xs.append(100.0 * air)
        ys.append(mean)
        es.append(std)
    axis.errorbar(xs, ys, yerr=es, marker="o", capsize=3,
                  label="opportunistic (anchor cadence swept)")
    ref_keys = [key for key in cache if key.startswith("C|conv-scheduled|8|")]
    if ref_keys:
        air = sum(cache[key]["airtime"] for key in ref_keys) / len(ref_keys)
        worst = sum(cache[key]["worst_mrad"] for key in ref_keys) / len(ref_keys)
        axis.plot([100.0 * air], [worst], marker="*", markersize=14,
                  linestyle="none", label="dedicated two-way (scheduled)")
    axis.set_xscale("log")
    axis.set_xlabel("synchronization airtime (% of frame)")
    axis.set_ylabel("worst-station phase residual (mrad)")
    axis.set_title("Residual phase error vs synchronization airtime, N=8")
    axis.legend()
    fig.savefig(os.path.join(FIGDIR, "figC1_anchor_frontier.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("figures written", flush=True)


def report() -> None:
    cache = load_cache()
    schemes = ["conv-uniform", "conv-scheduled", "opportunistic",
               "hybrid", "no-sync"]
    print("EXPERIMENT A (worst mrad / gain % / airtime %):")
    header = "N".rjust(4) + "".join(s.rjust(24) for s in schemes)
    print(header)
    for n in N_LIST:
        row = f"{n:>4}"
        for scheme in schemes:
            prefix = f"A|{scheme}|{n}|"
            if not any(key.startswith(prefix) for key in cache):
                row += " " * 24
                continue
            worst, wstd = agg(cache, prefix, "worst_mrad")
            gain, _ = agg(cache, prefix, "gain")
            air, _ = agg(cache, prefix, "airtime")
            row += f" {worst:7.0f}±{wstd:<4.0f}/{100*gain:4.1f}/{100*air:5.1f}"
        print(row)
    print("\nEXPERIMENT C (N=8, per K: worst mrad @ airtime %):")
    for k in K_LIST:
        keys = [key for key in cache
                if key.startswith("C|opportunistic|8|")
                and key.endswith(f"K{k}")]
        if not keys:
            continue
        worst = [cache[key]["worst_mrad"] for key in keys]
        mean = sum(worst) / len(worst)
        std = (sum((v - mean) ** 2 for v in worst) / len(worst)) ** 0.5
        air = sum(cache[key]["airtime"] for key in keys) / len(keys)
        print(f"  K={k:>3}  {mean:6.1f}±{std:<5.1f} mrad @ {100*air:6.3f}%")
    ref = [key for key in cache if key.startswith("C|conv-scheduled|8|")]
    if ref:
        worst = sum(cache[key]["worst_mrad"] for key in ref) / len(ref)
        air = sum(cache[key]["airtime"] for key in ref) / len(ref)
        print(f"  dedicated ref: {worst:6.1f} mrad @ {100*air:6.3f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", required=True,
                        choices=["a-conv", "a-opp", "a-hyb", "a-nosync",
                                 "c", "figs", "report"])
    args = parser.parse_args()
    if args.part == "a-conv":
        part_a(["conv-uniform", "conv-scheduled"])
    elif args.part == "a-opp":
        part_a(["opportunistic"])
    elif args.part == "a-hyb":
        part_a(["hybrid"])
    elif args.part == "a-nosync":
        part_a(["no-sync"])
    elif args.part == "c":
        part_c()
    elif args.part == "figs":
        figures()
    else:
        report()


if __name__ == "__main__":
    main()
