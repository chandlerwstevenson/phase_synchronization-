"""Array-size scaling of the five-metric scheduling-policy comparison.

The multi-metric study scored the four scheduling policies (uniform,
roundrobin, scheduled, oracle) only at a 10-station array. This study
sweeps N in {6, 10, 14, 20} at two capacity levels per N:

  contended     about 22% of the array's links may sync per interval
  comfortable   about 45% of the links

using max(1, round(fraction * (N - 1))) exchanges per interval, i.e.
capacities 1/2 (N=6), 2/4 (N=10), 3/6 (N=14), 4/9 (N=20). The N=10
rows regression-anchor against the published multi_metric_study table
(same scoring code, same trials).

Metrics per cell (definitions in multi_metric_study.py): probability
of detection, mean and 95%-likely spectral efficiency at the coverage
edge, beam quality, sync airtime, net throughput, detection range.

The study is chunked so each command finishes in the foreground:

    .venv/bin/python scheduling_scaling_study.py --stations 6 --level contended
    ...                                          (one command per cell group)
    .venv/bin/python scheduling_scaling_study.py --demand
    .venv/bin/python scheduling_scaling_study.py --report

Results accumulate in scheduling_scaling_cache.json; --report prints
the tables and the four scaling answers from the cache.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from detection import DetectionParams
from detection.viability import detection_range_m
from multi_metric_study import (
    best_per_metric,
    print_table,
    score,
    star_residual_matrix,
)
from ota_sync import SDRSimulationConfig
from ota_sync.scheduled import run_scheduled_star

CACHE = "scheduling_scaling_cache.json"
POLICIES = ("uniform", "roundrobin", "scheduled", "oracle")
LEVELS = {"contended": 0.22, "comfortable": 0.45}


def capacity_for(num_stations: int, level: str) -> int:
    return max(1, round(LEVELS[level] * (num_stations - 1)))


def load_cache() -> dict:
    if os.path.exists(CACHE):
        with open(CACHE) as handle:
            return json.load(handle)
    return {"cells": [], "demand": []}


def save_cache(cache: dict) -> None:
    with open(CACHE, "w") as handle:
        json.dump(cache, handle, indent=1)


def run_cells(
    num_stations: int,
    level: str,
    policies: list[str],
    seeds: list[int],
    iterations: int,
    trials: int,
    h0_trials: int,
) -> None:
    capacity = capacity_for(num_stations, level)
    params = DetectionParams(tx_power_w=0.5)
    cache = load_cache()
    for policy in policies:
        key = (num_stations, level, policy)
        if any(
            (c["n"], c["level"], c["policy"]) == key for c in cache["cells"]
        ):
            print(f"  cached: N={num_stations} {level} {policy}")
            continue
        per_seed = []
        airtimes = []
        for seed in seeds:
            settings = SDRSimulationConfig(
                num_iterations=iterations, seed=seed, device="cpu"
            )
            result = run_scheduled_star(
                settings,
                num_stations=num_stations,
                policy=policy,
                max_exchanges_per_interval=capacity,
            )
            per_seed.append((result.positions, star_residual_matrix(result)))
            airtimes.append(result.airtime_used_fraction)
        row = score(
            f"{policy} N={num_stations} @cap{capacity}",
            per_seed,
            float(np.mean(airtimes)),
            params,
            trials,
            h0_trials,
        )
        cache["cells"].append(
            {
                "n": num_stations,
                "level": level,
                "policy": policy,
                "capacity": capacity,
                "seeds": seeds,
                **{
                    k: row[k]
                    for k in (
                        "label", "pd", "mean_se", "likely_se", "gain",
                        "airtime", "net", "range_m", "overfit",
                    )
                },
            }
        )
        save_cache(cache)
        print(f"  scored N={num_stations} {level} {policy}")


def run_demand(seeds: list[int], iterations: int) -> None:
    """Sync demand per policy if the channel never refuses service:
    informed policies measured at capacity N-1; uniform/roundrobin
    demand is the full (N-1)-exchange rate by definition."""

    cache = load_cache()
    done = {(d["n"], d["policy"]) for d in cache["demand"]}
    for num_stations in (6, 10, 14, 20):
        for policy in ("scheduled", "oracle"):
            if (num_stations, policy) in done:
                continue
            airtimes = []
            uniform_demand = None
            for seed in seeds:
                settings = SDRSimulationConfig(
                    num_iterations=iterations, seed=seed, device="cpu"
                )
                result = run_scheduled_star(
                    settings,
                    num_stations=num_stations,
                    policy=policy,
                    max_exchanges_per_interval=num_stations - 1,
                )
                airtimes.append(result.airtime_used_fraction)
                uniform_demand = result.airtime_uniform_fraction
            cache["demand"].append(
                {
                    "n": num_stations,
                    "policy": policy,
                    "airtime": float(np.mean(airtimes)),
                    "uniform_demand": uniform_demand,
                }
            )
            save_cache(cache)
            print(
                f"  demand N={num_stations} {policy}: "
                f"{100 * float(np.mean(airtimes)):.1f}% "
                f"(uniform would demand {100 * uniform_demand:.1f}%)"
            )


def report() -> None:
    cache = load_cache()
    cells = cache["cells"]
    if not cells:
        print("no cells cached yet")
        return

    for level in LEVELS:
        rows = [
            c for c in cells if c["level"] == level
        ]
        rows.sort(key=lambda c: (c["n"], POLICIES.index(c["policy"])))
        if rows:
            print_table(
                rows,
                f"{level} capacity (~{int(100 * LEVELS[level])}% of links)",
            )
            best_per_metric(rows)

    # (1) informed-scheduling advantage vs N (contended level)
    print("\n=== scaling answer 1: net-throughput advantage over uniform "
          "(contended capacity) ===")
    print(f"  {'N':>3} {'uniform':>8} {'scheduled':>10} {'oracle':>8} "
          f"{'sched adv':>10} {'oracle adv':>11}")
    for n in sorted({c["n"] for c in cells}):
        by = {
            c["policy"]: c for c in cells
            if c["n"] == n and c["level"] == "contended"
        }
        if {"uniform", "scheduled", "oracle"} <= set(by):
            u, s, o = by["uniform"]["net"], by["scheduled"]["net"], by["oracle"]["net"]
            print(
                f"  {n:>3} {u:>8.2f} {s:>10.2f} {o:>8.2f} "
                f"{s - u:>+10.2f} {o - u:>+11.2f}"
            )

    # (2) airtime wall per policy
    demand = cache["demand"]
    if demand:
        print("\n=== scaling answer 2: sync demand if never refused "
              "(the airtime wall) ===")
        print(f"  {'N':>3} {'uniform/roundrobin':>19} {'scheduled':>10} "
              f"{'oracle':>8}")
        for n in sorted({d["n"] for d in demand}):
            by = {d["policy"]: d for d in demand if d["n"] == n}
            uniform = next(iter(by.values()))["uniform_demand"]
            sched = by.get("scheduled", {}).get("airtime", float("nan"))
            oracle = by.get("oracle", {}).get("airtime", float("nan"))
            print(
                f"  {n:>3} {100 * uniform:>18.1f}% {100 * sched:>9.1f}% "
                f"{100 * oracle:>7.1f}%"
            )
        # linear wall estimates
        ns = sorted({d["n"] for d in demand})
        uniform_pts = [
            (n, next(d for d in demand if d["n"] == n)["uniform_demand"])
            for n in ns
        ]
        slope = (uniform_pts[-1][1] - uniform_pts[0][1]) / (
            uniform_pts[-1][0] - uniform_pts[0][0]
        )
        wall_uniform = uniform_pts[0][0] + (1.0 - uniform_pts[0][1]) / slope
        print(f"  uniform/roundrobin demand crosses 100% at N ~= "
              f"{wall_uniform:.1f}")
        for policy in ("scheduled", "oracle"):
            pts = [
                (d["n"], d["airtime"]) for d in demand
                if d["policy"] == policy
            ]
            if len(pts) >= 2:
                slope = (pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0])
                if slope > 0:
                    wall = pts[0][0] + (1.0 - pts[0][1]) / slope
                    print(f"  {policy} demand crosses 100% at N ~= {wall:.1f}")

    # (3) does the quality-metric pathology worsen with N
    print("\n=== scaling answer 3: quality metrics vs realizability ===")
    for n in sorted({c["n"] for c in cells}):
        rows = [c for c in cells if c["n"] == n]
        best_beam = max(rows, key=lambda c: c["gain"])
        best_net = max(rows, key=lambda c: c["net"])
        print(
            f"  N={n}: best beam quality = {best_beam['label']} "
            f"(airtime {100 * best_beam['airtime']:.0f}%"
            f"{', NOT realizable' if best_beam['overfit'] else ''}) | "
            f"best net throughput = {best_net['label']} "
            f"(airtime {100 * best_net['airtime']:.0f}%)"
        )

    # (4) detection range vs the perfect-sync reference
    print("\n=== scaling answer 4: detection range vs perfect-sync "
          "reference (contended capacity) ===")
    params = DetectionParams(tx_power_w=0.5)
    print(f"  {'N':>3} {'perfect':>8} " + "".join(
        f"{p:>11}" for p in POLICIES
    ))
    for n in sorted({c["n"] for c in cells}):
        by = {
            c["policy"]: c for c in cells
            if c["n"] == n and c["level"] == "contended"
        }
        perfect = detection_range_m(n, 1.0, params)
        cols = "".join(
            f"{by[p]['range_m']:>10.0f}m" if p in by else f"{'-':>11}"
            for p in POLICIES
        )
        print(f"  {n:>3} {perfect:>7.0f}m {cols}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="N-scaling of the scheduling-policy multi-metric table"
    )
    parser.add_argument("--stations", type=int)
    parser.add_argument("--level", choices=tuple(LEVELS))
    parser.add_argument("--policies", type=str, default=",".join(POLICIES))
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--h0-trials", type=int, default=12000)
    parser.add_argument("--demand", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    if args.report:
        report()
        return
    if args.demand:
        run_demand([int(s) for s in args.seeds.split(",")], args.iterations)
        return
    if args.stations is None or args.level is None:
        parser.error("need --stations and --level (or --demand/--report)")
    run_cells(
        args.stations,
        args.level,
        [p.strip() for p in args.policies.split(",")],
        [int(s) for s in args.seeds.split(",")],
        args.iterations,
        args.trials,
        args.h0_trials,
    )


if __name__ == "__main__":
    main()
