"""Array-size scaling of the membership family.

Every membership result so far (posterior gate, 1-bit alignment
feedback, hybrid two-tier combiner) was measured at N=10 stations.
This study repeats the five-metric scoring at N = 6, 10, 14, 20 and
answers the scaling questions directly.

Design choices, stated up front:
- Contention is held comparable across N by scaling the sync channel
  capacity with the number of links: capacity = max(1, round(0.22 x
  (N-1))) -> 1/2/3/4 exchanges per interval at N = 6/10/14/20 (the
  N=10 point reproduces the original capacity-2 regime).
- Detection powers: 0.5 W per station (the operational/near-saturated
  point) and an "N-matched" power 0.05 x (10/N)^3 W. Detection
  signal-to-noise ratio scales as N^3 x P for a perfectly synchronized
  array (transmit focusing N^2 x receive combining N), so the matched
  power holds the perfect-sync detection budget constant across N -
  at N=10 it is exactly the 0.05 W that separated the methods before.
- Cheap metrics (beam quality, throughput, range, net throughput) use
  seeds 0-2. Counted detection uses seeds 0-1 at N = 6, 10 and seed 0
  at N = 14, 20 to bound runtime; stated in the output.
- Results cache to membership_scaling_cache.json per cell, so a
  timed-out invocation resumes instead of recomputing.

Usage (one N per invocation is fine; the cache accumulates):
    .venv/bin/python membership_scaling_study.py --stations 6
    .venv/bin/python membership_scaling_study.py --stations 10
    ...
    .venv/bin/python membership_scaling_study.py --report   # tables + answers
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch

from detection import DetectionParams
from gating_study import evaluation_mask, phase_matrix, run_star_with_posteriors
from metrics import (
    detection_range_m,
    mean_array_gain,
    net_throughput,
    probability_of_detection,
    spectral_efficiency,
)
from metrics_membership_study import METHODS, method_weights
from ota_sync import SDRSimulationConfig

CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "membership_scaling_cache.json",
)
DEFAULT_STATIONS = (6, 10, 14, 20)
CHEAP_SEEDS = (0, 1, 2)
COMM_POWER_W = 0.5


def capacity_for(num_stations: int) -> int:
    return max(1, round(0.22 * (num_stations - 1)))


def matched_power_w(num_stations: int) -> float:
    """Hold the perfect-sync detection budget (proportional to N^3 x P)
    at its N=10 / 0.05 W value."""

    return 0.05 * (10.0 / num_stations) ** 3


def detect_seeds_for(num_stations: int) -> tuple[int, ...]:
    return (0, 1) if num_stations <= 10 else (0,)


def _load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as handle:
            return json.load(handle)
    return {}


def _save_cache(cache: dict) -> None:
    with open(CACHE_PATH, "w") as handle:
        json.dump(cache, handle, indent=0, sort_keys=True)


def score_stations(
    num_stations: int,
    iterations: int = 60,
    trials: int = 250,
    h0_trials: int = 8000,
    cache: dict | None = None,
) -> dict:
    """Score every method at one array size, filling the cache."""

    if cache is None:
        cache = _load_cache()
    capacity = capacity_for(num_stations)
    powers = {"0.5W": 0.5, "matched": matched_power_w(num_stations)}
    d_seeds = detect_seeds_for(num_stations)

    for seed in CHEAP_SEEDS:
        prefix = f"{num_stations}/{seed}"
        want_detect = seed in d_seeds
        have_cheap = all(
            f"{prefix}/{m}/{k}" in cache
            for m in METHODS
            for k in ("gain", "se_edge", "se_edge95", "se_near", "net", "range")
        ) and f"{prefix}/airtime" in cache
        have_detect = (not want_detect) or all(
            f"{prefix}/{m}/pd@{tag}" in cache
            for m in METHODS
            for tag in powers
        )
        if have_cheap and have_detect:
            continue

        settings = SDRSimulationConfig(
            num_iterations=iterations, seed=seed, device="cpu"
        )
        result, sigma_full = run_star_with_posteriors(
            settings,
            num_stations=num_stations,
            policy="uniform",
            max_exchanges_per_interval=capacity,
        )
        cache[f"{prefix}/airtime"] = result.airtime_used_fraction
        mask = evaluation_mask(result)
        phases = phase_matrix(result)[:, mask]
        sigma = sigma_full[:, mask]
        positions = result.positions
        centroid = positions.mean(axis=0)
        edge_targets = np.array(
            [centroid + [1200.0, 150.0], centroid + [-1200.0, 150.0]]
        )
        near_user = centroid + np.array([400.0, 0.0])
        range_params = DetectionParams(tx_power_w=COMM_POWER_W)

        dedupe: dict[bytes, list[float]] = {}
        for name in METHODS:
            weights, tier = method_weights(name, phases, sigma, seed)
            gain_key = f"{prefix}/{name}/gain"
            if gain_key not in cache:
                gain = mean_array_gain(phases, weights)
                cache[gain_key] = gain
                cache[f"{prefix}/{name}/range"] = detection_range_m(
                    num_stations, gain, range_params
                )
                se_edge = spectral_efficiency(
                    phases, weights, positions, edge_targets[0],
                    COMM_POWER_W, noncoherent_weights=tier,
                )
                se_near = spectral_efficiency(
                    phases, weights, positions, near_user,
                    COMM_POWER_W, noncoherent_weights=tier,
                )
                cache[f"{prefix}/{name}/se_edge"] = se_edge.mean_bps_hz
                cache[f"{prefix}/{name}/se_edge95"] = se_edge.likely95_bps_hz
                cache[f"{prefix}/{name}/se_near"] = se_near.mean_bps_hz
                cache[f"{prefix}/{name}/net"] = net_throughput(
                    se_near.mean_bps_hz, result.airtime_used_fraction
                )
            if not want_detect:
                continue
            combiner = (
                "two-tier-noncoherent" if tier is not None
                else "two-tier-discard"
            )
            for tag, power in powers.items():
                pd_key = f"{prefix}/{name}/pd@{tag}"
                if pd_key in cache:
                    continue
                dedupe_key = (
                    f"{combiner}@{tag}".encode()
                    + weights.to(torch.uint8).numpy().tobytes()
                )
                if dedupe_key not in dedupe:
                    detect = probability_of_detection(
                        f"N{num_stations}/{name}@s{seed}/{tag}",
                        positions, phases, weights, edge_targets,
                        combiner=combiner,
                        params=DetectionParams(tx_power_w=power),
                        trials=trials, h0_trials=h0_trials, seed=seed,
                    )
                    dedupe[dedupe_key] = list(detect.pd_measured)
                cache[pd_key] = dedupe[dedupe_key]
                _save_cache(cache)
        _save_cache(cache)
        print(f"  scored N={num_stations} seed {seed}", flush=True)
    return cache


def _mean(cache: dict, num_stations: int, name: str, key: str,
          seeds: tuple[int, ...]) -> float:
    values = [cache[f"{num_stations}/{s}/{name}/{key}"] for s in seeds]
    return float(np.mean(values))


def _mean_pd(cache: dict, num_stations: int, name: str, tag: str) -> float:
    seeds = detect_seeds_for(num_stations)
    rows = [cache[f"{num_stations}/{s}/{name}/pd@{tag}"] for s in seeds]
    return float(np.mean(rows))


def report(cache: dict, station_list: list[int]) -> None:
    metric_keys = [
        ("beam quality", lambda n, m: 100.0 * _mean(cache, n, m, "gain", CHEAP_SEEDS)),
        ("detect @0.5 W", lambda n, m: 100.0 * _mean_pd(cache, n, m, "0.5W")),
        ("detect @matched", lambda n, m: 100.0 * _mean_pd(cache, n, m, "matched")),
        ("thru edge mean", lambda n, m: _mean(cache, n, m, "se_edge", CHEAP_SEEDS)),
        ("thru edge 95%", lambda n, m: _mean(cache, n, m, "se_edge95", CHEAP_SEEDS)),
        ("range (m)", lambda n, m: _mean(cache, n, m, "range", CHEAP_SEEDS)),
        ("net thru", lambda n, m: _mean(cache, n, m, "net", CHEAP_SEEDS)),
    ]

    for n in station_list:
        airtime = float(np.mean(
            [cache[f"{n}/{s}/airtime"] for s in CHEAP_SEEDS]
        ))
        print(
            f"\n=== N={n} stations, capacity {capacity_for(n)}/{n - 1}, "
            f"matched power {1e3 * matched_power_w(n):.1f} mW, sync "
            f"airtime {100 * airtime:.1f}%, detection seeds "
            f"{list(detect_seeds_for(n))} ==="
        )
        header = f"{'method':<14}" + "".join(
            f"{label:>16}" for label, _ in metric_keys
        )
        print(header)
        for name in METHODS:
            cells = ""
            for label, fn in metric_keys:
                value = fn(n, name)
                if "detect" in label or "beam" in label:
                    cells += f"{value:>15.1f}%"
                elif "range" in label:
                    cells += f"{value:>16.0f}"
                else:
                    cells += f"{value:>16.2f}"
            print(f"{name:<14}" + cells)

    # ---- the four scaling answers ---------------------------------
    print("\n=== scaling answers ===")

    print("(1) 1-bit rank per metric per N (1 = best):")
    for n in station_list:
        ranks = []
        for label, fn in metric_keys:
            values = {m: fn(n, m) for m in METHODS}
            ordered = sorted(values, key=lambda m: -values[m])
            ranks.append(f"{label.split()[0]}={ordered.index('1-bit') + 1}")
        print(f"  N={n:>2}: " + "  ".join(ranks))

    print("(2) detection gap vs all-in at matched power (percentage points):")
    for n in station_list:
        base = _mean_pd(cache, n, "all-in", "matched")
        gate = _mean_pd(cache, n, "post-gate", "matched")
        onebit = _mean_pd(cache, n, "1-bit", "matched")
        hybrid = _mean_pd(cache, n, "hybrid", "matched")
        print(
            f"  N={n:>2}: gate {100 * (gate - base):+5.1f}  "
            f"1-bit {100 * (onebit - base):+5.1f}  "
            f"hybrid {100 * (hybrid - base):+5.1f}"
        )

    print("(3) all-in mean-vs-guaranteed throughput inversion "
          "(all-in minus posterior gate, edge user):")
    for n in station_list:
        d_mean = (
            _mean(cache, n, "all-in", "se_edge", CHEAP_SEEDS)
            - _mean(cache, n, "post-gate", "se_edge", CHEAP_SEEDS)
        )
        d_95 = (
            _mean(cache, n, "all-in", "se_edge95", CHEAP_SEEDS)
            - _mean(cache, n, "post-gate", "se_edge95", CHEAP_SEEDS)
        )
        verdict = "inverted" if (d_mean > 0 > d_95) else "not inverted"
        print(
            f"  N={n:>2}: mean {d_mean:+.2f}, 95%-likely {d_95:+.2f}  "
            f"-> {verdict}"
        )

    print("(4) absolute scaling per method (range m | edge throughput "
          "bits/s/Hz), with the perfect-array reference:")
    reference_n = station_list[0]
    for name in METHODS + ("perfect-reference",):
        cells = []
        for n in station_list:
            if name == "perfect-reference":
                base_range = _mean(
                    cache, reference_n, "all-in", "range", CHEAP_SEEDS
                )
                cells.append(
                    f"N={n}: {base_range * (n / reference_n) ** 0.75:>5.0f}"
                    " |  ---"
                )
            else:
                cells.append(
                    f"N={n}: {_mean(cache, n, name, 'range', CHEAP_SEEDS):>5.0f}"
                    f" | {_mean(cache, n, name, 'se_edge', CHEAP_SEEDS):.2f}"
                )
        print(f"  {name:<18}" + "   ".join(cells))
    print(
        "  (perfect-reference: range of an ideal array scales as "
        "N^(3/4) via the N^3 detection budget at fixed power; the "
        "reference row anchors at the first N's all-in range)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="membership methods vs array size, five metrics"
    )
    parser.add_argument("--stations", type=str, default=None,
                        help="comma list; default runs all of "
                        f"{DEFAULT_STATIONS}")
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--trials", type=int, default=250)
    parser.add_argument("--h0-trials", type=int, default=8000)
    parser.add_argument("--report", action="store_true",
                        help="print tables/answers from the cache only")
    args = parser.parse_args()

    station_list = (
        [int(v) for v in args.stations.split(",")]
        if args.stations
        else list(DEFAULT_STATIONS)
    )
    cache = _load_cache()
    if not args.report:
        for n in station_list:
            print(f"scoring N={n} (capacity {capacity_for(n)})...",
                  flush=True)
            score_stations(
                n,
                iterations=args.iterations,
                trials=args.trials,
                h0_trials=args.h0_trials,
                cache=cache,
            )
    complete = [
        n for n in (args.stations and station_list or DEFAULT_STATIONS)
        if f"{n}/0/all-in/gain" in cache
    ]
    if complete:
        report(cache, complete)


if __name__ == "__main__":
    main()
