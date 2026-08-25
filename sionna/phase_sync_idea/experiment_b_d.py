"""Experiments B and D of the phase_sync_idea test plan.

B - environment coherence time: residual vs channel motion (converted
    to coherence time T_c = 0.423/f_D) at two anchor cadences, with
    the observability theory's Doppler-ramp boundary overlaid.
B boundary: locked ramp bias ~ pi * f_D * T * K = budget  =>
    T_c* = 0.423 * pi * T * K / budget.

D - the U-curve hypothesis test: residual and locked misattribution
    bias vs free-observation rate n, in static and slowly-moving
    environments. External reviewer predicts a U (too many obs =>
    channel drift misread as oscillator drift); our observability
    theory predicts the motion bias is n-independent (no U). Statistic
    excludes the first 4 anchor cycles (the documented acquisition
    transient), so this is a steady-state measurement by construction.

All runs: N=2 stations, OFDM one-way observations, seeds as stated,
results cached to JSON so partial runs resume.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from clutter_sync_ofdm import run_piggyback_star
from ota_sync import SDRSimulationConfig

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "experiment_b_d_cache.json")
CARRIER_HZ = 915e6
SPEED_OF_LIGHT = 299792458.0
BUDGET = 0.314

B_SPEEDS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
B_CADENCES = [10, 40]
B_SEEDS = [0, 1, 2]

D_RATES = [1, 2, 5, 10, 20]
D_SPEEDS = [0.0, 0.02, 0.05, 0.1]
D_SEEDS = [0, 1, 2, 3, 4]
D_CADENCE = 40
D_TOTAL_CYCLES = 8   # run 8 anchor cycles ...
D_SKIP_CYCLES = 4    # ... and keep only the last 4 in the statistic


def coherence_time_s(speed_mps: float) -> float:
    if speed_mps <= 0.0:
        return float("inf")
    doppler = speed_mps * CARRIER_HZ / SPEED_OF_LIGHT
    return 0.423 / doppler


def boundary_coherence_time_s(cadence: int, sync_interval: float) -> float:
    """T_c at which the predicted locked ramp bias equals the budget."""

    return 0.423 * math.pi * sync_interval * cadence / BUDGET


def tail_stats(result, skip_intervals: int, n_obs: int) -> tuple[float, float]:
    """(rms mrad, |circular-mean| bias mrad) over valid substeps after
    the first `skip_intervals` intervals."""

    row = result.station_residuals[0]
    valid = result.station_valid[0].clone()
    valid[: skip_intervals * n_obs] = False
    if not torch.any(valid):
        return float("nan"), float("nan")
    tail = row[valid]
    rms = 1e3 * torch.sqrt(torch.mean(tail.square())).item()
    bias = 1e3 * abs(math.atan2(
        torch.mean(torch.sin(tail)).item(),
        torch.mean(torch.cos(tail)).item(),
    ))
    return rms, bias


def load_cache() -> dict:
    if os.path.exists(CACHE):
        with open(CACHE) as handle:
            return json.load(handle)
    return {}


def save_cache(cache: dict) -> None:
    with open(CACHE, "w") as handle:
        json.dump(cache, handle, indent=1)


def run_cell(key: str, cache: dict, *, speed: float, cadence: int,
             n_obs: int, intervals: int, skip_intervals: int,
             seed: int) -> dict:
    if key in cache:
        return cache[key]
    settings = SDRSimulationConfig(
        num_iterations=intervals, seed=seed, device="cpu",
        channel_speed_mps=speed,
    )
    result = run_piggyback_star(
        settings, num_stations=2, anchor_every_intervals=cadence,
        obs_per_interval=n_obs, waveform="ofdm",
    )
    rms, bias = tail_stats(result, skip_intervals, n_obs)
    cache[key] = {
        "rms_mrad": rms,
        "bias_mrad": bias,
        "full_rms_mrad": result.worst_rms_mrad,
        "airtime": result.piggyback_airtime,
        "detection": result.detection_rate,
    }
    save_cache(cache)
    return cache[key]


def part_b() -> None:
    cache = load_cache()
    print("Experiment B: residual vs environment coherence time "
          f"(N=2, OFDM, n_obs=5, seeds {B_SEEDS})")
    print(f"{'v m/s':>7} {'T_c s':>9} {'K':>4} {'rms mrad':>12} "
          f"{'bias mrad':>10}")
    for cadence in B_CADENCES:
        intervals = max(60, 4 * cadence)
        for speed in B_SPEEDS:
            rows = [
                run_cell(
                    f"B|{speed}|{cadence}|{seed}", cache,
                    speed=speed, cadence=cadence, n_obs=5,
                    intervals=intervals, skip_intervals=0, seed=seed,
                )
                for seed in B_SEEDS
            ]
            rms = [r["rms_mrad"] for r in rows if r["rms_mrad"] == r["rms_mrad"]]
            bias = [r["bias_mrad"] for r in rows if r["bias_mrad"] == r["bias_mrad"]]
            mean_rms = sum(rms) / len(rms) if rms else float("nan")
            std_rms = (
                math.sqrt(sum((v - mean_rms) ** 2 for v in rms) / len(rms))
                if len(rms) > 1 else 0.0
            )
            mean_bias = sum(bias) / len(bias) if bias else float("nan")
            tc = coherence_time_s(speed)
            tc_text = f"{tc:9.2f}" if tc != float("inf") else "      inf"
            print(f"{speed:>7.3f} {tc_text} {cadence:>4} "
                  f"{mean_rms:>7.1f}±{std_rms:<4.1f} {mean_bias:>10.1f}")
    for cadence in B_CADENCES:
        print(f"theory boundary K={cadence}: T_c* = "
              f"{boundary_coherence_time_s(cadence, 0.05):.2f} s "
              f"(v* = {0.423 / boundary_coherence_time_s(cadence, 0.05) / (CARRIER_HZ / SPEED_OF_LIGHT):.4f} m/s)")


def part_d() -> None:
    cache = load_cache()
    intervals = D_TOTAL_CYCLES * D_CADENCE
    skip = D_SKIP_CYCLES * D_CADENCE
    print("Experiment D: residual vs observation rate "
          f"(N=2, K={D_CADENCE}, {intervals} intervals, first "
          f"{skip} discarded, seeds {D_SEEDS})")
    print(f"{'v m/s':>7} {'n':>4} {'rms mrad':>13} {'bias mrad':>13}")
    for speed in D_SPEEDS:
        for n_obs in D_RATES:
            rows = [
                run_cell(
                    f"D|{speed}|{n_obs}|{seed}", cache,
                    speed=speed, cadence=D_CADENCE, n_obs=n_obs,
                    intervals=intervals, skip_intervals=skip, seed=seed,
                )
                for seed in D_SEEDS
            ]
            rms = [r["rms_mrad"] for r in rows if r["rms_mrad"] == r["rms_mrad"]]
            bias = [r["bias_mrad"] for r in rows if r["bias_mrad"] == r["bias_mrad"]]
            mean_rms = sum(rms) / len(rms) if rms else float("nan")
            std_rms = (
                math.sqrt(sum((v - mean_rms) ** 2 for v in rms) / len(rms))
                if len(rms) > 1 else 0.0
            )
            mean_bias = sum(bias) / len(bias) if bias else float("nan")
            std_bias = (
                math.sqrt(sum((v - mean_bias) ** 2 for v in bias) / len(bias))
                if len(bias) > 1 else 0.0
            )
            print(f"{speed:>7.3f} {n_obs:>4} "
                  f"{mean_rms:>8.1f}±{std_rms:<4.1f} "
                  f"{mean_bias:>8.1f}±{std_bias:<4.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="experiments B and D")
    parser.add_argument("--part", choices=["b", "d", "both"], default="both")
    args = parser.parse_args()
    if args.part in ("b", "both"):
        part_b()
    if args.part in ("d", "both"):
        part_d()


if __name__ == "__main__":
    main()
