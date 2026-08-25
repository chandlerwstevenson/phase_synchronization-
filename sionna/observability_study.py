"""Measured identifiability of the oscillator/channel split vs the
mismatched-Riccati prediction (observability_analysis.py).

Grid: observation rate n in {1, 2, 5, 10} per interval x anchor cadence
K in {10, 40, 160} intervals x channel motion {0, 0.05, 0.2} m/s, N=2,
seeds 0-2, 60 intervals, real OFDM piggyback observations. Per cell:

  measured residual rms   - steady window (valid & second half)
  measured locked bias    - |circular mean| of the steady residual per
                            seed, averaged (the misattribution offset)
  predicted residual std  - true_theta_std from the mismatched cycle
  predicted split std     - the hidden (theta-psi)/sqrt2 coordinate

K=160 exceeds the 60-interval horizon (one acquisition anchor, then
coast); the theory is evaluated at the effective cycle
K_eff = min(K, iterations) and that is what "K=160" means here.

Usage:
    .venv/bin/python observability_study.py            # full grid
    .venv/bin/python observability_study.py --speeds 0.2
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from clutter_sync_ofdm import run_piggyback_star
from observability_analysis import (
    jakes_channel_innovation,
    los_ramp_bias_cycle,
    split_uncertainty_cycle,
)
from ota_sync import SDRSimulationConfig

CACHE = Path(__file__).resolve().parent / "observability_cache.json"
N_OBS = (1, 2, 5, 10)
CADENCES = (10, 40, 160)
SPEEDS = (0.0, 0.05, 0.2)
SEEDS = (0, 1, 2)
ITERATIONS = 60


def steady_bias_and_rms(result) -> tuple[float, float]:
    """(|circular mean|, rms) of the steady residual, pooled over the
    valid second half of the run - row 0 (N=2 has one link)."""

    row = result.station_residuals[0]
    valid = result.station_valid[0].clone()
    half = row.numel() // 2
    valid[:half] = False
    if not torch.any(valid):
        return float("nan"), float("nan")
    steady = row[valid].to(torch.float64)
    mean_phasor = torch.mean(torch.exp(1j * steady.to(torch.complex128)))
    bias = abs(float(torch.angle(mean_phasor)))
    rms = float(torch.sqrt(torch.mean(steady.square())))
    return bias, rms


def run_cell(n_obs: int, cadence: int, speed: float) -> dict:
    biases, rmses, phase_vars, freq_vars = [], [], [], []
    for seed in SEEDS:
        settings = SDRSimulationConfig(
            num_iterations=ITERATIONS,
            seed=seed,
            device="cpu",
            channel_speed_mps=speed,
        )
        result = run_piggyback_star(
            settings,
            num_stations=2,
            anchor_every_intervals=cadence,
            obs_per_interval=n_obs,
            waveform="ofdm",
        )
        bias, rms = steady_bias_and_rms(result)
        if bias == bias:
            biases.append(bias)
            rmses.append(rms)
        phase_vars.append(result.oneway_phase_var)
        freq_vars.append(result.oneway_frequency_var)
    settings = SDRSimulationConfig(
        num_iterations=ITERATIONS, seed=0, device="cpu"
    )
    mean_phase_var = sum(phase_vars) / len(phase_vars)
    mean_freq_var = sum(freq_vars) / len(freq_vars)
    q_psi = jakes_channel_innovation(
        speed, settings.sync_interval, settings.carrier_frequency_hz
    )
    effective_cadence = min(cadence, ITERATIONS)
    prediction = split_uncertainty_cycle(
        settings, n_obs, effective_cadence, q_psi,
        mean_phase_var, mean_freq_var,
    )
    ramp_bias = los_ramp_bias_cycle(
        settings, n_obs, effective_cadence, speed,
        mean_phase_var, mean_freq_var,
    )
    predicted_total = math.sqrt(
        prediction.true_theta_std**2 + ramp_bias**2
    )
    return {
        "n_obs": n_obs,
        "cadence": cadence,
        "speed": speed,
        "measured_bias_mrad": 1e3 * sum(biases) / max(len(biases), 1),
        "measured_rms_mrad": 1e3
        * math.sqrt(sum(r * r for r in rmses) / max(len(rmses), 1)),
        "predicted_theta_mrad": 1e3 * prediction.true_theta_std,
        "predicted_ramp_bias_mrad": 1e3 * ramp_bias,
        "predicted_total_mrad": 1e3 * predicted_total,
        "believed_theta_mrad": 1e3 * prediction.believed_theta_std,
        "predicted_split_mrad": 1e3 * prediction.true_split_std,
        "oneway_phase_var": mean_phase_var,
        "oneway_freq_var": mean_freq_var,
        "seeds_valid": len(biases),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="oscillator/channel split identifiability grid"
    )
    parser.add_argument("--speeds", type=str, default=None)
    args = parser.parse_args()
    speeds = (
        tuple(float(s) for s in args.speeds.split(","))
        if args.speeds
        else SPEEDS
    )

    cache: dict[str, dict] = {}
    if CACHE.exists():
        cache = json.loads(CACHE.read_text())

    print(
        "Split identifiability: measured vs mismatched-Riccati prediction "
        f"(N=2, {ITERATIONS} intervals, seeds {list(SEEDS)})"
    )
    print(
        f"{'v m/s':>6} {'n':>3} {'K':>4} | {'meas rms':>9} {'meas bias':>10} "
        f"| {'pred rand':>9} {'pred bias':>10} {'pred total':>10} | {'ratio':>6}"
    )
    for speed in speeds:
        for cadence in CADENCES:
            for n_obs in N_OBS:
                key = f"{speed}-{cadence}-{n_obs}"
                if key not in cache or "predicted_total_mrad" not in cache[key]:
                    cache[key] = run_cell(n_obs, cadence, speed)
                    CACHE.write_text(json.dumps(cache, indent=1))
                cell = cache[key]
                ratio = (
                    cell["measured_rms_mrad"] / cell["predicted_total_mrad"]
                    if cell["predicted_total_mrad"] > 0
                    else float("nan")
                )
                print(
                    f"{speed:>6.2f} {n_obs:>3} {cadence:>4} | "
                    f"{cell['measured_rms_mrad']:>7.1f}  "
                    f"{cell['measured_bias_mrad']:>8.1f}  | "
                    f"{cell['predicted_theta_mrad']:>7.1f}  "
                    f"{cell['predicted_ramp_bias_mrad']:>8.1f}  "
                    f"{cell['predicted_total_mrad']:>8.1f}  | "
                    f"{ratio:>6.2f}"
                )


if __name__ == "__main__":
    main()
