"""Experiment E part 1: component ablation of the opportunistic sync
loop - remove one component at a time and measure exactly what breaks.

Baseline: N=2, OFDM observations, K=40 anchors, static channel,
seeds 0-2, 160 intervals (4 anchor cycles). Ablations:

  no-decomposition   2-state filter (channel state pinned to zero):
                     the composite observation is treated as pure
                     oscillator phase
  no-anchors         one acquisition anchor, then none (K -> inf);
                     branch check also off so the gauge drift is
                     measured un-reflected; 10 seeds, ensemble
                     variance growth vs the q_theta/2 prediction
  no-branch-check    12 seeds with and without the check; count
                     anti-phase captures (tail-quarter mean |residual|
                     beyond pi/2)
  zc-waveform        dedicated preamble observations instead of OFDM
  dynamic-channel    environment moving at 0.1 m/s, K=40 and K=10

Results cached to ablation_results.json (delete to re-run).
"""

from __future__ import annotations

import json
import math
import os

import torch

from ota_sync import SDRSimulationConfig
from piggyback_variant_e import run_piggyback_variant_e

PY_CACHE = "ablation_results.json"
BASE_ITERATIONS = 160
SEEDS = [0, 1, 2]
CHECK_SEEDS = list(range(12))


def tail_stats(result, fraction: float = 0.25):
    """(worst-station rms mrad over valid samples, tail-quarter mean
    residual mrad per station, tail-quarter mean |residual| mrad)."""

    rows = result.station_residuals
    total = rows.shape[1]
    tail = slice(int(total * (1.0 - fraction)), total)
    rms = result.worst_rms_mrad
    bias = [1e3 * float(torch.mean(row[tail])) for row in rows]
    magnitude = [
        1e3 * float(torch.mean(torch.abs(row[tail]))) for row in rows
    ]
    return rms, bias, magnitude


def run_cell(label, seeds, iterations=BASE_ITERATIONS, num_stations=2,
             collect_rows=False, **kwargs):
    out = []
    for seed in seeds:
        settings_kwargs = dict(
            num_iterations=iterations, seed=seed, device="cpu"
        )
        speed = kwargs.pop("channel_speed_mps", None)
        if speed is not None:
            settings_kwargs["channel_speed_mps"] = speed
        settings = SDRSimulationConfig(**settings_kwargs)
        if speed is not None:
            kwargs["channel_speed_mps"] = speed  # restore for next seed
            kwargs2 = {k: v for k, v in kwargs.items()
                       if k != "channel_speed_mps"}
        else:
            kwargs2 = dict(kwargs)
        result = run_piggyback_variant_e(
            settings, num_stations=num_stations, **kwargs2
        )
        rms, bias, magnitude = tail_stats(result)
        record = {
            "seed": seed,
            "worst_rms_mrad": rms,
            "tail_bias_mrad": bias,
            "tail_abs_mrad": magnitude,
            "mean_gain": float(torch.mean(
                result.array_gain[result.all_valid]
            )) if bool(result.all_valid.any()) else float("nan"),
            "airtime": result.piggyback_airtime,
            "detection_rate": result.detection_rate,
        }
        if collect_rows:
            record["residual_row0"] = [
                float(v) for v in result.station_residuals[0]
            ]
        out.append(record)
        print(f"  {label} seed {seed}: worst rms {rms:.1f} mrad, "
              f"tail bias {bias[0]:+.1f} mrad")
    return out


def main() -> None:
    if os.path.exists(PY_CACHE):
        print(f"{PY_CACHE} exists - delete to re-run")
        return
    results = {}

    print("baseline (full architecture, OFDM, K=40)")
    results["full"] = run_cell("full", SEEDS, anchor_every_intervals=40)

    print("baseline at N=6 (spot check)")
    results["full_n6"] = run_cell(
        "full_n6", SEEDS, num_stations=6, anchor_every_intervals=40
    )

    print("ablation 1: no channel-state decomposition (2-state filter)")
    results["no_decomposition"] = run_cell(
        "no-decomp", SEEDS, anchor_every_intervals=40, channel_state=False
    )

    print("ablation 2: no anchors after acquisition (check off, 10 seeds)")
    results["no_anchors"] = run_cell(
        "no-anchors", list(range(10)), iterations=60,
        anchor_every_intervals=10**6, branch_check=False,
        collect_rows=True,
    )

    print("ablation 3a: branch check ON, 12 seeds, 60 intervals")
    results["check_on"] = run_cell(
        "check-on", CHECK_SEEDS, iterations=60, anchor_every_intervals=40
    )
    print("ablation 3b: branch check OFF, 12 seeds, 60 intervals")
    results["check_off"] = run_cell(
        "check-off", CHECK_SEEDS, iterations=60, anchor_every_intervals=40,
        branch_check=False,
    )

    print("ablation 4: dedicated-preamble observations (zc)")
    results["zc_waveform"] = run_cell(
        "zc", SEEDS, anchor_every_intervals=40, waveform="zc"
    )

    print("ablation 5: dynamic channel 0.1 m/s, K=40")
    results["dynamic_k40"] = run_cell(
        "dyn-k40", SEEDS, anchor_every_intervals=40,
        channel_speed_mps=0.1,
    )
    print("ablation 5b: dynamic channel 0.1 m/s, K=10")
    results["dynamic_k10"] = run_cell(
        "dyn-k10", SEEDS, anchor_every_intervals=10,
        channel_speed_mps=0.1,
    )

    # Prediction constants for the no-anchor drift analysis.
    settings = SDRSimulationConfig(device="cpu")
    substeps = 5
    dt_samples = int(round(settings.sync_interval / substeps
                           * settings.sample_rate))
    q_theta_sub = (
        2.0 * settings.phase_process_std_rad**2 / substeps
        + settings.phase_noise_std_rad**2 * dt_samples
    )
    results["_predictions"] = {
        "q_theta_per_substep": q_theta_sub,
        "null_drift_per_substep": q_theta_sub / 2.0,
        "q_psi_filter_per_substep": 0.01**2 / substeps,
    }

    with open(PY_CACHE, "w") as handle:
        json.dump(results, handle)
    print(f"saved {PY_CACHE}")


if __name__ == "__main__":
    main()
