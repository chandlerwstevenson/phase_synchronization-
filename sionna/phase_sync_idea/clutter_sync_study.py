"""Clutter-referenced synchronization: can the static environment carry
the sync load the channel currently pays airtime for?

The idea (user-proposed): the array radiates sensing bursts anyway
(detection/waveform.py transmits ~49 OFDM frames per 50 ms interval per
station). When station i radiates, station j hears the direct
inter-station path plus echoes off STATIC clutter; either way the
received phase is wrap(theta_i - theta_j + phi_path) with phi_path
CONSTANT while the environment is static. Every sensing frame is
therefore a free one-way relative-phase observation; dedicated two-way
exchanges are needed only to (re-)pin the unknown static phi_path and
for acquisition.

This is exactly the hybrid model's estimation structure
(hybrid_calibration/hybrid.py: a 3-state EKF where cheap one-way
pilots observe theta + phi_c and sparse two-way anchors every K
intervals re-pin the split) - and the hybrid's one-way pilot already
propagates through the full Sionna TDL channel, i.e. through the
frozen LOS + clutter composite. What changes here is the ACCOUNTING,
not the physics: the hybrid charges its one-way frames and micro
pilots to the sync budget; under the piggyback reading they ride on
frames the array transmits for sensing regardless, so the
sync-attributable airtime is the anchors alone. We charge BOTH anchor
captures (forward and reverse) - conservative, since the forward
anchor frame could itself piggyback.

Honest fidelity gap, stated up front: the simulated one-way pilot uses
the Zadoff-Chu sync preamble, not the OFDM sensing frame. Since the
sensing receiver knows the transmitted frame (detection/waveform.py
matched-filters its own reference), phase estimation off the OFDM
burst is the same operation at comparable time-bandwidth product, but
its measurement covariance would differ; closing that gap means
running the one-way estimator on the actual OFDM waveform. Until
then, this study bounds what the accounting is worth, on the repo's
own physical layer.

Known defect respected: the hybrid's one-way frequency observation is
biased by LOS Doppler (RESEARCH_IDEAS.md; needs a 4th state). The
headline runs are static; the moving-clutter stressor is included
precisely to show where the scheme breaks.

Usage:
    .venv/bin/python clutter_sync_study.py            # full study
    .venv/bin/python clutter_sync_study.py --quick    # 1 seed, no sweeps
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace

import torch

from hybrid_calibration import run_hybrid_simulation
from ota_sync import SDRSimulationConfig, run_two_way_simulation
from ota_sync.scheduled import run_scheduled_star
from ota_sync.sdr import SDRRadioLink, make_sync_preamble
from ota_sync.core import resolve_device


SPEED_OF_LIGHT = 299792458.0


def full_capture_samples(settings: SDRSimulationConfig) -> int:
    """Length of one full-frame capture, from the same link construction
    the hybrid uses (probe object only; no simulation run)."""

    device = resolve_device("cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(settings.seed + 1)
    preamble = make_sync_preamble(settings, device)
    link = SDRRadioLink(settings, preamble, device, generator)
    return link.input_length + link.l_tot - 1


def piggyback_airtime_fraction(
    settings: SDRSimulationConfig, anchor_every_intervals: int
) -> float:
    """Sync-attributable airtime when one-way pilots ride on sensing
    frames: only the two-way anchors are charged, both directions."""

    interval_samples = int(
        round(settings.sync_interval * settings.sample_rate)
    )
    return (
        2.0
        * full_capture_samples(settings)
        / (anchor_every_intervals * interval_samples)
    )


def matched_channel_drift_std(
    settings: SDRSimulationConfig, speed_mps: float
) -> float:
    """Per-interval channel-phase drift prior matched to Jakes
    decorrelation at the given speed: std ~ sqrt(2*(1 - J0(2*pi*fD*T))).
    Small-decorrelation approximation of the composite phase walk; the
    static default (0.01 rad) is kept as a floor."""

    doppler_hz = speed_mps * settings.carrier_frequency_hz / SPEED_OF_LIGHT
    argument = torch.tensor(
        2.0 * math.pi * doppler_hz * settings.sync_interval,
        dtype=torch.float64,
    )
    decorrelation = 2.0 * (1.0 - torch.special.bessel_j0(argument).item())
    return max(0.01, math.sqrt(max(decorrelation, 0.0)))


def run_clutter_referenced(
    settings: SDRSimulationConfig,
    anchor_every_intervals: int,
    micro_pilots_per_interval: int = 4,
    channel_drift_std_rad: float = 0.01,
):
    """Hybrid physics, piggyback accounting. Returns (result,
    native_airtime, piggyback_airtime)."""

    result = run_hybrid_simulation(
        settings,
        micro_pilots_per_interval=micro_pilots_per_interval,
        anchor_every_intervals=anchor_every_intervals,
        channel_drift_std_rad=channel_drift_std_rad,
    )
    return (
        result,
        result.airtime_fraction,
        piggyback_airtime_fraction(settings, anchor_every_intervals),
    )


def _mean_std(values: list[float]) -> tuple[float, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return tensor.mean().item(), (
        tensor.std().item() if len(values) > 1 else 0.0
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="clutter/sensing-referenced sync vs paid-airtime sync"
    )
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--anchor-cadences", type=str, default="5,10,20,40")
    parser.add_argument("--micro-pilots", type=int, default=4)
    parser.add_argument("--quick", action="store_true",
                        help="1 seed, skip SNR and motion stressors")
    args = parser.parse_args()

    seeds = [0] if args.quick else [int(s) for s in args.seeds.split(",")]
    cadences = [int(k) for k in args.anchor_cadences.split(",")]

    print(
        "Clutter-referenced sync (hybrid physics, piggyback accounting), "
        f"{args.iterations} intervals, seeds {seeds}"
    )
    print(
        "piggyback reading: one-way frames + micro pilots ride on the "
        "sensing bursts the array transmits anyway; only two-way anchors "
        "(both directions) are charged to sync"
    )

    # ---- baselines --------------------------------------------------
    print("\n=== baselines (paid airtime) ===")
    rows = {"twoway": [], "micro-star": []}
    for seed in seeds:
        settings = SDRSimulationConfig(
            num_iterations=args.iterations, seed=seed, device="cpu"
        )
        twoway = run_two_way_simulation(settings)
        rows["twoway"].append(
            (twoway.steady_state_phase_rms, twoway.mean_coherent_gain,
             twoway.airtime_fraction)
        )
        star = run_scheduled_star(
            settings, num_stations=2, policy="scheduled",
            multi_fidelity=True,
        )
        rows["micro-star"].append(
            (star.station_steady_rms[0], star.mean_array_gain,
             star.airtime_used_fraction)
        )
    for label, values in rows.items():
        rms_mean, rms_std = _mean_std([1e3 * v[0] for v in values])
        gain_mean, _ = _mean_std([v[1] for v in values])
        air_mean, _ = _mean_std([v[2] for v in values])
        print(
            f"  {label:<22} rms {rms_mean:6.1f}±{rms_std:4.1f} mrad  "
            f"gain {100 * gain_mean:6.2f}%  sync airtime {100 * air_mean:5.1f}%"
        )

    # ---- clutter-referenced vs anchor cadence ----------------------
    print(
        "\n=== clutter-referenced (static environment), by anchor "
        "cadence K ==="
    )
    print(
        f"  {'K':>4} {'rms mrad':>14} {'gain %':>8} "
        f"{'native airtime %':>17} {'piggyback %':>12} {'detect %':>9}"
    )
    for cadence in cadences:
        cells = []
        for seed in seeds:
            settings = SDRSimulationConfig(
                num_iterations=args.iterations, seed=seed, device="cpu"
            )
            result, native, piggyback = run_clutter_referenced(
                settings, cadence, args.micro_pilots
            )
            cells.append(
                (result.steady_state_phase_rms, result.mean_coherent_gain,
                 native, piggyback, result.detection_rate)
            )
        rms_mean, rms_std = _mean_std([1e3 * c[0] for c in cells])
        gain_mean, _ = _mean_std([c[1] for c in cells])
        native_mean, _ = _mean_std([c[2] for c in cells])
        piggy_mean, _ = _mean_std([c[3] for c in cells])
        detect_mean, _ = _mean_std([c[4] for c in cells])
        print(
            f"  {cadence:>4} {rms_mean:8.1f}±{rms_std:4.1f} "
            f"{100 * gain_mean:8.2f} {100 * native_mean:17.1f} "
            f"{100 * piggy_mean:12.2f} {100 * detect_mean:9.1f}"
        )

    if args.quick:
        return

    # ---- echo-SNR stressor: referencing weaker clutter echoes ------
    print(
        "\n=== echo-SNR stressor (K=20, seed 0): one-way reference SNR "
        "swept below the direct-path budget ==="
    )
    print(f"  {'SNR dB':>7} {'rms mrad':>10} {'gain %':>8} {'detect %':>9}")
    for snr_db in (20.0, 10.0, 5.0, 0.0):
        settings = SDRSimulationConfig(
            num_iterations=args.iterations, seed=0, device="cpu",
            snr_db=snr_db,
        )
        result, _, _ = run_clutter_referenced(settings, 20, args.micro_pilots)
        print(
            f"  {snr_db:7.0f} {1e3 * result.steady_state_phase_rms:10.1f} "
            f"{100 * result.mean_coherent_gain:8.2f} "
            f"{100 * result.detection_rate:9.1f}"
        )

    # ---- moving-clutter stressor -----------------------------------
    print(
        "\n=== moving-clutter stressor (seed 0, matched channel prior): "
        "static-reference assumption removed ==="
    )
    print(
        f"  {'v m/s':>6} {'K':>4} {'prior rad':>10} {'rms mrad':>10} "
        f"{'gain %':>8} {'piggyback %':>12}"
    )
    for speed in (0.2, 0.5):
        for cadence in (5, 1):
            settings = SDRSimulationConfig(
                num_iterations=args.iterations, seed=0, device="cpu",
                channel_speed_mps=speed,
            )
            prior = matched_channel_drift_std(settings, speed)
            result, _, piggyback = run_clutter_referenced(
                settings, cadence, args.micro_pilots,
                channel_drift_std_rad=prior,
            )
            print(
                f"  {speed:6.1f} {cadence:>4} {prior:10.3f} "
                f"{1e3 * result.steady_state_phase_rms:10.1f} "
                f"{100 * result.mean_coherent_gain:8.2f} "
                f"{100 * piggyback:12.2f}"
            )


if __name__ == "__main__":
    main()
