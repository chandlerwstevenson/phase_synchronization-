"""Experiment E part 2: the per-observation noise decomposition

    sigma^2_obs(L) = sigma^2_thermal(SNR, L) + sigma^2_walk(L) + rest

established by sweeping the dedicated preamble's length and overlaying
ex-ante predictions - nothing fitted to the measured totals.

Components, all computed before looking at the measurements:
  thermal + white-PM:  (1/(2*SNR_lin) + sigma_wpm^2) / L_int
      L_int = integrated samples (reps x long length for the ZC
      preamble; 960 for the OFDM burst)
  intra-capture oscillator walk: the capture applies ONE white-FM
      random walk (per-sample std sigma_pn = 2e-4) to the received
      stream; its effect on the estimate is computed by running the
      ACTUAL estimator on clean waveform x exp(j*walk) - a Monte Carlo
      of the model term only (no thermal noise, no channel), 200 draws.
      The naive closed form sigma_pn^2 * L_span / 3 (mean-of-walk vs
      end reference) is printed alongside for intuition.
  rest: multipath resampling + RF impairments - whatever the first two
      do not explain; reported as the residual gap, not fitted.

Measured totals come from calibrate_oneway_noise (all impairments,
frozen oscillators between captures, intra-capture LO noise active).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import replace

import torch

import clutter_sync_ofdm
from clutter_sync_ofdm import (
    OFDMOneWayEstimator,
    calibrate_oneway_noise,
    make_ofdm_frame_pool,
)


def fresh_calibration(settings, mode, device, captures=160):
    """calibrate_oneway_noise with the module cache cleared first: its
    cache key omits the preamble length, so a length sweep would
    otherwise return the first length's numbers for every length."""

    clutter_sync_ofdm._CALIBRATION_CACHE.clear()
    return calibrate_oneway_noise(settings, mode, device, captures=captures)
from ota_sync import SDRSimulationConfig
from ota_sync.core import REAL_DTYPE, COMPLEX_DTYPE, resolve_device
from ota_sync.sdr import SDRSynchronizer, make_sync_preamble

LENGTHS = [127, 255, 511, 1023, 2047, 4095, 8191]
SNRS_DB = [10.0, 20.0, 30.0]
WALK_DRAWS = 200
CACHE = "capture_model_results.json"


def zc_settings(length: int, snr_db: float) -> SDRSimulationConfig:
    return SDRSimulationConfig(
        device="cpu",
        snr_db=snr_db,
        long_sequence_length=length,
        long_cp_length=min(128, max(8, length // 4)),
    )


def walk_only_variance_zc(settings: SDRSimulationConfig) -> float:
    """Estimator response to the intra-capture walk alone: clean
    preamble times exp(j*walk), no noise, no channel, no offsets."""

    device = resolve_device("cpu")
    preamble = make_sync_preamble(settings, device)
    synchronizer = SDRSynchronizer(settings, preamble)
    generator = torch.Generator().manual_seed(4242)
    pad = 64
    estimates = []
    for _ in range(WALK_DRAWS):
        walk = torch.cumsum(
            torch.randn(preamble.waveform.numel(), dtype=REAL_DTYPE,
                        generator=generator)
            * settings.phase_noise_std_rad,
            dim=0,
        )
        stream = torch.cat(
            (
                torch.zeros(pad, dtype=COMPLEX_DTYPE),
                preamble.waveform * torch.exp(1j * walk),
                torch.zeros(pad, dtype=COMPLEX_DTYPE),
            )
        )
        measurement = synchronizer.estimate(stream)
        if measurement.detected:
            estimates.append(measurement.phase)
    if len(estimates) < WALK_DRAWS // 2:
        return float("nan")
    stack = torch.stack(estimates)
    return float(torch.var(stack))


def walk_only_variance_ofdm(settings: SDRSimulationConfig) -> float:
    device = resolve_device("cpu")
    pool = make_ofdm_frame_pool(8, device, 1234)
    estimator = OFDMOneWayEstimator(settings)
    generator = torch.Generator().manual_seed(4242)
    pad = 64
    zero_omega = torch.zeros((), dtype=REAL_DTYPE)
    estimates = []
    for draw in range(WALK_DRAWS):
        waveform = pool[draw % len(pool)].waveform
        walk = torch.cumsum(
            torch.randn(waveform.numel(), dtype=REAL_DTYPE,
                        generator=generator)
            * settings.phase_noise_std_rad,
            dim=0,
        )
        stream = torch.cat(
            (
                torch.zeros(pad, dtype=COMPLEX_DTYPE),
                waveform * torch.exp(1j * walk),
                torch.zeros(pad, dtype=COMPLEX_DTYPE),
            )
        )
        estimate = estimator.estimate(stream, waveform, zero_omega)
        if estimate.detected:
            estimates.append(estimate.phase)
    stack = torch.stack(estimates)
    return float(torch.var(stack))


def main() -> None:
    if os.path.exists(CACHE):
        print(f"{CACHE} exists - delete to re-run")
        return
    device = resolve_device("cpu")
    results = {"zc": [], "ofdm": []}

    for snr_db in SNRS_DB:
        snr = 10.0 ** (snr_db / 10.0)
        for length in LENGTHS:
            settings = zc_settings(length, snr_db)
            preamble = make_sync_preamble(settings, device)
            l_int = settings.long_repetitions * length
            span = preamble.length
            thermal = (1.0 / (2.0 * snr)
                       + settings.phase_noise_white_pm_std_rad**2) / l_int
            walk = walk_only_variance_zc(settings)
            naive_walk = settings.phase_noise_std_rad**2 * span / 3.0
            measured = fresh_calibration(
                settings, "zc", device, captures=160
            )
            results["zc"].append({
                "snr_db": snr_db, "length": length, "l_int": l_int,
                "span": span,
                "thermal_var": thermal, "walk_var": walk,
                "naive_walk_var": naive_walk,
                "measured_var": measured[0],
                "detect": measured[2],
            })
            print(
                f"zc  SNR {snr_db:4.0f} L {length:5d} span {span:6d}: "
                f"measured {1e3 * math.sqrt(max(measured[0], 0)):6.2f} mrad"
                f"  pred(th+walk) "
                f"{1e3 * math.sqrt(thermal + walk):6.2f}"
                f"  [th {1e3 * math.sqrt(thermal):5.2f}"
                f" walk {1e3 * math.sqrt(walk):5.2f}]"
                f"  detect {measured[2]:.2f}"
            )
        settings = SDRSimulationConfig(device="cpu", snr_db=snr_db)
        l_int = 960
        thermal = (1.0 / (2.0 * snr)
                   + settings.phase_noise_white_pm_std_rad**2) / l_int
        walk = walk_only_variance_ofdm(settings)
        measured = fresh_calibration(settings, "ofdm", device,
                                          captures=160)
        results["ofdm"].append({
            "snr_db": snr_db, "length": 960, "l_int": l_int, "span": 960,
            "thermal_var": thermal, "walk_var": walk,
            "naive_walk_var": settings.phase_noise_std_rad**2 * 960 / 3.0,
            "measured_var": measured[0],
            "detect": measured[2],
        })
        print(
            f"ofdm SNR {snr_db:4.0f} L   960 span    960: "
            f"measured {1e3 * math.sqrt(max(measured[0], 0)):6.2f} mrad"
            f"  pred(th+walk) {1e3 * math.sqrt(thermal + walk):6.2f}"
            f"  [th {1e3 * math.sqrt(thermal):5.2f}"
            f" walk {1e3 * math.sqrt(walk):5.2f}]"
            f"  detect {measured[2]:.2f}"
        )

    # Predicted optimum length at each SNR from the two analytic terms
    # (thermal ~ c1/L, walk ~ c2*L): L* = sqrt(c1/c2).
    for snr_db in SNRS_DB:
        snr = 10.0 ** (snr_db / 10.0)
        settings = SDRSimulationConfig(device="cpu")
        c1 = (1.0 / (2.0 * snr)
              + settings.phase_noise_white_pm_std_rad**2) / 2.0  # per L (reps=2)
        c2 = settings.phase_noise_std_rad**2 * (2.4 / 3.0)  # span ~ 2.4x L
        optimum = math.sqrt(c1 / c2)
        print(f"predicted optimum long-sequence length at "
              f"{snr_db:.0f} dB: ~{optimum:.0f} samples")

    with open(CACHE, "w") as handle:
        json.dump(results, handle)
    print(f"saved {CACHE}")


if __name__ == "__main__":
    main()
