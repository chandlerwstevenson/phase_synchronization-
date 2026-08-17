"""Residual and airtime as a function of link SNR - the measured
anchors for figures/studies/SNR_LAW.md.

Three curves per SNR point:
  1. measured steady residual of a fully-serviced two-station link
     (every interval synced - the residual floor at that SNR)
  2. the ex-ante DARE posterior phase std at that SNR (theory)
  3. the ex-ante coast time tau(SNR) at a fixed budget, with and
     without the resampling-noise term, and the airtime it implies
"""

from __future__ import annotations

import math

import torch

from coast_law import (
    dare_posterior,
    link_matrices,
    predict_coast_time,
)
from ota_sync import SDRSimulationConfig
from ota_sync.scheduled import run_scheduled_star

SNRS = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
SEEDS = [0, 1, 2]
BUDGET = 0.314
RESAMPLING_VAR = 0.153**2


def main() -> None:
    print("SNR sweep: measured residual (N=2, serviced every interval, "
          f"60 intervals, seeds {SEEDS}) vs ex-ante theory")
    print(f"{'SNR dB':>7} {'measured rms':>14} {'DARE floor':>11} "
          f"{'tau(sdr)':>9} {'tau(tcxo)':>10} {'tau(tcxo,+rs)':>14} "
          f"{'airtime/link':>13}")
    for snr in SNRS:
        measured = []
        for seed in SEEDS:
            settings = SDRSimulationConfig(
                num_iterations=60, seed=seed, device="cpu", snr_db=snr
            )
            result = run_scheduled_star(
                settings, num_stations=2, policy="uniform"
            )
            rms = result.station_steady_rms[0]
            if rms == rms:
                measured.append(1e3 * rms)
        mean = sum(measured) / len(measured) if measured else float("nan")
        base = SDRSimulationConfig(device="cpu")
        matrices = link_matrices(
            base, "sdr", snr, 60 * base.sync_interval
        )
        posterior = dare_posterior(matrices)
        floor = 1e3 * math.sqrt(max(float(posterior[0, 0]), 0.0))
        tau_sdr = predict_coast_time(
            "sdr", snr, 1, base.sync_interval, BUDGET, base
        )
        tau_tcxo = predict_coast_time(
            "tcxo", snr, 1, base.sync_interval, BUDGET, base
        )
        tau_tcxo_rs = predict_coast_time(
            "tcxo", snr, 1, base.sync_interval, BUDGET, base,
            extra_phase_measurement_var=RESAMPLING_VAR,
        )
        # one full two-way exchange = 2 captures; airtime per link at
        # cadence tau, as a fraction of the frame
        capture_fraction = 0.19124 / 2.0  # one capture (measured: two-way
        # every interval = 19.124% at N=2, so one capture = half per T)
        airtime = 2.0 * capture_fraction * base.sync_interval / max(
            tau_tcxo, 1e-9
        )
        print(f"{snr:>7.0f} {mean:>11.1f} mrad {floor:>8.1f} mrad "
              f"{tau_sdr:>8.3f}s {tau_tcxo:>9.3f}s {tau_tcxo_rs:>13.3f}s "
              f"{100 * airtime:>12.2f}%")


if __name__ == "__main__":
    main()
