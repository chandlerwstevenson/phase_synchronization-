"""Experiment E addendum: the branch-check ablation under adverse
acquisition.

At the default initial offset (1.2 rad < pi/2) the acquisition anchor
never picks the wrong branch, so the check never fires and its
ablation shows nothing - itself a finding worth recording. Here the
initial oscillator offset is set beyond pi/2 (2.2 rad), where the
half-difference's branch pick at acquisition is wrong by construction,
and the check's rescue (or its absence) becomes visible. 12 seeds,
60 intervals, N=2, K=40, OFDM.
"""

from __future__ import annotations

import json

import torch

from ota_sync import SDRSimulationConfig
from piggyback_variant_e import run_piggyback_variant_e

SEEDS = list(range(12))


def run_arm(branch_check: bool):
    rows = []
    for seed in SEEDS:
        settings = SDRSimulationConfig(
            num_iterations=60, seed=seed, device="cpu",
            slave_initial_phase=2.2,
        )
        result = run_piggyback_variant_e(
            settings, num_stations=2, anchor_every_intervals=40,
            branch_check=branch_check,
        )
        row = result.station_residuals[0]
        tail = row[int(row.numel() * 0.75):]
        tail_abs = 1e3 * float(torch.mean(torch.abs(tail)))
        captured = tail_abs > 1e3 * 3.14159 / 2.0
        rows.append({
            "seed": seed,
            "tail_abs_mrad": tail_abs,
            "captured": bool(captured),
            "worst_rms_mrad": result.worst_rms_mrad,
        })
        print(f"  check={'on ' if branch_check else 'off'} seed {seed}: "
              f"tail |residual| {tail_abs:7.1f} mrad "
              f"{'ANTI-PHASE' if captured else ''}")
    return rows


def main() -> None:
    results = json.load(open("ablation_results.json"))
    print("adverse acquisition (initial offset 2.2 rad), check ON")
    results["adverse_check_on"] = run_arm(True)
    print("adverse acquisition (initial offset 2.2 rad), check OFF")
    results["adverse_check_off"] = run_arm(False)
    on = sum(r["captured"] for r in results["adverse_check_on"])
    off = sum(r["captured"] for r in results["adverse_check_off"])
    print(f"anti-phase captures: {on}/12 with check, {off}/12 without")
    with open("ablation_results.json", "w") as handle:
        json.dump(results, handle)


if __name__ == "__main__":
    main()
