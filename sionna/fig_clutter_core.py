"""Figures for the core clutter-referenced sync study (fresh runs,
plain default matplotlib).

Produces, in figures/studies/:
  clutter_residual_vs_cadence.png
  clutter_moving_environment.png
  clutter_reference_snr.png
  clutter_ofdm_vs_preamble.png

Cells cache to fig_cache_clutter_core.json so an interrupted run
resumes; delete the cache for a fully fresh pass.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from clutter_sync_study import (
    matched_channel_drift_std,
    run_clutter_referenced,
)
import clutter_sync_ofdm
from clutter_sync_ofdm import calibrate_oneway_noise
from ota_sync import SDRSimulationConfig, run_two_way_simulation
from ota_sync.scheduled import run_scheduled_star

CACHE = Path(__file__).resolve().parent / "fig_cache_clutter_core.json"
FIGDIR = Path(__file__).resolve().parent / "figures" / "studies"
ITERATIONS = 60
SEEDS = [0, 1, 2]


def save(figure, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / f"{name}.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(path)


def _load() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return {}


def _store(cache: dict) -> None:
    CACHE.write_text(json.dumps(cache, indent=1))


def _cell(cache: dict, key: str, compute) -> list:
    if key not in cache:
        cache[key] = compute()
        _store(cache)
        print(f"  computed {key}: {cache[key]}")
    return cache[key]


def collect(cache: dict) -> dict:
    data: dict = {}

    # Baselines (paid airtime), seeds 0-2.
    def baselines():
        two, micro = [], []
        for seed in SEEDS:
            settings = SDRSimulationConfig(
                num_iterations=ITERATIONS, seed=seed, device="cpu"
            )
            t = run_two_way_simulation(settings)
            two.append(
                [1e3 * t.steady_state_phase_rms, t.airtime_fraction]
            )
            s = run_scheduled_star(
                settings, num_stations=2, policy="scheduled",
                multi_fidelity=True,
            )
            micro.append(
                [1e3 * s.station_steady_rms[0], s.airtime_used_fraction]
            )
        return [two, micro]

    data["baselines"] = _cell(cache, "baselines", baselines)

    # Residual vs anchor cadence, seeds 0-2.
    for cadence in (5, 10, 20, 40):
        def cadence_cell(cadence=cadence):
            rows = []
            for seed in SEEDS:
                settings = SDRSimulationConfig(
                    num_iterations=ITERATIONS, seed=seed, device="cpu"
                )
                result, _, piggy = run_clutter_referenced(
                    settings, cadence, 4
                )
                rows.append(
                    [1e3 * result.steady_state_phase_rms, piggy,
                     result.detection_rate]
                )
            return rows
        data[f"K{cadence}"] = _cell(cache, f"K{cadence}", cadence_cell)

    # Reference-strength stressor: K=20, seed 0, one-way SNR swept.
    for snr in (20, 10, 5, 0):
        def snr_cell(snr=snr):
            settings = SDRSimulationConfig(
                num_iterations=ITERATIONS, seed=0, device="cpu",
                snr_db=float(snr),
            )
            result, _, _ = run_clutter_referenced(settings, 20, 4)
            return [1e3 * result.steady_state_phase_rms,
                    result.detection_rate]
        data[f"snr{snr}"] = _cell(cache, f"snr{snr}", snr_cell)

    # Moving environment: seed 0, matched channel prior.
    for speed in (0.2, 0.5):
        for cadence in (5, 1):
            def move_cell(speed=speed, cadence=cadence):
                settings = SDRSimulationConfig(
                    num_iterations=ITERATIONS, seed=0, device="cpu",
                    channel_speed_mps=speed,
                )
                prior = matched_channel_drift_std(settings, speed)
                result, _, piggy = run_clutter_referenced(
                    settings, cadence, 4, channel_drift_std_rad=prior
                )
                return [1e3 * result.steady_state_phase_rms, piggy]
            data[f"move{speed}_K{cadence}"] = _cell(
                cache, f"move{speed}_K{cadence}", move_cell
            )

    # Static K=1 anchors for the moving figure's speed-0 point.
    def static_k1():
        settings = SDRSimulationConfig(
            num_iterations=ITERATIONS, seed=0, device="cpu"
        )
        result, _, piggy = run_clutter_referenced(settings, 1, 4)
        return [1e3 * result.steady_state_phase_rms, piggy]

    data["move0_K1"] = _cell(cache, "move0_K1", static_k1)

    # OFDM sensing burst vs dedicated ZC preamble, per-observation noise.
    def waveform_noise():
        out = []
        for mode in ("ofdm", "zc"):
            clutter_sync_ofdm._CALIBRATION_CACHE.clear()
            settings = SDRSimulationConfig(
                num_iterations=ITERATIONS, seed=0, device="cpu"
            )
            phase_var, freq_var, detect = calibrate_oneway_noise(
                settings, mode, torch.device("cpu")
            )
            out.append(
                [1e3 * math.sqrt(max(phase_var, 0.0)),
                 math.sqrt(max(freq_var, 0.0)) / (2.0 * math.pi), detect]
            )
        clutter_sync_ofdm._CALIBRATION_CACHE.clear()
        return out

    data["waveform_noise"] = _cell(cache, "waveform_noise", waveform_noise)
    return data


def mean_std(rows: list, index: int) -> tuple[float, float]:
    values = torch.tensor([r[index] for r in rows], dtype=torch.float64)
    return values.mean().item(), (
        values.std().item() if len(rows) > 1 else 0.0
    )


def fig_cadence(data: dict) -> None:
    cadences = [5, 10, 20, 40]
    means, stds = [], []
    for cadence in cadences:
        m, s = mean_std(data[f"K{cadence}"], 0)
        means.append(m)
        stds.append(s)
    air_low = 100.0 * data["K40"][0][1]
    air_high = 100.0 * data["K5"][0][1]
    two_rms, _ = mean_std(data["baselines"][0], 0)
    two_air, _ = mean_std(data["baselines"][0], 1)
    micro_rms, _ = mean_std(data["baselines"][1], 0)
    micro_air, _ = mean_std(data["baselines"][1], 1)

    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    axis.errorbar(
        cadences, means, yerr=stds, marker="o", capsize=3,
        label=f"piggyback ({air_low:.1f}-{air_high:.1f}% sync airtime)",
    )
    axis.axhline(
        two_rms, color="C1",
        label=f"two-way baseline ({100 * two_air:.1f}% sync airtime)",
    )
    axis.axhline(
        micro_rms, color="C2", linestyle="--",
        label=f"micro-pilot star ({100 * micro_air:.1f}% sync airtime)",
    )
    axis.set_xlabel("anchor cadence K (intervals between two-way anchors)")
    axis.set_ylabel("steady clock error (mrad RMS)")
    axis.set_xticks(cadences)
    axis.set_ylim(bottom=0)
    axis.set_title(
        f"Steady clock error vs anchor cadence "
        f"(N=2, {ITERATIONS} intervals, seeds 0-2)"
    )
    axis.legend(loc="center left", bbox_to_anchor=(0.02, 0.72))
    save(figure, "clutter_residual_vs_cadence")


def fig_moving(data: dict) -> None:
    speeds = [0.0, 0.2, 0.5]
    k5 = [
        mean_std(data["K5"], 0)[0],
        data["move0.2_K5"][0],
        data["move0.5_K5"][0],
    ]
    k1 = [
        data["move0_K1"][0],
        data["move0.2_K1"][0],
        data["move0.5_K1"][0],
    ]
    two_rms, _ = mean_std(data["baselines"][0], 0)
    two_air, _ = mean_std(data["baselines"][0], 1)
    k5_air = 100.0 * data["K5"][0][1]
    k1_air = 100.0 * data["move0_K1"][1]

    figure, axis = plt.subplots(figsize=(6.2, 4.4))
    axis.plot(
        speeds, k5, marker="o",
        label=f"piggyback, anchors every 5 ({k5_air:.1f}% airtime)",
    )
    axis.plot(
        speeds, k1, marker="s",
        label=f"piggyback, anchors every interval ({k1_air:.1f}% airtime)",
    )
    axis.axhline(
        two_rms, color="C2",
        label=f"two-way baseline, static ({100 * two_air:.1f}% airtime)",
    )
    axis.set_xlabel("environment motion (m/s)")
    axis.set_ylabel("steady clock error (mrad RMS)")
    axis.set_xticks(speeds)
    axis.set_ylim(bottom=0)
    axis.set_title(
        "Steady clock error vs environment motion (N=2, seed 0)"
    )
    axis.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=9
    )
    save(figure, "clutter_moving_environment")


def fig_snr(data: dict) -> None:
    snrs = [20, 10, 5, 0]
    rms = [data[f"snr{s}"][0] for s in snrs]
    detect = [100.0 * data[f"snr{s}"][1] for s in snrs]

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(6.2, 5.4), sharex=True
    )
    top.plot(snrs, rms, marker="o")
    top.set_ylabel("steady clock error (mrad RMS)")
    top.set_ylim(bottom=0)
    top.set_title(
        "Clock error and observation detection vs one-way reference "
        "SNR\n(K=20, N=2, seed 0)"
    )
    bottom.plot(snrs, detect, marker="o")
    bottom.set_ylabel("observations detected (%)")
    bottom.set_xlabel("one-way reference signal-to-noise ratio (dB)")
    bottom.set_ylim(-5, 105)
    top.invert_xaxis()
    save(figure, "clutter_reference_snr")


def fig_waveform(data: dict) -> None:
    (ofdm_phase, _, _), (zc_phase, _, _) = data["waveform_noise"]
    figure, axis = plt.subplots(figsize=(4.8, 4.0))
    axis.bar(
        [0, 1], [ofdm_phase, zc_phase], width=0.55,
        color=["C0", "C1"],
    )
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["OFDM sensing burst", "dedicated ZC preamble"])
    axis.set_ylabel("phase noise per observation (mrad)")
    axis.set_title(
        "Per-observation phase noise by pilot waveform (seed 0)"
    )
    save(figure, "clutter_ofdm_vs_preamble")


def main() -> None:
    cache = _load()
    data = collect(cache)
    fig_cadence(data)
    fig_moving(data)
    fig_snr(data)
    fig_waveform(data)


if __name__ == "__main__":
    main()
