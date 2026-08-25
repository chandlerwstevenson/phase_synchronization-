"""OFDM-loop gap closure: the abstract attributes two results to the
OFDM-observation piggyback loop that were measured on the
preamble-observation variant. This study produces the OFDM loop's own
numbers for both.

Part 1 (headline): run_piggyback_star with waveform="ofdm" vs the "zc"
control at N=2, K in {5, 40, 160}, static default channel, seeds 0-2,
run length >= 4 anchor cycles per K (the interior_optimum_study
lesson: 60 intervals at K=40 is anchor-starved).

Part 2 (environments): the TDL letters (D/E/A/B/C) and the TDL-D
delay-spread sweep (30/100/300/1000 ns) at K=40, OFDM observations,
plus a same-machinery ZC control per cell so waveform is isolated
from implementation. Ray-traced scenes are composed through
environment_dependence_study.injected_channel, which already patches
clutter_sync_ofdm's SDRRadioLink.

Per cell the calibrated per-observation noise is recorded (the
calibration cache key omits channel parameters, so it is cleared
before every run - otherwise a stale calibration from another
environment would miscalibrate the filter's measurement covariance).

Usage:
    .venv/bin/python ofdm_environment_check.py --part headline
    .venv/bin/python ofdm_environment_check.py --part env
    .venv/bin/python ofdm_environment_check.py --part rt
    .venv/bin/python ofdm_environment_check.py --part report
"""

from __future__ import annotations

import argparse
import json
import math
import os

import torch

import clutter_sync_ofdm
from clutter_sync_ofdm import run_piggyback_star
from environment_dependence_study import (
    cir_to_frozen_taps,
    injected_channel,
    rt_station_pair_cir,
)
from ota_sync import SDRSimulationConfig

CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ofdm_environment_cache.json"
)
SEEDS = [0, 1, 2]
CADENCES = (5, 40, 160)
ENV_CADENCE = 40


def intervals_for(cadence: int) -> int:
    return max(60, 4 * cadence)


def _load() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as handle:
            return json.load(handle)
    return {}


def _save(cache: dict) -> None:
    with open(CACHE_PATH, "w") as handle:
        json.dump(cache, handle, indent=1)


def run_point(
    settings: SDRSimulationConfig,
    cadence: int,
    waveform: str,
    taps: torch.Tensor | None = None,
) -> dict:
    """One piggyback run; returns worst-station rms (mrad), anchor
    airtime fraction, observation detect rate, and the calibrated
    per-observation phase noise (mrad)."""

    clutter_sync_ofdm._CALIBRATION_CACHE.clear()
    with injected_channel(taps):
        result = run_piggyback_star(
            settings,
            num_stations=2,
            anchor_every_intervals=cadence,
            waveform=waveform,
        )
    clutter_sync_ofdm._CALIBRATION_CACHE.clear()
    return {
        "rms_mrad": result.worst_rms_mrad,
        "airtime": result.piggyback_airtime,
        "detect": result.detection_rate,
        "obs_mrad": 1e3 * math.sqrt(max(result.oneway_phase_var, 0.0)),
    }


def cell(
    key: str,
    cache: dict,
    cadence: int,
    waveform: str,
    taps: torch.Tensor | None = None,
    seeds: list[int] | None = None,
    **config,
) -> None:
    if key in cache:
        return
    rows = []
    for seed in seeds if seeds is not None else SEEDS:
        settings = SDRSimulationConfig(
            num_iterations=intervals_for(cadence),
            seed=seed,
            device="cpu",
            **config,
        )
        rows.append(run_point(settings, cadence, waveform, taps))
        print(
            f"    {key} seed {seed}: {rows[-1]['rms_mrad']:.1f} mrad "
            f"obs {rows[-1]['obs_mrad']:.1f}",
            flush=True,
        )
    cache[key] = rows
    _save(cache)


def _mean_std(rows: list[dict], field: str) -> tuple[float, float]:
    values = [row[field] for row in rows if row[field] == row[field]]
    mean = sum(values) / len(values)
    std = (
        math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        if len(values) > 1
        else 0.0
    )
    return mean, std


def summarize(key: str, cache: dict) -> str:
    rows = cache[key]
    rms, rms_std = _mean_std(rows, "rms_mrad")
    air, _ = _mean_std(rows, "airtime")
    obs, _ = _mean_std(rows, "obs_mrad")
    det, _ = _mean_std(rows, "detect")
    return (
        f"{rms:7.1f}±{rms_std:<5.1f} mrad @{100 * air:5.2f}% "
        f"obs {obs:5.1f} mrad det {100 * det:5.1f}%"
    )


TDL_LETTERS = ("D", "E", "A", "B", "C")
SPREADS_NS = (30, 100, 300, 1000)
RT_KINDS = ("tworay", "urban-los", "urban-nlos")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--part", choices=("headline", "env", "rt", "report"),
        default="report",
    )
    args = parser.parse_args()
    cache = _load()

    if args.part == "headline":
        for waveform in ("ofdm", "zc"):
            for cadence in CADENCES:
                cell(f"headline/{waveform}/K{cadence}", cache, cadence, waveform)

    elif args.part == "env":
        for letter in TDL_LETTERS:
            for waveform in ("ofdm", "zc"):
                cell(
                    f"tdl{letter}/{waveform}", cache, ENV_CADENCE, waveform,
                    tdl_model=letter,
                )
        for spread in SPREADS_NS:
            for waveform in ("ofdm", "zc"):
                cell(
                    f"spread{spread}/{waveform}", cache, ENV_CADENCE, waveform,
                    delay_spread_s=spread * 1e-9,
                )

    elif args.part == "rt":
        for kind in RT_KINDS:
            gains, delays, diag = rt_station_pair_cir(kind)
            base = SDRSimulationConfig(
                num_iterations=intervals_for(ENV_CADENCE), seed=0, device="cpu"
            )
            taps, dropped = cir_to_frozen_taps(gains, delays, base)
            print(
                f"  [{kind}] paths {diag['num_paths']} "
                f"LOS {'yes' if diag['has_los'] else 'NO'} "
                f"spread {diag['rms_spread_ns']:.1f} ns"
                + (f" ({dropped} dropped)" if dropped else "")
            )
            for waveform in ("ofdm", "zc"):
                cell(
                    f"rt-{kind}/{waveform}", cache, ENV_CADENCE, waveform,
                    taps=taps, seeds=SEEDS[:2],
                )

    print("\n=== OFDM headline (N=2, static default channel, "
          "intervals = max(60, 4K)) ===")
    for cadence in CADENCES:
        for waveform in ("ofdm", "zc"):
            key = f"headline/{waveform}/K{cadence}"
            if key in cache:
                print(f"  K={cadence:<4} {waveform:<5} {summarize(key, cache)}")

    print("\n=== OFDM x environment (K=40) ===")
    for group, keys in (
        ("TDL letters", [f"tdl{letter}" for letter in TDL_LETTERS]),
        ("delay spread", [f"spread{s}" for s in SPREADS_NS]),
        ("ray-traced", [f"rt-{k}" for k in RT_KINDS]),
    ):
        for base_key in keys:
            for waveform in ("ofdm", "zc"):
                key = f"{base_key}/{waveform}"
                if key in cache:
                    print(f"  {base_key:<12} {waveform:<5} "
                          f"{summarize(key, cache)}")


if __name__ == "__main__":
    main()
