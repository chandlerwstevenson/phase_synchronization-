"""Measurement companion to resampling_law.py.

Frozen oscillators (all LO/oscillator noise off), static channel,
N=2: every variation in the two-way half-difference across exchanges
is measurement-chain error. The grids separate the four candidate
mechanisms (M1 integer insertion jitter, M2 fractional clock carry,
M3 noise-driven argmax toggling, M4 thermal) by toggling the jitter
draw and the clock offset independently, then validate the law's
predictions (saturation values, whiteness conditions) with zero
fitted constants.

Outputs: printed tables + resampling_law_results.json.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

from ota_sync.core import REAL_DTYPE, Oscillator, resolve_device, wrap_phase
from ota_sync.sdr import (
    SDRRadioLink,
    SDRSimulationConfig,
    SDRSynchronizer,
    make_sync_preamble,
)
from resampling_law import _quiet_settings, saturation_sigma

F_CARRIER = 915e6
RESULTS = Path(__file__).resolve().parent / "resampling_law_results.json"


class WalkJitterLink(SDRRadioLink):
    """Insertion jitter as a reflected +-1 random walk (correlated),
    instead of the stock i.i.d. uniform draw."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._walk_state = self.settings.timing_jitter_samples // 2

    def _random_start(self):
        step = int(
            torch.randint(
                0, 2, (), device=self.device, generator=self.generator
            ).item()
        ) * 2 - 1
        self._walk_state = min(
            max(self._walk_state + step, 0), self.settings.timing_jitter_samples
        )
        return self.settings.capture_guard_samples + self._walk_state


def run_exchanges(
    jitter: int,
    cfo_hz: float,
    exchanges: int = 160,
    seed: int = 0,
    snr_db: float = 20.0,
    tdl_model: str = "D",
    delay_spread_ns: float = 100.0,
    jitter_mode: str = "iid",
    lo_noise: bool = False,
):
    """Two-way exchanges with frozen oscillators; returns per-exchange
    half-difference errors (rad) plus per-direction diagnostics."""

    device = resolve_device("cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    base = SDRSimulationConfig(num_iterations=exchanges, seed=seed, device="cpu")
    overrides = dict(
        snr_db=snr_db,
        timing_jitter_samples=jitter,
        tdl_model=tdl_model,
        delay_spread_s=delay_spread_ns * 1e-9,
        num_iterations=exchanges,
    )
    if lo_noise:
        # Re-enable the intra-capture LO processes (white-FM walk and
        # white-PM) at their defaults; oscillator STATES stay frozen, so
        # this isolates the per-capture LO contribution to the
        # half-difference (the walk-end carry is deliberately dropped).
        defaults = SDRSimulationConfig()
        overrides.update(
            phase_noise_std_rad=defaults.phase_noise_std_rad,
            phase_noise_white_pm_std_rad=defaults.phase_noise_white_pm_std_rad,
        )
    quiet = _quiet_settings(base, **overrides)
    preamble = make_sync_preamble(quiet, device)
    link_cls = WalkJitterLink if jitter_mode == "walk" else SDRRadioLink
    forward = link_cls(quiet, preamble, device, generator)
    reverse = link_cls(quiet, preamble, device, generator, mirror_of=forward)
    synchronizer = SDRSynchronizer(quiet, preamble)
    zero_cov = torch.zeros(2, 2, dtype=REAL_DTYPE, device=device)
    master = Oscillator(0.3, 0.0, quiet.sync_interval, zero_cov, device, generator)
    slave = Oscillator(
        -0.5, 2.0 * math.pi * cfo_hz, quiet.sync_interval, zero_cov, device, generator
    )
    truth = wrap_phase(master.state[0] - slave.state[0]).item()
    sfo = cfo_hz / F_CARRIER * 1e6

    errors, misses = [], 0
    diag = {"align_f": [], "align_r": [], "carry_f": [], "carry_r": []}
    for iteration in range(exchanges):
        capture_f = forward.capture(master, slave, iteration, sfo)
        capture_r = reverse.capture(slave, master, iteration, -sfo)
        est_f = synchronizer.estimate(capture_f.samples)
        est_r = synchronizer.estimate(capture_r.samples)
        if not (est_f.detected and est_r.detected):
            misses += 1
            continue
        half = wrap_phase((est_f.phase - est_r.phase) / 2.0).item()
        # the half-difference determines truth modulo pi: wrap into
        # (-pi/2, pi/2] around zero error
        raw = half - truth
        error = math.remainder(raw, math.pi)
        errors.append(error)
        diag["align_f"].append(est_f.timing_index - capture_f.inserted_start)
        diag["align_r"].append(est_r.timing_index - capture_r.inserted_start)
        diag["carry_f"].append(forward._timing_carry)
        diag["carry_r"].append(reverse._timing_carry)
    return errors, misses, diag


def stats(errors):
    if len(errors) < 4:
        return dict(mean=float("nan"), std=float("nan"), lag1=float("nan"))
    tensor = torch.tensor(errors, dtype=torch.float64)
    mean = tensor.mean()
    centered = tensor - mean
    var = centered.square().mean()
    lags = {}
    for lag in (1, 2, 5, 10):
        if len(errors) > lag + 4:
            lags[f"lag{lag}"] = float(
                (centered[:-lag] * centered[lag:]).mean() / var
            )
    toggles = None
    return dict(
        mean=float(mean),
        std=float(var.sqrt()),
        **lags,
    )


def main() -> None:
    quick = "--quick" in sys.argv
    exchanges = 80 if quick else 160
    seeds = [0] if quick else [0, 1, 2]
    results = {}

    print("=== GRID 1: mechanism decomposition (TDL-D 100 ns, 20 dB, ZC) ===")
    print(f"{'jitter':>7} {'cfo Hz':>7} | {'std mrad':>9} {'mean mrad':>10} "
          f"{'lag1':>6} {'lag5':>6} {'argmax-toggle rate':>18}")
    grid1 = [
        (32, 1500.0), (0, 1500.0), (32, 0.0), (0, 0.0),
        (1, 1500.0), (2, 1500.0), (4, 1500.0), (8, 1500.0), (16, 1500.0),
    ]
    for jitter, cfo in grid1:
        all_errors, all_lag1, all_lag5, toggle_rates = [], [], [], []
        for seed in seeds:
            errors, misses, diag = run_exchanges(
                jitter, cfo, exchanges, seed
            )
            s = stats(errors)
            all_errors.append((s["mean"], s["std"]))
            all_lag1.append(s.get("lag1", float("nan")))
            all_lag5.append(s.get("lag5", float("nan")))
            align = torch.tensor(diag["align_f"], dtype=torch.float64)
            toggle_rates.append(
                float((align[1:] != align[:-1]).to(torch.float64).mean())
                if align.numel() > 1 else float("nan")
            )
        stds = [b for _, b in all_errors]
        means = [a for a, _ in all_errors]
        row = dict(
            jitter=jitter, cfo=cfo,
            std_mrad=1e3 * sum(stds) / len(stds),
            mean_mrad=1e3 * sum(means) / len(means),
            lag1=sum(all_lag1) / len(all_lag1),
            lag5=sum(all_lag5) / len(all_lag5),
            toggle=sum(toggle_rates) / len(toggle_rates),
        )
        results[f"g1_j{jitter}_c{int(cfo)}"] = row
        print(f"{jitter:>7} {int(cfo):>7} | {row['std_mrad']:>9.1f} "
              f"{row['mean_mrad']:>10.1f} {row['lag1']:>6.2f} "
              f"{row['lag5']:>6.2f} {row['toggle']:>18.2f}")

    one_way_sat, two_way_sat = saturation_sigma("D")
    print(f"\nlaw saturation (TDL-D): one-way {1e3*one_way_sat:.0f} mrad, "
          f"two-way half-difference {1e3*two_way_sat:.0f} mrad")

    print("\n=== GRID 2: channel scaling (jitter 32, cfo 1500) ===")
    print(f"{'model':>6} {'spread ns':>9} | {'std mrad':>9} {'mean mrad':>10} "
          f"{'law 2-way mrad':>14}")
    grid2 = [
        ("D", 100.0), ("E", 100.0), ("A", 100.0),
        ("D", 30.0), ("D", 300.0), ("D", 1000.0), ("D", 0.001),
    ]
    for model, spread in grid2:
        stds, means = [], []
        for seed in seeds:
            errors, misses, _ = run_exchanges(
                32, 1500.0, exchanges, seed, tdl_model=model,
                delay_spread_ns=spread,
            )
            s = stats(errors)
            stds.append(s["std"])
            means.append(s["mean"])
        _, law = saturation_sigma(model)
        row = dict(
            model=model, spread=spread,
            std_mrad=1e3 * sum(stds) / len(stds),
            mean_mrad=1e3 * sum(means) / len(means),
            law_mrad=1e3 * law if law == law else None,
        )
        results[f"g2_{model}_{spread}"] = row
        law_txt = f"{1e3*law:>14.0f}" if law == law else f"{'n/a (NLOS)':>14}"
        print(f"{model:>6} {spread:>9.0f} | {row['std_mrad']:>9.1f} "
              f"{row['mean_mrad']:>10.1f} {law_txt}")

    print("\n=== GRID 3: SNR and correlated jitter ===")
    print(f"{'config':>28} | {'std mrad':>9} {'lag1':>6} {'lag5':>6}")
    grid3 = [
        ("40dB j32 cfo1500", dict(jitter=32, cfo_hz=1500.0, snr_db=40.0)),
        ("40dB j0 cfo0 (thermal)", dict(jitter=0, cfo_hz=0.0, snr_db=40.0)),
        ("40dB j2 cfo1500", dict(jitter=2, cfo_hz=1500.0, snr_db=40.0)),
        ("walk-jitter 20dB cfo1500",
         dict(jitter=32, cfo_hz=1500.0, jitter_mode="walk")),
        ("LO-noise-on j32 cfo1500",
         dict(jitter=32, cfo_hz=1500.0, lo_noise=True)),
    ]
    for label, kwargs in grid3:
        stds, lag1s, lag5s = [], [], []
        for seed in seeds:
            errors, _, _ = run_exchanges(exchanges=exchanges, seed=seed, **kwargs)
            s = stats(errors)
            stds.append(s["std"])
            lag1s.append(s.get("lag1", float("nan")))
            lag5s.append(s.get("lag5", float("nan")))
        row = dict(
            std_mrad=1e3 * sum(stds) / len(stds),
            lag1=sum(lag1s) / len(lag1s),
            lag5=sum(lag5s) / len(lag5s),
        )
        results[f"g3_{label}"] = row
        print(f"{label:>28} | {row['std_mrad']:>9.1f} "
              f"{row['lag1']:>6.2f} {row['lag5']:>6.2f}")

    RESULTS.write_text(json.dumps(results, indent=1))
    print(f"\nsaved {RESULTS.name}")


if __name__ == "__main__":
    main()
