"""Measurement companion to ``pi_ambiguity_analysis.py``.

Three experiments on the pi ambiguity of two-way sync:
  (a) capture demonstration - with the branch check disabled, initial
      offsets beyond +-pi/2 lock the loop at anti-phase; the 1-bit
      check eliminates every capture,
  (b) check-period sweep - anti-phase dwell fraction vs check period,
      against the renewal prediction p_cross * (C+1)/2 / (1 + ...),
  (c) bit-error stressor - a noisy check bit inserts pi-jumps; dwell
      grows with the bit error rate as the same renewal formula with
      p_cross -> p_cross + eps.

The star loop below is a line-faithful copy of
ota_sync.scheduled.run_scheduled_star restricted to the uniform
policy, with three added knobs (branch_check_every, bit_error,
service_every). At the defaults (1, 0.0, 1) it reproduces the
original bit-for-bit (regression-tested in
tests/test_pi_ambiguity.py) - no pre-existing file is modified.

Usage:
    .venv/bin/python pi_ambiguity_study.py           # all three parts
    .venv/bin/python pi_ambiguity_study.py --part a  # one part
"""

from __future__ import annotations

import argparse
import math
import random as pyrandom
from dataclasses import dataclass

import numpy as np
import torch
from sionna.phy import config as sionna_config

from coast_law import dare_posterior, link_matrices, station_snr_db
from ota_sync.coherent import _pick_half_phase
from ota_sync.core import (
    REAL_DTYPE,
    Oscillator,
    PhaseFrequencyEKF,
    resolve_device,
    wrap_phase,
)
from ota_sync.network import MAX_LINK_SNR_DB, place_stations
from ota_sync.oscillators import (
    LEGACY_PROFILE_NAME,
    OSCILLATOR_PROFILES,
    resolve_oscillator_noise,
)
from ota_sync.sdr import (
    SDRRadioLink,
    SDRSimulationConfig,
    SDRSynchronizer,
    _FlickerFrequencyNoise,
    _measurement_covariance,
    _quantize_correction,
    make_sync_preamble,
)
from pi_ambiguity_analysis import anti_phase_fraction, p_cross


@dataclass(frozen=True)
class BranchRunResult:
    residuals: torch.Tensor  # (stations-1, intervals)
    steady: torch.Tensor
    array_gain: torch.Tensor
    flips: int  # pi-corrections applied by the check

    def anti_phase_dwell(self, settle: int = 10) -> float:
        """Fraction of post-settle intervals with |residual| > pi/2."""

        tail = self.residuals[:, settle:]
        return torch.mean(
            (torch.abs(tail) > math.pi / 2.0).to(torch.float64)
        ).item()

    def tail_capture(self, last: int = 10) -> bool:
        """True if the run ends locked on the anti-phase branch."""

        tail = self.residuals[0, -last:]
        mean = torch.angle(torch.mean(torch.exp(1j * tail.to(torch.complex128))))
        return bool(abs(mean.item()) > math.pi / 2.0)


def run_branch_star(
    settings: SDRSimulationConfig = SDRSimulationConfig(),
    num_stations: int = 2,
    branch_check_every: int | None = 1,
    bit_error: float = 0.0,
    service_every: int = 1,
    radius_m: float = 500.0,
    path_loss_exponent: float = 2.7,
    reference_distance_m: float = 500.0,
    oscillator_profiles: list[str] | None = None,
) -> BranchRunResult:
    """Uniform-policy star, line-faithful to run_scheduled_star, with
    the pi-branch check parameterized."""

    if oscillator_profiles is not None:
        if len(oscillator_profiles) != num_stations:
            raise ValueError("need one oscillator profile per station")
        for profile_name in oscillator_profiles:
            if (
                profile_name != LEGACY_PROFILE_NAME
                and profile_name not in OSCILLATOR_PROFILES
            ):
                raise ValueError(f"unknown profile '{profile_name}'")
    device = resolve_device(settings.device)
    torch.manual_seed(settings.seed)
    sionna_config.seed = settings.seed
    generator = torch.Generator(device=device)
    generator.manual_seed(settings.seed + 1)
    bit_rng = pyrandom.Random(settings.seed * 7919 + 13)

    positions = place_stations(num_stations, radius_m, settings.seed)

    def _station_noise(station: int):
        if oscillator_profiles is None:
            return {}, None
        return resolve_oscillator_noise(
            oscillator_profiles[station],
            settings.carrier_frequency_hz,
            settings.sample_rate,
            settings.sync_interval,
        )

    reference_noise, _ = _station_noise(0)
    reference_phase_walk_std = reference_noise.get(
        "phase_noise_std_rad", settings.phase_noise_std_rad
    )
    frequency_process_std = 2.0 * math.pi * reference_noise.get(
        "frequency_process_std_hz", settings.frequency_process_std_hz
    )
    oscillator_covariance = torch.diag(
        torch.tensor(
            [
                reference_noise.get(
                    "phase_process_std_rad", settings.phase_process_std_rad
                )
                ** 2,
                frequency_process_std**2,
            ],
            dtype=REAL_DTYPE,
            device=device,
        )
    )
    reference = Oscillator(
        settings.master_initial_phase,
        2.0 * math.pi * settings.master_initial_frequency_hz,
        settings.sync_interval,
        oscillator_covariance,
        device,
        generator,
    )
    flicker = _FlickerFrequencyNoise(
        reference_noise.get(
            "flicker_frequency_std_hz", settings.flicker_frequency_std_hz
        ),
        settings.sync_interval,
        settings.num_iterations * settings.sync_interval,
        device,
        generator,
    )
    flicker_previous = torch.zeros((), dtype=REAL_DTYPE, device=device)

    preamble = make_sync_preamble(settings, device)
    interval_samples = int(round(settings.sync_interval * settings.sample_rate))

    links = []
    for station in range(1, num_stations):
        distance = max(
            float(np.linalg.norm(positions[station] - positions[0])), 1.0
        )
        snr_db = min(
            settings.snr_db
            - 10.0
            * path_loss_exponent
            * math.log10(distance / reference_distance_m),
            MAX_LINK_SNR_DB,
        )
        station_noise, station_cfo = _station_noise(station)
        overrides: dict[str, float] = {
            "snr_db": snr_db,
            "slave_initial_phase": settings.slave_initial_phase
            * station
            / max(num_stations - 1, 1),
            "slave_initial_frequency_hz": (
                settings.slave_initial_frequency_hz
                * station
                / max(num_stations - 1, 1)
            ),
        }
        overrides.update(station_noise)
        if station_cfo is not None:
            overrides["slave_initial_frequency_hz"] = (
                station_cfo * station / max(num_stations - 1, 1)
            )
        link_settings = SDRSimulationConfig(
            **{
                **{
                    field: getattr(settings, field)
                    for field in settings.__dataclass_fields__
                },
                **overrides,
            }
        )
        slave_covariance = oscillator_covariance
        if oscillator_profiles is not None:
            slave_frequency_std = (
                2.0 * math.pi * link_settings.frequency_process_std_hz
            )
            slave_covariance = torch.diag(
                torch.tensor(
                    [
                        link_settings.phase_process_std_rad**2,
                        slave_frequency_std**2,
                    ],
                    dtype=REAL_DTYPE,
                    device=device,
                )
            )
        slave = Oscillator(
            link_settings.slave_initial_phase,
            2.0 * math.pi * link_settings.slave_initial_frequency_hz,
            settings.sync_interval,
            slave_covariance,
            device,
            generator,
        )
        forward = SDRRadioLink(link_settings, preamble, device, generator)
        reverse = SDRRadioLink(
            link_settings, preamble, device, generator, mirror_of=forward
        )
        measurement_noise = 0.5 * _measurement_covariance(
            link_settings, preamble, device
        )
        white_fm_phase_variance = (
            0.5
            * (
                reference_phase_walk_std**2
                + link_settings.phase_noise_std_rad**2
            )
            * interval_samples
        )
        ekf = PhaseFrequencyEKF(
            settings.sync_interval,
            oscillator_covariance
            + slave_covariance
            + torch.diag(
                torch.tensor(
                    [white_fm_phase_variance, flicker.innovation_variance],
                    dtype=REAL_DTYPE,
                    device=device,
                )
            ),
            measurement_noise,
            device,
            initial_covariance=torch.diag(
                torch.tensor(
                    [math.pi**2, (2.0 * math.pi * 50e3) ** 2],
                    dtype=REAL_DTYPE,
                    device=device,
                )
            ),
        )
        links.append(
            {
                "station": station,
                "settings": link_settings,
                "slave": slave,
                "forward": forward,
                "reverse": reverse,
                "synchronizer": SDRSynchronizer(link_settings, preamble),
                "noise": measurement_noise,
                "ekf": ekf,
                "pending": {},
                "corrections": torch.zeros((), dtype=REAL_DTYPE, device=device),
                "acquired": False,
                "loaded": False,
                "settled": 0,
                "calibrated": False,
            }
        )

    capture_samples = (
        links[0]["forward"].input_length + links[0]["forward"].l_tot - 1
    )
    chain_bias = math.radians(settings.twoway_chain_asymmetry_deg)

    residual_rows: list[list[torch.Tensor]] = [[] for _ in links]
    steady_history: list[bool] = []
    gain_history: list[torch.Tensor] = []
    flips = 0

    for iteration in range(settings.num_iterations):
        reference.step()
        flicker_now = flicker.step()
        reference.state[1] = reference.state[1] + (flicker_now - flicker_previous)
        flicker_previous = flicker_now
        for link in links:
            link["slave"].step()
            due = link["pending"].pop(iteration, None)
            if due is not None:
                link["slave"].apply_correction(due)
                link["corrections"] = link["corrections"] + due[1]
                link["loaded"] = True
            link["ekf"].predict()
            if due is not None:
                link["ekf"].reset_after_correction(due)
            # The parameterized pi-branch check. At the defaults
            # (every interval, zero bit error) this block is the
            # original run_scheduled_star check verbatim.
            if link["loaded"]:
                link["settled"] += 1
                if link["settled"] >= 3:
                    if branch_check_every is not None and (
                        not link["calibrated"]
                        or iteration % branch_check_every == 0
                    ):
                        cosine = torch.cos(
                            reference.state[0] - link["slave"].state[0]
                        )
                        flip = bool(cosine < -0.2) or (
                            not link["calibrated"] and bool(cosine < 0.0)
                        )
                        if bit_error > 0.0 and bit_rng.random() < bit_error:
                            flip = not flip
                        if flip:
                            link["slave"].apply_correction(
                                torch.tensor(
                                    [math.pi, 0.0],
                                    dtype=REAL_DTYPE,
                                    device=device,
                                )
                            )
                            flips += 1
                    link["calibrated"] = True

        if iteration % service_every == 0:
            chosen = [(link, "full") for link in links]
        else:
            chosen = []

        for link, _mode in chosen:
            slave = link["slave"]
            if settings.sample_clock_offset_ppm is not None:
                sfo = settings.sample_clock_offset_ppm
            else:
                physical = slave.state[1] - link["corrections"]
                sfo = float(
                    (physical - reference.state[1]).item()
                    / (2.0 * math.pi * settings.carrier_frequency_hz)
                    * 1e6
                )
            capture_fwd = link["forward"].capture(
                reference, slave, iteration, sfo
            )
            reference.state[0] = wrap_phase(
                reference.state[0] + capture_fwd.lo_walk_end
            )
            capture_rev = link["reverse"].capture(
                slave, reference, iteration, -sfo
            )
            slave.state[0] = wrap_phase(
                slave.state[0] + capture_rev.lo_walk_end
            )
            forward = link["synchronizer"].estimate(capture_fwd.samples)
            reverse = link["synchronizer"].estimate(capture_rev.samples)
            if not (forward.detected and reverse.detected):
                continue
            combined_frequency = (forward.frequency - reverse.frequency) / 2.0
            combined_half = wrap_phase(
                wrap_phase(forward.phase - reverse.phase) / 2.0
                + chain_bias
                - combined_frequency * settings.tdd_turnaround_s / 2.0
            )
            ekf = link["ekf"]
            if not link["acquired"]:
                measurement = _pick_half_phase(
                    combined_half, torch.zeros_like(combined_half)
                )
                ekf.state = torch.stack((measurement, combined_frequency))
                ekf.covariance = torch.diag(
                    torch.stack((link["noise"][0, 0], link["noise"][2, 2]))
                )
                link["acquired"] = True
            else:
                measurement = _pick_half_phase(
                    combined_half, wrap_phase(ekf.state[0])
                )
                ekf.update(
                    torch.stack(
                        (
                            torch.cos(measurement),
                            torch.sin(measurement),
                            combined_frequency,
                        )
                    )
                )
            predicted = ekf.state.clone()
            for _ in range(settings.correction_latency_intervals):
                predicted = ekf.transition @ predicted
            link["pending"][
                iteration + max(settings.correction_latency_intervals, 1)
            ] = _quantize_correction(predicted, settings)

        remainder = max(0, interval_samples - 2 * capture_samples)
        if remainder > 0:
            for link in links:
                phase_walk_std = link["settings"].phase_noise_std_rad
                if phase_walk_std <= 0.0:
                    continue
                walk_std = phase_walk_std * math.sqrt(remainder)
                link["slave"].state[0] = wrap_phase(
                    link["slave"].state[0]
                    + torch.randn(
                        (), dtype=REAL_DTYPE, device=device,
                        generator=generator,
                    )
                    * walk_std
                )

        for row, link in zip(residual_rows, links):
            row.append(
                wrap_phase(
                    reference.state[0] - link["slave"].state[0]
                ).clone()
            )
        steady_history.append(
            all(link["loaded"] and link["calibrated"] for link in links)
        )
        phases = torch.stack(
            [torch.zeros((), dtype=REAL_DTYPE, device=device)]
            + [
                wrap_phase(link["slave"].state[0] - reference.state[0])
                for link in links
            ]
        )
        phasors = torch.exp(1j * phases.to(torch.complex128))
        gain_history.append(
            (
                torch.abs(torch.sum(phasors)) ** 2 / num_stations**2
            ).real.to(torch.float64)
        )

    return BranchRunResult(
        residuals=torch.stack(
            [torch.stack(row).detach().cpu() for row in residual_rows]
        ),
        steady=torch.tensor(steady_history, dtype=torch.bool),
        array_gain=torch.stack(gain_history).detach().cpu(),
        flips=flips,
    )


def predicted_sigma(profile: str, settings: SDRSimulationConfig) -> float:
    """Ex-ante one-service-interval phase prediction-error std for the
    N=2 link: sqrt of the prior covariance (F P+ F' + Q)[0,0] at the
    every-interval steady state."""

    snr = station_snr_db(settings, 2, 1)
    matrices = link_matrices(
        settings, profile, snr, settings.num_iterations * settings.sync_interval
    )
    posterior = dare_posterior(matrices)
    prior = (
        matrices.transition @ posterior @ matrices.transition.T
        + matrices.process
    )
    return math.sqrt(max(float(prior[0, 0]), 0.0))


# ---------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------

def part_a() -> None:
    print("=== (a) capture demonstration ===")
    print("zero initial frequency offset, legacy oscillator, 40 intervals,")
    print("seeds 0-2; capture = run ends with |circular-mean tail| > pi/2")
    thetas = [x / 10.0 for x in range(-30, 31, 4)]
    print(f"{'theta0 rad':>11} {'no check: captured':>19} "
          f"{'with check: captured':>21}")
    boundary_ok = True
    for theta0 in thetas:
        captured = {None: 0, 1: 0}
        for check in (None, 1):
            for seed in (0, 1, 2):
                settings = SDRSimulationConfig(
                    num_iterations=40, seed=seed, device="cpu",
                    slave_initial_phase=theta0,
                    slave_initial_frequency_hz=0.0,
                )
                result = run_branch_star(
                    settings, branch_check_every=check
                )
                captured[check] += int(result.tail_capture())
        expected = abs(theta0) > math.pi / 2.0
        if (captured[None] >= 2) != expected and abs(
            abs(theta0) - math.pi / 2.0
        ) > 0.25:
            boundary_ok = False
        print(f"{theta0:>11.1f} {captured[None]:>14}/3 "
              f"{captured[1]:>17}/3")
    print(f"boundary matches |theta0| > pi/2 (outside +-0.25 rad "
          f"of the boundary): {boundary_ok}")

    print("\nrealistic acquisition (default 1.5 kHz initial frequency "
          "offset, theta0 = 1.2, seeds 0-19):")
    for check in (None, 1):
        captures = 0
        for seed in range(20):
            settings = SDRSimulationConfig(
                num_iterations=40, seed=seed, device="cpu"
            )
            captures += int(
                run_branch_star(settings, branch_check_every=check)
                .tail_capture()
            )
        label = "no check" if check is None else "1-bit check every interval"
        print(f"  {label:<28} captured {captures}/20")


def part_b() -> None:
    print("=== (b) check-period sweep ===")
    for profile, periods, seeds in (
        ("sdr", [1, 3, 6, 12, 24, None], range(15)),
        ("tcxo", [6, 24, None], range(5)),
    ):
        settings0 = SDRSimulationConfig(num_iterations=60, seed=0, device="cpu")
        sigma = predicted_sigma(profile, settings0)
        print(f"\n{profile}: ex-ante prediction-error sigma "
              f"{1e3 * sigma:.0f} mrad, p_cross {p_cross(sigma):.3e}")
        print(f"{'check every':>12} {'measured dwell %':>17} "
              f"{'predicted %':>12} {'flips/run':>10}")
        for period in periods:
            dwells, flips = [], []
            for seed in seeds:
                settings = SDRSimulationConfig(
                    num_iterations=60, seed=seed, device="cpu"
                )
                result = run_branch_star(
                    settings,
                    branch_check_every=period,
                    oscillator_profiles=[profile, profile],
                )
                dwells.append(result.anti_phase_dwell())
                flips.append(result.flips)
            predicted = anti_phase_fraction(sigma, period)
            label = "none" if period is None else str(period)
            print(f"{label:>12} {100 * float(np.mean(dwells)):>16.1f}% "
                  f"{100 * predicted:>11.1f}% "
                  f"{float(np.mean(flips)):>10.1f}")


def part_c() -> None:
    print("=== (c) bit-error stressor (sdr, check every 3) ===")
    settings0 = SDRSimulationConfig(num_iterations=60, seed=0, device="cpu")
    sigma = predicted_sigma("sdr", settings0)
    print(f"{'bit error':>10} {'measured dwell %':>17} {'predicted %':>12} "
          f"{'flips/run':>10}")
    for eps in (0.0, 0.05, 0.2):
        dwells, flips = [], []
        for seed in range(10):
            settings = SDRSimulationConfig(
                num_iterations=60, seed=seed, device="cpu"
            )
            result = run_branch_star(
                settings,
                branch_check_every=3,
                bit_error=eps,
                oscillator_profiles=["sdr", "sdr"],
            )
            dwells.append(result.anti_phase_dwell())
            flips.append(result.flips)
        predicted = anti_phase_fraction(sigma, 3, bit_error=eps)
        print(f"{eps:>10.2f} {100 * float(np.mean(dwells)):>16.1f}% "
              f"{100 * predicted:>11.1f}% {float(np.mean(flips)):>10.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="pi-ambiguity study")
    parser.add_argument("--part", choices=["a", "b", "c"], default=None)
    args = parser.parse_args()
    if args.part in (None, "a"):
        part_a()
    if args.part in (None, "b"):
        part_b()
    if args.part in (None, "c"):
        part_c()


if __name__ == "__main__":
    main()
