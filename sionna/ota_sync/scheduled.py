"""Smart allocation of phase synchronization (uncertainty-driven,
task-aware pilot scheduling) on an N-station star.

The idea under test: synchronization is a scarce resource — every
two-way exchange occupies the shared channel — and a uniform schedule
wastes most of it. Here a scheduler holds every link's Kalman
posterior and services a link ONLY when its predicted phase
uncertainty approaches that station's phase budget. Budgets are
task-aware: stations whose propagation legs matter most for the
current sensing task (e.g., the detection coverage edge) get tight
budgets; low-utility stations are allowed to coast.

Mechanics per sync interval:
  1. Every oscillator steps; due corrections load; every link's EKF
     runs its predict step (coasting links grow their covariance —
     exactly the physics the error-floor formula describes).
  2. The scheduler ranks links by predicted-phase-std / budget and
     services the worst offenders, up to the channel capacity
     (max_exchanges_per_interval) and only those above the trigger
     fraction of their budget. policy="uniform" services every link
     every interval instead (the conventional baseline).
  3. Serviced links run a REAL two-way exchange (both captures through
     the full physical layer, half-difference measurement, EKF update,
     forward-predicted quantized correction with latency, pi-branch
     calibration) — the same machinery as run_two_way_simulation, per
     link, with ONE shared reference oscillator.

Airtime is counted as exchanges actually performed; the freed channel
time is the dividend the sensing side would collect.

This star shares the reference oscillator across links (an upgrade on
the independent pairwise network simulation: the reference's noise is
common-mode here, as in reality).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from sionna.phy import config as sionna_config

from .coherent import _pick_half_phase
from .core import (
    REAL_DTYPE,
    Oscillator,
    PhaseFrequencyEKF,
    resolve_device,
    wrap_phase,
)
from .network import MAX_LINK_SNR_DB, place_stations
from .sdr import (
    SDRRadioLink,
    SDRSimulationConfig,
    SDRSynchronizer,
    _FlickerFrequencyNoise,
    _measurement_covariance,
    _quantize_correction,
    make_sync_preamble,
)


@dataclass(frozen=True)
class ScheduledSyncResult:
    """Per-interval metrics of a scheduled (or uniform) star."""

    positions: np.ndarray
    budgets_rad: list[float]
    residuals: torch.Tensor  # (stations-1, intervals) vs the reference
    serviced: torch.Tensor  # (stations-1, intervals) bool
    steady: torch.Tensor  # (intervals,) bool
    array_gain: torch.Tensor
    airtime_used_fraction: float
    airtime_uniform_fraction: float
    device: torch.device

    @property
    def num_stations(self) -> int:
        return self.residuals.shape[0] + 1

    @property
    def mean_array_gain(self) -> float:
        if not torch.any(self.steady):
            return float("nan")
        return torch.mean(self.array_gain[self.steady]).item()

    @property
    def station_steady_rms(self) -> list[float]:
        values = []
        for row in self.residuals:
            if torch.any(self.steady):
                values.append(
                    torch.sqrt(torch.mean(row[self.steady].square())).item()
                )
            else:
                values.append(float("nan"))
        return values

    @property
    def exchange_rate(self) -> float:
        """Mean exchanges per interval actually performed."""

        return torch.mean(
            torch.sum(self.serviced.to(torch.float64), dim=0)
        ).item()

    def residual_matrix(self) -> torch.Tensor:
        """(stations, steady-samples), row 0 the reference — for the
        detection pipeline."""

        steady = self.steady
        rows = [
            torch.zeros(int(steady.sum().item()), dtype=torch.float64)
        ]
        for row in self.residuals:
            rows.append(row[steady])
        return torch.stack(rows)


def run_scheduled_star(
    settings: SDRSimulationConfig = SDRSimulationConfig(),
    num_stations: int = 6,
    budgets_rad: list[float] | None = None,
    policy: str = "scheduled",
    trigger_fraction: float = 0.5,
    max_exchanges_per_interval: int | None = None,
    radius_m: float = 500.0,
    path_loss_exponent: float = 2.7,
    reference_distance_m: float = 500.0,
) -> ScheduledSyncResult:
    """Uncertainty-driven two-way sync scheduling on a star network."""

    if policy not in ("scheduled", "uniform"):
        raise ValueError("policy must be 'scheduled' or 'uniform'")
    device = resolve_device(settings.device)
    torch.manual_seed(settings.seed)
    sionna_config.seed = settings.seed
    generator = torch.Generator(device=device)
    generator.manual_seed(settings.seed + 1)

    positions = place_stations(num_stations, radius_m, settings.seed)
    if budgets_rad is None:
        budgets_rad = [0.314] * (num_stations - 1)
    if len(budgets_rad) != num_stations - 1:
        raise ValueError("need one budget per non-reference station")

    frequency_process_std = 2.0 * math.pi * settings.frequency_process_std_hz
    oscillator_covariance = torch.diag(
        torch.tensor(
            [settings.phase_process_std_rad**2, frequency_process_std**2],
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
        settings.flicker_frequency_std_hz,
        settings.sync_interval,
        settings.num_iterations * settings.sync_interval,
        device,
        generator,
    )
    flicker_previous = torch.zeros((), dtype=REAL_DTYPE, device=device)

    preamble = make_sync_preamble(settings, device)
    interval_samples = int(round(settings.sync_interval * settings.sample_rate))
    white_fm_phase_variance = settings.phase_noise_std_rad**2 * interval_samples

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
        link_settings = SDRSimulationConfig(
            **{
                **{
                    field: getattr(settings, field)
                    for field in settings.__dataclass_fields__
                },
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
        )
        slave = Oscillator(
            link_settings.slave_initial_phase,
            2.0 * math.pi * link_settings.slave_initial_frequency_hz,
            settings.sync_interval,
            oscillator_covariance,
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
        ekf = PhaseFrequencyEKF(
            settings.sync_interval,
            2.0 * oscillator_covariance
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
                "budget": float(budgets_rad[station - 1]),
                "pending": {},
                "corrections": torch.zeros(
                    (), dtype=REAL_DTYPE, device=device
                ),
                "acquired": False,
                "loaded": False,
                "settled": 0,
                "calibrated": False,
            }
        )

    capture_samples = links[0]["forward"].input_length + links[0]["forward"].l_tot - 1
    exchange_fraction = (
        2.0 * capture_samples / (settings.sync_interval * settings.sample_rate)
    )
    capacity = max_exchanges_per_interval
    if capacity is None:
        capacity = max(1, int(1.0 / exchange_fraction))
    chain_bias = math.radians(settings.twoway_chain_asymmetry_deg)

    residual_rows: list[list[torch.Tensor]] = [[] for _ in links]
    serviced_rows: list[list[bool]] = [[] for _ in links]
    steady_history: list[bool] = []
    gain_history: list[torch.Tensor] = []
    exchanges_done = 0

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
            # Pi-branch calibration, then a periodic 1-bit re-check:
            # a coasting station can drift across the pi branch of the
            # half-difference, so the one-shot check is not enough
            # (same lesson as the decentralized mesh).
            if link["loaded"]:
                link["settled"] += 1
                if link["settled"] >= 3:
                    if torch.cos(
                        reference.state[0] - link["slave"].state[0]
                    ) < -0.2 or (
                        not link["calibrated"]
                        and torch.cos(
                            reference.state[0] - link["slave"].state[0]
                        )
                        < 0.0
                    ):
                        link["slave"].apply_correction(
                            torch.tensor(
                                [math.pi, 0.0],
                                dtype=REAL_DTYPE,
                                device=device,
                            )
                        )
                    link["calibrated"] = True

        # ---- the scheduler --------------------------------------
        if policy == "uniform":
            chosen = links[: capacity]
        else:
            candidates = []
            for link in links:
                predicted_std = math.sqrt(
                    max(link["ekf"].covariance[0, 0].item(), 0.0)
                )
                urgency = predicted_std / link["budget"]
                settling = not (
                    link["calibrated"] and link["settled"] >= 6
                )
                if settling:
                    urgency = float("inf")
                if settling or urgency >= trigger_fraction:
                    candidates.append((urgency, link))
            candidates.sort(key=lambda item: -item[0])
            chosen = [link for _, link in candidates[:capacity]]

        for link in chosen:
            exchanges_done += 1
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
                serviced = True  # airtime spent even on a miss
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
                    torch.stack(
                        (link["noise"][0, 0], link["noise"][2, 2])
                    )
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
        if settings.phase_noise_std_rad > 0.0 and remainder > 0:
            walk_std = settings.phase_noise_std_rad * math.sqrt(remainder)
            for link in links:
                link["slave"].state[0] = wrap_phase(
                    link["slave"].state[0]
                    + torch.randn(
                        (), dtype=REAL_DTYPE, device=device,
                        generator=generator,
                    )
                    * walk_std
                )

        chosen_set = {id(link) for link in chosen}
        for row, serviced_row, link in zip(
            residual_rows, serviced_rows, links
        ):
            row.append(
                wrap_phase(
                    reference.state[0] - link["slave"].state[0]
                ).clone()
            )
            serviced_row.append(id(link) in chosen_set)
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

    airtime_used = (
        exchanges_done / settings.num_iterations
    ) * exchange_fraction
    return ScheduledSyncResult(
        positions=positions,
        budgets_rad=list(budgets_rad),
        residuals=torch.stack(
            [torch.stack(row).detach().cpu() for row in residual_rows]
        ),
        serviced=torch.tensor(serviced_rows, dtype=torch.bool),
        steady=torch.tensor(steady_history, dtype=torch.bool),
        array_gain=torch.stack(gain_history).detach().cpu(),
        airtime_used_fraction=airtime_used,
        airtime_uniform_fraction=(num_stations - 1) * exchange_fraction,
        device=device,
    )
