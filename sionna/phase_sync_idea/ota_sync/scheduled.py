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

Extensions (all opt-in; every default reproduces the original run
bit-for-bit, including the random draw order):

  policy="roundrobin"   uninformed baseline: links serviced in fixed
                        rotation at the same capacity (acquisition is
                        still forced first, otherwise the baseline
                        never locks and the comparison is vacuous).
  policy="oracle"       genie upper bound: ranks links by their TRUE
                        instantaneous residual over budget - state no
                        online policy can observe without spending the
                        airtime. Same trigger semantics as "scheduled".
  policy="whittle"      myopic Whittle-style index: ranks links by the
                        one-interval GROWTH in predicted budget-
                        violation probability 2Q(budget/sigma) if the
                        link coasts; services those whose next-interval
                        violation risk clears the trigger. The restless-
                        bandit view of pilot scheduling: you only
                        observe a link's phase by paying its airtime.
  oscillator_profiles   per-station oscillator classes (one name per
                        station, index 0 = the reference), e.g.
                        ["ocxo", "ocxo", "sdr", ...]: heterogeneous
                        fleets where coast times differ per station.
                        Slave initial CFO follows each profile's
                        datasheet accuracy. The link's radio chain uses
                        the slave-side profile for capture-time LO
                        noise; heterogeneity enters the oscillator
                        processes and each link's EKF process noise.
  budget_updates        {iteration: [budgets...]} re-targets budgets
                        mid-run - the hook for sensing-in-the-loop
                        scheduling (budgets follow the target
                        hypothesis as it moves).
  multi_fidelity=True   two service modalities: a full two-way frame,
                        or - once a link is settled and its frequency
                        posterior is tight - a cheap reciprocal
                        phase-only micro-pilot (the microsync
                        machinery), priced at its true sample cost.
                        Capacity is then accounted in full-exchange
                        units, so cheap pilots pack tighter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

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
from .microsync import (
    _estimate_micro_phase,
    _make_micro_preamble,
    _micro_measurement_covariance,
)
from .network import MAX_LINK_SNR_DB, place_stations
from .oscillators import (
    LEGACY_PROFILE_NAME,
    OSCILLATOR_PROFILES,
    resolve_oscillator_noise,
)
from .sdr import (
    SDRRadioLink,
    SDRSimulationConfig,
    SDRSynchronizer,
    _FlickerFrequencyNoise,
    _measurement_covariance,
    _quantize_correction,
    make_sync_preamble,
)

SCHEDULER_POLICIES = (
    "scheduled",
    "uniform",
    "roundrobin",
    "oracle",
    "whittle",
)


def _violation_probability(sigma: float, budget: float) -> float:
    """P(|phase error| > budget) for a zero-mean Gaussian posterior."""

    if sigma <= 0.0:
        return 0.0
    return math.erfc(budget / (sigma * math.sqrt(2.0)))


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
    # Extensions (defaults keep older constructions valid):
    serviced_micro: torch.Tensor | None = None  # bool, multi-fidelity only

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

    def residual_matrix(
        self, interval_slice: slice | None = None
    ) -> torch.Tensor:
        """(stations, steady-samples), row 0 the reference — for the
        detection pipeline. ``interval_slice`` restricts to a window of
        intervals (e.g. while the target was near one waypoint)."""

        steady = self.steady
        if interval_slice is not None:
            window = torch.zeros_like(steady)
            window[interval_slice] = True
            steady = steady & window
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
    oscillator_profiles: list[str] | None = None,
    budget_updates: dict[int, list[float]] | None = None,
    multi_fidelity: bool = False,
    micro_sequence_length: int = 255,
    micro_cp_length: int = 32,
) -> ScheduledSyncResult:
    """Uncertainty-driven two-way sync scheduling on a star network.

    See the module docstring for the extension knobs; defaults
    reproduce the original scheduled/uniform behavior exactly.
    """

    if policy not in SCHEDULER_POLICIES:
        raise ValueError(f"policy must be one of {SCHEDULER_POLICIES}")
    if oscillator_profiles is not None:
        if len(oscillator_profiles) != num_stations:
            raise ValueError("need one oscillator profile per station")
        for profile_name in oscillator_profiles:
            if (
                profile_name != LEGACY_PROFILE_NAME
                and profile_name not in OSCILLATOR_PROFILES
            ):
                raise ValueError(
                    f"unknown oscillator profile '{profile_name}'; choose "
                    f"from {(LEGACY_PROFILE_NAME, *OSCILLATOR_PROFILES)}"
                )
    if budget_updates is not None:
        for update_iteration, update in budget_updates.items():
            if len(update) != num_stations - 1:
                raise ValueError(
                    "budget_updates entries need one budget per "
                    f"non-reference station (iteration {update_iteration})"
                )
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

    # Per-station noise fields: identical to `settings` unless a
    # profile list is given (index 0 = the reference station).
    def _station_noise(station: int) -> tuple[dict[str, float], float | None]:
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
    micro_preamble = (
        _make_micro_preamble(micro_sequence_length, micro_cp_length, device)
        if multi_fidelity
        else None
    )

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
        # White-FM capture-time walk of the PAIR: reference-side plus
        # slave-side (identical to the original expression when the
        # two share one noise class).
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
        link = {
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
        if multi_fidelity:
            micro_settings = replace(link_settings, timing_jitter_samples=0)
            micro_forward = SDRRadioLink(
                micro_settings, micro_preamble, device, generator,
                mirror_of=forward,
            )
            micro_reverse = SDRRadioLink(
                micro_settings, micro_preamble, device, generator,
                mirror_of=forward,
            )
            link["micro_forward"] = micro_forward
            link["micro_reverse"] = micro_reverse
            link["micro_noise"] = _micro_measurement_covariance(
                link_settings, micro_sequence_length, micro_cp_length,
                device,
            )
            link["micro_expected_start"] = (
                micro_settings.capture_guard_samples - micro_forward.l_min
            )
        links.append(link)

    capture_samples = links[0]["forward"].input_length + links[0]["forward"].l_tot - 1
    exchange_fraction = (
        2.0 * capture_samples / (settings.sync_interval * settings.sample_rate)
    )
    capacity = max_exchanges_per_interval
    if capacity is None:
        capacity = max(1, int(1.0 / exchange_fraction))
    chain_bias = math.radians(settings.twoway_chain_asymmetry_deg)
    micro_cost = 1.0
    if multi_fidelity:
        micro_capture_samples = (
            links[0]["micro_forward"].input_length
            + links[0]["micro_forward"].l_tot
            - 1
        )
        micro_cost = micro_capture_samples / capture_samples
    whittle_trigger = _violation_probability(trigger_fraction, 1.0)

    def _settling(link) -> bool:
        return not (link["calibrated"] and link["settled"] >= 6)

    residual_rows: list[list[torch.Tensor]] = [[] for _ in links]
    serviced_rows: list[list[bool]] = [[] for _ in links]
    micro_rows: list[list[bool]] = [[] for _ in links]
    steady_history: list[bool] = []
    gain_history: list[torch.Tensor] = []
    exchanges_done = 0.0
    roundrobin_next = 0

    for iteration in range(settings.num_iterations):
        if budget_updates is not None and iteration in budget_updates:
            for link, new_budget in zip(links, budget_updates[iteration]):
                link["budget"] = float(new_budget)
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
            ordered = list(links)
        elif policy == "roundrobin":
            rotation = [
                links[(roundrobin_next + index) % len(links)]
                for index in range(len(links))
            ]
            ordered = [link for link in links if _settling(link)] + [
                link for link in rotation if not _settling(link)
            ]
            roundrobin_next = (roundrobin_next + capacity) % len(links)
        else:
            candidates = []
            for link in links:
                settling = _settling(link)
                if policy == "oracle":
                    # Genie: the TRUE residual, unobservable online.
                    urgency = abs(
                        wrap_phase(
                            reference.state[0] - link["slave"].state[0]
                        ).item()
                    ) / link["budget"]
                    eligible = urgency >= trigger_fraction
                elif policy == "whittle":
                    # One-step growth of the predicted budget-violation
                    # probability if this link coasts one more interval.
                    ekf = link["ekf"]
                    sigma_now = math.sqrt(
                        max(ekf.covariance[0, 0].item(), 0.0)
                    )
                    coasted = (
                        ekf.transition
                        @ ekf.covariance
                        @ ekf.transition.T
                        + ekf.process_covariance
                    )
                    sigma_next = math.sqrt(max(coasted[0, 0].item(), 0.0))
                    risk_next = _violation_probability(
                        sigma_next, link["budget"]
                    )
                    urgency = risk_next - _violation_probability(
                        sigma_now, link["budget"]
                    )
                    eligible = risk_next >= whittle_trigger
                else:  # "scheduled" (the original rule)
                    predicted_std = math.sqrt(
                        max(link["ekf"].covariance[0, 0].item(), 0.0)
                    )
                    urgency = predicted_std / link["budget"]
                    eligible = urgency >= trigger_fraction
                if settling:
                    urgency = float("inf")
                if settling or eligible:
                    candidates.append((urgency, link))
            candidates.sort(key=lambda item: -item[0])
            ordered = [link for _, link in candidates]

        if multi_fidelity:
            chosen = []
            capacity_left = float(capacity)
            for link in ordered:
                mode = "full"
                if not _settling(link) and link["acquired"]:
                    # Micro-eligible: the frequency posterior predicts
                    # less phase drift over the correction horizon than
                    # half the budget, so a phase-only pilot suffices.
                    frequency_std = math.sqrt(
                        max(link["ekf"].covariance[1, 1].item(), 0.0)
                    )
                    horizon = (
                        settings.correction_latency_intervals + 1
                    ) * settings.sync_interval
                    if frequency_std * horizon < 0.5 * link["budget"]:
                        mode = "micro"
                cost = micro_cost if mode == "micro" else 1.0
                if capacity_left + 1e-9 < cost:
                    continue
                capacity_left -= cost
                chosen.append((link, mode))
        else:
            chosen = [(link, "full") for link in ordered[:capacity]]

        for link, service_mode in chosen:
            exchanges_done += micro_cost if service_mode == "micro" else 1.0
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
            if service_mode == "micro":
                ekf = link["ekf"]
                capture_fwd = link["micro_forward"].capture(
                    reference, slave, iteration, sfo
                )
                reference.state[0] = wrap_phase(
                    reference.state[0] + capture_fwd.lo_walk_end
                )
                capture_rev = link["micro_reverse"].capture(
                    slave, reference, iteration, -sfo
                )
                slave.state[0] = wrap_phase(
                    slave.state[0] + capture_rev.lo_walk_end
                )
                micro_start = (
                    link["micro_expected_start"] + micro_cp_length
                )
                detected_f, phase_f = _estimate_micro_phase(
                    capture_fwd.samples,
                    micro_preamble.long_sequence,
                    micro_start,
                    ekf.state[1],
                    link["settings"].sample_period,
                )
                detected_r, phase_r = _estimate_micro_phase(
                    capture_rev.samples,
                    micro_preamble.long_sequence,
                    micro_start,
                    -ekf.state[1],
                    link["settings"].sample_period,
                )
                if not (detected_f and detected_r):
                    continue  # airtime spent even on a miss
                combined_half = wrap_phase(
                    wrap_phase(phase_f - phase_r) / 2.0 + chain_bias
                )
                measurement = _pick_half_phase(
                    combined_half, wrap_phase(ekf.state[0])
                )
                frequency_holdover = ekf.state[1].clone()
                ekf.measurement_covariance = link["micro_noise"]
                ekf.update(
                    torch.stack(
                        (
                            torch.cos(measurement),
                            torch.sin(measurement),
                            frequency_holdover,
                        )
                    )
                )
                ekf.measurement_covariance = link["noise"]
                predicted = ekf.state.clone()
                for _ in range(settings.correction_latency_intervals):
                    predicted = ekf.transition @ predicted
                link["pending"][
                    iteration + max(settings.correction_latency_intervals, 1)
                ] = _quantize_correction(predicted, settings)
                continue

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

        chosen_set = {id(link) for link, _ in chosen}
        micro_set = {
            id(link) for link, mode in chosen if mode == "micro"
        }
        for row, serviced_row, micro_row, link in zip(
            residual_rows, serviced_rows, micro_rows, links
        ):
            row.append(
                wrap_phase(
                    reference.state[0] - link["slave"].state[0]
                ).clone()
            )
            serviced_row.append(id(link) in chosen_set)
            micro_row.append(id(link) in micro_set)
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
        serviced_micro=(
            torch.tensor(micro_rows, dtype=torch.bool)
            if multi_fidelity
            else None
        ),
    )
