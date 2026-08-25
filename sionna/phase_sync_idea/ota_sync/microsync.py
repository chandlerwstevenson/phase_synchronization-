"""Two-tier reciprocal synchronization: full frames plus micro-pilots.

The closed-loop residual of the plain two-way loop is dominated by terms
that depend only on correction cadence: the LO white-FM walk accumulated
over the dead time between frames, and frequency uncertainty propagated
over the correction staleness. Both shrink if phase is re-measured more
often -- and phase, unlike timing or wide-range CFO, can be measured from a
very short pilot once the loop is locked.

This module therefore runs the full two-way frame (detection, timing,
coarse/fine CFO, phase) once per synchronization interval, and inserts
short reciprocal phase-only micro-pilots (a cyclic-prefixed Zadoff-Chu
sequence, no training fields) at evenly spaced sub-intervals in between.
Micro-pilots ride on the tracking state: the receiver derotates with its
current frequency estimate and correlates at the expected arrival, so they
need no detection preamble. Corrections are issued every sub-interval. The
oscillators, channel, and all analog impairments are simulated at the same
fidelity as the main loop; micro-pilot captures pass through the identical
transmit/channel/receive chain.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
from sionna.phy import config as sionna_config

from .coherent import _pick_half_phase
from .core import (
    COMPLEX_DTYPE,
    REAL_DTYPE,
    Oscillator,
    PhaseFrequencyEKF,
    resolve_device,
    wrap_phase,
)
from .sdr import (
    SDRRadioLink,
    SDRSimulationConfig,
    SDRSynchronizer,
    SyncPreamble,
    _FlickerFrequencyNoise,
    _measurement_covariance,
    _quantize_correction,
    _zadoff_chu,
    make_sync_preamble,
)


@dataclass(frozen=True)
class MicroSyncResult:
    """Per-substep metrics from a two-tier reciprocal synchronization run."""

    true_phase: torch.Tensor
    physical_relative_frequency: torch.Tensor
    estimated_phase: torch.Tensor
    post_correction_phase: torch.Tensor
    post_correction_frequency: torch.Tensor
    coherent_gain: torch.Tensor
    detected: torch.Tensor
    is_full_frame: torch.Tensor
    correction_active: torch.Tensor
    calibrated: torch.Tensor
    airtime_fraction: float
    device: torch.device

    @property
    def detection_rate(self) -> float:
        return torch.mean(self.detected.to(torch.float64)).item()

    @property
    def steady_state_phase_rms(self) -> float:
        valid = self.detected & self.correction_active & self.calibrated
        if not torch.any(valid):
            return float("nan")
        return torch.sqrt(
            torch.mean(self.post_correction_phase[valid].square())
        ).item()

    @property
    def mean_coherent_gain(self) -> float:
        valid = self.detected & self.correction_active & self.calibrated
        if not torch.any(valid):
            return float("nan")
        return torch.mean(self.coherent_gain[valid]).item()

    @property
    def final_phase_error(self) -> float:
        return self.post_correction_phase[-1].item()

    @property
    def final_frequency_error_hz(self) -> float:
        return self.post_correction_frequency[-1].item() / (2.0 * math.pi)


def _make_micro_preamble(
    sequence_length: int, cp_length: int, device: torch.device
) -> SyncPreamble:
    """A phase-only pilot: one CP-protected Zadoff-Chu sequence."""

    root = 25
    while math.gcd(sequence_length, root) != 1:
        root += 1
    sequence = _zadoff_chu(sequence_length, root, device)
    cyclic_prefix = sequence[-cp_length:] if cp_length else sequence[:0]
    waveform = torch.cat((cyclic_prefix, sequence))
    return SyncPreamble(
        waveform=waveform,
        short_sequence=sequence[:1],
        long_sequence=sequence,
        short_length=0,
        long_block_length=waveform.numel(),
    )


def _estimate_micro_phase(
    samples: torch.Tensor,
    reference: torch.Tensor,
    expected_start: int,
    frequency: torch.Tensor,
    sample_period: float,
    search: int = 4,
    threshold: float = 0.2,
) -> tuple[bool, torch.Tensor]:
    """Tracking-mode phase estimate at a known schedule.

    The receiver derotates with its current frequency estimate and
    correlates against the known sequence within a small window around the
    expected arrival; no detection preamble is required.
    """

    samples = samples - torch.mean(samples)
    length = reference.numel()
    index = torch.arange(samples.numel(), dtype=REAL_DTYPE, device=samples.device)
    center = expected_start + (length - 1) / 2.0
    derotated = samples * torch.exp(
        -1j * frequency * (index - center) * sample_period
    )
    reference_energy = torch.sum(torch.abs(reference).square())

    best_metric = -1.0
    best_correlation = torch.zeros((), dtype=COMPLEX_DTYPE, device=samples.device)
    for offset in range(-search, search + 1):
        start = expected_start + offset
        if start < 0 or start + length > samples.numel():
            continue
        segment = derotated[start : start + length]
        correlation = torch.sum(torch.conj(reference) * segment)
        energy = torch.sum(torch.abs(segment).square())
        metric = (
            torch.abs(correlation)
            / torch.sqrt((energy * reference_energy).clamp_min(1e-15))
        ).item()
        if metric > best_metric:
            best_metric = metric
            best_correlation = correlation
    return best_metric >= threshold, torch.angle(best_correlation)


def _micro_measurement_covariance(
    settings: SDRSimulationConfig,
    sequence_length: int,
    cp_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Phase-only observation covariance for the reciprocal micro-pilot."""

    snr = 10.0 ** (settings.snr_db / 10.0)
    phase_variance = 1.0 / (2.0 * snr * sequence_length)
    # Intra-pilot LO walk averaged over the sequence, plus white PM.
    frame_length = cp_length + sequence_length
    phase_variance += settings.phase_noise_std_rad**2 * (
        cp_length + frame_length / 3.0
    )
    phase_variance += settings.phase_noise_white_pm_std_rad**2 / sequence_length
    # Two directions averaged in the half-difference.
    phase_variance *= 0.5
    # The pilot carries no usable frequency information.
    return torch.diag(
        torch.tensor(
            [phase_variance, phase_variance, 1e9], dtype=REAL_DTYPE, device=device
        )
    )


def run_micro_two_way_simulation(
    settings: SDRSimulationConfig = SDRSimulationConfig(),
    micro_pilots_per_interval: int = 4,
    micro_sequence_length: int = 255,
    micro_cp_length: int = 32,
) -> MicroSyncResult:
    """Run the two-tier reciprocal loop: full frames plus micro-pilots."""

    if micro_pilots_per_interval < 0:
        raise ValueError("micro_pilots_per_interval cannot be negative")
    substeps = micro_pilots_per_interval + 1
    dt = settings.sync_interval / substeps
    dt_samples = int(round(dt * settings.sample_rate))

    device = resolve_device(settings.device)
    torch.manual_seed(settings.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(settings.seed)
    sionna_config.seed = settings.seed
    generator = torch.Generator(device=device)
    generator.manual_seed(settings.seed + 1)

    frequency_process_std = 2.0 * math.pi * settings.frequency_process_std_hz
    substep_covariance = torch.diag(
        torch.tensor(
            [
                settings.phase_process_std_rad**2 / substeps,
                frequency_process_std**2 / substeps,
            ],
            dtype=REAL_DTYPE,
            device=device,
        )
    )
    node_a = Oscillator(
        settings.master_initial_phase,
        2.0 * math.pi * settings.master_initial_frequency_hz,
        dt,
        substep_covariance,
        device,
        generator,
    )
    node_b = Oscillator(
        settings.slave_initial_phase,
        2.0 * math.pi * settings.slave_initial_frequency_hz,
        dt,
        substep_covariance,
        device,
        generator,
    )

    full_preamble = make_sync_preamble(settings, device)
    link_ab = SDRRadioLink(
        settings, full_preamble, device, generator, captures_per_interval=substeps
    )
    link_ba = SDRRadioLink(
        settings,
        full_preamble,
        device,
        generator,
        mirror_of=link_ab,
        captures_per_interval=substeps,
    )
    micro_preamble = _make_micro_preamble(
        micro_sequence_length, micro_cp_length, device
    )
    # Micro-pilots are scheduled inside a locked loop: no timing jitter.
    micro_settings = replace(settings, timing_jitter_samples=0)
    micro_ab = SDRRadioLink(
        micro_settings,
        micro_preamble,
        device,
        generator,
        mirror_of=link_ab,
        captures_per_interval=substeps,
    )
    micro_ba = SDRRadioLink(
        micro_settings,
        micro_preamble,
        device,
        generator,
        mirror_of=link_ab,
        captures_per_interval=substeps,
    )
    synchronizer = SDRSynchronizer(settings, full_preamble)

    full_noise = 0.5 * _measurement_covariance(settings, full_preamble, device)
    micro_noise = _micro_measurement_covariance(
        settings, micro_sequence_length, micro_cp_length, device
    )
    white_fm_substep = settings.phase_noise_std_rad**2 * dt_samples
    flicker = _FlickerFrequencyNoise(
        settings.flicker_frequency_std_hz,
        dt,
        settings.num_iterations * settings.sync_interval,
        device,
        generator,
    )
    ekf = PhaseFrequencyEKF(
        dt,
        2.0 * substep_covariance
        + torch.diag(
            torch.tensor(
                [white_fm_substep, flicker.innovation_variance],
                dtype=REAL_DTYPE,
                device=device,
            )
        ),
        full_noise,
        device,
        initial_covariance=torch.diag(
            torch.tensor(
                [math.pi**2, (2.0 * math.pi * 50e3) ** 2],
                dtype=REAL_DTYPE,
                device=device,
            )
        ),
    )

    full_capture_samples = link_ab.input_length + link_ab.l_tot - 1
    micro_capture_samples = micro_ab.input_length + micro_ab.l_tot - 1
    airtime_fraction = (
        2.0
        * (full_capture_samples + micro_pilots_per_interval * micro_capture_samples)
        / (settings.sync_interval * settings.sample_rate)
    )
    micro_expected_start = micro_settings.capture_guard_samples - micro_ab.l_min

    pending: dict[int, torch.Tensor] = {}
    carried_lo_walk = torch.zeros((), dtype=REAL_DTYPE, device=device)
    flicker_previous = torch.zeros((), dtype=REAL_DTYPE, device=device)
    slave_frequency_corrections = torch.zeros((), dtype=REAL_DTYPE, device=device)
    correction_has_loaded = False
    acquired = False
    chain_bias = math.radians(settings.twoway_chain_asymmetry_deg)
    settled_corrections = 0
    pi_calibrated = False

    history: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "true_phase",
            "physical_relative_frequency",
            "estimated_phase",
            "post_correction_phase",
            "post_correction_frequency",
            "coherent_gain",
            "detected",
            "is_full_frame",
            "correction_active",
            "calibrated",
        )
    }

    total_substeps = settings.num_iterations * substeps
    for substep in range(total_substeps):
        iteration = substep // substeps
        is_full = substep % substeps == 0

        node_a.step()
        node_b.step()
        node_a.state[0] = wrap_phase(node_a.state[0] + carried_lo_walk)
        flicker_now = flicker.step()
        node_a.state[1] = node_a.state[1] + (flicker_now - flicker_previous)
        flicker_previous = flicker_now

        due = pending.pop(substep, None)
        if due is not None:
            node_b.apply_correction(due)
            slave_frequency_corrections = slave_frequency_corrections + due[1]
            correction_has_loaded = True

        # Two-way half-difference acquisition carries a global pi ambiguity.
        # A deployment resolves it once after lock with a coarse combining
        # check (one bit: constructive or destructive); modeled here as a
        # single flip of the node-B NCO if the settled pair combines
        # destructively.
        if correction_has_loaded and not pi_calibrated:
            settled_corrections += 1
            if settled_corrections >= 3:
                if torch.cos(node_a.state[0] - node_b.state[0]) < 0.0:
                    # The half-difference measurement is invariant under pi
                    # shifts, so the filter needs no reset: its near-zero
                    # state re-attaches to the true branch after the flip.
                    node_b.apply_correction(
                        torch.tensor(
                            [math.pi, 0.0], dtype=REAL_DTYPE, device=device
                        )
                    )
                pi_calibrated = True

        # Physical (correction-free) node-B oscillator: what the hardware
        # would do with no synchronization running at all.
        physical_b = node_b.state[1] - slave_frequency_corrections
        if settings.sample_clock_offset_ppm is not None:
            sfo_forward = settings.sample_clock_offset_ppm
        else:
            sfo_forward = float(
                (physical_b - node_a.state[1]).item()
                / (2.0 * math.pi * settings.carrier_frequency_hz)
                * 1e6
            )
        history["physical_relative_frequency"].append(
            (node_a.state[1] - physical_b).clone()
        )

        relative_state = node_a.state - node_b.state

        if is_full:
            capture_ab = link_ab.capture(node_a, node_b, iteration, sfo_forward)
            node_a.state[0] = wrap_phase(node_a.state[0] + capture_ab.lo_walk_end)
            capture_ba = link_ba.capture(node_b, node_a, iteration, -sfo_forward)
            node_b.state[0] = wrap_phase(node_b.state[0] + capture_ba.lo_walk_end)
            forward = synchronizer.estimate(capture_ab.samples)
            reverse = synchronizer.estimate(capture_ba.samples)
            detected = forward.detected and reverse.detected
            phase_f, phase_r = forward.phase, reverse.phase
            frequency_obs = (forward.frequency - reverse.frequency) / 2.0
            capture_len = full_capture_samples
            noise = full_noise
        else:
            capture_ab = micro_ab.capture(node_a, node_b, iteration, sfo_forward)
            node_a.state[0] = wrap_phase(node_a.state[0] + capture_ab.lo_walk_end)
            capture_ba = micro_ba.capture(node_b, node_a, iteration, -sfo_forward)
            node_b.state[0] = wrap_phase(node_b.state[0] + capture_ba.lo_walk_end)
            detected_f, phase_f = _estimate_micro_phase(
                capture_ab.samples,
                micro_preamble.long_sequence,
                micro_expected_start + micro_cp_length,
                ekf.state[1],
                settings.sample_period,
            )
            detected_r, phase_r = _estimate_micro_phase(
                capture_ba.samples,
                micro_preamble.long_sequence,
                micro_expected_start + micro_cp_length,
                -ekf.state[1],
                settings.sample_period,
            )
            detected = detected_f and detected_r and acquired
            frequency_obs = None
            capture_len = micro_capture_samples
            noise = micro_noise

        remainder = max(0, dt_samples - 2 * capture_len)
        if settings.phase_noise_std_rad > 0.0 and remainder > 0:
            carried_lo_walk = torch.randn(
                (), dtype=REAL_DTYPE, device=device, generator=generator
            ) * (settings.phase_noise_std_rad * math.sqrt(remainder))
        else:
            carried_lo_walk = torch.zeros((), dtype=REAL_DTYPE, device=device)

        ekf.predict()
        if due is not None:
            ekf.reset_after_correction(due)

        correction: torch.Tensor | None = None
        if detected:
            combined_half = wrap_phase(
                wrap_phase(phase_f - phase_r) / 2.0 + chain_bias
            )
            if not acquired and is_full:
                phase_measurement = _pick_half_phase(
                    combined_half, torch.zeros_like(combined_half)
                )
                ekf.state = torch.stack((phase_measurement, frequency_obs))
                ekf.covariance = torch.diag(
                    torch.stack((full_noise[0, 0], full_noise[2, 2]))
                )
                acquired = True
            elif acquired:
                phase_measurement = _pick_half_phase(
                    combined_half, wrap_phase(ekf.state[0])
                )
                observation_frequency = (
                    frequency_obs if frequency_obs is not None else ekf.state[1]
                )
                ekf.measurement_covariance = noise
                ekf.update(
                    torch.stack(
                        (
                            torch.cos(phase_measurement),
                            torch.sin(phase_measurement),
                            observation_frequency,
                        )
                    )
                )
            if acquired:
                predicted = ekf.transition @ ekf.state
                correction = _quantize_correction(predicted, settings)
                pending[substep + 1] = correction

        history["true_phase"].append(wrap_phase(relative_state[0]).clone())
        history["estimated_phase"].append(wrap_phase(ekf.state[0]).clone())
        history["detected"].append(
            torch.tensor(detected, dtype=torch.bool, device=device)
        )
        history["is_full_frame"].append(
            torch.tensor(is_full, dtype=torch.bool, device=device)
        )
        history["correction_active"].append(
            torch.tensor(correction_has_loaded, dtype=torch.bool, device=device)
        )
        history["calibrated"].append(
            torch.tensor(pi_calibrated, dtype=torch.bool, device=device)
        )
        residual_state = node_a.state - node_b.state
        residual_phase = wrap_phase(residual_state[0])
        history["post_correction_phase"].append(residual_phase.clone())
        history["post_correction_frequency"].append(residual_state[1].clone())
        history["coherent_gain"].append(torch.cos(residual_phase / 2.0).square())

    return MicroSyncResult(
        **{
            name: torch.stack(values).detach().cpu() for name, values in history.items()
        },
        airtime_fraction=airtime_fraction,
        device=device,
    )
