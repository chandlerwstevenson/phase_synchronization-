"""Two coherent-collaboration architectures built on the SDR sync machinery.

Closed-loop (CSI-aided) joint transmission: a user's channel estimates absorb
any static per-station phase bias, so the achievable coherent gain depends
only on the differential phase drift between CSI refreshes. That gain is
evaluated directly from a one-way synchronization run.

Open-loop coherence for passive detection: with no user feedback, the
inter-station channel phase must be removed before the station carriers are
truly aligned at the antennas. A reciprocal two-way exchange over the same
channel realization cancels it: the half-difference of the forward and
reverse phase measurements observes the pure oscillator offset.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from sionna.phy import config as sionna_config

from .core import (
    REAL_DTYPE,
    Oscillator,
    PhaseFrequencyEKF,
    resolve_device,
    wrap_phase,
    _unwrap_phase,
)
from .sdr import (
    SDRRadioLink,
    SDRSimulationConfig,
    SDRSimulationResult,
    SDRSynchronizer,
    _FlickerFrequencyNoise,
    _measurement_covariance,
    _quantize_correction,
    make_sync_preamble,
)


@dataclass(frozen=True)
class TwoWaySimulationResult:
    """Metrics from a reciprocal two-way synchronization run.

    Unlike the one-way loop, the oscillator offset is identifiable, so all
    phase metrics are referenced to the true relative oscillator state.
    """

    true_phase: torch.Tensor
    measured_phase: torch.Tensor
    estimated_phase: torch.Tensor
    true_frequency: torch.Tensor
    estimated_frequency: torch.Tensor
    phase_error: torch.Tensor
    frequency_error: torch.Tensor
    post_correction_phase: torch.Tensor
    post_correction_frequency: torch.Tensor
    coherent_gain: torch.Tensor
    detected: torch.Tensor
    correction_active: torch.Tensor
    device: torch.device

    @property
    def detection_rate(self) -> float:
        return torch.mean(self.detected.to(torch.float64)).item()

    @property
    def phase_rmse(self) -> float:
        valid = self.detected
        if not torch.any(valid):
            return float("nan")
        return torch.sqrt(torch.mean(self.phase_error[valid].square())).item()

    @property
    def steady_state_phase_rms(self) -> float:
        valid = self.detected & self.correction_active
        if not torch.any(valid):
            return float("nan")
        return torch.sqrt(
            torch.mean(self.post_correction_phase[valid].square())
        ).item()

    @property
    def mean_coherent_gain(self) -> float:
        valid = self.detected & self.correction_active
        if not torch.any(valid):
            return float("nan")
        return torch.mean(self.coherent_gain[valid]).item()

    @property
    def final_phase_error(self) -> float:
        return self.post_correction_phase[-1].item()

    @property
    def final_frequency_error_hz(self) -> float:
        return self.post_correction_frequency[-1].item() / (2.0 * math.pi)


def _pick_half_phase(
    combined_half: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    """Resolve the pi ambiguity of a half-difference phase.

    (a - b)/2 is only defined modulo pi. The candidate closer to the
    reference (the EKF prediction after acquisition) is chosen; the global
    pi ambiguity at acquisition is a real property of two-way carrier
    alignment and is resolved in hardware by a one-time combining check.
    """

    alternate = wrap_phase(combined_half + math.pi)
    keep = torch.abs(wrap_phase(combined_half - reference)) <= torch.abs(
        wrap_phase(alternate - reference)
    )
    return torch.where(keep, combined_half, alternate)


def run_two_way_simulation(
    settings: SDRSimulationConfig = SDRSimulationConfig(),
) -> TwoWaySimulationResult:
    """Run reciprocal two-way OTA synchronization for open-loop coherence."""

    device = resolve_device(settings.device)
    torch.manual_seed(settings.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(settings.seed)
    sionna_config.seed = settings.seed
    generator = torch.Generator(device=device)
    generator.manual_seed(settings.seed + 1)

    frequency_process_std = 2.0 * math.pi * settings.frequency_process_std_hz
    oscillator_covariance = torch.diag(
        torch.tensor(
            [settings.phase_process_std_rad**2, frequency_process_std**2],
            dtype=REAL_DTYPE,
            device=device,
        )
    )
    master = Oscillator(
        settings.master_initial_phase,
        2.0 * math.pi * settings.master_initial_frequency_hz,
        settings.sync_interval,
        oscillator_covariance,
        device,
        generator,
    )
    slave = Oscillator(
        settings.slave_initial_phase,
        2.0 * math.pi * settings.slave_initial_frequency_hz,
        settings.sync_interval,
        oscillator_covariance,
        device,
        generator,
    )
    preamble = make_sync_preamble(settings, device)
    link_forward = SDRRadioLink(settings, preamble, device, generator)
    link_reverse = SDRRadioLink(
        settings, preamble, device, generator, mirror_of=link_forward
    )
    synchronizer = SDRSynchronizer(settings, preamble)
    # Averaging two independent directional measurements halves the noise.
    measurement_noise = 0.5 * _measurement_covariance(settings, preamble, device)

    interval_samples = int(round(settings.sync_interval * settings.sample_rate))
    white_fm_phase_variance = settings.phase_noise_std_rad**2 * interval_samples
    flicker = _FlickerFrequencyNoise(
        settings.flicker_frequency_std_hz,
        settings.sync_interval,
        settings.num_iterations * settings.sync_interval,
        device,
        generator,
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

    chain_bias = math.radians(settings.twoway_chain_asymmetry_deg)
    capture_samples = link_forward.input_length + link_forward.l_tot - 1
    remainder_samples = max(0, interval_samples - 2 * capture_samples)
    pending_corrections: dict[int, torch.Tensor] = {}
    carried_lo_walk = torch.zeros((), dtype=REAL_DTYPE, device=device)
    flicker_previous = torch.zeros((), dtype=REAL_DTYPE, device=device)
    slave_frequency_corrections = torch.zeros((), dtype=REAL_DTYPE, device=device)
    correction_has_loaded = False
    acquired = False

    history: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "true_phase",
            "measured_phase",
            "estimated_phase",
            "true_frequency",
            "estimated_frequency",
            "phase_error",
            "frequency_error",
            "post_correction_phase",
            "post_correction_frequency",
            "coherent_gain",
            "detected",
            "correction_active",
        )
    }

    for iteration in range(settings.num_iterations):
        master.step()
        slave.step()
        master.state[0] = wrap_phase(master.state[0] + carried_lo_walk)
        flicker_now = flicker.step()
        master.state[1] = master.state[1] + (flicker_now - flicker_previous)
        flicker_previous = flicker_now

        due_correction = pending_corrections.pop(iteration, None)
        if due_correction is not None:
            slave.apply_correction(due_correction)
            slave_frequency_corrections = (
                slave_frequency_corrections + due_correction[1]
            )
            correction_has_loaded = True

        if settings.sample_clock_offset_ppm is not None:
            sfo_forward = settings.sample_clock_offset_ppm
        else:
            physical_slave_frequency = slave.state[1] - slave_frequency_corrections
            sfo_forward = float(
                (physical_slave_frequency - master.state[1]).item()
                / (2.0 * math.pi * settings.carrier_frequency_hz)
                * 1e6
            )

        relative_state = master.state - slave.state
        # Forward frame (master transmits); its intra-frame LO walk becomes
        # part of the true state before the reverse frame is exchanged, so
        # the two directions see one continuous noise process.
        capture_forward = link_forward.capture(master, slave, iteration, sfo_forward)
        master.state[0] = wrap_phase(master.state[0] + capture_forward.lo_walk_end)
        capture_reverse = link_reverse.capture(slave, master, iteration, -sfo_forward)
        slave.state[0] = wrap_phase(slave.state[0] + capture_reverse.lo_walk_end)

        if settings.phase_noise_std_rad > 0.0 and remainder_samples > 0:
            carried_lo_walk = torch.randn(
                (), dtype=REAL_DTYPE, device=device, generator=generator
            ) * (settings.phase_noise_std_rad * math.sqrt(remainder_samples))
        else:
            carried_lo_walk = torch.zeros((), dtype=REAL_DTYPE, device=device)

        forward = synchronizer.estimate(capture_forward.samples)
        reverse = synchronizer.estimate(capture_reverse.samples)
        detected = forward.detected and reverse.detected
        # Reciprocity: the channel phase is common to both directions and
        # cancels in the half-difference, leaving the oscillator offset.
        combined_half = wrap_phase(
            wrap_phase(forward.phase - reverse.phase) / 2.0 + chain_bias
        )
        combined_frequency = (forward.frequency - reverse.frequency) / 2.0

        ekf.predict()
        if due_correction is not None:
            ekf.reset_after_correction(due_correction)

        if detected:
            if not acquired:
                phase_measurement = _pick_half_phase(
                    combined_half, torch.zeros_like(combined_half)
                )
                ekf.state = torch.stack((phase_measurement, combined_frequency))
                ekf.covariance = torch.diag(
                    torch.stack((measurement_noise[0, 0], measurement_noise[2, 2]))
                )
                acquired = True
            else:
                phase_measurement = _pick_half_phase(
                    combined_half, wrap_phase(ekf.state[0])
                )
                ekf.update(
                    torch.stack(
                        (
                            torch.cos(phase_measurement),
                            torch.sin(phase_measurement),
                            combined_frequency,
                        )
                    )
                )
            predicted = ekf.state.clone()
            for _ in range(settings.correction_latency_intervals):
                predicted = ekf.transition @ predicted
            correction = _quantize_correction(predicted, settings)
        else:
            phase_measurement = combined_half
            correction = None

        true_phase = wrap_phase(relative_state[0])
        estimated_phase = wrap_phase(ekf.state[0])
        history["true_phase"].append(true_phase.clone())
        history["measured_phase"].append(phase_measurement.clone())
        history["estimated_phase"].append(estimated_phase.clone())
        history["true_frequency"].append(relative_state[1].clone())
        history["estimated_frequency"].append(ekf.state[1].clone())
        history["phase_error"].append(wrap_phase(true_phase - estimated_phase))
        history["frequency_error"].append(
            (relative_state[1] - ekf.state[1]).clone()
        )
        history["detected"].append(
            torch.tensor(detected, dtype=torch.bool, device=device)
        )

        if correction is not None:
            if settings.correction_latency_intervals == 0:
                slave.apply_correction(correction)
                ekf.reset_after_correction(correction)
                slave_frequency_corrections = (
                    slave_frequency_corrections + correction[1]
                )
                correction_has_loaded = True
            else:
                pending_corrections[
                    iteration + settings.correction_latency_intervals
                ] = correction

        history["correction_active"].append(
            torch.tensor(correction_has_loaded, dtype=torch.bool, device=device)
        )
        residual_state = master.state - slave.state
        residual_phase = wrap_phase(residual_state[0])
        history["post_correction_phase"].append(residual_phase.clone())
        history["post_correction_frequency"].append(residual_state[1].clone())
        # Two-element open-loop combining toward a passive target:
        # |1 + exp(j*residual)|^2 / 4 = cos^2(residual / 2).
        history["coherent_gain"].append(torch.cos(residual_phase / 2.0).square())

    return TwoWaySimulationResult(
        **{
            name: torch.stack(values).detach().cpu() for name, values in history.items()
        },
        device=device,
    )


def evaluate_csi_joint_transmission(
    result: SDRSimulationResult,
    refresh_intervals: tuple[int, ...] = (1, 2, 5, 10, 20),
    pilot_phase_std_rad: float = 0.01,
    seed: int = 0,
) -> dict[int, float]:
    """Mean two-station coherent JT gain at a user vs. CSI refresh cadence.

    Combining gain at the user depends only on the differential phase between
    the stations, so any static bias (including the one-way channel-phase
    bias) is absorbed by the user's channel estimate at each refresh. What
    remains is the drift of the differential phase since the last refresh
    plus the pilot estimation error, evaluated here from the closed-loop
    oscillator residual of a one-way synchronization run.
    """

    valid = result.detected & result.correction_active
    if not torch.any(valid):
        return {k: float("nan") for k in refresh_intervals}
    psi = _unwrap_phase(result.post_correction_oscillator_phase[valid])
    generator = torch.Generator()
    generator.manual_seed(seed)

    gains: dict[int, float] = {}
    for refresh in refresh_intervals:
        if refresh < 1:
            raise ValueError("refresh cadence must be at least one interval")
        epochs = torch.arange(psi.numel()) // refresh
        pilot_error = (
            torch.randn(int(epochs.max().item()) + 1, dtype=psi.dtype, generator=generator)
            * pilot_phase_std_rad
        )
        reference = psi[epochs * refresh] + pilot_error[epochs]
        gains[refresh] = torch.mean(
            torch.cos((psi - reference) / 2.0).square()
        ).item()
    return gains
