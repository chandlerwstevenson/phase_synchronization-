"""Hybrid one-way/two-way OTA phase calibration with channel prediction.

A joint 3-state EKF over [oscillator phase, oscillator frequency, channel
phase] separates the fast power-law clock noise from the slow channel-phase
dynamics. One-way pilots (full frame once per interval, phase-only
micro-pilots at sub-intervals) observe the sum of oscillator and channel
phase at high cadence; reciprocal two-way anchors every K intervals observe
the two components separately (half-difference and half-sum) and re-pin the
split. Reciprocity is paid for at the channel coherence timescale rather
than the oscillator timescale.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
from sionna.phy import config as sionna_config

from ota_sync.coherent import _pick_half_phase
from ota_sync.core import (
    REAL_DTYPE,
    Oscillator,
    resolve_device,
    wrap_phase,
)
from ota_sync.microsync import _estimate_micro_phase, _make_micro_preamble
from ota_sync.sdr import (
    SDRRadioLink,
    SDRSimulationConfig,
    SDRSynchronizer,
    _FlickerFrequencyNoise,
    _measurement_covariance,
    _quantize_correction,
    make_sync_preamble,
)


class _JointPhaseChannelEKF:
    """Iterated EKF over [theta, omega, phi_c].

    theta/omega evolve as the usual oscillator model; phi_c is a slow random
    walk. Observation models are supplied per update, since one-way pilots
    observe theta + phi_c while two-way anchors observe theta and phi_c
    separately.
    """

    def __init__(
        self,
        dt: float,
        process_covariance: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.device = device
        self.state = torch.zeros(3, dtype=REAL_DTYPE, device=device)
        self.covariance = torch.diag(
            torch.tensor(
                [math.pi**2, (2.0 * math.pi * 50e3) ** 2, math.pi**2],
                dtype=REAL_DTYPE,
                device=device,
            )
        )
        self.transition = torch.tensor(
            [[1.0, dt, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=REAL_DTYPE,
            device=device,
        )
        self.process_covariance = process_covariance

    def predict(self) -> None:
        self.state = self.transition @ self.state
        self.covariance = (
            self.transition @ self.covariance @ self.transition.T
            + self.process_covariance
        )

    def update(self, measurement, measurement_covariance, observe, jacobian) -> None:
        prior_state = self.state.clone()
        prior_covariance = self.covariance.clone()
        posterior = prior_state
        for _ in range(6):
            self.state = posterior
            jac = jacobian(self.state)
            innovation_covariance = (
                jac @ prior_covariance @ jac.T + measurement_covariance
            )
            gain = torch.linalg.solve(
                innovation_covariance, jac @ prior_covariance
            ).T
            innovation = (
                measurement
                - observe(self.state)
                - jac @ (prior_state - posterior)
            )
            next_state = prior_state + gain @ innovation
            if torch.max(torch.abs(next_state - posterior)) < 1e-10:
                posterior = next_state
                break
            posterior = next_state
        self.state = posterior
        identity = torch.eye(3, dtype=REAL_DTYPE, device=self.device)
        residual_map = identity - gain @ jac
        self.covariance = (
            residual_map @ prior_covariance @ residual_map.T
            + gain @ measurement_covariance @ gain.T
        )

    def reset_after_correction(self, correction: torch.Tensor) -> None:
        # NCO corrections act on the oscillator states, never the channel.
        self.state = self.state - torch.cat(
            (correction, torch.zeros(1, dtype=REAL_DTYPE, device=self.device))
        )


def _observe_sum_with_frequency(x: torch.Tensor) -> torch.Tensor:
    total = x[0] + x[2]
    return torch.stack((torch.cos(total), torch.sin(total), x[1]))


def _jacobian_sum_with_frequency(x: torch.Tensor) -> torch.Tensor:
    total = x[0] + x[2]
    s, c = torch.sin(total), torch.cos(total)
    zero = torch.zeros((), dtype=REAL_DTYPE, device=x.device)
    one = torch.ones((), dtype=REAL_DTYPE, device=x.device)
    return torch.stack(
        (
            torch.stack((-s, zero, -s)),
            torch.stack((c, zero, c)),
            torch.stack((zero, one, zero)),
        )
    )


def _observe_sum_only(x: torch.Tensor) -> torch.Tensor:
    total = x[0] + x[2]
    return torch.stack((torch.cos(total), torch.sin(total)))


def _jacobian_sum_only(x: torch.Tensor) -> torch.Tensor:
    total = x[0] + x[2]
    s, c = torch.sin(total), torch.cos(total)
    zero = torch.zeros((), dtype=REAL_DTYPE, device=x.device)
    return torch.stack(
        (torch.stack((-s, zero, -s)), torch.stack((c, zero, c)))
    )


def _observe_anchor(x: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            torch.cos(x[0]),
            torch.sin(x[0]),
            x[1],
            torch.cos(x[2]),
            torch.sin(x[2]),
        )
    )


def _jacobian_anchor(x: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros((), dtype=REAL_DTYPE, device=x.device)
    one = torch.ones((), dtype=REAL_DTYPE, device=x.device)
    return torch.stack(
        (
            torch.stack((-torch.sin(x[0]), zero, zero)),
            torch.stack((torch.cos(x[0]), zero, zero)),
            torch.stack((zero, one, zero)),
            torch.stack((zero, zero, -torch.sin(x[2]))),
            torch.stack((zero, zero, torch.cos(x[2]))),
        )
    )


@dataclass(frozen=True)
class HybridSyncResult:
    """Per-substep metrics from a hybrid calibration run."""

    true_phase: torch.Tensor
    estimated_phase: torch.Tensor
    estimated_channel_phase: torch.Tensor
    post_correction_phase: torch.Tensor
    post_correction_frequency: torch.Tensor
    coherent_gain: torch.Tensor
    detected: torch.Tensor
    is_anchor: torch.Tensor
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


def run_hybrid_simulation(
    settings: SDRSimulationConfig = SDRSimulationConfig(),
    micro_pilots_per_interval: int = 4,
    anchor_every_intervals: int = 5,
    micro_sequence_length: int = 255,
    micro_cp_length: int = 32,
    channel_drift_std_rad: float = 0.01,
) -> HybridSyncResult:
    """Run the hybrid one-way/two-way calibration loop.

    Every interval carries a one-way full frame (timing/CFO/phase) plus
    ``micro_pilots_per_interval`` one-way phase-only micro-pilots; every
    ``anchor_every_intervals``-th interval the full frame becomes a
    reciprocal two-way exchange that separates oscillator and channel
    phase. ``channel_drift_std_rad`` is the assumed channel-phase random
    walk per interval (the estimator's knob for channel coherence time).
    """

    if micro_pilots_per_interval < 0:
        raise ValueError("micro_pilots_per_interval cannot be negative")
    if anchor_every_intervals < 1:
        raise ValueError("anchor_every_intervals must be at least one")
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
    link_ab = SDRRadioLink(settings, full_preamble, device, generator)
    link_ba = SDRRadioLink(
        settings, full_preamble, device, generator, mirror_of=link_ab
    )
    micro_preamble = _make_micro_preamble(
        micro_sequence_length, micro_cp_length, device
    )
    micro_settings = replace(settings, timing_jitter_samples=0)
    micro_ab = SDRRadioLink(
        micro_settings,
        micro_preamble,
        device,
        generator,
        mirror_of=link_ab,
        captures_per_interval=substeps,
    )
    synchronizer = SDRSynchronizer(settings, full_preamble)

    # Observation covariances. One-way pilots use the full one-way variance;
    # the anchor's half-difference and half-sum each average two directions.
    oneway_noise = _measurement_covariance(settings, full_preamble, device)
    snr = 10.0 ** (settings.snr_db / 10.0)
    micro_phase_variance = 1.0 / (2.0 * snr * micro_sequence_length)
    micro_frame = micro_cp_length + micro_sequence_length
    micro_phase_variance += settings.phase_noise_std_rad**2 * (
        micro_cp_length + micro_frame / 3.0
    )
    micro_phase_variance += (
        settings.phase_noise_white_pm_std_rad**2 / micro_sequence_length
    )
    micro_noise = torch.diag(
        torch.tensor(
            [micro_phase_variance, micro_phase_variance],
            dtype=REAL_DTYPE,
            device=device,
        )
    )
    anchor_noise = torch.diag(
        torch.stack(
            (
                0.5 * oneway_noise[0, 0],
                0.5 * oneway_noise[1, 1],
                0.5 * oneway_noise[2, 2],
                0.5 * oneway_noise[0, 0],
                0.5 * oneway_noise[1, 1],
            )
        )
    )

    white_fm_substep = settings.phase_noise_std_rad**2 * dt_samples
    flicker = _FlickerFrequencyNoise(
        settings.flicker_frequency_std_hz,
        dt,
        settings.num_iterations * settings.sync_interval,
        device,
        generator,
    )
    process = torch.diag(
        torch.tensor(
            [
                2.0 * substep_covariance[0, 0].item() + white_fm_substep,
                2.0 * substep_covariance[1, 1].item()
                + flicker.innovation_variance,
                channel_drift_std_rad**2 / substeps,
            ],
            dtype=REAL_DTYPE,
            device=device,
        )
    )
    ekf = _JointPhaseChannelEKF(dt, process, device)

    full_capture_samples = link_ab.input_length + link_ab.l_tot - 1
    micro_capture_samples = micro_ab.input_length + micro_ab.l_tot - 1
    interval_samples = int(round(settings.sync_interval * settings.sample_rate))
    airtime_fraction = (
        full_capture_samples * (1.0 + 1.0 / anchor_every_intervals)
        + micro_pilots_per_interval * micro_capture_samples
    ) / interval_samples
    micro_expected_start = (
        micro_settings.capture_guard_samples - micro_ab.l_min + micro_cp_length
    )

    pending: dict[int, torch.Tensor] = {}
    carried_lo_walk = torch.zeros((), dtype=REAL_DTYPE, device=device)
    flicker_previous = torch.zeros((), dtype=REAL_DTYPE, device=device)
    slave_frequency_corrections = torch.zeros((), dtype=REAL_DTYPE, device=device)
    correction_has_loaded = False
    acquired = False
    settled_corrections = 0
    pi_calibrated = False
    chain_bias = math.radians(settings.twoway_chain_asymmetry_deg)

    history: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "true_phase",
            "estimated_phase",
            "estimated_channel_phase",
            "post_correction_phase",
            "post_correction_frequency",
            "coherent_gain",
            "detected",
            "is_anchor",
            "correction_active",
            "calibrated",
        )
    }

    total_substeps = settings.num_iterations * substeps
    for substep in range(total_substeps):
        iteration = substep // substeps
        at_frame_slot = substep % substeps == 0
        is_anchor = at_frame_slot and iteration % anchor_every_intervals == 0

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

        # One-time pi-ambiguity calibration (see run_two_way_simulation).
        # On a flip the channel-phase state shifts by -pi so the tracked sum
        # stays consistent with the one-way observations.
        if correction_has_loaded and not pi_calibrated:
            settled_corrections += 1
            if settled_corrections >= 3:
                if torch.cos(node_a.state[0] - node_b.state[0]) < 0.0:
                    node_b.apply_correction(
                        torch.tensor(
                            [math.pi, 0.0], dtype=REAL_DTYPE, device=device
                        )
                    )
                    ekf.state[2] = wrap_phase(ekf.state[2] - math.pi)
                pi_calibrated = True

        if settings.sample_clock_offset_ppm is not None:
            sfo_forward = settings.sample_clock_offset_ppm
        else:
            physical_b = node_b.state[1] - slave_frequency_corrections
            sfo_forward = float(
                (physical_b - node_a.state[1]).item()
                / (2.0 * math.pi * settings.carrier_frequency_hz)
                * 1e6
            )

        relative_state = node_a.state - node_b.state

        used_samples = 0
        if at_frame_slot:
            capture_ab = link_ab.capture(node_a, node_b, iteration, sfo_forward)
            node_a.state[0] = wrap_phase(node_a.state[0] + capture_ab.lo_walk_end)
            forward = synchronizer.estimate(capture_ab.samples)
            used_samples += full_capture_samples
            if is_anchor:
                capture_ba = link_ba.capture(
                    node_b, node_a, iteration, -sfo_forward
                )
                node_b.state[0] = wrap_phase(
                    node_b.state[0] + capture_ba.lo_walk_end
                )
                reverse = synchronizer.estimate(capture_ba.samples)
                used_samples += full_capture_samples
                detected = forward.detected and reverse.detected
            else:
                detected = forward.detected and acquired
        else:
            capture_ab = micro_ab.capture(node_a, node_b, iteration, sfo_forward)
            node_a.state[0] = wrap_phase(node_a.state[0] + capture_ab.lo_walk_end)
            detected_f, micro_phase = _estimate_micro_phase(
                capture_ab.samples,
                micro_preamble.long_sequence,
                micro_expected_start,
                ekf.state[1],
                settings.sample_period,
            )
            used_samples += micro_capture_samples
            detected = detected_f and acquired

        remainder = max(0, dt_samples - used_samples)
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
            if is_anchor:
                combined_half = wrap_phase(
                    wrap_phase(forward.phase - reverse.phase) / 2.0 + chain_bias
                )
                if not acquired:
                    theta_obs = _pick_half_phase(
                        combined_half, torch.zeros_like(combined_half)
                    )
                    frequency_obs = (forward.frequency - reverse.frequency) / 2.0
                    channel_obs = wrap_phase(forward.phase - theta_obs)
                    ekf.state = torch.stack(
                        (theta_obs, frequency_obs, channel_obs)
                    )
                    ekf.covariance = torch.diag(
                        torch.stack(
                            (
                                anchor_noise[0, 0],
                                anchor_noise[2, 2],
                                anchor_noise[3, 3],
                            )
                        )
                    )
                    acquired = True
                else:
                    theta_obs = _pick_half_phase(
                        combined_half, wrap_phase(ekf.state[0])
                    )
                    frequency_obs = (forward.frequency - reverse.frequency) / 2.0
                    channel_obs = wrap_phase(forward.phase - theta_obs)
                    ekf.update(
                        torch.stack(
                            (
                                torch.cos(theta_obs),
                                torch.sin(theta_obs),
                                frequency_obs,
                                torch.cos(channel_obs),
                                torch.sin(channel_obs),
                            )
                        ),
                        anchor_noise,
                        _observe_anchor,
                        _jacobian_anchor,
                    )
            elif at_frame_slot:
                ekf.update(
                    torch.stack(
                        (
                            torch.cos(forward.phase),
                            torch.sin(forward.phase),
                            forward.frequency,
                        )
                    ),
                    oneway_noise,
                    _observe_sum_with_frequency,
                    _jacobian_sum_with_frequency,
                )
            else:
                ekf.update(
                    torch.stack(
                        (torch.cos(micro_phase), torch.sin(micro_phase))
                    ),
                    micro_noise,
                    _observe_sum_only,
                    _jacobian_sum_only,
                )
            if acquired:
                predicted = ekf.transition @ ekf.state
                correction = _quantize_correction(predicted[:2], settings)
                pending[substep + 1] = correction

        history["true_phase"].append(wrap_phase(relative_state[0]).clone())
        history["estimated_phase"].append(wrap_phase(ekf.state[0]).clone())
        history["estimated_channel_phase"].append(
            wrap_phase(ekf.state[2]).clone()
        )
        history["detected"].append(
            torch.tensor(detected, dtype=torch.bool, device=device)
        )
        history["is_anchor"].append(
            torch.tensor(is_anchor, dtype=torch.bool, device=device)
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

    return HybridSyncResult(
        **{
            name: torch.stack(values).detach().cpu() for name, values in history.items()
        },
        airtime_fraction=airtime_fraction,
        device=device,
    )
