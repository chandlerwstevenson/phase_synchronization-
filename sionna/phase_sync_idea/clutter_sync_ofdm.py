"""Clutter-referenced sync on the REAL sensing waveform, at N stations.

Closes the two realism gaps flagged by clutter_sync_study.py:

1. WAVEFORM. The piggybacked one-way observations there rode on the
   Zadoff-Chu sync preamble (4606 samples of dedicated training). Here
   they ride on the actual OFDM sensing burst the array radiates for
   detection (detection/waveform.py `_ofdm_burst`: random QPSK
   subcarriers + CP, 960 samples, a DIFFERENT random frame per burst,
   known to the receiver because the array transmits it). Phase and
   frequency come from matched-filtering the known frame - no genie
   equalization; the estimator sees the multipath composite exactly as
   the ZC path does. Its observation noise is CALIBRATED empirically
   (a receiver measures its own estimator variance during
   commissioning) and reported next to the ZC preamble's.

2. SCALE. A full N-station star: one shared reference whose sensing
   bursts every station overhears (one broadcast serves all links -
   the piggyback observations stay free at any N), per-station slave
   oscillators / TDL channels / path-loss SNR (same construction as
   ota_sync/scheduled.py), per-link 3-state EKF [theta, omega, phi_c]
   (hybrid_calibration machinery, imported), and STAGGERED two-way ZC
   anchors every K intervals per station - the only thing charged to
   sync airtime, both directions, conservative.

Fidelity notes (stated, not hidden):
  - Anchors remain dedicated ZC two-way exchanges: they are paid
    airtime, so using the sync preamble there is legitimate.
  - Both the reference and every slave receive an independent
    between-capture white-FM walk each substep. The two-node hybrid
    applied one walk to the pair; here each oscillator walks, which is
    the conservative (noisier) choice and is applied identically to
    the OFDM and ZC modes, so the waveform comparison is internal.
  - The OFDM link steps the shadowing process per capture with the
    correlation time rescaled by the substep count, so the shadowing
    dynamics match the interval-based links'.
  - The hybrid's known LOS-Doppler frequency-bias defect is respected:
    headline runs are static (channel_speed_mps = 0).

Usage:
    .venv/bin/python clutter_sync_ofdm.py                # full study
    .venv/bin/python clutter_sync_ofdm.py --quick        # smoke run
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace

import numpy as np
import torch

from detection.waveform import _ofdm_burst
from hybrid_calibration.hybrid import (
    _JointPhaseChannelEKF,
    _jacobian_anchor,
    _jacobian_sum_only,
    _jacobian_sum_with_frequency,
    _observe_anchor,
    _observe_sum_only,
    _observe_sum_with_frequency,
)
from ota_sync import SDRSimulationConfig
from ota_sync.coherent import _pick_half_phase
from ota_sync.core import REAL_DTYPE, COMPLEX_DTYPE, Oscillator, resolve_device, wrap_phase
from ota_sync.network import MAX_LINK_SNR_DB, place_stations
from ota_sync.scheduled import run_scheduled_star
from ota_sync.sdr import (
    SDRRadioLink,
    SDRSynchronizer,
    SyncPreamble,
    _FlickerFrequencyNoise,
    _measurement_covariance,
    _quantize_correction,
    make_sync_preamble,
)
from sionna.phy import config as sionna_config


OFDM_PULSE_LENGTH = 1023  # detection/waveform.py default -> 960 samples


def make_ofdm_frame_pool(
    count: int, device: torch.device, seed: int, pulse_length: int = OFDM_PULSE_LENGTH
) -> list[SyncPreamble]:
    """Distinct random-QPSK OFDM sensing bursts, wrapped so SDRRadioLink
    can transmit them. The short/long fields are placeholders - only the
    dedicated ZC synchronizer reads those, and it never sees these."""

    generator = torch.Generator().manual_seed(seed)
    pool = []
    for _ in range(count):
        burst = _ofdm_burst(pulse_length, generator).to(COMPLEX_DTYPE).to(device)
        pool.append(
            SyncPreamble(
                waveform=burst,
                short_sequence=burst[:16],
                long_sequence=burst,
                short_length=16,
                long_block_length=burst.numel(),
            )
        )
    return pool


@dataclass(frozen=True)
class OneWayEstimate:
    detected: bool
    phase: torch.Tensor
    frequency: torch.Tensor
    metric: float


class OFDMOneWayEstimator:
    """Matched-filter phase/frequency estimator for a known OFDM burst.

    The receiver transmitted (or was told) the frame, so it correlates
    against the exact waveform - the same operation the detection
    pipeline's matched filter performs. Timing by normalized
    cross-correlation, frequency by split-half correlation phase slope
    around the EKF's predicted CFO, phase at the burst center after
    derotation (the same phase convention as SDRSynchronizer).
    """

    def __init__(self, settings: SDRSimulationConfig, detection_threshold: float | None = None):
        self.settings = settings
        self.threshold = (
            settings.detection_threshold
            if detection_threshold is None
            else detection_threshold
        )

    def estimate(
        self, samples: torch.Tensor, waveform: torch.Tensor, predicted_omega: torch.Tensor
    ) -> OneWayEstimate:
        ts = self.settings.sample_period
        samples = samples - torch.mean(samples)
        length = waveform.numel()
        index = torch.arange(samples.numel(), dtype=REAL_DTYPE, device=samples.device)
        derotated = samples * torch.exp(-1j * predicted_omega * index * ts)

        if derotated.numel() < length:
            zero = torch.zeros((), dtype=REAL_DTYPE, device=samples.device)
            return OneWayEstimate(False, zero, predicted_omega.clone(), 0.0)
        windows = derotated.unfold(0, length, 1)
        correlations = torch.sum(windows * torch.conj(waveform), dim=-1)
        window_energy = torch.sum(torch.abs(windows).square(), dim=-1)
        reference_energy = torch.sum(torch.abs(waveform).square())
        normalized = torch.abs(correlations) / torch.sqrt(
            (window_energy * reference_energy).clamp_min(1e-15)
        )
        timing = int(torch.argmax(normalized).item())
        metric = normalized[timing].item()

        # Phase and frequency are measured on the RAW segment with all
        # derotations CENTERED on the segment, so no constant phase bias
        # is introduced - the same convention as SDRSynchronizer, whose
        # transmit/receive carriers are referenced to the frame center.
        segment = samples[timing : timing + length]
        relative = torch.arange(length, dtype=REAL_DTYPE, device=samples.device)
        centered = relative - (length - 1) / 2.0
        predicted_removed = segment * torch.exp(
            -1j * predicted_omega * centered * ts
        )
        half = length // 2
        first = torch.sum(
            torch.conj(waveform[:half]) * predicted_removed[:half]
        )
        second = torch.sum(
            torch.conj(waveform[half:]) * predicted_removed[half:]
        )
        # Phase slope between the two half-frame correlations; their
        # centers are half a frame apart.
        residual_omega = torch.angle(second * torch.conj(first)) / (half * ts)
        frequency = predicted_omega + residual_omega

        corrected = segment * torch.exp(-1j * frequency * centered * ts)
        phase = torch.angle(torch.sum(torch.conj(waveform) * corrected))
        return OneWayEstimate(metric >= self.threshold, phase, frequency, metric)


_CALIBRATION_CACHE: dict[tuple, tuple[float, float, float]] = {}


def calibrate_oneway_noise(
    settings: SDRSimulationConfig,
    waveform_mode: str,
    device: torch.device,
    captures: int = 160,
    pool_seed: int = 1234,
) -> tuple[float, float, float]:
    """Measured per-observation (phase var, frequency var, detect rate)
    of the one-way estimator, all RF/RX impairments active, oscillators
    and channel frozen so the observable is a constant. This is what a
    receiver measures during commissioning - no analytic shortcut, and
    the multipath/timing-jitter resampling noise is inside it. Cached
    per (waveform, SNR, noise profile) - the commissioning measurement
    is done once per link class, not once per run."""

    cache_key = (
        waveform_mode,
        round(settings.snr_db, 3),
        settings.phase_noise_std_rad,
        settings.phase_noise_white_pm_std_rad,
        captures,
    )
    if cache_key in _CALIBRATION_CACHE:
        return _CALIBRATION_CACHE[cache_key]

    frozen = replace(
        settings,
        phase_process_std_rad=0.0,
        frequency_process_std_hz=0.0,
        flicker_frequency_std_hz=0.0,
        slave_initial_phase=0.7,
        slave_initial_frequency_hz=0.0,
        shadowing_std_db=0.0,
        channel_speed_mps=0.0,
        num_iterations=max(settings.num_iterations, captures),
    )
    torch.manual_seed(frozen.seed + 77)
    sionna_config.seed = frozen.seed + 77
    generator = torch.Generator(device=device)
    generator.manual_seed(frozen.seed + 78)
    master = _frozen_oscillator(frozen.master_initial_phase, device)
    slave = _frozen_oscillator(frozen.slave_initial_phase, device)

    if waveform_mode == "ofdm":
        pool = make_ofdm_frame_pool(8, device, pool_seed)
        link = SDRRadioLink(frozen, pool[0], device, generator)
        estimator = OFDMOneWayEstimator(frozen)
    else:
        preamble = make_sync_preamble(frozen, device)
        link = SDRRadioLink(frozen, preamble, device, generator)
        synchronizer = SDRSynchronizer(frozen, preamble)

    zero_omega = torch.zeros((), dtype=REAL_DTYPE, device=device)
    phases, frequencies, detects = [], [], 0
    for capture_index in range(captures):
        if waveform_mode == "ofdm":
            link.preamble = pool[capture_index % len(pool)]
        capture = link.capture(master, slave, capture_index % frozen.num_iterations, 0.0)
        if waveform_mode == "ofdm":
            estimate = estimator.estimate(
                capture.samples, link.preamble.waveform, zero_omega
            )
            detected, phase, frequency = (
                estimate.detected, estimate.phase, estimate.frequency,
            )
        else:
            measurement = synchronizer.estimate(capture.samples)
            detected, phase, frequency = (
                measurement.detected, measurement.phase, measurement.frequency,
            )
        if detected:
            detects += 1
            phases.append(phase)
            frequencies.append(frequency)
    if not phases:
        result = (float("nan"), float("nan"), 0.0)
        _CALIBRATION_CACHE[cache_key] = result
        return result
    phase_tensor = torch.stack(phases)
    mean_angle = torch.angle(torch.sum(torch.exp(1j * phase_tensor.to(torch.complex128))))
    phase_variance = torch.mean(
        wrap_phase(phase_tensor - mean_angle).square()
    ).item()
    frequency_tensor = torch.stack(frequencies)
    frequency_variance = torch.var(frequency_tensor).item()
    result = (phase_variance, frequency_variance, detects / captures)
    _CALIBRATION_CACHE[cache_key] = result
    return result


def _frozen_oscillator(phase: float, device: torch.device) -> Oscillator:
    covariance = torch.zeros(2, 2, dtype=REAL_DTYPE, device=device)
    return Oscillator(phase, 0.0, 1.0, covariance, device, None)


@dataclass(frozen=True)
class PiggybackStarResult:
    """Per-substep metrics of the N-station piggyback star."""

    station_residuals: torch.Tensor  # (stations-1, substeps)
    station_valid: torch.Tensor  # (stations-1, substeps) bool
    array_gain: torch.Tensor  # (substeps,)
    all_valid: torch.Tensor  # (substeps,) bool
    detection_rate: float
    piggyback_airtime: float
    paid_airtime_if_dedicated: float
    oneway_phase_var: float
    oneway_frequency_var: float

    @property
    def station_rms_mrad(self) -> list[float]:
        values = []
        for row, valid in zip(self.station_residuals, self.station_valid):
            if torch.any(valid):
                values.append(
                    1e3 * torch.sqrt(torch.mean(row[valid].square())).item()
                )
            else:
                values.append(float("nan"))
        return values

    @property
    def worst_rms_mrad(self) -> float:
        return max(v for v in self.station_rms_mrad if v == v)

    @property
    def mean_array_gain(self) -> float:
        if not torch.any(self.all_valid):
            return float("nan")
        return torch.mean(self.array_gain[self.all_valid]).item()


def run_piggyback_star(
    settings: SDRSimulationConfig,
    num_stations: int = 2,
    anchor_every_intervals: int = 5,
    obs_per_interval: int = 5,
    waveform: str = "ofdm",
    radius_m: float = 500.0,
    path_loss_exponent: float = 2.7,
    reference_distance_m: float = 500.0,
    ofdm_pool_size: int = 16,
    calibration_captures: int = 160,
) -> PiggybackStarResult:
    """N-station star where every substep carries a FREE one-way
    observation (the reference's sensing burst, OFDM or ZC-control) and
    each station runs a charged two-way ZC anchor every K intervals,
    staggered across stations."""

    if waveform not in ("ofdm", "zc"):
        raise ValueError("waveform must be 'ofdm' or 'zc'")
    if num_stations < 2:
        raise ValueError("need at least two stations")
    if obs_per_interval < 1:
        raise ValueError("obs_per_interval must be at least one")

    substeps = obs_per_interval
    dt = settings.sync_interval / substeps
    dt_samples = int(round(dt * settings.sample_rate))
    device = resolve_device(settings.device)
    torch.manual_seed(settings.seed)
    sionna_config.seed = settings.seed
    generator = torch.Generator(device=device)
    generator.manual_seed(settings.seed + 1)

    positions = place_stations(num_stations, radius_m, settings.seed)
    zc_preamble = make_sync_preamble(settings, device)
    ofdm_pool = (
        make_ofdm_frame_pool(ofdm_pool_size, device, settings.seed + 500)
        if waveform == "ofdm"
        else None
    )

    substep_covariance = torch.diag(
        torch.tensor(
            [
                settings.phase_process_std_rad**2 / substeps,
                (2.0 * math.pi * settings.frequency_process_std_hz) ** 2 / substeps,
            ],
            dtype=REAL_DTYPE,
            device=device,
        )
    )
    reference = Oscillator(
        settings.master_initial_phase,
        2.0 * math.pi * settings.master_initial_frequency_hz,
        dt,
        substep_covariance,
        device,
        generator,
    )
    flicker = _FlickerFrequencyNoise(
        settings.flicker_frequency_std_hz,
        dt,
        settings.num_iterations * settings.sync_interval,
        device,
        generator,
    )
    flicker_previous = torch.zeros((), dtype=REAL_DTYPE, device=device)

    links = []
    for station in range(1, num_stations):
        distance = max(
            float(np.linalg.norm(positions[station] - positions[0])), 1.0
        )
        snr_db = min(
            settings.snr_db
            - 10.0 * path_loss_exponent * math.log10(distance / reference_distance_m),
            MAX_LINK_SNR_DB,
        )
        link_settings = replace(
            settings,
            snr_db=snr_db,
            slave_initial_phase=settings.slave_initial_phase
            * station / max(num_stations - 1, 1),
            slave_initial_frequency_hz=settings.slave_initial_frequency_hz
            * station / max(num_stations - 1, 1),
        )
        slave = Oscillator(
            link_settings.slave_initial_phase,
            2.0 * math.pi * link_settings.slave_initial_frequency_hz,
            dt,
            substep_covariance,
            device,
            generator,
        )
        # The one-way link owns the TDL realization; the anchor links
        # mirror its channel (reciprocity). Its shadowing correlation is
        # rescaled so per-capture stepping matches the interval physics.
        oneway_settings = replace(
            link_settings,
            shadowing_correlation_s=link_settings.shadowing_correlation_s * substeps,
        )
        if waveform == "ofdm":
            oneway = SDRRadioLink(
                oneway_settings, ofdm_pool[0], device, generator,
                captures_per_interval=substeps,
            )
        else:
            oneway = SDRRadioLink(
                oneway_settings, zc_preamble, device, generator,
                captures_per_interval=substeps,
            )
        anchor_forward = SDRRadioLink(
            link_settings, zc_preamble, device, generator, mirror_of=oneway
        )
        anchor_reverse = SDRRadioLink(
            link_settings, zc_preamble, device, generator, mirror_of=oneway
        )

        zc_noise = _measurement_covariance(link_settings, zc_preamble, device)
        phase_var, freq_var, _ = calibrate_oneway_noise(
            link_settings, waveform, device, captures=calibration_captures
        )
        oneway_noise3 = torch.diag(
            torch.tensor(
                [phase_var, phase_var, freq_var], dtype=REAL_DTYPE, device=device
            )
        )
        oneway_noise2 = oneway_noise3[:2, :2].clone()
        anchor_noise = torch.diag(
            torch.stack(
                (
                    0.5 * zc_noise[0, 0],
                    0.5 * zc_noise[1, 1],
                    0.5 * zc_noise[2, 2],
                    0.5 * zc_noise[0, 0],
                    0.5 * zc_noise[1, 1],
                )
            )
        )

        white_fm_substep = link_settings.phase_noise_std_rad**2 * dt_samples
        process = torch.diag(
            torch.tensor(
                [
                    2.0 * substep_covariance[0, 0].item() + white_fm_substep,
                    2.0 * substep_covariance[1, 1].item()
                    + flicker.innovation_variance,
                    0.01**2 / substeps,  # static-channel drift prior
                ],
                dtype=REAL_DTYPE,
                device=device,
            )
        )
        links.append(
            {
                "station": station,
                "settings": link_settings,
                "slave": slave,
                "oneway": oneway,
                "anchor_forward": anchor_forward,
                "anchor_reverse": anchor_reverse,
                "synchronizer": SDRSynchronizer(link_settings, zc_preamble),
                "estimator": OFDMOneWayEstimator(link_settings)
                if waveform == "ofdm"
                else None,
                "ekf": _JointPhaseChannelEKF(dt, process, device),
                "oneway_noise3": oneway_noise3,
                "oneway_noise2": oneway_noise2,
                "anchor_noise": anchor_noise,
                "oneway_phase_var": phase_var,
                "oneway_freq_var": freq_var,
                "pending": {},
                "corrections": torch.zeros((), dtype=REAL_DTYPE, device=device),
                "loaded": False,
                "acquired": False,
                "settled": 0,
                "calibrated": False,
                "capture_count": 0,
            }
        )

    zc_capture_samples = (
        links[0]["anchor_forward"].input_length
        + links[0]["anchor_forward"].l_tot - 1
    )
    interval_samples = int(round(settings.sync_interval * settings.sample_rate))
    piggyback_airtime = (
        (num_stations - 1)
        * 2.0
        * zc_capture_samples
        / (anchor_every_intervals * interval_samples)
    )
    oneway_capture_samples = (
        links[0]["oneway"].input_length + links[0]["oneway"].l_tot - 1
    )
    paid_airtime = piggyback_airtime + (
        (num_stations - 1) * substeps * oneway_capture_samples / interval_samples
    )

    residual_rows = [[] for _ in links]
    valid_rows = [[] for _ in links]
    gain_history, all_valid_history = [], []
    detected_count, observation_count = 0, 0

    total_substeps = settings.num_iterations * substeps
    for substep in range(total_substeps):
        iteration = substep // substeps
        at_frame_slot = substep % substeps == 0

        reference.step()
        flicker_now = flicker.step()
        reference.state[1] = reference.state[1] + (flicker_now - flicker_previous)
        flicker_previous = flicker_now
        for link in links:
            link["slave"].step()
            due = link["pending"].pop(substep, None)
            if due is not None:
                link["slave"].apply_correction(due)
                link["corrections"] = link["corrections"] + due[1]
                link["loaded"] = True
                link["ekf"].reset_after_correction(due)
            # Periodic 1-bit pi-branch check (ota_sync/scheduled.py
            # convention). The one-shot hybrid check is insufficient
            # here: at low observation rates the loop can settle into
            # the anti-phase fixed point AFTER an early check passes -
            # the same lesson the repo learned twice (scheduler
            # coasting, decentralized mesh), re-encountered at
            # obs_per_interval <= 2 during this study.
            if link["loaded"]:
                link["settled"] += 1
                if link["settled"] >= 3:
                    misaligned = torch.cos(
                        reference.state[0] - link["slave"].state[0]
                    ) < (-0.2 if link["calibrated"] else 0.0)
                    if misaligned:
                        link["slave"].apply_correction(
                            torch.tensor(
                                [math.pi, 0.0], dtype=REAL_DTYPE, device=device
                            )
                        )
                        link["ekf"].state[2] = wrap_phase(
                            link["ekf"].state[2] - math.pi
                        )
                    link["calibrated"] = True

        for link in links:
            slave = link["slave"]
            ekf = link["ekf"]
            is_anchor = (
                at_frame_slot
                and (iteration - link["station"]) % anchor_every_intervals == 0
            )
            physical = slave.state[1] - link["corrections"]
            sfo = float(
                (physical - reference.state[1]).item()
                / (2.0 * math.pi * settings.carrier_frequency_hz)
                * 1e6
            )

            ekf.predict()
            observation_count += 1
            if is_anchor:
                fwd_capture = link["anchor_forward"].capture(
                    reference, slave, iteration, sfo
                )
                reference.state[0] = wrap_phase(
                    reference.state[0] + fwd_capture.lo_walk_end
                )
                rev_capture = link["anchor_reverse"].capture(
                    slave, reference, iteration, -sfo
                )
                slave.state[0] = wrap_phase(
                    slave.state[0] + rev_capture.lo_walk_end
                )
                forward = link["synchronizer"].estimate(fwd_capture.samples)
                reverse = link["synchronizer"].estimate(rev_capture.samples)
                detected = forward.detected and reverse.detected
                if detected:
                    detected_count += 1
                    combined_half = wrap_phase(
                        wrap_phase(forward.phase - reverse.phase) / 2.0
                        + math.radians(settings.twoway_chain_asymmetry_deg)
                    )
                    frequency_obs = (forward.frequency - reverse.frequency) / 2.0
                    if not link["acquired"]:
                        theta_obs = _pick_half_phase(
                            combined_half, torch.zeros_like(combined_half)
                        )
                        channel_obs = wrap_phase(forward.phase - theta_obs)
                        ekf.state = torch.stack(
                            (theta_obs, frequency_obs, channel_obs)
                        )
                        ekf.covariance = torch.diag(
                            torch.stack(
                                (
                                    link["anchor_noise"][0, 0],
                                    link["anchor_noise"][2, 2],
                                    link["anchor_noise"][3, 3],
                                )
                            )
                        )
                        link["acquired"] = True
                    else:
                        theta_obs = _pick_half_phase(
                            combined_half, wrap_phase(ekf.state[0])
                        )
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
                            link["anchor_noise"],
                            _observe_anchor,
                            _jacobian_anchor,
                        )
            else:
                # FREE observation: the reference's sensing burst.
                oneway = link["oneway"]
                if waveform == "ofdm":
                    oneway.preamble = ofdm_pool[
                        link["capture_count"] % len(ofdm_pool)
                    ]
                link["capture_count"] += 1
                capture = oneway.capture(reference, slave, iteration, sfo)
                reference.state[0] = wrap_phase(
                    reference.state[0] + capture.lo_walk_end
                )
                if waveform == "ofdm":
                    estimate = link["estimator"].estimate(
                        capture.samples, oneway.preamble.waveform, ekf.state[1]
                    )
                    detected = estimate.detected and link["acquired"]
                    obs_phase, obs_frequency = estimate.phase, estimate.frequency
                else:
                    measurement = link["synchronizer"].estimate(capture.samples)
                    detected = measurement.detected and link["acquired"]
                    obs_phase, obs_frequency = (
                        measurement.phase, measurement.frequency,
                    )
                if detected:
                    detected_count += 1
                    if at_frame_slot:
                        ekf.update(
                            torch.stack(
                                (
                                    torch.cos(obs_phase),
                                    torch.sin(obs_phase),
                                    obs_frequency,
                                )
                            ),
                            link["oneway_noise3"],
                            _observe_sum_with_frequency,
                            _jacobian_sum_with_frequency,
                        )
                    else:
                        ekf.update(
                            torch.stack(
                                (torch.cos(obs_phase), torch.sin(obs_phase))
                            ),
                            link["oneway_noise2"],
                            _observe_sum_only,
                            _jacobian_sum_only,
                        )
            if detected and link["acquired"]:
                predicted = ekf.transition @ ekf.state
                link["pending"][substep + 1] = _quantize_correction(
                    predicted[:2], settings
                )

        # Between-capture white-FM walk for every oscillator.
        remainder = max(0, dt_samples - oneway_capture_samples)
        if settings.phase_noise_std_rad > 0.0 and remainder > 0:
            walk_std = settings.phase_noise_std_rad * math.sqrt(remainder)
            reference.state[0] = wrap_phase(
                reference.state[0]
                + torch.randn((), dtype=REAL_DTYPE, device=device, generator=generator)
                * walk_std
            )
            for link in links:
                link["slave"].state[0] = wrap_phase(
                    link["slave"].state[0]
                    + torch.randn(
                        (), dtype=REAL_DTYPE, device=device, generator=generator
                    )
                    * walk_std
                )

        phases = [torch.zeros((), dtype=REAL_DTYPE, device=device)]
        substep_all_valid = True
        for row, valid_row, link in zip(residual_rows, valid_rows, links):
            residual = wrap_phase(reference.state[0] - link["slave"].state[0])
            row.append(residual.clone())
            is_valid = link["loaded"] and link["calibrated"]
            valid_row.append(is_valid)
            substep_all_valid = substep_all_valid and is_valid
            phases.append(-residual)
        phasors = torch.exp(
            1j * torch.stack(phases).to(torch.complex128)
        )
        gain_history.append(
            (torch.abs(torch.sum(phasors)) ** 2 / num_stations**2).real
        )
        all_valid_history.append(substep_all_valid)

    return PiggybackStarResult(
        station_residuals=torch.stack(
            [torch.stack(row).detach().cpu() for row in residual_rows]
        ),
        station_valid=torch.tensor(valid_rows, dtype=torch.bool),
        array_gain=torch.stack(gain_history).detach().cpu().to(torch.float64),
        all_valid=torch.tensor(all_valid_history, dtype=torch.bool),
        detection_rate=detected_count / max(observation_count, 1),
        piggyback_airtime=piggyback_airtime,
        paid_airtime_if_dedicated=paid_airtime,
        oneway_phase_var=float(
            np.mean([link["oneway_phase_var"] for link in links])
        ),
        oneway_frequency_var=float(
            np.mean([link["oneway_freq_var"] for link in links])
        ),
    )


def _mean_std(values: list[float]) -> tuple[float, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return tensor.mean().item(), (
        tensor.std().item() if len(values) > 1 else 0.0
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="clutter-referenced sync on the real OFDM sensing waveform"
    )
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    seeds = [0] if args.quick else [int(s) for s in args.seeds.split(",")]
    iterations = 20 if args.quick else args.iterations

    print(
        "Clutter-referenced sync, realistic-waveform edition "
        f"({iterations} intervals, seeds {seeds})"
    )

    # ---- estimator noise: OFDM sensing burst vs ZC preamble ---------
    device = resolve_device("cpu")
    base = SDRSimulationConfig(num_iterations=iterations, seed=0, device="cpu")
    print("\n=== per-observation estimator noise (measured, all impairments) ===")
    for mode in ("zc", "ofdm"):
        phase_var, freq_var, rate = calibrate_oneway_noise(base, mode, device)
        print(
            f"  {mode:<5} phase std {1e3 * math.sqrt(phase_var):7.1f} mrad   "
            f"freq std {math.sqrt(freq_var) / (2 * math.pi):7.2f} Hz   "
            f"detect {100 * rate:5.1f}%"
        )

    # ---- headline: N=2, OFDM vs ZC one-way, K=5 and K=40 ------------
    print("\n=== N=2 piggyback star: real OFDM sensing bursts vs ZC control ===")
    print(
        f"  {'waveform':<9} {'K':>4} {'rms mrad':>14} {'gain %':>8} "
        f"{'piggyback %':>12} {'detect %':>9}"
    )
    for mode in ("ofdm", "zc"):
        for cadence in (5, 40):
            cells = []
            for seed in seeds:
                settings = SDRSimulationConfig(
                    num_iterations=iterations, seed=seed, device="cpu"
                )
                result = run_piggyback_star(
                    settings, num_stations=2,
                    anchor_every_intervals=cadence, waveform=mode,
                )
                cells.append(
                    (result.worst_rms_mrad, result.mean_array_gain,
                     result.piggyback_airtime, result.detection_rate)
                )
            rms_mean, rms_std = _mean_std([c[0] for c in cells])
            gain_mean, _ = _mean_std([c[1] for c in cells])
            air_mean, _ = _mean_std([c[2] for c in cells])
            det_mean, _ = _mean_std([c[3] for c in cells])
            print(
                f"  {mode:<9} {cadence:>4} {rms_mean:8.1f}±{rms_std:4.1f} "
                f"{100 * gain_mean:8.2f} {100 * air_mean:12.2f} "
                f"{100 * det_mean:9.1f}"
            )

    if args.quick:
        return

    # ---- N=6 star: piggyback OFDM vs the paid baselines --------------
    print("\n=== N=6 star: piggyback OFDM vs scheduled two-way vs micro ===")
    print(
        f"  {'scheme':<22} {'worst rms':>10} {'gain %':>8} {'airtime %':>10}"
    )
    for cadence in (5, 40):
        cells = []
        for seed in seeds:
            settings = SDRSimulationConfig(
                num_iterations=iterations, seed=seed, device="cpu"
            )
            result = run_piggyback_star(
                settings, num_stations=6,
                anchor_every_intervals=cadence, waveform="ofdm",
            )
            cells.append(
                (result.worst_rms_mrad, result.mean_array_gain,
                 result.piggyback_airtime)
            )
        rms_mean, rms_std = _mean_std([c[0] for c in cells])
        gain_mean, _ = _mean_std([c[1] for c in cells])
        air_mean, _ = _mean_std([c[2] for c in cells])
        print(
            f"  piggyback-ofdm K={cadence:<3}   {rms_mean:6.1f}±{rms_std:4.0f} "
            f"{100 * gain_mean:8.2f} {100 * air_mean:10.2f}"
        )
    for label, kwargs in (
        ("scheduled two-way", {}),
        ("micro-pilot star", {"multi_fidelity": True}),
    ):
        cells = []
        for seed in seeds:
            settings = SDRSimulationConfig(
                num_iterations=iterations, seed=seed, device="cpu"
            )
            star = run_scheduled_star(
                settings, num_stations=6, policy="scheduled", **kwargs
            )
            worst = max(v for v in star.station_steady_rms if v == v)
            cells.append(
                (1e3 * worst, star.mean_array_gain, star.airtime_used_fraction)
            )
        rms_mean, rms_std = _mean_std([c[0] for c in cells])
        gain_mean, _ = _mean_std([c[1] for c in cells])
        air_mean, _ = _mean_std([c[2] for c in cells])
        print(
            f"  {label:<22} {rms_mean:6.1f}±{rms_std:4.0f} "
            f"{100 * gain_mean:8.2f} {100 * air_mean:10.2f}"
        )

    # ---- observation-rate sweep: does the white resampling noise ----
    # average down as 1/sqrt(n)?
    print(
        "\n=== observation-rate sweep (N=2, OFDM, K=40): residual vs "
        "observations/interval ==="
    )
    print(f"  {'n obs':>6} {'rms mrad':>14} {'x expected 1/sqrt(n)':>22}")
    baseline_rms = None
    for n_obs in (1, 2, 5, 10):
        cells = []
        for seed in seeds:
            settings = SDRSimulationConfig(
                num_iterations=iterations, seed=seed, device="cpu"
            )
            result = run_piggyback_star(
                settings, num_stations=2, anchor_every_intervals=40,
                obs_per_interval=n_obs, waveform="ofdm",
            )
            cells.append(result.worst_rms_mrad)
        rms_mean, rms_std = _mean_std(cells)
        if baseline_rms is None:
            baseline_rms = rms_mean
            expected = 1.0
        else:
            expected = 1.0 / math.sqrt(n_obs)
        print(
            f"  {n_obs:>6} {rms_mean:8.1f}±{rms_std:4.1f} "
            f"{rms_mean / baseline_rms:11.2f} vs {expected:5.2f}"
        )


if __name__ == "__main__":
    main()
