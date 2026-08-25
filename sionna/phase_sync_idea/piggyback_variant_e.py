"""Parameterized copy of clutter_sync_ofdm.run_piggyback_star for the
component-ablation study (experiment E).

Three flags, defaults reproduce the original bit-for-bit (verified by
ablation_study.py before any ablation runs):

  branch_check=False    disable the periodic 1-bit pi-branch check
                        (calibrated flag still set so validity masking
                        is unchanged; no corrective pi flip applied)
  channel_state=False   remove the channel state from the estimator:
                        q_psi = 0, psi pinned to 0 at acquisition with
                        ~zero variance. The 3-state machinery then
                        degenerates to a 2-state [theta, omega] filter
                        that treats the composite one-way observation
                        as pure oscillator phase (Kalman gain on psi is
                        exactly zero thereafter).
  anchor_every_intervals=BIG   gives exactly one acquisition anchor per
                        station and none after (K -> infinity ablation).
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import torch

from clutter_sync_ofdm import (
    OFDM_PULSE_LENGTH,
    OFDMOneWayEstimator,
    PiggybackStarResult,
    calibrate_oneway_noise,
    make_ofdm_frame_pool,
)
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
from ota_sync.core import REAL_DTYPE, Oscillator, resolve_device, wrap_phase
from ota_sync.network import MAX_LINK_SNR_DB, place_stations
from ota_sync.sdr import (
    SDRRadioLink,
    SDRSynchronizer,
    _FlickerFrequencyNoise,
    _measurement_covariance,
    _quantize_correction,
    make_sync_preamble,
)
from sionna.phy import config as sionna_config


def run_piggyback_variant_e(
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
    branch_check: bool = True,
    channel_state: bool = True,
) -> PiggybackStarResult:
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
                    (0.01**2 / substeps) if channel_state else 0.0,
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
            if link["loaded"]:
                link["settled"] += 1
                if link["settled"] >= 3:
                    if branch_check:
                        misaligned = torch.cos(
                            reference.state[0] - link["slave"].state[0]
                        ) < (-0.2 if link["calibrated"] else 0.0)
                        if misaligned:
                            link["slave"].apply_correction(
                                torch.tensor(
                                    [math.pi, 0.0], dtype=REAL_DTYPE, device=device
                                )
                            )
                            if channel_state:
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
                        if not channel_state:
                            channel_obs = torch.zeros_like(channel_obs)
                        ekf.state = torch.stack(
                            (theta_obs, frequency_obs, channel_obs)
                        )
                        third_var = (
                            link["anchor_noise"][3, 3]
                            if channel_state
                            else torch.tensor(
                                1e-12, dtype=REAL_DTYPE, device=device
                            )
                        )
                        ekf.covariance = torch.diag(
                            torch.stack(
                                (
                                    link["anchor_noise"][0, 0],
                                    link["anchor_noise"][2, 2],
                                    third_var,
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
