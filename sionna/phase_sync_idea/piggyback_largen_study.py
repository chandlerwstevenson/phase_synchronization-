"""Root-cause and fix the piggyback star's error creep at large N.

piggyback_scaling_study.py measured worst-station residual growing
63 -> 212 mrad from N=6 to N=14 while the airtime advantage held.
Candidate mechanisms, each discriminated by a variant of the star run
on identical physics (run_piggyback_star copied here with flags; the
original file is untouched, and with every flag at its default this
copy is regression-locked bit-for-bit against the original):

  stagger      each station anchors every K intervals offset by its
               index (the original schedule). Per-station spacing is K
               regardless of N by construction, so if de-staggering
               ("none": everyone anchors the same interval) changes
               nothing, the staggering hypothesis is dead.
  reference-walk artifact: the simulation serializes one-way captures
               per link, and EVERY capture advances the shared
               reference oscillator's phase by its capture-time walk.
               Physically the reference broadcasts ONE burst per
               substep that all stations overhear, so the reference
               should walk once - at N stations the simulated
               reference accrues (N-1) capture walks per substep, an
               excess of (N-2)*capture_samples of white-FM variance
               that no link's filter models. broadcast_reference=True
               applies the walk once per substep (the physical
               reading); inflate_process=True instead keeps the walk
               and tells each filter about it. If either flattens the
               creep, this is the mechanism.
  order statistics / link budget: more stations at the same radius =
               more draws from the same per-station error
               distribution (worst-of-N grows even if nothing is
               wrong) and more chance of a weak link. Discriminated by
               per-station diagnostics: mean-station error vs
               worst-station error, and error vs link distance.

Usage:
    .venv/bin/python piggyback_largen_study.py --part diag
    .venv/bin/python piggyback_largen_study.py --part sweep
    .venv/bin/python piggyback_largen_study.py --part env
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace

import numpy as np
import torch

import clutter_sync_ofdm as base_module
from clutter_sync_ofdm import (
    OFDMOneWayEstimator,
    PiggybackStarResult,
    _frozen_oscillator,
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
from ota_sync.scheduled import run_scheduled_star
from ota_sync.sdr import (
    SDRSynchronizer,
    _FlickerFrequencyNoise,
    _measurement_covariance,
    _quantize_correction,
    make_sync_preamble,
)
from sionna.phy import config as sionna_config


@dataclass(frozen=True)
class VariantResult:
    """PiggybackStarResult plus the diagnostics the root-cause needs."""

    star: PiggybackStarResult
    station_distances_m: list[float]
    station_snrs_db: list[float]
    anchor_log: list[tuple[int, int]]  # (interval, station)

    @property
    def per_station_spacing(self) -> dict[int, list[int]]:
        by_station: dict[int, list[int]] = {}
        for interval, station in self.anchor_log:
            by_station.setdefault(station, []).append(interval)
        return {
            s: [b - a for a, b in zip(v, v[1:])]
            for s, v in by_station.items()
        }


def _calibrate_on_own_channel(
    link_settings: SDRSimulationConfig,
    waveform: str,
    device: torch.device,
    oneway,
    ofdm_pool,
    zc_preamble,
    captures: int,
    station: int,
) -> tuple[float, float]:
    """Commission the one-way estimator on the LINK'S OWN frozen
    channel (mirror shares the taps tensor), instead of the fresh
    independent draw clutter_sync_ofdm.calibrate_oneway_noise uses.
    A real receiver measures its estimator variance in place, so this
    is the physically honest reading; it also captures the
    draw-dependent multipath/jitter resampling noise the shared
    calibration misses."""

    frozen = replace(
        link_settings,
        phase_process_std_rad=0.0,
        frequency_process_std_hz=0.0,
        flicker_frequency_std_hz=0.0,
        slave_initial_phase=0.7,
        slave_initial_frequency_hz=0.0,
        shadowing_std_db=0.0,
        channel_speed_mps=0.0,
        num_iterations=max(link_settings.num_iterations, captures),
    )
    sionna_config.seed = frozen.seed + 77 + station
    cal_generator = torch.Generator(device=device)
    cal_generator.manual_seed(frozen.seed + 78 + station)
    master = _frozen_oscillator(frozen.master_initial_phase, device)
    slave = _frozen_oscillator(frozen.slave_initial_phase, device)
    if waveform == "ofdm":
        cal_link = base_module.SDRRadioLink(
            frozen, ofdm_pool[0], device, cal_generator, mirror_of=oneway
        )
        estimator = OFDMOneWayEstimator(frozen)
    else:
        cal_link = base_module.SDRRadioLink(
            frozen, zc_preamble, device, cal_generator, mirror_of=oneway
        )
        synchronizer = SDRSynchronizer(frozen, zc_preamble)

    zero_omega = torch.zeros((), dtype=REAL_DTYPE, device=device)
    # The mirror shares the parent's tap tensor, which is sized to the
    # RUN's frame count (the channel is frozen, so any frame is the
    # same realization) - index within that bound.
    parent_frames = link_settings.num_iterations
    phases, frequencies = [], []
    for capture_index in range(captures):
        if waveform == "ofdm":
            cal_link.preamble = ofdm_pool[capture_index % len(ofdm_pool)]
        capture = cal_link.capture(
            master, slave, capture_index % parent_frames, 0.0
        )
        if waveform == "ofdm":
            estimate = estimator.estimate(
                capture.samples, cal_link.preamble.waveform, zero_omega
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
            phases.append(phase)
            frequencies.append(frequency)
    if not phases:
        return float("nan"), float("nan")
    phase_tensor = torch.stack(phases)
    mean_angle = torch.angle(
        torch.sum(torch.exp(1j * phase_tensor.to(torch.complex128)))
    )
    phase_variance = torch.mean(
        wrap_phase(phase_tensor - mean_angle).square()
    ).item()
    frequency_variance = torch.var(torch.stack(frequencies)).item()
    return phase_variance, frequency_variance


def run_piggyback_variant(
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
    stagger: str = "station",
    broadcast_reference: bool = False,
    inflate_process: bool = False,
    per_link_calibration: bool = False,
    frequency_every_obs: bool = False,
    correct_at_frame_only: bool = False,
    slip_compensation: bool = False,
) -> VariantResult:
    """run_piggyback_star with the three discriminating knobs.

    Defaults reproduce the original bit-for-bit (regression-tested).
    The radio links are constructed through the module attribute
    ``clutter_sync_ofdm.SDRRadioLink`` so environment_dependence_study's
    injected-channel context manager patches this copy too.
    """

    if waveform not in ("ofdm", "zc"):
        raise ValueError("waveform must be 'ofdm' or 'zc'")
    if num_stations < 2:
        raise ValueError("need at least two stations")
    if obs_per_interval < 1:
        raise ValueError("obs_per_interval must be at least one")
    if stagger not in ("station", "none"):
        raise ValueError("stagger must be 'station' or 'none'")

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
                (2.0 * math.pi * settings.frequency_process_std_hz) ** 2
                / substeps,
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
    distances, snrs = [], []
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
        distances.append(distance)
        snrs.append(snr_db)
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
            shadowing_correlation_s=link_settings.shadowing_correlation_s
            * substeps,
        )
        if waveform == "ofdm":
            oneway = base_module.SDRRadioLink(
                oneway_settings, ofdm_pool[0], device, generator,
                captures_per_interval=substeps,
            )
        else:
            oneway = base_module.SDRRadioLink(
                oneway_settings, zc_preamble, device, generator,
                captures_per_interval=substeps,
            )
        anchor_forward = base_module.SDRRadioLink(
            link_settings, zc_preamble, device, generator, mirror_of=oneway
        )
        anchor_reverse = base_module.SDRRadioLink(
            link_settings, zc_preamble, device, generator, mirror_of=oneway
        )

        zc_noise = _measurement_covariance(link_settings, zc_preamble, device)
        if per_link_calibration:
            phase_var, freq_var = _calibrate_on_own_channel(
                link_settings, waveform, device, oneway, ofdm_pool,
                zc_preamble, calibration_captures, station,
            )
        else:
            phase_var, freq_var, _ = calibrate_oneway_noise(
                link_settings, waveform, device,
                captures=calibration_captures,
            )
        oneway_noise3 = torch.diag(
            torch.tensor(
                [phase_var, phase_var, freq_var],
                dtype=REAL_DTYPE,
                device=device,
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
        oneway_capture_len = oneway.input_length + oneway.l_tot - 1
        excess_reference_walk = 0.0
        if inflate_process and not broadcast_reference:
            # The serialized captures walk the shared reference an
            # extra (N-2) capture-lengths of white FM per substep that
            # the per-link filter otherwise never hears about.
            excess_reference_walk = (
                max(0, num_stations - 2)
                * oneway_capture_len
                * link_settings.phase_noise_std_rad**2
            )
        process = torch.diag(
            torch.tensor(
                [
                    2.0 * substep_covariance[0, 0].item()
                    + white_fm_substep
                    + excess_reference_walk,
                    2.0 * substep_covariance[1, 1].item()
                    + flicker.innovation_variance,
                    0.01**2 / substeps,
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
                "slip_accum": 0.0,
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
        (num_stations - 1)
        * substeps
        * oneway_capture_samples
        / interval_samples
    )

    residual_rows = [[] for _ in links]
    valid_rows = [[] for _ in links]
    gain_history, all_valid_history = [], []
    detected_count, observation_count = 0, 0
    anchor_log: list[tuple[int, int]] = []

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

        reference_walk_applied = False
        for link in links:
            slave = link["slave"]
            ekf = link["ekf"]
            if stagger == "station":
                anchor_due = (
                    iteration - link["station"]
                ) % anchor_every_intervals == 0
            else:
                anchor_due = iteration % anchor_every_intervals == 0
            is_anchor = at_frame_slot and anchor_due
            physical = slave.state[1] - link["corrections"]
            sfo = float(
                (physical - reference.state[1]).item()
                / (2.0 * math.pi * settings.carrier_frequency_hz)
                * 1e6
            )

            ekf.predict()
            observation_count += 1
            if is_anchor:
                anchor_log.append((iteration, link["station"]))
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
                oneway = link["oneway"]
                if waveform == "ofdm":
                    oneway.preamble = ofdm_pool[
                        link["capture_count"] % len(ofdm_pool)
                    ]
                link["capture_count"] += 1
                capture = oneway.capture(reference, slave, iteration, sfo)
                if broadcast_reference:
                    # One physical broadcast per substep: the reference
                    # LO advances once, not once per overhearing
                    # station.
                    if not reference_walk_applied:
                        reference.state[0] = wrap_phase(
                            reference.state[0] + capture.lo_walk_end
                        )
                        reference_walk_applied = True
                else:
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
                if slip_compensation:
                    # The receiver's sample clock slips against the
                    # carrier at the physical oscillator offset rate,
                    # putting a 2*pi*f_phys*dt phase ramp into every
                    # one-way observation (the anchors are immune: the
                    # ramp cancels in the two-way half-difference).
                    # The receiver already measures this drift - it
                    # re-centers its capture window on it every frame -
                    # so subtract the accumulated slip phase.
                    link["slip_accum"] += (
                        2.0
                        * math.pi
                        * (sfo * 1e-6 * settings.carrier_frequency_hz)
                        * dt
                    )
                    obs_phase = wrap_phase(
                        obs_phase - torch.tensor(
                            link["slip_accum"],
                            dtype=REAL_DTYPE,
                            device=device,
                        )
                    )
                if detected:
                    detected_count += 1
                    if at_frame_slot or frequency_every_obs:
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
                if not correct_at_frame_only or at_frame_slot or is_anchor:
                    predicted = ekf.transition @ ekf.state
                    link["pending"][substep + 1] = _quantize_correction(
                        predicted[:2], settings
                    )

        remainder = max(0, dt_samples - oneway_capture_samples)
        if settings.phase_noise_std_rad > 0.0 and remainder > 0:
            walk_std = settings.phase_noise_std_rad * math.sqrt(remainder)
            reference.state[0] = wrap_phase(
                reference.state[0]
                + torch.randn(
                    (), dtype=REAL_DTYPE, device=device, generator=generator
                )
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

    star = PiggybackStarResult(
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
    return VariantResult(
        star=star,
        station_distances_m=distances,
        station_snrs_db=snrs,
        anchor_log=anchor_log,
    )


# ---------------------------------------------------------------------
# Study parts
# ---------------------------------------------------------------------

def _ms(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    std = (
        math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        if len(values) > 1
        else 0.0
    )
    return mean, std


VARIANTS = {
    "current": {},
    "de-staggered": {"stagger": "none"},
    "K-tightened(15)": {"anchor_every_intervals": 15},
    "broadcast-ref": {"broadcast_reference": True},
    "inflated-noise": {"inflate_process": True},
    "per-link-cal": {"per_link_calibration": True},
    "cal+inflated": {"per_link_calibration": True, "inflate_process": True},
    "freq-every-obs": {"frequency_every_obs": True},
    "correct-at-frame": {"correct_at_frame_only": True},
    "slip-comp": {"slip_compensation": True},
}


def part_diag(iterations: int, seeds: list[int], n: int, k: int) -> None:
    print(
        f"PART 1 - root-cause variants at N={n}, K={k}, "
        f"{iterations} intervals, seeds {seeds}"
    )
    print(
        f"  {'variant':<17} {'worst rms':>12} {'mean rms':>10} "
        f"{'gain':>7} {'airtime':>8}"
    )
    for name, flags in VARIANTS.items():
        worst, mean_station, gain, air = [], [], [], []
        per_station_last = None
        for seed in seeds:
            settings = SDRSimulationConfig(
                num_iterations=iterations, seed=seed, device="cpu"
            )
            kwargs = dict(num_stations=n, anchor_every_intervals=k)
            kwargs.update(flags)
            res = run_piggyback_variant(settings, **kwargs)
            rms = res.star.station_rms_mrad
            worst.append(res.star.worst_rms_mrad)
            mean_station.append(sum(rms) / len(rms))
            gain.append(res.star.mean_array_gain)
            air.append(res.star.piggyback_airtime)
            per_station_last = (rms, res.station_distances_m, res.station_snrs_db)
        wm, ws = _ms(worst)
        mm, _ = _ms(mean_station)
        gm, _ = _ms(gain)
        am, _ = _ms(air)
        per_seed = ",".join(f"{w:.0f}" for w in worst)
        print(
            f"  {name:<17} {wm:>7.1f}±{ws:<4.1f} {mm:>10.1f} "
            f"{100 * gm:>6.1f}% {100 * am:>7.2f}%  [per-seed worst: {per_seed}]"
        )
        if name == "current" and per_station_last is not None:
            rms, dist, snr = per_station_last
            print("    per-station (seed "
                  f"{seeds[-1]}): "
                  + "  ".join(
                      f"s{i + 1}:{r:.0f}mrad@{d:.0f}m/{s:.0f}dB"
                      for i, (r, d, s) in enumerate(zip(rms, dist, snr))
                  ))


def part_sweep(iterations: int, seeds: list[int], k: int,
               station_counts: list[int], fix: str) -> None:
    flags = VARIANTS[fix]
    print(
        f"\nPART 2 - N sweep with fix '{fix}', K={k}, seeds {seeds}"
    )
    print(
        f"  {'N':>3} {'fixed worst-rms':>16} {'mean rms':>9} {'gain':>7} "
        f"{'airtime':>8} {'| two-way worst':>15} {'airtime':>8} {'ratio':>7}"
    )
    for n in station_counts:
        worst, mean_station, gain, air = [], [], [], []
        two_worst, two_air = [], []
        for seed in seeds:
            settings = SDRSimulationConfig(
                num_iterations=iterations, seed=seed, device="cpu"
            )
            kwargs = dict(num_stations=n, anchor_every_intervals=k)
            kwargs.update(flags)
            res = run_piggyback_variant(settings, **kwargs)
            rms = res.star.station_rms_mrad
            worst.append(res.star.worst_rms_mrad)
            mean_station.append(sum(rms) / len(rms))
            gain.append(res.star.mean_array_gain)
            air.append(res.star.piggyback_airtime)
            two = run_scheduled_star(
                settings, num_stations=n, policy="scheduled"
            )
            tw = max(
                (v for v in two.station_steady_rms if v == v),
                default=float("nan"),
            )
            two_worst.append(1e3 * tw)
            two_air.append(two.airtime_used_fraction)
        wm, ws = _ms(worst)
        mm, _ = _ms(mean_station)
        gm, _ = _ms(gain)
        am, _ = _ms(air)
        twm, tws = _ms(two_worst)
        tam, _ = _ms(two_air)
        ratio = tam / am if am > 0 else float("inf")
        print(
            f"  {n:>3} {wm:>10.1f}±{ws:<5.1f} {mm:>9.1f} {100 * gm:>6.1f}% "
            f"{100 * am:>7.2f}% | {twm:>9.1f}±{tws:<4.1f} "
            f"{100 * tam:>7.2f}% {ratio:>6.1f}x"
        )


def part_env(iterations: int, seeds: list[int], n: int, k: int,
             fix: str) -> None:
    from environment_dependence_study import (
        cir_to_frozen_taps,
        injected_channel,
        rt_station_pair_cir,
    )

    flags = VARIANTS[fix]
    print(f"\nPART 3 - environment x N (N={n}, K={k}, fix '{fix}', "
          f"seeds {seeds})")
    print(f"  {'environment':<22} {'worst rms':>12} {'gain':>7} "
          f"{'detect':>7}")

    environments: list[tuple[str, dict, torch.Tensor | None]] = [
        ("TDL-D (headline)", {}, None),
        ("TDL-A (no LOS)", {"tdl_model": "A"}, None),
    ]
    gains_urban, delays_urban, diag = rt_station_pair_cir("urban-los")
    base = SDRSimulationConfig(num_iterations=iterations, seed=0, device="cpu")
    taps, _ = cir_to_frozen_taps(gains_urban, delays_urban, base)
    environments.append(
        (
            f"RT urban-los ({diag['num_paths']}p)",
            {},
            taps,
        )
    )

    for label, overrides, injected in environments:
        worst, gain, det = [], [], []
        for seed in seeds:
            settings = SDRSimulationConfig(
                num_iterations=iterations, seed=seed, device="cpu",
                **overrides,
            )
            kwargs = dict(num_stations=n, anchor_every_intervals=k)
            kwargs.update(flags)
            with injected_channel(injected):
                res = run_piggyback_variant(settings, **kwargs)
            worst.append(res.star.worst_rms_mrad)
            gain.append(res.star.mean_array_gain)
            det.append(res.star.detection_rate)
        wm, ws = _ms(worst)
        gm, _ = _ms(gain)
        dm, _ = _ms(det)
        print(
            f"  {label:<22} {wm:>7.1f}±{ws:<4.1f} {100 * gm:>6.1f}% "
            f"{100 * dm:>6.1f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="root-cause and fix piggyback error creep at large N"
    )
    parser.add_argument("--part", choices=("diag", "sweep", "env"),
                        required=True)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--anchor-every", type=int, default=40)
    parser.add_argument("--diag-stations", type=int, default=10)
    parser.add_argument("--stations", type=str, default="6,10,14,20")
    parser.add_argument("--fix", type=str, default="broadcast-ref",
                        choices=tuple(VARIANTS))
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    if args.part == "diag":
        part_diag(args.iterations, seeds, args.diag_stations,
                  args.anchor_every)
    elif args.part == "sweep":
        part_sweep(
            args.iterations, seeds, args.anchor_every,
            [int(v) for v in args.stations.split(",")], args.fix,
        )
    else:
        part_env(args.iterations, seeds, 6, args.anchor_every, args.fix)


if __name__ == "__main__":
    main()
