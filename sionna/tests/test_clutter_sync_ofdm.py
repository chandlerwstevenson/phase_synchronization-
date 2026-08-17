"""Tests for the realistic-waveform piggyback star (clutter_sync_ofdm).

Load-bearing checks: the OFDM matched-filter estimator reads the same
constant observable (theta + channel phase) as the ZC synchronizer on
the same frozen channel, the piggyback airtime accounting charges only
the staggered anchors, and the N=2 star converges under both waveforms.
"""

import math

import pytest
import torch

from clutter_sync_ofdm import (
    OFDMOneWayEstimator,
    calibrate_oneway_noise,
    make_ofdm_frame_pool,
    run_piggyback_star,
)
from ota_sync import SDRSimulationConfig
from ota_sync.core import REAL_DTYPE, resolve_device, wrap_phase
from ota_sync.sdr import SDRRadioLink, SDRSynchronizer, make_sync_preamble

FAST = SDRSimulationConfig(num_iterations=16, seed=0, device="cpu")


def _frozen_pair(device):
    from clutter_sync_ofdm import _frozen_oscillator

    return (
        _frozen_oscillator(0.3, device),
        _frozen_oscillator(-0.9, device),
    )


def test_ofdm_estimator_matches_zc_observable():
    # Same frozen channel (mirrored links), frozen oscillators: both
    # estimators must read theta + phi_c up to their own noise.
    device = resolve_device("cpu")
    from dataclasses import replace

    quiet = replace(
        FAST,
        phase_noise_std_rad=0.0,
        phase_noise_white_pm_std_rad=0.0,
        shadowing_std_db=0.0,
        snr_db=35.0,
        timing_jitter_samples=0,
    )
    torch.manual_seed(3)
    generator = torch.Generator(device=device)
    generator.manual_seed(3)
    pool = make_ofdm_frame_pool(2, device, 42)
    ofdm_link = SDRRadioLink(quiet, pool[0], device, generator)
    zc_preamble = make_sync_preamble(quiet, device)
    zc_link = SDRRadioLink(quiet, zc_preamble, device, generator, mirror_of=ofdm_link)

    master, slave = _frozen_pair(device)
    zero = torch.zeros((), dtype=REAL_DTYPE, device=device)
    estimator = OFDMOneWayEstimator(quiet)

    ofdm_capture = ofdm_link.capture(master, slave, 0, 0.0)
    ofdm = estimator.estimate(ofdm_capture.samples, pool[0].waveform, zero)
    zc_capture = zc_link.capture(master, slave, 0, 0.0)
    zc = SDRSynchronizer(quiet, zc_preamble).estimate(zc_capture.samples)

    assert ofdm.detected and zc.detected
    difference = wrap_phase(ofdm.phase - zc.phase).abs().item()
    assert difference < 0.05  # both read theta + phi_c of the shared taps


def test_ofdm_estimator_rejects_wrong_frame():
    device = resolve_device("cpu")
    torch.manual_seed(4)
    generator = torch.Generator(device=device)
    generator.manual_seed(4)
    pool = make_ofdm_frame_pool(2, device, 43)
    link = SDRRadioLink(FAST, pool[0], device, generator)
    master, slave = _frozen_pair(device)
    zero = torch.zeros((), dtype=REAL_DTYPE, device=device)
    estimator = OFDMOneWayEstimator(FAST)
    capture = link.capture(master, slave, 0, 0.0)
    right = estimator.estimate(capture.samples, pool[0].waveform, zero)
    wrong = estimator.estimate(capture.samples, pool[1].waveform, zero)
    assert right.detected
    assert wrong.metric < right.metric * 0.5  # unknown frame does not correlate


def test_calibration_returns_finite_and_caches():
    device = resolve_device("cpu")
    first = calibrate_oneway_noise(FAST, "ofdm", device, captures=40)
    second = calibrate_oneway_noise(FAST, "ofdm", device, captures=40)
    assert first == second  # cache hit is exact
    phase_var, freq_var, rate = first
    assert phase_var > 0.0 and freq_var > 0.0
    assert rate > 0.9


def test_piggyback_airtime_charges_anchors_only():
    result = run_piggyback_star(
        FAST, num_stations=2, anchor_every_intervals=8,
        waveform="zc", calibration_captures=30,
    )
    half = run_piggyback_star(
        FAST, num_stations=2, anchor_every_intervals=4,
        waveform="zc", calibration_captures=30,
    )
    assert result.piggyback_airtime == pytest.approx(
        half.piggyback_airtime / 2.0
    )
    # Free observations do not enter the piggyback figure but do enter
    # the paid figure.
    assert result.paid_airtime_if_dedicated > result.piggyback_airtime


def test_star_converges_both_waveforms():
    for mode in ("zc", "ofdm"):
        result = run_piggyback_star(
            FAST, num_stations=2, anchor_every_intervals=4,
            waveform=mode, calibration_captures=30,
        )
        assert torch.any(result.all_valid)
        assert result.worst_rms_mrad < 400.0, mode
        assert result.mean_array_gain > 0.9, mode
        assert result.detection_rate > 0.7, mode


def test_bad_arguments_rejected():
    with pytest.raises(ValueError, match="waveform"):
        run_piggyback_star(FAST, waveform="chirp")
    with pytest.raises(ValueError, match="stations"):
        run_piggyback_star(FAST, num_stations=1)
