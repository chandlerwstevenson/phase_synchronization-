"""Tests for the OFDM-loop environment check (ofdm_environment_check)."""

import clutter_sync_ofdm
from ofdm_environment_check import intervals_for, run_point
from ota_sync import SDRSimulationConfig


def test_intervals_respect_anchor_cycles():
    assert intervals_for(5) == 60
    assert intervals_for(40) == 160
    assert intervals_for(160) == 640


def test_run_point_finite_and_cache_cleared():
    # Seed the calibration cache with a poison entry keyed like a
    # different environment; run_point must clear it (the cache key
    # omits channel parameters, so a stale entry would silently
    # miscalibrate the filter in a new environment).
    clutter_sync_ofdm._CALIBRATION_CACHE[("poison",)] = (1.0, 1.0, 1.0)
    settings = SDRSimulationConfig(
        num_iterations=12, seed=0, device="cpu"
    )
    out = run_point(settings, cadence=4, waveform="zc")
    assert out["rms_mrad"] == out["rms_mrad"]  # finite
    assert 0.0 < out["airtime"] < 1.0
    assert 0.0 < out["obs_mrad"] < 1e4
    assert ("poison",) not in clutter_sync_ofdm._CALIBRATION_CACHE
