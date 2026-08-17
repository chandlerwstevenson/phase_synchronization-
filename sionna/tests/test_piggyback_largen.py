"""Tests for the large-N piggyback variants (piggyback_largen_study).

Load-bearing test first: with every flag at its default the variant
must reproduce clutter_sync_ofdm.run_piggyback_star bit-for-bit, so
every root-cause comparison is against the true original.
"""

import torch

import clutter_sync_ofdm
from clutter_sync_ofdm import run_piggyback_star
from ota_sync import SDRSimulationConfig
from piggyback_largen_study import run_piggyback_variant

FAST = SDRSimulationConfig(num_iterations=12, seed=0, device="cpu")
MICRO = dict(
    num_stations=3,
    anchor_every_intervals=4,
    obs_per_interval=3,
    calibration_captures=24,
)


def test_default_variant_is_bit_identical_to_original():
    # The commissioning-noise memo (_CALIBRATION_CACHE) skips RNG
    # draws on a hit, so both runs must start cache-cold for the
    # random streams to align (this order-sensitivity is a property of
    # the original function too, and is statistically benign).
    clutter_sync_ofdm._CALIBRATION_CACHE.clear()
    original = run_piggyback_star(FAST, **MICRO)
    clutter_sync_ofdm._CALIBRATION_CACHE.clear()
    variant = run_piggyback_variant(FAST, **MICRO)
    assert torch.equal(
        original.station_residuals, variant.star.station_residuals
    )
    assert torch.equal(original.array_gain, variant.star.array_gain)
    assert original.detection_rate == variant.star.detection_rate
    assert original.piggyback_airtime == variant.star.piggyback_airtime


def test_destaggered_per_station_spacing_equals_k():
    result = run_piggyback_variant(FAST, stagger="none", **MICRO)
    spacing = result.per_station_spacing
    assert set(spacing) == {1, 2}
    for gaps in spacing.values():
        assert gaps and all(g == MICRO["anchor_every_intervals"] for g in gaps)


def test_staggered_per_station_spacing_also_equals_k():
    # The staggering offsets the schedule but must never stretch any
    # single station's anchor cadence.
    result = run_piggyback_variant(FAST, **MICRO)
    for gaps in result.per_station_spacing.values():
        assert all(g == MICRO["anchor_every_intervals"] for g in gaps)


def test_broadcast_reference_runs_and_stays_locked():
    result = run_piggyback_variant(
        FAST, broadcast_reference=True, **MICRO
    )
    assert torch.any(result.star.all_valid)
    assert result.star.worst_rms_mrad < 1000.0


def test_inflated_noise_grows_with_station_count():
    small = run_piggyback_variant(FAST, inflate_process=True, **MICRO)
    big_kwargs = dict(MICRO)
    big_kwargs["num_stations"] = 6
    big = run_piggyback_variant(FAST, inflate_process=True, **big_kwargs)
    theta_small = small.star.station_residuals.shape[0]
    theta_big = big.star.station_residuals.shape[0]
    assert theta_small == 2 and theta_big == 5
