"""Tests for the Doppler coasting study instrumentation.

Load-bearing: the recording swap (EKF + synchronizer + radio link)
with fade_aware=False must reproduce the plain run bit-for-bit, so
every number in doppler_coast_study.py's fixed-R rows is the same
physics every published study measured.
"""

import math

import torch

from doppler_coast_study import (
    channel_energy,
    replay_update_with_true_noise,
    run_star_instrumented,
    service_records,
    _metric_to_snr,
)
from ota_sync import SDRSimulationConfig
from ota_sync.scheduled import run_scheduled_star

FAST = SDRSimulationConfig(num_iterations=18, seed=0, device="cpu")


def test_instrumentation_changes_nothing():
    plain = run_scheduled_star(FAST, num_stations=3)
    instrumented, tape = run_star_instrumented(FAST, num_stations=3)
    assert torch.equal(plain.residuals, instrumented.residuals)
    assert torch.equal(plain.array_gain, instrumented.array_gain)
    assert plain.airtime_used_fraction == instrumented.airtime_used_fraction
    assert len(tape["ekfs"]) == 2
    assert len(tape["forward_links"]) == 2


def test_service_records_align_with_serviced_mask():
    result, tape = run_star_instrumented(FAST, num_stations=3)
    records = service_records(result, tape)
    for k, link_records in enumerate(records):
        assert len(link_records) == int(result.serviced[k].sum())
        for record in link_records:
            assert bool(result.serviced[k][record["interval"]])
            assert 0.0 <= record["metric_min"] <= 1.0


def test_channel_energy_shape_and_positivity():
    result, tape = run_star_instrumented(FAST, num_stations=3)
    for link in tape["forward_links"]:
        energy = channel_energy(link)
        assert energy.shape == (FAST.num_iterations,)
        assert torch.all(energy > 0.0)


def test_metric_to_snr_monotone():
    assert _metric_to_snr(0.9) > _metric_to_snr(0.5) > _metric_to_snr(0.1)
    # nominal-metric round trip: gamma=100 -> m=sqrt(100/101)
    m = math.sqrt(100.0 / 101.0)
    assert abs(_metric_to_snr(m) - 100.0) / 100.0 < 1e-9


def test_replay_with_unit_scale_matches_believed():
    result, tape = run_star_instrumented(FAST, num_stations=3)
    ekf = tape["ekfs"][0]
    record = dict(ekf.updates[-1])
    record["fade_scale"] = 1.0
    replayed = replay_update_with_true_noise(ekf, record)
    believed = record["posterior_covariance"]
    # Same gain, same R -> the replay must reproduce the believed
    # posterior up to the iterated-EKF linearization point.
    assert torch.allclose(replayed, believed, rtol=0.05, atol=1e-12)


def test_replay_with_fade_inflates_covariance():
    result, tape = run_star_instrumented(FAST, num_stations=3)
    ekf = tape["ekfs"][0]
    record = dict(ekf.updates[-1])
    record["fade_scale"] = 10.0
    inflated = replay_update_with_true_noise(ekf, record)
    believed = record["posterior_covariance"]
    assert inflated[0, 0] > believed[0, 0]
    assert inflated[1, 1] > believed[1, 1]


def test_fade_aware_still_locks_at_zero_speed():
    result, _ = run_star_instrumented(
        FAST, fade_aware=True, num_stations=3
    )
    assert result.mean_array_gain > 0.9
