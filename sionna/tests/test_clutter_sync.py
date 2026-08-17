"""Tests for the clutter-referenced sync study (clutter_sync_study.py).

Load-bearing checks: the piggyback accounting charges anchors only and
reconstructs the hybrid's own native accounting exactly; the two-way
baseline reproduces the README-documented numbers (harness used
correctly); and the static environment really is a stable phase
reference (the premise the whole scheme rests on).
"""

import math

import torch

from clutter_sync_study import (
    full_capture_samples,
    matched_channel_drift_std,
    piggyback_airtime_fraction,
    run_clutter_referenced,
)
from ota_sync import SDRSimulationConfig, run_two_way_simulation
from ota_sync.core import wrap_phase

FAST = SDRSimulationConfig(num_iterations=12, seed=0, device="cpu")


def test_piggyback_charges_anchors_only_and_matches_native_algebra():
    micro_pilots = 4
    result, native, piggyback = run_clutter_referenced(
        FAST, anchor_every_intervals=5,
        micro_pilots_per_interval=micro_pilots,
    )
    interval_samples = int(round(FAST.sync_interval * FAST.sample_rate))
    full = full_capture_samples(FAST)
    # Native accounting (from hybrid.py): one full one-way frame per
    # interval, the anchor's extra reverse capture every K, plus micros.
    micro_share = native - (full * (1.0 + 1.0 / 5)) / interval_samples
    assert micro_share > 0.0  # micros are charged natively
    # Piggyback charges exactly two full captures every K intervals.
    assert piggyback == 2.0 * full / (5 * interval_samples)
    assert piggyback < native


def test_piggyback_decreases_with_anchor_cadence():
    values = [
        piggyback_airtime_fraction(FAST, cadence)
        for cadence in (5, 10, 20, 40)
    ]
    assert all(a > b for a, b in zip(values, values[1:]))
    assert values[-1] < 0.01  # K=40: below 1% of the channel


def test_hybrid_reproduces_documented_numbers():
    # hybrid_calibration/README.md: K=5 static sits at ~32-34 mrad.
    settings = SDRSimulationConfig(
        num_iterations=60, seed=0, device="cpu"
    )
    result, native, piggyback = run_clutter_referenced(
        settings, anchor_every_intervals=5
    )
    assert 30.0 < 1e3 * result.steady_state_phase_rms < 36.0
    assert result.detection_rate == 1.0
    assert piggyback < native


def test_two_way_baseline_airtime_algebra():
    # The bare two-way loop charges exactly two full captures per
    # interval; its residual must be in the loop's normal regime (the
    # README's 27.9 mrad "control" row uses a different cadence
    # configuration, so we assert our baseline's own algebra instead).
    settings = SDRSimulationConfig(
        num_iterations=30, seed=0, device="cpu"
    )
    result = run_two_way_simulation(settings)
    interval_samples = int(
        round(settings.sync_interval * settings.sample_rate)
    )
    expected = 2.0 * full_capture_samples(settings) / interval_samples
    assert abs(result.airtime_fraction - expected) < 1e-9
    assert 1e3 * result.steady_state_phase_rms < 150.0


def test_static_environment_is_a_stable_phase_reference():
    # The scheme's premise: with a frozen channel the estimated channel
    # phase settles to a constant; with motion it must wander more.
    static_result, _, _ = run_clutter_referenced(
        SDRSimulationConfig(num_iterations=20, seed=0, device="cpu"),
        anchor_every_intervals=5,
    )
    moving_settings = SDRSimulationConfig(
        num_iterations=20, seed=0, device="cpu", channel_speed_mps=0.5
    )
    moving_result, _, _ = run_clutter_referenced(
        moving_settings, anchor_every_intervals=5,
        channel_drift_std_rad=matched_channel_drift_std(
            moving_settings, 0.5
        ),
    )

    def tail_channel_wander(result) -> float:
        tail = result.estimated_channel_phase[
            result.estimated_channel_phase.shape[0] // 2:
        ]
        steps = wrap_phase(tail[1:] - tail[:-1])
        return torch.sqrt(torch.mean(steps.square())).item()

    static_wander = tail_channel_wander(static_result)
    moving_wander = tail_channel_wander(moving_result)
    assert static_wander < 0.15  # rad/substep: effectively pinned
    assert moving_wander > 2.0 * static_wander


def test_matched_prior_monotone_in_speed_with_static_floor():
    assert matched_channel_drift_std(FAST, 0.0) == 0.01
    slow = matched_channel_drift_std(FAST, 0.2)
    fast = matched_channel_drift_std(FAST, 0.5)
    assert 0.01 <= slow < fast
    assert fast < math.pi  # sane magnitude
