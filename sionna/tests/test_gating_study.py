"""Tests for posterior-gated membership (gating_study.py).

The two load-bearing tests: (1) instrumenting the star from the
outside changes NOTHING - residuals, gains, and airtime are
bit-identical to the unpatched run; (2) the weighted detection
pipeline with unit weights reproduces detection/waveform.py exactly.
Everything downstream (gating itself) is bookkeeping whose invariants
are checked directly.
"""

import math

import numpy as np
import pytest
import torch

from detection import DetectionParams
from detection.waveform import run_waveform_detection
from gating_study import (
    evaluation_mask,
    greedy_oracle_weights,
    membership_weights,
    oracle_gate_weights,
    phase_matrix,
    posterior_gate_weights,
    posterior_soft_weights,
    run_gated_waveform_detection,
    run_star_with_posteriors,
    weighted_gain,
)
from ota_sync import SDRSimulationConfig
from ota_sync.scheduled import run_scheduled_star

FAST = SDRSimulationConfig(num_iterations=18, seed=0, device="cpu")


def test_instrumentation_changes_nothing():
    plain = run_scheduled_star(FAST, num_stations=3)
    instrumented, sigma = run_star_with_posteriors(FAST, num_stations=3)
    assert torch.equal(plain.residuals, instrumented.residuals)
    assert torch.equal(plain.array_gain, instrumented.array_gain)
    assert torch.equal(plain.serviced, instrumented.serviced)
    assert plain.airtime_used_fraction == instrumented.airtime_used_fraction
    assert sigma.shape == (2, FAST.num_iterations)
    assert torch.all(sigma > 0.0)


def test_starved_station_posterior_grows():
    # Capacity 1 under uniform services only the first link; the
    # starved links' posteriors must grow monotonically (predict-only,
    # never an update) and dwarf the serviced link's.
    result, sigma = run_star_with_posteriors(
        FAST, num_stations=4, policy="uniform",
        max_exchanges_per_interval=1,
    )
    assert bool(result.serviced[0].any())
    assert not bool(result.serviced[2].any())
    starved = sigma[2]
    assert torch.all(starved[1:] >= starved[:-1])
    assert starved[-1] > 4.0 * sigma[0][-1]


def test_unit_weights_reproduce_stored_gain():
    result, sigma = run_star_with_posteriors(FAST, num_stations=3)
    phases = phase_matrix(result)
    gain = weighted_gain(phases, torch.ones_like(phases))
    assert torch.allclose(gain, result.array_gain, atol=1e-12)


def test_gate_weight_limits():
    result, sigma = run_star_with_posteriors(FAST, num_stations=3)
    phases = phase_matrix(result)
    wide = posterior_gate_weights(sigma, 3, 1e9)
    assert torch.equal(wide, torch.ones_like(phases))
    shut = posterior_gate_weights(sigma, 3, 0.0)
    gain = weighted_gain(phases, shut)
    # Only the reference transmits: gain must be exactly 1/N^2.
    assert torch.allclose(gain, torch.full_like(gain, 1.0 / 9.0))


def test_soft_weights_bounded_and_ordered():
    result, sigma = run_star_with_posteriors(FAST, num_stations=3)
    soft = posterior_soft_weights(sigma, 3)
    # Pre-acquisition posteriors are enormous (frequency uncertainty
    # couples into predicted phase), so exp(-sigma^2/2) legitimately
    # underflows to exactly 0 - the station is benched.
    assert torch.all(soft >= 0.0) and torch.all(soft <= 1.0)
    # Larger posterior std must never get a larger weight.
    flat_sigma = sigma.flatten()
    flat_weight = soft[1:].flatten()
    order = torch.argsort(flat_sigma)
    sorted_weights = flat_weight[order]
    assert torch.all(sorted_weights[1:] <= sorted_weights[:-1] + 1e-12)


def test_oracle_gate_benches_only_past_threshold():
    phases = torch.tensor(
        [[0.0, 0.0], [0.3, 3.0], [-2.0, 0.1]], dtype=torch.float64
    )
    weights = oracle_gate_weights(phases, math.pi / 2.0)
    expected = torch.tensor(
        [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64
    )
    assert torch.equal(weights, expected)


def test_greedy_never_below_all_in():
    generator = torch.Generator().manual_seed(7)
    phases = (
        torch.rand(6, 40, generator=generator, dtype=torch.float64)
        * 2.0 * math.pi
        - math.pi
    )
    phases[0] = 0.0
    greedy = greedy_oracle_weights(phases)
    all_in = weighted_gain(phases, torch.ones_like(phases))
    gated = weighted_gain(phases, greedy)
    assert torch.all(gated >= all_in - 1e-12)
    assert torch.all(greedy[0] == 1.0)  # reference stays in


def test_evaluation_mask_falls_back_to_tail():
    result, _ = run_star_with_posteriors(
        FAST, num_stations=4, policy="uniform",
        max_exchanges_per_interval=1,
    )
    assert not torch.any(result.steady)  # starved: never steady
    mask = evaluation_mask(result)
    quarter = max(1, FAST.num_iterations // 4)
    assert int(mask.sum()) == quarter
    assert bool(mask[-1])


def test_unknown_membership_variant_rejected():
    phases = torch.zeros(3, 4, dtype=torch.float64)
    sigma = torch.zeros(2, 4, dtype=torch.float64)
    with pytest.raises(ValueError, match="membership"):
        membership_weights("fifo", phases, sigma, 1.0)


def test_unit_weight_detection_matches_original_pipeline():
    # With all-ones weights the gated pipeline must reproduce the
    # original counted-detection numbers EXACTLY: the H0 weight
    # columns come from a separate generator, so the main draw
    # sequence is untouched.
    generator = torch.Generator().manual_seed(3)
    positions = np.array(
        [[0.0, 0.0], [400.0, 100.0], [-300.0, 250.0]]
    )
    residuals = (
        torch.rand(3, 12, generator=generator, dtype=torch.float64) - 0.5
    )
    residuals[0] = 0.0
    targets = np.array([[900.0, 120.0]])
    params = DetectionParams(tx_power_w=0.5)
    kwargs = dict(
        params=params,
        pulse_length=160,
        trials=200,
        h0_trials=4000,
        seed=5,
    )
    original = run_waveform_detection(
        "plain", positions, residuals, targets, **kwargs
    )
    gated = run_gated_waveform_detection(
        "unit", positions, residuals, torch.ones_like(residuals),
        targets, **kwargs
    )
    assert gated.pd_measured == original.pd_measured
    assert gated.measured_pfa == original.measured_pfa


def test_benched_station_removed_from_both_legs():
    # Bench station 2 everywhere: detection must behave as if the
    # array were the remaining stations - and with station 2 carrying
    # a huge residual, benching it must not hurt Pd.
    generator = torch.Generator().manual_seed(11)
    positions = np.array(
        [[0.0, 0.0], [400.0, 100.0], [-300.0, 250.0]]
    )
    residuals = torch.zeros(3, 12, dtype=torch.float64)
    residuals[2] = math.pi  # anti-phase liability
    targets = np.array([[900.0, 120.0]])
    params = DetectionParams(tx_power_w=0.002)
    kwargs = dict(
        params=params,
        pulse_length=160,
        trials=300,
        h0_trials=4000,
        seed=5,
    )
    all_in = run_gated_waveform_detection(
        "all-in", positions, residuals, torch.ones_like(residuals),
        targets, **kwargs
    )
    weights = torch.ones_like(residuals)
    weights[2] = 0.0
    benched = run_gated_waveform_detection(
        "benched", positions, residuals, weights, targets, **kwargs
    )
    assert benched.pd_measured[0] >= all_in.pd_measured[0]
    assert benched.combining_loss_db[0] > all_in.combining_loss_db[0]
