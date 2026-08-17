"""Tests for 1-bit opportunistic membership
(opportunistic_membership_study.py). The load-bearing fact: with
error-free bits the onebit rule IS gating_study's oracle gate - the
'genie' is implementable with one feedback bit per station/interval."""

import math

import torch

from gating_study import (
    oracle_gate_weights,
    phase_matrix,
    run_star_with_posteriors,
    weighted_gain,
)
from opportunistic_membership_study import (
    QUADRANT_WEIGHTS,
    alignment_bits,
    onebit_hysteresis_weights,
    onebit_weights,
    quantized2_weights,
)
from ota_sync import SDRSimulationConfig

FAST = SDRSimulationConfig(num_iterations=18, seed=0, device="cpu")


def _real_phases():
    result, _ = run_star_with_posteriors(
        FAST, num_stations=4, policy="uniform",
        max_exchanges_per_interval=1,
    )
    return phase_matrix(result)


def test_error_free_onebit_equals_oracle_gate():
    phases = _real_phases()
    bits = alignment_bits(phases, 0.0, None)
    assert torch.equal(
        onebit_weights(bits), oracle_gate_weights(phases, math.pi / 2.0)
    )


def test_bit_errors_degrade_monotonically():
    phases = _real_phases()
    oracle = torch.mean(
        weighted_gain(phases, oracle_gate_weights(phases, math.pi / 2.0))
    ).item()

    def gain_at(eps):
        values = []
        for seed in range(8):
            generator = torch.Generator().manual_seed(seed)
            weights = onebit_weights(
                alignment_bits(phases, eps, generator)
            )
            values.append(
                torch.mean(weighted_gain(phases, weights)).item()
            )
        return sum(values) / len(values)

    g_small, g_half = gain_at(0.1), gain_at(0.5)
    assert g_small < oracle + 1e-12
    assert g_half < g_small  # heavier bit noise, lower gain


def test_hysteresis_needs_two_aligned_bits_to_reenter():
    # Alternating aligned/misaligned: after the first bench the streak
    # never reaches 2, so the station must stay benched forever.
    bits = torch.tensor(
        [[True] * 6, [True, False, True, False, True, False]]
    )
    weights = onebit_hysteresis_weights(bits)
    assert torch.equal(
        weights[1], torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    # Two consecutive aligned bits do re-enter.
    bits = torch.tensor([[True] * 5, [True, False, True, True, True]])
    weights = onebit_hysteresis_weights(bits)
    assert torch.equal(
        weights[1], torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0])
    )


def test_quantized2_weights_follow_quadrants():
    phases = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.1, 1.0, 2.0, 3.0],  # quadrants 0,1,2,3
        ],
        dtype=torch.float64,
    )
    weights = quantized2_weights(phases)
    expected = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [
                QUADRANT_WEIGHTS[0],
                QUADRANT_WEIGHTS[1],
                QUADRANT_WEIGHTS[2],
                QUADRANT_WEIGHTS[3],
            ],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(weights, expected)
