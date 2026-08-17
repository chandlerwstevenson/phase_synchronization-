"""Tests for the hybrid coherent/noncoherent combiner
(hybrid_combiner_study.py).

Load-bearing anchors: with nobody benched the two-tier statistic must
reproduce the existing pipelines EXACTLY (same draws, same numbers);
the bench-set handling is then checked at both extremes and on the
anti-phase rescue case.
"""

import math

import numpy as np
import pytest
import torch

from detection import DetectionParams
from gating_study import run_gated_waveform_detection
from hybrid_combiner_study import (
    VARIANTS,
    run_hybrid_waveform_detection,
    variant_weights,
)

POSITIONS = np.array([[0.0, 0.0], [400.0, 100.0], [-300.0, 250.0]])
TARGETS = np.array([[900.0, 120.0]])
FAST = dict(pulse_length=160, trials=200, h0_trials=4000, seed=5)


def _residuals(generator_seed=3):
    generator = torch.Generator().manual_seed(generator_seed)
    residuals = (
        torch.rand(3, 12, generator=generator, dtype=torch.float64) - 0.5
    )
    residuals[0] = 0.0
    return residuals


def test_unit_weights_discard_reproduces_gated_pipeline():
    residuals = _residuals()
    params = DetectionParams(tx_power_w=0.5)
    ones = torch.ones_like(residuals)
    reference = run_gated_waveform_detection(
        "ref", POSITIONS, residuals, ones, TARGETS, params=params, **FAST
    )
    hybrid = run_hybrid_waveform_detection(
        "unit", POSITIONS, residuals, ones, "discard", TARGETS,
        params=params, **FAST
    )
    assert hybrid.pd_measured == reference.pd_measured
    assert hybrid.measured_pfa == reference.measured_pfa


def test_unit_weights_noncoherent_equals_discard():
    # Empty bench set: the noncoherent tier contributes nothing, so
    # both modes are the same statistic on the same draws.
    residuals = _residuals()
    params = DetectionParams(tx_power_w=0.5)
    ones = torch.ones_like(residuals)
    discard = run_hybrid_waveform_detection(
        "d", POSITIONS, residuals, ones, "discard", TARGETS,
        params=params, **FAST
    )
    noncoh = run_hybrid_waveform_detection(
        "n", POSITIONS, residuals, ones, "noncoherent", TARGETS,
        params=params, **FAST
    )
    assert discard.pd_measured == noncoh.pd_measured
    assert discard.measured_pfa == noncoh.measured_pfa


def test_all_benched_discard_detects_nothing():
    residuals = _residuals()
    params = DetectionParams(tx_power_w=0.5)
    zeros = torch.zeros_like(residuals)
    result = run_hybrid_waveform_detection(
        "z", POSITIONS, residuals, zeros, "discard", TARGETS,
        params=params, **FAST
    )
    assert result.pd_measured[0] == 0.0


def test_all_benched_noncoherent_still_detects():
    # Full noncoherent fusion keeps the echo power (phase-blind), so
    # at a comfortable power it must detect essentially everything
    # while the discard combiner (previous test) sees nothing.
    residuals = _residuals()
    params = DetectionParams(tx_power_w=0.5)
    zeros = torch.zeros_like(residuals)
    result = run_hybrid_waveform_detection(
        "nc", POSITIONS, residuals, zeros, "noncoherent", TARGETS,
        params=params, **FAST
    )
    assert result.pd_measured[0] > 0.9
    assert result.measured_pfa <= 3.0 * result.threshold_pfa


def test_hybrid_rescues_anti_phase_station():
    # Station 2 anti-phase: coherent all-in suffers cancellation; the
    # hybrid combiner benches it into the noncoherent tier and must do
    # at least as well as all-in coherent.
    residuals = torch.zeros(3, 12, dtype=torch.float64)
    residuals[2] = math.pi
    params = DetectionParams(tx_power_w=0.002)
    ones = torch.ones_like(residuals)
    weights = torch.ones_like(residuals)
    weights[2] = 0.0
    all_in = run_hybrid_waveform_detection(
        "all", POSITIONS, residuals, ones, "discard", TARGETS,
        params=params, **FAST
    )
    hybrid = run_hybrid_waveform_detection(
        "hyb", POSITIONS, residuals, weights, "noncoherent", TARGETS,
        params=params, **FAST
    )
    assert hybrid.pd_measured[0] >= all_in.pd_measured[0]


def test_variant_table_and_weights_sources():
    phases = torch.zeros(3, 4, dtype=torch.float64)
    sigma = torch.zeros(2, 4, dtype=torch.float64)
    assert set(VARIANTS) == {
        "all-in", "gate-discard", "hybrid-post", "hybrid-oracle",
        "noncoh-all",
    }
    assert torch.equal(
        variant_weights("ones", phases, sigma, 1.0), torch.ones_like(phases)
    )
    assert torch.equal(
        variant_weights("zeros", phases, sigma, 1.0),
        torch.zeros_like(phases),
    )
    with pytest.raises(ValueError, match="membership source"):
        variant_weights("fifo", phases, sigma, 1.0)
