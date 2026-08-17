"""Tests for the split-weight detection pipeline
(gating_dissociation_study.py).

Load-bearing: with tx_weights == rx_weights the split pipeline must
reproduce gating_study.run_gated_waveform_detection EXACTLY (same
generator discipline), which itself reproduces detection/waveform.py
at unit weights - so the whole chain is anchored to the original
counted-detection numbers.
"""

import math

import numpy as np
import torch

from detection import DetectionParams
from detection.waveform import run_waveform_detection
from gating_dissociation_study import run_split_waveform_detection
from gating_study import run_gated_waveform_detection

POSITIONS = np.array([[0.0, 0.0], [400.0, 100.0], [-300.0, 250.0]])
TARGETS = np.array([[900.0, 120.0]])
KWARGS = dict(
    params=DetectionParams(tx_power_w=0.5),
    pulse_length=160,
    trials=200,
    h0_trials=4000,
    seed=5,
)


def _residuals(seed=3):
    generator = torch.Generator().manual_seed(seed)
    residuals = (
        torch.rand(3, 12, generator=generator, dtype=torch.float64) - 0.5
    )
    residuals[0] = 0.0
    return residuals


def test_equal_weights_reproduce_gated_pipeline():
    residuals = _residuals()
    generator = torch.Generator().manual_seed(21)
    weights = torch.rand(
        3, 12, generator=generator, dtype=torch.float64
    )
    weights[0] = 1.0
    gated = run_gated_waveform_detection(
        "gated", POSITIONS, residuals, weights, TARGETS, **KWARGS
    )
    split = run_split_waveform_detection(
        "split", POSITIONS, residuals, weights, weights, TARGETS, **KWARGS
    )
    assert split.pd_measured == gated.pd_measured
    assert split.measured_pfa == gated.measured_pfa
    assert split.combining_loss_db == gated.combining_loss_db


def test_unit_weights_reproduce_original_pipeline():
    residuals = _residuals()
    ones = torch.ones_like(residuals)
    original = run_waveform_detection(
        "plain", POSITIONS, residuals, TARGETS, **KWARGS
    )
    split = run_split_waveform_detection(
        "unit", POSITIONS, residuals, ones, ones, TARGETS, **KWARGS
    )
    assert split.pd_measured == original.pd_measured
    assert split.measured_pfa == original.measured_pfa


def test_rx_only_keeps_transmit_field():
    # rx-only gating must not change the transmit combining loss:
    # the beam at the target is untouched, only the combiner changes.
    residuals = _residuals()
    ones = torch.ones_like(residuals)
    benched = ones.clone()
    benched[2] = 0.0
    all_in = run_split_waveform_detection(
        "all-in", POSITIONS, residuals, ones, ones, TARGETS, **KWARGS
    )
    rx_only = run_split_waveform_detection(
        "rx-only", POSITIONS, residuals, ones, benched, TARGETS, **KWARGS
    )
    assert rx_only.combining_loss_db == all_in.combining_loss_db


def test_tx_only_keeps_h0_threshold():
    # tx-only gating leaves the receive membership all-in, so the H0
    # calibration must land on the identical measured Pfa as all-in
    # (same noise draws, same unit combiner weights).
    residuals = _residuals()
    ones = torch.ones_like(residuals)
    benched = ones.clone()
    benched[2] = 0.0
    all_in = run_split_waveform_detection(
        "all-in", POSITIONS, residuals, ones, ones, TARGETS, **KWARGS
    )
    tx_only = run_split_waveform_detection(
        "tx-only", POSITIONS, residuals, benched, ones, TARGETS, **KWARGS
    )
    assert tx_only.measured_pfa == all_in.measured_pfa


def test_tx_benching_antiphase_station_helps():
    residuals = torch.zeros(3, 12, dtype=torch.float64)
    residuals[2] = math.pi
    ones = torch.ones_like(residuals)
    benched = ones.clone()
    benched[2] = 0.0
    kwargs = dict(KWARGS, params=DetectionParams(tx_power_w=0.002))
    all_in = run_split_waveform_detection(
        "all-in", POSITIONS, residuals, ones, ones, TARGETS, **kwargs
    )
    tx_only = run_split_waveform_detection(
        "tx-only", POSITIONS, residuals, benched, ones, TARGETS, **kwargs
    )
    assert tx_only.pd_measured[0] >= all_in.pd_measured[0]
    assert tx_only.combining_loss_db[0] > all_in.combining_loss_db[0]


def test_weight_shape_mismatch_rejected():
    residuals = _residuals()
    ones = torch.ones_like(residuals)
    bad = torch.ones(3, 5, dtype=torch.float64)
    for tx_w, rx_w in ((bad, ones), (ones, bad)):
        try:
            run_split_waveform_detection(
                "bad", POSITIONS, residuals, tx_w, rx_w, TARGETS, **KWARGS
            )
        except ValueError as error:
            assert "align" in str(error)
        else:
            raise AssertionError("shape mismatch was not rejected")
