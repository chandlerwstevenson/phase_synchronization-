"""Tests for the pi-ambiguity analysis and study.

The load-bearing test: run_branch_star at its defaults (check every
interval, zero bit error, service every interval) must reproduce
run_scheduled_star's uniform N=2 run bit-for-bit, so every measured
capture/dwell number rests on the same physics as the published runs.
"""

import math

import torch

from ota_sync import SDRSimulationConfig
from ota_sync.scheduled import run_scheduled_star
from pi_ambiguity_analysis import (
    anti_phase_fraction,
    max_check_period,
    p_cross,
)
from pi_ambiguity_study import run_branch_star

FAST = SDRSimulationConfig(num_iterations=18, seed=0, device="cpu")


def test_defaults_reproduce_run_scheduled_star():
    original = run_scheduled_star(FAST, num_stations=2, policy="uniform")
    copy = run_branch_star(FAST, num_stations=2)
    assert torch.equal(copy.residuals, original.residuals)
    assert torch.equal(copy.array_gain, original.array_gain)
    assert torch.equal(copy.steady, original.steady)


def test_p_cross_monotone_and_limits():
    assert p_cross(0.0) == 0.0
    values = [p_cross(s) for s in (0.1, 0.3, 0.6, 1.0, 2.0)]
    assert all(b > a for a, b in zip(values, values[1:]))
    assert p_cross(0.13) < 1e-30  # tcxo-class: effectively never
    assert 0.001 < p_cross(0.64) < 0.1  # sdr-class: routine
    assert max_check_period(0.13, 0.05) > 1e25
    assert max_check_period(0.64, 0.05) < 100.0


def test_renewal_fraction_behaviour():
    assert anti_phase_fraction(0.6, None) == 1.0
    f1 = anti_phase_fraction(0.6, 1)
    f24 = anti_phase_fraction(0.6, 24)
    assert 0.0 < f1 < f24 < 1.0
    # bit errors add dwell
    assert anti_phase_fraction(0.6, 3, bit_error=0.2) > anti_phase_fraction(
        0.6, 3
    )


def test_disabled_check_captures_and_one_bit_rescues():
    # Anti-phase initial offset, zero CFO: without the check the loop
    # must lock at the pi fixed point; the 1-bit check must align it.
    settings = SDRSimulationConfig(
        num_iterations=25, seed=0, device="cpu",
        slave_initial_phase=2.8, slave_initial_frequency_hz=0.0,
    )
    unchecked = run_branch_star(settings, branch_check_every=None)
    checked = run_branch_star(settings, branch_check_every=1)
    assert unchecked.tail_capture()
    assert not checked.tail_capture()
    assert checked.flips >= 1
    # the captured run's loop *believes* it is locked: gain collapses
    assert torch.mean(unchecked.array_gain[-5:]).item() < 0.2
    assert torch.mean(checked.array_gain[-5:]).item() > 0.8
