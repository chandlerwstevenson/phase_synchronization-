"""Sanity tests for the five-metric comparison layer."""

import math

import numpy as np
import torch

from multi_metric_study import (
    mean_gain_from_matrix,
    net_throughput,
    spectral_efficiency_draws,
    summarize_se,
    user_amplitudes,
)

POSITIONS = np.array([[0.0, 0.0], [300.0, 0.0], [-300.0, 100.0]])
USER = np.array([[1200.0, 150.0]])


def test_reference_amplitude_matches_link_budget():
    # A station exactly at the reference distance must give the
    # reference signal-to-noise ratio on its own (20 dB -> amp 10).
    amps = user_amplitudes(np.array([[500.0, 0.0]]), np.zeros(2))
    assert torch.allclose(amps, torch.tensor([10.0], dtype=torch.float64))


def test_perfect_sync_beats_scrambled_phases():
    aligned = torch.zeros(3, 200, dtype=torch.float64)
    generator = torch.Generator().manual_seed(0)
    scrambled = (
        torch.rand(3, 200, generator=generator, dtype=torch.float64)
        * 2.0 * math.pi - math.pi
    )
    scrambled[0] = 0.0
    se_aligned = summarize_se(
        spectral_efficiency_draws(POSITIONS, aligned, USER)
    )
    se_scrambled = summarize_se(
        spectral_efficiency_draws(POSITIONS, scrambled, USER)
    )
    assert se_aligned[0] > se_scrambled[0]
    assert se_aligned[1] > se_scrambled[1]
    assert mean_gain_from_matrix(aligned) > mean_gain_from_matrix(scrambled)


def test_net_throughput_prices_airtime():
    assert net_throughput(8.0, 0.25) == 6.0
    assert net_throughput(8.0, 0.0) == 8.0
    # Sync demand beyond the frame leaves no airtime for data.
    assert net_throughput(8.0, 1.5) == 0.0


def test_gain_of_perfect_alignment_is_one():
    aligned = torch.zeros(4, 50, dtype=torch.float64)
    assert abs(mean_gain_from_matrix(aligned) - 1.0) < 1e-12
