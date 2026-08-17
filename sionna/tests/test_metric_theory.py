"""Tests for the metric-prediction layer (metric_theory_study.py)."""

import math

import numpy as np

from metric_theory_study import (
    EDGE_SNR_PER_STATION,
    curve_summary,
    metric_draws,
    perfect_metrics,
    score_metrics,
    spearman,
)


def test_perfect_sync_reproduces_closed_form_spectral_efficiency():
    # Zero residuals, equal amplitudes: the coherent sum is exactly
    # N^2 x per-station SNR, so log2(1 + N^2 * snr1) must come out.
    n = 8
    scores = score_metrics(np.zeros((1, n)), n, seed=3, airtime=0.0)
    expected = math.log2(1.0 + n**2 * EDGE_SNR_PER_STATION)
    assert abs(scores["se_edge"] - expected) < 1e-12
    assert scores["gain"] == 1.0


def test_concavity_mean_of_log_below_log_of_mean():
    # Jensen: averaging log2(1+SNR) over draws must sit at or below
    # plugging the mean SNR into the log.
    rng = np.random.default_rng(0)
    n = 6
    phases = np.zeros((5000, n))
    phases[:, 1:] = rng.normal(0.0, 0.8, (5000, n - 1))
    amps = np.full(n, math.sqrt(EDGE_SNR_PER_STATION))
    draws = metric_draws(phases, amps)
    field = np.sum(amps[None, :] * np.exp(1j * phases), axis=1)
    naive = math.log2(1.0 + float(np.mean(np.abs(field) ** 2)))
    assert float(np.mean(draws)) <= naive + 1e-12
    assert float(np.mean(draws)) < naive  # strictly below with spread


def test_monte_carlo_gain_matches_expected_phasor_formula():
    # The model's closed form [(sum w)^2 + sum(1-w^2)] / N^2 with
    # w = exp(-s^2/2) is the independent-phase expectation of the
    # Monte Carlo gain estimator.
    rng = np.random.default_rng(1)
    n, s = 5, np.array([0.3, 0.9, 1.5, 2.2])
    phases = np.zeros((200000, n))
    phases[:, 1:] = rng.normal(0.0, s[None, :], (200000, n - 1))
    gain_mc = float(np.mean(
        np.abs(np.sum(np.exp(1j * phases), axis=1)) ** 2 / n**2
    ))
    w = np.concatenate(([1.0], np.exp(-0.5 * s**2)))
    gain_formula = (np.sum(w) ** 2 + np.sum(1.0 - w**2)) / n**2
    assert abs(gain_mc - gain_formula) < 0.01


def test_spearman_limits():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(a, 10 * a + 3) == 1.0
    assert spearman(a, -a) == -1.0


def test_curve_summary_knee_rule():
    per_capacity = {1: 0.3, 2: 0.7, 3: 0.95, 4: 0.99}
    plateau, knee = curve_summary(per_capacity, perfect=1.0)
    assert plateau == 0.99 and knee == 3
    plateau, knee = curve_summary({1: 0.2, 2: 0.4}, perfect=1.0)
    assert knee is None


def test_perfect_metrics_detection_range_positive():
    scores = perfect_metrics(8, seed=3)
    assert scores["range_m"] > 1000.0  # N=8 perfect sync reaches km scale
    assert scores["net_throughput"] == scores["se_edge"]  # zero airtime
