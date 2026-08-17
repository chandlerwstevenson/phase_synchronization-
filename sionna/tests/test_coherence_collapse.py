"""Tests for the coherence data-collapse study machinery."""

import math
import os

import numpy as np
import pytest

from coherence_collapse_study import (
    _coast_time_physical,
    compute_rhos,
    isotonic_fit,
    load_cache,
    run_cell,
    save_cache,
)


def test_coast_time_solves_the_formula():
    drift, sigma_omega, latency, budget = 4e-8 * 1e6, 2.0 * math.pi * 0.12, 0.05, 0.314
    tau = _coast_time_physical(drift, sigma_omega, latency, budget)
    assert tau > 0.0
    lhs = drift * tau + (sigma_omega * (tau + latency)) ** 2
    assert lhs == pytest.approx(budget**2, rel=1e-9)


def test_coast_time_edge_cases():
    # Latency alone already blows the budget -> zero coast.
    assert _coast_time_physical(1e-3, 100.0, 1.0, 0.1) == 0.0
    # No frequency uncertainty -> pure drift crossing.
    tau = _coast_time_physical(0.01, 0.0, 0.05, 0.1)
    assert tau == pytest.approx(0.1**2 / 0.01)


def test_rho_monotone_in_budget_and_oscillator_class():
    d_loose, _, _ = compute_rhos(4, "ocxo", 0, 1, 0.6, num_iterations=10)
    d_tight, _, _ = compute_rhos(4, "ocxo", 0, 1, 0.2, num_iterations=10)
    assert d_loose < d_tight  # looser budget -> longer coasts -> less demand
    d_ocxo, _, _ = compute_rhos(4, "ocxo", 0, 1, 0.314, num_iterations=10)
    d_sdr, _, _ = compute_rhos(4, "sdr", 0, 1, 0.314, num_iterations=10)
    assert d_ocxo < d_sdr  # better crystals demand less service


def test_isotonic_fit_monotone_and_exact_on_monotone_input():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.1, 0.2, 0.6, 0.9])
    fitted, r2 = isotonic_fit(x, y)
    assert np.allclose(fitted, y)
    assert r2 == pytest.approx(1.0)
    fitted2, r2_2 = isotonic_fit(x, np.array([0.5, 0.1, 0.9, 0.8]))
    order = np.argsort(x)
    assert np.all(np.diff(fitted2[order]) >= -1e-12)
    assert 0.0 <= r2_2 <= 1.0


def test_micro_cell_and_cache_roundtrip(tmp_path):
    rec = run_cell(3, "sdr", 1, 0, 1, 0.314, iterations=8)
    for key in (
        "gain", "rho_phys", "rho_trig", "naive", "worst_rms",
        "frac_met", "airtime", "demand_phys",
    ):
        assert key in rec
    assert 0.0 <= rec["gain"] <= 1.0
    assert rec["rho_phys"] > 0.0
    assert len(rec["tau_phys_s"]) == 2
    path = os.path.join(tmp_path, "cache.json")
    save_cache([rec], path)
    assert load_cache(path) == [rec]
