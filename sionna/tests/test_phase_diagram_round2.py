"""Tests for the round-2 phase-diagram blind test.

Load-bearing: (1) the fast closed-form coast cycle reproduces
coast_law.cycle_posterior's cadence exactly; (2) the frozen prediction
path never touches the simulator (blind protocol enforced by
construction); (3) model monotonicities.
"""

import math

import numpy as np
import pytest

import phase_diagram_round2 as r2
from coast_law import cycle_posterior, link_matrices, station_snr_db
from ota_sync import SDRSimulationConfig


def _matrices(profile="tcxo", snr=20.0):
    settings = SDRSimulationConfig(device="cpu")
    return link_matrices(settings, profile, snr, 3.0)


@pytest.mark.parametrize("profile,threshold", [
    ("tcxo", 0.157), ("tcxo", 0.314), ("sdr", 0.314), ("ocxo", 0.157),
])
def test_fast_cycle_matches_coast_law(profile, threshold):
    matrices = _matrices(profile)
    p_ref, m_ref = cycle_posterior(matrices, threshold)
    p_fast, m_fast = r2.fast_cycle(matrices, threshold)
    assert m_fast == m_ref
    assert np.allclose(p_fast, p_ref.numpy(), rtol=1e-8, atol=1e-16)


def test_p00_trajectory_matches_stepwise():
    matrices = _matrices("tcxo")
    Q = matrices.process.numpy()
    T = float(matrices.transition[0, 1])
    P = np.diag([0.01, 4.0])
    trajectory = r2._p00_trajectory(
        P, float(Q[0, 0]), float(Q[1, 1]), T, 25
    )
    F = np.array([[1.0, T], [0.0, 1.0]])
    Qd = np.diag([float(Q[0, 0]), float(Q[1, 1])])
    stepped = P.copy()
    for j in range(25):
        stepped = F @ stepped @ F.T + Qd
        assert trajectory[j] == pytest.approx(stepped[0, 0], rel=1e-12)


def test_prediction_never_calls_simulator(monkeypatch):
    import ota_sync.scheduled as scheduled_module

    def _forbidden(*args, **kwargs):
        raise AssertionError("blind protocol violated: simulator called "
                             "during prediction")

    monkeypatch.setattr(
        scheduled_module, "run_scheduled_star", _forbidden
    )
    monkeypatch.setattr(r2, "run_scheduled_star", _forbidden)
    result = r2.predict_condition(
        6, r2.fleet_r2("tcxo", 6), seed=3, latency=1,
        budget=0.25, capacity=3,
    )
    assert 0.0 <= result["gain_pred"] <= 1.0
    assert result["kappa"] >= 1.0


def test_gain_monotone_in_capacity():
    profiles = r2.fleet_r2("tcxo", 8)
    gains = [
        r2.predict_condition(8, profiles, 3, 1, 0.25, c)["gain_pred"]
        for c in (1, 3, 5, 7)
    ]
    assert all(b >= a - 1e-9 for a, b in zip(gains, gains[1:]))


def test_latency_lowers_plateau():
    profiles = r2.fleet_r2("tcxo", 8)
    low = r2.predict_condition(8, profiles, 3, 1, 0.25, 7)["gain_pred"]
    high = r2.predict_condition(8, profiles, 3, 4, 0.25, 7)["gain_pred"]
    assert high < low


def test_sdr_member_raises_phi_and_lowers_gain():
    ocxo = r2.predict_condition(
        8, r2.fleet_r2("ocxo", 8), 3, 1, 0.25, 7
    )
    sdr1 = r2.predict_condition(
        8, r2.fleet_r2("sdr1", 8), 3, 1, 0.25, 7
    )
    assert sdr1["phi_max"] > ocxo["phi_max"]
    assert sdr1["gain_pred"] < ocxo["gain_pred"]


def test_snr_helper_consistent_with_geometry():
    settings = SDRSimulationConfig(device="cpu", seed=3)
    snr = station_snr_db(settings, 8, 1)
    assert math.isfinite(snr)
