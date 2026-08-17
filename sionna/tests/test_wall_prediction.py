"""Tests for the blind wall-prediction study (wall_prediction_study.py).

The invariants that make the ex-ante claim honest: predictions are
computable with the simulator disabled, coast times order by
oscillator class quality, and the closed form prices latency.
"""

import math

import pytest

import wall_prediction_study as wps
from ota_sync import SDRSimulationConfig

SETTINGS = SDRSimulationConfig(num_iterations=10, seed=0, device="cpu")


def _coast(profile: str) -> int:
    _, steps, _ = wps.predict_demand(
        SETTINGS, 3, [profile] * 3, seed=0
    )
    return min(steps)


def test_coast_orders_by_oscillator_class():
    # Better crystals must coast longer: ocxo > tcxo > sdr.
    ocxo, tcxo, sdr = _coast("ocxo"), _coast("tcxo"), _coast("sdr")
    assert ocxo > tcxo > sdr >= 1


def test_closed_form_prices_latency():
    q_phase, sigma_omega, target, interval = 2e-3, 0.8, 0.157, 0.05
    taus = [
        wps.closed_form_coast(q_phase, sigma_omega, target, interval, lat)
        for lat in (0, 1, 3)
    ]
    assert taus[0] > taus[1] > taus[2] > 0.0


def test_predictions_need_no_simulation(monkeypatch):
    # The ex-ante guarantee: with the simulator disabled, every
    # prediction still computes.
    def _forbidden(*args, **kwargs):
        raise AssertionError("prediction touched the simulator")

    monkeypatch.setattr(wps, "run_scheduled_star", _forbidden)
    predictions = wps.predict_all()
    assert set(predictions) == {"p1", "p2", "p3"}
    for entry in predictions["p1"].values():
        assert entry["demand"] > 0.0
        assert entry["knee"] >= 1


def test_demand_grows_with_stations():
    small, _, _ = wps.predict_demand(SETTINGS, 4, ["sdr"] * 4, seed=0)
    large, _, _ = wps.predict_demand(SETTINGS, 8, ["sdr"] * 8, seed=0)
    assert large > small


def test_knee_is_ceiling_of_demand():
    predictions = wps.predict_all()
    for entry in predictions["p1"].values():
        assert entry["knee"] == math.ceil(entry["demand"] - 1e-9)
