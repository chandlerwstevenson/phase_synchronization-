"""Tests for the theory N-scaling study (theory_nscaling_study.py)."""

import json
import math

import pytest

from phase_diagram_round2 import fleet_r2, predict_curve
from theory_nscaling_study import (
    CURVES,
    freeze_nscaling_predictions,
    load_runs,
)


def test_curves_are_fresh_n_values():
    # The whole point is extrapolation: every curve must use an N the
    # round-2 blind test never ran (its CURVES used 8 and 12).
    for n, _, _, latency in CURVES:
        assert n in (16, 20)
        assert latency == 1  # clean regime only


def test_predicted_gain_monotone_in_capacity():
    per_cap, plateau, knee = predict_curve(
        8, fleet_r2("tcxo", 8), (3,), 1, 0.25,
        [1, 2, 3, 4, 7], 60,
    )
    values = [per_cap[c] for c in sorted(per_cap)]
    for lower, higher in zip(values, values[1:]):
        assert higher >= lower - 1e-9
    assert 0.0 < plateau <= 1.0
    if knee is not None:
        assert per_cap[knee] >= 0.90


def test_freeze_produces_scoreable_curves():
    frozen = freeze_nscaling_predictions(
        curves=[(16, "ocxo", 0.25, 1)], seeds=(3,)
    )
    assert len(frozen) == 1
    curve = frozen[0]
    for key in ("n", "fleet", "budget", "latency", "capacities",
                "pred_gain", "plateau", "knee", "reachable"):
        assert key in curve
    assert set(curve["pred_gain"]) == set(curve["capacities"])
    assert all(0.0 <= g <= 1.0 for g in curve["pred_gain"].values())


def test_run_cache_roundtrip(tmp_path):
    path = tmp_path / "runs.json"
    runs = [{
        "n": 16, "fleet": "tcxo", "capacity": 3, "seed": 3,
        "latency": 1, "budget": 0.25, "gain": 0.97,
    }]
    path.write_text(json.dumps(runs))
    loaded = load_runs(str(path))
    assert loaded == runs
    assert load_runs(str(tmp_path / "missing.json")) == []
