"""Tests for the membership array-size scaling study."""

import torch

from membership_scaling_study import (
    CHEAP_SEEDS,
    capacity_for,
    detect_seeds_for,
    matched_power_w,
    score_stations,
)
from metrics_membership_study import METHODS


def test_capacity_rule_matches_documented_values():
    assert [capacity_for(n) for n in (6, 10, 14, 20)] == [1, 2, 3, 4]


def test_matched_power_anchors_at_n10():
    assert abs(matched_power_w(10) - 0.05) < 1e-12
    # Larger arrays need less power for the same perfect-sync budget.
    assert matched_power_w(20) < matched_power_w(10) < matched_power_w(6)


def test_detect_seed_policy():
    assert detect_seeds_for(6) == (0, 1)
    assert detect_seeds_for(14) == (0,)


def test_micro_grid_produces_all_cells(tmp_path, monkeypatch):
    import membership_scaling_study as study

    monkeypatch.setattr(
        study, "CACHE_PATH", str(tmp_path / "cache.json")
    )
    monkeypatch.setattr(study, "CHEAP_SEEDS", (0,))
    monkeypatch.setattr(study, "detect_seeds_for", lambda n: (0,))
    cache = study.score_stations(
        4, iterations=12, trials=40, h0_trials=1500, cache={}
    )
    for method in METHODS:
        assert f"4/0/{method}/gain" in cache
        assert f"4/0/{method}/net" in cache
        pd = cache[f"4/0/{method}/pd@matched"]
        assert len(pd) == 2 and all(0.0 <= v <= 1.0 for v in pd)
    assert 0.0 < cache["4/0/airtime"] <= 1.0
