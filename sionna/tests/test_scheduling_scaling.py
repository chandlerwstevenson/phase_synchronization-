"""Tests for the scheduling N-scaling study machinery."""

import json

import pytest

import scheduling_scaling_study as study


def test_capacity_levels_match_published_convention():
    # N=10 must reproduce the multi_metric_study capacities (2 and 4),
    # anchoring the N sweep to the published table.
    assert study.capacity_for(10, "contended") == 2
    assert study.capacity_for(10, "comfortable") == 4
    assert study.capacity_for(6, "contended") == 1
    assert study.capacity_for(6, "comfortable") == 2
    assert study.capacity_for(14, "contended") == 3
    assert study.capacity_for(14, "comfortable") == 6
    assert study.capacity_for(20, "contended") == 4
    assert study.capacity_for(20, "comfortable") == 9


def test_capacity_never_below_one():
    assert study.capacity_for(2, "contended") == 1


def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "CACHE", str(tmp_path / "cache.json"))
    cache = study.load_cache()
    assert cache == {"cells": [], "demand": []}
    cache["cells"].append({"n": 6, "level": "contended", "policy": "uniform"})
    study.save_cache(cache)
    again = study.load_cache()
    assert again["cells"][0]["n"] == 6
    with open(study.CACHE) as handle:
        assert json.load(handle) == again


def test_report_handles_empty_cache(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(study, "CACHE", str(tmp_path / "cache.json"))
    study.report()
    assert "no cells cached" in capsys.readouterr().out


def test_unknown_level_rejected():
    with pytest.raises(KeyError):
        study.capacity_for(10, "luxurious")
