"""Tests for the scheduler extensions (contention baselines,
heterogeneous fleets, budget re-targeting, multi-fidelity pilots).

The first test is the load-bearing one: with every extension knob at
its default, the extended run_scheduled_star must reproduce the
original scheduled/uniform behavior exactly, so every previously
published scheduling number survives the change.
"""

import math

import pytest
import torch

from ota_sync import SDRSimulationConfig
from ota_sync.scheduled import SCHEDULER_POLICIES, run_scheduled_star

FAST = SDRSimulationConfig(num_iterations=18, seed=0, device="cpu")


def _run(policy="scheduled", **kwargs):
    return run_scheduled_star(
        FAST, num_stations=3, policy=policy, **kwargs
    )


def test_default_paths_unchanged_by_extensions():
    # Golden digests recorded from the pre-extension implementation
    # (equiv_check.py against git HEAD b9101df): num_stations=4,
    # 25 iterations, seed 0.
    settings = SDRSimulationConfig(num_iterations=25, seed=0, device="cpu")
    scheduled = run_scheduled_star(settings, num_stations=4)
    uniform = run_scheduled_star(settings, num_stations=4, policy="uniform")
    assert scheduled.mean_array_gain == pytest.approx(0.988227, abs=5e-6)
    assert scheduled.airtime_used_fraction == pytest.approx(
        0.275386, abs=5e-6
    )
    assert uniform.mean_array_gain == pytest.approx(0.994492, abs=5e-6)
    assert uniform.airtime_used_fraction == pytest.approx(0.573720, abs=5e-6)
    assert scheduled.serviced_micro is None


@pytest.mark.parametrize("policy", ["roundrobin", "whittle", "oracle"])
def test_new_policies_lock_and_hold_gain(policy):
    result = _run(policy=policy)
    assert result.mean_array_gain > 0.9
    assert torch.any(result.steady)
    # every station must be serviced at least once (no starvation)
    assert bool(result.serviced.any(dim=1).all())


def test_unknown_policy_rejected():
    with pytest.raises(ValueError, match="policy"):
        _run(policy="fifo")
    assert "scheduled" in SCHEDULER_POLICIES


def test_informed_policies_use_less_airtime_than_roundrobin():
    roundrobin = _run(policy="roundrobin")
    whittle = _run(policy="whittle")
    assert (
        whittle.airtime_used_fraction < roundrobin.airtime_used_fraction
    )


def test_heterogeneous_profiles_shape_and_validation():
    with pytest.raises(ValueError, match="one oscillator profile"):
        _run(oscillator_profiles=["ocxo", "tcxo"])
    with pytest.raises(ValueError, match="unknown oscillator profile"):
        _run(oscillator_profiles=["ocxo", "tcxo", "rubidium"])
    result = _run(oscillator_profiles=["ocxo", "ocxo", "tcxo"])
    assert torch.any(result.steady)
    # the tcxo station's residual should be the worse of the two
    rms = result.station_steady_rms
    assert rms[1] > rms[0]


def test_scheduler_services_the_noisy_station_more():
    result = _run(oscillator_profiles=["ocxo", "ocxo", "tcxo"])
    rates = result.serviced.to(dtype=torch.float64).mean(dim=1)
    assert rates[1] > rates[0]


def test_budget_updates_validation_and_effect():
    with pytest.raises(ValueError, match="budget_updates"):
        _run(budget_updates={4: [0.3]})
    tight_then_loose = _run(
        budgets_rad=[0.05, 0.05],
        budget_updates={9: [1.5, 1.5]},
    )
    loose_throughout = _run(budgets_rad=[1.5, 1.5])
    # after the loosening lands, service demand should drop toward the
    # loose-budget run's level
    tail = slice(11, None)
    assert (
        tight_then_loose.serviced[:, tail].sum()
        <= tight_then_loose.serviced[:, slice(0, 9)].sum()
    )
    assert torch.any(loose_throughout.steady)


def test_multi_fidelity_reduces_airtime_at_matching_gain():
    settings = SDRSimulationConfig(
        num_iterations=40, seed=0, device="cpu"
    )
    full = run_scheduled_star(
        settings, num_stations=3, policy="uniform"
    )
    micro = run_scheduled_star(
        settings, num_stations=3, policy="uniform", multi_fidelity=True
    )
    assert micro.serviced_micro is not None
    assert int(micro.serviced_micro.sum()) > 0
    assert micro.airtime_used_fraction < 0.6 * full.airtime_used_fraction
    assert micro.mean_array_gain > full.mean_array_gain - 0.02
    # micro services are a subset of services
    assert bool((micro.serviced | ~micro.serviced_micro).all())


def test_residual_matrix_window_slicing():
    result = _run(policy="uniform")
    full = result.residual_matrix()
    window = result.residual_matrix(interval_slice=slice(10, 18))
    assert window.shape[0] == full.shape[0]
    assert 0 < window.shape[1] <= min(8, full.shape[1])
    assert math.isfinite(float(window.abs().sum()))


def test_capacity_starves_uniform_but_not_roundrobin():
    capped_uniform = _run(policy="uniform", max_exchanges_per_interval=1)
    capped_rr = _run(policy="roundrobin", max_exchanges_per_interval=1)
    # uniform always picks the same first link: station 2 never serviced
    assert not bool(capped_uniform.serviced[1].any())
    assert bool(capped_rr.serviced.any(dim=1).all())
