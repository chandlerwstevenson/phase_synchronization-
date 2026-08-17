"""Tests for the ex-ante coast-time predictor (coast_law.py)."""

import math

import pytest
import torch

from coast_law import (
    dare_posterior,
    link_matrices,
    predict_coast_time,
    station_snr_db,
    supply_demand_ratio,
)
from ota_sync import SDRSimulationConfig
from ota_sync.core import PhaseFrequencyEKF
import ota_sync.scheduled as scheduled_module

SETTINGS = SDRSimulationConfig(
    num_iterations=40, seed=0, device="cpu"
)


def test_closed_form_solves_its_own_equation():
    settings = SETTINGS
    matrices = link_matrices(settings, "tcxo", 20.0, 2.0)
    from coast_law import cycle_posterior

    budget = 0.314
    posterior, _ = cycle_posterior(matrices, budget)
    tau = predict_coast_time(
        "tcxo", 20.0, 1, settings.sync_interval, budget,
        settings=settings, horizon_s=2.0, mode="closed",
        include_latency=True,
    )
    a = matrices.white_fm_rate
    sigma_w2 = posterior[1, 1].item()
    c0 = posterior[0, 0].item()
    horizon = tau + 1 * settings.sync_interval
    residual = a * tau + sigma_w2 * horizon**2 + c0 - budget**2
    assert abs(residual) < 1e-9


def test_monotonicities():
    def tau(profile, budget, latency, mode="cycle", **kw):
        return predict_coast_time(
            profile, 20.0, latency, 0.05, budget,
            settings=SETTINGS, horizon_s=2.0, mode=mode, **kw
        )

    # worse oscillator class -> shorter coast
    assert tau("ocxo", 0.314, 1) > tau("tcxo", 0.314, 1) >= tau(
        "sdr", 0.314, 1
    )
    # higher budget -> longer coast
    assert tau("tcxo", 0.6, 1) > tau("tcxo", 0.2, 1)
    # more latency -> shorter safe coast (closed form, L in horizon)
    assert tau(
        "tcxo", 0.314, 4, mode="closed", include_latency=True
    ) < tau("tcxo", 0.314, 1, mode="closed", include_latency=True)


def test_dare_matches_running_filter():
    # Instrument the star from the outside (gating_study pattern) and
    # compare the every-interval DARE posterior to the running EKF's
    # post-update covariance late in a run serviced every interval
    # (budget far below reachable -> trigger always on).
    records: dict[int, list[torch.Tensor]] = {}

    class CovarianceRecorder(PhaseFrequencyEKF):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._key = len(records)
            records[self._key] = []

        def update(self, measurement):
            super().update(measurement)
            records[self._key].append(self.covariance.clone())

    original = scheduled_module.PhaseFrequencyEKF
    scheduled_module.PhaseFrequencyEKF = CovarianceRecorder
    try:
        scheduled_module.run_scheduled_star(
            SETTINGS,
            num_stations=3,
            policy="scheduled",
            trigger_fraction=1.0,
            budgets_rad=[0.02, 0.02],
        )
    finally:
        scheduled_module.PhaseFrequencyEKF = original

    horizon = SETTINGS.num_iterations * SETTINGS.sync_interval
    for key, station in ((0, 1), (1, 2)):
        assert len(records[key]) > 20  # serviced nearly every interval
        running = records[key][-1]
        snr = station_snr_db(SETTINGS, 3, station)
        predicted = dare_posterior(
            link_matrices(SETTINGS, "custom", snr, horizon)
        )
        for index in ((0, 0), (1, 1)):
            ratio = predicted[index].item() / running[index].item()
            assert 0.8 < ratio < 1.25, (station, index, ratio)


def test_supply_demand_ratio_behaves():
    with pytest.raises(ValueError, match="SNR"):
        supply_demand_ratio(["sdr"], [20.0, 20.0], 2, 1, 0.05, 0.314)
    good = supply_demand_ratio(
        ["ocxo"] * 5, [20.0] * 5, 2, 1, 0.05, 0.314, settings=SETTINGS
    )
    bad = supply_demand_ratio(
        ["sdr"] * 5, [20.0] * 5, 2, 1, 0.05, 0.314, settings=SETTINGS
    )
    assert good > bad  # better fleet -> less demand -> larger rho
    assert bad == pytest.approx(0.4)  # sdr coasts 1 interval: 2/5
