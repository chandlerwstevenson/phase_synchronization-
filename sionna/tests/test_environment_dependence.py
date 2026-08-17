"""Tests for the environment-dependence machinery (RT-tap injection).

The anchors: the CIR-to-taps converter behaves correctly in degenerate
configurations, the injection context patches and restores cleanly and
actually lands the taps in every link (mirrors included), and a
non-default TDL letter runs end to end.
"""

import math

import numpy as np
import torch

import hybrid_calibration.hybrid as hybrid_module
import ota_sync.coherent as coherent_module
from environment_dependence_study import (
    _InjectedRadioLink,
    cir_to_frozen_taps,
    injected_channel,
    run_cell,
)
from ota_sync import SDRSimulationConfig
from ota_sync.sdr import SDRRadioLink, make_sync_preamble
from ota_sync.core import resolve_device
from sionna.phy.channel import time_lag_discrete_time_channel


FAST = SDRSimulationConfig(num_iterations=8, seed=0, device="cpu")


def test_single_path_converter_is_unit_energy_delta():
    taps, dropped = cir_to_frozen_taps(
        np.array([1.0 + 0.0j]), np.array([0.0]), FAST
    )
    assert dropped == 0
    energy = torch.sum(torch.abs(taps) ** 2).item()
    assert math.isclose(energy, 1.0, rel_tol=1e-9)
    l_min, _ = time_lag_discrete_time_channel(
        FAST.sample_rate, FAST.maximum_channel_delay_s
    )
    peak = torch.argmax(torch.abs(taps)).item()
    assert peak == -l_min  # first arrival gated to lag zero


def test_converter_drops_paths_outside_window_and_regates():
    # Absolute delays: gating subtracts the first arrival, so a 2 us
    # spread survives the 3 us window; a 10 us straggler is dropped.
    taps, dropped = cir_to_frozen_taps(
        np.array([1.0, 0.5, 0.25], dtype=np.complex128),
        np.array([5e-6, 7e-6, 15e-6]),
        FAST,
    )
    assert dropped == 1
    assert math.isclose(
        torch.sum(torch.abs(taps) ** 2).item(), 1.0, rel_tol=1e-9
    )


def test_injection_lands_in_forward_and_mirror_links():
    taps, _ = cir_to_frozen_taps(np.array([1.0 + 0j]), np.array([0.0]), FAST)
    device = resolve_device("cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(1)
    preamble = make_sync_preamble(FAST, device)
    _InjectedRadioLink.injected_taps = taps
    try:
        forward = _InjectedRadioLink(FAST, preamble, device, generator)
        mirror = _InjectedRadioLink(
            FAST, preamble, device, generator, mirror_of=forward
        )
    finally:
        _InjectedRadioLink.injected_taps = None
    flat = forward.channel_taps.reshape(
        -1, forward.channel_taps.shape[-1]
    )
    expected = taps.to(forward.channel_taps.dtype)
    assert torch.allclose(flat[0], expected, atol=1e-12)
    assert torch.allclose(flat[-1], expected, atol=1e-12)  # frozen in time
    assert mirror.channel_taps.data_ptr() == forward.channel_taps.data_ptr()


def test_context_manager_restores_originals():
    taps, _ = cir_to_frozen_taps(np.array([1.0 + 0j]), np.array([0.0]), FAST)
    with injected_channel(taps):
        assert hybrid_module.SDRRadioLink is _InjectedRadioLink
        assert coherent_module.SDRRadioLink is _InjectedRadioLink
    assert hybrid_module.SDRRadioLink is SDRRadioLink
    assert coherent_module.SDRRadioLink is SDRRadioLink
    assert _InjectedRadioLink.injected_taps is None
    with injected_channel(None):  # no-op path
        assert hybrid_module.SDRRadioLink is SDRRadioLink


def test_nlos_tdl_letter_runs_end_to_end():
    settings = SDRSimulationConfig(
        num_iterations=8, seed=0, device="cpu", tdl_model="A"
    )
    cell = run_cell(settings, (4,))
    rms = cell["K4"][0]
    assert math.isfinite(rms) and rms > 0.0
    assert math.isfinite(cell["twoway"][0])
