import math

import torch

from ota_sync import (
    PilotReceiver,
    SDRSimulationConfig,
    SimulationConfig,
    make_sync_preamble,
    run_sdr_simulation,
    run_simulation,
    wrap_phase,
)


def test_wrap_phase_uses_principal_interval():
    angles = torch.tensor(
        [-3.0 * math.pi, -math.pi, 0.0, math.pi, 3.0 * math.pi],
        dtype=torch.float64,
    )
    expected = torch.tensor(
        [-math.pi, -math.pi, 0.0, -math.pi, -math.pi],
        dtype=torch.float64,
    )
    torch.testing.assert_close(wrap_phase(angles), expected)


def test_pilot_receiver_recovers_phase_and_frequency_without_noise():
    times = (torch.arange(501, dtype=torch.float64) - 250.0) * 1e-5
    phase = -1.1
    frequency = 31.0
    received = torch.exp(1j * (phase + frequency * times))

    measurement = PilotReceiver(times).estimate(received)

    torch.testing.assert_close(
        measurement,
        torch.tensor(
            [math.cos(phase), math.sin(phase), frequency], dtype=torch.float64
        ),
        rtol=1e-10,
        atol=1e-10,
    )


def test_noiseless_ota_loop_synchronizes_slave():
    result = run_simulation(
        SimulationConfig(
            num_iterations=5,
            snr_db=120.0,
            phase_process_variance=0.0,
            frequency_process_variance=0.0,
            seed=7,
            device="cpu",
        )
    )

    assert abs(result.final_phase_error) < 1e-5
    assert abs(result.final_frequency_error) < 1e-3
    assert torch.all(torch.linalg.eigvalsh(result.covariance) >= -1e-12)


def test_seeded_sionna_awgn_run_is_reproducible():
    settings = SimulationConfig(num_iterations=8, seed=11, device="cpu")
    first = run_simulation(settings)
    second = run_simulation(settings)

    torch.testing.assert_close(
        first.post_correction_phase, second.post_correction_phase
    )
    torch.testing.assert_close(
        first.post_correction_frequency, second.post_correction_frequency
    )


def test_sdr_preamble_has_repeated_training_fields():
    settings = SDRSimulationConfig(num_iterations=1, device="cpu")
    preamble = make_sync_preamble(settings, torch.device("cpu"))

    short = preamble.waveform[: preamble.short_length]
    torch.testing.assert_close(
        short[: settings.short_sequence_length],
        short[settings.short_sequence_length : 2 * settings.short_sequence_length],
    )
    assert preamble.length == (
        settings.short_sequence_length * settings.short_repetitions
        + (settings.long_cp_length + settings.long_sequence_length)
        * settings.long_repetitions
    )


def test_sdr_tdl_link_acquires_and_corrects_effective_ota_carrier():
    result = run_sdr_simulation(
        SDRSimulationConfig(num_iterations=5, seed=3, device="cpu")
    )

    assert result.detection_rate == 1.0
    assert torch.max(torch.abs(result.timing_error_samples)) <= 1.0
    assert abs(result.final_ota_phase_error) < 0.01
    assert abs(result.final_frequency_error_hz) < 2.0
    assert torch.all(result.adc_clip_rate < 0.01)
