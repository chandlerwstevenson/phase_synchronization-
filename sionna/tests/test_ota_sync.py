import math

import torch

from ota_sync import (
    ConsensusStatsConfig,
    OSCILLATOR_PROFILES,
    PilotReceiver,
    SDRSimulationConfig,
    SimulationConfig,
    evaluate_csi_joint_transmission,
    make_sync_preamble,
    resolve_oscillator_noise,
    run_consensus_ota_simulation,
    run_consensus_stats,
    run_sdr_simulation,
    run_simulation,
    run_two_way_simulation,
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


def test_sdr_truth_reference_matches_measurement_when_clean():
    result = run_sdr_simulation(
        SDRSimulationConfig(
            num_iterations=3,
            snr_db=100.0,
            sample_clock_offset_ppm=0.0,
            phase_noise_std_rad=0.0,
            phase_noise_white_pm_std_rad=0.0,
            flicker_frequency_std_hz=0.0,
            shadowing_std_db=0.0,
            iq_gain_imbalance_db=0.0,
            iq_phase_imbalance_deg=0.0,
            dc_offset=0j,
            adc_bits=14,
            seed=5,
            device="cpu",
        )
    )

    torch.testing.assert_close(
        result.measured_ota_phase, result.true_ota_phase, atol=2e-3, rtol=0.0
    )
    torch.testing.assert_close(
        result.measured_frequency, result.true_ota_frequency, atol=10.0, rtol=0.0
    )


def test_sdr_tdl_link_acquires_and_corrects_effective_ota_carrier():
    result = run_sdr_simulation(
        SDRSimulationConfig(num_iterations=5, seed=3, device="cpu")
    )

    assert result.detection_rate == 1.0
    assert torch.max(torch.abs(result.timing_error_samples)) <= 1.0
    # The closed-loop residual floor is the LO white-FM walk accumulated over
    # one sync interval (~45 mrad RMS at defaults), which no controller can
    # predict; the bound leaves headroom for tail draws.
    assert abs(result.final_ota_phase_error) < 0.25
    assert abs(result.final_frequency_error_hz) < 2.0
    assert torch.all(result.adc_clip_rate < 0.01)


def test_sdr_delayed_corrections_converge_without_phase_noise():
    result = run_sdr_simulation(
        SDRSimulationConfig(
            num_iterations=10,
            phase_noise_std_rad=0.0,
            phase_noise_white_pm_std_rad=0.0,
            flicker_frequency_std_hz=0.0,
            shadowing_std_db=0.0,
            iq_gain_imbalance_db=0.0,
            iq_phase_imbalance_deg=0.0,
            dc_offset=0j,
            correction_latency_intervals=1,
            seed=4,
            device="cpu",
        )
    )

    assert result.detection_rate == 1.0
    # With the LO walk disabled, the floor of a delayed-correction loop is the
    # frequency-estimate uncertainty propagated over the latency interval
    # (roughly 40 mrad at the default oscillator frequency random walk), far
    # below the initial 1.2 rad offset the loop must remove.
    assert abs(result.final_ota_phase_error) < 0.1
    assert torch.sqrt(
        torch.mean(result.post_correction_ota_phase[5:].square())
    ).item() < 0.08


def test_two_way_sync_cancels_channel_phase_bias():
    result = run_two_way_simulation(
        SDRSimulationConfig(num_iterations=12, seed=3, device="cpu")
    )

    assert result.detection_rate == 1.0
    # One-way sync leaves a channel-phase bias of about -2.9 rad in the raw
    # oscillator residual; reciprocity must remove it, leaving only the
    # oscillator-noise/latency floor.
    assert abs(result.final_phase_error) < 0.4
    assert result.steady_state_phase_rms < 0.4
    assert result.mean_coherent_gain > 0.9


def test_two_way_clean_loop_reaches_estimation_floor():
    result = run_two_way_simulation(
        SDRSimulationConfig(
            num_iterations=10,
            phase_noise_std_rad=0.0,
            phase_noise_white_pm_std_rad=0.0,
            flicker_frequency_std_hz=0.0,
            shadowing_std_db=0.0,
            iq_gain_imbalance_db=0.0,
            iq_phase_imbalance_deg=0.0,
            dc_offset=0j,
            seed=4,
            device="cpu",
        )
    )

    assert result.detection_rate == 1.0
    assert abs(result.final_phase_error) < 0.1
    assert result.mean_coherent_gain > 0.99


def test_consensus_stats_converges_and_respects_eq27_bound():
    settings = ConsensusStatsConfig(
        num_nodes=20, connectivity=0.2, num_iterations=150, seed=1
    )
    dfpc = run_consensus_stats(settings)

    # Initial spread is ~100 ppm of 1 GHz; consensus must collapse it by
    # orders of magnitude, and the paper's Eq. 27 upper-bounds the residual
    # for sparse graphs (moderate connectivity sits below the bound).
    assert dfpc.frequency_spread_hz[-1] < 1e-2 * dfpc.frequency_spread_hz[0]
    assert dfpc.final_phase_error_std < 1.5 * dfpc.eq27_bound_rad


def test_consensus_stats_kalman_variant_reduces_residual():
    base = dict(num_nodes=20, connectivity=0.2, num_iterations=150, seed=1)
    dfpc = run_consensus_stats(ConsensusStatsConfig(algorithm="dfpc", **base))
    kf = run_consensus_stats(ConsensusStatsConfig(algorithm="kf-dfpc", **base))

    tail = slice(100, None)
    assert torch.mean(kf.total_phase_error_std[tail]) < torch.mean(
        dfpc.total_phase_error_std[tail]
    )


def test_naive_consensus_ota_captures_at_anti_phase():
    result = run_consensus_ota_simulation(
        SDRSimulationConfig(num_iterations=25, seed=0, device="cpu"),
        "dfpc",
        reciprocal=False,
    )

    # The paper's channel-free assumption makes naive OTA consensus
    # bistable: wrapped symmetric updates converge to relative phase 0 or
    # pi depending on the channel-phase realization. Seed 0 (channel phase
    # about -2.9 rad against a 1.2 rad initial offset) wraps on the first
    # update and locks at the anti-phase fixed point.
    assert result.detection_rate == 1.0
    assert result.steady_state_phase_rms > 1.5
    assert result.mean_coherent_gain < 0.3


def test_reciprocal_consensus_ota_aligns_and_filtering_helps():
    settings = SDRSimulationConfig(num_iterations=25, seed=3, device="cpu")
    dfpc = run_consensus_ota_simulation(settings, "dfpc")
    kf = run_consensus_ota_simulation(settings, "kf-dfpc")

    assert dfpc.detection_rate == 1.0
    # With the exchanged half-difference the channel phase cancels and the
    # pair genuinely aligns.
    assert dfpc.steady_state_phase_rms < 0.6
    assert kf.steady_state_phase_rms < 0.4
    # The paper's own claim, reproduced over a physical link: filtering
    # reduces the residual relative to raw consensus.
    assert kf.steady_state_phase_rms < dfpc.steady_state_phase_rms


def test_micro_pilot_loop_beats_plain_two_way():
    from ota_sync import run_micro_two_way_simulation

    settings = SDRSimulationConfig(num_iterations=40, seed=0, device="cpu")
    baseline = run_two_way_simulation(settings)
    micro = run_micro_two_way_simulation(settings, micro_pilots_per_interval=4)

    assert micro.detection_rate == 1.0
    # Re-measuring phase 5x per interval cuts the walk and staleness terms;
    # the residual should improve by well over 2x at modest extra airtime.
    assert micro.steady_state_phase_rms < 0.5 * baseline.steady_state_phase_rms
    assert micro.mean_coherent_gain > 0.999
    assert micro.airtime_fraction < 0.35


def test_hybrid_calibration_matches_micro_at_much_lower_airtime():
    from hybrid_calibration import run_hybrid_simulation

    settings = SDRSimulationConfig(num_iterations=40, seed=0, device="cpu")
    baseline = run_two_way_simulation(settings)
    hybrid = run_hybrid_simulation(
        settings, micro_pilots_per_interval=4, anchor_every_intervals=5
    )

    assert hybrid.detection_rate == 1.0
    # One-way micro-pilots plus sparse reciprocal anchors must beat the
    # plain two-way loop on residual while spending less airtime than it.
    assert hybrid.steady_state_phase_rms < 0.6 * baseline.steady_state_phase_rms
    assert hybrid.airtime_fraction < 0.16
    assert hybrid.mean_coherent_gain > 0.999


def test_hybrid_doppler_requires_matched_channel_prior():
    from hybrid_calibration import run_hybrid_simulation

    settings = SDRSimulationConfig(
        num_iterations=40, seed=0, device="cpu", channel_speed_mps=0.5
    )
    mismatched = run_hybrid_simulation(
        settings, anchor_every_intervals=5, channel_drift_std_rad=0.01
    )
    matched = run_hybrid_simulation(
        settings, anchor_every_intervals=5, channel_drift_std_rad=0.48
    )

    # Under channel Doppler the split between oscillator and channel phase
    # rests on the process priors: a static-channel prior lets the filter
    # chase the channel with NCO corrections, while a matched prior keeps
    # the oscillator residual usable at the same sparse anchor cadence.
    assert matched.steady_state_phase_rms < 0.5 * mismatched.steady_state_phase_rms
    assert matched.mean_coherent_gain > 0.95


def test_csi_joint_transmission_gain_degrades_with_stale_csi():
    result = run_sdr_simulation(
        SDRSimulationConfig(num_iterations=60, seed=3, device="cpu")
    )
    gains = evaluate_csi_joint_transmission(result, refresh_intervals=(1, 20))

    assert gains[1] > 0.99
    assert gains[20] < gains[1]
    assert gains[20] > 0.5


def test_oscillator_profiles_rank_by_stability():
    carrier = 915e6
    knobs = {
        name: OSCILLATOR_PROFILES[name].noise_settings(carrier, 1e6, 0.05)
        for name in ("ocxo", "tcxo", "sdr")
    }
    # A base-station OCXO is quieter than a small-cell TCXO, which is
    # quieter than a bench-SDR TCXO, in every noise dimension.
    for key in (
        "phase_noise_std_rad",
        "flicker_frequency_std_hz",
        "frequency_process_std_hz",
    ):
        assert knobs["ocxo"][key] < knobs["tcxo"][key] < knobs["sdr"][key]
    cfo = {
        name: OSCILLATOR_PROFILES[name].expected_cfo_hz(carrier)
        for name in ("ocxo", "tcxo", "sdr")
    }
    assert cfo["ocxo"] < cfo["tcxo"] < cfo["sdr"]
    # The legacy profile keeps the config defaults untouched.
    noise, default_cfo = resolve_oscillator_noise("custom", carrier, 1e6, 0.05)
    assert noise == {}
    assert default_cfo is None


def test_two_way_turnaround_gap_is_compensated():
    # A 5 ms half-duplex turnaround with a 1500 Hz initial CFO shifts the
    # reverse frame by ~24 rad of deterministic drift; the loop's own CFO
    # estimate must remove it, leaving the clean loop at its floor.
    result = run_two_way_simulation(
        SDRSimulationConfig(
            num_iterations=10,
            tdd_turnaround_s=5e-3,
            phase_noise_std_rad=0.0,
            phase_noise_white_pm_std_rad=0.0,
            flicker_frequency_std_hz=0.0,
            shadowing_std_db=0.0,
            iq_gain_imbalance_db=0.0,
            iq_phase_imbalance_deg=0.0,
            dc_offset=0j,
            seed=4,
            device="cpu",
        )
    )

    assert result.detection_rate == 1.0
    assert abs(result.final_phase_error) < 0.1
    assert result.mean_coherent_gain > 0.99


def test_station_placement_is_reproducible_and_separated():
    from ota_sync import place_stations

    first = place_stations(8, 400.0, seed=3, min_separation_m=25.0)
    second = place_stations(8, 400.0, seed=3, min_separation_m=25.0)
    assert (first == second).all()
    for i in range(8):
        assert (first[i] ** 2).sum() ** 0.5 <= 400.0
        for j in range(i + 1, 8):
            distance = ((first[i] - first[j]) ** 2).sum() ** 0.5
            assert distance >= 25.0


def test_three_station_network_reaches_coherence():
    from ota_sync import run_network_simulation

    settings = SDRSimulationConfig(
        num_iterations=10,
        phase_noise_std_rad=0.0,
        phase_noise_white_pm_std_rad=0.0,
        flicker_frequency_std_hz=0.0,
        shadowing_std_db=0.0,
        iq_gain_imbalance_db=0.0,
        iq_phase_imbalance_deg=0.0,
        dc_offset=0j,
        seed=4,
        device="cpu",
    )

    def extract(result):
        mask = result.detected & result.correction_active & result.calibrated
        return (
            result.post_correction_phase,
            mask,
            result.detection_rate,
            result.airtime_fraction,
        )

    network = run_network_simulation(
        settings,
        3,
        run_two_way_simulation,
        extract,
        radius_m=300.0,
        path_loss_exponent=2.7,
        reference_distance_m=300.0,
    )

    assert network.num_stations == 3
    assert network.min_detection_rate == 1.0
    # Every link's SNR reflects its distance through the path-loss model.
    for link in network.links:
        assert link.snr_db <= 50.0
        assert link.distance_m > 0.0
    assert network.mean_array_gain > 0.95
    # Two links sharing the channel cost twice one link's pilots.
    assert abs(
        network.total_airtime_fraction - 2 * network.links[0].airtime_fraction
    ) < 1e-9


def test_decentralized_hybrid_matches_centralized_quality():
    from hybrid_calibration import run_hybrid_simulation

    settings = SDRSimulationConfig(num_iterations=12, seed=3, device="cpu")
    centralized = run_hybrid_simulation(settings)
    decentralized = run_hybrid_simulation(settings, decentralized=True)

    # Symmetric control changes WHO moves, not the relative dynamics:
    # the decentralized loop must reach the same quality class.
    assert decentralized.detection_rate == 1.0
    assert decentralized.steady_state_phase_rms < 0.4
    assert decentralized.mean_coherent_gain > 0.9
    assert (
        decentralized.steady_state_phase_rms
        < 3.0 * centralized.steady_state_phase_rms + 0.05
    )


def test_decentralized_hybrid_mesh_converges_beyond_two_nodes():
    from hybrid_calibration import run_decentralized_hybrid_mesh

    settings = SDRSimulationConfig(
        num_iterations=16,
        phase_noise_std_rad=0.0,
        phase_noise_white_pm_std_rad=0.0,
        flicker_frequency_std_hz=0.0,
        shadowing_std_db=0.0,
        iq_gain_imbalance_db=0.0,
        iq_phase_imbalance_deg=0.0,
        dc_offset=0j,
        seed=4,
        device="cpu",
    )
    result = run_decentralized_hybrid_mesh(settings, num_nodes=3)

    # Symmetric consensus control on shared oscillators must still lock
    # the whole chain: no anti-phase capture, high datum-free array gain.
    assert result.detection_rate == 1.0
    assert result.mean_array_gain > 0.9
    assert result.worst_edge_rms < 0.35


def test_mesh_scheduling_beats_simultaneous_consensus():
    from hybrid_calibration import run_decentralized_hybrid_mesh

    settings = SDRSimulationConfig(num_iterations=20, seed=0, device="cpu")
    symmetric = run_decentralized_hybrid_mesh(
        settings, num_nodes=4, control="symmetric"
    )
    alternating = run_decentralized_hybrid_mesh(
        settings, num_nodes=4, control="alternating"
    )

    # The consensus tax is a scheduling artifact: taking turns
    # (Gauss-Seidel) must decisively beat simultaneous degree-weighted
    # updates (Jacobi) under identical physics.
    assert alternating.mean_array_gain > symmetric.mean_array_gain + 0.1
    assert alternating.mean_array_gain > 0.95


def test_detection_viability_math_is_consistent():
    from detection import (
        DetectionParams,
        coherent_snr_factor,
        detection_range_m,
        probability_of_detection,
        required_snr,
    )

    params = DetectionParams()
    snr_req = required_snr(params.pfa, params.pd_target)
    # The required-SNR inversion must round-trip through the detector.
    assert abs(probability_of_detection(snr_req, params.pfa) - 0.9) < 1e-9
    # Perfect N-station coherence buys exactly N^3 in SNR = N^(3/4) in range.
    single = detection_range_m(1, 1.0, params)
    assert abs(detection_range_m(6, 1.0, params) / single - 6 ** 0.75) < 1e-9
    # Free-running (G = 1/N) collapses to incoherent N, not N^3.
    free = detection_range_m(6, 1.0 / 6.0, params)
    assert abs(free / single - 6 ** 0.25) < 1e-9
    # Range is monotone in sync gain.
    assert detection_range_m(6, 0.9, params) > detection_range_m(6, 0.5, params)


def test_ofdm_detection_burst_is_unit_power_and_detects():
    import numpy as np
    from detection.waveform import _ofdm_burst, run_waveform_detection
    import torch as _torch

    burst = _ofdm_burst(1040, _torch.Generator().manual_seed(0))
    assert abs(_torch.mean(_torch.abs(burst) ** 2).item() - 1.0) < 1e-9

    # A close target with perfect sync must be detected essentially
    # always; the empirical threshold must reproduce its target Pfa.
    positions = np.array([[0.0, 0.0], [200.0, 50.0], [-150.0, 120.0]])
    result = run_waveform_detection(
        "perfect",
        positions,
        _torch.zeros(3, 8, dtype=_torch.float64),
        np.array([[150.0, 400.0]]),
        trials=300,
        h0_trials=20000,
        seed=1,
    )
    assert result.pd_measured[0] > 0.99
    assert abs(result.measured_pfa - result.threshold_pfa) < 5e-4
    assert result.combining_loss_db[0] > -0.01


def test_clutter_model_produces_clutter_limited_regime():
    import numpy as np
    from detection import DetectionParams
    from detection.realistic import (
        RealisticDetectionConfig,
        _clutter_gate_power,
        _range_gates,
    )

    positions = np.array(
        [[0.0, 0.0], [300.0, 100.0], [-200.0, 250.0], [150.0, -300.0]]
    )
    waypoints = np.array([[800.0, 100.0], [-800.0, 100.0]])
    params = DetectionParams()
    config = RealisticDetectionConfig()
    gates, _ = _range_gates(positions, waypoints, config, 1e6)
    power = _clutter_gate_power(positions, gates, params, config, 1e6)

    assert power.shape == (4, len(gates))
    assert (power >= 0.0).all()
    # UHF ground clutter must dominate thermal noise by tens of dB —
    # the regime that forces Doppler processing.
    from detection.viability import BOLTZMANN_T0

    noise = (
        BOLTZMANN_T0
        * 10.0 ** (params.noise_figure_db / 10.0)
        * 10.0 ** (params.losses_db / 10.0)
        * 1e6
    )
    assert power.max() / noise > 1e3  # > 30 dB CNR
