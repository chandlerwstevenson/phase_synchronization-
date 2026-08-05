"""Command-line entry point for the Sionna OTA synchronization examples."""

from __future__ import annotations

import argparse
import math

import torch

from ota_sync import (
    LEGACY_PROFILE_NAME,
    OSCILLATOR_PROFILES,
    SDRSimulationConfig,
    SDRSimulationResult,
    SimulationConfig,
    evaluate_csi_joint_transmission,
    resolve_oscillator_noise,
    run_network_simulation,
    run_consensus_ota_simulation,
    run_micro_two_way_simulation,
    run_sdr_simulation,
    run_simulation,
    run_two_way_simulation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate one-way OTA phase/frequency synchronization with Sionna."
    )
    parser.add_argument(
        "--model",
        choices=(
            "sdr",
            "twoway",
            "micro",
            "hybrid",
            "dhybrid",
            "dfpc",
            "kfdfpc",
            "compare",
            "ideal",
        ),
        default="sdr",
        help="one-way sampled-IQ SDR model (default), reciprocal two-way "
        "sync, two-tier micro-pilot sync, hybrid one-way+anchor sync, "
        "decentralized hybrid (no master: both nodes retune half-way), "
        "Rashid & Nanzer's consensus algorithms over the physical layer "
        "(dfpc/kfdfpc), a side-by-side comparison of all open-loop "
        "approaches, or the ideal AWGN model",
    )
    parser.add_argument(
        "--micro-pilots",
        type=int,
        default=4,
        help="phase-only micro-pilot exchanges per interval (micro model)",
    )
    parser.add_argument(
        "--anchor-every",
        type=int,
        default=5,
        help="intervals between reciprocal two-way anchors (hybrid model)",
    )
    parser.add_argument(
        "--csi-gain",
        action="store_true",
        help="with --model sdr: also report coherent joint-transmission gain "
        "at a user vs. CSI refresh cadence",
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--snr-db", type=float, default=20.0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="display the standard scoreboard: |residual| vs time and coherent "
        "gain, identical axes/units for every model so methods compare fairly",
    )
    parser.add_argument(
        "--plot-story",
        action="store_true",
        help="one-way model only: the guided walkthrough figure (the rollover "
        "problem, the residual, the estimator's eyesight)",
    )
    network = parser.add_argument_group("multi-station network")
    network.add_argument(
        "--stations",
        type=int,
        default=None,
        help="number of base stations: station 0 becomes the reference and "
        "every other station syncs to it with the chosen model (sdr, twoway, "
        "micro, or hybrid), time-multiplexed on one channel. Stations are "
        "placed uniformly at random in a disc and each link's SNR follows "
        "log-distance path loss",
    )
    network.add_argument(
        "--sweep-stations",
        nargs="?",
        const="2,4,8,12",
        default=None,
        metavar="N_LIST",
        help="scalability sweep: run the network at several station counts "
        "(comma-separated; bare flag sweeps 2,4,8,12) and show array "
        "coherence, worst-station residual, and total airtime vs N",
    )
    network.add_argument(
        "--area-radius-m",
        type=float,
        default=500.0,
        help="radius of the random deployment disc (meters)",
    )
    network.add_argument(
        "--path-loss-exponent",
        type=float,
        default=2.7,
        help="log-distance path-loss exponent (2 = free space, 2.7-3.9 = "
        "urban macro per 3GPP TR 38.901)",
    )
    network.add_argument(
        "--mesh-control",
        choices=("symmetric", "alternating", "directed"),
        default="alternating",
        help="control law for the dhybrid mesh: 'symmetric' = all edges "
        "correct simultaneously with degree-weighted halves (DFPC-style; "
        "pays the consensus tax), 'alternating' = edges take turns so "
        "corrections apply at full strength (default; still no master), "
        "'directed' = each node retunes fully toward its elected-root "
        "parent (decentralized fault structure, centralized control)",
    )
    network.add_argument(
        "--ref-distance-m",
        type=float,
        default=500.0,
        help="distance at which --snr-db is defined (default: the disc "
        "radius, so edge-to-center links get roughly the nominal SNR); "
        "links farther are noisier, closer ones cleaner",
    )
    parser.add_argument(
        "--sweep-interval",
        nargs="?",
        const="10,20,50,100,200,500",
        default=None,
        metavar="MS_LIST",
        help="run the chosen model at several sync intervals (comma-separated "
        "milliseconds; bare flag sweeps 10,20,50,100,200,500) and show "
        "accuracy vs cost as the pilot cadence changes — how 50 ms was chosen",
    )
    parser.add_argument(
        "--plot-all",
        action="store_true",
        help="display full SDR acquisition and channel diagnostics",
    )
    parser.add_argument(
        "--plot-iq",
        action="store_true",
        help="show one capture at the IQ-sample level: transmitted waveform, "
        "received samples, constellation, and detection metrics",
    )

    sdr = parser.add_argument_group("SDR model")
    sdr.add_argument(
        "--oscillator",
        choices=(LEGACY_PROFILE_NAME, "ocxo", "tcxo", "sdr"),
        default=LEGACY_PROFILE_NAME,
        help="oscillator noise profile anchored to real parts: 'ocxo' = macro "
        "base-station Stratum 3E OCXO (Rakon ROM1490E class, 3GPP +-50 ppb); "
        "'tcxo' = small-cell ultra-stable TCXO (+-100 ppb); 'sdr' = bench SDR "
        "TCXO without GPSDO (USRP B2xx class, +-2 ppm); 'custom' = this "
        "repository's original hand-tuned values (default, keeps published "
        "numbers reproducible)",
    )
    sdr.add_argument("--sample-rate", type=float, default=1e6, help="IQ sample rate")
    sdr.add_argument("--carrier-mhz", type=float, default=915.0)
    sdr.add_argument(
        "--cfo-hz",
        type=float,
        default=None,
        help="initial carrier frequency offset; default = the oscillator "
        "profile's rated accuracy (1500 Hz for the custom profile)",
    )
    sdr.add_argument(
        "--sfo-ppm",
        type=float,
        default=None,
        help="fixed sample-clock offset; default derives it from the carrier "
        "offset (shared reference crystal)",
    )
    sdr.add_argument(
        "--flicker-std-hz",
        type=float,
        default=None,
        help="RMS flicker FM frequency deviation of the references; default = "
        "the oscillator profile's value (0.05 Hz for the custom profile)",
    )
    sdr.add_argument(
        "--turnaround-ms",
        type=float,
        default=1.0,
        help="half-duplex (TDD) turnaround between the forward and reverse "
        "frames of a two-way exchange; the oscillators keep drifting across "
        "the gap (twoway model; 0 restores the instant-reciprocity idealization)",
    )
    sdr.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="run the chosen model at N consecutive seeds and report "
        "mean +- std of the key metrics (Monte Carlo over channel and "
        "noise realizations); works with --model compare too",
    )
    sdr.add_argument(
        "--shadowing-std-db",
        type=float,
        default=2.0,
        help="std of the temporally correlated log-normal shadowing",
    )
    sdr.add_argument("--tdl-model", choices=("A", "B", "C", "D", "E"), default="D")
    sdr.add_argument("--delay-spread-ns", type=float, default=100.0)
    sdr.add_argument("--speed-mps", type=float, default=0.0)
    sdr.add_argument("--adc-bits", type=int, default=12)
    sdr.add_argument(
        "--correction-latency",
        type=int,
        default=1,
        help="processing latency in sync intervals before an NCO command loads",
    )
    sdr.add_argument(
        "--no-rf-impairments",
        action="store_true",
        help="disable SFO, phase noise, IQ imbalance, and DC offset",
    )

    ideal = parser.add_argument_group("ideal model")
    ideal.add_argument("--pilot-length", type=int, default=500)
    return parser.parse_args()


def _render_figure_and_panels(
    suptitle: str | None,
    panels,
    shape: tuple[int, int],
    figsize: tuple[float, float],
    sharex: bool = True,
    top: float = 0.92,
) -> None:
    """Draw the annotated combined figure PLUS one clean figure per panel.

    ``panels`` is a list of (explanatory_title, draw_fn) pairs. Each
    draw_fn(axis) renders its panel completely — data, axis labels,
    legend — but without the explanatory text. The combined figure adds
    the wordy titles and the suptitle; the individual figures stay clean
    so they can go straight into slides or a paper.
    """

    import matplotlib.pyplot as plt

    rows, cols = shape
    figure, axes = plt.subplots(rows, cols, figsize=figsize, sharex=sharex)
    flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for axis, (title, draw) in zip(flat, panels):
        draw(axis)
        if title:
            axis.set_title(title, fontsize=10)
        axis.grid(True, alpha=0.3, which="both")
    if sharex:
        # With a shared x axis only the bottom row keeps its x label.
        for axis in flat[: (rows - 1) * cols]:
            axis.set_xlabel("")
    if suptitle:
        figure.suptitle(suptitle, fontsize=11)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, top) if suptitle else None)

    for _, draw in panels:
        single_figure, single_axis = plt.subplots(figsize=(9.0, 4.8))
        draw(single_axis)
        single_axis.grid(True, alpha=0.3, which="both")
        single_figure.tight_layout()
    plt.show()


def plot_ideal_result(result) -> None:
    iteration = range(len(result.true_phase))

    def pre_phase(axis):
        axis.plot(iteration, result.true_phase, label="true")
        axis.plot(iteration, result.estimated_phase, "--", label="EKF")
        axis.set_ylabel("phase (rad)")
        axis.set_xlabel("synchronization interval")
        axis.legend()

    def pre_frequency(axis):
        axis.plot(iteration, result.true_frequency, label="true")
        axis.plot(iteration, result.estimated_frequency, "--", label="EKF")
        axis.set_ylabel("angular frequency (rad/s)")
        axis.set_xlabel("synchronization interval")
        axis.legend()

    def post_phase(axis):
        axis.plot(iteration, result.post_correction_phase)
        axis.set_ylabel("phase residual (rad)")
        axis.set_xlabel("synchronization interval")

    def post_frequency(axis):
        axis.plot(iteration, result.post_correction_frequency)
        axis.set_ylabel("frequency residual (rad/s)")
        axis.set_xlabel("synchronization interval")

    _render_figure_and_panels(
        None,
        [
            ("Pre-correction relative phase", pre_phase),
            ("Pre-correction relative frequency", pre_frequency),
            ("After correction", post_phase),
            ("After correction", post_frequency),
        ],
        (2, 2),
        (11, 7),
    )


def plot_sdr_result(
    result: SDRSimulationResult, settings: SDRSimulationConfig
) -> None:
    import numpy as np

    step_seconds = settings.sync_interval
    pilot_samples = (
        settings.short_sequence_length * settings.short_repetitions
        + settings.long_repetitions
        * (settings.long_cp_length + settings.long_sequence_length)
    )
    pilot_ms = 1e3 * pilot_samples / settings.sample_rate
    latency = settings.correction_latency_intervals
    if latency == 0:
        latency_note = "loads instantly"
    else:
        plural = "s" if latency > 1 else ""
        latency_note = f"loads {latency} interval{plural} later"

    iteration = np.arange(len(result.true_phase))
    true_ota_phase = result.true_ota_phase.numpy()
    measured_phase = result.measured_ota_phase.numpy()
    post_phase_mrad = 1e3 * result.post_correction_ota_phase.numpy()
    steady = (result.detected & result.correction_active).numpy()
    if np.any(steady):
        post_rms_mrad = float(np.sqrt(np.mean(np.square(post_phase_mrad[steady]))))
    else:
        post_rms_mrad = float("nan")

    # What the phase offset would do with no synchronization at all: integrate
    # the physical (correction-free) relative oscillator frequency. The phase
    # itself lives on a circle, so wrap it to [-pi, pi) — the accumulated total
    # only matters as a count of slipped cycles.
    unsync_phase = (
        np.cumsum(result.physical_relative_frequency.numpy()) * step_seconds
    )
    unsync_wrapped = np.angle(np.exp(1j * unsync_phase))
    slipped_cycles = abs(unsync_phase[-1]) / (2.0 * math.pi)
    cycles_per_interval = slipped_cycles / max(len(iteration), 1)

    measured_error_mrad = 1e3 * np.angle(
        np.exp(1j * (measured_phase - true_ota_phase))
    )
    # ota_phase_error is wrap(true - estimate); negate for the same
    # "value minus truth" convention as the raw measurement curve.
    tracking_error_mrad = -1e3 * result.ota_phase_error.numpy()

    def rollover_panel(axis):
        axis.scatter(
            iteration,
            unsync_wrapped,
            s=12,
            color="tab:red",
            label="free-running (no corrections)",
        )
        axis.plot(
            iteration,
            true_ota_phase,
            color="tab:blue",
            linewidth=1.6,
            label="with the sync loop",
        )
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
        axis.set_ylim(-math.pi * 1.15, math.pi * 1.15)
        axis.set_yticks((-math.pi, 0.0, math.pi))
        axis.set_yticklabels((r"$-\pi$", "0", r"$\pi$"))
        axis.set_ylabel("phase offset (rad)")
        axis.set_xlabel("synchronization interval")
        axis.legend(fontsize="small", loc="upper right")

    def residual_panel(axis):
        axis.plot(iteration, post_phase_mrad, linewidth=1.4, label="residual")
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
        axis.axhline(
            post_rms_mrad,
            color="tab:red",
            linestyle=":",
            linewidth=1.0,
            label=f"steady RMS = {post_rms_mrad:.3f} mrad",
        )
        axis.axhline(
            -post_rms_mrad,
            color="tab:red",
            linestyle=":",
            linewidth=1.0,
        )
        axis.set_ylabel("phase residual (mrad)")
        axis.set_xlabel("synchronization interval")
        axis.legend(fontsize="small")

    def eyesight_panel(axis):
        axis.plot(
            iteration,
            measured_error_mrad,
            linewidth=1.0,
            alpha=0.55,
            label="raw pilot measurement - truth",
        )
        axis.plot(
            iteration,
            tracking_error_mrad,
            linewidth=1.5,
            label=(
                "EKF estimate - truth "
                f"(RMSE {1e3 * result.ota_phase_rmse:.1f} mrad)"
            ),
        )
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
        axis.set_ylabel("estimation error (mrad)")
        axis.set_xlabel("synchronization interval")
        axis.legend(fontsize="small")

    _render_figure_and_panels(
        "One-way OTA phase synchronization (--model sdr)\n"
        f"Method: master sends a known {pilot_ms:.1f} ms Zadoff-Chu pilot every "
        f"{1e3 * step_seconds:.0f} ms → slave measures its phase & "
        "frequency → EKF smooths →\n"
        f"slave retunes its own oscillator (correction {latency_note}). "
        f"Tracking RMSE={1e3 * result.ota_phase_rmse:.3f} mrad, "
        f"steady residual RMS={post_rms_mrad:.3f} mrad",
        [
            (
                "The problem: with no corrections the offset never stops "
                f"rolling over — ~{cycles_per_interval:.0f} full cycles slip "
                "BETWEEN pilots\n"
                f"({slipped_cycles:,.0f} rollovers this run); sampled once "
                "per interval, the spin aliases into the red ramps sweeping "
                "the circle.\n"
                "The sync loop (blue) pins the offset at zero.",
                rollover_panel,
            ),
            (
                "Zoom on the blue curve (milliradians now): the loop's "
                "leftover misalignment. Every capture already\n"
                "includes all previously loaded corrections, so this IS the "
                "residual. Spike at 0: the initial offset,\n"
                "seen before the first correction has loaded.",
                residual_panel,
            ),
            (
                "The estimator's eyesight: it SEES the phase to "
                f"~{1e3 * result.ota_phase_rmse:.0f} mrad. The EKF hugs each "
                "raw measurement because the\n"
                "crystals drift too much between pilots for averaging old "
                f"pilots to help. The loop holds ~{post_rms_mrad:.0f} mrad "
                "(above),\n"
                "not this, because corrections load one interval late and "
                "jitter keeps arriving — only correcting faster helps.",
                eyesight_panel,
            ),
        ],
        (3, 1),
        (10.5, 10),
    )


def plot_sdr_diagnostics(result: SDRSimulationResult) -> None:
    import numpy as np

    iteration = np.arange(len(result.true_phase))
    true_ota_phase = result.true_ota_phase.numpy()
    measured_phase = result.measured_ota_phase.numpy()
    estimated_phase = result.estimated_ota_phase.numpy()
    oscillator_phase = np.unwrap(result.true_phase.numpy())
    channel_phase = np.unwrap(result.channel_phase.numpy())
    effective_phase = oscillator_phase + channel_phase
    post_phase_mrad = 1e3 * result.post_correction_ota_phase.numpy()
    detection_metric = result.detection_metric.numpy()
    detected = result.detected.numpy()
    timing_error = result.timing_error_samples.numpy()

    def observable_panel(axis):
        axis.plot(iteration, true_ota_phase, linewidth=1.7, label="true")
        axis.plot(
            iteration, measured_phase, linewidth=1.0, alpha=0.7, label="measured"
        )
        axis.plot(
            iteration, estimated_phase, "--", linewidth=1.2, label="EKF estimate"
        )
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
        axis.set_ylabel("phase (rad)")
        axis.set_xlabel("synchronization interval")
        axis.legend()

    def decomposition_panel(axis):
        axis.plot(iteration, oscillator_phase, label="oscillator")
        axis.plot(iteration, channel_phase, label="channel")
        axis.plot(
            iteration,
            effective_phase,
            "--",
            linewidth=1.5,
            label="observable sum",
        )
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
        axis.set_ylabel("phase (rad)")
        axis.set_xlabel("synchronization interval")
        axis.legend(ncols=3, fontsize="small")

    def cfo_panel(axis):
        axis.plot(
            iteration,
            result.true_ota_frequency / (2.0 * math.pi),
            linewidth=1.5,
            label="true",
        )
        axis.plot(
            iteration,
            result.estimated_frequency / (2.0 * math.pi),
            "--",
            linewidth=1.2,
            label="EKF",
        )
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
        axis.set_yscale("symlog", linthresh=1.0)
        axis.set_ylabel("CFO (Hz)")
        axis.set_xlabel("synchronization interval")
        axis.legend()

    def residual_panel(axis):
        axis.plot(iteration, post_phase_mrad, linewidth=1.3)
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
        axis.set_ylabel("phase residual (mrad)")
        axis.set_xlabel("synchronization interval")

    def detection_panel(axis):
        axis.plot(iteration, detection_metric, linewidth=1.3, label="score")
        axis.axhline(
            0.25,
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
            label="threshold",
        )
        if np.any(~detected):
            axis.scatter(
                iteration[~detected],
                detection_metric[~detected],
                color="tab:red",
                marker="x",
                label="missed",
                zorder=3,
            )
        axis.set_ylabel("normalized score")
        axis.set_xlabel("synchronization interval")
        axis.set_ylim(0.0, 1.05)
        axis.legend(fontsize="small")

    def timing_panel(axis):
        axis.step(iteration, timing_error, where="mid")
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
        axis.set_ylabel("timing error (samples)")
        axis.set_xlabel("synchronization interval")
        if np.all(timing_error == 0.0):
            axis.set_ylim(-0.5, 0.5)
            axis.text(
                0.5,
                0.85,
                "All packets acquired at the correct sample",
                ha="center",
                transform=axis.transAxes,
                fontsize="small",
            )

    _render_figure_and_panels(
        "SDR OTA synchronization diagnostics — one-way method: "
        "known pilot → detect → measure phase/CFO → EKF → retune\n"
        f"detection={100.0 * result.detection_rate:.1f}%, "
        f"OTA phase RMSE={1e3 * result.ota_phase_rmse:.3f} mrad, "
        f"CFO RMSE={result.frequency_rmse / (2.0 * math.pi):.3f} Hz",
        [
            ("Observable OTA phase before correction", observable_panel),
            ("Unwrapped phase decomposition", decomposition_panel),
            ("CFO acquisition and tracking (symmetric log scale)", cfo_panel),
            ("Post-correction OTA phase", residual_panel),
            ("Packet detection confidence", detection_panel),
            ("Packet timing", timing_panel),
        ],
        (3, 2),
        (12, 9),
        top=0.94,
    )


def plot_iq_diagnostics(settings: SDRSimulationConfig) -> None:
    """One capture at the IQ-sample level: waveform, channel, receiver DSP."""

    import matplotlib.pyplot as plt
    import numpy as np
    from dataclasses import replace

    import torch
    from sionna.phy import config as sionna_config

    from ota_sync.core import REAL_DTYPE, Oscillator, resolve_device
    from ota_sync.sdr import SDRRadioLink, make_sync_preamble

    settings = replace(settings, num_iterations=1)
    device = resolve_device("cpu")
    torch.manual_seed(settings.seed)
    sionna_config.seed = settings.seed
    generator = torch.Generator(device=device)
    generator.manual_seed(settings.seed + 1)

    frequency_process_std = 2.0 * math.pi * settings.frequency_process_std_hz
    covariance = torch.diag(
        torch.tensor(
            [settings.phase_process_std_rad**2, frequency_process_std**2],
            dtype=REAL_DTYPE,
            device=device,
        )
    )
    master = Oscillator(
        settings.master_initial_phase,
        2.0 * math.pi * settings.master_initial_frequency_hz,
        settings.sync_interval,
        covariance,
        device,
        generator,
    )
    slave = Oscillator(
        settings.slave_initial_phase,
        2.0 * math.pi * settings.slave_initial_frequency_hz,
        settings.sync_interval,
        covariance,
        device,
        generator,
    )
    preamble = make_sync_preamble(settings, device)
    link = SDRRadioLink(settings, preamble, device, generator)
    master.step()
    slave.step()
    if settings.sample_clock_offset_ppm is not None:
        sfo_ppm = settings.sample_clock_offset_ppm
    else:
        sfo_ppm = float(
            (slave.state[1] - master.state[1]).item()
            / (2.0 * math.pi * settings.carrier_frequency_hz)
            * 1e6
        )
    capture = link.capture(master, slave, 0, sfo_ppm)

    received = capture.samples.numpy()
    oracle = capture.oracle_samples.numpy()
    # The oracle path carries no AGC; rescale it to the received RMS so the
    # two are directly comparable in the plots.
    oracle = oracle * (
        np.sqrt(np.mean(np.abs(received) ** 2))
        / (np.sqrt(np.mean(np.abs(oracle) ** 2)) + 1e-12)
    )
    tx = preamble.waveform.numpy()
    start = capture.expected_arrival
    length = preamble.length

    def sliding(values, width):
        cumulative = np.concatenate(([0.0], np.cumsum(values)))
        return cumulative[width:] - cumulative[:-width]

    lag = settings.short_sequence_length
    width = preamble.short_length - lag
    product = np.conj(received[:-lag]) * received[lag:]
    energy_a = np.abs(received[:-lag]) ** 2
    energy_b = np.abs(received[lag:]) ** 2
    stf_correlation = sliding(product, width)
    metric = np.abs(stf_correlation) / (
        np.sqrt(sliding(energy_a, width) * sliding(energy_b, width)) + 1e-12
    )

    # As in the receiver: remove the coarse CFO (measured from the STF
    # self-correlation angle) before matched filtering, otherwise the
    # carrier rotation smears the correlation peak across the frame.
    coarse_index = int(np.argmax(metric))
    coarse_cfo = np.angle(stf_correlation[coarse_index]) / (
        lag * settings.sample_period
    )
    index = np.arange(received.size)
    derotated = received * np.exp(
        -1j * coarse_cfo * index * settings.sample_period
    )
    correlation = np.correlate(derotated, tx, mode="valid")
    window_energy = sliding(np.abs(received) ** 2, length)
    reference_energy = np.sum(np.abs(tx) ** 2)
    timing_metric = np.abs(correlation) / (
        np.sqrt(window_energy * reference_energy) + 1e-12
    )

    def tx_real_panel(axis):
        axis.plot(np.real(tx[:400]), linewidth=0.8)
        axis.axvline(preamble.short_length, color="tab:red", linestyle="--",
                     linewidth=1.0, label="STF ends / LTF CP begins")
        axis.set_xlabel("sample")
        axis.set_ylabel("real part")
        axis.legend(fontsize="small")

    def tx_envelope_panel(axis):
        axis.plot(np.abs(tx), linewidth=0.6)
        axis.set_ylim(0.0, 1.3)
        axis.set_xlabel("sample")
        axis.set_ylabel("|x[n]|")

    def received_panel(axis):
        axis.plot(np.real(received), linewidth=0.5, alpha=0.9,
                  label="received (noise + impairments)")
        axis.plot(np.real(oracle), linewidth=0.5, alpha=0.6,
                  label="oracle twin (clean, AGC-matched)")
        axis.axvline(start, color="tab:red", linestyle="--", linewidth=1.0,
                     label="frame arrival")
        axis.set_xlabel("sample")
        axis.set_ylabel("real part")
        axis.legend(fontsize="small")

    active = received[start : start + length]
    active_oracle = oracle[start : start + length]

    def constellation_panel(axis):
        axis.scatter(np.real(active), np.imag(active), s=2, alpha=0.25,
                     label="received")
        axis.scatter(np.real(active_oracle), np.imag(active_oracle), s=2,
                     alpha=0.25, label="oracle")
        axis.set_aspect("equal")
        axis.set_xlabel("I")
        axis.set_ylabel("Q")
        axis.legend(fontsize="small", markerscale=4)

    def stf_metric_panel(axis):
        axis.plot(metric, linewidth=0.8)
        axis.axhline(settings.detection_threshold, color="tab:red",
                     linestyle="--", linewidth=1.0, label="detection threshold")
        axis.set_xlabel("window position d")
        axis.set_ylabel("normalized score")
        axis.set_ylim(0.0, 1.05)
        axis.legend(fontsize="small")

    peak = int(np.argmax(timing_metric))

    def timing_metric_panel(axis):
        axis.plot(timing_metric, linewidth=0.8)
        axis.axvline(peak, color="tab:red", linestyle="--", linewidth=1.0,
                     label=f"peak at d={peak}")
        axis.set_xlabel("candidate start d")
        axis.set_ylabel("normalized correlation")
        axis.legend(fontsize="small")

    _render_figure_and_panels(
        "One capture at the IQ level "
        f"(TDL-{settings.tdl_model}, {settings.snr_db:.0f} dB SNR, "
        f"CFO {settings.slave_initial_frequency_hz:.0f} Hz)",
        [
            (
                "Transmitted preamble, real part (first 400 samples):\n"
                "16-sample repeats of the STF are visible",
                tx_real_panel,
            ),
            (
                "Transmitted frame envelope |x[n]| (all 4606 samples):\n"
                "constant amplitude --- the ZC property the PA likes",
                tx_envelope_panel,
            ),
            (
                "Received capture window, real part:\n"
                "guard noise, then the frame through the multipath channel",
                received_panel,
            ),
            (
                "IQ constellation of the frame samples:\n"
                "a ring (constant envelope, AGC-scaled) blurred by noise",
                constellation_panel,
            ),
            (
                "STF self-correlation metric M(d):\n"
                "high wherever the 16-sample repetition is present",
                stf_metric_panel,
            ),
            (
                "Matched-filter timing correlation:\n"
                "the single sharp ZC peak that gives sample-level timing",
                timing_metric_panel,
            ),
        ],
        (3, 2),
        (13, 9),
        sharex=False,
        top=0.94,
    )


def plot_standard_result(
    title: str,
    step_seconds: float,
    curves,
    gain,
    airtime: float | None = None,
    free_running=None,
    locked=None,
) -> None:
    """The standard scoreboard, identical for every model.

    Same panels, same units, same thresholds regardless of method, so
    figures from different runs can be compared side by side:
      top    the wrapped free-running phase offset (red dots, rolling over
             constantly) vs the loop holding it (blue) — the problem and
             the fix on one circle-aware axis
      middle |phase residual| in mrad on a log axis, against the
             314 mrad = 18 degree = 90%-coherent-gain threshold
             (the same axes --model compare uses)
      bottom 2-station coherent gain in percent, against the 90% line

    ``curves`` is a list of (label, residual_radians, steady_rms_or_None).
    ``airtime`` is the fraction of transmission time spent on sync pilots —
    the price paid for the accuracy shown.
    ``free_running`` is the physical (correction-free) relative frequency
    per step, integrated to reconstruct the no-sync phase.
    ``locked`` is (label, wrapped_phase_radians) for the blue curve.
    """

    import numpy as np

    panels = []

    if free_running is not None:
        unsync_phase = np.cumsum(np.asarray(free_running)) * step_seconds
        unsync_wrapped = np.angle(np.exp(1j * unsync_phase))
        slipped_cycles = abs(unsync_phase[-1]) / (2.0 * math.pi)

        def rollover_panel(axis):
            time = (np.arange(unsync_wrapped.size) + 1) * step_seconds
            axis.scatter(
                time,
                unsync_wrapped,
                s=12,
                color="tab:red",
                label="free-running (no corrections)",
            )
            if locked is not None:
                locked_label, locked_values = locked
                axis.plot(
                    time,
                    np.asarray(locked_values),
                    color="tab:blue",
                    linewidth=1.4,
                    label=locked_label,
                )
            axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
            axis.set_ylim(-math.pi * 1.15, math.pi * 1.15)
            axis.set_yticks((-math.pi, 0.0, math.pi))
            axis.set_yticklabels((r"$-\pi$", "0", r"$\pi$"))
            axis.set_ylabel("phase offset (rad)")
            axis.set_xlabel("time (s)")
            axis.legend(fontsize="small", loc="upper right")

        panels.append(
            (
                "The problem: free-running, the offset rolls over "
                f"{slipped_cycles:,.0f} times this run (red dots, aliased by "
                "once-per-step sampling); the loop holds it (blue)",
                rollover_panel,
            )
        )

    def residual_panel(axis):
        for label, residual, rms in curves:
            residual = np.abs(np.asarray(residual))
            time = (np.arange(residual.size) + 1) * step_seconds
            if rms is not None:
                label = f"{label} (steady RMS {1e3 * rms:.0f} mrad)"
            axis.semilogy(time, 1e3 * residual + 1e-2, linewidth=1.3, label=label)
        axis.axhline(
            314.0,
            color="red",
            linestyle="--",
            linewidth=1.0,
            label="314 mrad = 18\N{DEGREE SIGN} = 90% coherent gain",
        )
        axis.set_ylabel("|phase residual| (mrad, log)")
        axis.set_xlabel("time (s)")
        axis.legend(fontsize="small")

    residual_title = (
        "Lower is better; staying below the red line means usable coherence"
    )
    if airtime is not None:
        residual_title += (
            f"\ncost: sync pilots occupy {100.0 * airtime:.1f}% of the shared "
            "channel's time (not per beam/user)"
        )
    panels.append((residual_title, residual_panel))

    def gain_panel(axis):
        gain_values = np.asarray(gain)
        gain_time = (np.arange(gain_values.size) + 1) * step_seconds
        axis.plot(
            gain_time,
            100.0 * gain_values,
            linewidth=1.3,
            label=f"with the sync loop (mean {100.0 * np.mean(gain_values):.1f}%)",
        )
        if free_running is not None:
            unsync_gain = np.cos(unsync_wrapped / 2.0) ** 2
            axis.scatter(
                (np.arange(unsync_gain.size) + 1) * step_seconds,
                100.0 * unsync_gain,
                s=10,
                color="tab:red",
                alpha=0.6,
                label=(
                    "free-running, no sync "
                    f"(mean {100.0 * np.mean(unsync_gain):.0f}% this run; "
                    "long-run average ~50%, no coherent bonus)"
                ),
            )
        axis.axhline(
            90.0,
            color="red",
            linestyle="--",
            linewidth=1.0,
            label="90% threshold",
        )
        axis.set_ylabel("2-station coherent gain (%)")
        axis.set_xlabel("time (s)")
        axis.set_ylim(-2.0, 102.0)
        axis.legend(fontsize="small", loc="center right")

    panels.append((None, gain_panel))

    rows = len(panels)
    _render_figure_and_panels(
        title,
        panels,
        (rows, 1),
        (10.5, 7.5 if rows == 2 else 10.5),
    )


def plot_loop_result(title: str, step_seconds: float, result) -> None:
    """Standard scoreboard for any closed-loop (two-way-style) result."""

    plot_standard_result(
        title,
        step_seconds,
        [
            (
                "true crystal-to-crystal residual",
                result.post_correction_phase.numpy(),
                result.steady_state_phase_rms,
            )
        ],
        result.coherent_gain.numpy(),
        airtime=result.airtime_fraction,
        free_running=result.physical_relative_frequency.numpy(),
        locked=("with the sync loop", result.post_correction_phase.numpy()),
    )


def plot_sdr_scoreboard(
    result: SDRSimulationResult, settings: SDRSimulationConfig
) -> None:
    """Standard scoreboard for the one-way model, same axes as every other.

    Shows both the observable link phase (what one-way actually controls)
    and the true crystal offset (the quantity every other method is graded
    on) so the one-way model is directly comparable.
    """

    import numpy as np

    oscillator = result.post_correction_oscillator_phase.numpy()
    observable = result.post_correction_ota_phase.numpy()
    steady = (result.detected & result.correction_active).numpy()

    def steady_rms(values):
        if not np.any(steady):
            return None
        return float(np.sqrt(np.mean(np.square(values[steady]))))

    gain = np.cos(oscillator / 2.0) ** 2
    plot_standard_result(
        "One-way OTA phase synchronization (--model sdr) — standard "
        "scoreboard\n"
        "Insight: one-way pins the radio LINK phase (orange) but the crystals "
        "park at the absorbed channel phase\n"
        "(blue, above the red line) — so open-loop coherent gain is lost. "
        "Two-way exists to fix exactly this.",
        settings.sync_interval,
        [
            (
                "true crystal-to-crystal residual (comparable across methods)",
                oscillator,
                steady_rms(oscillator),
            ),
            (
                "observable link-phase residual (what one-way controls)",
                observable,
                steady_rms(observable),
            ),
        ],
        gain,
        airtime=result.airtime_fraction,
        free_running=result.physical_relative_frequency.numpy(),
        locked=(
            "with the sync loop (link phase, what one-way controls)",
            observable,
        ),
    )


def plot_method_comparison(series) -> None:
    """Overlay |residual| trajectories of several methods on one time axis."""

    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(11, 6.5))
    for label, step_seconds, result in series:
        residual = np.abs(result.post_correction_phase.numpy())
        time = (np.arange(residual.size) + 1) * step_seconds
        rms = 1e3 * result.steady_state_phase_rms
        airtime = 100.0 * result.airtime_fraction
        plt.semilogy(time, 1e3 * residual + 1e-2, linewidth=1.2,
                     label=f"{label} (steady {rms:.0f} mrad, {airtime:.0f}% air)")
    plt.axhline(314.0, color="red", linestyle="--", linewidth=1.0,
                label="18\N{DEGREE SIGN} = 90% coherent gain")
    plt.xlabel("time (s)")
    plt.ylabel("|oscillator phase residual| (mrad, log scale)")
    plt.title("Open-loop synchronization methods, identical physical conditions")
    plt.legend(fontsize="small")
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.show()


def metric(label: str, value: str, explanation: str) -> None:
    """Print one result line followed by a plain-language explanation."""
    print(f"{label}: {value}")
    print(f"    ({explanation})")


def run_ideal(args: argparse.Namespace) -> None:
    result = run_simulation(
        SimulationConfig(
            num_iterations=args.iterations,
            snr_db=args.snr_db,
            pilot_length=args.pilot_length,
            device=args.device,
            seed=args.seed,
        )
    )
    print("model: ideal AWGN")
    print(f"device: {result.device}")
    metric(
        "phase RMSE",
        f"{result.phase_rmse:.6g} rad",
        "avg error of the phase-offset estimate vs truth in this toy noise-only "
        "model (no multipath channel, no hardware impairments)",
    )
    metric(
        "frequency RMSE",
        f"{result.frequency_rmse:.6g} rad/s",
        "avg error of the frequency-offset estimate (divide by 2*pi for Hz)",
    )
    metric(
        "final phase residual",
        f"{result.final_phase_error:.6g} rad",
        "phase misalignment left after the last correction",
    )
    metric(
        "final frequency residual",
        f"{result.final_frequency_error:.6g} rad/s",
        "frequency mismatch left after the last correction",
    )
    if args.plot or args.plot_all:
        plot_ideal_result(result)


def sdr_settings_from_args(args: argparse.Namespace) -> SDRSimulationConfig:
    carrier_hz = args.carrier_mhz * 1e6
    default_interval = SDRSimulationConfig.__dataclass_fields__[
        "sync_interval"
    ].default
    profile_noise, profile_cfo = resolve_oscillator_noise(
        args.oscillator, carrier_hz, args.sample_rate, default_interval
    )
    if args.cfo_hz is not None:
        cfo_hz = args.cfo_hz
    elif profile_cfo is not None:
        cfo_hz = profile_cfo
    else:
        cfo_hz = 1500.0
    settings_values = {
        "num_iterations": args.iterations,
        "snr_db": args.snr_db,
        "sample_rate": args.sample_rate,
        "carrier_frequency_hz": carrier_hz,
        "slave_initial_frequency_hz": cfo_hz,
        "sample_clock_offset_ppm": args.sfo_ppm,
        "shadowing_std_db": args.shadowing_std_db,
        "tdl_model": args.tdl_model,
        "delay_spread_s": args.delay_spread_ns * 1e-9,
        "channel_speed_mps": args.speed_mps,
        "adc_bits": args.adc_bits,
        "correction_latency_intervals": args.correction_latency,
        "tdd_turnaround_s": args.turnaround_ms * 1e-3,
        "device": args.device,
        "seed": args.seed,
    }
    settings_values.update(profile_noise)
    if args.flicker_std_hz is not None:
        settings_values["flicker_frequency_std_hz"] = args.flicker_std_hz
    elif "flicker_frequency_std_hz" not in settings_values:
        settings_values["flicker_frequency_std_hz"] = 0.05
    if args.no_rf_impairments:
        settings_values.update(
            {
                "sample_clock_offset_ppm": 0.0,
                "phase_noise_std_rad": 0.0,
                "phase_noise_white_pm_std_rad": 0.0,
                "flicker_frequency_std_hz": 0.0,
                "shadowing_std_db": 0.0,
                "iq_gain_imbalance_db": 0.0,
                "iq_phase_imbalance_deg": 0.0,
                "dc_offset": 0j,
            }
        )
    return SDRSimulationConfig(**settings_values)


def run_sdr(args: argparse.Namespace) -> None:
    settings = sdr_settings_from_args(args)
    result = run_sdr_simulation(settings)
    steady = result.detected & result.correction_active
    if torch.any(steady):
        post_ota_rms = torch.sqrt(
            torch.mean(result.post_correction_ota_phase[steady].square())
        ).item()
    else:
        post_ota_rms = float("nan")
    print("model: sampled-IQ SDR")
    print(f"channel: Sionna 3GPP TDL-{settings.tdl_model}")
    print(f"device: {result.device}")
    metric(
        "packet detection rate",
        f"{100.0 * result.detection_rate:.2f}%",
        "how often the slave found the master's pilot in the noise; 100% = never missed",
    )
    metric(
        "pilot airtime fraction",
        f"{100.0 * result.airtime_fraction:.1f}%",
        "share of transmission time spent on sync pilots — the cost side of "
        "every accuracy number below",
    )
    metric(
        "OTA phase tracking RMSE",
        f"{result.ota_phase_rmse:.6g} rad",
        "avg error of the slave's phase-offset ESTIMATE vs ground truth — how well "
        f"it can SEE the offset ({math.degrees(result.ota_phase_rmse):.2g} deg)",
    )
    metric(
        "frequency tracking RMSE",
        f"{result.frequency_rmse / (2.0 * math.pi):.6g} Hz",
        "avg error of the carrier-frequency-offset estimate — compare to the "
        f"{settings.slave_initial_frequency_hz:.0f} Hz offset the slave started with",
    )
    metric(
        "steady-state OTA phase residual RMS",
        f"{post_ota_rms:.6g} rad",
        "misalignment LEFT OVER after corrections, once locked "
        f"({math.degrees(post_ota_rms):.2g} deg) — the bottom-line 'how "
        "synchronized are we' number; bigger than tracking RMSE because crystals "
        "jitter between corrections and each correction loads one interval late",
    )
    metric(
        "final OTA phase residual",
        f"{result.final_ota_phase_error:.6g} rad",
        "same residual but at the very last interval only — one noisy sample, "
        "less meaningful than the RMS above",
    )
    metric(
        "final raw oscillator phase residual",
        f"{result.final_oscillator_phase_error:.6g} rad",
        "gap between the two PHYSICAL crystals — stays large because one-way sync "
        "locks oscillator+channel phase combined, absorbing the channel's phase; "
        "use --model twoway to cancel it",
    )
    metric(
        "final CFO residual",
        f"{result.final_frequency_error_hz:.6g} Hz",
        "leftover carrier-frequency mismatch after correction at the last interval",
    )
    if args.csi_gain:
        gains = evaluate_csi_joint_transmission(result, seed=args.seed)
        print("coherent JT gain at user vs. CSI refresh cadence:")
        for refresh, gain in gains.items():
            print(f"  every {refresh:>2} interval(s): {100.0 * gain:.2f}%")
    if args.plot_all:
        plot_sdr_diagnostics(result)
    if args.plot_story:
        plot_sdr_result(result, settings)
    if args.plot:
        plot_sdr_scoreboard(result, settings)


def run_twoway(args: argparse.Namespace) -> None:
    result = run_two_way_simulation(sdr_settings_from_args(args))
    print("model: reciprocal two-way SDR (open-loop coherence)")
    print(f"device: {result.device}")
    metric(
        "detection rate (both directions)",
        f"{100.0 * result.detection_rate:.2f}%",
        "how often BOTH the forward and reverse pilots were found; each interval "
        "needs both to form a measurement",
    )
    metric(
        "pilot airtime fraction",
        f"{100.0 * result.airtime_fraction:.1f}%",
        "share of transmission time spent on sync pilots (two full frames per "
        "interval) — the cost side of the accuracy below",
    )
    metric(
        "oscillator phase tracking RMSE",
        f"{result.phase_rmse:.6g} rad",
        "avg error of the estimate vs the TRUE crystal-to-crystal offset — the "
        "two-way exchange cancels the channel phase, so the real oscillator gap "
        "is now observable",
    )
    metric(
        "steady-state oscillator phase residual RMS",
        f"{result.steady_state_phase_rms:.6g} rad",
        "true physical misalignment of the two crystals after corrections, once "
        "locked — the bottom-line number for transmitting coherently without "
        "channel knowledge",
    )
    metric(
        "mean open-loop 2-station coherent gain",
        f"{100.0 * result.mean_coherent_gain:.2f}%",
        "if both stations transmitted together, fraction of the ideal combined "
        "power actually delivered (cos^2 of half the residual); 100% = perfectly "
        "in phase",
    )
    metric(
        "final oscillator phase residual",
        f"{result.final_phase_error:.6g} rad",
        "crystal-to-crystal error at the very last interval — one noisy sample",
    )
    metric(
        "final CFO residual",
        f"{result.final_frequency_error_hz:.6g} Hz",
        "leftover carrier-frequency mismatch after correction at the last interval",
    )
    if args.plot or args.plot_all:
        settings = sdr_settings_from_args(args)
        plot_loop_result(
            "Reciprocal two-way synchronization (--model twoway)\n"
            "Method: both stations send each other a pilot over the SAME "
            "channel each interval; half the difference of\n"
            "the two measured phases cancels the channel, leaving the true "
            "crystal-to-crystal offset → EKF → slave retunes",
            settings.sync_interval,
            result,
        )


def run_consensus(args: argparse.Namespace, algorithm: str) -> None:
    result = run_consensus_ota_simulation(sdr_settings_from_args(args), algorithm)
    print(f"model: {algorithm} over the sampled-IQ physical layer (2 nodes)")
    print(f"device: {result.device}")
    metric(
        "detection rate (both directions)",
        f"{100.0 * result.detection_rate:.2f}%",
        "how often both nodes' pilots were found; each consensus step needs both",
    )
    metric(
        "pilot airtime fraction",
        f"{100.0 * result.airtime_fraction:.1f}%",
        "share of transmission time spent on sync pilots — the cost side of "
        "the accuracy below",
    )
    metric(
        "steady-state oscillator phase residual RMS",
        f"{result.steady_state_phase_rms:.6g} rad",
        "true physical misalignment of the two crystals after each node retunes "
        "toward the other, once locked",
    )
    metric(
        "mean open-loop 2-station coherent gain",
        f"{100.0 * result.mean_coherent_gain:.2f}%",
        "if both stations transmitted together, fraction of the ideal combined "
        "power actually delivered; near 0% means the loop locked half a cycle off "
        "(anti-phase)",
    )
    metric(
        "final oscillator phase residual",
        f"{result.final_phase_error:.6g} rad",
        "crystal-to-crystal error at the very last interval — one noisy sample",
    )
    metric(
        "final CFO residual",
        f"{result.final_frequency_error_hz:.6g} Hz",
        "leftover carrier-frequency mismatch at the last interval",
    )
    if args.plot or args.plot_all:
        settings = sdr_settings_from_args(args)
        plot_loop_result(
            f"{algorithm} consensus over the physical layer "
            f"(--model {algorithm.replace('-', '')})\n"
            "Method (Rashid & Nanzer 2023): each node measures the other's "
            "pilot and retunes its own oscillator by\n"
            "half its estimated offset each interval, so the two drift "
            "toward each other (no master/slave roles)",
            settings.sync_interval,
            result,
        )


def run_micro(args: argparse.Namespace) -> None:
    result = run_micro_two_way_simulation(
        sdr_settings_from_args(args), micro_pilots_per_interval=args.micro_pilots
    )
    print(
        f"model: two-tier reciprocal sync ({args.micro_pilots} micro-pilots "
        "per interval)"
    )
    print(f"device: {result.device}")
    metric(
        "airtime fraction",
        f"{100.0 * result.airtime_fraction:.1f}%",
        "share of air time spent sending sync pilots instead of useful traffic — "
        "the cost of syncing more often",
    )
    metric(
        "detection rate",
        f"{100.0 * result.detection_rate:.2f}%",
        "how often the pilots (full frames and micro-pilots) were found",
    )
    metric(
        "steady-state oscillator phase residual RMS",
        f"{result.steady_state_phase_rms:.6g} rad",
        "true physical misalignment of the two crystals after corrections, once "
        "locked — extra micro-pilots correct drift sooner, shrinking this",
    )
    metric(
        "mean open-loop 2-station coherent gain",
        f"{100.0 * result.mean_coherent_gain:.2f}%",
        "if both stations transmitted together, fraction of the ideal combined "
        "power actually delivered; 100% = perfectly in phase",
    )
    metric(
        "final oscillator phase residual",
        f"{result.final_phase_error:.6g} rad",
        "crystal-to-crystal error at the very last step — one noisy sample",
    )
    metric(
        "final CFO residual",
        f"{result.final_frequency_error_hz:.6g} Hz",
        "leftover carrier-frequency mismatch at the last step",
    )
    if args.plot or args.plot_all:
        settings = sdr_settings_from_args(args)
        plot_loop_result(
            f"Two-tier micro-pilot synchronization (--model micro, "
            f"{args.micro_pilots} micro-pilots/interval)\n"
            "Method: one full two-way pilot exchange per interval (as in "
            "twoway), plus tiny reciprocal phase-only\n"
            "pilots in between — correcting several times per interval "
            "before drift can accumulate",
            settings.sync_interval / (args.micro_pilots + 1),
            result,
        )


def run_hybrid(args: argparse.Namespace) -> None:
    from hybrid_calibration import run_hybrid_simulation

    result = run_hybrid_simulation(
        sdr_settings_from_args(args),
        micro_pilots_per_interval=args.micro_pilots,
        anchor_every_intervals=args.anchor_every,
    )
    print(
        f"model: hybrid calibration ({args.micro_pilots} one-way micro-pilots "
        f"per interval, two-way anchor every {args.anchor_every} intervals)"
    )
    print(f"device: {result.device}")
    metric(
        "airtime fraction",
        f"{100.0 * result.airtime_fraction:.1f}%",
        "share of air time spent sending sync pilots instead of useful traffic — "
        "hybrid's point is to cut this by making most pilots cheap one-way ones",
    )
    metric(
        "detection rate",
        f"{100.0 * result.detection_rate:.2f}%",
        "how often the pilots (one-way frames, micro-pilots, anchors) were found",
    )
    metric(
        "steady-state oscillator phase residual RMS",
        f"{result.steady_state_phase_rms:.6g} rad",
        "true physical misalignment of the two crystals after corrections — the "
        "sparse two-way anchors keep the channel phase from leaking into it",
    )
    metric(
        "mean open-loop 2-station coherent gain",
        f"{100.0 * result.mean_coherent_gain:.2f}%",
        "if both stations transmitted together, fraction of the ideal combined "
        "power actually delivered; 100% = perfectly in phase",
    )
    metric(
        "final oscillator phase residual",
        f"{result.final_phase_error:.6g} rad",
        "crystal-to-crystal error at the very last step — one noisy sample",
    )
    metric(
        "final CFO residual",
        f"{result.final_frequency_error_hz:.6g} Hz",
        "leftover carrier-frequency mismatch at the last step",
    )
    if args.plot or args.plot_all:
        settings = sdr_settings_from_args(args)
        plot_loop_result(
            f"Hybrid calibration (--model hybrid, two-way anchor every "
            f"{args.anchor_every} intervals)\n"
            "Method: cheap ONE-way pilots track oscillator+channel phase "
            "jointly; an occasional two-way exchange\n"
            "(the anchor) separates the two again — most of two-way's "
            "accuracy at a fraction of the airtime",
            settings.sync_interval / (args.micro_pilots + 1),
            result,
        )


def run_dhybrid(args: argparse.Namespace) -> None:
    from hybrid_calibration import run_hybrid_simulation

    result = run_hybrid_simulation(
        sdr_settings_from_args(args),
        micro_pilots_per_interval=args.micro_pilots,
        anchor_every_intervals=args.anchor_every,
        decentralized=True,
    )
    print(
        f"model: DECENTRALIZED hybrid calibration ({args.micro_pilots} "
        f"one-way micro-pilots per interval, two-way anchor every "
        f"{args.anchor_every} intervals; no master — each node retunes "
        "half-way toward the other)"
    )
    print(f"device: {result.device}")
    metric(
        "airtime fraction",
        f"{100.0 * result.airtime_fraction:.1f}%",
        "share of air time spent sending sync pilots instead of useful traffic",
    )
    metric(
        "detection rate",
        f"{100.0 * result.detection_rate:.2f}%",
        "how often the pilots (one-way frames, micro-pilots, anchors) were found",
    )
    metric(
        "steady-state oscillator phase residual RMS",
        f"{result.steady_state_phase_rms:.6g} rad",
        "true physical misalignment of the two crystals after corrections — "
        "same estimator as hybrid; only the CONTROL is symmetric, so this "
        "should match the centralized hybrid",
    )
    metric(
        "mean open-loop 2-station coherent gain",
        f"{100.0 * result.mean_coherent_gain:.2f}%",
        "if both stations transmitted together, fraction of the ideal "
        "combined power actually delivered; 100% = perfectly in phase",
    )
    metric(
        "final oscillator phase residual",
        f"{result.final_phase_error:.6g} rad",
        "crystal-to-crystal error at the very last step — one noisy sample",
    )
    metric(
        "final CFO residual",
        f"{result.final_frequency_error_hz:.6g} Hz",
        "leftover carrier-frequency mismatch at the last step; note the "
        "PAIR's common (average) clock now drifts freely — decentralized "
        "control has no fixed datum",
    )
    if args.plot or args.plot_all:
        settings = sdr_settings_from_args(args)
        plot_loop_result(
            f"DECENTRALIZED hybrid calibration (--model dhybrid, anchor "
            f"every {args.anchor_every} intervals)\n"
            "Method: identical estimation to hybrid (cheap one-way pilots + "
            "sparse two-way anchors), but NO master —\n"
            "each node retunes HALF-way toward the other (symmetric "
            "consensus control); the pair meets at its average clock",
            settings.sync_interval / (args.micro_pilots + 1),
            result,
        )


def run_compare(args: argparse.Namespace) -> None:
    from hybrid_calibration import run_hybrid_simulation

    settings = sdr_settings_from_args(args)
    interval = settings.sync_interval
    substep = interval / (args.micro_pilots + 1)
    rows = []
    series = []
    for label, step_seconds, runner in (
        ("two-way EKF (ours)", interval, lambda: run_two_way_simulation(settings)),
        (
            "two-tier micro-pilot (ours)",
            substep,
            lambda: run_micro_two_way_simulation(
                settings, micro_pilots_per_interval=args.micro_pilots
            ),
        ),
        (
            "hybrid 1-way+anchors (ours)",
            substep,
            lambda: run_hybrid_simulation(
                settings,
                micro_pilots_per_interval=args.micro_pilots,
                anchor_every_intervals=args.anchor_every,
            ),
        ),
        (
            "decentralized hybrid (ours)",
            substep,
            lambda: run_hybrid_simulation(
                settings,
                micro_pilots_per_interval=args.micro_pilots,
                anchor_every_intervals=args.anchor_every,
                decentralized=True,
            ),
        ),
        (
            "DFPC naive (as published)",
            interval,
            lambda: run_consensus_ota_simulation(settings, "dfpc", reciprocal=False),
        ),
        (
            "DFPC + reciprocity",
            interval,
            lambda: run_consensus_ota_simulation(settings, "dfpc"),
        ),
        (
            "KF-DFPC + reciprocity",
            interval,
            lambda: run_consensus_ota_simulation(settings, "kf-dfpc"),
        ),
    ):
        result = runner()
        rows.append(
            (
                label,
                result.steady_state_phase_rms,
                result.mean_coherent_gain,
                result.detection_rate,
                result.airtime_fraction,
            )
        )
        series.append((label, step_seconds, result))
    print("open-loop synchronization comparison (identical physical conditions)")
    print(
        "columns: phase RMS = true crystal-to-crystal misalignment after\n"
        "correction, in thousandths of a radian (lower = better; 314 mrad = 18\n"
        "deg = the 90%-gain threshold); coherent gain = fraction of ideal\n"
        "combined power if both stations transmitted together; detect = how\n"
        "often pilots were found; airtime = share of transmission time spent\n"
        "on sync pilots (the cost of the accuracy)"
    )
    print(
        f"{'approach':<27}{'phase RMS (mrad)':>18}{'coherent gain':>15}"
        f"{'detect':>9}{'airtime':>10}"
    )
    for label, rms, gain, detect, airtime in rows:
        print(
            f"{label:<27}{1e3 * rms:>18.1f}{100.0 * gain:>14.2f}%"
            f"{100.0 * detect:>8.0f}%{100.0 * airtime:>9.1f}%"
        )
    print(
        "note: naive DFPC consenses on raw one-way measurements (the paper's\n"
        "channel-free assumption); over a real channel the wrapped symmetric\n"
        "update is bistable and can capture at the anti-phase fixed point,\n"
        "as it does for this channel realization. The reciprocity rows\n"
        "exchange measurements over the paper's assumed side channel."
    )
    if args.plot or args.plot_all:
        plot_method_comparison(series)


def plot_interval_sweep(model: str, rows) -> None:
    """Accuracy and cost against the pilot cadence, default 50 ms marked."""

    import numpy as np

    intervals_ms = np.array([row[0] for row in rows])
    rms_mrad = 1e3 * np.array([row[1] for row in rows])
    airtime_pct = 100.0 * np.array([row[3] for row in rows])

    def accuracy_panel(axis):
        axis.loglog(intervals_ms, rms_mrad, "o-", linewidth=1.4)
        axis.axhline(
            314.0,
            color="red",
            linestyle="--",
            linewidth=1.0,
            label="314 mrad = 18\N{DEGREE SIGN} = 90% coherent gain",
        )
        axis.axvline(
            50.0, color="gray", linestyle=":", linewidth=1.2, label="50 ms default"
        )
        axis.set_ylabel("steady residual RMS (mrad, log)")
        axis.set_xlabel("sync interval (ms, log)")
        axis.legend(fontsize="small")

    def cost_panel(axis):
        axis.semilogx(
            intervals_ms, airtime_pct, "o-", color="tab:orange", linewidth=1.4
        )
        axis.axhline(
            100.0,
            color="red",
            linestyle="--",
            linewidth=1.0,
            label="100% = channel completely consumed by pilots",
        )
        axis.axvline(
            50.0, color="gray", linestyle=":", linewidth=1.2, label="50 ms default"
        )
        axis.set_ylabel("airtime (%)")
        axis.set_xlabel("sync interval (ms, log)")
        axis.legend(fontsize="small")

    _render_figure_and_panels(
        f"Choosing the pilot cadence (--model {model})\n"
        "The default 50 ms is one point on this trade curve, not a magic "
        "number: pick the longest interval whose\n"
        "accuracy still meets your target, because that costs the least "
        "airtime.",
        [
            (
                "Accuracy: the longer between pilots, the more the crystals "
                "drift before each correction",
                accuracy_panel,
            ),
            (
                "Cost: the shorter between pilots, the more of the channel "
                "sync consumes",
                cost_panel,
            ),
        ],
        (2, 1),
        (10.5, 8),
        top=0.9,
    )


def _model_metrics(
    args: argparse.Namespace, settings: SDRSimulationConfig
) -> tuple[float, float, float, float]:
    """Run one model once: (residual RMS, mean gain, detect rate, airtime)."""

    if args.model == "twoway":
        result = run_two_way_simulation(settings)
        return (
            result.steady_state_phase_rms,
            result.mean_coherent_gain,
            result.detection_rate,
            result.airtime_fraction,
        )
    if args.model == "micro":
        result = run_micro_two_way_simulation(
            settings, micro_pilots_per_interval=args.micro_pilots
        )
        return (
            result.steady_state_phase_rms,
            result.mean_coherent_gain,
            result.detection_rate,
            result.airtime_fraction,
        )
    if args.model in ("hybrid", "dhybrid"):
        from hybrid_calibration import run_hybrid_simulation

        result = run_hybrid_simulation(
            settings,
            micro_pilots_per_interval=args.micro_pilots,
            anchor_every_intervals=args.anchor_every,
            decentralized=args.model == "dhybrid",
        )
        return (
            result.steady_state_phase_rms,
            result.mean_coherent_gain,
            result.detection_rate,
            result.airtime_fraction,
        )
    if args.model in ("dfpc", "kfdfpc"):
        algorithm = "dfpc" if args.model == "dfpc" else "kf-dfpc"
        result = run_consensus_ota_simulation(settings, algorithm)
        return (
            result.steady_state_phase_rms,
            result.mean_coherent_gain,
            result.detection_rate,
            result.airtime_fraction,
        )
    result = run_sdr_simulation(settings)
    steady = result.detected & result.correction_active
    if torch.any(steady):
        rms = torch.sqrt(
            torch.mean(result.post_correction_ota_phase[steady].square())
        ).item()
        gain = torch.mean(
            torch.cos(result.post_correction_oscillator_phase[steady] / 2.0).square()
        ).item()
    else:
        rms = float("nan")
        gain = float("nan")
    return rms, gain, result.detection_rate, result.airtime_fraction


def run_interval_sweep(args: argparse.Namespace) -> None:
    """Run the chosen model at several pilot cadences and compare."""

    from dataclasses import replace

    if args.model in ("compare", "ideal"):
        print("--sweep-interval supports sdr, twoway, micro, hybrid, dfpc, kfdfpc")
        return

    intervals_ms = [float(value) for value in args.sweep_interval.split(",")]
    base = sdr_settings_from_args(args)
    rows = []
    print(f"sync-interval sweep, --model {args.model} "
          f"({args.iterations} intervals per point)")
    if args.model == "sdr":
        print(
            "note: residual is the observable link phase (what one-way "
            "controls); gain is the open-loop crystal gain, which one-way "
            "does not optimize"
        )
    for interval_ms in intervals_ms:
        settings = replace(base, sync_interval=interval_ms * 1e-3)
        rms, gain, _, airtime = _model_metrics(args, settings)
        rows.append((interval_ms, rms, gain, airtime))
        note = ""
        if airtime >= 1.0:
            note = "  <- pilots do not fit in the interval"
        print(
            f"  {interval_ms:>6g} ms: residual RMS {1e3 * rms:>8.1f} mrad, "
            f"mean gain {100.0 * gain:>6.2f}%, "
            f"airtime {100.0 * airtime:>6.1f}%{note}"
        )
    if args.plot or args.plot_all:
        plot_interval_sweep(args.model, rows)


def _mean_std(values: list[float]) -> tuple[float, float]:
    import statistics

    clean = [value for value in values if value == value]  # drop NaN
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.mean(clean), statistics.stdev(clean)


def run_monte_carlo(args: argparse.Namespace) -> None:
    """Repeat the chosen model over consecutive seeds; report mean +- std.

    Every number the single-seed runs print is one channel/noise
    realization; this turns them into statistically defensible values.
    """

    from dataclasses import replace

    if args.model == "ideal":
        print("--seeds supports sdr, twoway, micro, hybrid, dfpc, kfdfpc, compare")
        return

    seeds = list(range(args.seed, args.seed + args.seeds))
    base = sdr_settings_from_args(args)

    if args.model == "compare":
        from hybrid_calibration import run_hybrid_simulation

        methods = (
            ("two-way EKF", lambda s: run_two_way_simulation(s)),
            (
                "two-tier micro-pilot",
                lambda s: run_micro_two_way_simulation(
                    s, micro_pilots_per_interval=args.micro_pilots
                ),
            ),
            (
                "hybrid 1-way+anchors",
                lambda s: run_hybrid_simulation(
                    s,
                    micro_pilots_per_interval=args.micro_pilots,
                    anchor_every_intervals=args.anchor_every,
                ),
            ),
            (
                "DFPC + reciprocity",
                lambda s: run_consensus_ota_simulation(s, "dfpc"),
            ),
            (
                "KF-DFPC + reciprocity",
                lambda s: run_consensus_ota_simulation(s, "kf-dfpc"),
            ),
        )
        print(
            f"Monte Carlo comparison over seeds {seeds[0]}..{seeds[-1]} "
            f"({args.seeds} runs per method, mean +- std)"
        )
        for label, runner in methods:
            rms_values, gain_values = [], []
            for seed in seeds:
                result = runner(replace(base, seed=seed))
                rms_values.append(result.steady_state_phase_rms)
                gain_values.append(result.mean_coherent_gain)
            rms_mean, rms_std = _mean_std(rms_values)
            gain_mean, gain_std = _mean_std(gain_values)
            print(
                f"  {label:<22} residual {1e3 * rms_mean:>7.1f} +- "
                f"{1e3 * rms_std:>5.1f} mrad, gain {100.0 * gain_mean:>6.2f} "
                f"+- {100.0 * gain_std:.2f}%"
            )
        return

    print(
        f"Monte Carlo, --model {args.model}, seeds {seeds[0]}..{seeds[-1]} "
        f"({args.seeds} runs, {args.iterations} intervals each)"
    )
    rms_values, gain_values, detect_values = [], [], []
    airtime = float("nan")
    for seed in seeds:
        rms, gain, detect, airtime = _model_metrics(
            args, replace(base, seed=seed)
        )
        rms_values.append(rms)
        gain_values.append(gain)
        detect_values.append(detect)
        print(
            f"  seed {seed:>3}: residual RMS {1e3 * rms:>8.1f} mrad, "
            f"gain {100.0 * gain:>6.2f}%, detect {100.0 * detect:>5.1f}%"
        )
    rms_mean, rms_std = _mean_std(rms_values)
    gain_mean, gain_std = _mean_std(gain_values)
    detect_mean, detect_std = _mean_std(detect_values)
    print("mean +- std over seeds:")
    print(f"  steady residual RMS: {1e3 * rms_mean:.1f} +- {1e3 * rms_std:.1f} mrad")
    print(f"  mean coherent gain:  {100.0 * gain_mean:.2f} +- {100.0 * gain_std:.2f} %")
    print(f"  detection rate:      {100.0 * detect_mean:.1f} +- {100.0 * detect_std:.1f} %")
    print(f"  airtime fraction:    {100.0 * airtime:.1f} % (deterministic)")


def _network_link_scheme(args: argparse.Namespace):
    """(link_runner, extract, step_seconds_fn) for the chosen model."""

    def loop_extract(result):
        mask = result.detected & result.correction_active & result.calibrated
        return (
            result.post_correction_phase,
            mask,
            result.detection_rate,
            result.airtime_fraction,
        )

    if args.model == "twoway":
        return (
            run_two_way_simulation,
            loop_extract,
            lambda settings: settings.sync_interval,
        )
    if args.model == "micro":
        return (
            lambda settings: run_micro_two_way_simulation(
                settings, micro_pilots_per_interval=args.micro_pilots
            ),
            loop_extract,
            lambda settings: settings.sync_interval / (args.micro_pilots + 1),
        )
    if args.model == "hybrid":
        from hybrid_calibration import run_hybrid_simulation

        return (
            lambda settings: run_hybrid_simulation(
                settings,
                micro_pilots_per_interval=args.micro_pilots,
                anchor_every_intervals=args.anchor_every,
            ),
            loop_extract,
            lambda settings: settings.sync_interval / (args.micro_pilots + 1),
        )
    if args.model == "sdr":

        def sdr_extract(result):
            mask = result.detected & result.correction_active
            return (
                result.post_correction_oscillator_phase,
                mask,
                result.detection_rate,
                result.airtime_fraction,
            )

        return (
            run_sdr_simulation,
            sdr_extract,
            lambda settings: settings.sync_interval,
        )
    return None


def _run_network_once(args: argparse.Namespace, num_stations: int):
    scheme = _network_link_scheme(args)
    if scheme is None:
        return None, None
    link_runner, extract, step_fn = scheme
    settings = sdr_settings_from_args(args)
    result = run_network_simulation(
        settings,
        num_stations,
        link_runner,
        extract,
        radius_m=args.area_radius_m,
        path_loss_exponent=args.path_loss_exponent,
        reference_distance_m=args.ref_distance_m,
    )
    return result, step_fn(settings)


def plot_network_result(args, result, step_seconds: float) -> None:
    import numpy as np

    positions = result.positions
    time = (np.arange(result.array_gain.shape[0]) + 1) * step_seconds

    def map_panel(axis):
        axis.scatter(
            positions[1:, 0], positions[1:, 1], s=40, label="stations"
        )
        axis.scatter(
            positions[0, 0],
            positions[0, 1],
            s=140,
            marker="*",
            color="tab:red",
            label="reference (station 0)",
            zorder=3,
        )
        for link in result.links:
            axis.plot(
                [positions[0, 0], positions[link.station, 0]],
                [positions[0, 1], positions[link.station, 1]],
                color="gray",
                linewidth=0.7,
                alpha=0.5,
            )
            axis.annotate(
                f"{link.snr_db:.0f} dB",
                positions[link.station],
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_aspect("equal")
        axis.legend(fontsize="small")

    def residual_panel(axis):
        for link in result.links:
            residual = np.abs(link.residual.numpy())
            axis.semilogy(
                time,
                1e3 * residual + 1e-2,
                linewidth=1.0,
                alpha=0.8,
                label=(
                    f"station {link.station} ({link.distance_m:.0f} m, "
                    f"{link.snr_db:.0f} dB, {1e3 * link.steady_rms:.0f} mrad)"
                ),
            )
        axis.axhline(
            314.0,
            color="red",
            linestyle="--",
            linewidth=1.0,
            label="314 mrad = 18\N{DEGREE SIGN}",
        )
        axis.set_ylabel("|residual| (mrad, log)")
        axis.set_xlabel("time (s)")
        axis.legend(fontsize="x-small", ncols=2)

    def gain_panel(axis):
        axis.plot(
            time,
            100.0 * result.array_gain.numpy(),
            linewidth=1.3,
            label=(
                f"{result.num_stations}-station array gain "
                f"(steady mean {100.0 * result.mean_array_gain:.1f}%)"
            ),
        )
        axis.axhline(
            90.0, color="red", linestyle="--", linewidth=1.0, label="90%"
        )
        axis.set_ylabel("array coherent gain (%)")
        axis.set_xlabel("time (s)")
        axis.set_ylim(-2.0, 102.0)
        axis.legend(fontsize="small")

    airtime = result.total_airtime_fraction
    airtime_note = "" if airtime < 1.0 else " — DOES NOT FIT, pilots overrun the channel"
    _render_figure_and_panels(
        f"{result.num_stations}-station synchronization (--model {args.model}, "
        "star topology to the reference)\n"
        f"random deployment in a {args.area_radius_m:.0f} m disc, path-loss "
        f"exponent {args.path_loss_exponent:g}, SNR {args.snr_db:.0f} dB at "
        f"{args.ref_distance_m:.0f} m\n"
        f"total pilot airtime {100.0 * airtime:.0f}%{airtime_note}",
        [
            (
                "Where the stations landed (annotation: per-link SNR after "
                "path loss)",
                map_panel,
            ),
            (
                "Per-station true crystal residual vs the reference — "
                "distant/noisy links track worse",
                residual_panel,
            ),
            (
                "N-station array coherent gain: fraction of the ideal N^2 "
                "power if ALL stations transmitted together",
                gain_panel,
            ),
        ],
        (3, 1),
        (10.5, 12),
        sharex=False,
        top=0.9,
    )


def run_mesh_network(args: argparse.Namespace) -> None:
    """N-node decentralized hybrid: a chain mesh, no reference station."""

    from hybrid_calibration import run_decentralized_hybrid_mesh

    settings = sdr_settings_from_args(args)
    result = run_decentralized_hybrid_mesh(
        settings,
        num_nodes=args.stations,
        micro_pilots_per_interval=args.micro_pilots,
        anchor_every_intervals=args.anchor_every,
        radius_m=args.area_radius_m,
        path_loss_exponent=args.path_loss_exponent,
        reference_distance_m=args.ref_distance_m,
        control=args.mesh_control,
    )
    print(
        f"{result.num_nodes}-station DECENTRALIZED hybrid mesh (no "
        f"designated reference): nearest-neighbor chain, "
        f"'{args.mesh_control}' control law"
    )
    print(f"chain order (nearest-neighbor from station 0): {result.chain}")
    print(f"{'edge':>12}{'distance':>10}{'link SNR':>10}{'steady residual':>17}")
    for index in range(len(result.edge_distances_m)):
        pair = f"{result.chain[index]}-{result.chain[index + 1]}"
        rms = result.edge_steady_rms[index]
        print(
            f"{pair:>12}{result.edge_distances_m[index]:>9.0f}m"
            f"{result.edge_snrs_db[index]:>8.1f}dB{1e3 * rms:>12.1f} mrad"
        )
    metric(
        "detection rate (all edges)",
        f"{100.0 * result.detection_rate:.2f}%",
        "fraction of sub-steps where every edge's pilot was found",
    )
    metric(
        "N-station array coherent gain (steady mean)",
        f"{100.0 * result.mean_array_gain:.2f}%",
        "datum-free |sum of station phasors|^2 / N^2 — decentralized "
        "control has no reference, so this is measured about the array's "
        "own average clock",
    )
    metric(
        "total pilot airtime",
        f"{100.0 * result.airtime_fraction:.1f}%",
        "per-edge hybrid pilot cost summed over the chain (no broadcast "
        "amortization credited — conservative)",
    )
    if args.plot or args.plot_all:
        plot_mesh_result(args, result, settings)


def plot_mesh_result(args, result, settings) -> None:
    import numpy as np

    positions = result.positions
    step_seconds = settings.sync_interval / (args.micro_pilots + 1)
    time = (np.arange(result.array_gain.shape[0]) + 1) * step_seconds

    def map_panel(axis):
        axis.scatter(positions[:, 0], positions[:, 1], s=50)
        for index, (x, y) in enumerate(positions):
            axis.annotate(
                str(index), (x, y), fontsize=9, xytext=(5, 5),
                textcoords="offset points",
            )
        for k in range(len(result.chain) - 1):
            p, q = result.chain[k], result.chain[k + 1]
            axis.plot(
                [positions[p, 0], positions[q, 0]],
                [positions[p, 1], positions[q, 1]],
                color="tab:green",
                linewidth=1.2,
                alpha=0.8,
            )
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_aspect("equal")

    def residual_panel(axis):
        for index, row in enumerate(result.edge_residuals):
            pair = f"{result.chain[index]}-{result.chain[index + 1]}"
            rms = result.edge_steady_rms[index]
            axis.semilogy(
                time,
                1e3 * np.abs(row.numpy()) + 1e-2,
                linewidth=1.0,
                alpha=0.85,
                label=f"edge {pair} ({1e3 * rms:.0f} mrad)",
            )
        axis.axhline(
            314.0, color="red", linestyle="--", linewidth=1.0,
            label="314 mrad = 18\N{DEGREE SIGN}",
        )
        axis.set_ylabel("|edge residual| (mrad, log)")
        axis.set_xlabel("time (s)")
        axis.legend(fontsize="x-small", ncols=2)

    def gain_panel(axis):
        axis.plot(
            time,
            100.0 * result.array_gain.numpy(),
            linewidth=1.3,
            label=(
                f"{result.num_nodes}-station array gain "
                f"(steady mean {100.0 * result.mean_array_gain:.1f}%)"
            ),
        )
        axis.axhline(
            90.0, color="red", linestyle="--", linewidth=1.0, label="90%"
        )
        axis.set_ylabel("array coherent gain (%)")
        axis.set_xlabel("time (s)")
        axis.set_ylim(-2.0, 102.0)
        axis.legend(fontsize="small")

    _render_figure_and_panels(
        f"{result.num_nodes}-station DECENTRALIZED hybrid mesh "
        "(--model dhybrid --stations N)\n"
        "No reference: nearest-neighbor chain, symmetric degree-weighted "
        "corrections, periodic 1-bit branch checks.\n"
        "Interior edges pay a consensus tax (under-relaxed + disturbed by "
        "neighbors) — compare with the centralized star.",
        [
            ("Deployment and the chain the mesh formed", map_panel),
            ("Per-edge crystal residuals", residual_panel),
            ("Array coherent gain about the array's own average clock",
             gain_panel),
        ],
        (3, 1),
        (10.5, 12),
        sharex=False,
        top=0.9,
    )


def run_network(args: argparse.Namespace) -> None:
    if args.model == "dhybrid":
        run_mesh_network(args)
        return
    result, step_seconds = _run_network_once(args, args.stations)
    if result is None:
        print(
            "--stations supports --model sdr, twoway, micro, hybrid "
            "(star topology) and dhybrid (decentralized chain mesh); the "
            "physical-layer consensus models are 2-node"
        )
        return
    print(
        f"{result.num_stations}-station network, --model {args.model}, "
        "star topology: every station syncs to station 0 over a shared "
        "TDMA channel"
    )
    print(
        f"deployment: uniform random in a {args.area_radius_m:.0f} m disc "
        f"(seed {args.seed}); path-loss exponent "
        f"{args.path_loss_exponent:g}; SNR {args.snr_db:.0f} dB at "
        f"{args.ref_distance_m:.0f} m"
    )
    print(
        f"{'station':>8}{'distance':>10}{'link SNR':>10}"
        f"{'steady residual':>17}{'detect':>8}"
    )
    for link in result.links:
        print(
            f"{link.station:>8}{link.distance_m:>9.0f}m"
            f"{link.snr_db:>8.1f}dB{1e3 * link.steady_rms:>12.1f} mrad"
            f"{100.0 * link.detection_rate:>7.0f}%"
        )
    metric(
        "N-station array coherent gain (steady mean)",
        f"{100.0 * result.mean_array_gain:.2f}%",
        "fraction of the ideal N^2 combined power if all stations "
        "transmitted together; errors compound across stations, so this "
        "falls as the array grows",
    )
    airtime = result.total_airtime_fraction
    note = (
        "the links share one channel (TDMA), so pilot cost grows linearly "
        "with N"
    )
    if airtime >= 1.0:
        note += " — DOES NOT FIT: pilots alone overrun the channel"
    metric(
        "total pilot airtime",
        f"{100.0 * airtime:.1f}%",
        note,
    )
    if args.plot or args.plot_all:
        plot_network_result(args, result, step_seconds)


def run_station_sweep(args: argparse.Namespace) -> None:
    counts = [int(value) for value in args.sweep_stations.split(",")]
    if _network_link_scheme(args) is None:
        print(
            "--sweep-stations supports --model sdr, twoway, micro, hybrid"
        )
        return
    print(
        f"station-count sweep, --model {args.model} "
        f"({args.iterations} intervals per network)"
    )
    rows = []
    for count in counts:
        result, _ = _run_network_once(args, count)
        rows.append(
            (
                count,
                result.mean_array_gain,
                result.worst_station_rms,
                result.total_airtime_fraction,
                result.min_detection_rate,
            )
        )
        note = "" if result.total_airtime_fraction < 1.0 else "  <- does not fit"
        print(
            f"  N={count:>3}: array gain {100.0 * result.mean_array_gain:>6.2f}%, "
            f"worst station {1e3 * result.worst_station_rms:>7.1f} mrad, "
            f"airtime {100.0 * result.total_airtime_fraction:>6.1f}%, "
            f"min detect {100.0 * result.min_detection_rate:>5.1f}%{note}"
        )
    if not (args.plot or args.plot_all):
        return

    import numpy as np

    stations = np.array([row[0] for row in rows])
    gains = 100.0 * np.array([row[1] for row in rows])
    worst = 1e3 * np.array([row[2] for row in rows])
    airtimes = 100.0 * np.array([row[3] for row in rows])

    def gain_panel(axis):
        axis.plot(stations, gains, "o-", linewidth=1.4)
        axis.axhline(
            90.0, color="red", linestyle="--", linewidth=1.0, label="90%"
        )
        axis.set_ylabel("array coherent gain (%)")
        axis.set_xlabel("number of stations")
        axis.set_ylim(-2.0, 102.0)
        axis.legend(fontsize="small")

    def worst_panel(axis):
        axis.semilogy(stations, worst, "o-", linewidth=1.4, color="tab:green")
        axis.axhline(
            314.0,
            color="red",
            linestyle="--",
            linewidth=1.0,
            label="314 mrad = 18\N{DEGREE SIGN}",
        )
        axis.set_ylabel("worst-station residual (mrad, log)")
        axis.set_xlabel("number of stations")
        axis.legend(fontsize="small")

    def airtime_panel(axis):
        axis.plot(stations, airtimes, "o-", linewidth=1.4, color="tab:orange")
        axis.axhline(
            100.0,
            color="red",
            linestyle="--",
            linewidth=1.0,
            label="100% = pilots alone consume the channel",
        )
        axis.set_ylabel("total pilot airtime (%)")
        axis.set_xlabel("number of stations")
        axis.legend(fontsize="small")

    _render_figure_and_panels(
        f"Scalability (--model {args.model}): coherence and cost vs array "
        "size\n"
        "Bigger arrays promise N^2 combined power, but per-station errors "
        "compound (gain falls) and TDMA pilot\n"
        "airtime grows linearly until sync alone consumes the channel — "
        "the two scaling walls.",
        [
            ("Coherence: N-station array gain", gain_panel),
            ("Accuracy: worst station's residual", worst_panel),
            ("Cost: total pilot airtime", airtime_panel),
        ],
        (3, 1),
        (10.5, 11),
        sharex=True,
        top=0.9,
    )


def main() -> None:
    args = parse_args()
    if args.model != "ideal" and args.oscillator != LEGACY_PROFILE_NAME:
        profile = OSCILLATOR_PROFILES[args.oscillator]
        print(f"oscillator profile: {args.oscillator} — {profile.description}")
    if args.seeds > 1 and args.sweep_interval:
        print(
            "--seeds and --sweep-interval cannot be combined; "
            "running the sweep at the base seed only"
        )
    if args.sweep_interval:
        run_interval_sweep(args)
        return
    if args.sweep_stations:
        run_station_sweep(args)
        return
    if args.stations is not None:
        if args.stations < 2:
            print("--stations needs at least 2")
            return
        run_network(args)
        return
    if args.seeds > 1:
        run_monte_carlo(args)
        return
    if args.plot_iq and args.model != "ideal":
        plot_iq_diagnostics(sdr_settings_from_args(args))
    if args.model == "ideal":
        run_ideal(args)
    elif args.model == "twoway":
        run_twoway(args)
    elif args.model == "micro":
        run_micro(args)
    elif args.model == "hybrid":
        run_hybrid(args)
    elif args.model == "dhybrid":
        run_dhybrid(args)
    elif args.model == "dfpc":
        run_consensus(args, "dfpc")
    elif args.model == "kfdfpc":
        run_consensus(args, "kf-dfpc")
    elif args.model == "compare":
        run_compare(args)
    else:
        run_sdr(args)


if __name__ == "__main__":
    main()
