"""Command-line entry point for the Sionna OTA synchronization examples."""

from __future__ import annotations

import argparse
import math

import torch

from ota_sync import (
    SDRSimulationConfig,
    SDRSimulationResult,
    SimulationConfig,
    evaluate_csi_joint_transmission,
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
            "dfpc",
            "kfdfpc",
            "compare",
            "ideal",
        ),
        default="sdr",
        help="one-way sampled-IQ SDR model (default), reciprocal two-way "
        "sync, two-tier micro-pilot sync, Rashid & Nanzer's consensus "
        "algorithms over the physical layer (dfpc/kfdfpc), a side-by-side "
        "comparison of all open-loop approaches, or the ideal AWGN model",
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
        "--plot", action="store_true", help="display the phase-synchronization plot"
    )
    parser.add_argument(
        "--plot-all",
        action="store_true",
        help="display full SDR acquisition and channel diagnostics",
    )

    sdr = parser.add_argument_group("SDR model")
    sdr.add_argument("--sample-rate", type=float, default=1e6, help="IQ sample rate")
    sdr.add_argument("--carrier-mhz", type=float, default=915.0)
    sdr.add_argument("--cfo-hz", type=float, default=1500.0)
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
        default=0.05,
        help="RMS flicker FM frequency deviation of the references",
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


def plot_ideal_result(result) -> None:
    import matplotlib.pyplot as plt

    iteration = range(len(result.true_phase))
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

    axes[0, 0].plot(iteration, result.true_phase, label="true")
    axes[0, 0].plot(iteration, result.estimated_phase, "--", label="EKF")
    axes[0, 0].set_ylabel("phase (rad)")
    axes[0, 0].set_title("Pre-correction relative phase")
    axes[0, 0].legend()

    axes[0, 1].plot(iteration, result.true_frequency, label="true")
    axes[0, 1].plot(iteration, result.estimated_frequency, "--", label="EKF")
    axes[0, 1].set_ylabel("angular frequency (rad/s)")
    axes[0, 1].set_title("Pre-correction relative frequency")
    axes[0, 1].legend()

    axes[1, 0].plot(iteration, result.post_correction_phase)
    axes[1, 0].set_ylabel("phase residual (rad)")
    axes[1, 0].set_xlabel("synchronization interval")
    axes[1, 0].set_title("After correction")

    axes[1, 1].plot(iteration, result.post_correction_frequency)
    axes[1, 1].set_ylabel("frequency residual (rad/s)")
    axes[1, 1].set_xlabel("synchronization interval")
    axes[1, 1].set_title("After correction")

    for axis in axes.flat:
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    plt.show()


def plot_sdr_result(result: SDRSimulationResult) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    iteration = np.arange(len(result.true_phase))
    true_ota_phase = result.true_ota_phase.numpy()
    measured_phase = result.measured_ota_phase.numpy()
    estimated_phase = result.estimated_ota_phase.numpy()
    post_phase_mrad = 1e3 * result.post_correction_ota_phase.numpy()
    steady = (result.detected & result.correction_active).numpy()
    if np.any(steady):
        post_rms_mrad = float(np.sqrt(np.mean(np.square(post_phase_mrad[steady]))))
    else:
        post_rms_mrad = float("nan")

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(iteration, true_ota_phase, linewidth=1.8, label="true OTA")
    axes[0].plot(iteration, measured_phase, linewidth=1.0, alpha=0.7, label="measured")
    axes[0].plot(iteration, estimated_phase, "--", linewidth=1.3, label="EKF estimate")
    axes[0].axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    axes[0].set_ylabel("phase (rad)")
    axes[0].set_title("OTA phase before correction")
    axes[0].legend()

    axes[1].plot(iteration, post_phase_mrad, linewidth=1.4, label="residual")
    axes[1].axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    axes[1].axhline(
        post_rms_mrad,
        color="tab:red",
        linestyle=":",
        linewidth=1.0,
        label=f"steady RMS = {post_rms_mrad:.3f} mrad",
    )
    axes[1].axhline(
        -post_rms_mrad,
        color="tab:red",
        linestyle=":",
        linewidth=1.0,
    )
    axes[1].set_ylabel("phase residual (mrad)")
    axes[1].set_xlabel("synchronization interval")
    axes[1].set_title("OTA phase after correction")
    axes[1].legend()

    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.suptitle(
        "OTA phase synchronization\n"
        f"tracking RMSE={1e3 * result.ota_phase_rmse:.3f} mrad, "
        f"steady residual RMS={post_rms_mrad:.3f} mrad",
        fontsize=13,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    plt.show()


def plot_sdr_diagnostics(result: SDRSimulationResult) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    iteration = np.arange(len(result.true_phase))
    figure, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)

    true_ota_phase = result.true_ota_phase.numpy()
    measured_phase = result.measured_ota_phase.numpy()
    estimated_phase = result.estimated_ota_phase.numpy()
    oscillator_phase = np.unwrap(result.true_phase.numpy())
    channel_phase = np.unwrap(result.channel_phase.numpy())
    effective_phase = oscillator_phase + channel_phase

    axes[0, 0].plot(iteration, true_ota_phase, linewidth=1.7, label="true")
    axes[0, 0].plot(
        iteration, measured_phase, linewidth=1.0, alpha=0.7, label="measured"
    )
    axes[0, 0].plot(
        iteration, estimated_phase, "--", linewidth=1.2, label="EKF estimate"
    )
    axes[0, 0].axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    axes[0, 0].set_ylabel("phase (rad)")
    axes[0, 0].set_title("Observable OTA phase before correction")
    axes[0, 0].legend()

    axes[0, 1].plot(iteration, oscillator_phase, label="oscillator")
    axes[0, 1].plot(iteration, channel_phase, label="channel")
    axes[0, 1].plot(
        iteration,
        effective_phase,
        "--",
        linewidth=1.5,
        label="observable sum",
    )
    axes[0, 1].axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    axes[0, 1].set_ylabel("phase (rad)")
    axes[0, 1].set_title("Unwrapped phase decomposition")
    axes[0, 1].legend(ncols=3, fontsize="small")

    axes[1, 0].plot(
        iteration,
        result.true_ota_frequency / (2.0 * math.pi),
        linewidth=1.5,
        label="true",
    )
    axes[1, 0].plot(
        iteration,
        result.estimated_frequency / (2.0 * math.pi),
        "--",
        linewidth=1.2,
        label="EKF",
    )
    axes[1, 0].axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    axes[1, 0].set_yscale("symlog", linthresh=1.0)
    axes[1, 0].set_ylabel("CFO (Hz)")
    axes[1, 0].set_title("CFO acquisition and tracking (symmetric log scale)")
    axes[1, 0].legend()

    post_phase_mrad = 1e3 * result.post_correction_ota_phase.numpy()
    axes[1, 1].plot(iteration, post_phase_mrad, linewidth=1.3)
    axes[1, 1].axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    axes[1, 1].set_ylabel("phase residual (mrad)")
    axes[1, 1].set_title("Post-correction OTA phase")

    detection_metric = result.detection_metric.numpy()
    detected = result.detected.numpy()
    axes[2, 0].plot(iteration, detection_metric, linewidth=1.3, label="score")
    axes[2, 0].axhline(
        0.25,
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
        label="threshold",
    )
    if np.any(~detected):
        axes[2, 0].scatter(
            iteration[~detected],
            detection_metric[~detected],
            color="tab:red",
            marker="x",
            label="missed",
            zorder=3,
        )
    axes[2, 0].set_ylabel("normalized score")
    axes[2, 0].set_xlabel("synchronization interval")
    axes[2, 0].set_title("Packet detection confidence")
    axes[2, 0].set_ylim(0.0, 1.05)
    axes[2, 0].legend(fontsize="small")

    timing_error = result.timing_error_samples.numpy()
    axes[2, 1].step(iteration, timing_error, where="mid")
    axes[2, 1].axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    axes[2, 1].set_ylabel("timing error (samples)")
    axes[2, 1].set_xlabel("synchronization interval")
    axes[2, 1].set_title("Packet timing")
    if np.all(timing_error == 0.0):
        axes[2, 1].set_ylim(-0.5, 0.5)
        axes[2, 1].text(
            0.5,
            0.85,
            "All packets acquired at the correct sample",
            ha="center",
            transform=axes[2, 1].transAxes,
            fontsize="small",
        )

    for axis in axes.flat:
        axis.grid(True, alpha=0.3)
    figure.suptitle(
        "SDR OTA synchronization diagnostics\n"
        f"detection={100.0 * result.detection_rate:.1f}%, "
        f"OTA phase RMSE={1e3 * result.ota_phase_rmse:.3f} mrad, "
        f"CFO RMSE={result.frequency_rmse / (2.0 * math.pi):.3f} Hz",
        fontsize=13,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    plt.show()


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
    print(f"phase RMSE: {result.phase_rmse:.6g} rad")
    print(f"frequency RMSE: {result.frequency_rmse:.6g} rad/s")
    print(f"final phase residual: {result.final_phase_error:.6g} rad")
    print(f"final frequency residual: {result.final_frequency_error:.6g} rad/s")
    if args.plot or args.plot_all:
        plot_ideal_result(result)


def sdr_settings_from_args(args: argparse.Namespace) -> SDRSimulationConfig:
    settings_values = {
        "num_iterations": args.iterations,
        "snr_db": args.snr_db,
        "sample_rate": args.sample_rate,
        "carrier_frequency_hz": args.carrier_mhz * 1e6,
        "slave_initial_frequency_hz": args.cfo_hz,
        "sample_clock_offset_ppm": args.sfo_ppm,
        "flicker_frequency_std_hz": args.flicker_std_hz,
        "shadowing_std_db": args.shadowing_std_db,
        "tdl_model": args.tdl_model,
        "delay_spread_s": args.delay_spread_ns * 1e-9,
        "channel_speed_mps": args.speed_mps,
        "adc_bits": args.adc_bits,
        "correction_latency_intervals": args.correction_latency,
        "device": args.device,
        "seed": args.seed,
    }
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
    print(f"packet detection rate: {100.0 * result.detection_rate:.2f}%")
    print(f"OTA phase tracking RMSE: {result.ota_phase_rmse:.6g} rad")
    print(f"frequency tracking RMSE: {result.frequency_rmse / (2.0 * math.pi):.6g} Hz")
    print(f"steady-state OTA phase residual RMS: {post_ota_rms:.6g} rad")
    print(f"final OTA phase residual: {result.final_ota_phase_error:.6g} rad")
    print(
        "final raw oscillator phase residual: "
        f"{result.final_oscillator_phase_error:.6g} rad"
    )
    print(f"final CFO residual: {result.final_frequency_error_hz:.6g} Hz")
    if args.csi_gain:
        gains = evaluate_csi_joint_transmission(result, seed=args.seed)
        print("coherent JT gain at user vs. CSI refresh cadence:")
        for refresh, gain in gains.items():
            print(f"  every {refresh:>2} interval(s): {100.0 * gain:.2f}%")
    if args.plot_all:
        plot_sdr_diagnostics(result)
    elif args.plot:
        plot_sdr_result(result)


def run_twoway(args: argparse.Namespace) -> None:
    result = run_two_way_simulation(sdr_settings_from_args(args))
    print("model: reciprocal two-way SDR (open-loop coherence)")
    print(f"device: {result.device}")
    print(f"detection rate (both directions): {100.0 * result.detection_rate:.2f}%")
    print(f"oscillator phase tracking RMSE: {result.phase_rmse:.6g} rad")
    print(
        "steady-state oscillator phase residual RMS: "
        f"{result.steady_state_phase_rms:.6g} rad"
    )
    print(
        "mean open-loop 2-station coherent gain: "
        f"{100.0 * result.mean_coherent_gain:.2f}%"
    )
    print(f"final oscillator phase residual: {result.final_phase_error:.6g} rad")
    print(f"final CFO residual: {result.final_frequency_error_hz:.6g} Hz")


def run_consensus(args: argparse.Namespace, algorithm: str) -> None:
    result = run_consensus_ota_simulation(sdr_settings_from_args(args), algorithm)
    print(f"model: {algorithm} over the sampled-IQ physical layer (2 nodes)")
    print(f"device: {result.device}")
    print(f"detection rate (both directions): {100.0 * result.detection_rate:.2f}%")
    print(
        "steady-state oscillator phase residual RMS: "
        f"{result.steady_state_phase_rms:.6g} rad"
    )
    print(
        "mean open-loop 2-station coherent gain: "
        f"{100.0 * result.mean_coherent_gain:.2f}%"
    )
    print(f"final oscillator phase residual: {result.final_phase_error:.6g} rad")
    print(f"final CFO residual: {result.final_frequency_error_hz:.6g} Hz")


def run_micro(args: argparse.Namespace) -> None:
    result = run_micro_two_way_simulation(
        sdr_settings_from_args(args), micro_pilots_per_interval=args.micro_pilots
    )
    print(
        f"model: two-tier reciprocal sync ({args.micro_pilots} micro-pilots "
        "per interval)"
    )
    print(f"device: {result.device}")
    print(f"airtime fraction: {100.0 * result.airtime_fraction:.1f}%")
    print(f"detection rate: {100.0 * result.detection_rate:.2f}%")
    print(
        "steady-state oscillator phase residual RMS: "
        f"{result.steady_state_phase_rms:.6g} rad"
    )
    print(
        "mean open-loop 2-station coherent gain: "
        f"{100.0 * result.mean_coherent_gain:.2f}%"
    )
    print(f"final oscillator phase residual: {result.final_phase_error:.6g} rad")
    print(f"final CFO residual: {result.final_frequency_error_hz:.6g} Hz")


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
    print(f"airtime fraction: {100.0 * result.airtime_fraction:.1f}%")
    print(f"detection rate: {100.0 * result.detection_rate:.2f}%")
    print(
        "steady-state oscillator phase residual RMS: "
        f"{result.steady_state_phase_rms:.6g} rad"
    )
    print(
        "mean open-loop 2-station coherent gain: "
        f"{100.0 * result.mean_coherent_gain:.2f}%"
    )
    print(f"final oscillator phase residual: {result.final_phase_error:.6g} rad")
    print(f"final CFO residual: {result.final_frequency_error_hz:.6g} Hz")


def run_compare(args: argparse.Namespace) -> None:
    from hybrid_calibration import run_hybrid_simulation

    settings = sdr_settings_from_args(args)
    rows = []
    for label, runner in (
        ("two-way EKF (ours)", lambda: run_two_way_simulation(settings)),
        (
            "two-tier micro-pilot (ours)",
            lambda: run_micro_two_way_simulation(
                settings, micro_pilots_per_interval=args.micro_pilots
            ),
        ),
        (
            "hybrid 1-way+anchors (ours)",
            lambda: run_hybrid_simulation(
                settings,
                micro_pilots_per_interval=args.micro_pilots,
                anchor_every_intervals=args.anchor_every,
            ),
        ),
        (
            "DFPC naive (as published)",
            lambda: run_consensus_ota_simulation(settings, "dfpc", reciprocal=False),
        ),
        (
            "DFPC + reciprocity",
            lambda: run_consensus_ota_simulation(settings, "dfpc"),
        ),
        (
            "KF-DFPC + reciprocity",
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
            )
        )
    print("open-loop synchronization comparison (identical physical conditions)")
    print(f"{'approach':<27}{'phase RMS (mrad)':>18}{'coherent gain':>15}{'detect':>9}")
    for label, rms, gain, detect in rows:
        print(
            f"{label:<27}{1e3 * rms:>18.1f}{100.0 * gain:>14.2f}%{100.0 * detect:>8.0f}%"
        )
    print(
        "note: naive DFPC consenses on raw one-way measurements (the paper's\n"
        "channel-free assumption); over a real channel the wrapped symmetric\n"
        "update is bistable and can capture at the anti-phase fixed point,\n"
        "as it does for this channel realization. The reciprocity rows\n"
        "exchange measurements over the paper's assumed side channel."
    )


def main() -> None:
    args = parse_args()
    if args.model == "ideal":
        run_ideal(args)
    elif args.model == "twoway":
        run_twoway(args)
    elif args.model == "micro":
        run_micro(args)
    elif args.model == "hybrid":
        run_hybrid(args)
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
