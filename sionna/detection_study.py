"""Drone-detection viability of every synchronization method.

Runs each sync approach at N stations on the same random deployment,
takes its MEASURED steady array coherent gain G, and converts it into
passive-target detection performance via SNR_N = N^3 * G^2 * SNR_1 and
a Swerling-1 detector (see detection/viability.py for the math and the
honest scope). Prints the table and, by default, plots P_d-vs-range
curves and detection-range bars (combined annotated figure + clean
panels).

Usage:
    .venv/bin/python detection_study.py                    # N=6, all methods
    .venv/bin/python detection_study.py --stations 8 --iterations 40
    .venv/bin/python detection_study.py --rcs-dbsm -20 --tx-power-w 2

Pure addition: drives the existing simulators through their public
APIs only.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from detection import DetectionParams, detection_range_m, evaluate_method
from hybrid_calibration import run_hybrid_simulation
from hybrid_calibration.mesh import (
    run_decentralized_hybrid_mesh,
    run_dfpc_mesh,
)
from ota_sync import (
    SDRSimulationConfig,
    run_micro_two_way_simulation,
    run_network_simulation,
    run_two_way_simulation,
)


def loop_extract(result):
    mask = result.detected & result.correction_active & result.calibrated
    return (
        result.post_correction_phase,
        mask,
        result.detection_rate,
        result.airtime_fraction,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="drone-detection viability per synchronization method"
    )
    parser.add_argument("--stations", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--snr-db", type=float, default=20.0)
    parser.add_argument("--area-radius-m", type=float, default=500.0)
    parser.add_argument("--micro-pilots", type=int, default=4)
    parser.add_argument("--anchor-every", type=int, default=5)
    parser.add_argument(
        "--rcs-dbsm",
        type=float,
        default=-15.0,
        help="target radar cross-section in dBsm (-15 = 0.03 m^2, a small "
        "quadcopter; published UHF drone RCS spans about -20..-10)",
    )
    parser.add_argument("--tx-power-w", type=float, default=1.0)
    parser.add_argument("--antenna-gain-dbi", type=float, default=6.0)
    parser.add_argument("--integration-ms", type=float, default=50.0)
    parser.add_argument("--pfa", type=float, default=1e-6)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    params = DetectionParams(
        tx_power_w=args.tx_power_w,
        antenna_gain_dbi=args.antenna_gain_dbi,
        rcs_m2=10.0 ** (args.rcs_dbsm / 10.0),
        integration_time_s=args.integration_ms * 1e-3,
        pfa=args.pfa,
    )
    settings = SDRSimulationConfig(
        num_iterations=args.iterations,
        snr_db=args.snr_db,
        seed=args.seed,
        device="cpu",
    )
    n = args.stations
    geometry = {
        "radius_m": args.area_radius_m,
        "path_loss_exponent": 2.7,
        "reference_distance_m": args.area_radius_m,
    }

    print(
        f"Drone-detection viability, N={n} stations, "
        f"RCS {args.rcs_dbsm:.0f} dBsm, {args.tx_power_w:g} W/station, "
        f"{args.integration_ms:.0f} ms integration, "
        f"Pd>=0.9 @ Pfa={args.pfa:g} (Swerling 1)"
    )
    single_range = detection_range_m(1, 1.0, params)
    perfect_range = detection_range_m(n, 1.0, params)
    print(
        f"reference ranges: single station {single_range:.0f} m; "
        f"perfect {n}-station coherence {perfect_range:.0f} m (N^3 = "
        f"{10.0 * 3.0 * __import__('math').log10(n):.1f} dB)"
    )

    def star_gain(runner) -> float:
        network = run_network_simulation(
            settings, n, runner, loop_extract, **geometry
        )
        return network.mean_array_gain

    print("running synchronization methods (this is the slow part)...")
    methods = []
    methods.append(
        (
            "star: two-way",
            star_gain(run_two_way_simulation),
        )
    )
    methods.append(
        (
            "star: micro",
            star_gain(
                lambda s: run_micro_two_way_simulation(
                    s, micro_pilots_per_interval=args.micro_pilots
                )
            ),
        )
    )
    methods.append(
        (
            "star: hybrid",
            star_gain(
                lambda s: run_hybrid_simulation(
                    s,
                    micro_pilots_per_interval=args.micro_pilots,
                    anchor_every_intervals=args.anchor_every,
                )
            ),
        )
    )
    for control in ("symmetric", "alternating", "directed"):
        mesh = run_decentralized_hybrid_mesh(
            settings,
            num_nodes=n,
            micro_pilots_per_interval=args.micro_pilots,
            anchor_every_intervals=args.anchor_every,
            control=control,
            **geometry,
        )
        methods.append((f"mesh: dhybrid/{control}", mesh.mean_array_gain))
    for use_kf, label in ((False, "mesh: DFPC"), (True, "mesh: KF-DFPC")):
        mesh = run_dfpc_mesh(settings, num_nodes=n, use_kf=use_kf, **geometry)
        methods.append((label, mesh.mean_array_gain))
    methods.append(("free-running (no sync)", 1.0 / n))
    methods.append(("perfect sync (bound)", 1.0))

    rows = [
        evaluate_method(label, n, gain, params) for label, gain in methods
    ]
    print()
    print(
        f"{'method':<26}{'sync gain':>10}{'SNR factor':>12}"
        f"{'detect range':>14}{'vs single':>11}{'vs perfect':>12}"
    )
    for row in rows:
        print(
            f"{row.label:<26}{100.0 * row.sync_gain:>9.1f}%"
            f"{row.snr_factor_db:>10.1f}dB{row.range_m:>12.0f} m"
            f"{row.range_vs_single:>10.2f}x{100.0 * row.range_vs_perfect:>11.1f}%"
        )
    print(
        "\nscope: link-budget + Swerling-1 statistics on top of the MEASURED"
        "\nsync residuals; no waveform-level echo, clutter, or micro-Doppler"
        "\nsimulation yet (next fidelity step)."
    )

    if args.no_plot:
        return

    import numpy as np

    from simulation import _render_figure_and_panels

    ranges_km = np.linspace(0.05, 1.4 * perfect_range / 1e3, 400)

    def pd_panel(axis):
        for row in rows:
            style = "-"
            if row.label.startswith("free") or row.label.startswith("perfect"):
                style = "--"
            axis.plot(
                ranges_km,
                [row.pd_at(1e3 * r) for r in ranges_km],
                style,
                linewidth=1.3,
                label=f"{row.label} ({100 * row.sync_gain:.0f}%)",
            )
        axis.axhline(0.9, color="red", linestyle=":", linewidth=1.0,
                     label="Pd = 0.9 target")
        axis.set_xlabel("target range (km)")
        axis.set_ylabel("probability of detection")
        axis.set_ylim(0.0, 1.02)
        axis.legend(fontsize="x-small", ncols=2)

    def range_panel(axis):
        labels = [row.label for row in rows]
        values = [row.range_m / 1e3 for row in rows]
        colors = [
            "tab:gray" if r.label.startswith(("free", "perfect")) else "tab:blue"
            for r in rows
        ]
        axis.barh(labels, values, color=colors)
        axis.axvline(
            single_range / 1e3, color="black", linestyle=":", linewidth=1.0,
            label="single station",
        )
        axis.set_xlabel("detection range for Pd>=0.9 (km)")
        axis.invert_yaxis()
        axis.legend(fontsize="small")

    _render_figure_and_panels(
        f"Drone-detection viability of each synchronization method "
        f"(N={n}, RCS {args.rcs_dbsm:.0f} dBsm, {args.tx_power_w:g} W/station)\n"
        "Detection SNR = N^3 * G^2 x single station, where G is each "
        "method's MEASURED array coherent gain —\n"
        "sync errors cost detection range twice (transmit focusing and "
        "receive combining).",
        [
            ("Probability of detecting the drone vs range (Swerling 1, "
             f"Pfa={args.pfa:g})", pd_panel),
            ("Detection range per method", range_panel),
        ],
        (2, 1),
        (11.0, 9.0),
        sharex=False,
        top=0.9,
    )


if __name__ == "__main__":
    main()
