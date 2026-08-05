"""Waveform-level drone-detection test of each synchronization method.

Real software testing end to end: run the sync method (full physical
layer, ZC pilots — that is what BS-to-BS sync uses), take its
per-station residual phase SERIES, then Monte-Carlo the actual radar
detection with a 5G-style OFDM burst as the sensing waveform:
transmitted frames, geometric spreading, Swerling-1 target draws, real
noise sample streams, matched filtering, coherent combining, empirical
threshold — and COUNT detections. No detection formulas in the loop
(see detection/waveform.py).

Scenario: a drone flies a straight path across the deployment; every
waypoint is a detection trial set. The scene figure shows the station
locations, the drone path colored by the measured P_d, and the beam
steering (every station refocuses on the moving target).

Usage:
    .venv/bin/python waveform_detection_study.py
    .venv/bin/python waveform_detection_study.py --stations 6 --trials 3000
    .venv/bin/python waveform_detection_study.py --waveform zc   # probe option
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from detection import DetectionParams
from detection.waveform import run_waveform_detection
from hybrid_calibration import run_hybrid_simulation
from hybrid_calibration.mesh import run_decentralized_hybrid_mesh
from ota_sync import (
    SDRSimulationConfig,
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


def star_residual_matrix(network) -> torch.Tensor:
    """(stations, steady-samples) phase residuals; row 0 = reference."""

    steady = network.array_steady_mask
    rows = [torch.zeros(int(steady.sum().item()), dtype=torch.float64)]
    for link in network.links:
        rows.append(link.residual[steady])
    return torch.stack(rows)


def mesh_residual_matrix(mesh) -> torch.Tensor:
    """Node phases relative to the chain root, from per-edge residuals."""

    steady = mesh.steady
    count = int(steady.sum().item())
    node_rows = {mesh.chain[0]: torch.zeros(count, dtype=torch.float64)}
    running = torch.zeros(count, dtype=torch.float64)
    for index in range(len(mesh.chain) - 1):
        # edge residual = phi_parent - phi_child  =>  child = parent - res
        running = running - mesh.edge_residuals[index][steady]
        node_rows[mesh.chain[index + 1]] = running.clone()
    return torch.stack([node_rows[i] for i in range(len(mesh.chain))])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="waveform-level detection test per sync method"
    )
    parser.add_argument("--stations", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=1500)
    parser.add_argument("--rcs-dbsm", type=float, default=-15.0)
    parser.add_argument("--tx-power-w", type=float, default=1.0)
    parser.add_argument(
        "--pulse-length", type=int, default=1040,
        help="detection burst length in samples at 1 MS/s (default = 13 "
        "OFDM symbols of 64 subcarriers + 16 CP)",
    )
    parser.add_argument(
        "--waveform", choices=("ofdm", "zc"), default="ofdm",
        help="detection waveform: 5G-style OFDM burst (default; ZC stays "
        "the BS-to-BS sync waveform) or a ZC probe for comparison",
    )
    parser.add_argument(
        "--threshold-pfa", type=float, default=1e-3,
        help="false-alarm rate for the EMPIRICAL threshold (1e-6 would "
        "need >1e7 target-absent trials; comparisons are unaffected)",
    )
    parser.add_argument("--path-offset-m", type=float, default=150.0,
                        help="drone path lateral offset from the centroid")
    parser.add_argument("--waypoints", type=int, default=12)
    parser.add_argument(
        "--propagation",
        choices=("rt-ground", "rt-freespace", "analytic"),
        default="rt-ground",
        help="station<->target propagation: Sionna-RT with a dielectric "
        "ground plane (default: adds the two-ray lobing that dominates "
        "low-altitude UHF), RT in free space (validates against the "
        "analytic model), or the closed-form spreading factors",
    )
    parser.add_argument("--bs-height-m", type=float, default=15.0)
    parser.add_argument("--drone-alt-m", type=float, default=60.0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    n = args.stations
    pulse_seconds = args.pulse_length / 1e6
    params = DetectionParams(
        tx_power_w=args.tx_power_w,
        rcs_m2=10.0 ** (args.rcs_dbsm / 10.0),
        integration_time_s=pulse_seconds,
        pfa=args.threshold_pfa,
    )
    settings = SDRSimulationConfig(
        num_iterations=args.iterations, seed=args.seed, device="cpu"
    )
    geometry = {
        "radius_m": 500.0,
        "path_loss_exponent": 2.7,
        "reference_distance_m": 500.0,
    }

    print(
        f"Waveform-level detection: N={n}, {args.waveform.upper()} burst "
        f"({1e3 * pulse_seconds:.2f} ms), RCS {args.rcs_dbsm:.0f} dBsm, "
        f"{args.tx_power_w:g} W/station, empirical threshold at "
        f"Pfa={args.threshold_pfa:g}, {args.trials} trials/waypoint"
    )
    print("running sync methods to obtain their residual series...")

    residual_sets: list[tuple[str, torch.Tensor, np.ndarray]] = []
    network = run_network_simulation(
        settings, n, run_two_way_simulation, loop_extract, **geometry
    )
    positions = network.positions
    residual_sets.append(
        ("star: two-way", star_residual_matrix(network), positions)
    )
    network_hybrid = run_network_simulation(
        settings,
        n,
        lambda s: run_hybrid_simulation(
            s, micro_pilots_per_interval=4, anchor_every_intervals=5
        ),
        loop_extract,
        **geometry,
    )
    residual_sets.append(
        ("star: hybrid", star_residual_matrix(network_hybrid), positions)
    )
    for control in ("symmetric", "alternating"):
        mesh = run_decentralized_hybrid_mesh(
            settings, num_nodes=n, control=control, **geometry
        )
        residual_sets.append(
            (
                f"mesh: dhybrid/{control}",
                mesh_residual_matrix(mesh),
                mesh.positions,
            )
        )
    residual_sets.append(
        ("perfect sync (bound)", torch.zeros(n, 16, dtype=torch.float64),
         positions)
    )
    rng = torch.Generator().manual_seed(args.seed)
    free = (
        torch.rand(n, 512, dtype=torch.float64, generator=rng) * 2.0 - 1.0
    ) * math.pi
    free[0] = 0.0
    residual_sets.append(("free-running (no sync)", free, positions))

    # Drone path: straight crossing, offset from the centroid, spanning
    # from beyond one side of the deployment to beyond the other.
    centroid = positions.mean(axis=0)
    span = 2.4 * geometry["radius_m"]
    path_x = np.linspace(-span, span, args.waypoints)
    waypoints = np.stack(
        (
            centroid[0] + path_x,
            np.full(args.waypoints, centroid[1] + args.path_offset_m),
        ),
        axis=1,
    )

    leg_gains = None
    if args.propagation.startswith("rt"):
        from detection.rt_echo import rt_steered_legs

        print(
            f"ray tracing station->target legs (Sionna RT, "
            f"{'ground plane' if args.propagation == 'rt-ground' else 'free space'}, "
            f"BS masts {args.bs_height_m:.0f} m, drone at "
            f"{args.drone_alt_m:.0f} m)..."
        )
        leg_gains = rt_steered_legs(
            positions,
            waypoints,
            station_height_m=args.bs_height_m,
            target_height_m=args.drone_alt_m,
            carrier_frequency_hz=params.carrier_frequency_hz,
            antenna_gain_dbi=params.antenna_gain_dbi,
            with_ground=args.propagation == "rt-ground",
        )

    print("running waveform Monte Carlo along the drone path...")
    results = []
    for label, matrix, pos in residual_sets:
        result = run_waveform_detection(
            label,
            pos,
            matrix,
            waypoints,
            params=params,
            pulse_length=args.pulse_length,
            trials=args.trials,
            threshold_pfa=args.threshold_pfa,
            seed=args.seed,
            waveform=args.waveform,
            leg_gains=leg_gains,
        )
        results.append(result)
        print(
            f"  {label:<26} combining loss {result.combining_loss_db[0]:>6.2f} dB, "
            "Pd along path: "
            + " ".join(f"{100.0 * pd:3.0f}" for pd in result.pd_measured)
            + " %"
        )
    print(
        f"measured Pfa at the empirical threshold: "
        f"{results[0].measured_pfa:.2e} (target {args.threshold_pfa:g})"
    )
    scope_propagation = {
        "rt-ground": "Sionna-RT legs with ground-bounce multipath (two-ray); "
        "target RCS analytic at the probe point (shooting-ray RT cannot "
        "sample drone-sized targets)",
        "rt-freespace": "Sionna-RT free-space legs (validates the analytic "
        "model)",
        "analytic": "closed-form spreading factors",
    }[args.propagation]
    print(
        f"scope: {scope_propagation}; cued single-gate detection, no clutter "
        "or direct-path, PA effects not applied to the detection burst; "
        "every Pd is a counted Monte-Carlo fraction over real sample streams."
    )

    if args.no_plot:
        return

    from simulation import _render_figure_and_panels

    headline = results[1]  # star: hybrid
    path_km = path_x / 1e3

    def scene_panel(axis):
        axis.scatter(
            positions[:, 0], positions[:, 1], s=70, marker="^",
            color="tab:blue", label="base stations", zorder=3,
        )
        for index, (x, y) in enumerate(positions):
            axis.annotate(
                f"BS{index}", (x, y), fontsize=8, xytext=(5, 5),
                textcoords="offset points",
            )
        scatter = axis.scatter(
            waypoints[:, 0], waypoints[:, 1],
            c=headline.pd_measured, cmap="RdYlGn", vmin=0.0, vmax=1.0,
            s=45, zorder=3, label="drone waypoints (color = measured Pd)",
        )
        axis.plot(
            waypoints[:, 0], waypoints[:, 1], color="gray",
            linewidth=0.8, alpha=0.6,
        )
        axis.annotate(
            "drone path",
            (waypoints[-1, 0], waypoints[-1, 1]),
            fontsize=9, xytext=(8, 8), textcoords="offset points",
        )
        # Beam steering: every station refocuses on the target as it
        # moves; draw the steering rays for three highlighted waypoints.
        highlight = [1, args.waypoints // 2, args.waypoints - 2]
        for waypoint_index in highlight:
            target = waypoints[waypoint_index]
            for station in positions:
                axis.plot(
                    [station[0], target[0]],
                    [station[1], target[1]],
                    color="tab:orange", linewidth=0.6, alpha=0.35,
                )
        axis.plot([], [], color="tab:orange", linewidth=1.0, alpha=0.6,
                  label="beam steering (3 snapshots)")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_aspect("equal")
        axis.legend(fontsize="x-small", loc="lower left")
        import matplotlib.pyplot as plt

        plt.colorbar(scatter, ax=axis, label="measured Pd", shrink=0.8)

    def pd_panel(axis):
        for result in results:
            style = "--o" if result.label.startswith(("perfect", "free")) else "-o"
            axis.plot(
                path_km, result.pd_measured, style, linewidth=1.2,
                markersize=4, label=result.label,
            )
        axis.axhline(0.9, color="red", linestyle=":", linewidth=1.0,
                     label="Pd = 0.9")
        axis.set_xlabel("drone position along path (km from centroid)")
        axis.set_ylabel("measured Pd")
        axis.set_ylim(0.0, 1.03)
        axis.legend(fontsize="x-small", ncols=2)

    def loss_panel(axis):
        labels = [r.label for r in results]
        losses = [r.combining_loss_db[0] for r in results]
        axis.barh(labels, losses, color="tab:purple")
        axis.set_xlabel(
            "waveform-measured combining loss vs perfect sync "
            "(dB, transmit + receive)"
        )
        axis.invert_yaxis()

    _render_figure_and_panels(
        f"Waveform-level drone detection along a flight path (N={n}, "
        f"{args.waveform.upper()} burst, RCS {args.rcs_dbsm:.0f} dBsm, "
        f"propagation: {args.propagation})\n"
        "ZC pilots synchronize the stations; a 5G-style OFDM burst does "
        "the sensing; station→target legs are ray-traced\n"
        "(Sionna RT, ground bounce included). Pd is counted from real "
        "sample streams with the measured sync residuals.",
        [
            (
                "Scene: stations, drone path (colored by measured Pd for "
                "star: hybrid), and beam steering",
                scene_panel,
            ),
            ("Measured Pd along the path, every method", pd_panel),
            ("Coherent combining loss measured from the waveforms",
             loss_panel),
        ],
        (3, 1),
        (11.0, 14.0),
        sharex=False,
        top=0.92,
    )


if __name__ == "__main__":
    main()
