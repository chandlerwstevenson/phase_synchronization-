"""Clutter-limited drone detection study — the field-realism layer.

Everything the single-pulse waveform study had, plus: gamma-model
ground clutter with internal motion (injected into the raw streams,
leaking across gates through real range sidelobes), direct-path
self-interference with actual least-squares cancellation, an
aspect-dependent micro-Doppler drone signature instead of a Swerling
scalar, pulse-train range-Doppler processing, and CA-CFAR search with
the clutter ridge notched. Ray-traced station->target legs (ground
bounce) as before. See detection/realistic.py for the models.

Usage:
    .venv/bin/python realistic_detection_study.py
    .venv/bin/python realistic_detection_study.py --clutter-gamma-db -10
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from detection import DetectionParams
from detection.realistic import (
    RealisticDetectionConfig,
    run_realistic_detection,
)
from detection.rt_echo import rt_steered_legs
from hybrid_calibration import run_hybrid_simulation
from hybrid_calibration.mesh import run_decentralized_hybrid_mesh
from ota_sync import SDRSimulationConfig, run_network_simulation
from waveform_detection_study import (
    loop_extract,
    mesh_residual_matrix,
    star_residual_matrix,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="clutter-limited CPI-level drone detection per method"
    )
    parser.add_argument("--stations", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=150)
    parser.add_argument("--h0-trials", type=int, default=800)
    parser.add_argument("--waypoints", type=int, default=8)
    parser.add_argument("--path-offset-m", type=float, default=150.0)
    parser.add_argument("--tx-power-w", type=float, default=1.0)
    parser.add_argument("--clutter-gamma-db", type=float, default=-15.0)
    parser.add_argument("--drone-speed-mps", type=float, default=15.0)
    parser.add_argument("--buildings", type=int, default=8,
                        help="random concrete buildings in the RT scene "
                        "(blockage/reflections on the target legs)")
    parser.add_argument("--span-factor", type=float, default=3.0,
                        help="path half-span as a multiple of the disc radius")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    n = args.stations
    params = DetectionParams(tx_power_w=args.tx_power_w)
    config = RealisticDetectionConfig(
        clutter_gamma_db=args.clutter_gamma_db,
        drone_speed_mps=args.drone_speed_mps,
    )
    settings = SDRSimulationConfig(
        num_iterations=args.iterations, seed=args.seed, device="cpu"
    )
    geometry = {
        "radius_m": 500.0,
        "path_loss_exponent": 2.7,
        "reference_distance_m": 500.0,
    }

    cpi_ms = 1e3 * config.num_pulses * config.pri_s
    print(
        f"Clutter-limited detection: N={n}, {config.num_pulses}-pulse CPI "
        f"({cpi_ms:.0f} ms), clutter gamma {args.clutter_gamma_db:.0f} dB, "
        f"drone {args.drone_speed_mps:.0f} m/s with rotor micro-Doppler, "
        f"window Pfa {config.window_pfa:g} (empirical), "
        f"{args.trials} trials/waypoint"
    )
    print("running sync methods...")
    residual_sets = []
    network_hybrid = run_network_simulation(
        settings,
        n,
        lambda s: run_hybrid_simulation(
            s, micro_pilots_per_interval=4, anchor_every_intervals=5
        ),
        loop_extract,
        **geometry,
    )
    positions = network_hybrid.positions
    residual_sets.append(
        ("star: hybrid", star_residual_matrix(network_hybrid))
    )
    mesh = run_decentralized_hybrid_mesh(
        settings, num_nodes=n, control="alternating", **geometry
    )
    residual_sets.append(
        ("mesh: dhybrid/alternating", mesh_residual_matrix(mesh))
    )
    residual_sets.append(
        ("perfect sync (bound)", torch.zeros(n, 16, dtype=torch.float64))
    )
    rng = torch.Generator().manual_seed(args.seed)
    free = (
        torch.rand(n, 512, dtype=torch.float64, generator=rng) * 2.0 - 1.0
    ) * math.pi
    free[0] = 0.0
    residual_sets.append(("free-running (no sync)", free))

    centroid = positions.mean(axis=0)
    span = args.span_factor * geometry["radius_m"]
    path_x = np.linspace(-span, span, args.waypoints)
    waypoints = np.stack(
        (
            centroid[0] + path_x,
            np.full(args.waypoints, centroid[1] + args.path_offset_m),
        ),
        axis=1,
    )

    print(f"ray tracing station->target legs (ground bounce + "
      f"{args.buildings} buildings)...")
    leg_gains = rt_steered_legs(
        positions,
        waypoints,
        station_height_m=config.station_height_m,
        target_height_m=config.target_height_m,
        carrier_frequency_hz=params.carrier_frequency_hz,
        antenna_gain_dbi=params.antenna_gain_dbi,
        with_ground=True,
        num_buildings=args.buildings,
        building_seed=args.seed,
    )

    print("running CPI Monte Carlo (clutter + direct path + micro-Doppler)...")
    results = []
    for label, matrix in residual_sets:
        result = run_realistic_detection(
            label,
            positions,
            matrix,
            waypoints,
            leg_gains,
            params=params,
            config=config,
            trials=args.trials,
            h0_trials=args.h0_trials,
            seed=args.seed,
        )
        results.append(result)
        print(
            f"  {label:<26} Pd: "
            + " ".join(f"{100.0 * pd:3.0f}" for pd in result.pd_measured)
            + f" %  (measured window Pfa {result.measured_window_pfa:.3f})"
        )
    first = results[0]
    print(
        f"peak clutter-to-noise ratio: {first.clutter_to_noise_db:.1f} dB "
        "(clutter-limited regime confirmed)"
        if first.clutter_to_noise_db > 10
        else f"peak clutter-to-noise ratio: {first.clutter_to_noise_db:.1f} dB"
    )
    before, after = first.direct_before_after_db
    print(
        f"direct-path power vs noise before/after LS cancellation (clutter-free probe): "
        f"{before:.1f} -> {after:.1f} dB"
    )
    print(
        "scope: gamma-model clutter (no discrete clutter), LS direct-path "
        "cancellation after assumed "
        f"{config.analog_isolation_db:.0f} dB analog isolation, single CPI "
        "(no tracking), simulation only — no hardware validation."
    )

    if args.no_plot:
        return

    from simulation import _render_figure_and_panels

    path_km = path_x / 1e3

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

    def map_panel(axis):
        example = results[0].example_map
        gate_t, doppler_bin = results[0].example_truth
        display = 10.0 * np.log10(np.fft.fftshift(example, axes=1) + 1e-12)
        image = axis.imshow(
            display,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            extent=(
                -0.5 / config.pri_s / 1e3 / 2.0 * 2.0,
                0.5 / config.pri_s / 1e3 / 2.0 * 2.0,
                0,
                example.shape[0],
            ),
        )
        shifted_bin = (
            doppler_bin + config.num_pulses // 2
        ) % config.num_pulses
        doppler_khz = (
            (shifted_bin - config.num_pulses / 2.0)
            / (config.num_pulses * config.pri_s)
            / 1e3
        )
        axis.plot(
            doppler_khz, gate_t, "rx", markersize=10, markeredgewidth=2,
            label="true drone cell",
        )
        axis.set_xlabel("Doppler (kHz)")
        axis.set_ylabel("range gate")
        axis.legend(fontsize="small", loc="upper right")
        import matplotlib.pyplot as plt

        plt.colorbar(image, ax=axis, label="CFAR ratio (dB)", shrink=0.8)

    def scene_panel(axis):
        axis.scatter(
            positions[:, 0], positions[:, 1], s=70, marker="^",
            color="tab:blue", label="base stations", zorder=3,
        )
        scatter = axis.scatter(
            waypoints[:, 0], waypoints[:, 1],
            c=results[0].pd_measured, cmap="RdYlGn", vmin=0, vmax=1,
            s=45, zorder=3, label="drone waypoints (Pd, star hybrid)",
        )
        axis.plot(waypoints[:, 0], waypoints[:, 1], color="gray",
                  linewidth=0.8, alpha=0.6)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_aspect("equal")
        axis.legend(fontsize="x-small", loc="lower left")
        import matplotlib.pyplot as plt

        plt.colorbar(scatter, ax=axis, label="measured Pd", shrink=0.8)

    _render_figure_and_panels(
        f"Clutter-limited drone detection (N={n}, {config.num_pulses}-pulse "
        f"CPI, gamma {args.clutter_gamma_db:.0f} dB, micro-Doppler target)\n"
        "Ground clutter + direct-path interference + CFAR search over the "
        "range-Doppler map; Pd counted from real streams.\n"
        f"Peak CNR {results[0].clutter_to_noise_db:.0f} dB — detection "
        "survives by Doppler separation, not link budget.",
        [
            ("Scene: stations and drone path (color = measured Pd)",
             scene_panel),
            ("Example fused range-Doppler map, CFAR-normalized (one trial, "
             "mid-path)", map_panel),
            ("Measured Pd along the path", pd_panel),
        ],
        (3, 1),
        (11.0, 14.0),
        sharex=False,
        top=0.92,
    )


if __name__ == "__main__":
    main()
