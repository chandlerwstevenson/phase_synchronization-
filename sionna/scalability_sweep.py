"""2-D scalability sweep: method x sync interval x station count.

Reproduces the "which method scales best" experiment: for each model
and each pilot cadence, find the largest station count that physically
fits the shared channel (total pilot airtime < 100%), run the full
network simulation at that N, and report the array coherent gain.

Usage:
    .venv/bin/python scalability_sweep.py                 # all models
    .venv/bin/python scalability_sweep.py twoway hybrid   # a subset
    .venv/bin/python scalability_sweep.py --intervals-ms 50,100 --cap 12

Runs the repository's public simulation functions unchanged; expect
roughly 15-30 minutes for the full default grid on a laptop CPU.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from ota_sync import (
    SDRSimulationConfig,
    run_micro_two_way_simulation,
    run_network_simulation,
    run_two_way_simulation,
)
from hybrid_calibration import run_hybrid_simulation


MODEL_COLORS = {
    "twoway": "tab:blue",
    "micro": "tab:orange",
    "hybrid": "tab:green",
}


def loop_extract(result):
    mask = result.detected & result.correction_active & result.calibrated
    return (
        result.post_correction_phase,
        mask,
        result.detection_rate,
        result.airtime_fraction,
    )


def plot_grid(rows_by_model: dict, cap: int) -> None:
    """Combined annotated figure + clean per-panel figures for the grid.

    Each row is (interval_ms, per_link, n_fit, n_run, gain, worst_rms).
    """

    from simulation import _render_figure_and_panels

    def stations_panel(axis):
        for model, rows in rows_by_model.items():
            intervals = [row[0] for row in rows]
            fits = [row[2] for row in rows]
            axis.semilogx(
                intervals,
                fits,
                "o-",
                linewidth=1.4,
                color=MODEL_COLORS[model],
                label=model,
            )
        axis.axhline(
            cap,
            color="gray",
            linestyle=":",
            linewidth=1.0,
            label=f"simulation cap (N={cap})",
        )
        axis.set_ylabel("max stations that fit the channel")
        axis.set_xlabel("sync interval (ms, log)")
        axis.legend(fontsize="small")

    def gain_panel(axis):
        for model, rows in rows_by_model.items():
            intervals = [row[0] for row in rows]
            gains = [100.0 * row[4] for row in rows]
            axis.semilogx(
                intervals,
                gains,
                "o-",
                linewidth=1.4,
                color=MODEL_COLORS[model],
                label=model,
            )
            for row in rows:
                axis.annotate(
                    f"N={row[3]}",
                    (row[0], 100.0 * row[4]),
                    fontsize=7,
                    xytext=(3, 4),
                    textcoords="offset points",
                    color=MODEL_COLORS[model],
                )
        axis.axhline(
            90.0, color="red", linestyle="--", linewidth=1.0, label="90%"
        )
        axis.set_ylabel("array gain at max-fitting N (%)")
        axis.set_xlabel("sync interval (ms, log)")
        axis.legend(fontsize="small")

    def residual_panel(axis):
        for model, rows in rows_by_model.items():
            intervals = [row[0] for row in rows]
            worst = [1e3 * row[5] for row in rows]
            axis.loglog(
                intervals,
                worst,
                "o-",
                linewidth=1.4,
                color=MODEL_COLORS[model],
                label=model,
            )
        axis.axhline(
            314.0,
            color="red",
            linestyle="--",
            linewidth=1.0,
            label="314 mrad = 18\N{DEGREE SIGN}",
        )
        axis.set_ylabel("worst-station residual (mrad, log)")
        axis.set_xlabel("sync interval (ms, log)")
        axis.legend(fontsize="small")

    _render_figure_and_panels(
        "Which method scales best? cadence × station count, per method\n"
        "Longer intervals fit more stations (pilots get cheaper) but let "
        "crystals drift more between corrections.\n"
        "The winner holds the most stations while staying accurate — read "
        "all three panels together.",
        [
            (
                "Capacity: stations that fit the shared channel "
                "(airtime < 100%)",
                stations_panel,
            ),
            (
                "Coherence: array gain measured at that station count "
                "(labels = N actually simulated)",
                gain_panel,
            ),
            (
                "Accuracy: worst station's residual — crossing the red line "
                "means losing coherence",
                residual_panel,
            ),
        ],
        (3, 1),
        (10.5, 11),
        sharex=True,
        top=0.9,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="method x interval x station-count scalability sweep"
    )
    parser.add_argument(
        "models",
        nargs="*",
        default=["twoway", "micro", "hybrid"],
        choices=["twoway", "micro", "hybrid"],
        help="which methods to sweep (default: all three)",
    )
    parser.add_argument(
        "--intervals-ms",
        default="25,50,100,200",
        help="comma-separated sync intervals in milliseconds",
    )
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--cap",
        type=int,
        default=20,
        help="largest station count to actually simulate (the analytic "
        "airtime wall is still reported when it exceeds the cap)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--micro-pilots", type=int, default=4)
    parser.add_argument("--anchor-every", type=int, default=5)
    parser.add_argument("--area-radius-m", type=float, default=500.0)
    parser.add_argument("--path-loss-exponent", type=float, default=2.7)
    parser.add_argument("--ref-distance-m", type=float, default=500.0)
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="skip the figures and print only the table",
    )
    args = parser.parse_args()

    runners = {
        "twoway": run_two_way_simulation,
        "micro": lambda s: run_micro_two_way_simulation(
            s, micro_pilots_per_interval=args.micro_pilots
        ),
        "hybrid": lambda s: run_hybrid_simulation(
            s,
            micro_pilots_per_interval=args.micro_pilots,
            anchor_every_intervals=args.anchor_every,
        ),
    }
    intervals_ms = [float(v) for v in args.intervals_ms.split(",")]
    base = SDRSimulationConfig(
        num_iterations=args.iterations, seed=args.seed, device="cpu"
    )
    geometry = {
        "radius_m": args.area_radius_m,
        "path_loss_exponent": args.path_loss_exponent,
        "reference_distance_m": args.ref_distance_m,
    }

    print(
        "2-D scalability sweep: per cell, the largest N that fits the "
        "channel and the array gain measured there"
    )
    print(
        f"({args.iterations} intervals, seed {args.seed}, "
        f"{args.area_radius_m:.0f} m disc, exponent "
        f"{args.path_loss_exponent:g}, N capped at {args.cap})"
    )
    rows_by_model: dict[str, list] = {}
    for model in args.models:
        runner = runners[model]
        rows_by_model[model] = []
        print(f"model={model}")
        for interval_ms in intervals_ms:
            settings = replace(base, sync_interval=interval_ms * 1e-3)
            probe = run_network_simulation(
                settings, 2, runner, loop_extract, **geometry
            )
            per_link = probe.links[0].airtime_fraction
            n_fit = int(1.0 / per_link) + 1
            n_run = min(n_fit, args.cap)
            capped = "+" if n_fit > args.cap else ""
            if n_run <= 2:
                result = probe
            else:
                result = run_network_simulation(
                    settings, n_run, runner, loop_extract, **geometry
                )
            print(
                f"  T={interval_ms:>5.0f} ms: per-link airtime "
                f"{100 * per_link:5.1f}% -> max N that fits = "
                f"{n_fit:>3}{capped}; ran N={result.num_stations:>3}: "
                f"array gain {100 * result.mean_array_gain:6.2f}%, "
                f"worst station {1e3 * result.worst_station_rms:7.1f} mrad, "
                f"total airtime {100 * result.total_airtime_fraction:6.1f}%, "
                f"min detect {100 * result.min_detection_rate:5.1f}%"
            )
            rows_by_model[model].append(
                (
                    interval_ms,
                    per_link,
                    n_fit,
                    result.num_stations,
                    result.mean_array_gain,
                    result.worst_station_rms,
                )
            )

    if not args.no_plot:
        plot_grid(rows_by_model, args.cap)


if __name__ == "__main__":
    main()
