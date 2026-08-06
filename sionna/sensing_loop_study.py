"""Sensing-in-the-loop pilot scheduling: budgets FOLLOW the target.

smart_sync_study.py sets task-aware budgets once, for a fixed coverage
edge. Here the sensing side closes the loop: a drone crosses the
deployment, and every segment of its track re-targets the budgets -
stations whose ray-traced legs dominate the CURRENT target hypothesis
get tight budgets, the rest are released to coast. This is the
`budget_updates` hook of run_scheduled_star driven by the same RT legs
the detection test uses: sensing -> sync allocation -> sensing.

Compared on identical physics:
  uniform          fixed cadence (the airtime ceiling)
  static-edge      scheduler with budgets set once for the coverage
                   edge (smart_sync_study's policy)
  target-tracking  scheduler with budgets re-targeted each segment as
                   the drone moves

Verification: counted waveform detection AT EACH probe waypoint, using
only the sync residuals from the intervals when the drone was actually
there (ScheduledSyncResult.residual_matrix(interval_slice)).

Usage:
    .venv/bin/python sensing_loop_study.py
    .venv/bin/python sensing_loop_study.py --waypoints 8 --iterations 96
"""

from __future__ import annotations

import argparse

import numpy as np

from detection import DetectionParams
from detection.rt_echo import rt_steered_legs
from detection.waveform import run_waveform_detection
from ota_sync import SDRSimulationConfig
from ota_sync.scheduled import run_scheduled_star


def main() -> None:
    parser = argparse.ArgumentParser(
        description="pilot budgets that follow the sensing target"
    )
    parser.add_argument("--stations", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=72)
    parser.add_argument("--waypoints", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tight-budget", type=float, default=0.20)
    parser.add_argument("--loose-budget", type=float, default=0.60)
    parser.add_argument("--trials", type=int, default=600)
    parser.add_argument("--path-offset-m", type=float, default=150.0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    n = args.stations
    if args.iterations % args.waypoints:
        parser.error("--iterations must divide evenly by --waypoints")
    segment = args.iterations // args.waypoints
    settings = SDRSimulationConfig(
        num_iterations=args.iterations, seed=args.seed, device="cpu"
    )
    params = DetectionParams(tx_power_w=0.5)

    probe = run_scheduled_star(
        SDRSimulationConfig(num_iterations=1, seed=args.seed, device="cpu"),
        num_stations=n,
        policy="uniform",
    )
    positions = probe.positions
    centroid = positions.mean(axis=0)
    span = 1200.0
    path_x = np.linspace(-span, span, args.waypoints)
    waypoints = np.stack(
        (
            centroid[0] + path_x,
            np.full(args.waypoints, centroid[1] + args.path_offset_m),
        ),
        axis=1,
    )
    print("ray tracing the station->waypoint legs (once)...")
    legs = rt_steered_legs(positions, waypoints)  # (waypoints, stations)

    def budgets_for(waypoint_index: int) -> list[float]:
        utility = np.abs(legs[waypoint_index])[1:]  # skip the reference
        budgets = [args.loose_budget] * (n - 1)
        for rank in np.argsort(-utility)[: (n - 1) // 2 + 1]:
            budgets[rank] = args.tight_budget
        return budgets

    tracking_updates = {
        segment * index: budgets_for(index)
        for index in range(args.waypoints)
    }
    static_budgets = budgets_for(0)  # the entry edge, fixed forever

    print(
        f"Sensing-in-the-loop scheduling, N={n} star, "
        f"{args.waypoints} track segments x {segment} intervals; "
        f"tight/loose budgets {1e3 * args.tight_budget:.0f}/"
        f"{1e3 * args.loose_budget:.0f} mrad"
    )
    for index in range(args.waypoints):
        print(
            f"  segment {index}: budgets (mrad) "
            + " ".join(f"{1e3 * b:4.0f}" for b in tracking_updates[segment * index])
        )

    runs = [
        ("uniform", dict(policy="uniform")),
        ("static-edge", dict(policy="scheduled", budgets_rad=static_budgets)),
        (
            "target-tracking",
            dict(
                policy="scheduled",
                budgets_rad=budgets_for(0),
                budget_updates=tracking_updates,
            ),
        ),
    ]
    results = []
    for label, kwargs in runs:
        result = run_scheduled_star(settings, num_stations=n, **kwargs)
        results.append((label, result))
        print(
            f"  {label:<16} airtime {100 * result.airtime_used_fraction:5.1f}% "
            f"gain {100 * result.mean_array_gain:6.2f}%"
        )

    print("counted detection at each waypoint, from that segment's "
          "residuals...")
    pd_rows = {label: [] for label, _ in results}
    for index in range(args.waypoints):
        window = slice(segment * index, segment * (index + 1))
        for label, result in results:
            matrix = result.residual_matrix(interval_slice=window)
            if matrix.shape[1] == 0:
                pd_rows[label].append(float("nan"))
                continue
            detect = run_waveform_detection(
                f"{label}@wp{index}",
                positions,
                matrix,
                waypoints[index : index + 1],
                params=params,
                trials=args.trials,
                seed=args.seed + index,
                leg_gains=legs[index : index + 1],
            )
            pd_rows[label].append(float(detect.pd_measured[0]))
    header = "  ".join(f"wp{i:<4d}" for i in range(args.waypoints))
    print(f"  {'policy':<16} {header}")
    for label, result in results:
        row = "  ".join(
            f"{100 * pd:5.1f}%" if pd == pd else "  n/a "
            for pd in pd_rows[label]
        )
        print(f"  {label:<16} {row}")

    base = results[0][1]
    track = results[2][1]
    saved = (
        base.airtime_used_fraction - track.airtime_used_fraction
    ) / base.airtime_used_fraction
    print(
        f"\nheadline: target-tracking budgets return {100 * saved:.0f}% of "
        "the sync airtime vs uniform while budgets follow the drone "
        "(closed sensing->sync loop)."
    )

    if args.no_plot:
        return

    from simulation import _render_figure_and_panels

    def budget_panel(axis):
        time = np.arange(args.iterations) * settings.sync_interval
        tracking = results[2][1]
        for index in range(n - 1):
            budgets = np.concatenate(
                [
                    np.full(segment, tracking_updates[segment * s][index])
                    for s in range(args.waypoints)
                ]
            )
            axis.step(time, 1e3 * budgets, where="post",
                      label=f"station {index + 1}")
            row = np.abs(tracking.residuals[index].numpy())
            axis.semilogy(time, 1e3 * row + 1e-2, linewidth=0.7, alpha=0.6,
                          color=axis.lines[-1].get_color())
        axis.set_ylabel("budget / |residual| (mrad, log)")
        axis.set_xlabel("time (s) - budgets step as the drone crosses")
        axis.legend(fontsize="x-small", ncols=3)

    def pd_panel(axis):
        x = np.arange(args.waypoints)
        width = 0.25
        for offset, (label, _) in zip((-1, 0, 1), results):
            axis.bar(x + offset * width, pd_rows[label], width, label=label)
        axis.axhline(0.9, color="red", linestyle=":", linewidth=1.0)
        axis.set_xticks(x)
        axis.set_xticklabels([f"wp{i}" for i in range(args.waypoints)])
        axis.set_ylabel("counted Pd at the waypoint")
        axis.set_ylim(0, 1.05)
        axis.legend(fontsize="small")

    def airtime_panel(axis):
        labels = [label for label, _ in results]
        airtimes = [100 * r.airtime_used_fraction for _, r in results]
        gains = [100 * r.mean_array_gain for _, r in results]
        bars = axis.barh(labels, airtimes, color="tab:orange")
        for bar, gain in zip(bars, gains):
            axis.annotate(
                f"gain {gain:.1f}%",
                (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=8,
            )
        axis.set_xlabel("sync airtime used (%)")
        axis.invert_yaxis()

    _render_figure_and_panels(
        f"Sensing-in-the-loop pilot scheduling (N={n}): budgets follow "
        "the drone across the deployment\nEach segment re-targets the "
        "budgets from the ray-traced legs toward the CURRENT target "
        "hypothesis.",
        [
            ("Per-station budgets stepping with the track, residuals "
             "under them", budget_panel),
            ("Counted detection at each waypoint, per policy", pd_panel),
            ("Sync airtime per policy (annotation: array gain)",
             airtime_panel),
        ],
        (3, 1),
        (11.0, 12.5),
        sharex=False,
        top=0.91,
    )


if __name__ == "__main__":
    main()
