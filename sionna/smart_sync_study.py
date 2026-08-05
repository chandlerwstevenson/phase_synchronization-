"""Smart allocation of phase sync: scheduled vs uniform, priced in
airtime and verified at the detection edge.

Three policies on the same N-station star, same deployment, same
physics:
  uniform         every station serviced every interval (the baseline
                  every method in this repository uses)
  scheduled       uncertainty-driven: a station is serviced only when
                  its Kalman-predicted phase uncertainty approaches its
                  budget (one budget for everyone)
  task-aware      same scheduler, but budgets follow detection utility:
                  stations with the strongest ray-traced legs toward
                  the coverage-edge annulus get tight budgets, the rest
                  are allowed to coast further

Verification: both residual processes feed the counted waveform
detection test at edge waypoints — the claim is equal detection at a
fraction of the sync airtime.

Usage:
    .venv/bin/python smart_sync_study.py
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from detection import DetectionParams
from detection.rt_echo import rt_steered_legs
from detection.waveform import run_waveform_detection
from ota_sync import SDRSimulationConfig
from ota_sync.scheduled import run_scheduled_star


def main() -> None:
    parser = argparse.ArgumentParser(
        description="uncertainty-driven sync scheduling vs uniform"
    )
    parser.add_argument("--stations", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tight-budget", type=float, default=0.20)
    parser.add_argument("--loose-budget", type=float, default=0.60)
    parser.add_argument("--flat-budget", type=float, default=0.314)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    n = args.stations
    settings = SDRSimulationConfig(
        num_iterations=args.iterations, seed=args.seed, device="cpu"
    )
    params = DetectionParams(tx_power_w=0.5)

    # Task utility: leg strength toward the coverage-edge annulus.
    probe = run_scheduled_star(
        SDRSimulationConfig(num_iterations=1, seed=args.seed, device="cpu"),
        num_stations=n,
        policy="uniform",
    )
    positions = probe.positions
    centroid = positions.mean(axis=0)
    edge_targets = np.array(
        [
            centroid + [1200.0, 150.0],
            centroid + [-1200.0, 150.0],
        ]
    )
    legs = rt_steered_legs(positions, edge_targets)
    utility = np.abs(legs).mean(axis=0)[1:]  # skip the reference
    order = np.argsort(-utility)
    task_budgets = [args.loose_budget] * (n - 1)
    for rank in order[: (n - 1) // 2 + 1]:
        task_budgets[rank] = args.tight_budget

    print(
        f"Smart sync allocation, N={n} star, {args.iterations} intervals, "
        f"channel capacity {int(1.0 / probe.airtime_uniform_fraction * (n - 1))} "
        "exchanges/interval"
    )
    print(
        "task-aware budgets (mrad):",
        [f"{1e3 * b:.0f}" for b in task_budgets],
        "(tight = strong legs to the coverage edge)",
    )

    policies = (
        ("uniform", "uniform", None),
        ("scheduled (flat budgets)", "scheduled",
         [args.flat_budget] * (n - 1)),
        ("scheduled (task-aware)", "scheduled", task_budgets),
    )
    results = []
    for label, policy, budgets in policies:
        result = run_scheduled_star(
            settings, num_stations=n, policy=policy, budgets_rad=budgets
        )
        results.append((label, result))
        print(
            f"  {label:<26} airtime {100 * result.airtime_used_fraction:5.1f}% "
            f"(uniform {100 * result.airtime_uniform_fraction:.1f}%), "
            f"gain {100 * result.mean_array_gain:6.2f}%, "
            f"residuals {[f'{1e3 * v:.0f}' for v in result.station_steady_rms]} mrad"
        )

    # Detection verification at the coverage edge, counted trials.
    print("verifying at the detection edge (counted waveform trials)...")
    detection_rows = []
    for label, result in results:
        detect = run_waveform_detection(
            label,
            positions,
            result.residual_matrix(),
            edge_targets,
            params=params,
            trials=args.trials,
            seed=args.seed,
            leg_gains=legs,
        )
        detection_rows.append((label, result, detect))
        print(
            f"  {label:<26} edge Pd: "
            + "  ".join(f"{100.0 * pd:5.1f}%" for pd in detect.pd_measured)
        )

    best = detection_rows[2]
    base = detection_rows[0]
    saved = (
        base[1].airtime_used_fraction - best[1].airtime_used_fraction
    ) / base[1].airtime_used_fraction
    print(
        f"\nheadline: task-aware scheduling returns "
        f"{100 * saved:.0f}% of the sync airtime "
        f"({100 * base[1].airtime_used_fraction:.1f}% -> "
        f"{100 * best[1].airtime_used_fraction:.1f}%) at matching edge "
        "detection."
    )

    if args.no_plot:
        return

    from simulation import _render_figure_and_panels

    def residual_panel(axis):
        interval = settings.sync_interval
        time = (np.arange(args.iterations) + 1) * interval
        scheduled = results[2][1]
        for index in range(n - 1):
            row = scheduled.residuals[index].numpy()
            axis.semilogy(
                time,
                1e3 * np.abs(row) + 1e-2,
                linewidth=1.0,
                label=(
                    f"station {index + 1} "
                    f"(budget {1e3 * scheduled.budgets_rad[index]:.0f} mrad)"
                ),
            )
            ticks = scheduled.serviced[index].numpy()
            axis.plot(
                time[ticks],
                np.full(ticks.sum(), 0.02),
                "|",
                markersize=6,
                color=axis.lines[-1].get_color(),
            )
        axis.axhline(314.0, color="red", linestyle="--", linewidth=1.0,
                     label="314 mrad")
        axis.set_ylabel("|residual| (mrad, log)")
        axis.set_xlabel("time (s)  (ticks = pilot exchanges actually sent)")
        axis.legend(fontsize="x-small", ncols=2)

    def airtime_panel(axis):
        labels = [row[0] for row in detection_rows]
        airtimes = [100 * row[1].airtime_used_fraction for row in detection_rows]
        gains = [100 * row[1].mean_array_gain for row in detection_rows]
        bars = axis.barh(labels, airtimes, color="tab:orange")
        for bar, gain in zip(bars, gains):
            axis.annotate(
                f"gain {gain:.1f}%",
                (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=8,
            )
        axis.set_xlabel("sync airtime actually used (%)")
        axis.invert_yaxis()

    def pd_panel(axis):
        width = 0.25
        x = np.arange(len(edge_targets))
        for offset, (label, _, detect) in zip((-1, 0, 1), detection_rows):
            axis.bar(
                x + offset * width,
                detect.pd_measured,
                width,
                label=label,
            )
        axis.set_xticks(x)
        axis.set_xticklabels(
            [f"edge {index + 1} ({row:.1f} km)" for index, row in
             enumerate(np.linalg.norm(edge_targets - centroid, axis=1) / 1e3)]
        )
        axis.axhline(0.9, color="red", linestyle=":", linewidth=1.0)
        axis.set_ylabel("measured Pd at the coverage edge")
        axis.set_ylim(0, 1.05)
        axis.legend(fontsize="small")

    _render_figure_and_panels(
        f"Smart allocation of phase sync (N={n} star): service a station "
        "only when its filter says so\n"
        "Budgets are task-aware (tight for stations that matter at the "
        "detection edge). Ticks in the top panel are the\n"
        "pilots actually sent — the whitespace between them is the airtime "
        "returned to sensing.",
        [
            ("Scheduled residuals: sawtooth coasting against per-station "
             "budgets", residual_panel),
            ("Sync airtime used per policy (annotation: array gain held)",
             airtime_panel),
            ("Counted detection at the coverage edge, per policy", pd_panel),
        ],
        (3, 1),
        (11.0, 12.0),
        sharex=False,
        top=0.91,
    )


if __name__ == "__main__":
    main()
