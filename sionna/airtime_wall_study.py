"""The airtime-wall shift: how far each scheduling policy moves the
array-size wall (PUBLISHABLE.md result 3's harmonic-mean prediction,
measured instead of asserted).

Fixed-cadence sync demands (N-1) exchanges every interval, but the
shared channel physically fits only ~5 at the default frame length -
that is the wall that caps array size in every --sweep-stations run.
An uncertainty-driven scheduler services links only when their filters
demand it, so its EFFECTIVE demand is sum(1/coast_time) instead of
(N-1)/T and the wall moves right.

This study sweeps N with the channel capacity pinned at its physical
limit and measures, per policy: the steady array gain, the airtime
actually used, and N_max = the largest N still holding >= 90% gain.

Usage:
    .venv/bin/python airtime_wall_study.py
    .venv/bin/python airtime_wall_study.py --stations-list 4,8,12,16 \
        --policies scheduled,whittle --multi-fidelity
"""

from __future__ import annotations

import argparse

import torch

from ota_sync import SDRSimulationConfig
from ota_sync.scheduled import SCHEDULER_POLICIES, run_scheduled_star


def effective_gain(result) -> tuple[float, str]:
    gain = result.mean_array_gain
    if gain == gain:
        return gain, ""
    tail = result.array_gain[-max(1, result.array_gain.numel() // 4):]
    return torch.mean(tail).item(), "*"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="array-size wall vs scheduling policy"
    )
    parser.add_argument("--stations-list", type=str, default="4,6,8,10,12")
    parser.add_argument(
        "--policies", type=str,
        default="uniform,roundrobin,scheduled,whittle",
        help=f"comma list from {SCHEDULER_POLICIES}",
    )
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flat-budget", type=float, default=0.314)
    parser.add_argument("--gain-threshold", type=float, default=0.90)
    parser.add_argument("--multi-fidelity", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    stations_list = [int(v) for v in args.stations_list.split(",")]
    policies = [p.strip() for p in args.policies.split(",")]
    rows = []  # (n, policy, gain, flag, airtime)

    # Physical channel capacity at the default frame length: how many
    # full two-way exchanges fit one interval (same for every N).
    probe = run_scheduled_star(
        SDRSimulationConfig(num_iterations=1, seed=args.seed, device="cpu"),
        num_stations=2,
        policy="uniform",
    )
    per_exchange = probe.airtime_uniform_fraction  # one link's fraction
    capacity = max(1, int(1.0 / per_exchange))
    print(
        "Airtime-wall study: channel fits "
        f"{capacity} two-way exchanges per {1e3 * 0.05:.0f} ms interval "
        f"({100 * per_exchange:.1f}% each); fixed cadence hits the wall "
        f"at N = {capacity + 1}"
        + (", multi-fidelity pilots ON" if args.multi_fidelity else "")
    )

    for n in stations_list:
        settings = SDRSimulationConfig(
            num_iterations=args.iterations, seed=args.seed, device="cpu"
        )
        print(f"\n--- N = {n} (demand {n - 1}/interval) ---")
        for policy in policies:
            result = run_scheduled_star(
                settings,
                num_stations=n,
                policy=policy,
                budgets_rad=[args.flat_budget] * (n - 1),
                max_exchanges_per_interval=capacity,
                multi_fidelity=args.multi_fidelity,
            )
            gain, flag = effective_gain(result)
            rows.append((n, policy, gain, flag, result.airtime_used_fraction))
            print(
                f"  {policy:<11} gain {100 * gain:6.2f}%{flag or ' '} "
                f"airtime {100 * result.airtime_used_fraction:5.1f}%"
            )

    print("\nN_max holding >= "
          f"{100 * args.gain_threshold:.0f}% array gain:")
    for policy in policies:
        passing = [
            n for n, p, gain, flag, _ in rows
            if p == policy and gain >= args.gain_threshold
        ]
        wall = max(passing) if passing else None
        beyond = (
            " (wall beyond the sweep - extend --stations-list)"
            if wall == max(stations_list)
            else ""
        )
        print(f"  {policy:<11} N_max = {wall}{beyond}")

    if args.no_plot:
        return

    from simulation import _render_figure_and_panels

    def gain_panel(axis):
        for policy in policies:
            xs = [n for n, p, *_ in rows if p == policy]
            ys = [100 * g for n, p, g, f, a in rows if p == policy]
            axis.plot(xs, ys, "o-", label=policy)
        axis.axhline(100 * args.gain_threshold, color="red",
                     linestyle=":", linewidth=1.0)
        axis.set_xlabel("stations N")
        axis.set_ylabel("array coherent gain (%)")
        axis.legend(fontsize="small")

    def airtime_panel(axis):
        for policy in policies:
            xs = [n for n, p, *_ in rows if p == policy]
            ys = [100 * a for n, p, g, f, a in rows if p == policy]
            axis.plot(xs, ys, "o-", label=policy)
        axis.axhline(100.0, color="black", linestyle="--", linewidth=1.0)
        axis.set_xlabel("stations N")
        axis.set_ylabel("sync airtime used (%)")
        axis.legend(fontsize="small")

    _render_figure_and_panels(
        "The airtime wall, per scheduling policy: gain and channel cost "
        "vs array size\n(channel capacity pinned at its physical limit; "
        "fixed cadence saturates it, scheduling spends only what the "
        "filters demand)",
        [
            ("Array gain vs N", gain_panel),
            ("Airtime used vs N", airtime_panel),
        ],
        (2, 1),
        (10.0, 9.0),
        sharex=True,
        top=0.90,
    )


if __name__ == "__main__":
    main()
