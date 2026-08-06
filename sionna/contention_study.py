"""Contended-channel scheduling: what happens when sync demand exceeds
channel capacity (PUBLISHABLE.md result 3's missing operating point).

The uncontended demonstration (smart_sync_study.py) shows the scheduler
RETURNS airtime the channel did not need. This study caps the channel
(``--capacity`` two-way exchanges per interval, swept) below what a
fixed-cadence schedule demands, and asks which policy keeps the array
coherent when there is not enough channel to go around:

  uniform      the conventional baseline: the same first-C links every
               interval. Under contention it permanently starves the
               rest of the array - the honest failure mode of fixed
               allocation, not a strawman.
  roundrobin   uninformed rotation at the same capacity (acquisition
               forced first). The fair share-the-channel baseline.
  scheduled    uncertainty-driven (predicted std vs budget).
  whittle      restless-bandit myopic index: service the links whose
               predicted budget-violation probability GROWS fastest if
               they coast.
  oracle       genie upper bound: ranks by the true residual no online
               policy can see. The gap oracle-to-scheduled bounds what
               better estimation could still buy.

Verification is counted waveform detection at the two coverage-edge
waypoints, exactly as in smart_sync_study.py.

Usage:
    .venv/bin/python contention_study.py
    .venv/bin/python contention_study.py --capacities 1,2,4 --stations 10
    .venv/bin/python contention_study.py --multi-fidelity   # micro pilots
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from detection import DetectionParams
from detection.rt_echo import rt_steered_legs
from detection.waveform import run_waveform_detection
from ota_sync import SDRSimulationConfig
from ota_sync.scheduled import SCHEDULER_POLICIES, run_scheduled_star


def effective_gain(result) -> tuple[float, str]:
    """Steady-state array gain, falling back to the tail mean when the
    run never reaches all-stations-steady (uniform under contention
    never does: starved links never calibrate)."""

    gain = result.mean_array_gain
    if gain == gain:  # not NaN
        return gain, ""
    tail = result.array_gain[-max(1, result.array_gain.numel() // 4):]
    return torch.mean(tail).item(), "*"


def detection_matrix(result) -> torch.Tensor:
    """Residual matrix for the counted-detection test. When the run
    never reaches all-stations steady (starved stations), fall back to
    the tail intervals so the starved stations' free-running phases are
    measured honestly instead of crashing the test."""

    matrix = result.residual_matrix()
    if matrix.shape[1] > 0:
        return matrix
    intervals = result.residuals.shape[1]
    tail = slice(max(0, intervals - max(1, intervals // 4)), intervals)
    rows = [torch.zeros(tail.stop - tail.start, dtype=torch.float64)]
    for row in result.residuals:
        rows.append(row[tail])
    return torch.stack(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="scheduling policies on a contended sync channel"
    )
    parser.add_argument("--stations", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--capacities", type=str, default="2,4,8",
        help="comma list of channel capacities (two-way exchanges per "
        "interval) to sweep; the uncontended demand is stations-1",
    )
    parser.add_argument(
        "--policies", type=str,
        default="uniform,roundrobin,scheduled,whittle,oracle",
        help=f"comma list from {SCHEDULER_POLICIES}",
    )
    parser.add_argument("--flat-budget", type=float, default=0.314)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--multi-fidelity", action="store_true",
                        help="allow cheap phase-only micro-pilot services")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    n = args.stations
    capacities = [int(v) for v in args.capacities.split(",")]
    policies = [p.strip() for p in args.policies.split(",")]
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
    edge_targets = np.array(
        [centroid + [1200.0, 150.0], centroid + [-1200.0, 150.0]]
    )
    print("ray tracing the edge legs (once per deployment)...")
    legs = rt_steered_legs(positions, edge_targets)

    print(
        f"Contended-channel scheduling, N={n} star "
        f"({n - 1} links), {args.iterations} intervals, flat budget "
        f"{1e3 * args.flat_budget:.0f} mrad"
        + (", multi-fidelity pilots ON" if args.multi_fidelity else "")
    )
    print(
        "uncontended demand: every fixed-cadence scheme wants "
        f"{n - 1} exchanges/interval; the physical channel fits "
        f"{max(1, int(1.0 / probe.airtime_uniform_fraction * (n - 1)))}"
    )

    physical_fit = max(1, int(1.0 / probe.airtime_uniform_fraction * (n - 1)))
    table = []  # (capacity, policy, result, gain, flag, pd)
    for capacity in capacities:
        overfit = (
            "  [capacity above the physical fit of "
            f"{physical_fit}: airtime >100% is not realizable at this "
            "cadence - read it as demand, not spend]"
            if capacity > physical_fit
            else ""
        )
        print(f"\n--- capacity {capacity} exchanges/interval "
              f"(demand {n - 1}) ---{overfit}")
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
            detect = run_waveform_detection(
                f"{policy}@{capacity}",
                positions,
                detection_matrix(result),
                edge_targets,
                params=params,
                trials=args.trials,
                seed=args.seed,
                leg_gains=legs,
            )
            table.append((capacity, policy, result, gain, flag, detect))
            worst = max(
                (v for v in result.station_steady_rms if v == v),
                default=float("nan"),
            )
            micro = (
                result.serviced_micro.sum().item()
                if result.serviced_micro is not None
                else 0
            )
            print(
                f"  {policy:<11} airtime {100 * result.airtime_used_fraction:5.1f}%  "
                f"gain {100 * gain:6.2f}%{flag or ' '} "
                f"worst-rms {1e3 * worst:7.0f} mrad  "
                f"edge Pd "
                + " ".join(f"{100 * pd:5.1f}%" for pd in detect.pd_measured)
                + (f"  ({micro} micro)" if args.multi_fidelity else "")
            )
    print(
        "\n(* = never reached all-stations steady: gain is the tail mean; "
        "residual/Pd rows include the starved stations)"
    )

    if args.no_plot:
        return

    from simulation import _render_figure_and_panels

    def gain_panel(axis):
        for policy in policies:
            xs = [c for c, p, *_ in table if p == policy]
            ys = [100 * g for c, p, r, g, f, d in table if p == policy]
            axis.plot(xs, ys, "o-", label=policy)
        axis.axhline(90.0, color="red", linestyle=":", linewidth=1.0)
        axis.set_xlabel("channel capacity (two-way exchanges per interval)")
        axis.set_ylabel("array coherent gain (%)")
        axis.legend(fontsize="small")

    def pd_panel(axis):
        for policy in policies:
            xs = [c for c, p, *_ in table if p == policy]
            ys = [
                100 * float(np.mean(d.pd_measured))
                for c, p, r, g, f, d in table
                if p == policy
            ]
            axis.plot(xs, ys, "o-", label=policy)
        axis.axhline(90.0, color="red", linestyle=":", linewidth=1.0)
        axis.set_xlabel("channel capacity (two-way exchanges per interval)")
        axis.set_ylabel("mean edge Pd (%)")
        axis.legend(fontsize="small")

    _render_figure_and_panels(
        f"Scheduling under contention (N={n} star, demand {n - 1} "
        "exchanges/interval)\nWhen the channel cannot carry a fixed "
        "cadence, informed scheduling decides which stations stay "
        "coherent.",
        [
            ("Array gain vs channel capacity, per policy", gain_panel),
            ("Counted edge detection vs channel capacity, per policy",
             pd_panel),
        ],
        (2, 1),
        (10.0, 9.0),
        sharex=False,
        top=0.90,
    )


if __name__ == "__main__":
    main()
