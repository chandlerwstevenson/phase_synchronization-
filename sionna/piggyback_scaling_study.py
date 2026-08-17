"""Does clutter-referenced (piggyback) sync scale with the number of
base stations?

The structural argument: the free one-way observations are broadcasts
(one reference sensing burst serves every station at once), so their
cost is flat in N; the two-way anchors are per-station and serial, so
anchor airtime grows like (N-1)/K - the same (N-1) growth every
two-way scheme pays every interval. The advantage should therefore
hold at ratio ~K while both walls move. This script measures instead
of arguing: piggyback (K=40) vs the scheduled two-way star at N = 2,
4, 6, 10, 14, same seeds, same physics.

Usage:
    .venv/bin/python piggyback_scaling_study.py
    .venv/bin/python piggyback_scaling_study.py --quick   # 1 seed, N<=6
"""

from __future__ import annotations

import argparse
import math

import torch

from clutter_sync_ofdm import run_piggyback_star
from ota_sync import SDRSimulationConfig
from ota_sync.scheduled import run_scheduled_star


def main() -> None:
    parser = argparse.ArgumentParser(
        description="piggyback sync scaling with array size"
    )
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--anchor-every", type=int, default=40)
    parser.add_argument("--stations", type=str, default="2,4,6,10,14")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    station_counts = [int(n) for n in args.stations.split(",")]
    if args.quick:
        seeds = seeds[:1]
        station_counts = [n for n in station_counts if n <= 6]

    print(
        f"Piggyback scaling: K={args.anchor_every}, "
        f"{args.iterations} intervals, seeds {seeds}"
    )
    print(
        f"{'N':>3} {'piggyback worst-rms':>20} {'gain':>7} {'airtime':>8} "
        f"{'| two-way worst-rms':>20} {'gain':>7} {'airtime':>8} "
        f"{'| airtime ratio':>15}"
    )

    for n in station_counts:
        pig_rms, pig_gain, pig_air = [], [], []
        two_rms, two_gain, two_air = [], [], []
        for seed in seeds:
            settings = SDRSimulationConfig(
                num_iterations=args.iterations, seed=seed, device="cpu"
            )
            pig = run_piggyback_star(
                settings,
                num_stations=n,
                anchor_every_intervals=args.anchor_every,
            )
            pig_rms.append(pig.worst_rms_mrad)
            pig_gain.append(pig.mean_array_gain)
            pig_air.append(pig.piggyback_airtime)

            two = run_scheduled_star(
                settings, num_stations=n, policy="scheduled"
            )
            worst = max(
                (v for v in two.station_steady_rms if v == v),
                default=float("nan"),
            )
            two_rms.append(1e3 * worst)
            two_gain.append(two.mean_array_gain)
            two_air.append(two.airtime_used_fraction)

        def ms(vals):
            m = sum(vals) / len(vals)
            s = (
                math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
                if len(vals) > 1
                else 0.0
            )
            return m, s

        pr, ps = ms(pig_rms)
        tr, ts = ms(two_rms)
        pa, _ = ms(pig_air)
        ta, _ = ms(two_air)
        pg, _ = ms(pig_gain)
        tg, _ = ms(two_gain)
        ratio = ta / pa if pa > 0 else float("inf")
        realizable = "" if ta <= 1.0 else " (two-way not realizable!)"
        print(
            f"{n:>3} {pr:>9.1f}±{ps:<5.1f} mrad {100 * pg:>6.1f}% "
            f"{100 * pa:>7.2f}% "
            f"| {tr:>9.1f}±{ts:<5.1f} mrad {100 * tg:>6.1f}% "
            f"{100 * ta:>7.2f}% "
            f"| {ratio:>7.1f}x{realizable}"
        )

    print(
        "\n(airtime ratio = two-way sync airtime / piggyback sync airtime "
        "at the same N; two-way above 100% cannot fit in the frame)"
    )


if __name__ == "__main__":
    main()
