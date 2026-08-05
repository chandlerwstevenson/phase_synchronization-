"""One command: every method on the SAME random deployment, with figures.

Runs the N-station network simulation for each synchronization method
(two-way, micro, hybrid) on an identical random deployment (same seed =
same station positions), printing each method's per-station table and
opening its figures — deployment map, per-station residuals, array
coherent gain — plus the clean individual panels.

Usage:
    .venv/bin/python run_everything.py                 # 6 stations, all methods
    .venv/bin/python run_everything.py --stations 10
    .venv/bin/python run_everything.py --models twoway hybrid --iterations 60

Note: figures open after each method; close them to advance to the next
method. The station map is identical across methods (same seed), so the
comparison is apples to apples.
"""

from __future__ import annotations

import argparse
import sys

import simulation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="run every sync method on one shared random deployment"
    )
    parser.add_argument("--stations", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument(
        "--models",
        nargs="*",
        default=["twoway", "micro", "hybrid"],
        choices=["sdr", "twoway", "micro", "hybrid"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--snr-db", type=float, default=20.0)
    parser.add_argument("--area-radius-m", type=float, default=500.0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    for index, model in enumerate(args.models):
        print()
        print(f"########## method {index + 1}/{len(args.models)}: {model} "
              f"##########")
        argv = [
            "simulation.py",
            "--model", model,
            "--stations", str(args.stations),
            "--iterations", str(args.iterations),
            "--seed", str(args.seed),
            "--snr-db", str(args.snr_db),
            "--area-radius-m", str(args.area_radius_m),
        ]
        if not args.no_plot:
            argv.append("--plot")
        sys.argv = argv
        simulation.main()


if __name__ == "__main__":
    main()
