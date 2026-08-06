"""Heterogeneous fleets and channel motion: where scheduling pays most
and where coasting breaks.

Part 1 - mixed oscillator classes. Real deployments mix hardware:
macro sites carry Stratum-3E OCXOs, small cells carry TCXOs with a
~100x larger frequency walk. A fixed cadence must poll everyone at the
worst station's rate; the scheduler reads each link's Kalman posterior,
so good crystals coast and the airtime dividend GROWS with fleet
spread. Per-station service rates make the mechanism visible.

Part 2 - channel motion (--speed-mps sweep). Coasting assumes the
channel phase holds still between pilots; a moving channel decorrelates
it and silently converts coast time into phase error the filter never
saw. Sweeping Doppler speed finds where uncertainty-driven coasting
breaks - the third simulation-discovered failure mode candidate, after
never-coast-during-acquisition and the periodic pi-branch check.

Usage:
    .venv/bin/python heterogeneous_fleet_study.py
    .venv/bin/python heterogeneous_fleet_study.py \
        --profiles ocxo,ocxo,ocxo,tcxo,tcxo,tcxo --speeds 0,1,3
"""

from __future__ import annotations

import argparse

from ota_sync import SDRSimulationConfig
from ota_sync.scheduled import run_scheduled_star


def summarize(label: str, result, budgets=None) -> None:
    rates = result.serviced.to(dtype=float).mean(dim=1)
    rate_text = " ".join(f"{100 * rate:3.0f}%" for rate in rates)
    print(
        f"  {label:<22} airtime {100 * result.airtime_used_fraction:5.1f}% "
        f"gain {100 * result.mean_array_gain:6.2f}%  "
        f"rms {[f'{1e3 * v:.0f}' for v in result.station_steady_rms]} mrad"
    )
    print(f"  {'':<22} service rate per station: {rate_text}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="scheduling gains on mixed-hardware fleets + motion"
    )
    parser.add_argument(
        "--profiles", type=str, default="ocxo,ocxo,ocxo,tcxo,tcxo,tcxo",
        help="comma list, one oscillator class per station "
        "(index 0 = reference); classes: ocxo, tcxo, sdr, custom",
    )
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flat-budget", type=float, default=0.314)
    parser.add_argument(
        "--speeds", type=str, default="0,1,3",
        help="comma list of channel speeds (m/s) for the motion sweep",
    )
    args = parser.parse_args()

    profiles = [p.strip() for p in args.profiles.split(",")]
    n = len(profiles)
    budgets = [args.flat_budget] * (n - 1)
    settings = SDRSimulationConfig(
        num_iterations=args.iterations, seed=args.seed, device="cpu"
    )

    print(f"Part 1 - heterogeneous fleet, N={n}: {profiles}")
    print("(homogeneous 'custom' rows below for the spread-free baseline)")
    for label, profile_list in (
        ("homogeneous custom", None),
        ("mixed fleet", profiles),
    ):
        for policy in ("uniform", "scheduled"):
            result = run_scheduled_star(
                settings,
                num_stations=n,
                policy=policy,
                budgets_rad=budgets,
                oscillator_profiles=profile_list,
            )
            summarize(f"{label} / {policy}", result)

    speeds = [float(v) for v in args.speeds.split(",")]
    print(
        "\nPart 2 - channel motion: scheduled coasting vs Doppler "
        f"(speeds {speeds} m/s, homogeneous oscillators)"
    )
    for speed in speeds:
        moving = SDRSimulationConfig(
            num_iterations=args.iterations,
            seed=args.seed,
            device="cpu",
            channel_speed_mps=speed,
        )
        for policy in ("uniform", "scheduled"):
            result = run_scheduled_star(
                moving, num_stations=n, policy=policy, budgets_rad=budgets
            )
            summarize(f"{speed:g} m/s / {policy}", result)
    print(
        "\nreading: if the scheduled row's residuals blow past the "
        "budget while uniform holds, the filter's coast-time rule is "
        "missing a channel-decorrelation term - the motion analogue of "
        "the acquisition and pi-branch failure modes."
    )


if __name__ == "__main__":
    main()
