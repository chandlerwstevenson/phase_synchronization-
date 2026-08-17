"""Scheduling-policy and clutter-family figures, plain matplotlib.

Every run is fresh (no cache reads; the *_data.json files are written
afterward as a record only). Produces, in figures/studies/:

  fig_sched_gain_vs_capacity.png   array gain vs channel capacity, per policy
  fig_sched_net_throughput.png     net throughput per policy and capacity
                                   (unrealizable configs are zero by data)
  fig_clutter_family_tradeoff.png  worst-station residual vs sync airtime
                                   for the four ways an N=6 array pays for sync
  scheduling_deployment_map.png    top-down N=10 seed-0 deployment with the
                                   reference station and edge-user/detection
                                   target locations

Usage:  .venv/bin/python fig_scheduling_runs.py  [--quick]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from clutter_sync_ofdm import run_piggyback_star
from multi_metric_study import (
    edge_targets,
    net_throughput,
    spectral_efficiency_draws,
    star_residual_matrix,
    summarize_se,
)
from ota_sync import SDRSimulationConfig
from ota_sync.network import place_stations
from ota_sync.scheduled import run_scheduled_star

FIGURES_DIR = Path(__file__).resolve().parent / "figures" / "studies"

POLICIES = ("uniform", "roundrobin", "scheduled", "oracle")
POLICY_LABEL = {
    "uniform": "uniform",
    "roundrobin": "round-robin",
    "scheduled": "scheduled (posterior)",
    "oracle": "oracle",
}


def save(fig, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def record(name: str, data) -> None:
    with open(FIGURES_DIR / f"{name}_data.json", "w") as handle:
        json.dump(data, handle, indent=1)


def star(seed, iterations, **kwargs):
    settings = SDRSimulationConfig(
        num_iterations=iterations, seed=seed, device="cpu"
    )
    return run_scheduled_star(settings, **kwargs)


def tail_gain(result) -> float:
    gain = result.mean_array_gain
    if gain == gain:
        return gain
    tail = result.array_gain[-max(1, result.array_gain.numel() // 4):]
    return torch.mean(tail).item()


def worst_rms_mrad(result) -> float:
    values = [v for v in result.station_steady_rms if v == v]
    if not values:
        matrix = star_residual_matrix(result)
        values = [
            torch.sqrt(torch.mean(row.square())).item()
            for row in matrix[1:]
        ]
    return 1e3 * max(values)


def figure_gain_vs_capacity(seeds, iterations, capacities):
    print("figure 1: gain vs capacity", flush=True)
    table = {}
    for policy in POLICIES:
        means = []
        for capacity in capacities:
            gains = [
                tail_gain(
                    star(
                        seed, iterations, num_stations=10,
                        policy=policy,
                        max_exchanges_per_interval=capacity,
                    )
                )
                for seed in seeds
            ]
            means.append(100.0 * float(np.mean(gains)))
            print(
                f"  {policy} cap {capacity}: {means[-1]:.1f}%", flush=True
            )
        table[policy] = means
    record("gain_vs_capacity", table)

    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    for policy in POLICIES:
        axis.plot(
            capacities, table[policy], "o-", label=POLICY_LABEL[policy]
        )
    axis.set_xlabel("channel capacity (two-way sync exchanges per interval)")
    axis.set_ylabel("array gain (% of perfect)")
    axis.set_title("Array gain vs sync capacity, per policy (N=10)")
    axis.legend()
    axis.grid(True)
    return save(fig, "fig_sched_gain_vs_capacity")


def figure_net_throughput(seeds, iterations, capacities):
    print("figure 2: net throughput", flush=True)
    table = {}
    for policy in POLICIES:
        values, overfit = [], []
        for capacity in capacities:
            draws, airtimes = [], []
            for seed in seeds:
                result = star(
                    seed, iterations, num_stations=10, policy=policy,
                    max_exchanges_per_interval=capacity,
                )
                matrix = star_residual_matrix(result)
                draws.append(
                    spectral_efficiency_draws(
                        result.positions, matrix,
                        edge_targets(result.positions),
                    )
                )
                airtimes.append(result.airtime_used_fraction)
            mean_se, _ = summarize_se(torch.cat(draws))
            airtime = float(np.mean(airtimes))
            values.append(net_throughput(mean_se, airtime))
            overfit.append(airtime > 1.0)
            print(
                f"  {policy} cap {capacity}: net {values[-1]:.2f} "
                f"(airtime {100 * airtime:.0f}%)", flush=True,
            )
        table[policy] = {"net": values, "overfit": overfit}
    record("net_throughput", table)

    fig, axis = plt.subplots(figsize=(6.8, 4.2))
    width = 0.19
    for index, policy in enumerate(POLICIES):
        positions = np.arange(len(capacities)) + (index - 1.5) * width
        axis.bar(
            positions, table[policy]["net"], width=width * 0.92,
            label=POLICY_LABEL[policy],
        )
    axis.set_xticks(np.arange(len(capacities)))
    axis.set_xticklabels([f"capacity {c}" for c in capacities])
    axis.set_ylabel("net throughput (bits/s/Hz after sync overhead)")
    axis.set_title(
        "Net throughput by policy and sync capacity (N=10);\n"
        "zero bars = sync demand exceeds the frame"
    )
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    axis.grid(True, axis="y")
    return save(fig, "fig_sched_net_throughput")


def figure_clutter_family(seeds, iterations):
    print("figure 3: clutter family trade-off", flush=True)
    schemes = []  # (label, residuals, airtimes)
    for label, runner in (
        (
            "two-way scheduled",
            lambda s: star(
                s, iterations, num_stations=6, policy="scheduled"
            ),
        ),
        (
            "micro-pilot star",
            lambda s: star(
                s, iterations, num_stations=6, policy="scheduled",
                multi_fidelity=True,
            ),
        ),
    ):
        residuals, airtimes = [], []
        for seed in seeds:
            result = runner(seed)
            residuals.append(worst_rms_mrad(result))
            airtimes.append(result.airtime_used_fraction)
        schemes.append((label, residuals, airtimes))
        print(f"  {label}: {np.mean(residuals):.0f} mrad", flush=True)
    for label, cadence in (
        ("piggyback, anchors every 5", 5),
        ("piggyback, anchors every 40", 40),
    ):
        residuals, airtimes = [], []
        for seed in seeds:
            settings = SDRSimulationConfig(
                num_iterations=iterations, seed=seed, device="cpu"
            )
            result = run_piggyback_star(
                settings, num_stations=6, anchor_every_intervals=cadence,
            )
            residuals.append(result.worst_rms_mrad)
            airtimes.append(result.piggyback_airtime)
        schemes.append((label, residuals, airtimes))
        print(f"  {label}: {np.mean(residuals):.0f} mrad", flush=True)
    record(
        "clutter_family_tradeoff",
        {
            label: {
                "residuals_mrad": residuals,
                "airtime_fractions": airtimes,
            }
            for label, residuals, airtimes in schemes
        },
    )

    fig, axis = plt.subplots(figsize=(6.4, 4.4))
    for label, residuals, airtimes in schemes:
        axis.errorbar(
            100.0 * float(np.mean(airtimes)),
            float(np.mean(residuals)),
            yerr=float(np.std(residuals)),
            fmt="o", markersize=8, capsize=3, label=label,
        )
    axis.set_xlabel("sync airtime (% of the frame)")
    axis.set_ylabel("worst-station residual (mrad, root-mean-square)")
    axis.set_title(
        "Worst-station residual vs sync airtime (N=6, seeds 0-2)"
    )
    axis.set_xlim(left=-2)
    axis.legend()
    axis.grid(True)
    return save(fig, "fig_clutter_family_tradeoff")


def figure_deployment_map():
    """Top-down map of the N=10 seed-0 deployment used by the
    scheduling metric figures: station positions from the same
    place_stations call the simulator makes, the reference station,
    and the two edge-user / detection-target locations."""

    print("figure 4: deployment map", flush=True)
    positions = place_stations(10, 500.0, 0)
    targets = edge_targets(positions)
    centroid = positions.mean(axis=0)
    record(
        "scheduling_deployment_map",
        {
            "stations_m": positions.tolist(),
            "targets_m": targets.tolist(),
            "centroid_m": centroid.tolist(),
        },
    )

    fig, axis = plt.subplots(figsize=(6.0, 5.2))
    axis.scatter(
        positions[1:, 0], positions[1:, 1], marker="o", s=55,
        label="station",
    )
    axis.scatter(
        positions[0, 0], positions[0, 1], marker="s", s=90,
        label="reference station",
    )
    axis.scatter(
        targets[:, 0], targets[:, 1], marker="*", s=140,
        label="edge user / detection target",
    )
    axis.scatter(
        centroid[0], centroid[1], marker="+", s=90,
        label="array centroid",
    )
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title(
        "Deployment used by the scheduling metrics (N=10, seed 0)"
    )
    axis.set_aspect("equal")
    everything = np.vstack([positions, targets])
    axis.set_xlim(everything[:, 0].min() - 150, everything[:, 0].max() + 150)
    axis.set_ylim(everything[:, 1].min() - 150, everything[:, 1].max() + 150)
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    axis.grid(True)
    return save(fig, "scheduling_deployment_map")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    seeds = [0] if args.quick else [0, 1, 2]
    iterations = 20 if args.quick else 60
    capacities_line = [1, 4, 8] if args.quick else [1, 2, 3, 4, 6, 8]
    paths = [
        figure_gain_vs_capacity(
            seeds, 20 if args.quick else 50, capacities_line
        ),
        figure_net_throughput(seeds, iterations, [2, 4, 8]),
        figure_clutter_family(seeds, iterations),
        figure_deployment_map(),
    ]
    for path in paths:
        print("saved", path)


if __name__ == "__main__":
    main()
