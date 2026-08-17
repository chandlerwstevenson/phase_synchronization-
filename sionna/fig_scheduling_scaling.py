"""Scaling and metric-transfer figures, plain matplotlib, fresh runs.

Nothing is read from the scaling cache: the sync-demand and contended
cells are re-simulated here, and the metric-transfer study is re-run
with its measurement cache set aside so every number is regenerated.
Produces, in figures/studies/:

  fig_airtime_wall.png             sync demand vs array size per policy,
                                   with the 100% frame limit
  fig_range_vs_N.png               detection range vs array size per
                                   policy, perfect-sync reference dashed
  fig_metric_rank_correlation.png  rank correlation of each user-facing
                                   metric with beam quality
  fig_metric_knee.png              sync capacity each metric says you
                                   need, per condition

Usage:  .venv/bin/python fig_scheduling_scaling.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from detection import DetectionParams
from detection.viability import detection_range_m
from multi_metric_study import mean_gain_from_matrix, star_residual_matrix
from ota_sync import SDRSimulationConfig
from ota_sync.scheduled import run_scheduled_star

FIGURES_DIR = Path(__file__).resolve().parent / "figures" / "studies"
POLICIES = ("uniform", "roundrobin", "scheduled", "oracle")
POLICY_LABEL = {
    "uniform": "uniform",
    "roundrobin": "round-robin",
    "scheduled": "scheduled (posterior)",
    "oracle": "oracle",
}
STATION_COUNTS = (6, 10, 14, 20)
PYTHON = sys.executable


def save(fig, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def record(name: str, data) -> None:
    with open(FIGURES_DIR / f"{name}_data.json", "w") as handle:
        json.dump(data, handle, indent=1)


def star(seed, num_stations, policy, capacity, iterations=60):
    settings = SDRSimulationConfig(
        num_iterations=iterations, seed=seed, device="cpu"
    )
    return run_scheduled_star(
        settings,
        num_stations=num_stations,
        policy=policy,
        max_exchanges_per_interval=capacity,
    )


def fresh_demand(seeds) -> list[dict]:
    """Sync airtime per policy when the channel never refuses service
    (informed policies at capacity N-1; uniform/round-robin demand is
    the full rate by definition)."""

    rows = []
    for num_stations in STATION_COUNTS:
        for policy in ("scheduled", "oracle"):
            airtimes, uniform_demand = [], None
            for seed in seeds:
                result = star(
                    seed, num_stations, policy, num_stations - 1
                )
                airtimes.append(result.airtime_used_fraction)
                uniform_demand = result.airtime_uniform_fraction
            rows.append(
                {
                    "n": num_stations,
                    "policy": policy,
                    "airtime": float(np.mean(airtimes)),
                    "uniform_demand": uniform_demand,
                }
            )
            print(
                f"  demand N={num_stations} {policy}: "
                f"{100 * rows[-1]['airtime']:.1f}% "
                f"(uniform {100 * uniform_demand:.1f}%)",
                flush=True,
            )
    return rows


def fresh_contended(seeds) -> list[dict]:
    """Mean gain and detection range per policy at contended capacity
    (~22% of links)."""

    params = DetectionParams(tx_power_w=0.5)
    rows = []
    for num_stations in STATION_COUNTS:
        capacity = max(1, round(0.22 * (num_stations - 1)))
        for policy in POLICIES:
            gains = []
            for seed in seeds:
                result = star(seed, num_stations, policy, capacity)
                gains.append(
                    mean_gain_from_matrix(star_residual_matrix(result))
                )
            gain = float(np.mean(gains))
            rows.append(
                {
                    "n": num_stations,
                    "policy": policy,
                    "capacity": capacity,
                    "gain": gain,
                    "range_m": detection_range_m(num_stations, gain, params),
                }
            )
            print(
                f"  contended N={num_stations} {policy}: gain "
                f"{100 * gain:.1f}%, range {rows[-1]['range_m']:.0f} m",
                flush=True,
            )
    return rows


def figure_airtime_wall(demand: list[dict]):
    ns = sorted({d["n"] for d in demand})
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.axhline(
        100.0, linestyle="--", color="gray", label="frame limit (100%)"
    )
    uniform_curve = [
        100 * next(d for d in demand if d["n"] == n)["uniform_demand"]
        for n in ns
    ]
    axis.plot(
        ns, uniform_curve, "o-",
        label="uniform / round-robin (identical demand)",
    )
    for policy in ("scheduled", "oracle"):
        curve = [
            100 * next(
                d for d in demand
                if d["n"] == n and d["policy"] == policy
            )["airtime"]
            for n in ns
        ]
        axis.plot(ns, curve, "o-", label=POLICY_LABEL[policy])
    axis.set_xlabel("array size (stations)")
    axis.set_ylabel("sync airtime demanded (% of the frame)")
    axis.set_title("Sync airtime demand vs array size, per policy")
    axis.legend()
    axis.grid(True)
    return save(fig, "fig_airtime_wall")


def figure_range_vs_n(cells: list[dict]):
    ns = sorted({c["n"] for c in cells})
    params = DetectionParams(tx_power_w=0.5)
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.plot(
        ns,
        [detection_range_m(n, 1.0, params) for n in ns],
        "--", color="gray", label="perfect sync (reference)",
    )
    for policy in POLICIES:
        curve = [
            next(
                c for c in cells if c["n"] == n and c["policy"] == policy
            )["range_m"]
            for n in ns
        ]
        axis.plot(ns, curve, "o-", label=POLICY_LABEL[policy])
    axis.set_xlabel("array size (stations)")
    axis.set_ylabel("detection range (m)")
    axis.set_title(
        "Detection range vs array size, per policy (contended capacity)"
    )
    axis.legend()
    axis.grid(True)
    return save(fig, "fig_range_vs_N")


def metric_theory_output() -> str:
    """Re-run the metric-transfer study with its measurement cache set
    aside, so every measurement is fresh."""

    runs_path = "metric_theory_runs.json"
    backup = runs_path + ".pre_figures"
    if os.path.exists(runs_path):
        os.replace(runs_path, backup)
        print(f"  set aside {runs_path} -> {backup}", flush=True)
    result = subprocess.run(
        [PYTHON, "metric_theory_study.py"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def figure_rank_correlation(stdout: str):
    block = stdout.split("rank correlation")[1]
    pairs = re.findall(r"(\w+)\s+([+-]\d+\.\d+)", block)
    label_map = {
        "se_edge": "mean throughput, edge user",
        "se_edge_95": "guaranteed (95%) throughput, edge",
        "se_strong": "mean throughput, nearby user",
        "se_phys": "mean throughput, physical link",
        "range_m": "detection range",
        "net_throughput": "net throughput (after sync cost)",
    }
    names, values = [], []
    for key, value in pairs:
        if key in label_map:
            names.append(label_map[key])
            values.append(float(value))
    order = np.argsort(values)
    names = [names[i] for i in order]
    values = [values[i] for i in order]
    record(
        "metric_rank_correlation", dict(zip(names, values))
    )
    fig, axis = plt.subplots(figsize=(6.6, 3.6))
    axis.barh(names, values)
    axis.axvline(0.0, color="gray", linewidth=1.0)
    axis.set_xlabel(
        "rank correlation with beam quality (1.0 = same ordering)"
    )
    axis.set_title("Metric rank correlation with beam quality (219 runs)")
    axis.grid(True, axis="x")
    return save(fig, "fig_metric_rank_correlation")


def figure_knee(stdout: str):
    block = stdout.split("by metric, per curve")[1]
    rows = []
    for line in block.splitlines():
        match = re.match(
            r"\s+N=(\d+)\s+(\w+)\s+B=([\d.]+)\s+L=(\d+)\s+gain=(\S+)\s+"
            r"se_edge=(\S+)\s+se_strong=(\S+)\s+range_m=(\S+)\s+nt\*=(\S+)",
            line,
        )
        if match:
            rows.append(match.groups())
    record("metric_knee", rows)
    metrics = [
        ("gain", 4, "beam quality"),
        ("se_edge", 5, "edge-user throughput"),
        ("se_strong", 6, "nearby-user throughput"),
        ("range_m", 7, "detection range"),
        ("nt", 8, "net throughput"),
    ]
    fig, axis = plt.subplots(figsize=(6.8, 4.6))
    labels = []
    for row_index, row in enumerate(rows):
        labels.append(f"N={row[0]} {row[1]} B={row[2]} L={row[3]}")
    for metric_index, (_, column, metric_label) in enumerate(metrics):
        xs, ys = [], []
        for row_index, row in enumerate(rows):
            value = row[column]
            if value == "-":
                continue
            xs.append(int(value))
            # Stagger metrics vertically inside each row so coincident
            # knee values stay individually visible.
            ys.append(row_index + (metric_index - 2) * 0.16)
        axis.plot(xs, ys, "o", markersize=6, label=metric_label)
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel(
        "sync capacity needed to reach 90% of the metric's plateau"
    )
    axis.set_title("Required sync capacity by metric, per condition")
    axis.legend(fontsize=8, loc="lower right")
    axis.grid(True, axis="x")
    return save(fig, "fig_metric_knee")


def main() -> None:
    seeds = [0, 1, 2]
    print("fresh demand runs...", flush=True)
    demand = fresh_demand(seeds)
    record("airtime_wall", demand)
    print("fresh contended runs...", flush=True)
    cells = fresh_contended(seeds)
    record("range_vs_n", cells)
    print("saved", figure_airtime_wall(demand))
    print("saved", figure_range_vs_n(cells))
    print("re-running metric_theory_study.py (fresh measurements)...",
          flush=True)
    stdout = metric_theory_output()
    print("saved", figure_rank_correlation(stdout))
    print("saved", figure_knee(stdout))


if __name__ == "__main__":
    main()
