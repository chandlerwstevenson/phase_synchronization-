"""Figures for the base membership-gating study (fresh re-run).

Fig 1: mean array gain per membership policy under contention.
Fig 2: counted detection per membership policy at the two edge targets.
Plain default matplotlib; no in-axes annotations.
"""

from __future__ import annotations

import numpy as np
import torch

from detection import DetectionParams
from fig_membership_common import METHOD_COLORS, METHOD_LABELS, save_fig
from gating_study import (
    evaluation_mask,
    membership_weights,
    phase_matrix,
    run_gated_waveform_detection,
    run_star_with_posteriors,
    weighted_gain,
)
from ota_sync import SDRSimulationConfig
import matplotlib.pyplot as plt

N = 10
CAPACITY = 2
SEEDS = (0, 1, 2)
GAIN_METHODS = ("all-in", "gate", "oracle", "greedy")
DETECT_METHODS = ("all-in", "gate", "oracle")
GATE = float(np.pi / 2.0)


def main() -> None:
    runs = {}
    for seed in SEEDS:
        settings = SDRSimulationConfig(
            num_iterations=50, seed=seed, device="cpu"
        )
        runs[seed] = run_star_with_posteriors(
            settings, num_stations=N, policy="uniform",
            budgets_rad=[0.314] * (N - 1),
            max_exchanges_per_interval=CAPACITY,
        )
        print(f"sync run seed {seed} done")

    # ---- Fig 1: gain per membership, per-seed dots + mean bar ------
    gains = {m: [] for m in GAIN_METHODS}
    for seed in SEEDS:
        result, sigma = runs[seed]
        mask = evaluation_mask(result)
        phases = phase_matrix(result)[:, mask]
        sig = sigma[:, mask]
        for method in GAIN_METHODS:
            weights = membership_weights(method, phases, sig, GATE)
            gains[method].append(
                100.0 * torch.mean(weighted_gain(phases, weights)).item()
            )
    print("gains:", {m: [f"{v:.1f}" for v in gains[m]] for m in GAIN_METHODS})

    figure, axis = plt.subplots(figsize=(5.4, 3.4))
    xs = np.arange(len(GAIN_METHODS))
    means = [float(np.mean(gains[m])) for m in GAIN_METHODS]
    axis.bar(
        xs, means, width=0.62,
        color=[METHOD_COLORS[m] for m in GAIN_METHODS],
    )
    for x, method in zip(xs, GAIN_METHODS):
        axis.scatter([x] * len(SEEDS), gains[method], color="k", s=14)
    axis.set_xticks(xs)
    axis.set_xticklabels([METHOD_LABELS[m] for m in GAIN_METHODS])
    axis.set_ylabel("Mean array gain (% of perfect)")
    axis.set_title(
        f"Mean array gain by membership policy "
        f"(N={N}, capacity {CAPACITY}/{N - 1})"
    )
    print("saved", save_fig(figure, "membership_gain_by_policy"))

    # ---- Fig 2: detection per membership, both edge targets --------
    result0, sigma0 = runs[0]
    mask = evaluation_mask(result0)
    phases = phase_matrix(result0)[:, mask]
    sig = sigma0[:, mask]
    positions = result0.positions
    centroid = positions.mean(axis=0)
    targets = np.array(
        [centroid + [1200.0, 150.0], centroid + [-1200.0, 150.0]]
    )
    params = DetectionParams(tx_power_w=0.5)
    pds = {}
    for method in DETECT_METHODS:
        weights = membership_weights(method, phases, sig, GATE)
        detect = run_gated_waveform_detection(
            method, positions, phases, weights, targets,
            params=params, trials=300, h0_trials=12000, seed=0,
        )
        pds[method] = [100.0 * v for v in detect.pd_measured]
        print(f"detection {method}: {pds[method]}")

    figure, axis = plt.subplots(figsize=(5.4, 3.4))
    xs = np.arange(len(DETECT_METHODS))
    width = 0.34
    for offset, target_label in enumerate(
        ("edge target A", "edge target B")
    ):
        values = [pds[m][offset] for m in DETECT_METHODS]
        axis.bar(
            xs + (offset - 0.5) * width + width / 2, values, width=width,
            color=f"C{offset}", label=target_label,
        )
    axis.set_xticks(xs)
    axis.set_xticklabels([METHOD_LABELS[m] for m in DETECT_METHODS])
    axis.set_ylabel("Probability of detection (%)")
    axis.set_ylim(0, 105)
    axis.legend(loc="lower right")
    axis.set_title(
        f"Probability of detection by membership policy "
        f"(N={N}, capacity {CAPACITY}/{N - 1}, seed 0)"
    )
    print("saved", save_fig(figure, "membership_detection_by_policy"))


if __name__ == "__main__":
    main()
