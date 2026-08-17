"""Figure for the hybrid two-tier combiner study (fresh re-run).

Grouped bars: probability of detection per combiner across the five
operating regimes. Plain default matplotlib; no in-axes annotations.
Seeds 0-1 and reduced trial counts vs the original study (stated in
output).
"""

from __future__ import annotations

import math

import numpy as np

from detection import DetectionParams
from fig_membership_common import METHOD_COLORS, METHOD_LABELS, save_fig
from gating_study import (
    evaluation_mask,
    phase_matrix,
    run_star_with_posteriors,
)
from hybrid_combiner_study import (
    run_hybrid_waveform_detection,
    variant_weights,
)
from ota_sync import SDRSimulationConfig
import matplotlib.pyplot as plt

N = 10
SEEDS = (0, 1)
GATE = math.pi / 2.0
TRIALS = 250
H0 = 8000
REGIMES = (
    ("cap 2\n0.5 W", 2, 0.5),
    ("cap 3\n0.5 W", 3, 0.5),
    ("cap 2\n0.05 W", 2, 0.05),
    ("cap 8\n0.05 W", 8, 0.05),
    ("cap 8\n0.02 W", 8, 0.02),
)
VARIANTS = (
    ("all-in", "ones", "discard"),
    ("gate-discard", "posterior", "discard"),
    ("hybrid-post", "posterior", "noncoherent"),
    ("hybrid-oracle", "oracle", "noncoherent"),
    ("noncoh-all", "zeros", "noncoherent"),
)


def main() -> None:
    print(f"seeds {SEEDS}, trials={TRIALS}, h0={H0} "
          "(reduced vs study's 0-2 / 400 / 15000)")
    table = {}
    for label, capacity, power in REGIMES:
        params = DetectionParams(tx_power_w=power)
        for name in (v[0] for v in VARIANTS):
            table[(label, name)] = []
        for seed in SEEDS:
            settings = SDRSimulationConfig(
                num_iterations=60, seed=seed, device="cpu"
            )
            result, sigma = run_star_with_posteriors(
                settings, num_stations=N, policy="uniform",
                budgets_rad=[0.314] * (N - 1),
                max_exchanges_per_interval=capacity,
            )
            mask = evaluation_mask(result)
            phases = phase_matrix(result)[:, mask]
            sig = sigma[:, mask]
            positions = result.positions
            centroid = positions.mean(axis=0)
            targets = np.array(
                [centroid + [1200.0, 150.0], centroid + [-1200.0, 150.0]]
            )
            for name, source, mode in VARIANTS:
                weights = variant_weights(source, phases, sig, GATE)
                detect = run_hybrid_waveform_detection(
                    name, positions, phases, weights, mode, targets,
                    params=params, trials=TRIALS, h0_trials=H0,
                    seed=seed,
                )
                table[(label, name)].append(
                    100.0 * float(np.mean(detect.pd_measured))
                )
            print(f"regime {label!r} seed {seed} done")

    figure, axis = plt.subplots(figsize=(8.2, 3.9))
    xs = np.arange(len(REGIMES))
    width = 0.15
    names = [v[0] for v in VARIANTS]
    for index, name in enumerate(names):
        values = [
            float(np.mean(table[(label, name)]))
            for label, _, _ in REGIMES
        ]
        axis.bar(
            xs + (index - 2) * width, values, width=width,
            color=METHOD_COLORS[name], label=METHOD_LABELS[name],
        )
    axis.set_xticks(xs)
    axis.set_xticklabels([label for label, _, _ in REGIMES])
    axis.set_ylabel("Mean probability of detection (%)")
    axis.set_ylim(0, 105)
    axis.legend(loc="lower left", ncol=2, fontsize=8)
    axis.set_title(
        f"Probability of detection by combiner and regime (N={N})"
    )
    print("saved", save_fig(figure, "hybrid_combiner_by_regime"))
    for label, _, _ in REGIMES:
        row = {n: f"{np.mean(table[(label, n)]):.1f}" for n in names}
        print(label.replace(chr(10), " "), row)


if __name__ == "__main__":
    main()
