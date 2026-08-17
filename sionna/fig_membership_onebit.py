"""Figure for the 1-bit membership study (fresh re-run).

Mean array gain vs feedback bit-error rate for the 1-bit alignment
rule, with flat references (all-in, posterior gate, oracle, greedy) as
labeled horizontal lines. Plain default matplotlib; no in-axes
annotations.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from fig_membership_common import METHOD_COLORS, METHOD_LABELS, save_fig
from gating_study import (
    evaluation_mask,
    membership_weights,
    phase_matrix,
    run_star_with_posteriors,
    weighted_gain,
)
from opportunistic_membership_study import alignment_bits, onebit_weights
from ota_sync import SDRSimulationConfig
import matplotlib.pyplot as plt

N = 10
CAPACITY = 2
SEEDS = (0, 1, 2, 3, 4)
EPS_GRID = (0.0, 0.025, 0.05, 0.1, 0.15, 0.2)
GATE = math.pi / 2.0
REFS = ("all-in", "gate", "oracle", "greedy")


def main() -> None:
    prepared = []
    for seed in SEEDS:
        settings = SDRSimulationConfig(
            num_iterations=50, seed=seed, device="cpu"
        )
        result, sigma = run_star_with_posteriors(
            settings, num_stations=N, policy="uniform",
            budgets_rad=[0.314] * (N - 1),
            max_exchanges_per_interval=CAPACITY,
        )
        mask = evaluation_mask(result)
        prepared.append(
            (seed, phase_matrix(result)[:, mask], sigma[:, mask])
        )
        print(f"sync run seed {seed} done")

    onebit_curve = []
    for eps in EPS_GRID:
        values = []
        for seed, phases, sig in prepared:
            generator = torch.Generator().manual_seed(9000 + seed)
            bits = alignment_bits(phases, eps, generator)
            weights = onebit_weights(bits)
            values.append(
                100.0 * torch.mean(weighted_gain(phases, weights)).item()
            )
        onebit_curve.append(float(np.mean(values)))
    print("1-bit gain vs eps:", [f"{v:.1f}" for v in onebit_curve])

    refs = {}
    for name in REFS:
        values = []
        for seed, phases, sig in prepared:
            weights = membership_weights(name, phases, sig, GATE)
            values.append(
                100.0 * torch.mean(weighted_gain(phases, weights)).item()
            )
        refs[name] = float(np.mean(values))
    print("references:", {k: f"{v:.1f}" for k, v in refs.items()})

    figure, axis = plt.subplots(figsize=(6.0, 3.8))
    eps_pct = [100.0 * e for e in EPS_GRID]
    axis.plot(
        eps_pct, onebit_curve, "-o", color=METHOD_COLORS["1-bit"],
        label=METHOD_LABELS["1-bit"],
    )
    for name in REFS:
        axis.axhline(
            refs[name], color=METHOD_COLORS[name], linestyle="--",
            linewidth=1.2, label=METHOD_LABELS[name],
        )
    axis.set_xlabel("Feedback bit-error rate (%)")
    axis.set_ylabel("Mean array gain (% of perfect)")
    axis.legend(fontsize=8)
    axis.set_title(
        f"1-bit membership gain vs feedback bit-error rate "
        f"(N={N}, capacity {CAPACITY}/{N - 1})"
    )
    print("saved", save_fig(figure, "onebit_gain_vs_bit_error"))


if __name__ == "__main__":
    main()
