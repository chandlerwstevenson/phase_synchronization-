"""Figures for the detection/gain dissociation study (fresh re-run).

Fig 1: detection gap (posterior gate minus all-in) vs channel capacity.
Fig 2: mechanism split at capacity 2 (transmit-only / receive-only / both).
Fig 3: power-matched control.
Plain default matplotlib; no in-axes annotations. Reduced trial counts
vs the original study (stated in output).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from detection import DetectionParams
from fig_membership_common import save_fig
from gating_dissociation_study import (
    _edge_targets,
    run_split_waveform_detection,
)
from gating_study import (
    evaluation_mask,
    membership_weights,
    phase_matrix,
    run_star_with_posteriors,
    weighted_gain,
)
from ota_sync import SDRSimulationConfig
import matplotlib.pyplot as plt

N = 10
CAPACITIES = (1, 2, 3, 4)
SEEDS = (0, 1, 2, 3, 4)
MECH_SEEDS = (0, 1, 2)
GATE = math.pi / 2.0
TRIALS = 250
H0 = 10000
PARAMS = DetectionParams(tx_power_w=0.5)


def _run(seed: int, capacity: int):
    settings = SDRSimulationConfig(
        num_iterations=50, seed=seed, device="cpu"
    )
    return run_star_with_posteriors(
        settings, num_stations=N, policy="uniform",
        budgets_rad=[0.314] * (N - 1),
        max_exchanges_per_interval=capacity,
    )


def _prep(run):
    result, sigma = run
    mask = evaluation_mask(result)
    return phase_matrix(result)[:, mask], sigma[:, mask], result


def _detect(phases, positions, tx, rx, seed):
    detect = run_split_waveform_detection(
        "x", positions, phases, tx, rx, _edge_targets(positions),
        params=PARAMS, trials=TRIALS, h0_trials=H0, seed=seed,
    )
    return 100.0 * float(np.mean(detect.pd_measured))


def main() -> None:
    print(f"trials={TRIALS}, h0={H0} (reduced vs study's 300/15000)")

    # ---- Fig 1: gap vs capacity ----------------------------------
    gaps = {c: [] for c in CAPACITIES}
    for capacity in CAPACITIES:
        for seed in SEEDS:
            phases, sig, result = _prep(_run(seed, capacity))
            positions = result.positions
            ones = torch.ones_like(phases)
            gate_w = membership_weights("gate", phases, sig, GATE)
            pd_all = _detect(phases, positions, ones, ones, seed)
            pd_gate = _detect(phases, positions, gate_w, gate_w, seed)
            gaps[capacity].append(pd_gate - pd_all)
            print(f"cap {capacity} seed {seed}: gap {pd_gate - pd_all:+.1f}")

    figure, axis = plt.subplots(figsize=(5.2, 3.4))
    for capacity in CAPACITIES:
        axis.scatter(
            [capacity] * len(SEEDS), gaps[capacity], color="C0",
            alpha=0.45, s=22, label="per seed" if capacity == 1 else None,
        )
    means = [float(np.mean(gaps[c])) for c in CAPACITIES]
    axis.plot(CAPACITIES, means, "-o", color="C0", label="mean")
    axis.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    axis.set_xticks(list(CAPACITIES))
    axis.set_xlabel("Sync channel capacity (exchanges per interval)")
    axis.set_ylabel("Detection gap, gate minus all-in (points)")
    axis.legend()
    axis.set_title(
        f"Detection gap vs sync capacity (N={N}, demand {N - 1})"
    )
    print("saved", save_fig(figure, "dissociation_gap_vs_capacity"))

    # ---- Fig 2: mechanism split at capacity 2 --------------------
    lifts = {"transmit only": [], "receive only": [], "both": []}
    for seed in MECH_SEEDS:
        phases, sig, result = _prep(_run(seed, 2))
        positions = result.positions
        ones = torch.ones_like(phases)
        gate_w = membership_weights("gate", phases, sig, GATE)
        base = _detect(phases, positions, ones, ones, seed)
        lifts["transmit only"].append(
            _detect(phases, positions, gate_w, ones, seed) - base
        )
        lifts["receive only"].append(
            _detect(phases, positions, ones, gate_w, seed) - base
        )
        lifts["both"].append(
            _detect(phases, positions, gate_w, gate_w, seed) - base
        )
        print(f"mech seed {seed} done")
    print("mechanism lifts:", {k: [f"{v:.1f}" for v in lifts[k]] for k in lifts})

    figure, axis = plt.subplots(figsize=(4.6, 3.3))
    names = list(lifts)
    values = [float(np.mean(lifts[k])) for k in names]
    axis.bar(names, values, color="C0", width=0.6)
    for x, k in enumerate(names):
        axis.scatter([x] * len(MECH_SEEDS), lifts[k], color="k", s=14)
    axis.set_ylabel("Detection lift over all-in (points)")
    axis.set_title("Gate applied per side (capacity 2)")
    print("saved", save_fig(figure, "dissociation_mechanism_split"))

    # ---- Fig 3: power-matched control ----------------------------
    phases, sig, result = _prep(_run(0, 2))
    positions = result.positions
    ones = torch.ones_like(phases)
    gate_w = membership_weights("gate", phases, sig, GATE)
    gain_all = torch.mean(weighted_gain(phases, ones)).item()
    gain_gate = torch.mean(weighted_gain(phases, gate_w)).item()
    scale = math.sqrt(gain_gate / gain_all)
    pd_all = _detect(phases, positions, ones, ones, 0)
    pd_matched = _detect(phases, positions, scale * ones, ones, 0)
    pd_gate = _detect(phases, positions, gate_w, gate_w, 0)
    print(f"tx scale {scale:.3f}: all-in {pd_all:.1f}, "
          f"matched {pd_matched:.1f}, gate {pd_gate:.1f}")

    figure, axis = plt.subplots(figsize=(4.6, 3.3))
    labels = ["all-in", f"all-in,\npower x{scale:.2f}", "posterior gate"]
    axis.bar(labels, [pd_all, pd_matched, pd_gate],
             color=["C0", "C0", "C1"], width=0.6)
    axis.set_ylabel("Mean probability of detection (%)")
    axis.set_ylim(0, 105)
    axis.set_title("Power-matched control (capacity 2, seed 0)")
    print("saved", save_fig(figure, "dissociation_power_matched"))


if __name__ == "__main__":
    main()
