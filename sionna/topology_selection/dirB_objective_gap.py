"""Direction B: the coherent-gain objective is not the phase-MSE
objective, and topology selection under a budget can tell them apart.

Setup. A selected set E of two-way sync edges (each measurement
variance r2, unit airtime cost) gives BLUE phase-error pair variances

    Var(e_i - e_k) = r2 * R_ik(E),

R_ik the unit-conductance effective resistance (machinery verified in
../phase_sync_idea/openloop_graph_theory.py; the formulation is prior
art - Barooah-Hespanha, Karp et al., Howard et al. - cited, not
claimed). For Gaussian errors and beamforming amplitudes a_i the
expected coherent gain is EXACT (not approximate):

    E[G(E)] = sum_ik a_i a_k exp(-r2 R_ik(E)/2) / (sum_i a_i)^2

because E[exp(j(e_i-e_k))] = exp(-Var/2) for joint Gaussians (the
pairwise variance absorbs the correlation term). Pairs split across
components have infinite variance and contribute zero - so the gain
objective remains finite on disconnected graphs and can choose to
abandon a node; the MSE objective cannot.

The three objectives compared, all on the same covariance:

    GAIN  maximize  sum_ik a_i a_k exp(-r2 R_ik/2)
    MSE   minimize  sum_{i<k} R_ik      (Kirchhoff index; A-optimality,
                                         gauge-symmetric; infinite if
                                         disconnected)
    WORST minimize  max_ik R_ik

Why they can disagree - two mechanisms, derived:

1. AMPLITUDE WEIGHTING (first order). Expanding exp for small
   r2*R: maximizing gain is minimizing sum_ik a_i a_k R_ik - a
   WEIGHTED Kirchhoff index. Whenever amplitudes are heterogeneous
   this differs from the unweighted sum at first order: the gain
   objective buys resistance reduction between strong-amplitude pairs.

2. SATURATION (second order, present even at equal amplitudes).
   exp(-x) is convex, so by Jensen, at FIXED total resistance the
   gain objective prefers an UNEVEN resistance profile: a pair whose
   variance is already ~2 rad^2 contributes ~nothing regardless, so
   further resistance on a written-off pair is free for gain but
   costly for MSE. Gain concentrates budget on saveable pairs; MSE
   equalizes. Divergence requires the curvature to matter:
   r2 * spread(R) = O(1). For r2*R_ik << 1 for all pairs, the two
   objectives coincide to first order at equal amplitudes - the
   regime condition.

Predictions stated before measurement (discipline):
  P1: the exact-gain formula matches Monte-Carlo draws to MC error.
  P2: with equal amplitudes and small r2, argmax-gain == argmin-MSE
      almost always; divergence frequency grows with r2.
  P3: mismatch cost grows with amplitude heterogeneity at fixed r2.
  P4: at tight budgets and strong heterogeneity the gain-optimal
      edge set is sometimes DISCONNECTED (abandons a weak node) -
      infinite-MSE territory, feeding direction C.

Run: python dirB_objective_gap.py    (from this directory)
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase_sync_idea"))

from openloop_graph_theory import (  # noqa: E402
    blue_theta_covariance,
    component_count,
    effective_resistance,
    pair_variance,
)

RESULTS = {}


def components(num_nodes, edges):
    parent = list(range(num_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        parent[find(i)] = find(j)
    return [find(i) for i in range(num_nodes)]


def pair_resistances(num_nodes, edges):
    """Unit-conductance effective resistance; inf across components."""

    labels = components(num_nodes, edges)
    resistance = effective_resistance(
        num_nodes, list(edges), [1.0] * len(edges)
    )
    for i in range(num_nodes):
        for k in range(num_nodes):
            if labels[i] != labels[k]:
                resistance[i, k] = np.inf
    return resistance


def gain(amplitudes, resistance, r2):
    weight = np.exp(-r2 * np.where(np.isinf(resistance), np.inf, resistance) / 2.0)
    weight[np.isinf(resistance)] = 0.0
    total = amplitudes.sum() ** 2
    return float(amplitudes @ weight @ amplitudes) / total


def kirchhoff(resistance):
    upper = resistance[np.triu_indices_from(resistance, 1)]
    return float(np.sum(upper))  # inf if disconnected


def enumerate_subsets(num_nodes, budget):
    candidates = list(itertools.combinations(range(num_nodes), 2))
    for subset in itertools.combinations(candidates, budget):
        yield subset


def best_topologies(num_nodes, budget, amplitudes, r2):
    """Exhaustive argmax-gain (any subset) and argmin-MSE (connected)."""

    best_gain, best_gain_edges = -1.0, None
    best_mse, best_mse_edges = np.inf, None
    for subset in enumerate_subsets(num_nodes, budget):
        resistance = pair_resistances(num_nodes, subset)
        g = gain(amplitudes, resistance, r2)
        if g > best_gain:
            best_gain, best_gain_edges = g, subset
        m = kirchhoff(resistance)
        if m < best_mse:
            best_mse, best_mse_edges = m, subset
    mse_gain = gain(
        amplitudes, pair_resistances(num_nodes, best_mse_edges), r2
    ) if best_mse_edges else 0.0
    return {
        "gain_edges": best_gain_edges,
        "gain_value": best_gain,
        "mse_edges": best_mse_edges,
        "gain_of_mse_choice": mse_gain,
        "mismatch_cost": best_gain - mse_gain,
        # differ = the MSE choice actually LOSES gain. Set-inequality
        # alone overcounts: symmetric ties (e.g. all stars share one
        # Kirchhoff index) pick different representatives at zero cost.
        "differ": (best_gain - mse_gain) > 1e-9,
        "edge_sets_differ": set(best_gain_edges) != set(best_mse_edges),
        "gain_choice_disconnected": (
            component_count(num_nodes, list(best_gain_edges)) > 1
        ),
    }


def experiment_formula_validation(seed=0, draws=1_000_000):
    """P1: exact-gain formula vs Monte-Carlo Gaussian draws."""

    rng = np.random.default_rng(seed)
    n, r2 = 6, 0.3
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 3), (1, 4)]
    amplitudes = rng.uniform(0.3, 2.0, n)
    cov = blue_theta_covariance(n, [], edges, 1.0, r2)
    draws_z = rng.multivariate_normal(np.zeros(n - 1), cov, size=draws)
    theta = np.concatenate([np.zeros((draws, 1)), draws_z], axis=1)
    field = (amplitudes * np.exp(1j * theta)).sum(axis=1)
    mc = float(np.mean(np.abs(field) ** 2)) / amplitudes.sum() ** 2
    resistance = pair_resistances(n, edges)
    formula = gain(amplitudes, resistance, r2)
    consistency = max(
        abs(r2 * resistance[i, k] - pair_variance(cov, i, k))
        for i in range(n) for k in range(i + 1, n)
    )
    RESULTS["E1"] = {
        "mc": mc, "formula": formula,
        "relative_error": abs(mc - formula) / formula,
        "resistance_vs_blue_max_gap": consistency,
    }


def experiment_divergence(seeds=(0, 1, 2)):
    """P2/P3/P4: exhaustive search for objective divergence, N=6."""

    n = 6
    table = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        base = rng.standard_normal(n)
        for budget in (5, 6, 7):
            for r2 in (0.05, 0.5):
                for hetero, label in ((0.0, "equal"), (0.8, "hetero")):
                    amplitudes = np.exp(hetero * base)
                    out = best_topologies(n, budget, amplitudes, r2)
                    table.append({
                        "seed": seed, "budget": budget, "r2": r2,
                        "amps": label,
                        "differ": out["differ"],
                        "mismatch_cost": out["mismatch_cost"],
                        "gain_value": out["gain_value"],
                        "gain_disconnected": out["gain_choice_disconnected"],
                    })
    RESULTS["E2"] = table


def experiment_heterogeneity_sweep(seeds=(0, 1, 2)):
    """P3: mismatch cost vs amplitude heterogeneity, r2 fixed."""

    n, budget, r2 = 6, 5, 0.2
    sweep = []
    for h in (0.0, 0.2, 0.4, 0.6, 0.8, 1.2, 1.6):
        costs, differs = [], 0
        for seed in seeds:
            rng = np.random.default_rng(seed)
            amplitudes = np.exp(h * rng.standard_normal(n))
            out = best_topologies(n, budget, amplitudes, r2)
            costs.append(out["mismatch_cost"])
            differs += int(out["differ"])
        sweep.append({
            "h": h, "mean_cost": float(np.mean(costs)),
            "max_cost": float(np.max(costs)), "differ_count": differs,
        })
    RESULTS["E3"] = sweep


def experiment_residual_sweep(seeds=(0, 1, 2)):
    """P2 saturation branch: equal amplitudes, sweep r2."""

    n, budget = 6, 5
    sweep = []
    amplitudes = np.ones(n)
    for r2 in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
        costs, differs = [], 0
        for seed in seeds:  # seed only sets tie-breaking via enumeration order; kept for symmetry
            out = best_topologies(n, budget, amplitudes, r2)
            costs.append(out["mismatch_cost"])
            differs += int(out["differ"])
        sweep.append({
            "r2": r2, "mean_cost": float(np.mean(costs)),
            "differ_count": differs,
        })
    RESULTS["E4"] = sweep


def experiment_larger_instance():
    """One N=7, budget 6 heterogeneous instance (116k subsets)."""

    n, budget, r2 = 7, 6, 0.4
    rng = np.random.default_rng(0)
    amplitudes = np.exp(0.9 * rng.standard_normal(n))
    out = best_topologies(n, budget, amplitudes, r2)
    RESULTS["E5"] = {
        "differ": out["differ"],
        "mismatch_cost": out["mismatch_cost"],
        "gain_value": out["gain_value"],
        "gain_edges": list(map(list, out["gain_edges"])),
        "mse_edges": list(map(list, out["mse_edges"])),
        "gain_disconnected": out["gain_choice_disconnected"],
        "amplitudes": amplitudes.round(3).tolist(),
    }


def make_figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    e3 = RESULTS["E3"]
    axes[0].plot([r["h"] for r in e3], [100 * r["mean_cost"] for r in e3],
                 "o-", label="mean over seeds")
    axes[0].plot([r["h"] for r in e3], [100 * r["max_cost"] for r in e3],
                 "s--", label="max over seeds")
    axes[0].set_xlabel("amplitude heterogeneity h (a = exp(h z))")
    axes[0].set_ylabel("gain lost by MSE-optimal topology (points)")
    axes[0].set_title("Mismatch cost vs amplitude heterogeneity")
    axes[0].legend()
    e4 = RESULTS["E4"]
    axes[1].plot([r["r2"] for r in e4], [100 * r["mean_cost"] for r in e4],
                 "o-")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("per-measurement phase variance r2 (rad^2)")
    axes[1].set_ylabel("gain lost by MSE-optimal topology (points)")
    axes[1].set_title("Equal amplitudes: saturation mechanism")
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "figures" / "dirB_mismatch_cost.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200)
    return out


def main():
    print("P1-P4 predictions are in the module docstring (written first).")
    experiment_formula_validation()
    print(f"E1 formula vs MC: {RESULTS['E1']}")
    experiment_divergence()
    differ = [row for row in RESULTS["E2"] if row["differ"]]
    print(f"E2: {len(differ)}/{len(RESULTS['E2'])} configs diverge")
    for row in differ:
        print("   ", row)
    experiment_heterogeneity_sweep()
    print("E3 heterogeneity sweep:")
    for row in RESULTS["E3"]:
        print("   ", row)
    experiment_residual_sweep()
    print("E4 equal-amplitude residual sweep:")
    for row in RESULTS["E4"]:
        print("   ", row)
    experiment_larger_instance()
    print(f"E5 N=7 instance: {RESULTS['E5']}")
    path = Path(__file__).resolve().parent / "dirB_results.json"
    path.write_text(json.dumps(RESULTS, indent=1, default=str))
    figure = make_figure()
    print(f"figure: {figure}")


if __name__ == "__main__":
    main()
