"""Figures for experiments B and D (plain default matplotlib, no
in-axes annotations). Reads experiment_b_d_cache.json."""

from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_b_d import (
    B_CADENCES,
    B_SEEDS,
    B_SPEEDS,
    D_RATES,
    D_SEEDS,
    D_SPEEDS,
    boundary_coherence_time_s,
    coherence_time_s,
    load_cache,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")


def mean_std(values):
    values = [v for v in values if v == v]
    mean = sum(values) / len(values)
    std = (
        math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        if len(values) > 1 else 0.0
    )
    return mean, std


def fig_b(cache):
    figure, axis = plt.subplots(figsize=(7.0, 4.6))
    for cadence in B_CADENCES:
        xs, ys, errs = [], [], []
        static_level = None
        for speed in B_SPEEDS:
            rms, err = mean_std([
                cache[f"B|{speed}|{cadence}|{seed}"]["rms_mrad"]
                for seed in B_SEEDS
            ])
            if speed == 0.0:
                static_level = rms
            else:
                xs.append(coherence_time_s(speed))
                ys.append(rms)
                errs.append(err)
        line = axis.errorbar(
            xs, ys, yerr=errs, marker="o", capsize=3,
            label=f"anchors every {cadence} intervals",
        )
        color = line.lines[0].get_color()
        axis.axhline(
            static_level, linestyle=":", color=color,
            label=f"static environment level (K={cadence})",
        )
        axis.axvline(
            boundary_coherence_time_s(cadence, 0.05), linestyle="--",
            color=color,
            label=f"theory boundary (K={cadence})",
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("environment coherence time (s)")
    axis.set_ylabel("residual phase error (mrad, rms)")
    axis.set_title("Residual phase error vs environment coherence time")
    axis.legend(fontsize=8)
    figure.tight_layout()
    path = os.path.join(FIGDIR, "expB_residual_vs_coherence_time.png")
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def fig_d(cache):
    figure, axis = plt.subplots(figsize=(7.0, 4.6))
    for speed in D_SPEEDS:
        ys, errs = [], []
        for n_obs in D_RATES:
            rms, err = mean_std([
                cache[f"D|{speed}|{n_obs}|{seed}"]["rms_mrad"]
                for seed in D_SEEDS
            ])
            ys.append(rms)
            errs.append(err)
        label = "static" if speed == 0.0 else f"{speed:g} m/s"
        axis.errorbar(
            D_RATES, ys, yerr=errs, marker="o", capsize=3, label=label
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks(D_RATES)
    axis.set_xticklabels([str(n) for n in D_RATES])
    axis.set_xlabel("free observations per sync interval")
    axis.set_ylabel("residual phase error (mrad, rms)")
    axis.set_title(
        "Residual phase error vs observation rate, by environment motion"
    )
    axis.legend(title="environment motion", fontsize=8)
    figure.tight_layout()
    path = os.path.join(FIGDIR, "expD_residual_vs_observation_rate.png")
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    cache = load_cache()
    print(fig_b(cache))
    print(fig_d(cache))


if __name__ == "__main__":
    main()
