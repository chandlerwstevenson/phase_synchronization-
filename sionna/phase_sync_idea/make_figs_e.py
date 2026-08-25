"""Figures for experiment E (plain default matplotlib, no in-axes
annotations)."""

from __future__ import annotations

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def fig_e1():
    r = json.load(open("ablation_results.json"))

    def mean_rms(key):
        return float(np.mean([x["worst_rms_mrad"] for x in r[key]]))

    def mean_absresid(key):
        return float(np.mean([x["tail_abs_mrad"][0] for x in r[key]]))

    labels = [
        "full\n(baseline)",
        "no channel\nstate",
        "no anchors\n(static)",
        "no branch check\n(adverse acq.)",
        "preamble\nobservations",
        "moving env.\n0.1 m/s, K=40",
        "moving env.\n0.1 m/s, K=10",
    ]
    values = [
        mean_rms("full"),
        mean_rms("no_decomposition"),
        mean_rms("no_anchors"),
        float(np.mean([x["tail_abs_mrad"] for x in r["adverse_check_off"]])),
        mean_rms("zc_waveform"),
        mean_rms("dynamic_k40"),
        mean_rms("dynamic_k10"),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)
    ax.bar(range(len(labels)), values)
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("worst-station residual (mrad, log scale)")
    ax.set_title("Component ablation: residual with each component removed")
    ax.axhline(values[0], linestyle="--", linewidth=1.0, color="gray",
               label="full architecture baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/fig_e1_ablation.png", bbox_inches="tight")
    plt.close(fig)


def fig_e2():
    r = json.load(open("capture_model_results.json"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=200, sharey=True)
    for axis, snr_db in zip(axes, [10.0, 20.0, 30.0]):
        rows = [x for x in r["zc"] if x["snr_db"] == snr_db]
        lengths = [x["length"] for x in rows]
        measured = [1e3 * math.sqrt(max(x["measured_var"], 0)) for x in rows]
        thermal = [1e3 * math.sqrt(x["thermal_var"]) for x in rows]
        walk = [1e3 * math.sqrt(x["walk_var"]) for x in rows]
        total = [
            1e3 * math.sqrt(x["thermal_var"] + x["walk_var"]) for x in rows
        ]
        axis.plot(lengths, measured, "o-", label="measured (preamble)")
        axis.plot(lengths, total, "--", label="predicted thermal + walk")
        axis.plot(lengths, thermal, ":", label="thermal + white-PM term")
        axis.plot(lengths, walk, ":", label="oscillator-walk term")
        ofdm = [x for x in r["ofdm"] if x["snr_db"] == snr_db][0]
        axis.plot(
            [ofdm["length"]],
            [1e3 * math.sqrt(max(ofdm["measured_var"], 0))],
            "s", label="measured (OFDM burst)",
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("pilot long-sequence length (samples)")
        axis.set_title(f"link SNR {snr_db:.0f} dB")
    axes[0].set_ylabel("per-observation phase error (mrad)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Per-observation noise vs capture length: measured and "
                 "predicted components")
    fig.tight_layout()
    fig.savefig("figures/fig_e2_capture_model.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_e1()
    fig_e2()
    print("saved figures/fig_e1_ablation.png, figures/fig_e2_capture_model.png")
