"""Render the two headline figures the paper tables imply but the
mechanism plots don't show: the topology-ranking inversion across
protocols (N=8) and the N=16 collapse of bidirectional protocols.
Data: dirA2_cache.json (the 118-cell fork campaign) - no re-simulation.
Plain default matplotlib, no in-axes annotations.
"""

import json
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

cache = json.load(open("dirA2_cache.json"))
LAWS = ["symmetric", "alternating", "directed"]


def cells(prefix, topo, law):
    vals = [
        v["gain"] * 100.0
        for k, v in cache.items()
        if k.startswith(f"{prefix}|{topo}|{law}|")
    ]
    return (
        statistics.mean(vals),
        statistics.stdev(vals) if len(vals) > 1 else 0.0,
    )


# --- Figure 1: inversion at N=8 (degree campaign: star / deg3tree / mst)
topos = ["star", "deg3tree", "mst"]
labels = ["star (hub degree 7)", "degree-3 tree", "min-variance tree"]
fig, ax = plt.subplots(figsize=(7.5, 4.5))
width = 0.25
for i, law in enumerate(LAWS):
    means = [cells("deg", t, law)[0] for t in topos]
    errs = [cells("deg", t, law)[1] for t in topos]
    ax.bar(
        [x + (i - 1) * width for x in range(len(topos))],
        means,
        width,
        yerr=errs,
        capsize=3,
        label=law,
    )
ax.set_xticks(range(len(topos)))
ax.set_xticklabels(labels)
ax.set_ylabel("coherent gain (%)")
ax.legend(title="protocol")
ax.set_ylim(0, 105)
fig.tight_layout()
fig.savefig("paper_figures/fig1_inversion_n8.png", dpi=200)
plt.close(fig)

# --- Figure 2: N=16 scaling (scl campaign: mst16 / star16 / ring16)
topos16 = ["mst16", "star16", "ring16"]
labels16 = ["min-variance tree", "star (hub degree 15)", "ring"]
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for i, law in enumerate(LAWS):
    means, errs, xs = [], [], []
    for j, t in enumerate(topos16):
        vals = [
            v["gain"] * 100.0
            for k, v in cache.items()
            if k.startswith(f"scl|{t}|{law}|")
        ]
        if vals:
            xs.append(j + (i - 1) * width)
            means.append(statistics.mean(vals))
            errs.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
    ax.bar(xs, means, width, yerr=errs, capsize=3, label=law)
ax.set_xticks(range(len(topos16)))
ax.set_xticklabels(labels16)
ax.set_ylabel("coherent gain (%)")
ax.legend(title="protocol")
ax.set_ylim(0, 105)
fig.tight_layout()
fig.savefig("paper_figures/fig2_scaling_n16.png", dpi=200)
plt.close(fig)

print("wrote paper_figures/fig1_inversion_n8.png and fig2_scaling_n16.png")
for t in topos:
    print(t, {law: round(cells("deg", t, law)[0], 1) for law in LAWS})
