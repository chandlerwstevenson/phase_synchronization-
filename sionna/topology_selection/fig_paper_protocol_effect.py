"""Figure 6 (redesigned): the protocol, not the topology, determines
coherent gain. Lines = topologies, x = protocol. The visual claim:
the topology lines scatter under bidirectional protocols and
CONVERGE under the directed protocol - choose the protocol correctly
and the topology choice stops mattering; choose it wrong and no
topology saves you (N=16). Data: dirA2_cache.json. Plain matplotlib.
"""

import json
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

cache = json.load(open("dirA2_cache.json"))
LAWS = ["symmetric", "alternating", "directed"]
XLABELS = ["simultaneous", "sequential", "directed"]


def mean_std(prefix, topo, law):
    v = [
        val["gain"] * 100.0
        for k, val in cache.items()
        if k.startswith(f"{prefix}|{topo}|{law}|")
    ]
    return (
        statistics.mean(v),
        statistics.stdev(v) if len(v) > 1 else 0.0,
    )


fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)

# Left: N=8, three topologies
ax = axes[0]
for topo, label, marker in [
    ("star", "star", "o"),
    ("deg3tree", "degree-3 tree", "s"),
    ("mst", "min-variance tree", "^"),
]:
    means, errs = zip(*[mean_std("deg", topo, law) for law in LAWS])
    ax.errorbar(
        range(3), means, yerr=errs, marker=marker, capsize=3,
        linewidth=2, markersize=7, label=label,
    )
ax.set_xticks(range(3))
ax.set_xticklabels(XLABELS)
ax.set_ylabel("coherent gain (%)")
ax.set_title("N=8")
ax.legend(title="topology", loc="upper left")
ax.set_ylim(0, 105)

# Right: N=16, star and mst
ax = axes[1]
for topo, label, marker in [
    ("star16", "star", "o"),
    ("mst16", "min-variance tree", "^"),
]:
    means, errs = zip(*[mean_std("scl", topo, law) for law in LAWS])
    ax.errorbar(
        range(3), means, yerr=errs, marker=marker, capsize=3,
        linewidth=2, markersize=7, label=label,
    )
ax.set_xticks(range(3))
ax.set_xticklabels(XLABELS)
ax.set_title("N=16")
ax.legend(title="topology", loc="upper left")

fig.tight_layout()
fig.savefig("paper_figures/fig6_protocol_effect.png", dpi=200)
print("wrote fig6")
