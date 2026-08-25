"""Topology-effect figure for a topology-focused presentation: at a
FIXED synchronization protocol (the standard simultaneous convention),
the choice of measurement topology alone spans coherent gain from
~17% to ~84% at identical airtime, N=8, seeds 0-2. Bars sorted by
gain; colors group the mechanism that limits each graph: low-degree
acyclic (works), high-degree hub (Jacobi divergence), cyclic (winding
states). Y = coherent gain, error bars = standard error. Plain
matplotlib. Data: dirA2_cache.json (deg + cyc campaigns).
"""

import json
import math
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

cache = json.load(open("dirA2_cache.json"))

TOPOS = [
    ("mst", "min-variance\ntree", "tree"),
    ("chain", "chain", "tree"),
    ("deg3tree", "degree-3\ntree", "tree"),
    ("star", "star\n(hub degree 7)", "hub"),
    ("complete", "complete\ngraph", "cycles"),
    ("mst2c", "tree + 2\nchords", "cycles"),
    ("ring", "ring", "cycles"),
]
GROUP_COLOR = {"tree": "C0", "hub": "C1", "cycles": "C3"}
GROUP_LABEL = {
    "tree": "acyclic, low degree",
    "hub": "high-degree hub",
    "cycles": "contains cycles",
}


def stats(topo):
    vals = [
        v["gain"] * 100.0
        for k, v in cache.items()
        if k.split("|")[:3:2] == [k.split("|")[0], "symmetric"]
        and k.split("|")[1] == topo
        and k.split("|")[0] in ("deg", "cyc")
    ]
    return statistics.mean(vals), statistics.stdev(vals) / math.sqrt(len(vals))


rows = sorted(
    [(name, label, group, *stats(name)) for name, label, group in TOPOS],
    key=lambda r: -r[3],
)

fig, ax = plt.subplots(figsize=(9, 5))
seen = set()
for i, (name, label, group, mean, sem) in enumerate(rows):
    ax.bar(
        i, mean, 0.65, yerr=sem, capsize=4,
        color=GROUP_COLOR[group],
        label=GROUP_LABEL[group] if group not in seen else None,
    )
    seen.add(group)
ax.set_xticks(range(len(rows)))
ax.set_xticklabels([r[1] for r in rows])
ax.set_ylabel("coherent gain (%)")
ax.legend(title="graph property")
ax.set_ylim(0, 100)
fig.tight_layout()
fig.savefig("paper_figures/fig8_topology_effect.png", dpi=200)
for r in rows:
    print(f"{r[0]:10s} {r[3]:5.1f} ± {r[4]:.1f}")
print("wrote paper_figures/fig8_topology_effect.png")
