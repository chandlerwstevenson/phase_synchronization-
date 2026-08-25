"""Cycle-mechanism figure: chain vs ring (chain + one chord) under
the two bidirectional protocols, 10 seeds each (cyc| cells including
the 2026-08-25 overnight seed extension). Y = coherent gain, error
bars = standard error. Plain default matplotlib, no in-axes
annotations. Data: dirA2_cache.json. Replaces the original 3-seed
dirA2_cycle.png.
"""

import json
import math
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

cache = json.load(open("dirA2_cache.json"))


def stats(topo, law):
    vals = [
        v["gain"] * 100.0
        for k, v in cache.items()
        if k.startswith(f"cyc|{topo}|{law}|")
    ]
    return statistics.mean(vals), statistics.stdev(vals) / math.sqrt(len(vals)), len(vals)


LAWS = [("symmetric", "simultaneous"), ("alternating", "sequential")]
fig, ax = plt.subplots(figsize=(7, 4.5))
width = 0.3
for i, (law, label) in enumerate(LAWS):
    means, sems = zip(*[stats(t, law)[:2] for t in ("chain", "ring")])
    ax.bar(
        [x + (i - 0.5) * width for x in range(2)],
        means, width, yerr=sems, capsize=4, label=label,
    )
ax.set_xticks(range(2))
ax.set_xticklabels(["chain (no cycle)", "ring (chain + one chord)"])
ax.set_ylabel("coherent gain (%)")
ax.legend(title="protocol")
ax.set_ylim(0, 100)
fig.tight_layout()
fig.savefig("paper_figures/dirA2_cycle.png", dpi=200)
for t in ("chain", "ring"):
    for law, _ in LAWS:
        m, s, n = stats(t, law)
        print(f"{t:6s} {law:12s} {m:5.1f} ± {s:.1f} (n={n})")
print("wrote paper_figures/dirA2_cycle.png")
