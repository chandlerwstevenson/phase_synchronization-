"""Figure 7 (final): the ranking reversal as four bars. Y axis =
coherent gain (%), nothing else. Two protocol groups; within each,
the min-variance tree and the star. Error bars = standard error of
the mean (n = 17 seeds) - the uncertainty of the compared means, not
the population spread. Under the simultaneous protocol the tree bar
is higher; under the directed protocol the star bar is higher.
Win counts and sign-test p-values are printed for the caption.
Fixed N=8, fixed 8.6% sync airtime. Plain matplotlib.
"""

import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

cache = {}
for f in ("dirA2_cache.json", "dirA3_cache.json", "dirA5_cache.json"):
    if Path(f).exists():
        cache.update(json.loads(Path(f).read_text()))


def seed_vals(topo, law):
    # Uniform-geometry campaigns only. The revc| cells are the
    # clustered-geometry replication and must not mix in here.
    out = {}
    for k, v in cache.items():
        p = k.split("|")
        if (
            len(p) == 5 and p[0] in ("rev", "cad", "par")
            and p[1] == topo and p[2] == law and p[3] == "B2"
        ):
            out[p[4]] = v["gain"] * 100.0
    return out


fig, ax = plt.subplots(figsize=(7.5, 5))
width = 0.32
for j, (topo, label) in enumerate(
    [("mst", "min-variance tree"), ("star", "star")]
):
    means, sems = [], []
    for law in ("symmetric", "directed"):
        star, mst = seed_vals("star", law), seed_vals("mst", law)
        common = sorted(set(star) & set(mst))
        vals = [seed_vals(topo, law)[s] for s in common]
        means.append(statistics.mean(vals))
        sems.append(statistics.stdev(vals) / math.sqrt(len(vals)))
    ax.bar(
        [x + (j - 0.5) * width for x in range(2)],
        means, width, yerr=sems, capsize=4, label=label,
    )
ax.set_xticks(range(2))
ax.set_xticklabels(["simultaneous protocol", "directed protocol"])
ax.set_ylabel("coherent gain (%)")
ax.legend(title="topology")
ax.set_ylim(0, 100)
fig.tight_layout()
fig.savefig("paper_figures/fig7_reversal.png", dpi=200)

for law in ("symmetric", "directed"):
    star, mst = seed_vals("star", law), seed_vals("mst", law)
    common = sorted(set(star) & set(mst))
    wins = sum(star[s] > mst[s] for s in common)
    n = len(common)
    tail = sum(
        math.comb(n, k) for k in range(max(wins, n - wins), n + 1)
    ) / 2 ** n
    print(f"{law}: star wins {wins}/{n}, p={min(1.0, 2 * tail):.4f}")
print("wrote paper_figures/fig7_reversal.png")
