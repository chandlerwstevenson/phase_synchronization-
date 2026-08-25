"""Redesigned degree-mechanism figure: the star (hub with 7 links)
under four update rules. The story in one group of bars: simultaneous
updates collapse the hub; turn-taking only half-fixes it (it dilutes
each link's service rate); damping the simultaneous updates or
directing them fully cures it. Y = coherent gain, error bars =
standard error (3 seeds). Overwrites figures/dirA2_degree.png and
paper_figures/dirA2_degree.png. Plain matplotlib.
"""

import json
import math
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

cache = json.load(open("dirA2_cache.json"))

VARIANTS = [
    ("symmetric", "simultaneous"),
    ("alternating", "turn-taking"),
    ("symmetric-dw", "simultaneous\n+ damping"),
    ("directed", "directed"),
]

fig, ax = plt.subplots(figsize=(7.5, 5))
for i, (key, label) in enumerate(VARIANTS):
    vals = [
        v["gain"] * 100.0
        for k, v in cache.items()
        if k.startswith(f"deg|star|{key}|")
    ]
    mean = statistics.mean(vals)
    sem = statistics.stdev(vals) / math.sqrt(len(vals))
    ax.bar(i, mean, 0.62, yerr=sem, capsize=4)
    print(label.replace("\n", " "), round(mean, 1), "±", round(sem, 1),
          f"(n={len(vals)})")
ax.set_xticks(range(len(VARIANTS)))
ax.set_xticklabels([label for _, label in VARIANTS])
ax.set_ylabel("coherent gain (%)")
ax.set_ylim(0, 105)
fig.tight_layout()
for path in ("figures/dirA2_degree.png", "paper_figures/dirA2_degree.png"):
    fig.savefig(path, dpi=200)
print("wrote dirA2_degree.png (both locations)")
