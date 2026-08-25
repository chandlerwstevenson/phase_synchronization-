"""Strengthen the fixed-N fixed-airtime ranking reversal: run seeds
3-9 for {star, mst} x {symmetric, directed} at budgets B2 and B3
(N=8), extending the existing seeds 0-2 in the caches. Uses the
dirA2_threelaw machinery unchanged. Results appended to
dirA5_cache.json; prints the combined per-seed table and win counts.
"""

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "../phase_sync_idea")

from dirA_selection import make_positions
from dirA2_threelaw import cell, get_sigma2, topology

CACHE = Path("dirA5_cache.json")
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

positions = make_positions("uniform")
sigma2 = get_sigma2(positions)

for budget in (2, 3):
    for name in ("star", "mst"):
        edges = topology(name, positions, sigma2)
        for law in ("symmetric", "directed"):
            for seed in range(3, 10):
                key = f"rev|{name}|{law}|B{budget}|s{seed}"
                if key not in cache:
                    cell(cache, key, positions, edges, law, budget, seed)
                    CACHE.write_text(json.dumps(cache, indent=1))
                    print(key, round(cache[key]["gain"] * 100, 1), flush=True)

# combined report with the original seeds 0-2
old = {}
for f in ("dirA2_cache.json", "dirA3_cache.json"):
    old.update(json.loads(Path(f).read_text()))

print("\n=== combined (seeds 0-9) ===")
for budget in (2, 3):
    print(f"--- B{budget} ({budget * 4.276:.1f}% airtime) ---")
    values = {}
    for name in ("star", "mst"):
        for law in ("symmetric", "directed"):
            vals = []
            for s in range(10):
                for src in (old, cache):
                    for k, v in src.items():
                        p = k.split("|")
                        if (len(p) == 5 and p[1] == name and p[2] == law
                                and p[3] == f"B{budget}" and p[4] == f"s{s}"):
                            vals.append((s, v["gain"] * 100))
            vals = dict(vals)
            values[(name, law)] = vals
            mean = statistics.mean(vals.values())
            std = statistics.stdev(vals.values())
            print(f"  {name:5s} {law:11s} {mean:5.1f} ± {std:4.1f} (n={len(vals)})")
    for law in ("symmetric", "directed"):
        star, mst = values[("star", law)], values[("mst", law)]
        seeds = sorted(set(star) & set(mst))
        wins = sum(star[s] > mst[s] for s in seeds)
        print(f"  {law}: star beats mst in {wins}/{len(seeds)} seeds")
