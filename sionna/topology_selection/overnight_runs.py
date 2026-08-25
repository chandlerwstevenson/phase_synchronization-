"""Overnight hardening runs for the slide backing document.

PRE-REGISTERED PREDICTIONS (stated before running):
  A. Clustered-geometry reversal replication (the important one): the
     fixed-conditions reversal was found post-hoc on the uniform
     geometry. Prediction: on the CLUSTERED geometry at N=8, B=2
     (8.6% airtime), 10 seeds, the same double reversal appears -
     tree > star under simultaneous, star > tree under directed -
     because the mechanisms (hub Jacobi divergence; depth-vs-cadence
     under sparse service) are geometry-independent. If it does NOT
     replicate, that is a finding and goes in the document verbatim.
  B. N=16 seed extension (star16/mst16 x 3 protocols, seeds 2-4,
     total 18 new cells): prediction - means move little; directed
     star stays ~99, tree ~89; bidirectional stays collapsed ~14-20.
  C. Cycle-cell seed extension (chain, ring x simultaneous,
     alternating, seeds 3-9): prediction - chain stays ~70s with wide
     spread, ring stays far below chain under both protocols.

All cells go through the merge-safe cell() into dirA2-style caches
(prefixes revc| for clustered, scl| for N=16, cyc| extended seeds).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "../phase_sync_idea")

import numpy as np

from dirA_selection import make_positions
from dirA2_threelaw import cell, get_sigma2, load_cache, topology
import openloop_topology_study as topo

cache = load_cache()

# ---- A: clustered-geometry reversal replication -------------------
print("=== A: clustered-geometry reversal (B2, seeds 0-9) ===", flush=True)
pos_c = make_positions("clustered")
sig_c = get_sigma2(pos_c)
for name in ("star", "mst"):
    edges = topology(name, pos_c, sig_c)
    for law in ("symmetric", "directed"):
        for seed in range(10):
            key = f"revc|{name}|{law}|B2|s{seed}"
            if key not in cache:
                cell(cache, key, pos_c, edges, law, 2, seed)
                print(key, round(cache[key]["gain"] * 100, 1), flush=True)

# ---- B: N=16 seed extension --------------------------------------
print("=== B: N=16 seeds 2-4 ===", flush=True)
n = 16
pos16 = topo.place_stations(n, 500.0, 7)
dists = {
    (i, j): float(np.linalg.norm(pos16[i] - pos16[j]))
    for i in range(n) for j in range(i + 1, n)
}
parent = list(range(n))


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


mst16 = []
for (i, j) in sorted(dists, key=dists.get):
    ri, rj = find(i), find(j)
    if ri != rj:
        parent[ri] = rj
        mst16.append((i, j))
hub = min(
    range(n),
    key=lambda h: sum(dists[(min(h, j), max(h, j))] for j in range(n) if j != h),
)
star16 = [(min(hub, j), max(hub, j)) for j in range(n) if j != hub]
for name, edges in (("mst16", mst16), ("star16", star16)):
    for law in ("symmetric", "alternating", "directed"):
        for seed in (2, 3, 4):
            key = f"scl|{name}|{law}|B15|s{seed}"
            if key not in cache:
                cell(cache, key, pos16, edges, law, 15, seed, n=n)
                print(key, round(cache[key]["gain"] * 100, 1), flush=True)

# ---- C: cycle-cell seed extension --------------------------------
print("=== C: cycle cells seeds 3-9 ===", flush=True)
pos_u = make_positions("uniform")
sig_u = get_sigma2(pos_u)
for name in ("chain", "ring"):
    edges = topology(name, pos_u, sig_u)
    for law in ("symmetric", "alternating"):
        for seed in range(3, 10):
            key = f"cyc|{name}|{law}|Bnone|s{seed}"
            if key not in cache:
                cell(cache, key, pos_u, edges, law, None, seed)
                print(key, round(cache[key]["gain"] * 100, 1), flush=True)

# ---- report -------------------------------------------------------
import math
import statistics

final = load_cache()


def summarize(prefix, topo_name, law):
    vals = [
        v["gain"] * 100
        for k, v in final.items()
        if k.startswith(f"{prefix}|{topo_name}|{law}|")
    ]
    return (
        round(statistics.mean(vals), 1),
        round(statistics.stdev(vals), 1) if len(vals) > 1 else 0,
        len(vals),
    )


print("\n=== FINAL SUMMARY ===")
print("A. clustered reversal:")
values = {}
for name in ("star", "mst"):
    for law in ("symmetric", "directed"):
        values[(name, law)] = {
            k.split("|")[4]: v["gain"] * 100
            for k, v in final.items()
            if k.startswith(f"revc|{name}|{law}|")
        }
        print(" ", name, law, summarize("revc", name, law))
for law in ("symmetric", "directed"):
    star, mst = values[("star", law)], values[("mst", law)]
    common = sorted(set(star) & set(mst))
    wins = sum(star[s] > mst[s] for s in common)
    nn = len(common)
    tail = sum(
        math.comb(nn, k) for k in range(max(wins, nn - wins), nn + 1)
    ) / 2 ** nn
    print(f"  {law}: star wins {wins}/{nn} (two-sided p={min(1.0, 2*tail):.4f})")
print("B. N=16 (now 5 seeds):")
for name in ("star16", "mst16"):
    for law in ("symmetric", "alternating", "directed"):
        print(" ", name, law, summarize("scl", name, law))
print("C. cycles (now 10 seeds):")
for name in ("chain", "ring"):
    for law in ("symmetric", "alternating"):
        print(" ", name, law, summarize("cyc", name, law))
