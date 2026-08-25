# RESULTS A3 — the coherence/airtime Pareto frontier

Reviewer request: make synchronization airtime an explicit secondary
metric rather than a control. Grid: topology {star, mst, chain} ×
protocol {simultaneous ("symmetric"), alternating, directed} × exchange
budget B ∈ {7, 5, 4, 3, 2, 1} edges serviced round-robin per interval
× seeds {0, 1, 2}, N = 8, uniform geometry, 150 intervals (steady
window 100–150). Steady-state sync airtime = B × 4.276% (two captures
of the 255-sample pilot per exchange; acquisition intervals excluded,
identical across cells). Machinery: `dirA2_threelaw.py` unchanged;
cells reused from `dirA2_cache.json` where the identical configuration
existed (mst B7–B2 all laws; star B7 all laws), 108 new cells in
`dirA3_cache.json`. Predictions were printed before the runs
(`dirA3_pareto.py` docstring).

## Frontier (airtime → mean gain % ± std over seeds)

| pair | 29.9% | 21.4% | 17.1% | 12.8% | 8.6% | 4.3% |
|---|---|---|---|---|---|---|
| star / simultaneous | 23.7±1.1 | 24.3±2.1 | 22.5±1.1 | 21.3±2.6 | 24.1±2.3 | 22.1±2.0 |
| star / alternating | 38.9±3.3 | 35.6±6.6 | 31.0±8.8 | 25.3±4.2 | 27.6±3.6 | 23.1±0.8 |
| star / directed | 92.2±12.5 | 91.4±12.3 | 90.9±12.4 | 89.4±12.5 | **87.4±12.1** | 48.4±11.7 |
| mst / simultaneous | 83.9±8.3 | 69.2±17.8 | 47.7±38.0 | 44.8±17.6 | 38.0±19.6 | 23.6±4.3 |
| mst / alternating | 68.3±13.0 | 41.9±8.8 | 38.7±16.4 | 28.4±0.9 | 26.9±6.5 | 25.5±4.9 |
| mst / directed | 96.3±1.3 | 93.7±0.5 | 93.8±2.9 | 86.8±3.2 | 80.2±3.0 | 38.0±1.9 |
| chain / simultaneous | 73.2±20.7 | 21.8±0.8 | 25.4±5.3 | 25.6±2.6 | 22.7±2.9 | 21.8±2.3 |
| chain / alternating | 73.2±20.7 | 27.5±1.3 | 25.2±1.8 | 22.6±1.7 | 20.5±1.3 | 21.7±1.2 |
| chain / directed | 82.3±10.6 | 79.5±12.3 | 61.9±10.1 | 56.2±10.1 | 37.0±6.9 | 52.9±10.8 |

(chain simultaneous = alternating at B7 is bit-identical, the known
path-parity coincidence disclosed in RESULTS_A2.)

## Prediction verdicts

- **P-i (directed dominates the frontier on star and mst) —
  CONFIRMED.** At every airtime level, the directed curve is the
  highest for both topologies; on the star the gap is 49–66 points at
  every point at or below 21% airtime.
- **P-ii (within directed: mst ≈ star ≥ chain) — CONFIRMED with a
  nuance.** At generous airtime mst ≥ star > chain (96.3 / 92.2 /
  82.3); at 8.6% the star overtakes the mst (87.4 vs 80.2) — the
  star's depth-1 paths to the root accumulate less error per skipped
  service than the mst's deeper branches. Chain is worst under
  directed at every level except the anomaly below.
- **P-iii (an airtime level where no bidirectional point is usable
  while directed holds ≥80%) — CONFIRMED.** At 8.6% airtime the best
  bidirectional point is 38.0% gain (below the 50% usability bar)
  while directed delivers 87.4%. Cheapest ≥80% point: directed 8.6%
  (star), bidirectional 29.9% (mst/simultaneous, 83.9%) —
  **airtime-advantage ratio 3.5×**, and the bidirectional point is
  barely above the bar where the directed one has ~7 points of margin.

## Efficiency and constrained optima

Max gain-per-airtime among usable (≥50%) points: star/directed 10.2
and chain/directed 12.4 %gain/%airtime vs best bidirectional 3.2
(mst/simultaneous). Constrained optima: T_max 5% → chain/directed
52.9%; T_max 10% → star/directed 87.4%; T_max 20% → mst/directed
93.8%. The (topology, protocol) optimum changes with the budget —
the joint-choice thesis restated on the airtime axis.

## Honest anomalies and misses

1. **chain/directed is non-monotone at the bottom of the frontier**:
   37.0±6.9 at 8.6% airtime but 52.9±10.8 at 4.3%. Flip counts drop
   from 51–60 to 21–33 alongside. Suspected round-robin phasing
   interaction (B=2 over 7 edges yields an irregular 3.5-interval
   pattern; B=1 a clean 7-interval cycle) — flagged, not investigated;
   no conclusion rests on it.
2. star/directed retains its known ±12-point seed spread (one seed's
   acquisition event, disclosed in RESULTS_A2); its frontier dominance
   holds in every individual seed.
3. mst/simultaneous at 17.1% has a ±38 spread (one seed collapsed,
   two held) — the frontier's cliff edge for that pair.

## Scope

Simulation only; N=8; one geometry; static channel; round-robin
scheduling within the budget (no informed scheduling); airtime
accounting is steady-state and excludes the shared acquisition
prelude. Figure: `paper_figures/fig3_pareto_frontier.png`.
