# Direction C — joint node participation + edge selection

Question: which subset of radios should a distributed array spend
synchronization airtime on, over which links, to maximize expected
coherent beamforming gain — charged honestly against the full array
(a benched node's lost amplitude counts; normalization is the
all-N amplitude sum, matching the project's membership convention).

Scripts: `dirC_joint_selection.py` (model + sweep, cache
`dirC_cache.json`), `dirC_validate.py` (refinement, exactness,
waveform checks, figure `figures/dirC_gain_vs_budget.png`, cache
`dirC_validate_cache.json`).

## Model (and two model corrections caught by smoke runs)

Expected gain E[G(S, E)] is computed from the time-averaged phase
covariance of a two-state-per-node (phase + frequency, grounded)
Kalman recursion driven through the actual round-robin service
schedule the budget affords: budget A buys A/0.19124 exchanges per
interval, split across selected edges; per-edge measurement noise
from the same path-loss/SNR law as the waveform testbed plus the
intra-capture oscillator floor (corrected attribution — oscillator,
not multipath). Pairwise coherence credit e^(−var/2), Gaussian.

Two artifact catches during construction, per project discipline:
an edge-level model let dense graphs average away oscillator drift
that is physically common per node ("complete graph always wins" at
any budget — wrong); a phase-only model omitted the frequency walk,
making 67-interval coasts look benign in direct contradiction of the
project's validated coast-time law. Both were fixed before any
conclusions were drawn; the final model is the node-level
phase+frequency recursion.

## Prediction (registered before the sweep)

At tight budgets the joint optimum benches nodes and beats every
full-array topology; |S*| grows with budget; crossover expected
around 1–5%.

## Results

**1. The partial-participation window is real, large, and
band-limited (the prediction was right in the middle and wrong at
both ends).** Sweep: N=8, budgets {1,2,5,10,20}%, 2 geometries ×
3 seeds × 2 amplitude models (12 instances, seeds are deployment
draws):

| budget | outcome (all 12 instances agree) |
|---|---|
| 1–2% | nothing can cohere; best policy is all-in *unsynchronized* (G = incoherent floor 0.125–0.131); benching only loses diagonal power |
| 5% | **joint optimum benches 3 of 8 nodes and beats the best full-array topology in 12/12 instances**, by +0.083 to +0.121 absolute gain (e.g. 0.284 vs 0.160 — 78% relative) |
| 10–20% | budget affords the full array; |S*| = 8, best full topology (hub star) is optimal |

So partial participation wins not "at tight budgets" but in a
**window**: above the budget where a subset can cohere, below the
budget where everyone can. Below the window, benching is
counterproductive — an effect the honest all-N normalization exposes
and a selected-nodes-only normalization would hide.

**2. The participation staircase (refinement at {3,4,6,8}%,
path-gain amplitudes, seed 0):** |S*| = 8(incoherent) → 3–4 → 4 → 5 →
6 → 7 → 8 across 1→10%. Within the coherent window the optimal sets
are **nested** (greedy-consistent): uniform {2,4,5,6} ⊂ {2,4,5,6,7} ⊂
{1,2,4,5,6,7} ⊂ … ; clustered {1,4,7} ⊂ {1,3,4,7} ⊂ {1,2,3,4,7} ⊂ ….
The raw sweep's "12/12 chains jump" count is a boundary artifact: the
only non-nested step is entering the window (all-in-incoherent →
small subset), a regime change rather than a selection reversal.

**3. Selection is amplitude-vs-cost, not geometry.** In the clustered
instances the two outliers get opposite treatment: the target-side
outlier (amplitude 1.00, the strongest beam contributor) is in S*
from the very first coherent budget despite having the most expensive
sync links; the far-side outlier (amplitude 0.48) is benched at every
budget below 10%. A strongest-k or nearest-k heuristic makes neither
call correctly.

**4. Exactness.** The structured+greedy edge optimizer matches full
edge-subset enumeration exactly on the 5-node check at both 5% and
20% budgets (gap +0.00000).

## Waveform validation — honest status: anchored at single-hop,
confounded at multi-hop

Predictions were printed before measurement. Outcome:

| configuration | predicted G | measured G |
|---|---|---|
| N=2, one edge, serviced every interval | 1.0000 | **0.9972 ± 0.0008** (residual ~105 mrad, consistent with the project's recorded two-way loops) |
| tree8, 7-interval per-edge cycles | 0.9375 | 0.134 ± 0.009 (100 itv) → 0.164 ± 0.019 (200 itv) |
| complete8, 28-interval cycles | 0.7516 | 0.126 ± 0.012 |

The single-hop, densely-serviced anchor agrees to 0.3%. The
multi-hop failure is **not evidence against the model**: the shared
testbed fixes a symmetric degree-weighted consensus control law,
which this project previously measured to cost 20–25+ points of gain
even at every-interval service (its own results file registers ~N²
convergence time as a known caveat), and here per-edge service is
7–28× sparser. The 100→200-interval improvement (0.134→0.164, still
crawling) matches slow consensus convergence, not an estimation
limit. Conclusion: direction C's results hold at the level of the
estimation-limit model (centralized-filter bound), anchored by the
single-hop waveform point; full multi-hop waveform validation
requires a directed / turn-taking control variant of the testbed
(the control law the project showed reaches 99.8% at dense service)
— flagged as the follow-up, out of this run's scope.

## One-line summary

Under a fixed synchronization-airtime budget there is a coherence
window in which benching 3 of 8 radios and spending the whole budget
on the remainder beats every full-array topology by up to 78%
relative gain (12/12 instances); the optimal roster is nested as the
budget grows, and it keeps expensive-but-strong contributors while
benching cheap-but-weak ones — behavior no amplitude- or
geometry-only heuristic reproduces.
