# Direction A results — airtime-constrained topology selection

N=8, two frozen geometries (uniform disk; two clusters 800 m apart),
seeds 0–2, 150 intervals, pilot 255 samples (one two-way exchange =
4.28% of the frame), budgets B ∈ {2,3,4,5,7} exchanges/interval =
{8.6, 12.8, 17.1, 21.4, 29.9}% sync airtime. All runs on the
cadence-aware runner (`dirA_runner.py`; see "testbed corrections").
Predictions were computed and printed before every waveform run
(`dirA_selection.py`; cache `dirA_cache.json`; figures
`figures/dirA_gain_vs_budget_{uniform,clustered}.png`).

## Headline finding (not the one the direction assumed)

**At equal airtime, topology selection for this open-loop consensus
loop is dominated by dynamical stability, not estimation quality.**
The estimation layer (weighted effective resistance → expected gain)
predicts near-identical performance for all reasonable topologies;
what actually separates them is three measured stability constraints:

1. **Service-cadence cap.** Per-edge service interval m ≲ 2 intervals;
   gain decays gradually and with large across-seed variance beyond
   (MST, 3 seeds: 84% at m=1.0, 69% at 1.4, 48±31% at 1.75, 45% at
   2.33, 26% at 3.5). Control: the decay is the same at pilot lengths
   255/1023/2047, ruling out per-capture frequency noise; the driver
   is the frequency random walk between services interacting with the
   mod-π branch pick.
2. **Degree cap.** High-degree nodes destabilize under simultaneous
   half-corrections (the project's known consensus tax, reproduced
   here): at B=7, same geometry, same edge count — star (hub degree
   7): 23.7±0.9%, 113 flips; degree-capped tree (max 3): 80.2±19.5%,
   25 flips; MST (max 4): 83.9±6.8%.
3. **No chords.** Any cycle traps π-windings the tree-based branch
   check cannot fix (the cut-space structure from the graph-theory
   module, seen dynamically): index chain 73.2±16.9% → add one chord
   (ring) 38.2±2.1%, flips 26→65. This is why spectral (MST+2 chords,
   22%) and complete (22%) fail while plain MST works.

Consequence: **the minimum-variance spanning tree beats every other
strategy at every budget in both geometries** — including the
gain-per-airtime greedy built on the resistance objective, which
(correctly maximizing the estimation layer) chooses star-like or
chorded sets and loses. The estimation-optimal graph is dynamically
the worst. This gives mechanism to Ngo & Larsson's observed
"denser calibration topologies can degrade performance" and to
Larsson's unbounded-error topologies, in the open-loop consensus
setting.

## Campaign table (measured gain %, mean±std over 3 seeds; predicted in parens)

Uniform geometry:

| strategy | 8.6% | 12.8% | 17.1% | 21.4% | 29.9% |
|---|---|---|---|---|---|
| complete | 22.3±0.6 (32) | 24.3±1.0 (32) | 23.9±1.7 (32) | 21.0±1.4 (33) | 23.6±1.4 (35) |
| star | 24.1±1.8 (35) | 21.3±2.2 (52) | 22.5±0.9 (54) | 24.3±1.7 (73) | 23.7±0.9 (86) |
| ring | 49.0±31.4 (35) | 23.1±2.5 (47) | 20.7±1.3 (53) | 21.6±11.7 (62) | 26.7±7.6 (81) |
| **mst** | **38.0±16.0** (35) | **44.8±14.3** (52) | **47.7±31.0** (54) | **69.2±14.5** (73) | **83.9±6.8** (86) |
| spectral | 21.9±1.0 (34) | 20.8±2.0 (42) | 23.5±1.4 (52) | 22.7±0.3 (54) | 22.8±0.6 (77) |
| greedy | 24.5±1.6 (35) | 23.3±1.3 (52) | 21.8±1.2 (54) | 19.2±2.6 (73) | 20.9±2.0 (86) |

Clustered geometry: same ordering; MST 75.6±3.4% at 21.4% airtime,
all others ≤31%. (Full numbers in the cache/figures.)

## Answers to the direction's stated questions

- **Does gain-aware selection beat max-SNR tree and spectral at equal
  airtime?** No — it loses, decisively, in both geometries. The
  resistance-layer objective (with or without the measured stability
  curve) ranks star/chorded sets highest because it cannot see degree
  or winding costs. The honest positive claim is inverted: naive
  estimation-driven selection is actively harmful; the stable-and-good
  family is low-degree chord-free trees with per-edge service ≲ 2
  intervals, and within that family edge quality matters modestly
  (MST 84% ≥ degree-capped 80% ≥ arbitrary chain 73% at B=7).
- **Does the advantage need heterogeneity?** The MST-vs-rest gap is
  large in both geometries; heterogeneity (clustered) does not rescue
  estimation-driven selection.
- **Airtime floor:** with this pilot family and N=8, no spanning
  topology holds coherence below ~17% sync airtime (trees need
  B≥4 → m≤1.75); at 8.6% everything is at or near the incoherent
  floor. Budgets of 2–5% are unreachable for N=8 with dedicated
  exchanges alone at this frame design.
- **Submodularity:** the stabilized objective is far from submodular
  under rate dilution (271/400 empirical violations); the fixed-rate
  resistance objective is nearly submodular (10/400). No greedy
  guarantee should be claimed.

## Predicted-vs-measured verdict (honest miss)

The README's resistance objective — even after multiplying by a
stability curve calibrated on the MST cadence sweep (disclosed
calibration, one topology) — does NOT predict strategy ordering: it
is blind to the degree and chord mechanisms. Prediction quality is
good only within the tree family (MST predicted 86/73/54, measured
84/69/48 at B=7/5/4, uniform). A usable selection objective must be:
minimum-variance spanning tree, degree-capped, no chords, subject to
m = (N−1)/B ≤ ~2 — i.e., the feasible search space is trees, not
edge subsets.

## Testbed corrections made along the way (contained in this folder)

- `dirA_runner.py`: cadence-aware variant of the sandbox testbed —
  the original predicts each edge's filter only at service, which at
  sparse cadence propagates one interval of dynamics across an
  m-interval coast and flip-storms (controls: storms persist at zero
  initial CFO and every pilot length; soften only as m→1). The
  sibling file is untouched (correct for its own every-interval use).
- Stagger gap: the original parity-stagger engages only at
  budget=None; budget ≥ |E| degenerated to simultaneous issuance (the
  Jacobi storm it was built to prevent). Fixed in the variant.

## Caveats

All constraints are properties of THIS control law (symmetric
half-corrections + tree-flip branch check): the project's earlier mesh
work showed directed/alternating laws behave differently, so the
degree and chord limits should be read as "for symmetric consensus",
not universal. One controller family, N=8, two geometries, 3 seeds,
static channel, simulation only. The stability curve S(m) is
calibrated, not derived. Detection failures on weak edges (star,
complete, greedy in clustered geometry: 77–98% detect) are unmodeled
in the predictor and contribute to (but do not explain — see the
100%-detect failures) the misses.
