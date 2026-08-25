# Open-loop topology measurements (openloop_topology_study.py)

Status: PREDICTIONS REGISTERED, campaigns not yet run. Results are
appended below the prediction block after each campaign completes, in
run order — nothing above this file's "MEASURED RESULTS" line was
edited after the first campaign started.

Testbed: N-node open-loop sync over the full physical layer
(waveform captures, TDL channels, oscillator noise incl. flicker,
timing jitter, quantized corrections). Measurement graph is
configurable per edge: "two-way" edges use the half-difference of a
bidirectional exchange (propagation phase cancels); "one-way" edges
use a single capture whose phase is the relative oscillator phase
PLUS that edge's unknown propagation phase. Control law held fixed
across all topologies (symmetric degree-weighted consensus) so graph
comparisons are not confounded. Node 0 is a bookkeeping gauge only.

Trap controls carried from this project's four retracted findings:
randomized (non-grid) per-node initial frequency offsets; run lengths
covering ≥4 cycles of the sparsest service cadence; no use of the
length-blind calibration cache; ≥3 seeds everywhere with spread
reported; discriminating controls (zero-frequency-offset run,
environment-motion run) attached to the identifiability campaign.

## Predictions (registered before running)

1. **Identifiability (exp 1).** All-two-way trees and rings hold
   bounded pair phases. One-way-only graphs: frequency locks (a
   static channel adds nothing to a frequency measurement), but each
   pair's phase settles at MINUS its channel phase — stable,
   radian-scale biases in a static environment. Under environment
   motion (0.02 m/s) those biases track the channel: the array's
   internal phase wanders with the environment while the two-way
   topologies do not. (This refines the directive's "predict: drifts"
   — the failure mode is channel-valued bias, drifting if and only if
   the environment drifts.) A single two-way edge added to a one-way
   ring repairs only that edge's pair; distant pairs stay biased.
2. **Accuracy vs topology (exp 2).** At equal total airtime, the
   steady phase-error variance between nodes i and j tracks the
   graph's effective resistance R_ij (chain end-to-end worst, complete
   graph best; positive correlation between pair variance and R_ij
   within and across graphs).
3. **Size scaling (exp 3).** Chain end-to-end deviation std grows
   ~ sqrt(hops) (series resistance); ring antipodal ~ half that
   trend. Caveat registered up front: symmetric-consensus convergence
   time grows ~ N² on chains (spectral gap), so the largest cells may
   not be fully converged within bounded runs — if so it will show as
   first-half vs second-half disagreement and will be reported as
   non-convergence (a known property), not as a scaling discovery.
4. **Branch states (exp 4).** On a two-way ring with the branch check
   disabled and adverse acquisition, settled states have each edge
   near 0 or π with an EVEN number of π-edges around the cycle
   (physical consistency: pair phases sum to zero around a closed
   loop), and multiple distinct states occur across seeds.

---

## MEASURED RESULTS

### Testbed validation and defects found (before any campaign)

Pre-campaign validation at N=8 exposed three real mechanisms, each
confirmed by a discriminating control before any fix; all are part of
the record because they are physics, not bugs of the physics:

1. **Turnaround bias corrupts acquisition branches.** The two-way
   half-difference is biased by π·f_rel·τ_turnaround (2.4 rad at
   750 Hz pair offset, 1 ms turnaround). The production star
   compensates with the measured frequency; the mesh lineage this
   testbed derives from does not. Control: instrumenting the
   measurement-vs-truth error showed a persistent π-branch error on
   affected edges (3121.9 mrad mean, max 3141) that vanished for
   unaffected edges (9–20 mrad). Fixed by adding the star's
   compensation term.
2. **Branch migration under residual frequency slew.** Between
   corrections, residual pair frequency slews the phase across the
   mod-π fold faster than the filter's prediction follows, silently
   migrating the measurement branch. Zero-frequency-offset control:
   clean (adjacent edges 156–432 mrad); ±750 Hz randomized offsets:
   1–2 rad chaos with zero-mean ~1 rad/interval increments and
   frequency traces identical to the clean case — isolating the
   effect to the phase-measurement fold, not the clocks.
3. **Flip-on-excursion instability.** A branch check that fires on
   any anti-phase excursion *creates* π errors when the excursion was
   honest jitter the filter already tracked. Gating the check on the
   silent-wrong-lock signature (filter confident near zero AND truth
   anti-phase) cut flips ~5×. A flip-storm escape (re-acquire after
   two flips of the same edge within 4 intervals) handles the
   remaining mod-π frequency-alias locks (residual frequency errors
   of k/(2T) are invisible to mod-π phase tracking sampled every T).

Consequence for design: campaigns 1–4 run in the
**frequency-presynchronized regime** (zero initial frequency offset),
which isolates the graph-structure physics; a ±200 Hz column is
retained in campaign 1 to document how acquisition dynamics
contaminate topology comparisons. Legacy (SDR-class) oscillator
noise; symmetric consensus with per-node incident-correction
averaging (see campaign 2's control for why summation fails on hubs);
Gauss-Seidel parity staggering after a 12-correction settling phase.

### Campaign 1 — identifiability by topology (N=8, seeds 0–2, 160 intervals)

Worst-pair circular std / worst-pair |bias| / movement of pair means
between steady half-windows ("mean-move"), worst over seeds, mrad:

| topology | static: std / bias / move | moving 0.02 m/s: std / bias / move |
|---|---|---|
| two-way chain (tree) | 1462* / 1104 / 2760* | 1475* / 1136 / 2793* |
| two-way ring | 309 / 404 / 429 | 316 / 410 / 442 |
| one-way ring + one two-way edge | 863 / 3071 / 211 | 2495 / 3140 / 3135 |
| one-way chain only | 899 / 3094 / 302 | 2266 / 3020 / 3116 |
| one-way ring only | 715 / 3132 / 381 | 1544 / 3093 / 3125 |
| mixed random (half two-way) | 1093 / 2945 / 257 | 2537 / 3111 / 3108 |

(*the two-way chain's seed 0 is an isolated acquisition-branch event,
reproduced identically under motion — deterministic, catalogued in the
anomaly list; its other seeds are 271–300 std / 250–338 bias.)

**Prediction 1: CONFIRMED, in its refined form.** Two-way topologies
are bounded and *unchanged by environment motion* (ring: 309→316).
One-way-only topologies hold **stable channel-valued biases** (π-scale
biases, small mean-move) in a static environment — bounded bias, not
unbounded drift — and under 0.02 m/s motion those biases wander
(mean-move 86–380 → 908–3135 mrad): the array's internal phase is
hostage to the environment exactly when, and only when, the
environment moves. A single two-way edge repairs only its own pair.
The ±200 Hz frequency-offset column (cache key `exp1_cfo`) degrades
every topology including two-way ones — acquisition dynamics, not
graph structure.

### Campaign 2 — accuracy vs topology at equal airtime (N=8, budget 7 exchanges/interval, seeds 0–2)

| graph | edges | worst-pair std | best-pair std | corr(pair var, effective resistance) |
|---|---|---|---|---|
| chain | 7 | 727 | 109 | **+0.97** |
| ring | 8 | 229 | 113 | **+0.91** |
| star | 7 | 128 | 95 | +0.41 (R range only 1–2) |
| complete | 28 | 1612 | 366 | −0.00 (anomalous cell, see control) |
| random tree+1 | 8 | 247 | 110 | **+0.85** |
| random tree+4 | 11 | 272 | 112 | **+0.95** |

**Prediction 2: CONFIRMED on five of six graphs** — pair phase-error
variance tracks the graph's effective resistance (correlations
+0.85…+0.97 where the resistance range is wide; the star's weak +0.41
reflects its narrow resistance range 1–2, not a violation).

**Anomalous cell with control.** The complete graph at budget 7 (each
edge serviced every 4th interval) collapses (1.6–1.8 rad). Control:
full service → 143 ± 15 mrad; budget 14 (every 2nd interval) →
171 ± 9; budget 7 → 1823 ± 616. The collapse is a **service-staleness
threshold for high-degree nodes** (between service visits, an edge's
endpoints absorb ~12 corrections from other edges; between cadence 2
and 4 the disturbance-to-service ratio crosses stability), not a
property of the topology. Catalogued; mechanism story beyond the
control is deliberately not claimed.

Two control-law facts were required to get here (both measured, both
now part of the design record): plain summation of simultaneous
half-corrections overshoots hubs (star: seven at once, ×3.5 —
unstable at 1.8–2.5 rad until incident-correction averaging was
applied), and simultaneous degree-weighted Jacobi at N=8 storms when
combined with an aggressive branch check (the repo's own
consensus-tax result, reproduced).

### Campaign 4 — branch states on a two-way ring (N=8, adverse acquisition, check disabled, 24 seeds)

21 of 24 seeds settle; **21 distinct states**; 3 unsettled.
**Prediction 4: REFUTED in its stated form, and the truth is
cleaner.** Both even AND odd counts of π-labeled edges occur — but
inspecting the settled states shows the cycle sum of pair phases is
**exactly 2π·w (measured ±1.00 windings to three decimals)**: the
states are quantized *winding states*. An even-parity state realizes
its winding discretely (π-edges, zero tilt: offsets ≤0.03 rad); an
odd-parity state closes the cycle by distributing the defect
uniformly (~π/8 ≈ 0.32–0.46 rad per edge). The mod-π branch label is
the wrong invariant; the winding number is the right one — the
engineered loop reproduces coupled-oscillator twisted-state
multistability through the branch mechanism.

### Campaign 3 — size scaling (chains and rings, seeds 0–2)

| topology | N | far-pair std (mrad, per seed) | near-pair std |
|---|---|---|---|
| chain | 8 | 144, 92, 174 | 88–145 |
| chain | 16 | 79, 118, 547 | 94–158 |
| chain | 32 | 728, 672, 998 | 95–190 |
| ring | 8 | 76, 140, 151 | 57–116 |
| ring | 16 | 227, 266, 192 | 108–156 |
| ring | 32 | 734, 442, 784 | 123–252 |

**Prediction 3: qualitatively confirmed, quantitatively bounded by
the pre-registered caveat.** Near-pair (adjacent) accuracy is flat in
N (~90–190 mrad at every size) — local synchronization does not
degrade with array size. Far-pair error grows with size (chain
~137 → ~800 mrad mean from N=8→32); the growth at N=32 exceeds the
√N resistance prediction, but the first-half/second-half windows
disagree there (e.g. 996 vs 672), i.e. the runs are not fully
converged — exactly the consensus-convergence-time (~N²) caveat
registered before running. The clean statement supported: local
error flat, end-to-end error growing with hop distance, with the
N=32+ exponent not separable from convergence within these run
lengths. N=64 extension (300 intervals, ~77 s/seed): near-pair accuracy
remains flat (chain 153–203 mrad; ring 183–448) — **local
synchronization quality is size-independent out to 64 nodes** — while
far-pair windows disagree severely (chain seed 2: 1756 first half vs
444 second; ring seed 1: 585 vs 2092), i.e. N=64 is thoroughly
unconverged at 300 intervals, consistent with the ~N² consensus
convergence time (≥415 intervals needed). The far-pair scaling
exponent at N≥32 is therefore explicitly NOT claimed from these runs.

### Anomaly list (honest, controls attached)

1. Two-way chain seed 0: isolated 1.4 rad pair event, reproduced
   bit-identically under environment motion (deterministic
   acquisition branch event, not noise). Not investigated further.
2. Complete graph at quarter-rate service: collapse with a clean
   service-rate control (above); threshold between cadence 2 and 4.
3. Three of 24 ring seeds unsettled in campaign 4 within 80
   intervals (slow winding-state relaxation).
4. The mod-π frequency-alias lock (k/(2T) grid) and the branch-
   migration mechanism (validation section) are loop-design
   phenomena that any experimental realization of open-loop mod-π
   sync will face; they are reported as findings of the testbed
   validation, with instrumentation evidence.

### Wall-clock envelope (Mac Studio, CPU)

N=8 full-service graph runs: 1–2 s per 160 intervals; N=32: ~24 s per
128 intervals; complete graph (28 edges): ~4 s per 100 intervals;
the entire campaign set above: under 15 minutes sequential. N=64
chains/rings at 300 intervals: ~2 min per seed (background log).
