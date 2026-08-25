# Synchronization Topology and Update Protocol Are Not Separable — Technical Summary

*(Formerly one page; expanded to full technical depth. Every number
traces to `topology_selection/RESULTS_A.md`, `RESULTS_A2.md`,
`RESULTS_A3.md`, `RESULTS_A4.md`, and the run caches beside them.
Figures: `topology_selection/paper_figures/` with its own
`FIGURES.md`.)*

## 1. Problem and objective

A distributed coherent array realizes beamforming gain
G = |Σᵢ e^{jθᵢ}|²/N² only while its per-station carrier phases θᵢ stay
locked. Lock is maintained over a *measurement graph*: two-way
synchronization exchanges on selected links, where the half-difference
½·wrap(φ_fwd − φ_rev) cancels the reciprocal propagation phase and
measures the pairwise clock offset — but only modulo π, since the
half-difference observes wrap(2θ). Choosing this graph is treated in
the literature as an estimation problem: the best-linear-unbiased
phase-error covariance equals effective resistance on the graph
(Barooah–Hespanha), identifiability follows from graph structure
(Freris–Graham–Kumar, IEEE TAC 2011), and topology families are
analyzed against calibration-error scaling (Larsson, arXiv:2401.11730)
or consensus convergence (Rashid–Nanzer, arXiv:2201.08931). All of
these treat the graph as feeding an estimator. We ask what happens
when the graph feeds an *actuated closed loop* — nodes physically
steering their oscillators from the measurements, through the update
protocol that schedules and applies corrections — and the metric is
realized coherent gain rather than estimation error.

## 2. Experimental system

**Physics.** Waveform-level simulation: each exchange transmits a
Zadoff-Chu pilot through a tapped-delay-line channel with additive
noise, timing jitter, IQ imbalance, DC offset, quantized corrections,
TDD turnaround; oscillator noise (white-PM, white-FM, flicker-FM,
random-walk-FM) anchored to datasheet Allan deviation. Randomized
per-station frequency offsets (never grid-structured — a
previously-diagnosed aliasing trap). N = 8 primary, N = 16 scaling
cells; seeds ×3 (×2 at N = 16); run lengths ≥ 4 cycles of the
sparsest service cadence.

**Protocols, identical physics underneath.**
- *Simultaneous*: every active edge measures and both endpoints apply
  half-corrections in the same interval (Jacobi-like — the convention
  of the published consensus-synchronization literature).
- *Sequential*: edge-colored turn-taking; edges fire in a proper
  edge-coloring schedule so no node updates on two edges at once
  (Gauss-Seidel-like, masterless). Its intrinsic rate cost is
  1/chromatic-index.
- *Directed*: elected-root spanning tree; each node corrects fully
  toward its parent; corrections propagate root-outward.

**Accounting and discipline.** Equal synchronization airtime within
every comparison (airtime = exchanges × capture-length/interval; one
exchange = 2×1069 samples of a 50,000-sample interval = 4.276% per
per-interval exchange). Predictions and linearized iteration-matrix
spectral radii computed and *printed before any run*; the simultaneous
arm reproduces the prior study bit-identically (85.70% = 85.70%);
every mechanism claim carries a discriminating control
(zero-frequency-offset cells, pilot-length sweeps, per-run branch-flip
and winding-state counters). 118 cells in the fork campaign
(`dirA2_cache.json`) + 108 in the airtime campaign (`dirA3_cache.json`).

## 3. The motivating result (why the fork was needed)

A six-strategy topology-selection race under the simultaneous protocol
(complete graph, star, ring, cost-MST, max-SNR tree, spectral
max-connectivity, versus a greedy selector maximizing predicted
expected gain over pair resistances) produced an inversion of the
estimation-first premise: the gain-aware selector *lost* to simple
minimum-variance spanning trees at every airtime budget in both tested
geometries, because estimation quality was not the binding constraint —
loop stability was. Three empirical constraints emerged (service
cadence, node degree, cycle presence), all under one protocol —
raising the confound the fork experiment was built to resolve: are
these properties of topology, or of the protocol?

## 4. The three-protocol fork: three mechanisms, three origins

**Cadence — physical.** Sweeping per-edge service interval m from 1.0
to 3.5 on the MST: gain falls 84→38% (simultaneous), 68→27%
(sequential), 96→80% (directed). The ceiling near m ≈ 2 appears under
*every* protocol. Pilot-length controls (255/1023/2047 samples,
identical curves) exclude per-capture measurement noise; the zero-CFO
control (84% at the reference point) excludes deterministic frequency
offset; the driver is oscillator frequency walk accumulating between
services. Protocols differ in *expression*, not existence:
bidirectional protocols amplify branch-ambiguity errors into flip
cascades (32–91 flips/run), while the directed tree confines a branch
error to the subtree below it (3–9 flips).

**Degree — numerical.** A degree-7 hub (star) under simultaneous
updates collapses to 23.7 ± 0.9% gain with 113 branch flips/run —
exactly where its linearized Jacobi iteration diverges (pre-registered
spectral radius 1.107 > 1). Two cures work: *direction* (92.2 ± 10.2%)
and *degree-weighted damping* of simultaneous updates (92.2 ± 10.3%,
per-seed identical — the linear theory's own prescription). Sequencing
alone does **not** cure it (38.9 ± 2.7%): turn-taking on a degree-7
hub necessarily serves each incident edge every 7th slot, which
re-enters the cadence mechanism. The degree and cadence mechanisms are
*coupled* — protocol fixes for one can activate the other — which is
why no per-mechanism patch list substitutes for choosing the protocol
correctly.

**Cycles — topological.** Adding one chord to a chain halves gain
under both bidirectional protocols (73.2→16.9% simultaneous;
73.2→38.3% sequential; the chain values agree bit-identically across
campaigns as a cross-check). Directed protocols operate on trees and
are structurally exempt. The failure states are not noise: every
settled impaired configuration carries an exactly quantized winding
number (below).

## 5. The winding-number theory (proved, with two of our own conjectures refuted)

Define, for a cycle C with consistently oriented edges, the winding
w_C = (1/2π)·Σ_{e∈C} wrap(Δθ_e). Four results
(`dirA4_winding_theorem.py`, all numerically verified):

1. **Exact quantization at every instant.** w_C ∈ ℤ for *every* phase
   configuration — not only settled ones. (Unwrapped differences
   telescope to zero around a cycle; wrapping subtracts whole turns —
   the sync-graph form of the Goldstein residue identity from phase
   unwrapping.) Verified to 8.4×10⁻¹⁵ worst deviation over 10⁵ random
   configurations. Empirically, settled impaired states occupied 21
   distinct winding states over 24 seeds, cycle sums integer to
   ±0.001 — measured independently in a second testbed.
2. **Conservation under sub-boundary dynamics.** w_C changes only when
   a cycle edge's difference crosses ±π; any discrete update step
   conserves it provided each cycle edge moves less than its distance
   to the boundary (|Δd| < π − |wrap(d)|). This is a *step-size
   hypothesis*, not a universal claim — verified with zero violations
   over 35,683 bound-respecting steps, and all 2,620 winding jumps on
   deliberately boundary-crossing paths matched signed-crossing
   bookkeeping exactly.
3. **The flip action — our conjecture refuted.** We conjectured node
   π-flips (the branch-ambiguity resolution acting on a node) are
   cycle-neutral by cut/cycle orthogonality. False in the wrapped
   setting: the exact rule is Δw = −½(sgn r₊ + sgn r₋) over the two
   cycle edges adjacent to the flipped node, independent of flip sign.
   (Orthogonality holds unwrapped; wrapping breaks it.) Verified with
   zero mismatches over 8,295 generic flips; the predicted entry rate
   into wound states from near-lock equals the sign-agreement
   probability exactly (0.467 = 0.467).
4. **Metastability, not invariance — second conjecture refuted.** The
   hoped-for impossibility ("local corrections cannot unwind a
   cycle") is false: an unwinding flip exists at every node of a
   settled wound state. What is true — and is the actual robustness
   mechanism — is a *barrier*: every escape from a wound sector,
   whether by flip or by continuous slip, must transit an anti-lock
   event of magnitude π − 2π|w|/n, and the measured escape transients
   match this formula per cell. Wound states are therefore metastable
   under **every** local update protocol — an earned strengthening of
   the claim we had previously softened to "the tested three."

Terminology, precisely (a reviewer-prompted correction): the *set* of
branch-consistent configurations forms a coset of the graph's **cut
space** (node flips generate cuts); the *winding constraint* lives in
the **cycle space**. The two layers interact only through wrapping —
which is exactly what result 3 quantifies.

## 6. The linear theory column, and its honest limit

Pre-registered spectral radii of the linearized update iterations rank
the protocols correctly (star-simultaneous 1.107, divergent;
sequential 0.944, stable; damping brings simultaneous below 1). But
linear stability is necessary, not sufficient: the sequential-complete
cell has spectral radius 0.000 — exact one-step linear convergence —
yet measures 29% gain. This is a clean falsification of any purely
linear account: the binding nonlinearity is the modulo-π branch layer
of the two-way measurement, which no covariance or consensus-rate
analysis represents.

## 7. The inversion, and scaling

The star is the *worst* tested topology under simultaneous updates
(23.7%) and the *best* under directed updates (99.4 ± 0.0%). At
N = 16, every bidirectional protocol collapses on every tested
topology (13–20% gain, 113–242 flips/run) while the directed protocol
holds 89.3 ± 1.6% (MST) to 99.4 ± 0.0% (star).

Stated precisely, the central claim is a ranking non-invariance:
there exist topologies 𝒢₁, 𝒢₂ and protocols P₁, P₂ with
G(𝒢₁,P₁) > G(𝒢₂,P₁) while G(𝒢₁,P₂) < G(𝒢₂,P₂) — demonstrated here
with 𝒢₁ = MST, 𝒢₂ = star, P₁ = simultaneous (83.9% > 23.7%),
P₂ = directed at N = 16 (89.3% < 99.4%). The optimal topology is not
merely protocol-*sensitive*; its *identity changes* with the
protocol, so the standard design decomposition — choose the topology
first, choose the synchronization algorithm second — is invalid. The
same non-invariance appears on the resource axis: the
coherence–airtime Pareto-optimal (topology, protocol) pair changes
with the airtime budget (§8), so topology optimization and protocol
optimization do not commute with each other or with resource
allocation.

## 8. The coherence–airtime frontier

Making airtime an explicit axis (108 cells: {star, MST, chain} ×
three protocols × six exchange budgets, N = 8): the directed protocol
dominates the Pareto frontier at every airtime level on star and MST,
by 49–66 points at ≤ 21% airtime. The cheapest configuration
sustaining ≥ 80% gain costs **8.6% airtime under directed versus
29.9% under the best bidirectional configuration — a 3.5× airtime
advantage** at matched performance on the same graphs and physics.
Efficiency: 10.2–12.4 %-gain-per-%-airtime (directed) versus 3.2
(best bidirectional). The constrained optimum *changes with budget* —
5% → chain/directed, 10% → star/directed, 20% → MST/directed — with a
mechanistic nuance (the star overtakes the MST at tight airtime
because depth-1 paths accumulate less error per skipped service).
This restates non-separability on the resource axis a system designer
actually allocates. One disclosed, uninvestigated anomaly: the
chain/directed cell is non-monotone at the lowest budgets (suspected
service-schedule phasing; nothing rests on it).

## 9. Relation to prior work

Larsson (2401.11730): who-measures-on-whom topology versus
calibration-error scaling, including unbounded-error families — an
estimation analysis with no actuation loop. Rashid–Nanzer (2201.08931):
simultaneous consensus dynamics with drift and estimation error,
validated at statistics level — our degree and cycle mechanisms
operate exactly in that convention's blind spot (its own update
structure is the divergent one, and its validation cannot see branch
capture). Shandi et al. (2410.17356, 2405.18384): SDR implementations
showing connectivity affects convergence, including dynamic
connectivity — no protocol comparison, and a feasibility precedent for
our hardware plan. Ngo–Larsson (2509.03722): calibration topology
inside the TDD flow with the observation that denser graphs can
reduce spectral efficiency — heuristic link selection, no optimizer,
no protocol coupling. Mathematical ancestry, cited not claimed:
Jacobi/Gauss-Seidel convergence theory; Goldstein residues and phase
unwrapping; ℤ₂ group synchronization; effective-resistance estimation
(Barooah–Hespanha, Karp); identifiability on graphs
(Freris–Graham–Kumar). The protocol-coupled closed-loop result, the
ranking inversion, the quantized-winding metastability account, and
the frontier separation appear in none of these.

**The one prior protocol comparison, engaged head-on.** Ouassal,
Rocco, Yan & Nanzer (IEEE TAP 68(7), 2020) is, to our knowledge, the
only work in this subfield comparing directed versus bidirected
update structures — for quantized *frequency* consensus, at
statistics level — and concluded both achieve consensus and high
coherent gain, i.e., that the choice is benign. Our result
contradicts the natural generalization of that conclusion: once the
actuated *phase* loop, its modulo-π branch layer, and realized
beamforming gain are included, the choice is far from benign — it
reverses which topology is optimal. Their paper is therefore a
strengthening foil, not a precedent.

**Coverage behind the novelty claim (a claim of absence needs
evidence of search).** A dedicated adversarial audit of the exact
kill-intersections — "topology selection × update schedule × phase
synchronization × distributed antenna" and "topology × consensus
protocol × coherent beamforming × airtime" — returned
validated-as-open on both the ranking-inversion and Pareto claims:
17 ledgered keyword queries (12+ informative zero-hits recorded as
coverage evidence), plus citation walks reviewing ~143 works citing
the five nearest papers. The consensus-theory literature optimizes
topology *per algorithm* (Xiao–Boyd and gossip lines) but never
compares rankings *across* algorithms. Full query ledger:
`lit_review/ledger.json`.

## 9b. What the nearest formulations structurally cannot represent

A dedicated three-line gap analysis (each paper read in full where
accessible; every claimed omission verified against the text, not
assumed) found the same **five structural exclusions** recurring
across every neighboring framework — and our results live precisely
in the intersection those exclusions leave open.

**The five exclusions.** (i) *Phases modeled on the real line, not
the circle* — so branch ambiguity, winding quantization, and the
metastability barrier have no representation. (ii) *Estimation
without actuation* — corrections never feed back into future
measurements, so closed-loop instability cannot exist at any
parameter setting. (iii) *Per-algorithm formulation* — the update
protocol is not a variable, so cross-protocol ranking is not a
well-formed question. (iv) *Variance objectives* — amplitude-blind,
so the gain/MSE topology divergence (up to 8.6 points) does not
exist. (v) *No resource axis* — airtime absent, so the
coherence–airtime frontier and its budget-dependent optimum are
outside scope.

**Verified per line, with the texts' own words.**
- *Larsson (2401.11730)* is explicit about both boundary choices:
  "we can ignore the mod 2π operation" (small-error assumption), and
  its Section II-D *proves* adding measurements never worsens
  estimation — "beamforming accuracy… is always better when all
  measurements are used." True for estimation; measurably false for
  the actuated loop, where one added serviced chord halves a chain's
  gain (73.2→16.9%) and the framework's own Rayleigh-monotonicity
  ordering (complete ≻ ring ≻ chain) reverses outright: measured
  22.3% / 16.9% / 73.2%. Its line-topology unbounded-variance theorem
  is correct and matches our companion-testbed measurement.
- *Ngo–Larsson (2509.03722)* is the most complete estimation-side
  treatment and *does* handle ordinary 2π wrapping in its tracker (a
  correction to our earlier characterization); its boundaries are
  Assumption 3 (inter-station channels known — which removes the
  two-way half-difference and with it the entire two-element branch
  layer) and compensation-in-the-precoder rather than oscillator
  actuation. Notably, its "denser calibration can be worse" result is
  a *different, complementary mechanism* to ours — drift accumulated
  during longer measurement windows (estimation-side) versus cycles
  and simultaneous-update divergence (actuation-side); a complete
  design framework needs both, and neither appears in the other's
  model.
- *Rashid–Nanzer (2201.08931)* validates by drawing its error terms
  from assumed statistics — the channel enters as a uniform random
  phase constant, never as a wrapped measurement chain — so the
  anti-phase fixed point (zero innovation while transmitters cancel;
  our measured 3009 mrad "lock") is unexhibitable by construction,
  and its derived connectivity-helps monotonicity is the exact
  statement the actuated loop reverses. Its own observation that
  *faster* updating can worsen residuals is the statistics-level
  shadow of our exchange-noise-injection result — credited as such.
- *Ouassal et al. (TAP 2020)* — the subfield's only
  directed-vs-bidirected comparison — reached its "both work"
  conclusion for *frequency* consensus without an actuated phase
  loop; one layer up, the same choice is the decisive variable
  (23.7% vs 92.2% on the same graph). Its hardware demonstration of
  feedback-free directed structure is direct implementability
  precedent *for our winning protocol*.
- *Shandi et al. (2405.18384, 2410.17356)* tolerate extreme link
  sparsity for *time* consensus (one random link per iteration) —
  but random-link scheduling has no per-link cadence accounting, and
  the expected service interval it induces sits exactly in the
  regime (m ≳ 2) where actuated *phase* loops destabilize; sparser
  is reported as slower, never as cheaper, so the Pareto tradeoff is
  unposed. Their picosecond two-way machinery is the strongest
  hardware in the subfield and the feasibility base for our proposed
  experiment.
- *The theory frameworks* (Barooah–Hespanha resistance; FGK
  identifiability; Xiao–Boyd fastest averaging; Average TimeSynch):
  real-valued, open-loop, per-algorithm, variance-objective,
  resource-free — respectively excluding the winding layer, the
  attainability question (identifiable ≠ achievable: the same
  identifiable star delivers 23.7% or 99.4%), the cross-protocol
  question, the amplitude-weighted objective, and the frontier.
  Consensus convergence proofs guarantee agreement but are silent on
  *which sector* the protocol converges into — a wound state is a
  consensus fixed point that destroys the beam.

**The honesty cut.** Each framework's positive content survives in
our results inside its domain: the resistance law (validated on 5/6
graphs, correlations +0.85–0.97), the identifiability machinery (our
mixed-graph theorem is FGK applied to our observation model), the
Jacobi/Gauss–Seidel spectral ranking (pre-registered, ranked our
protocols correctly, and its damping prescription cures the star
exactly as predicted), and the drift-during-measurement density cost
(real, complementary to ours). The gaps are not errors in these
works; they are the boundaries of what their formulations were built
to ask.

## 10. Verification and reproducibility

Pre-registered predictions per mechanism (one partially falsified —
cadence severity is protocol-dependent — and reported as such); the
fork's simultaneous arm reproduces the motivating study bit-identically;
chain cells agree bit-identically across campaigns; every theorem
carries a numerical verification suite (10⁵ configurations, 35k steps,
8k flips); the winding quantization was measured independently in two
testbeds before being proved; all run caches
(`dirA2_cache.json`, `dirA3_cache.json`) and scripts are in
`topology_selection/`. Two of our own conjectures (flip
cycle-neutrality; local-unwinding impossibility) were refuted by our
own verification and the claims corrected before external review.

## 11. Limits and the next step

Simulation only. N ≤ 16 for the fork; one deployment geometry; static
channel; the directed protocol's advantage presumes an elected root
(re-election dynamics and root failure not analyzed; the alternating
protocol's chromatic-index rate cost is intrinsic). The result is
hardware-testable at small scale — 4–8 software-defined radios, star
and tree topologies, simultaneous versus directed updates, with the
ranking inversion as the single target observable
(star+simultaneous ≪ star+directed) — and the published four-to-six
node SDR synchronization systems demonstrate the regime is practical.
