# Research ideas and novelty map

Living document. Updated 2026-08-05 after a fresh literature pass
(see `LITERATURE_REVIEW.md` for the original 19-source verified review,
2026-07-28). Re-run searches before submitting anything — this field
publishes continuously (Nanzer group especially).

---

## 1. CONFIRMED NOVEL — the error-floor letter (primary paper)

**Claim:** closed-form steady-state phase residual of a closed-loop OTA
sync link:

    sigma^2 ~ (white-FM walk)·T_sync
            + (Kalman frequency-posterior std x correction LATENCY)^2
            + tracking terms

with latency a design parameter DISTINCT from the sync interval.
Nobody prices in the fact that corrections arrive late; our sim shows
it is the dominant term at defaults (sees 12 mrad, holds 70 mrad).

- Verified unpublished through Dec 2025 (19-source review), re-checked
  2026-08-05 against Rashid & Nanzer follow-ons: HA-DKF
  (arXiv:2302.09351) and Online-EM (arXiv:2207.11859) still have no
  actuation latency, no propagation channel, no airtime accounting.
- Near-miss to differentiate: Rashid & Nanzer TWC 2022
  (arXiv:2201.08931) Eq. 27 — five-term decomposition WITHOUT latency.
- Supporting deltas: Kalman steady-state (DARE) algebra instead of
  per-measurement CRLB; analytical flicker-FM term; impairment-complete
  waveform-level validation with oracle ground truth (theirs validates
  statistics against their own assumptions).
- Venue: IEEE WCL / LMWT letter.
- **OPEN KILL RISK:** classical delayed-feedback PLL / ADPLL loop-delay
  literature (Gardner-style) could contain an equivalent
  (frequency-uncertainty x delay)^2 term under different vocabulary.
  Searched 2026-08-05: only patents surfaced — inconclusive, NOT
  cleared. Also check IEEE 1588 servo phase-noise budgets. Must be
  resolved before writing.

## 2. RECOMMENDED NEXT PAPER — uncertainty-driven pilot scheduling

**Idea:** stop syncing every station at the same fixed cadence. Each
station's Kalman filter knows its current phase uncertainty; send a
station a pilot only when its uncertainty approaches the 18-degree
coherence line. Close/quiet stations coast; far/noisy/bad-crystal
stations get attention. The error-floor formula (idea #1) IS the
scheduling rule — it predicts how long station k can coast given its
oscillator class, link SNR, and latency.

- **Headline result to demonstrate:** same array coherence at X% less
  airtime, i.e. the airtime wall in `--sweep-stations` moves right —
  the same channel supports ~2x more stations.
- Everything needed already exists in the repo: per-link SNR from path
  loss (`ota_sync/network.py`), oscillator classes
  (`ota_sync/oscillators.py`), airtime accounting, latency modeling,
  Monte Carlo (`--seeds`). Missing: the scheduler loop itself
  (~a day) + comparison runs (fixed-rate vs adaptive).
- **Prior-art status (checked 2026-08-05):** event-triggered Kalman
  filtering exists in control theory (about saving communication of
  estimates, e.g. Automatica 2018, arXiv:1711.00493); fixed-cadence
  sync is universal in the distributed-array literature; the
  intersection — uncertainty-driven PILOT scheduling for carrier-phase
  coherence, validated at the waveform level — came up empty.
- Motivation citation: Mudumbai et al., "On the Feasibility of
  Distributed Beamforming," TWC 2007 (established the overhead wall).

**STATUS UPDATE (2026-08-05, evening): the missing experiments are
built and measured** (`ota_sync/scheduled.py` extensions +
`contention_study.py`, `airtime_wall_study.py`, `sensing_loop_study.py`,
`heterogeneous_fleet_study.py`; all opt-in, defaults regression-locked
bit-for-bit, 12 new tests):

- **Contended channel (the operating point the paper needs):** N=10,
  channel capped at 2 of 9 demanded exchanges/interval: uniform 10.5%
  gain (starved stations free-run), round-robin 84.8%, scheduled
  94.0%, genie oracle 97.8% — the threshold rule captures most of what
  perfect information could buy.
- **Airtime wall, measured per policy:** uniform's wall at exactly
  N=6 (27% gain by N=10); scheduled 98.4% at N=12 on 77% airtime.
  The "wall moves right" claim is now a curve, not a sentence.
- **Multi-fidelity servicing:** posterior-driven choice between full
  two-way frames and phase-only micro-pilots (priced at true sample
  cost): airtime 57.4% -> 10.4% at unchanged gain on a uniform star.
- **Heterogeneous fleets:** OCXO/TCXO mix — scheduler service rates
  12% vs 55% per class, 97.5% gain at 36% airtime. The dividend grows
  with fleet spread, as the coasting-time formula predicts.
- **Motion (candidate NEW TERM for idea #1):** scheduled coasting
  degrades with channel Doppler (145 -> 227 mrad at 3 m/s) while
  uniform holds — the filter's coast-time rule is missing a
  channel-decorrelation term. Same discovery shape as the latency
  term: derive it, show the corrected rule closes the gap.
- **Negative result worth a caution paragraph:** a myopic
  Whittle-style index UNDERPERFORMS the plain threshold rule under
  severe contention (64% vs 94% gain at capacity 2) — it chases
  already-blown links instead of protecting salvageable ones.
- **Sensing-in-the-loop budgets** (`budget_updates` re-targeted per
  track segment from the RT legs toward the current target
  hypothesis): 56% airtime returned at matching per-waypoint counted
  Pd — but it only TIES static-edge budgets uncontended; the
  differentiating experiment is tracking budgets + capped channel
  (both knobs now compose; not yet run).

**ENLARGED KILL RISK (must clear before writing):** the earlier
prior-art pass covered event-triggered Kalman filtering, but NOT the
Age-of-Information scheduling literature (Whittle-index freshness
scheduling, Kadota et al.) or remote-state-estimation sensor
scheduling (Shi/Sinopoli line) — both do "service the estimator whose
uncertainty grows fastest over a constrained channel" in the abstract.
The claim must be framed as the INTERSECTION: the scheduled resource
is carrier-phase coherence, the update is a physical two-way pilot
with real airtime cost, budgets are set by a sensing task's detection
utility, and validation is counted detection at waveform level. A
dedicated AoI/remote-estimation search is now the first
pre-submission task for this paper.

## 3. WEAKER BUT UNCLAIMED — anchor-cadence analysis (hybrid scheme)

**Idea:** the hybrid model's 3-state EKF makes the oscillator-vs-channel
split an explicit observability statement: cheap one-way pilots observe
only the SUM (theta + phi_c); sparse two-way anchors re-pin the split.
Derive the required anchor cadence as a function of channel coherence
time (Doppler) vs oscillator stability. The repo's Doppler experiments
already show the shape (static: K=20 anchors barely hurt; 0.2 m/s:
K=20 collapses to 1752 mrad; matched channel prior rescues it).

- No direct prior art found (BeamSync arXiv:2311.11070 and
  reciprocity-calibration papers do periodic recalibration but no
  cadence analysis vs Doppler/oscillator class).
- Probably a section in a larger paper, not standalone.
- Known defect to fix first: hybrid's one-way frequency observation is
  biased by LOS Doppler (needs a 4th state).

## 4. QUANTIFICATION ONLY — the N_max scaling law

**Idea:** one closed-form for the maximum number of stations one
channel can keep coherent: N_max(oscillator class, cadence, latency,
pilot length, coherence target). Our `--sweep-stations` measures it
(two-way at 50 ms stops fitting at N~6).

- **The CONCEPT is prior art:** Mudumbai et al. TWC 2007 already
  identified sync-overhead-vs-gain and the duplexing constraint as the
  fundamental limit. Frame any use of this as full-PHY quantification
  + the latency/oscillator-class axes, never as a new idea.

## 5. FOOTNOTE-GRADE — anti-phase capture of naive DFPC

**Finding:** Rashid & Nanzer's DFPC, run as published over a real
propagation channel, is bistable: seed 0 locks at anti-phase
(3009 mrad, 0.68% gain) because raw one-way phases include the channel
phase and the wrapped symmetric update has two fixed points.

- The MATH is known (Kuramoto-type phase consensus is multistable —
  textbook). The APPLICATION (their algorithm breaks over a real
  channel; reciprocity fixes it) is a legitimate caveat/critique —
  one paragraph or a short section in paper #1's validation story,
  not a paper.

## 6. IMPLEMENTED THROUGH N>2 — decentralized hybrid (mesh measured)

**Status:** N=2 built as `--model dhybrid`
(`hybrid_calibration/hybrid.py`, `decentralized=True`): 32 vs 33 mrad —
decentralizing the CONTROL is free at N=2. N>2 mesh built 2026-08-05
(`hybrid_calibration/mesh.py`, `--model dhybrid --stations N`): shared
oscillators, nearest-neighbor chain (tree: no winding states),
degree-weighted symmetric corrections with a side channel reporting
each node's applied deltas to its edges' filters, periodic 1-bit
branch checks at the anchor cadence (a ONE-SHOT check is insufficient —
the slow mesh convergence lets edges drift to the pi fixed point after
an early check passes; found and fixed by simulation).

**First N>2 measurements (30 intervals, seed 0, full impairments):**
the mesh CONVERGES (100% detection, no anti-phase capture), but
interior edges pay a consensus tax — each edge's correction is
under-relaxed by degree-weighting and actively disturbed by adjacent
edges' corrections: edge residuals 143/406/849 mrad at N=4, array gain
74% (N=4) and 80% (N=6), versus 99.9% for the centralized star at the
same N and airtime. THIS is the quantitative case for centralization:
control decentralization is free at N=2 and gets expensive with every
shared node. Publishable angle: the consensus tax scaling law
(residual vs node degree / chain depth), and whether smarter mesh
control (per-edge gain scheduling, alternating edge updates) can close
the gap.

**N>2 DFPC over the actual channel (2026-08-05,
`mesh.py:run_dfpc_mesh`)** — first waveform-level N-node consensus
numbers we know of (the paper's own N-node results are
statistics-level). Same harness/seed/geometry as the hybrid mesh.
N=4: DFPC 61.5% gain @ 57.4% airtime, KF-DFPC 54.5% @ 57.4%, vs
decentralized hybrid 74.0% @ 44.7%. N=6: DFPC 46.3% @ 95.6%, KF-DFPC
56.7% @ 95.6%, vs hybrid 80.0% @ 74.6%. Decentralized hybrid beats
both DFPC variants on gain AND airtime at every N tested; the
consensus tax (DFPC edges ~1 rad) is a property of symmetric control
on shared nodes, not of hybrid. Caveats: single seed (DFPC vs KF-DFPC
ordering inverts between N=4 and N=6 — within seed noise; the
hybrid-vs-DFPC gaps are much larger), chain topology, reciprocity
steelman granted to DFPC.

**THE CONSENSUS TAX IS A SCHEDULING ARTIFACT (2026-08-05, measured):**
three control laws in the same mesh harness, identical physics/seed.
Symmetric (Jacobi, DFPC-style simultaneous updates): 74%/80% array
gain at N=4/6. Alternating (Gauss-Seidel turn-taking, still fully
masterless): 99.9%/95.8%. Directed (elected-root tree, PTP-BMCA-style
— decentralized fault structure, asymmetric control): 99.9%/99.8% =
matches the centralized star. Same airtime in all cases. This
reframes the centralization argument: what costs 20-25 points of gain
is not decentralization per se but SIMULTANEOUS symmetric updates on
shared nodes — the DFPC literature's own update structure.
**Novelty check (2026-08-05): the MECHANISM is known** — asynchronous
gossip consensus is standard in WSN time sync (Average TimeSynch, PI
gossip controllers) precisely to avoid synchronized simultaneous
updates, and elected-root trees are PTP BMCA standard practice. What
appears new is the CONSEQUENCE measured in the carrier-phase
beamforming subfield: the DFPC line updates simultaneously and
validates at statistics level; nobody there has quantified the
schedule's cost in coherent gain over a real channel. Frame as a
bridging result inside a larger paper ("statistics-level validation
hid two costs: the channel [anti-phase] and the update schedule
[20-25 points of gain]") — NOT a standalone paper. Alternating's
residual gap at N=6 (95.8%) grows with chain depth — the depth
scaling law is the follow-on question. Regression-tested
(test_mesh_scheduling_beats_simultaneous_consensus).

Remaining open: broadcast amortization vs spatial reuse
accounting (star's cheap tier is O(1) transmissions vs mesh O(N); mesh
anchors can be spatially parallel, star's hub is serial); hierarchy
framing — centralized stars locally, consensus between hubs.

**Known accounting conservatism to fix before using N-station numbers
in a paper:** `ota_sync/network.py` charges hybrid's one-way pilots
per-link because links are simulated independently; physically they
broadcast, so the star's true airtime is lower than reported (hybrid's
27-station wall is an underestimate).

## 7. PROPOSED — posterior-gated membership (who is IN the array)

**Idea (2026-08-05):** subset participation driven by the sync
posterior. In a coherent array a station whose phase residual drifts
past ~90 degrees doesn't just stop helping — it SUBTRACTS from the
beam (measured here: anti-phase DFPC combines to ~0.7%). The Kalman
posterior predicts exactly when a coasting station crosses from asset
to liability, so schedule pilots AND membership from the same state:
a station that can't get airtime this interval is benched from the
coherent sum until re-synced, and the scheduler prices the bench at
N^2-vs-(N-1)^2 of array gain. Soft version: weight each station's
contribution by its expected phasor E[e^{j theta}] = e^{-sigma^2/2}
straight from the filter covariance — a scheduling-integrated form of
robust beamforming under phase error.

- **What is NOT novel (do not claim):** subset/node selection per se —
  sensor selection (Joshi & Boyd convex relaxation and descendants),
  antenna selection in MIMO, node selection + power allocation in
  distributed MIMO radar (Godrich/Petropulu/Poor), AP selection in
  cell-free massive MIMO. All select by SNR/geometry/power and assume
  a selected node WORKS; none gate on a time-varying synchronization
  posterior, and none model a member that actively harms the array.
- **The open-looking intersection:** joint pilot-airtime + membership
  scheduling from the sync posterior in an OTA carrier-phase array,
  where membership is perishable (coherence decays between pilots)
  and validation is counted detection. Adjacent literatures to clear:
  sensor selection, robust/Bayesian beamforming under phase errors
  (the e^{-sigma^2/2} weighting exists analytically there), and the
  AoI kill-risk noted in idea #2.
- **Cheap to test with what now exists:** residual matrices are
  already per-station/per-interval; gating is bookkeeping on the
  coherent sum. Experiment: all-in vs posterior-gated vs
  oracle-gated membership under the contended channel
  (`contention_study.py` regime, where stations genuinely go stale
  at radian-level residuals). Prediction: gating recovers a large
  fraction of uniform-under-contention's lost gain and finally
  separates the policies in counted Pd, not just array gain.
- Natural home: the section that completes the scheduling paper's
  triangle — budgets say what each station NEEDS, the scheduler says
  who gets AIRTIME, gating says what to do with the losers.

## NOT claims — validation machinery (never present as novelty)

- Two-way reciprocity sync itself (Nanzer group's program; SOTA:
  Merlo et al. TMTT 2025, arXiv:2506.07267)
- Two-tier micro-pilots (5G PTRS does this conceptually)
- Datasheet oscillator profiles (ocxo/tcxo/sdr), TDD turnaround with
  CFO compensation, --seeds Monte Carlo, random deployment + path loss
- Oracle-graded ground truth (a rigor point that strengthens
  validation, cite as methodology)

## Pre-submission checklist (any paper)

1. Resolve the delayed-PLL kill risk (idea #1 box above).
2. Re-run the full literature search — last full pass 2026-07-28,
   spot re-check 2026-08-05.
3. Re-check Nanzer group publications list (jeffreynanzer.com).
4. Mudumbai/Madhow and Quitin/Rahman lines: never produced verified
   claims in our review — re-search directly.
5. Everything here is simulation; frame claims accordingly (no
   hardware validation).

## Key references

- Rashid & Nanzer, IEEE TWC 2022 — arXiv:2201.08931 (near-miss, Eq. 27)
- Rashid & Nanzer, HA-DKF — arXiv:2302.09351 (checked: no latency)
- Merlo et al., TMTT 2025 — arXiv:2506.07267 (two-way SOTA)
- Mghabghab, Schlegel & Nanzer, IEEE Access 2021 (open-loop drift vs
  update interval; optimal-interval result)
- Mudumbai et al., TWC 2007 (overhead/duplexing feasibility limit)
- BeamSync — arXiv:2311.11070; array phase calibration —
  arXiv:2304.05144
- Event-triggered distributed KF — Automatica 2018
  (S0005109818300852), arXiv:1711.00493
- David & Brown, IEEE Aerospace 2015 (cabled USRP Kalman sync — never
  cite as OTA)
