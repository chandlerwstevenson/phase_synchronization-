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
