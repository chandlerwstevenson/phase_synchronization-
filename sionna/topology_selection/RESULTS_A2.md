# Direction A2 — the three-law fork experiment

Question (from external review of direction A): do the three
topology-stability mechanisms survive a change of control law, or are
they properties of simultaneous symmetric updates? Per the review, the
title claim "stability, not information, governs synchronization
topology" is only earned if they survive; if they vanish under
sequential updates, the narrower claim is "topology and update
protocol are coupled."

Setup: `dirA2_threelaw.py`. Same physics, geometry (uniform, frozen),
seeds 0–2, 150 intervals, pilot 255/64, airtime accounting, EKFs, and
branch check as direction A; only the correction law differs:

- **symmetric** — direction A's exact law (simultaneous ±c/2).
  Continuity verified bit-identical (MST B=7 s0: 85.70% = 85.70%),
  and the full cadence row reproduces RESULTS_A exactly.
- **alternating** — Gauss-Seidel turn-taking generalized by proper
  edge coloring; off-slot corrections forward-predicted to their slot.
- **directed** — elected-root tree; corrections applied fully to the
  child (trees only).
- **symmetric-dw** — control row: canonical degree-weighted
  simultaneous (±c/(2·deg)), star only.

Predictions P1–P3 and the linearized spectral radii were registered
and printed before any run (`--part predict`; they are in the module
docstring). Zero-CFO control cells run at the sparsest cadence.

## Verdict table (mechanism × law; gain % mean±std, 3 seeds)

| mechanism | symmetric | alternating | directed | verdict vs prediction |
|---|---|---|---|---|
| **P1 cadence ceiling** (MST, m=1→3.5) | 84→69→48→45→38 (flips 45→91) | 68→42→39→28→27 (flips 32→90) | **96→94→94→87→80 (flips 3→9)** | **PARTIALLY FALSIFIED** — predicted "survives all laws unchanged"; survives symmetric and alternating, but directed softens the ceiling drastically (80% at m=3.5; zero-CFO control 84%, so not an initial-offset artifact) |
| **P2 degree ceiling** (star hub-7 vs trees, B=7) | star 23.7±0.9 (113 flips) vs MST 83.9±6.8 | star 38.9±2.7 (98 flips) — recovers only partially | star **92.2±10.2** (3 flips); symmetric-dw control **92.2±10.3** (identical per seed) | **CONFIRMED with a sharpening** — vanishes under directed AND under damped simultaneous; sequencing alone does NOT rescue it (see coupling below) |
| **P3 cycle exclusion** (chain vs ring/chorded, all edges serviced) | chain 73.2±16.9 vs ring 16.9±11.1, mst2c 21.6, complete 22.3 | chain 73.2±16.9 (bit-identical to symmetric — the parity stagger IS the path's 2-coloring, an implementation cross-check) vs ring 38.3±3.5, mst2c 34.5, complete 29.0 | trees only (moot by construction) | **CONFIRMED** — the chord penalty survives sequencing (ring still 35 points below chain); winding states occupied under both laws, including a locked quiet winding (cycle sum 5.4 rad, 9 flips, 42.7% gain) |

## The earned title (per the review's menu)

Not "stability, not information" and not "everything is
protocol-coupled." The three mechanisms have **three different
characters**:

1. **Cadence** — a physical driver (frequency walk across the service
   gap; persists at zero CFO) whose *catastrophic expression* is
   protocol-dependent: bidirectional laws amplify a wrong branch pick
   into cross-edge flip cascades (45–99 flips/run); the directed law
   contains the damage to one subtree (3–12 flips) and coasts to 80%
   at m=3.5.
2. **Degree** — a numerical artifact of *undamped simultaneous*
   updates specifically. It vanishes under direction (92%) and under
   degree damping (92%, per-seed identical to directed on the star),
   but **sequencing alone fails to fix it**: edge-colored turn-taking
   on a degree-7 hub forces each hub edge to correct every 7th slot,
   which re-enters the cadence mechanism (38.9%, 98 flips). For a
   high-degree node, simultaneous → divergence, sequential → rate
   dilution; only unidirectional or damped control escapes both.
3. **Cycles** — a topological invariant, robust to protocol, exactly
   as the cut-space theory predicts (windings are locally stable
   sectors under any local update rule).

**Bonus finding (not pre-registered): the topology ranking inverts
across laws, and only the directed law scales.** At N=16, m=1, every
bidirectional law collapses on every topology (13–20% gain, 113–242
flips — flip cascades scale with N), while directed holds MST at
89.3±1.6 and the star at **99.4±0.0**. The star — the *worst*
topology under symmetric — is the *best* under directed (depth-1
correction paths). Topology selection and control law are not
separable design choices.

## Linearized-theory column (registered ex ante)

Spectral radii predicted star-symmetric divergence (1.107 > 1) and
star-alternating linear stability (0.944) — the collapse and the
partial recovery match. But linear stability is necessary, not
sufficient: alternating-complete is linearly nilpotent (radius 0.000)
yet measures 29.0%, because the binding nonlinearity is the mod-π
branch layer interacting with correction rate — which the small-angle
analysis cannot see. Jacobi/Gauss-Seidel intuition ranks the laws
correctly but does not predict magnitudes.

## Honest notes

- The alternating arm's per-edge correction rate is 1/(chromatic
  index) by construction — the intrinsic cost of turn-taking, not an
  implementation accident (on chains, where the chromatic index is 2
  and matches the symmetric law's parity stagger, the two laws agree
  bit-for-bit).
- My degree-capped tree instance (Kruskal with cap 3) is a different
  graph from direction A's ad-hoc degree-3 tree (theirs measured
  80.2±19.5, mine 44.9±3.3 under symmetric); the star and MST
  continuity anchors are exact, so nothing rests on that instance.
- One seed (s2) drags the star down to 77.7% identically under both
  directed and symmetric-dw — an acquisition-phase event upstream of
  the law choice, not a law effect.
- N=8/16, one geometry, 3 seeds (2 at N=16), static channel,
  simulation only.

Files: `dirA2_threelaw.py` (predictions + radii + runner + campaign,
cache `dirA2_cache.json`, 118 cells), figures
`figures/dirA2_cadence.png`, `figures/dirA2_degree.png`,
`figures/dirA2_cycle.png` (plain, inspected).
