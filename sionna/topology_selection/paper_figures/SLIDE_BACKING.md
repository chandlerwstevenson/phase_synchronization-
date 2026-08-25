# Slide-by-slide backing document

Every number here was read directly from the run caches
(`dirA2_cache.json`, `dirA3_cache.json`, `dirA5_cache.json`) on
2026-08-25. Nothing is quoted from older summaries.

Each slide gets a verdict:

- **SUPPORTED** — the data backs what the slide says.
- **NEEDS CORRECTION** — the slide says something the data does not.
- **BACKGROUND** — no data needed; checked for accuracy only.

"Gain" always means coherent gain: the fraction of the ideal beam
power the array actually delivers, from 0 to 100%. "Tree" means the
minimum-variance spanning tree unless said otherwise. Read the
MUST-FIX list first.

---

## MUST-FIX before presenting

**1. Slide 11 uses the wrong figure for the reversal claim.**
`fig1_inversion_n8.png` shows N=8 at full airtime. At that operating
point there is no reversal: the tree is best under every protocol
(directed tree 96.3%, directed star 92.2%). What fig1 does show is
huge protocol gaps — the star goes from 23.7% to 92.2% just by
changing the protocol. Present fig1 as the protocol-gap picture.
Present the reversal with fig7 (next item).

**2. Slide 12 says simultaneous updates "gave the opposite ordering"
at N=16. That is false.** At N=16 under simultaneous updates the star
and the tree both collapse to about 16%. That is a tie, not an
opposite ordering.

**The fix for both slides is the same figure: `fig7_reversal.png`.**
It holds everything fixed — 8 stations, 8.6% of airtime spent on
synchronization, the uniform station layout — and pairs the seeds, so
both topologies see the same placement, clocks, and channel. Result,
over 20 paired seeds:

- Simultaneous protocol: tree 32.4%, star 23.9%. The tree wins 16 of
  20 seeds (p = 0.012).
- Directed protocol: star 92.2%, tree 82.2%. The star wins 18 of 20
  seeds (p = 0.0004).

Same size, same airtime, and the best topology switches with the
protocol. Both directions are statistically significant.

**3. Say where the reversal lives: the uniform station layout.** We
reran the same test overnight on a clustered layout (details in the
next section). The simultaneous half repeated. The directed half did
not — star and tree tied. So on slides 11, 12, 33, and 40, say the
reversal is shown "at N=8, 8.6% airtime, uniform layout." Do not
present "the directed protocol prefers the star" as a general law.

Small number updates used throughout: on the cadence sweep, branch
flips run 30–99 per run for the two-way protocols and 2–12 for
directed (older text said 32–91 and 3–9). The star under simultaneous
updates flips 109–117 times per run.

---

## Overnight runs (finished 2026-08-25, predictions written first)

Three run sets. The predictions were written into the docstring of
`../overnight_runs.py` before anything ran. One prediction missed;
it is reported first.

**A. Clustered-layout repeat of the reversal — half worked, half did
not. The prediction (full reversal) was wrong.** Same operating point
as fig7, new clustered station layout, 10 paired seeds:

- Simultaneous: tree 27.0%, star 21.9%. Tree wins 8 of 10 (p = 0.11).
  Same direction as before, not significant at 10 seeds.
- Directed: star 73.7%, tree 73.5%. Star wins 5 of 10. **A tie. This
  half did not repeat.**

What did carry over: the protocol still dominates (directed lifts
both topologies from roughly 25% to roughly 74% at the same airtime),
and the flip counts still separate cleanly (92–115 per run under
simultaneous, 1–7 under directed). What did not carry over: the
directed protocol's preference for the star. A possible reason,
untested: in a clustered layout, both graphs must route through the
same few long links between clusters, and those weak links may set
the error no matter what shape the graph is. Bottom line for the
talk: the reversal is real and significant on the uniform layout, and
it is an operating-point result, not a universal one. Saying so
costs nothing — the core claim only needs one solid example.

**B. N=16 extended from 2 seeds to 5 — prediction held.** Directed
star 97.1% (four of five seeds land at 99.4–99.5; the new seed 4 came
in at 87.8). Directed tree 90.2%. Star above tree in 4 of 5 paired
seeds. Every two-way-protocol cell stays collapsed, between 13.4%
and 19.9%.

**C. Chain and ring extended from 3 seeds to 10 — prediction held.**
Chain: 72.1% under both two-way protocols. Ring (the same chain plus
one extra link, which closes a loop): 24.7% under simultaneous, 49.7%
under alternating. One extra measurement link costs 47 points of
mean gain under simultaneous updates. The ring's big spread is real:
runs either settle clean or settle twisted. Gain tracks the twist —
the only ring runs above 85% gain have loop phase sums under 0.11
radians, and every run with a loop sum near ±5 radians stays at or
below 42.7%.

---

## Slide 1 — title. BACKGROUND

The claim "the best topology can change when the update protocol
changes" is supported. Say "can change," not "always changes." At
generous airtime the ranking happens to stay put (slide 31 covers
this).

## Slide 2 — coherent gain. SUPPORTED

The formula on the slide is exactly what every experiment computes,
from the final settled window of each run. The slide's distinction is
the project's core point: the array needs the transmitters actually
aligned, not merely well estimated. Slide 26 proves the difference
with data — a case with perfect linear estimation and 29% gain.

## Slide 3 — the synchronization graph. SUPPORTED

Use `fig4_topologies.png`. Those five graphs are drawn by the
experiments' own graph-building code at the true station coordinates.
The star's hub is station 3, with 7 links. If someone asks why the
chain criss-crosses: it connects stations in index order, and that is
faithful to what was simulated.

## Slide 4 — what one link measures. SUPPORTED

The two-way exchange gives each pair their phase difference, but only
up to half a turn. The reason is clean: the measurement sees twice
the offset, wrapped, and doubling maps two different offsets to the
same reading. So there are always exactly two candidate answers, at
any signal quality. That is why one bit per link is exactly enough to
fix it — not roughly enough.

## Slide 5 — the hidden failure. SUPPORTED

The wrong choice between those two answers is not just possible. It
is stable: the loop settles there, its own error signals read zero,
and the transmitters cancel. We measured this — an anti-phase lock at
0.7% gain that reported itself as locked. That is a measured fact,
not a hypothetical.

## Slide 6 — the conventional view. BACKGROUND

Fair and documented. The estimation papers prove "more measurements
never hurt," which is true for estimation and measurably false inside
the loop (slide 19). One key paper states outright that it ignores
the phase wrap-around. Quotes are in the technical summary if
challenged.

## Slide 7 — the closed-loop pivot. BACKGROUND

Accurate framing. No numbers to check.

## Slide 8 — the three protocols. SUPPORTED

The descriptions match the code. Simultaneous: every link corrects in
the same interval. Sequential: links take turns so no station updates
two links at once; a 7-link hub then serves each link only every
seventh slot. Directed: corrections flow one way, toward an elected
root station. Airtime is equalized in every comparison; one exchange
costs 4.28% of an interval. Use `fig5_protocols.png`.

## Slide 9 — what is simulated. SUPPORTED

The list matches the code: real pilot waveforms, multipath channel,
noise, timing jitter, hardware imperfections, transmit/receive
turnaround, quantized corrections, and oscillator noise matched to a
datasheet. 8 stations for the main runs, 16 for scaling. The closing
point is true and worth saying: no failure is injected by hand;
everything emerges from the physics.

## Slide 10 — the motivating experiment. SUPPORTED

The selector that picked links to maximize predicted estimation
quality lost to plain spanning trees at every tested airtime, in both
station layouts. Loop stability was the binding constraint, not
estimation quality. Figures `dirA_gain_vs_budget_uniform.png` and
`_clustered.png`. Keep it framed as motivation.

## Slide 11 — headline result. NEEDS CORRECTION — see MUST-FIX 1

Every number on the slide is individually correct. The logic is not:
at full airtime the tree stays best under every protocol, so there is
no reversal in fig1. Use fig7 for the reversal and keep fig1 as the
protocol-gap picture.

## Slide 12 — scaling. NEEDS CORRECTION — see MUST-FIX 2

Verified numbers at N=16, now 5 seeds: directed star 97.1%, directed
tree 90.2%, star ahead in 4 of 5 paired seeds. Every two-way cell is
collapsed (13–20%). So the slide can honestly say: at N=16 only the
directed protocol still works, and it prefers the star. It cannot say
simultaneous gave the opposite ordering — that is a tie. The full
reversal belongs to fig7 (N=8, 8.6% airtime, uniform layout).

## Slide 13 — mechanism 1: service rate. SUPPORTED

Verified sweep on the tree, gain from most-frequent to least-frequent
service: simultaneous 84% → 38%, alternating 68% → 27%, directed
96% → 80%. Every protocol degrades once a link waits more than about
two correction cycles between measurements; the clocks drift in the
gap. That part is physics. How badly each protocol pays is the
protocol's doing. Figure `dirA2_cadence.png`.

## Slide 14 — the controls behind it. SUPPORTED

Two controls make the "physics" label stick. Longer pilots (4× and
8×) change nothing, so per-measurement noise is not the driver.
Removing the fixed frequency offsets changes nothing either, so the
driver is the random frequency wander between services.

## Slide 15 — drift becomes flips. SUPPORTED

Updated counts: 30–99 branch flips per run for the two-way protocols
on the sweep, versus 2–12 for directed. The chain on the slide is the
mechanism as measured: a long gap lets the phase wander past a
quarter turn, the loop picks the wrong one of its two candidate
answers, and the error spreads. The directed tree confines each flip
to the stations below it.

## Slide 16 — mechanism 2: the crowded hub. SUPPORTED

Star under simultaneous updates: 23.7%, with 109–117 flips per run.
The linear analysis predicted this before the runs: seven corrections
fighting at one hub make the update loop unstable (growth factor
1.107, above 1). Figure `dirA2_degree.png`.

## Slide 17 — the linear theory calls it. SUPPORTED

Damping the updates: 92.2%. Directing them: 92.2%. The two cures give
nearly identical results seed by seed, which is strong evidence the
linear mechanism is the right one — both work by making the same
growth factor drop below 1. Keep the slide's caveat that linear
theory does not explain everything; slide 26 is the proof.

## Slide 18 — why taking turns fails. SUPPORTED

Sequential star: 38.9%. The reason is arithmetic. Taking turns at a
7-link hub means each link is served every seventh slot — far past
the two-cycle limit from slide 13. Fixing the crowding problem
re-creates the service-rate problem. This coupling is the slide set's
most important subtle point, and the data supports it exactly.

## Slide 19 — mechanism 3: loops. SUPPORTED (updated to 10 seeds)

Chain: 72.1% under both two-way protocols, identical seed by seed — a
useful determinism check. Add one link to close the loop and gain
falls to 24.7% (simultaneous) or 49.7% (alternating). The ring's wide
spread is the mechanism itself: some seeds settle twisted, some
settle clean (see slide 21). Figure `dirA2_cycle.png`, regenerated
with the 10-seed data. If asked: the complete graph gives 22.3% and
tree-plus-two-chords 21.6% — every graph with loops is bad, not just
the ring.

## Slide 20 — the twist is quantized. SUPPORTED

The derivation on the slide is the actual proof: walk the phase
differences around a loop and the wrap-arounds force the total to a
whole number of turns. Checked numerically on 100,000 random
configurations; worst deviation from a whole number was 8.4×10⁻¹⁵.
It holds at every instant, not just at the end.

## Slide 21 — the twist is a hidden state. SUPPORTED

Measured 21 distinct settled twist values across 24 seeds, whole
numbers to ±0.001 — found in an independent second testbed before the
proof existed. Twisted states are locked: the loop's error signals
read zero while the beam is destroyed. New supporting data from the
10-seed ring runs: every run above 85% gain has a loop sum under 0.11
radians; every run with a loop sum near ±5 radians stays at or below
42.7%.

## Slide 22 — conserved, not constant. SUPPORTED

As long as each per-step correction stays away from the wrap point,
the twist cannot change: verified in 35,683 of 35,683 tested steps.
When a step does cross the wrap point, the change follows the signed
crossing rule: all 2,620 deliberate crossings matched. Mention the
history honestly: we first conjectured the twist could never change
and refuted our own conjecture. That strengthens the story.

## Slide 23 — why twisted states persist. SUPPORTED

Escape requires one link's error to swing past a threshold that
shrinks as the twist grows: π − 2π|w|/n. Measured escape events match
the formula. Also refuted along the way: "local updates can never
untwist" is false — an untwisting flip exists at every station. The
protection is a barrier, not impossibility.

## Slide 24 — flips and twists are linked. SUPPORTED

The flip rule Δw = −½(sgn r₊ + sgn r₋) predicted every one of 8,295
flips with zero mismatches, and it predicts the rate of falling into
twisted states exactly (0.467 predicted, 0.467 measured). The
terminology on the slide is correct and already survived a
terminology-specific review.

## Slide 25 — the three-mechanism table. SUPPORTED

The table matches the evidence. Plain row labels if the audience is
mixed: "the clocks drift between measurements" (physical), "seven
corrections fight at the hub" (numerical), "loops can trap a twisted
pattern that looks locked" (topological).

## Slide 26 — linear stability is not enough. SUPPORTED

The killer cell: sequential updates on the complete graph have a
growth factor of exactly 0 — the best possible linear score — and
still deliver only 29.0% gain. Whatever kills it is not linear. It is
the two-candidate-answers layer. Give this slide time; it is one of
the strongest.

## Slide 27 — the literature gap. BACKGROUND

Fair and documented. The three receipts: a paper that assumes away
the wrap-around, a theorem that more measurements always help
(estimation only), and one prior directed-versus-two-way comparison
that found the choice benign — for frequency, not phase. The slide's
stance — prior work answers different questions, not wrong ones — is
right. Keep it.

## Slide 28 — five things prior formulations exclude. SUPPORTED as analysis

Present as analysis of the published formulations, verified against
their texts — not as new measurements. The five exclusions match the
documented version.

## Slide 29 — the resource question. BACKGROUND

Correct framing. "More links is not free information" is exactly what
slides 18, 19, and 30 demonstrate.

## Slide 30 — gain versus airtime. SUPPORTED

At 8.6% airtime: best two-way configuration 38.0%, star/directed
87.4%. Cheapest way to hold 80% gain or better: 8.6% airtime with
directed, 29.9% with any two-way protocol — 3.5× cheaper. The three
directed curves sit above the entire two-way band at every tested
airtime. Figure `fig3_pareto_frontier.png`. One disclosed oddity if
asked: chain/directed wiggles at the two lowest airtimes; suspected
scheduling artifact; nothing rests on it.

## Slide 31 — the best choice moves with the budget. SUPPORTED

Verified: at 5% airtime the best pair is chain/directed (52.9%); at
10%, star/directed (87.4%); at 20%, tree/directed (93.8%). The reason
the star wins at tight airtime: its paths are one hop, so each
skipped service hurts less than on the tree's longer paths. That is
also exactly why the star wins the fig7 reversal.

## Slide 32 — the star, worst to best. SUPPORTED (one number updated)

7-link hub; predicted growth factor 1.107; 23.7% under simultaneous
updates. At N=16 directed: now say 97.1% over 5 seeds, four of them
above 99% (the old "99.4 ± 0.0" was 2 seeds). This is your best 60
seconds; "that is the paper in one example" is earned.

## Slide 33 — stronger than "protocol matters". SUPPORTED, scoped

The claim only needs one solid demonstration, and fig7 is it: 16 of
20 and 18 of 20, both significant, on the uniform layout. Do not
demonstrate it with full-airtime N=8 (no reversal there), the N=16
simultaneous side (tie), or the clustered layout (directed side
tied). Say where it lives; it costs nothing.

## Slide 34 — contributions. SUPPORTED, one wording change

For contribution 1 use: "shows the topology ranking can invert at
fixed array size and fixed airtime, with seed-by-seed significance."
That phrasing is bulletproof. The vaguer version invites the slide-11
objection.

## Slide 35 — verification. SUPPORTED

All checked this session: 100,000 configurations at 8.4×10⁻¹⁵;
35,683 conserved steps; 2,620 crossings; 8,295 flips, zero misses;
independent second testbed; bit-identical rerun of the simultaneous
arm. Worth adding aloud: the entire campaign was regenerated from
scratch during figure work and reproduced exactly, and the overnight
runs were pre-registered — predictions written before execution, one
of which missed and is reported.

## Slide 36 — what has NOT been shown. SUPPORTED — now stronger

Keep the list: simulation only; at most 16 stations; static channel;
root election assumed, root failure unanalyzed. Update one line: we
did test a second station layout overnight, and the directed-side
star preference did not carry over (it tied). Put that on the slide.
A scope slide that reports its own failed replication is more
credible than one that lists untested things.

## Slide 37 — hardware plan. BACKGROUND

Consistent with the documented plan. The right first observable is
the star under simultaneous versus directed — our largest and most
replicated effect.

## Slide 38 — how this could be wrong. BACKGROUND

All five risks are genuine. Number 5 — scheduling overhead eating the
airtime advantage — is the one the simulation can least rule out.
Acknowledge it if asked.

## Slide 39 — the causal picture. BACKGROUND

The chain matches the mechanism evidence end to end.

## Slide 40 — takeaway. SUPPORTED

"The best topology depends on the protocol" — supported, with fig7 as
the demonstration and its scope stated. The closing sentence — this
is a feedback system on a graph, not an estimation problem — is the
honest one-line thesis of everything measured here.

---

## The numbers, one place, all verified 2026-08-25

| quantity | value | source |
|---|---|---|
| Star, simultaneous, N=8 | 23.7 ± 1.1 % (109–117 flips) | dirA2 `deg` |
| Star, sequential, N=8 | 38.9 ± 3.3 % | dirA2 `deg` |
| Star, simultaneous + damping | 92.2 ± 12.6 % | dirA2 `deg` |
| Star, directed, N=8 | 92.2 ± 12.5 % | dirA2 `deg` |
| Tree, simultaneous, N=8 | 83.9 ± 8.3 % | dirA2 `deg`/`cad` |
| Tree, directed, N=8 | 96.3 ± 1.3 % | dirA2 `cad` |
| Service-rate sweep (tree, frequent → sparse) | 84→38 / 68→27 / 96→80 % | dirA2 `cad` |
| Chain / ring, simultaneous (10 seeds) | 72.1 ± 17.8 / 24.7 ± 25.9 % | dirA2 `cyc` |
| Chain / ring, sequential (10 seeds) | 72.1 ± 17.8 / 49.7 ± 28.6 % | dirA2 `cyc` |
| Complete graph, simultaneous | 22.3 ± 1.9 % | dirA2 `cyc` |
| Sequential on complete graph (growth factor 0) | 29.0 ± 4.1 % | dirA2 `cyc` |
| N=16 directed: star / tree (5 seeds) | 97.1 ± 5.2 / 90.2 ± 1.7 % (star wins 4/5) | dirA2 `scl` |
| N=16, all two-way cells (5 seeds) | 13.4–19.9 % per seed | dirA2 `scl` |
| Reversal, N=8, 8.6% airtime, uniform layout, 20 paired seeds | simultaneous: tree 32.4 vs star 23.9, tree wins 16/20 (p ≈ 0.012); directed: star 92.2 vs tree 82.2, star wins 18/20 (p ≈ 0.0004) | `rev`+`cad`+`par` B2 cells |
| Same test, clustered layout, 10 paired seeds (pre-registered) | simultaneous: tree 27.0 vs star 21.9, tree wins 8/10 (p = 0.11); directed: 73.7 vs 73.5, star wins 5/10 — tie, did not replicate | dirA2 `revc` |
| Cheapest airtime holding ≥80% gain | 8.6% directed vs 29.9% two-way (3.5×) | dirA3 |
| Best pair at 5 / 10 / 20% airtime | chain/dir 52.9 · star/dir 87.4 · tree/dir 93.8 % | dirA3 |
| Twist quantization | whole numbers to 8.4×10⁻¹⁵ (10⁵ configs); 21 states / 24 seeds | dirA4; second testbed |
| Conserved steps / crossings / flips | 35,683 / 2,620 / 8,295, zero mismatches | dirA4 |
| Escape barrier | π − 2π|w|/n, matched per cell | dirA4 |
| Predicted growth factors | star-simultaneous 1.107; sequential 0.944; sequential-complete 0.000 | dirA2 (written before the runs) |

Scope for everything: waveform-level simulation (Sionna channel and
noise, custom oscillator and loop code); at most 16 stations; static
channel; no hardware. The main campaign used the uniform station
layout. The reversal was also tested on the clustered layout with 10
pre-registered paired seeds: the simultaneous half repeated
directionally, the directed half tied.
