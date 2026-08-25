# RESULTS A4 — the winding conservation theorems (what "robust to local update rules" is allowed to mean)

Task: the external review softened our claim that the winding failure
sector is "robust to any local update rule" to "persists across the
tested protocols," and invited an upgrade if the invariance could be
proved. This deliverable proves exactly what is provable, refutes two
things we ourselves conjectured along the way, and states the upgraded
sentence with its precise hypothesis. Derivations and the runnable
verification suite: `dirA4_winding_theorem.py` (pure mathematics +
synthetic verification; nothing here depends on the RF simulator, so
none of it can be a configuration artifact).

## The four theorems, in plain language

**1 — Winding is exactly quantized, always.** For any cycle in the
measurement graph, the sum of wrapped pairwise phase offsets around it
is an integer multiple of 2π for *every* phase configuration — not
just settled ones. (The unwrapped differences telescope to exactly
zero; wrapping subtracts whole turns.) This is the phase-unwrapping
residue identity applied to sync graphs. Practical corollary: any
*instantaneous* measured cycle sum must be exactly 2πk — so the one
"cycle sum 5.4 rad" cell reported in RESULTS_A2 cannot be a wrapped sum
of simultaneous physical offsets; it is necessarily a time average or
a sum of estimator states (flagged for that file, one sentence).

**2 — Winding is conserved by all sub-boundary dynamics.** The winding
number changes only when some cycle edge's phase difference crosses
±π. Continuous evolution that never sends an edge through ±π conserves
it exactly; a discrete update step conserves it whenever each cycle
edge moves by less than its current distance to the boundary
(|Δd| < π − |wrap(d)| per edge). This is protocol-independent: *any*
update rule — simultaneous, sequential, directed, or one not yet
invented — conserves the winding sector while its corrections respect
that bound. The hypothesis is a step-size condition, not universality.

**3 — What a π branch flip does (our own conjecture refuted).** We had
conjectured node π-flips cannot change winding ("flips are cut
vectors, cuts ⟂ cycles"). That argument is valid for unwrapped sums —
which are identically zero — and **false** for the wrapped sum: the
exact rule, proved and verified on 8,295 generic random flips with
zero mismatches, is

  Δw = −½ (sgn r₊ + sgn r₋),

where r₊, r₋ are the wrapped differences on the two cycle edges at the
flipped node — *independent of the flip's sign*. A flip changes
winding exactly when both adjacent differences share a sign, and is
cycle-neutral when they don't. This is the entry mechanism: from
near-lock, residual signs are random, and the measured entry rate
matches the sign-agreement probability to three decimals (0.467 =
0.467, 20,000 trials).

**4 — Wound states are metastable; every escape transits anti-lock.**
In a settled wound state all cycle differences share the winding's
sign (verified), so an *unwinding flip exists at every node* — the
hoped-for impossibility statement ("local corrections cannot unwind")
is also false. What is true, and is the real content: unwinding by
flip throws the two adjacent edges to error magnitude π − 2π|w|/n (the
anti-lock scale; measured mean transients 1.65–2.67 rad match the
formula per cell), and unwinding continuously requires an edge to slip
through ±π. Combined with Theorem 2: **every sector transition,
discrete or continuous, under any protocol, transits an
anti-lock-scale event on at least one cycle edge.** That barrier — not
protocol invariance per se — is why wound states persist in practice.

## Verification summary (all in `dirA4_winding_theorem.py`)

| suite | check | result |
|---|---|---|
| V1 | quantization, 10⁵ random configs, rings n=3–12 | worst deviation from integer 8.4×10⁻¹⁵ |
| V2a | 400 continuous paths, 2,620 winding jumps | every jump matches signed boundary-crossing bookkeeping, 0 mismatches |
| V2b | 120,000 random-walk steps | bound held on 35,683 steps → **0** winding changes; 46,217 changes occurred only on bound-breaking steps |
| V3 | 8,295 generic on-cycle π-flips (both signs) + off-cycle flips | flip formula exact, 0 mismatches; off-cycle flips never change winding |
| V4 | wound rings (n, w) ∈ {(6,1),(8,1),(12,1),(8,2),(12,2)} | all same-sign ✓; unwinding flip at every node ✓; transient magnitude matches π − 2π|w|/n (e.g. 2.387 vs 2.356 at n=8, w=1) |
| V5 | 20,000 flips from near-lock | winding-entry rate = adjacent-sign-agreement rate exactly (0.467 = 0.467) |

Cross-check on recorded data: campaign 4 of the companion testbed
stored only branch signatures, not phase vectors, so its "±1.00
windings to 3 decimals" could not be recomputed here; it is consistent
with Theorem 1 (which requires exact quantization of any instantaneous
wrapped sum), and the one non-integer cycle sum on record (5.4 rad,
RESULTS_A2) is explained above as necessarily non-instantaneous.

## The upgraded sentence the paper may now use

Replacing "the failure sector persists across the tested local update
protocols":

> "The winding number is exactly quantized at every instant and is
> invariant under any update dynamics — continuous or discrete, under
> any protocol — whose per-step corrections keep each cycle edge's
> phase difference away from the ±π boundary; entering or leaving a
> wound sector requires an anti-lock event (an edge slipping through
> ±π, or a π branch correction applied where both adjacent residuals
> share a sign, Δw = −½(sgn r₊ + sgn r₋)), so settled wound states are
> metastable under every local protocol rather than specific to the
> tested ones."

Terminology note adopted from the review: winding constraints live in
the measurement graph's **cycle space**; branch flips form a
**cut-space** coset — and Theorem 3 is the precise statement of how the
two layers interact once wrapping is included: the orthogonality that
holds unwrapped does *not* survive wrapping, which is exactly why
flips can wind.

## Honest notes

- Two conjectures from our own side were refuted en route (flip
  cycle-neutrality; local-unwinding impossibility) — the surviving
  theorems are sharper for it.
- The step-size hypothesis in Theorem 2 is checkable per protocol:
  directed-tree "correct fully toward parent" steps can be large
  during acquisition, which is consistent with sector changes being
  observed there only in acquisition-phase churn.
- All verification is synthetic (exact arithmetic on phases); the
  connection to the RF system is through RESULTS_A2's measured
  winding occupancy, which these theorems explain rather than re-measure.
