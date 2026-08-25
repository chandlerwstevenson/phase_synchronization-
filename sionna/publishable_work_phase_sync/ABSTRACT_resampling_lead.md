# RETRACTED — do not use (see correction note below)

**Status 2026-08-24 (same day): this framing is withdrawn. Our own
control experiments refuted its central mechanism within hours of
writing it.** Two independent studies
(`../phase_sync_idea/RESULTS_DOMINANCE.md`,
`../phase_sync_idea/resampling_law.py`) established:

- Sample-timing jitter re-sampling the multipath contributes
  **< 0.05 mrad**, not ~100 mrad: the discrete channel is
  shift-invariant and the correlator tracks integer sample shifts
  exactly. The jitter knob is irrelevant at any setting.
- A real but much smaller related mechanism does exist —
  *clock-offset-induced fractional* resampling — worth **10–25 mrad**,
  set by the delay-separated diffuse power (not the naive Rice bound),
  with a validated zero-fit predictor. It is **not white** at
  realistic clock offsets (lag-1 correlation up to +0.95); whiteness
  requires the per-exchange alignment step to exceed the waveform's
  ambiguity width.
- The ~130 mrad loop-level excess that motivated this abstract is
  **oscillator noise**, not channel noise: it scales with oscillator
  class (28 / 130 / 650 mrad for oven-controlled / temperature-
  compensated / cheap) and is independent of channel, SNR, and jitter.
  Flicker-FM modeled as white is the prime suspect. The "153 mrad
  matches the 1/(2K) Rice estimate" agreement was a coincidence.

What survives for publication: (i) a previously unmodeled 10–25 mrad
measurement term with derived whiteness *conditions* and an exact
predictor; (ii) a real class-proportional 1.3–1.7× filter
overconfidence (model-misspecification finding); (iii) the
scheduling-reversal result, whose measurement always stood and whose
cause is now correctly attributed (see
`../phase_sync_idea/RESULTS_reversal.md`). The text below is kept only
as a record of the retracted framing.

---

# Abstract — resampling-noise-led framing (RETRACTED 2026-08-24)

**The Missing Term in the Distributed-Array Synchronization Budget:
Multipath Resampling Noise**

Error budgets for over-the-air synchronization of distributed
coherent arrays decompose the residual phase error into oscillator
drift, thermal estimation noise, and correction latency — and design
every synchronization schedule accordingly. We show these budgets are
missing their dominant term. In any realistic deployment, each
synchronization capture arrives with sample-timing jitter, and that
jitter re-samples the static multipath composite differently on every
capture: the channel itself, though frozen, is measured through a
different lens each time. The result is a white, per-exchange phase
noise — approximately 100 mrad per capture on a standard
line-of-sight channel model — that exceeds the oscillator and thermal
terms combined, and whose magnitude is predictable from the channel's
diffuse power fraction with no fitted parameters. The term carries a
falsifiable signature: a Kalman tracking filter built from the
textbook budget is systematically overconfident, with its
true-to-believed error ratio matching √(b² + σ_r²)/b across phase
budgets b to within a few percent. Two design consequences follow,
both measured in waveform-level simulation with full oscillator and
RF impairments and both inverting standard practice. First, because
every dedicated synchronization exchange injects this noise,
synchronizing *less often* improves accuracy: thinning two-way
exchanges by 160× reduces residual error from 129 to 86 mrad while
cutting synchronization airtime from 67% to 0.4% of the channel.
Second, once ordinary data and sensing transmissions carry the
tracking between rare exchanges, the required exchange rate is set by
the propagation environment's coherence time rather than oscillator
quality — sustaining 55–108 mrad of residual error from 2 to 64
stations on 0.5–30% airtime, where every conventional scheme tested
is incoherent or channel-saturated at 64. The mechanism is distinct
from the slowly-varying multipath bias known in satellite-navigation
carrier phase, and from asynchronous-medium-sampling noise in optical
time transfer; neither its dominance in array synchronization, its
predictive signature, nor its consequences for synchronization
scheduling appear in the existing literature.

---

Provenance: audit verdict 7/10, all four sub-claims standing
(`../lit_review/ledger.json`, claim `resampling-floor`). Must-cite
and distinguish: Sinclair et al., PRA 99.023844 (optical
asynchronous-sampling reciprocity noise — the mechanism's cousin in a
different domain); the Nanzer-line exchange-cadence result (sparser
is better for *frequency* via time baseline — distinct from our
*phase* mechanism); GNSS carrier-multipath literature (colored
Gauss-Markov bias, not white per-capture noise); MDPI 2026
discrete-sampling blind zones (deterministic, not stochastic).
Numbers: ~100 mrad/capture and flat-in-N (98/93/90 mrad at
N=6/10/14) from the resampling diagnostics; σ_r ≈ 153 mrad from the
1/(2·Rice-factor) diffuse-power estimate; overconfidence
×1.45/1.18/1.12 at budgets 200/314/600 mrad; frontier
129.5→86.0 mrad at 66.9%→0.42% airtime (phase_sync_idea experiment
C); scaling from experiment A. Simulation only; no hardware.
