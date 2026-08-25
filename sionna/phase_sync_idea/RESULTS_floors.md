# The floor budget of one open-loop two-way sync link

Scripts: `openloop_floors.py` (derivations in the module docstring,
predictors, harness, study); full output `openloop_floors_full.log`;
raw numbers `openloop_floors_results.json`. All runs: frozen
oscillator states (every number is per-exchange measurement floor,
not tracking error), 120 exchanges × seeds 0–2, predictions printed
before measurements, discriminating controls per term. Zero fitted
constants anywhere.

## Headline

**The complete per-exchange error budget of an open-loop two-way RF
carrier-sync link — six derived terms — verified additive against
all-on runs at eight grid points spanning oscillator class ×
environment speed × turnaround gap × exchange cadence × SNR, with
measured-to-predicted ratios 0.89–1.00.**

## Term 1 — channel nonreciprocity (the new term)

The two directions of an exchange sample the channel a turnaround gap
δt apart; the half-difference only cancels the part that didn't
change. Derived and verified:

- **Bias** = π·cos(π/4)·f_D·δt (the line-of-sight Doppler ramp across
  the gap; cos(π/4) is Sionna's LOS arrival angle, verified). Matches
  measurement to ≤2% at every cell:

  | v (m/s) | δt (ms) | predicted (mrad) | measured |
  |---|---|---|---|
  | 1 | 1 | −6.8 | −6.8 |
  | 3 | 1 | −20.3 | −20.4 |
  | 10 | 1 | −67.8 | −66.1 |
  | 30 | 1 | −203.4 | −199.6 |
  | 3 | 0.1 | −2.0 | −2.0 |
  | 3 | 10 | −203.4 | −200.3 |
  | 1 | 10 | −67.8 | −67.5 |

- **Crucially, the standard CFO correction cannot remove this bias**:
  the channel Doppler enters both directions' frequency estimates
  with the same sign and cancels out of the measured CFO
  ((f_fwd − f_rev)/2), so the receiver's turnaround correction is
  blind to it. Demonstrated: the all-on runs (correction active)
  carry the full predicted bias (−207.8 measured vs −203.4 predicted
  at 3 m/s, 10 ms).
- **Spread**: diffuse-decorrelation closed form
  σ² = (1 − J₀(2π·f_D·δt))/(4·K_eff) is uniformly ~1.5× low (it
  averages away the realization's tap weighting); the
  realization-exact predictor (phase of the summed taps, same channel
  draw, zero fits) matches measured spread at every cell (e.g. 8.9 vs
  7.9, 24.8 vs 22.9, 77.0 vs 69.0, 79.6 vs 86.0 mrad).
- **Controls**: machinery null (v=0) 0.08 mrad; same-taps-to-reverse
  control at v=3 gives 0.24 mrad — the effect comes entirely from the
  δt tap shift.
- Scope: the stock simulator holds taps static across the gap (a
  documented simplification); this module built the finer-grained
  channel to measure what that hides. Within-capture Doppler remains
  unmodeled.

## Term 2 — fractional resampling vs exchange cadence (law refined)

The sample-clock carry sweeps the receiver's fractional alignment
through the multipath composite. The refined law, exactly verified:

- **Magnitude is cadence-independent** (~5.7 mrad on this
  realization): every cadence sweeps the same ±half-sample alignment
  range, so the long-run spread equals the alignment profile's total
  spread regardless of step size. (This corrects the earlier
  "magnitude and color both depend on cadence" expectation — only
  color does.)
- **Color is cadence-controlled**: the per-exchange alignment step
  (∝ cadence × clock offset) sets the correlation structure, moving
  from strongly positively correlated (sawtooth) at small steps to
  anticorrelated at large ones. Predicted vs measured, all six
  cadences: std 5.7/5.7, 5.7/5.7, 5.6/5.6, 5.6/5.6, 5.6/5.6, 5.6/5.6
  mrad; lag-1 correlation +0.47/+0.47, +0.07/+0.08, −0.28/−0.28,
  −0.30/−0.29, −0.26/−0.25, −0.33/−0.31.
- The predictor is the noiseless alignment-phase profile of the same
  channel draw evaluated at the exact carry sequence both directions
  visit — the estimator itself defines the profile, no constants.

## Term 3 — intra-capture oscillator walk

Analytic approximation σ ≈ σ_pn·√(L/3)·√2/2 sits within 10% of the
numerically exact component (actual estimator on walk-only captures);
fresh-seed measurements match the exact predictor to 4–8%:
ocxo 1.6 vs 1.7 mrad, tcxo 16.2 vs 17.4, sdr 80.8 vs 86.9. This term
scales with oscillator class and dominates the budget for cheap
oscillators — consistent with (and refining) the earlier corrected
attribution of the "resampling floor."

## Terms 4–6 — turnaround walk, CFO-correction residue, thermal

Derived closed forms (module docstring): turnaround walk
√(σ_pn²·f_s·δt/2); CFO-correction residue (σ_f/√2)·δt/2 from the
ex-ante measurement covariance; thermal √(1/(4·SNR·L)) — 0.78 mrad
predicted at 20 dB vs 0.8 measured previously. Verified inside the
assembled budget below.

## Assembled budget — additivity verdict

σ_total² = quadrature sum of terms 1b–6, bias from term 1a. Eight
all-on grid points (everything enabled: moving fine-grained channel,
1500 Hz CFO, turnaround advance + per-class walk + CFO correction,
class LO noise, thermal):

| class | v | δt (ms) | M | SNR | measured bias/std | predicted | ratio |
|---|---|---|---|---|---|---|---|
| tcxo | 0 | 1 | 1 | 20 | −1.3 / 33.2 | 0 / 33.2 | 1.00 |
| tcxo | 3 | 1 | 1 | 40 | −20.7 / 31.8 | −20.3 / 33.9 | 0.94 |
| tcxo | 3 | 10 | 1 | 40 | −207.8 / 79.5 | −203.4 / 84.4 | 0.94 |
| tcxo | 0 | 1 | 8 | 20 | −1.1 / 31.8 | 0 / 33.1 | 0.96 |
| ocxo | 3 | 10 | 1 | 40 | −203.2 / 64.2 | −203.4 / 72.0 | 0.89 |
| sdr | 0 | 1 | 1 | 20 | −5.9 / 104.6 | 0 / 111.0 | 0.94 |
| tcxo | 3 | 10 | 8 | 20 | −208.3 / 82.9 | −203.4 / 84.3 | 0.98 |
| ocxo | 0 | 0.1 | 1 | 40 | −0.1 / 25.3 | 0 / 25.2 | 1.00 |

**Additivity verified to ≤11% (mean |1−ratio| ≈ 5%), with the
predictions slightly conservative-high in most cells.**

## Honest misses and notes

- The ensemble closed form for term 1's spread is ~1.5× low
  everywhere; only the realization-exact predictor is quantitative.
  The closed form's *bias* half is exact.
- Term 1's spread shows mild negative lag-1 correlation (−0.1 to
  −0.4) not modeled by either predictor; magnitude unaffected.
- An earlier draft of the term-2 predictor used total diffuse power
  (the naive Rice bound) and over-predicted by up to 190× — the same
  mistake the resampling-law history warns about (only
  delay-separated diffuse power matters at these delay spreads). It
  was caught by the quick-run validation and replaced with the exact
  profile predictor before any full run.
- **Tooling fact siblings should know**: Sionna channel realizations
  are pinned by `sionna.phy.config.seed`, NOT by
  `torch.manual_seed` — without setting it, every TDL construction
  draws a fresh channel and any "seed-matched" comparison is silently
  comparing different realizations (this produced large spurious
  predictor-measurement gaps before being found and fixed; any study
  that constructs its own TDL objects should check for the same bug).
- Everything here is per-exchange measurement floor with frozen
  oscillator states; closed-loop tracking behavior is out of scope by
  design.

## What this supports, stated carefully

For open-loop distributed carrier sync, the per-exchange floor is now
a six-term budget in which every term is computable before deployment
from the waveform, the link budget, the oscillator datasheet, and two
channel parameters (Doppler and the delay-separated diffuse
fraction) — and the new nonreciprocity term gives a closed-form,
CFO-correction-immune bias π·cos(θ_LOS)·f_D·δt that ties the
achievable open-loop accuracy directly to the product of environment
speed and turnaround gap. This is the RF carrier-phase analog of
two-way time transfer's nonreciprocity and the optical
asynchronous-sampling result, with a validated formula; the
literature sibling owns the novelty verdict.
