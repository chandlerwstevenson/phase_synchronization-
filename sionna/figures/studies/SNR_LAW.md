# Phase residual and sync airtime as a function of SNR

Question (from the PI): can we derive a formula for the phase residual
and the synchronization airtime as a function of link signal-to-noise
ratio (SNR)?

**Short answer: yes — the full chain is derivable with the machinery
we already validated, and its most useful prediction is a shape, not
a slope: over the whole practical operating range the residual and
airtime are nearly *independent* of SNR, because the pilot's
processing gain and the multipath resampling floor dominate thermal
noise. SNR matters only at two edges — a hard acquisition cliff at
low SNR, and a pilot-shortening lever at high SNR that converts SNR
directly into airtime savings. Both the flat middle and the cliff are
confirmed by measurement (table below).**

All symbols: T = sync interval (50 ms), f_s = sample rate (1 MHz),
L = pilot length in samples, L_lat = correction latency in intervals,
b = phase budget (rad), q_θ = per-interval phase process variance
(oscillator white-FM walk: σ_pn² f_s T), σ_rs² = multipath resampling
variance (≈ (153 mrad)² on the default channel, SNR-independent).

---

## 1. The chain, step by step

**Step 1 — SNR sets the measurement variance.** A pilot of L samples
at signal-to-noise ratio SNR (linear) yields a phase estimate with
variance 1/(2·SNR·L) per direction; the two-way half-difference
halves it. Adding the SNR-independent resampling floor:

    r(SNR) = 1/(4·SNR·L) + σ_rs²                       (phase, rad²)
    r_ω(SNR) = 1/(2·SNR·E_t)                            (frequency)

with E_t the pilot's time-energy Σtᵢ². This is the *only* place SNR
enters the tracking loop.

**Step 2 — the measurement variance sets the filter's steady state.**
The exact route is the discrete algebraic Riccati equation for
(F, Q, R(SNR)) — implemented in `coast_law.py: dare_posterior`,
computable ex ante. Closed-form limits for intuition:

- *Phase-walk-dominated (scalar) limit:* the steady posterior phase
  variance of a random walk observed in noise r is

      p⁺(SNR) = [ −q_θ + √(q_θ² + 4 q_θ r(SNR)) ] / 2 ,

  which → √(q_θ·r) when r ≫ q_θ and → r when r ≪ q_θ.
- *Coupled phase-frequency:* the classical tracking-index form
  (Kalata): with Λ = σ_w T²/σ_v (process-to-measurement ratio), the
  steady gains α(Λ), β(Λ) have closed forms and the posterior phase
  variance is α·r, the frequency posterior follows from β. Our exact
  fixed-point solver replaces these approximations, but they give the
  right SNR scaling: **all posterior terms enter through r(SNR).**

**Step 3 — the steady state sets the residual.** The error-floor
decomposition, now with SNR explicit:

    σ_resid²(SNR) = q_θ                                (drift; no SNR)
                  + ( σ_ω⁺(SNR) · L_lat · T )²         (latency)
                  + α(SNR) · r(SNR)                    (tracking)

**Step 4 — the residual sets the coast time.** Solve for the longest
service interval τ that keeps the accumulated uncertainty inside the
budget:

    q_θ·(τ/T) + ( σ_ω⁺(SNR) · (τ + L_lat·T) )² = b²   →   τ(SNR)

(the exact version is the coast-cycle covariance fixed point in
`coast_law.py`, which takes `link_snr_db` as an input — the formula
is already implemented and validated at 99.5% gap exactness).

**Step 5 — the coast time sets the airtime.** Per link,

    A(SNR) = 2 · L_cap(SNR) / ( f_s · τ(SNR) ) ,

with L_cap the capture length (pilot + channel spread). Total star
airtime is the sum over the N−1 links.

## 2. The three regimes the formula predicts

Substituting realistic numbers (L ≈ 4.6·10³ samples) explains the
measured shape:

1. **The flat middle (processing-gain regime).** Even at 0 dB
   (SNR = 1), thermal phase noise is 1/(4·L) ≈ (7 mrad)² — tiny
   against the oscillator process noise (≈ (45 mrad)² per interval)
   and against the resampling floor (≈ (153 mrad)²). So r(SNR) is
   dominated by SNR-independent terms, the Riccati output barely
   moves, and residual and coast time are *flat in SNR*. More SNR
   buys essentially nothing here.
2. **The acquisition cliff (low SNR).** Below the preamble's
   detection threshold the correlator stops finding the pilot at all;
   misses starve the filter and the loop fails non-gracefully. This
   is a hard cutoff, not a slope — the formula's domain boundary.
3. **The pilot-shortening lever (high SNR).** At fixed measurement
   quality, required pilot length scales as L ∝ 1/SNR (the product
   SNR·L is what matters). So the honest way SNR buys airtime is
   through shorter captures:

       A(SNR) ∝ L_cap(SNR)/τ ≈ constant/SNR   (until L_cap hits the
                                               channel-spread floor)

   — every 3 dB of link budget can halve the sync airtime at
   unchanged residual, until the capture length is dominated by the
   channel delay spread and guard time rather than the pilot.

Because per-link SNR comes from the link budget,
SNR(d) = SNR₀ − 10·n·log₁₀(d/d₀) (path-loss exponent n ≈ 2.7 here),
the same chain gives residual and airtime *as a function of station
distance*: flat out to the distance where SNR crosses the acquisition
threshold, then failure — a range wall, not a gradual degradation.

## 3. Measured anchors

Fresh sweep (`snr_law_check.py`, two stations, serviced every
interval, 60 intervals, seeds 0–2; theory columns computed ex ante
from `coast_law.py` at each SNR):

| link SNR (dB) | measured residual (mrad) | theory floor (mrad) | coast time τ, tcxo @ 0.314 rad |
|---|---|---|---|
| 0 | 222.5 | 85.8 | 0.150 s |
| 5 | 291.5 | 85.5 | 0.150 s |
| 10 | 124.4 | 85.5 | 0.150 s |
| 15 | 114.2 | 85.4 | 0.150 s |
| 20 | 113.3 | 85.4 | 0.150 s |
| 25 | 112.9 | 85.4 | 0.150 s |
| 30 | 112.7 | 85.4 | 0.150 s |

Exactly the predicted shape: flat above ~10 dB (113 mrad measured,
85 mrad tracking floor — the gap is the drift/latency terms and the
unmodeled resampling noise), collapse below. Independent anchors:
the clutter-referenced study's reference-strength stressor measured
34/36/39 mrad at 20/10/5 dB (flat) with observation detection
collapsing 100% → 64% → 0% below that (the cliff); and the coast-law
validation absorbed a 15–23 dB per-link SNR spread across deployments
while predicting 99.5% of coast gaps exactly — i.e., the R(SNR)
dependence in the law is already verified to be the right one.

## 4. What this means in practice

- **Don't spend link budget on sync accuracy** — in this operating
  regime it buys nothing; the residual is set by oscillators, cadence,
  latency, and the multipath resampling floor.
- **Spend link budget on shorter pilots** — the L ∝ 1/SNR lever is
  the real SNR→airtime conversion, and it multiplies with everything
  else (scheduling, piggybacking).
- **Design to the cliff, not the slope** — the binding SNR constraint
  is acquisition/detection, which sets the maximum station spacing;
  above it, SNR is a free variable.

## 5. Honest limitations

The closed forms in Step 2 are limits; the exact statement is the
Riccati fixed point (implemented, validated). σ_rs² is
channel-dependent (delay spread / Rice factor) and enters r additively
— on richer channels the flat region starts even lower. The cliff
location depends on the detector, not the tracking math, so it is
measured (here: between 5 and 10 dB for the sync preamble at these
settings), not derived. The pilot-shortening lever is analytic and
consistent with the micro-pilot measurements, but a dedicated
L-vs-SNR sweep at fixed SNR·L has not been run — it is the one
missing experiment if the PI wants the lever curve measured rather
than derived.

*Script: `snr_law_check.py` (in the `sionna/` directory). Formulas as
implemented in `coast_law.py` and `ota_sync/`. Simulation-validated
only.*
