# Experiments B and D — environment coherence time, and the U-curve test

Both experiments run fresh in this folder (`experiment_b_d.py`, cache
`experiment_b_d_cache.json`; figures via `figures_b_d.py`, plain
default matplotlib). Two stations, real random-data OFDM observations,
full oscillator and RF impairments. All numbers below are mean ± std
over seeds.

## Experiment B — residual error vs environment coherence time

Setup: environment motion swept from 0 to 1 m/s (coherence time
T_c = 0.423/f_D, with f_D the Doppler frequency at 915 MHz), two
anchor cadences (one two-way exchange every K = 10 or 40 intervals),
seeds 0–2, run length max(60, 4K) intervals, observation rate 5 per
interval.

| motion (m/s) | T_c (s) | K=10 residual (mrad) | K=40 residual (mrad) |
|---|---|---|---|
| 0 (static) | ∞ | 58.5 ± 16.3 | 54.9 ± 13.2 |
| 0.005 | 27.7 | 65.4 ± 17.4 | 98.3 ± 13.8 |
| 0.01 | 13.9 | 74.4 ± 17.0 | 171.4 ± 19.9 |
| 0.02 | 6.9 | 98.6 ± 9.0 | 306.6 ± 23.7 |
| 0.05 | 2.8 | 196.5 ± 18.4 | 781.2 ± 33.9 |
| 0.1 | 1.4 | 389.0 ± 38.3 | 956.3 ± 63.7 |
| 0.2 | 0.7 | 764.0 ± 20.7 | 1000.9 ± 32.5 |
| 0.5 | 0.28 | 896.8 ± 38.7 | 931.9 ± 25.8 |
| 1.0 | 0.14 | 898.9 ± 29.1 | 926.0 ± 7.9 |

**The three regimes are real and sit where the theory says.** The
observability analysis predicts the locked misattribution bias
π·f_D·T·K reaches the phase budget (0.314 rad) at T_c* = 2.12 s for
K=10 and T_c* = 8.46 s for K=40 (equivalently 0.066 and 0.016 m/s).
Measured: K=10 crosses budget-scale error between 0.05 and 0.1 m/s;
K=40 crosses between 0.01 and 0.02 m/s — both boundaries land inside
one grid step of the prediction, with no fitted constants.

- **Free regime** (T_c far above the boundary): residual sits at the
  static level (~55–58 mrad) and the measured bias is small.
- **Biased regime** (T_c near the boundary): the residual is
  dominated by a locked offset — measured bias ≈ measured rms (e.g.
  K=40 at 0.02 m/s: bias 256 of 307 mrad rms) — exactly the
  misattribution signature, growing in proportion to speed × K.
- **Broken regime** (T_c below the boundary): the residual saturates
  near the wrapped-uniform level (~0.9–1.0 rad) and the "bias"
  becomes meaningless (wrapped), i.e. the oscillator/channel split is
  lost, as the theory declares.

Figure: `figures/expB_residual_vs_coherence_time.png` (log-log; solid
lines = measurement, dotted horizontals = static levels, dashed
verticals = the two theory boundaries).

## Experiment D — the U-curve hypothesis test

The external reviewer predicted a U-shaped curve of error vs
free-observation rate in a slowly varying environment (too few
observations → oscillator drift; too many → channel evolution
misread as oscillator motion). Our observability theory predicts the
opposite: the per-observation leakage into the channel state shrinks
as the rate grows, so the motion bias is rate-independent and there
is no U.

Setup: observation rate n ∈ {1, 2, 5, 10, 20} per interval ×
environment motion {static, 0.02, 0.05, 0.1 m/s}, K = 40, five seeds,
320 intervals per run with the first four anchor cycles (160
intervals) excluded from the statistic — so this is a steady-state
measurement by construction, immune to the acquisition-transient
artifact that produced the original (retracted) interior-optimum
report.

Residual (mrad rms) and locked bias (mrad), mean ± std over 5 seeds:

| motion | n=1 | n=2 | n=5 | n=10 | n=20 |
|---|---|---|---|---|---|
| static, rms | 138.7 ± 21 | 85.1 ± 15 | 59.1 ± 19 | 47.5 ± 19 | 41.7 ± 21 |
| static, bias | 20.9 | 17.5 | 14.4 | 14.3 | 10.3 |
| 0.02 m/s, rms | 486 ± 204 | 434 ± 145 | 424 ± 144 | 455 ± 211 | 418 ± 143 |
| 0.02 m/s, bias | 327 | 297 | 290 | 323 | 316 |
| 0.05 m/s, rms | 821 ± 65 | 803 ± 62 | 795 ± 62 | 841 ± 86 | 791 ± 61 |
| 0.05 m/s, bias | 663 | 652 | 648 | 617 | 644 |
| 0.1 m/s, rms | 982 ± 40 | 1005 ± 49 | 1000 ± 49 | 993 ± 39 | 1001 ± 46 |
| 0.1 m/s, bias | 193 | 252 | 227 | 238 | 217 (wrapped; split lost) |

**Verdict: no U-curve, at any environment speed.**

- Static: monotone decreasing, 139 → 42 mrad from n=1 to n=20 —
  re-confirming (now at 5 seeds with a strictly post-convergence
  statistic) that the old interior optimum was an artifact.
- Slowly varying (0.02, 0.05 m/s): the curve is *flat* — the locked
  bias is independent of observation rate to within seed noise
  (0.02 m/s: 290–327 mrad across a 20× rate range), and the residual
  neither improves nor degrades meaningfully with more observations.
  More free observations do not hurt; they simply cannot fix a floor
  that anchoring cadence and environment motion set.
- Fast (0.1 m/s): flat at the wrapped saturation level — broken
  regime, rate irrelevant.

So the reviewer's proposed mechanism ("channel evolution gets
interpreted as oscillator motion *more* when there are more
observations") is refuted by measurement, and our theory's
prediction (bias set by speed × anchor spacing, not by observation
rate) is confirmed. The correct principle, supported by both
experiments together:

**The value of opportunistic observations is bounded by the
environment's coherence time through the anchor cadence — not
through the observation rate.** The tradeoff the reviewer intuited
exists, but it lives on the anchor-spacing axis (experiment B's
boundary), not the observation-rate axis (experiment D's flat lines).

Figure: `figures/expD_residual_vs_observation_rate.png`.
