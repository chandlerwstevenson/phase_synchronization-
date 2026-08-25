# Dominance-region study — result: the mechanism claim is refuted

Script: `dominance_region_study.py`, full output in
`dominance_region_output.log`. Purpose was the reviewer-demanded
dominance map for the "multipath resampling" term; the mechanism
checks it required ended up overturning the claim itself.

## 1. The claimed mechanism produces ~0.25 mrad, not ~100–150

With oscillator noise frozen, integer-sample timing jitter (0 vs 32
samples) over the frozen multipath composite changes the two-way
half-difference error by less than 0.05 mrad, identically across
TDL-E/D/A (all cells 0.23–0.27 mrad). The synchronizer's
correlation-based timing recovery removes integer sample shifts, so
"jitter re-samples the multipath" contributes essentially nothing in
this simulator.

## 2. What the per-capture noise actually is

Turning on only the intra-capture oscillator random walk (tcxo value)
gives ~30 mrad per half-difference — identical with jitter 0 or 32,
and identical with the multipath removed (1 ns delay spread). The
capture-level noise is the oscillators' own walk during the capture.
Channel and jitter are irrelevant.

## 3. The loop-level excess is oscillator-side, not propagation-side

Inferred excess measurement variance (bisected so the steady Kalman
prediction matches the measured serviced-every-interval residual),
12 corner cells, seeds 0–2:

- channel D/E/A at fixed class/jitter/SNR: 127.3 / 126.3 / 127.9 mrad
  — channel-independent
- SNR 10/20/30 dB: 126.9 / 127.1 / 127.3 — SNR-independent
- jitter 2/8/32 samples: 141.0 / 91.9 / 127.3 — no jitter trend
- class ocxo/tcxo/sdr: ~28–38 / ~127–141 / ~637–709 —
  **proportional to oscillator class**

Measured/predicted loop residual is ≈1.3× for tcxo and sdr (1.7× for
ocxo) — a roughly constant multiplicative under-prediction,
consistent with unmodeled *correlated* oscillator noise (the filter
treats flicker-FM as white; flicker scales with class). The earlier
"σ_r ≈ 153 mrad matches the 1/(2K) Rice estimate" was a numerical
coincidence with the tcxo/custom-class excess scale.

## 4. Dominance map (with the excess correctly attributed)

The unmodeled term is comparable to the drift term for every class
(0.7× for tcxo/sdr, 1.5× for ocxo); it dominates the budget only for
oven-controlled oscillators. And it is not a multipath term.

## 5. Overconfidence signature, re-tested out-of-sample

Using each corner's inferred excess (from uniform-service runs) to
predict the coasting-run residual-at-service ratio √(b²+e)/b across
budgets 0.15–0.6: tracks well at small budgets (1.32 vs 1.31 at
b=0.15) but the measured ratios are non-monotone at mid budgets
(0.91 measured vs 1.04 predicted at b=0.45), and are identical for
channels D and E — again channel-independent. The white-measurement-
noise model is an approximation, not a law.

## Verdict for the paper claim

"Dominant multipath resampling noise" must be **retracted as
mechanism and as framing** — it fails every discriminating test
(jitter-independence, channel-independence, SNR-independence,
class-proportionality). What survives, honestly stated: the textbook
white-noise oscillator model under-predicts the closed-loop residual
by a class-proportional ~1.3–1.7× (the filter is genuinely
overconfident), the prime suspect is flicker-FM treated as white,
and the earlier frontier/scheduling results stand as measurements
but their "each exchange injects propagation noise" explanation is
unsupported. The one-line downgrade: from "previously unmodeled
propagation term" to "class-proportional model misspecification,
likely flicker" — an honest calibration finding, not a discovery
about the channel.
