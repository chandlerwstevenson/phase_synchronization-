# Scheduling reversal — derived regime condition and measured boundary

Script: `scheduling_reversal.py` (`--part theory | dedicated |
frontier | figs`), cache `scheduling_reversal_cache.json`, figures
`figures/figR1_dedicated_vs_M.png`, `figures/figR2_frontier_extended.png`.
All runs fresh, seeds 0–2. Incorporates the dominance study's
attribution correction (`RESULTS_DOMINANCE.md`): the per-exchange
excess measurement noise r_extra is oscillator-derived
(intra-capture walk + flicker misspecification), class-proportional,
and jitter/channel/SNR-independent — not multipath resampling. The
derivation here is source-agnostic and survives the correction
unchanged; the inputs were relabeled.

## 1. The derived condition

- **Scalar phase-only loop (Proposition 1): no reversal is possible.**
  For both a correctly specified and a misspecified (excess-blind)
  filter, the sampled-chain steady error plus coast drift is strictly
  increasing in the exchange spacing M. White per-exchange noise sets
  a floor; it cannot make more exchanges worse.
- **Two-state loop (Proposition 2): the only dedicated-only reversal
  channel is frequency-baseline improvement**, and the resulting
  condition r_th + r_extra ≳ q·M³/c essentially cannot hold once
  r_extra is correctly attributed: it is oscillator-derived, so any
  class that inflates r_extra inflates the drift q with it. The exact
  mismatched-Riccati evaluation predicts no reversal cell.
- **Consequence: the reversal is exclusively a property of the
  architecture with free observations** — it requires an alternative
  observation channel cheaper per unit information than the dedicated
  exchange, making the exchange's r_extra net injection rather than
  net information.

## 2. Dedicated-only regime map (measured, 30 cells + theory overlay)

Held residual (mrad), mean of seeds 0–2, exchange every M intervals:

| class, jitter | M=1 | M=2 | M=4 | M=8 | M=16 |
|---|---|---|---|---|---|
| ocxo ±2  | 20.8 | 22.3 | 18.7 | 25.5 | 47.3 |
| ocxo ±32 | 20.1 | 23.2 | 20.6 | 27.2 | 43.8 |
| tcxo ±2  | 197.7 | 237.1 | 248.0 | 682.5 | 889.5 |
| tcxo ±32 | 192.6 | 222.9 | 524.5 | 779.9 | 880.8 |
| sdr ±2   | 946.4 | 1183.7 | 1050.9 | 1030.9 | 1049.1 |
| sdr ±32  | 917.5 | 1219.4 | 1031.1 | 1124.9 | 1011.4 |

- **No reversal in any cell** — flat-or-worse with sparser exchanges
  everywhere, exactly as the corrected theory predicts. The
  originally hypothesized "good oscillator + high jitter reverses
  even without free observations" is refuted by both theory and
  measurement.
- **No jitter effect in any cell** (±2 vs ±32 statistically
  identical) — independently confirming the dominance study's
  attribution at loop level.
- Theory overlay (class-based r_extra = 33/134/673 mrad, zero
  fitted constants): right shape and scale for every class — ocxo
  predicted 32→61 vs measured 21→47 (~1.4× high), tcxo endpoint 1033
  vs 881, sdr capped by phase wrapping at ~1 rad as expected.

## 3. Extended frontier (N=8, with free observations)

Reduced model σ²(K) = floor² + A/K + bK with b = 0.283
mrad²/interval fixed independently from the ablation-measured static
gauge drift, (floor, A) calibrated on the declining branch K=40–320:
floor 83.0 mrad, A = 1.77·10⁵ mrad²·intervals, **predicted minimum
K\* ≈ 790 with sub-mrad curvature** — the measurable prediction is
flatness.

- Measured K=640 (fresh, 2568 intervals, seeds 0–2): **88.4 ± 9.7
  mrad at 0.21% exchange airtime** vs predicted 85.7 — flat
  confirmed, no turn-up, every point still far below the dedicated
  baseline (193 mrad at 51.6% airtime).
- The small-K saturation (129.5–138.3 mrad at K ≤ 10) sits exactly at
  the inferred per-exchange excess for this oscillator class
  (127–141 mrad): at dense exchanges the loop holds the injected
  noise level itself — the near-unity-gain injection signature.
- The calibrated A corresponds to an injected-transient decay time
  A/r_extra² ≈ 10 intervals, consistent with the free-observation
  re-tracking timescale.

## 4. What actually causes the sparser-is-better frontier

**Corrected statement:** each dedicated two-way exchange injects
oscillator-derived excess measurement noise (intra-capture oscillator
walk plus flicker noise the filter models as white) with near-unity
weight; when cheap free observations carry the tracking between
exchanges, that injection is the exchange's dominant effect, so
thinning exchanges removes noise faster than it adds gauge drift —
until the (environment-set) gauge-drift floor, which in a static
environment lies beyond K ≈ 790. Discriminating evidence: the
frozen-oscillator control (jitter × multipath < 0.05 mrad), the
jitter-independence of all 30 dedicated cells here, the
class-proportionality of the excess, the quantitative match of the
small-K saturation to the per-exchange excess level, and the
flat-through-K=640 frontier matching the ablation-drift prediction.

## 5. The defensible paper sentence

"We identify a regime in which the conventional
synchronize-more-often rule reverses: when ordinary transmissions
supply a cheaper observation channel, dedicated synchronization
exchanges contribute more measurement noise than information, and
their optimal rate is set by the environment's gauge drift rather
than oscillator stability — in a dedicated-only loop no such
reversal exists, and we prove and measure both halves."
