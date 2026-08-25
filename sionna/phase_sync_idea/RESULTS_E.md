# Experiment E — component ablation and the capture-length model

Scripts: `piggyback_variant_e.py` (parameterized copy of the loop,
bit-identical at defaults with a warmed calibration cache),
`ablation_study.py`, `ablation_branch_addendum.py`,
`capture_model_study.py`, `make_figs_e.py`. Data:
`ablation_results.json`, `capture_model_results.json`. Figures:
`figures/fig_e1_ablation.png`, `figures/fig_e2_capture_model.png`.
Baseline conditions: two stations, OFDM observations, anchors every
40 intervals, static channel, seeds 0–2, 160 intervals (four anchor
cycles). All numbers are worst-station residual in mrad unless noted.

## 1. The what-breaks table

| configuration | residual (mrad) | what happens |
|---|---|---|
| Full architecture | 54.7 ± 13.2 (N=2), 63.0 ± 11.6 (N=6) | baseline; beam quality 99.9% |
| No channel state (2-state filter) | **1327 ± 100** | breaks completely: the filter absorbs the propagation phase into the oscillator estimate and steers the transmitter by it; beam quality collapses to 62.6% |
| No anchors after acquisition (static) | 40–62 (does **not** break in 60 intervals) | measured gauge drift 1.7·10⁻⁷ rad²/substep — about **1000× below** the covariance-bound rate (2.0·10⁻⁴) and 100× below the filter-leakage rate (2.0·10⁻⁵); see §2 |
| No branch check, benign acquisition (offset 1.2 rad) | identical to baseline | the check never fires — with the initial offset inside ±π/2 the acquisition anchor picks the right branch |
| No branch check, adverse acquisition (offset 2.2 rad) | **3103 ± 8** (anti-phase) | **12/12 seeds lock anti-phase without the check; 0/12 with it** (34–56 mrad) |
| Dedicated-preamble observations instead of OFDM | 51.0 ± 16.7 | statistically indistinguishable from OFDM at the loop level |
| Moving environment 0.1 m/s, K=40 | 956 ± 62 | broken (wrapped-saturation regime) |
| Moving environment 0.1 m/s, K=10 | 424 ± 10, locked bias −354 ± 4 | degraded with the predicted Doppler-misattribution bias: prediction π·f_D·T·K = 479 mrad vs measured ≈ 424 (agreement ×0.89) |

Reading: the two components that carry the architecture are the
channel state (without it, the loop is off by the propagation phase —
1.3 rad here) and the branch check (without it, any acquisition
beyond ±π/2 locks the transmitters in cancellation). Anchors, in a
truly static environment, are for acquisition and branch resolution
rather than steady-state gauge maintenance (§2); in a moving
environment they are the only defense and their required rate follows
the Doppler-misattribution prediction.

## 2. The no-anchor surprise, stated carefully

The observability analysis says the oscillator/channel split is
unobservable from one-way data, and the filter's *uncertainty* about
it grows at (q_θ + q_ψ)/2 per substep. The measured *realized error*
with zero post-acquisition anchors grows ~1000× slower (ensemble of
10 seeds: variance flat at ~(40–60 mrad)² over 300 substeps). These
are consistent: the covariance bound is a worst case over channel
realizations; when the true channel really is static, the true
system never excites the unobservable direction, and the filter's
θ-gain (≈95% of each innovation) keeps attribution honest in
realization even though the filter cannot *verify* the split. The
practical consequence matches the earlier interior-optimum
correction: anchor duty in static environments is acquisition plus
convergence, not continuous gauge repair — but the loop's
self-assessed uncertainty is genuinely growing, so a deployed system
still needs sparse anchors to *know* it is locked.

## 3. The capture-length model (why the OFDM burst beats the preamble)

Model, all terms computed before comparison, nothing fitted:

    per-observation variance ≈ (1/(2·SNR) + σ_wpm²)/L_int   (thermal + white-PM)
                             + walk(L)                      (intra-capture oscillator
                                                             random walk, evaluated by
                                                             running the actual estimator
                                                             on clean waveform × walk)

Measured (preamble length sweep 127–8191, three SNRs; full-impairment
calibration, cache cleared per length — the module cache does not key
on length and silently returns stale values otherwise):

- **The U-shape is real at every SNR.** Measured optima: ~1023
  samples at 10 dB (predicted 884), ~255–511 at 20 dB (predicted
  280), ~255 at 30 dB (predicted 91, shallow floor). Short captures
  are thermal-limited; long captures are oscillator-walk-limited.
- **The zero-fit prediction tracks the measured curve** within
  ~10–25% for lengths ≥ 511 (e.g. at 20 dB: 2047 → predicted 7.97 vs
  measured 8.75 mrad; 4095 → 10.91 vs 10.21). It over-predicts at
  8191 (16.3 vs 13.4 — the estimator's frequency-fit absorbs part of
  the very long walk; the error is in the conservative direction).
  At the shortest lengths there is an unexplained excess (10.2
  measured vs 5.6 predicted at 127/20 dB) from resampling and RF
  impairments — reported as the residual, not fitted away.
- **Verdict on 4.8 vs 8.7 mrad: capture geometry, not waveform
  magic.** The OFDM burst (960 contiguous samples) sits at the
  predicted optimum and integrates its whole span (no cyclic-prefix
  or short-field overhead), measured 4.78 vs predicted 4.25 mrad at
  20 dB (within 12%). The stock 2047 preamble loses because its
  4606-sample span sits on the walk-dominated branch of the U — and a
  *length-optimized* preamble (255 → 5.83 mrad) closes most of the
  gap to OFDM. The abstract's claim survives in its corrected form:
  the ordinary data burst gives up nothing, and the reviewer's
  σ²_thermal + σ²_osc(T_capture) decomposition is confirmed
  quantitatively.
- Bonus boundary: pilot detection itself collapses for very long
  pilots at low SNR (39% detection at 8191/10 dB) — long pilots fail
  twice.

## Caveats

Single link geometry (N=2 plus an N=6 baseline spot check); the
no-anchor result is 60 intervals (3 s) — longer horizons were not
measured; the moving-environment rows use one speed (0.1 m/s); the
walk-only Monte Carlo isolates the oscillator term but the "rest"
(resampling, impairments) is only bounded empirically, not modeled.
All simulation.
