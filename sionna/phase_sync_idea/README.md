# phase_sync_idea — the reviewer's test plan, executed

Self-contained sandbox (copies of `ota_sync/`, `hybrid_calibration/`,
`detection/`, and the sync study modules; nothing outside this folder
was touched). All experiments: real OFDM observations, full oscillator
and RF impairments, seeds 0–2 (experiment D: 5 seeds), fresh runs.
Detail per experiment: `RESULTS_A_C.md`, `RESULTS_B_D.md`,
`RESULTS_E.md`. Figures (plain matplotlib, no in-axes annotations):
`figures/`.

## Verdicts, one line each

| experiment | question | verdict |
|---|---|---|
| A — scheme × N (2→64) | does the advantage survive scale? | Opportunistic is the only scheme flat in accuracy over a 32× size change (55→108 mrad, ≥99.7% beam quality, 0.5→30% airtime); naive dedicated sync collapses at N=8, the best conventional baseline is incoherent at N=64 on 95.6% of the frame |
| B — coherence time | where does it stop working? | Three regimes (free / biased / broken) with the boundary predicted by π·f_D·T·K = budget, zero fitted constants, within one grid step at both anchor cadences |
| C — anchor frontier | did we just move the traffic into anchors? | No: cutting anchor airtime 160× (67% → 0.42%) *improves* the residual (each anchor injects multipath resampling noise); every frontier point beats the dedicated baseline on both axes |
| D — U-curve hypothesis | do more free observations eventually hurt? | **No U-curve at any environment speed** (5 seeds): static is monotone-improving, motion is flat in observation rate — the misattribution penalty is set by speed × anchor spacing, not observation rate. The reviewer's intuited tradeoff is real but lives on the anchor-spacing axis |
| E — ablations | which components are load-bearing? | The oscillator/channel decomposition (removing it: 1327 mrad — catastrophic) and the branch check under adverse acquisition (12/12 anti-phase without, 0/12 with). Waveform choice: irrelevant at loop level. Static-environment surprise: anchors are for acquisition and branch resolution — realized gauge drift runs ~1000× below the worst-case bound because a static channel never excites the unobservable direction |
| E — capture model | is 4.8-vs-8.7 mrad waveform magic? | No: a zero-fit thermal + oscillator-walk model predicts the capture-length U-shape at every SNR; the OFDM burst sits at the predicted optimal length (4.25 predicted vs 4.78 measured), and a length-optimized preamble closes most of the gap. Capture geometry, not waveform |

## The refined thesis these experiments support

Ordinary inter-station traffic can carry synchronization, but doing so
creates a latent oscillator/channel gauge problem. The decomposition is
what breaks if you skip it; sparse two-way anchors fix the gauge and
their required cadence is set by environment coherence (π·f_D·T·K < b),
not oscillator drift; free observations are monotonically beneficial in
every regime; and the observation-quality advantage of short data
bursts is a derivable capture-length optimum. At scale, this is the
difference between an architecture that is flat to 64 stations at 30%
airtime and every dedicated-signal alternative measured here.

## Known issues found during these runs

- `clutter_sync_ofdm.calibrate_oneway_noise` caches without keying on
  capture length (stale returns on any length sweep) — fixed in this
  sandbox's copy (`capture_model_study.py` works around it); the
  original in the parent directory still has it. Only affects studies
  that sweep pilot length.
- Mild non-monotonicity in the anchor frontier near K=10 coincides
  with the anchor-noise-vs-tracking crossover; unexplored.
