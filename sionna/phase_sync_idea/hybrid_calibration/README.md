# Hybrid one-way/two-way OTA phase calibration

Two-way exchanges pay double airtime purely to cancel the channel phase --
but for fixed stations the channel phase is nearly static while the
oscillators wander fast. The two processes have different spectral
signatures (power-law clock noise vs. slow channel dynamics), so a joint
estimator can separate them:

- A **3-state EKF** tracks the relative oscillator phase, relative
  frequency, and the link channel phase.
- Cheap **one-way** pilots (a full frame once per interval for timing/CFO,
  plus phase-only micro-pilots at sub-intervals) observe the *sum*
  oscillator + channel phase at high cadence and carry the oscillator
  tracking.
- Occasional **two-way anchors** (reciprocal full-frame exchanges every K
  intervals) observe the oscillator and channel phases *separately* via the
  half-difference / half-sum, re-pinning the split.

How often reciprocity must be paid for becomes a question about channel
coherence time (the channel-phase process noise), not oscillator quality.

Run from the repository root:

```bash
python simulation.py --model hybrid --micro-pilots 4 --anchor-every 5
python simulation.py --model compare      # includes the hybrid row
```

The global pi ambiguity of two-way acquisition is resolved by the same
modeled one-time combining calibration as the other reciprocal loops; on a
flip, the channel-phase state is shifted by -pi so the sum stays consistent
with the one-way observations.

## Validation (60 intervals, seed 0, defaults otherwise)

Static channel: residual is flat (~32-34 mrad) from K=1 to K=20 while
airtime falls 22.6% -> 13.5%; the fully two-way control sits at 27.9 mrad
and 26.0%. Under channel Doppler the decoupling inverts exactly as the
theory requires -- required anchor cadence follows channel coherence:

| speed | fD | scheme (K, prior)            | residual | gain  |
|------:|----:|------------------------------|---------:|------:|
| 0.2   |0.61| control (two-way)            |  43 mrad | 99.95%|
| 0.2   |0.61| hybrid K=5, static prior     | 446 mrad | 95.3% |
| 0.2   |0.61| hybrid K=5, matched prior    | 235 mrad | 98.7% |
| 0.5   |1.52| hybrid K=1, static prior     | 128 mrad | 99.6% |
| 0.5   |1.52| hybrid K=1, matched prior    |  44 mrad | 99.95%|
| 0.5   |1.52| hybrid K=5, matched prior    | 208 mrad | 98.9% |

The two-way control is immune (channel-free measurements every substep).
The one-way full-frame frequency observation is biased by the LOS Doppler;
a fourth state (channel Doppler rate) is the natural next refinement.
