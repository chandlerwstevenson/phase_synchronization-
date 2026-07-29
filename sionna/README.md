# SDR-like OTA synchronization with NVIDIA Sionna

This directory ports the one-way master/slave synchronization loop in
`../reg_python` to sampled complex IQ. The default simulation is built around
Sionna PHY's 3GPP TR 38.901 tapped-delay-line channel, time-domain channel
application, and complex AWGN blocks. The original ideal tone-plus-AWGN model is
still available for comparison.

## What the SDR model includes

- A framed preamble with a repeated short training field and CP-protected long
  Zadoff-Chu training fields.
- Packet detection, sample timing acquisition, wide-range coarse CFO, fine CFO,
  and phase estimation from received IQ only.
- Independent master/slave oscillator phase and frequency random walks.
- Sionna `TDL` channel snapshots with fractional-delay multipath and optional
  Doppler, converted to discrete taps with `cir_to_time_channel` and applied by
  `ApplyTimeChannel`.
- Sionna complex `AWGN` at a specified received SNR.
- A shared reference crystal per node: carrier CFO, sample-clock offset, and
  frame-timing drift all derive from the same fractional reference error and
  co-drift with it. NCO corrections are digital and never touch the physical
  clock. A fixed independent SFO can still be forced for experiments.
- Frame arrival drift that accumulates with the clock error: whole-sample
  drift steps the arrival inside the re-centered capture window and the
  sub-sample residual is applied as a fractional delay.
- Power-law oscillator noise: white PM sample jitter, a continuous white-FM
  random walk (the walk observed inside each frame carries forward through the
  dead time and becomes part of the true oscillator state), and flicker FM
  approximated by a bank of log-spaced first-order Gauss-Markov processes.
- Temporally correlated log-normal shadowing on top of the TDL multipath, with
  a fixed thermal noise floor so fading changes the actual received SNR
  (`snr_db` defines the link SNR at nominal channel gain).
- PA limiting, DAC/ADC quantization, AGC, IQ gain/phase imbalance, clipping,
  timing jitter, and receiver DC offset.
- An iterated EKF and quantized phase/frequency corrections, similar to an SDR
  NCO control loop. Corrections load into the NCO after a configurable
  processing latency (default one sync interval); the controller
  forward-predicts each command by its known latency, as a real design would.

The default profile is a stationary 915 MHz TDL-D line-of-sight channel, 1 MS/s,
1500 Hz initial CFO, 10 ppm sample-clock error, 12-bit converters, and 20 dB SNR.

## Coherent collaboration modes

Two architectures sit on top of the synchronization machinery, matching the
two ways base stations can collaborate coherently:

- **CSI-aided (closed-loop) joint transmission** — `python simulation.py
  --csi-gain`. A user's channel estimates absorb any static per-station phase
  bias, including the one-way channel-phase bias, so the achievable
  two-station combining gain depends only on differential phase drift between
  CSI refreshes. Reported as gain vs. refresh cadence, evaluated from the
  one-way run's closed-loop oscillator residual.
- **Open-loop coherence for passive detection** — `python simulation.py
  --model twoway`. With no user feedback, the inter-station channel phase
  must be removed before the carriers are truly aligned at the antennas. A
  reciprocal two-way exchange (both frames traverse the same channel
  realization and shadowing state) cancels it: the half-difference of the
  forward and reverse phase measurements observes the pure oscillator
  offset. The report includes the two-station open-loop coherent gain
  cos^2(residual/2). Caveats modeled honestly: the half-difference carries a
  global pi ambiguity (resolved in hardware by a one-time combining check),
  and residual TX/RX chain asymmetry (`twoway_chain_asymmetry_deg`) does not
  cancel and must come from loopback calibration.

## Two-tier micro-pilot synchronization

The closed-loop floor of the plain two-way loop is set by correction
cadence, not SNR: the LO walk accumulated over the dead time between frames
plus frequency uncertainty propagated over correction staleness.
`ota_sync/microsync.py` therefore adds a second tier: the full two-way
frame (detection, timing, CFO) runs once per interval, while short
reciprocal phase-only micro-pilots (a CP-protected ZC-255, ~0.3 ms,
estimated in tracking mode with no detection preamble) are exchanged at
evenly spaced sub-intervals, with corrections issued every sub-interval.
`python simulation.py --model micro --micro-pilots N`. Measured trade at
defaults (steady-state residual vs. airtime): 71 mrad at 19% (N=0), 35 at
23%, 28 at 26% (N=4), 23 at 35% (N=9) -- about 3x tighter lock than the
plain loop for ~7% more airtime. The global pi ambiguity of two-way
acquisition is resolved by a modeled one-time combining calibration (a
single coarse power check after lock flips the NCO by pi if the pair
combines destructively); steady-state metrics are masked to start after
that calibration.

## Hybrid one-way/two-way calibration (`hybrid_calibration/`)

A joint 3-state EKF over [oscillator phase, frequency, channel phase]
separates fast power-law clock noise from slow channel dynamics: cheap
one-way pilots observe the sum at high cadence, sparse reciprocal anchors
(every K intervals) observe the components separately and re-pin the
split. `python simulation.py --model hybrid --anchor-every K`. On a static
channel the residual is flat from K=1 to K=20 (~33 mrad) while airtime
falls to 13.5% -- reciprocity cadence is set by channel coherence, not
oscillator quality. See `hybrid_calibration/README.md`.

## Comparison with Rashid & Nanzer (IEEE TWC 2023)

`ota_sync/dfpc.py` implements their DFPC and KF-DFPC algorithms twice: a
faithful statistics-level reproduction of their Algorithms 1-2 (N nodes,
Metropolis-Hastings mixing, their error model, validated against their
Eq. 27 bound), and a two-node adaptation running their consensus update
rule over this repository's physical layer. `python simulation.py --model
compare` runs the head-to-head under identical conditions. Findings at
defaults: applied naively (their channel-free measurement assumption), the
wrapped symmetric consensus is bistable over a real channel and can capture
at anti-phase (residual pi, ~0% coherent gain); with a reciprocity
correction using their assumed side channel, DFPC reaches ~153 mrad
(unfiltered measurement noise feeds the NCO directly) and KF-DFPC ~83 mrad,
statistically identical to our two-way EKF (~82 mrad) -- reproducing their
claim that filtering helps, while quantifying the physical-layer penalty
their statistics-level model cannot see.

## Run

Python 3.11 or newer and PyTorch 2.9.1 or newer are required by Sionna 2.

```bash
python -m pip install -r requirements.txt
python simulation.py
```

Plot phase synchronization, request the full radio diagnostics, or change radio
conditions:

```bash
python simulation.py --plot
python simulation.py --plot-all
python simulation.py --snr-db 8 --cfo-hz 10000 --tdl-model C
python simulation.py --speed-mps 1.0 --sfo-ppm 20 --adc-bits 10
python simulation.py --device cuda --iterations 500
```

Disable the custom RF impairments while retaining the Sionna multipath channel:

```bash
python simulation.py --no-rf-impairments
```

Run the original idealized model with:

```bash
python simulation.py --model ideal --pilot-length 500
```

Tests:

```bash
python -m pip install '.[test]'
python -m pytest
```

## Signal flow

```text
master clock/NCO
  -> STF + LTF IQ frame
  -> PA + DAC
  -> Sionna 3GPP TDL multipath + Sionna AWGN
  -> slave LO + sample-clock offset + phase noise
  -> IQ imbalance + AGC + ADC
  -> packet detection and timing
  -> coarse CFO -> fine CFO -> phase
  -> iterated EKF -> quantized slave NCO correction
```

## Important physical interpretation

A one-way receiver observes `oscillator phase + channel phase`; it cannot identify
those two terms separately without another reference, channel calibration, or a
two-way protocol. The program therefore reports both:

- **OTA phase residual:** what the SDR can measure and drive toward zero.
- **Raw oscillator phase residual:** the simulated clock-only error, shown to
  expose the channel-phase bias rather than hiding it with oracle information.

## Ground-truth metrics

Every phase and frequency error metric is computed against a ground-truth
reference, never against the receiver's own noisy measurement. Each capture is
also passed through an oracle path that shares the frame, channel realization,
deterministic transmit hardware (PA limiting, DAC quantization), and sample
clock offset, but carries no AWGN, LO phase noise, IQ imbalance, DC offset, or
ADC quantization. Running the same synchronizer on the oracle capture yields
the true observable OTA phase and frequency. Reported RMSE values therefore
include estimator error from noise, phase noise, quantization, and analog
impairments; at the default settings, the LO phase-noise random walk (not
AWGN) dominates the per-measurement phase error. The EKF measurement and
process covariances model both the AWGN Cramer-Rao bound and the phase-noise
contribution, as a receiver designed from the LO's data-sheet spec would.

Because the LO white-FM walk continues through the dead time between frames,
the walk accumulated over one full sync interval (about 45 mrad RMS at the
default 2e-4 rad-per-sample noise, 1 MS/s, and 50 ms interval) is
unpredictable by any controller and sets the floor of the closed-loop phase
residual. Reaching a tighter phase lock requires a quieter oscillator, a
shorter sync interval, or both. With a nonzero correction latency, the
reported post-correction residual is simply the true observable phase at each
capture, which already reflects every command that has physically loaded into
the NCO, including the drift accumulated while waiting.

For a stationary channel, frequency synchronization remains identifiable and the
channel-phase bias is stable. Under mobility, channel Doppler and phase evolution
become part of the measured OTA loop, as they would on hardware.

The coarse CFO range is approximately
`+/- sample_rate / (2 * short_sequence_length)` Hz. TDL channel snapshots are
sampled once per synchronization interval, so select mobility and interval values
that avoid aliasing the channel Doppler.
