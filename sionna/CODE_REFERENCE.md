# Complete code reference: what this repository does, in detail

This is the exhaustive technical reference for the simulator. Every waveform,
parameter, processing step, algorithm, metric, and measured number, grounded in
the actual source files. Companion documents: `teaching_document.pdf` (the
concepts from the ground up), `simulation_design.pdf` (architecture summary),
`README.md` (quick start).

---

## 1. What this code is, and is not

**It is:** a sampled-IQ physical simulation of two radio stations achieving and
holding over-the-air carrier phase/frequency synchronization, plus faithful
reimplementations of a published alternative (Rashid & Nanzer, IEEE TWC 2023),
plus two schemes of our own design, all measured under identical physics.

**It is not:** a communication system. **No data is transmitted, no bits are
encoded, there is no modulation scheme (no QPSK, no OFDM), and no channel
coding.** Every transmission is a *pilot* — a waveform both sides agreed on in
advance — and the receiver's only job is to measure how that known waveform
arrived: when (timing), at what frequency offset (CFO), and at what phase.

**Signal representation:** everything is complex baseband IQ at 1 MS/s. Each
sample is one complex number `I + jQ` whose angle is the instantaneous phase
and magnitude the amplitude. All arithmetic is double precision
(`torch.float64` / `torch.complex128`) because the questions of interest live
at the milliradian level.

---

## 2. The waveforms ("encoding scheme")

All pilots are **Zadoff-Chu (ZC) sequences** (`ota_sync/sdr.py:_zadoff_chu`):

```
odd  length L:  x[n] = exp(-j * pi * u * n(n+1) / L)
even length L:  x[n] = exp(-j * pi * u * n^2 / L)
```

with root `u` coprime to `L`. Two properties motivate the choice:

1. **Constant envelope**: `|x[n]| = 1` for every sample, so the power
   amplifier runs at one level (no peak-to-average problem).
2. **Ideal periodic autocorrelation**: zero at every nonzero cyclic shift, so
   a matched filter yields a single sharp timing peak even through multipath.
   A cyclic prefix (the sequence's tail copied in front) makes the channel's
   linear convolution act cyclically, preserving the property.

### 2.1 The full synchronization frame (4606 samples = 4.606 ms at 1 MS/s)

Built by `make_sync_preamble`:

| segment | construction | samples | purpose |
|---|---|---|---|
| STF (short training field) | ZC length 16, root 1, repeated 16x | 256 | packet detection, coarse CFO |
| LTF 1 | 128-sample cyclic prefix + ZC length 2047, root 25 | 2175 | fine CFO (one end of baseline), phase |
| LTF 2 | identical block | 2175 | fine CFO (other end), phase |
| **total** | | **4606** | |

The capture window adds 2x64 guard samples + up to 32 samples of timing
jitter: `input_length = 4766`; after channel spreading (`l_tot` taps for a
3 microsecond maximum delay) one capture is ~4780 samples.

### 2.2 The micro-pilot (two-tier and hybrid schemes)

`ota_sync/microsync.py:_make_micro_preamble`: a 32-sample cyclic prefix + one
ZC of length 255 (root auto-selected as 26, the first value >= 25 coprime to
255) = 287 samples = 0.287 ms. No STF: micro-pilots are used only after lock,
when timing and frequency are already known, so no detection preamble is
needed.

---

## 3. Configuration reference (`SDRSimulationConfig`, `ota_sync/sdr.py`)

Every field, its default, and its meaning:

| field | default | meaning |
|---|---|---|
| `sync_interval` | 0.05 s | loop period T: one pilot exchange + correction per interval |
| `sample_rate` | 1e6 | IQ sample rate (Hz) |
| `num_iterations` | 100 | number of intervals simulated |
| `snr_db` | 20.0 | nominal link SNR, defined at nominal channel gain (see 5.3) |
| `carrier_frequency_hz` | 915e6 | carrier |
| `tdl_model` | "D" | 3GPP TR 38.901 TDL profile (D = line-of-sight dominant) |
| `delay_spread_s` | 100e-9 | channel RMS delay spread |
| `maximum_channel_delay_s` | 3e-6 | tap truncation for the discrete channel |
| `channel_speed_mps` | 0.0 | mobility; Doppler = v * f_c / c = 3.05 Hz per m/s |
| `short_sequence_length` | 16 | STF ZC length (sets coarse-CFO range +-fs/32) |
| `short_repetitions` | 16 | STF repeats |
| `long_sequence_length` | 2047 | LTF ZC length |
| `long_cp_length` | 128 | LTF cyclic prefix |
| `long_repetitions` | 2 | LTF blocks (2175-sample fine-CFO baseline) |
| `capture_guard_samples` | 64 | window guard on each side |
| `timing_jitter_samples` | 32 | random arrival slop per capture (0 for micro-pilots) |
| `master_initial_phase` | 0.0 | node A start phase (rad) |
| `master_initial_frequency_hz` | 0.0 | node A start CFO |
| `slave_initial_phase` | 1.2 | node B start phase (rad) |
| `slave_initial_frequency_hz` | 1500.0 | node B start CFO (=> 1.64 ppm reference error) |
| `phase_process_std_rad` | 0.002 | per-interval oscillator phase random walk (each node) |
| `frequency_process_std_hz` | 0.1 | per-interval oscillator frequency random walk (each node) |
| `sample_clock_offset_ppm` | None | None = SFO derived from carrier error (shared crystal); a number forces a fixed SFO |
| `phase_noise_std_rad` | 2e-4 | white-FM walk step per sample (=> sigma_y(1s) = 3.5e-11) |
| `phase_noise_white_pm_std_rad` | 0.005 | white PM jitter per sample |
| `flicker_frequency_std_hz` | 0.05 | RMS flicker FM (AR(1)-bank surrogate) |
| `shadowing_std_db` | 2.0 | log-normal shadowing std |
| `shadowing_correlation_s` | 10.0 | shadowing AR(1) correlation time |
| `iq_gain_imbalance_db` | 0.2 | receiver I/Q gain mismatch |
| `iq_phase_imbalance_deg` | 1.0 | receiver I/Q phase skew |
| `dc_offset` | 0.005+0.003j | receiver DC offset (post-AGC) |
| `adc_bits` / `dac_bits` | 12 / 12 | converter resolution |
| `tx_amplitude` | 0.5 | frame amplitude into the PA |
| `pa_clip_level` | 0.9 | soft limiter level |
| `agc_target_rms` | 0.25 | receiver AGC target |
| `detection_threshold` | 0.25 | normalized correlation threshold for "packet present" |
| `phase_correction_bits` | 16 | NCO phase quantization (2*pi/2^16) |
| `frequency_correction_resolution_hz` | 0.01 | NCO frequency quantization |
| `correction_latency_intervals` | 1 | intervals between computing and loading a correction |
| `twoway_chain_asymmetry_deg` | 0.0 | residual TX/RX chain asymmetry (does not cancel in two-way) |
| `seed` / `device` | 0 / "auto" | reproducibility / cpu-cuda |

---

## 4. The oscillator and clock model

### 4.1 State (`ota_sync/core.py:Oscillator`)

Each node holds `[phi, omega]` (rad, rad/s) evolving per interval as

```
[phi; omega] <- [[1, T],[0, 1]] @ [phi; omega] + N(0, diag(sigma_phi^2, sigma_omega^2))
```

### 4.2 The four noise processes and where they enter

1. **White FM (dominant)**: a per-sample phase random walk of std
   `phase_noise_std_rad`. Applied *inside each capture* as a cumulative sum on
   the received samples; its end value, plus an independent increment for the
   dead time, is **folded back into the oscillator state**
   (`carried_lo_walk`), making intra-frame and inter-frame noise one
   continuous process. Contributes phase variance `sigma_pn^2 * fs * T` per
   interval: 45 mrad RMS at defaults — the irreducible cadence floor.
2. **White PM**: independent `N(0, 5 mrad)` phase jitter per received sample.
3. **Flicker FM** (`_FlickerFrequencyNoise`): a bank of 4 AR(1) processes
   with correlation times log-spaced from 2 intervals to the run horizon;
   their sum approximates a 1/f frequency-noise spectrum. Folded into the
   master state's frequency each interval; its per-step innovation variance
   is added to the EKF's process covariance.
4. **Random-walk FM**: the per-interval `frequency_process_std_hz` on each
   oscillator (aging/temperature).

### 4.3 One crystal: coherent CFO, SFO, and timing drift

Carrier LO and sample clock share the reference. Unless
`sample_clock_offset_ppm` is forced, the loop derives the SFO **every
interval** from the *physical* (correction-free) relative frequency:

```
physical_slave_freq = slave.state[1] - (sum of applied NCO frequency corrections)
sfo_ppm = (physical_slave_freq - master.state[1]) / (2*pi*f_c) * 1e6
```

because NCO corrections are digital and never touch the crystal. The same
fractional error drives **frame-timing drift**: each capture advances a
per-link `_timing_carry` by `sfo_ppm * 1e-6 * interval_samples`; whole-sample
drift steps the frame's insertion point inside the capture window (a receiver
re-centering on each detection), and the sub-sample residual is applied as a
fractional delay in the resampler (`_resample_clock_offset` with
`delay_samples`).

---

## 5. Transmit chain, channel, receive chain (exact order, per capture)

`ota_sync/sdr.py:SDRRadioLink.capture`:

### 5.1 Transmit
1. Place the preamble at `start = guard + jitter + timing-drift step`.
2. Modulate onto the master carrier: multiply by
   `exp(j*(phi_m + omega_m * t))`, time referenced to the frame center.
3. Soft PA limiter (`_soft_limit`, level 0.9).
4. DAC quantization (12-bit per rail).

### 5.2 Channel
5. Per-interval 3GPP TDL taps (Sionna: `TDL` -> `cir_to_time_channel` ->
   `ApplyTimeChannel`; taps normalized; Jakes-correlated across intervals when
   speed > 0; one snapshot per interval, static within a capture).
6. AR(1) log-normal shadowing gain multiplies the taps (`_step_shadowing`;
   a mirrored reverse link reuses the forward link's taps and shadow state —
   reciprocity).
7. AWGN at a **fixed thermal noise floor**, established once from the
   nominal (unshadowed) signal power on the first frame — so per-frame SNR
   genuinely fades with the channel. (`snr_db` is the *nominal* link SNR.)

### 5.3 Receive
8. Resample at the current SFO with the accumulated fractional timing delay.
9. Downconvert by the slave LO: multiply by
   `exp(-j*(phi_s + omega_s * t))` on the receiver's own time base.
10. Intra-frame white-FM walk applied (cumulative), end value returned for
    state folding (see 4.2).
11. White-PM jitter per sample.
12. IQ imbalance: I scaled by `10^(0.2dB/40)`, Q inversely, plus a 1 degree
    quadrature skew.
13. AGC to 0.25 RMS; DC offset added; 12-bit ADC quantization (clip rate
    recorded).

**The oracle twin** is produced from the same capture: identical frame,
timing draw, channel taps, shadowing, PA/DAC, and SFO/timing resampling —
but no AWGN, no phase noise, no white PM, no IQ imbalance, no AGC/DC, no ADC.
Same estimator on the twin = ground truth (Section 8).

---

## 6. Receiver DSP (`SDRSynchronizer.estimate`)

Stage by stage, on the raw capture:

1. **DC removal**: subtract the sample mean.
2. **Detection + coarse timing + coarse CFO** (Schmidl-Cox style): with lag
   D=16, compute `P(d) = sum conj(r[d+m]) * r[d+m+D]` over a window of
   `short_length - D = 240` products, normalized by the two windows'
   energies. `M(d)` near 1 flags the repeated STF; argmax gives coarse
   timing; `angle(P)/(D*Ts)` gives coarse CFO, unambiguous to
   `+- fs/(2D) = +-31.25 kHz`.
3. **Fine timing**: derotate by coarse CFO, cross-correlate against the full
   4606-sample preamble (normalized); the ZC autocorrelation property gives a
   single sharp peak; a normalized score below `detection_threshold = 0.25`
   declares a miss.
4. **Coarse CFO refresh** on the timed STF.
5. **Fine CFO**: the phase of `sum conj(LTF1) * LTF2` across the 2175-sample
   separation, divided by the separation time. Total CFO = coarse + fine.
6. **Phase**: after full CFO removal (centered at the frame midpoint),
   matched-filter both LTFs against the known ZC; the angle of the summed
   correlation is the frame phase.

**Micro-pilot estimator** (`microsync.py:_estimate_micro_phase`, tracking
mode only): derotate by the *filter's current frequency estimate*, correlate
the known ZC-255 at the expected arrival +-4 samples, take the best
normalized peak; its angle is the phase. Works because after lock the timing
drifts only 0.016 samples per 10 ms and the residual CFO (~0.2 Hz) rotates
0.36 mrad across the pilot.

---

## 7. Estimation and control

### 7.1 The EKF (`ota_sync/core.py:PhaseFrequencyEKF`)

State `[theta, omega]` (relative phase/frequency). Observation
`z = [cos m, sin m, omega_meas]` (cos/sin avoids the +-pi wrap). Iterated
Gauss-Newton update (6 relinearizations), Joseph-form covariance.

**Measurement covariance R** (`_measurement_covariance`) — three physical
terms per component:

```
phase:  1/(2*SNR*L_tot)                     (AWGN CRLB; 1.1 mrad at defaults)
      + sigma_pn^2 * (offset + span/3)       (intra-frame walk;  ~8.3 mrad)
      + sigma_wpm^2 / L_tot                  (white PM)
freq:   1/(SNR*L_b*T_sep^2)                 (AWGN; 0.16 Hz)
      + sigma_pn^2 * N_sep / T_sep^2         (walk decorrelation; 0.68 Hz)
      + 2*sigma_wpm^2/(L_b*T_sep^2)          (white PM)
```

**Process covariance Q**: both nodes' random walks (2x the per-node values)
plus the white-FM walk over one interval (`sigma_pn^2 * fs * T`) on phase and
the flicker bank's innovation variance on frequency.

### 7.2 Acquisition, correction, latency, quantization

- First detected frame initializes the state directly from the measurement
  (the coarse+fine CFO must resolve cycle ambiguity before wrapped-phase
  tracking starts).
- Corrections are **forward-predicted** through `F^latency` (the controller
  knows its own latency), then quantized (phase to 2*pi/2^16, frequency to
  0.01 Hz), then queued; they load `correction_latency_intervals` later and
  the EKF state is reduced by the loaded correction at load time
  (predict -> reset -> update ordering).
- The resulting closed-loop error budget (verified by ablation):
  `sigma^2 ~ sigma_pn^2*fs*T  +  (sigma_omega_posterior * latency * T)^2  +  tracking^2`
  = 45 mrad + 38 mrad + ~12 mrad terms at defaults.

### 7.3 The pi ambiguity and its calibration (two-way loops)

The two-way half-difference is defined only modulo pi. Branch choice: nearest
zero at acquisition, nearest the filter prediction afterward. A wrong branch
is *self-consistent* (the loop parks the true phase at pi and every internal
statistic looks locked). Resolution, modeled as a deployment would do it:
after 3 loaded corrections, one external combining check (one bit:
constructive/destructive); if destructive, flip the slave NCO by pi. **The
filter is not reset at the flip** (the measurement is pi-invariant; the
near-zero state re-attaches to the true branch). In the hybrid, the channel
state is additionally shifted by -pi so the tracked sum stays consistent.
Steady-state metrics are masked to start after this calibration.

---

## 8. Ground-truth grading and metric definitions

**The trap this replaces:** comparing the filter posterior to the measurement
that just updated it leaves only `(1-kappa)` of the innovation; the original
code reported 0.105 mrad that way. True (oracle-graded) figure: 12.0-12.5
mrad — a factor ~100.

Result-object metrics (all graded against the oracle or the true oscillator
state, never against the estimator's own inputs):

| metric | definition |
|---|---|
| `ota_phase_error` (one-way) | wrap(true observable phase [oracle] - EKF estimate) |
| `ota_phase_rmse` | RMS of the above over detected frames |
| `post_correction_*` | residual after the most recently loaded correction (with latency: the true phase at each capture, drift included) |
| `post_correction_oscillator_phase` | wrap(true master-slave state): exposes the one-way channel bias (-2.92 rad) |
| `phase_rmse` (two-way) | RMS of wrap(true relative oscillator phase - estimate) |
| `steady_state_phase_rms` | RMS of the residual over frames that are detected AND after first correction AND after pi-calibration |
| `coherent_gain` | cos^2(residual/2), the two-station combining gain |
| `detection_rate`, `timing_error_samples`, `agc_gain`, `adc_clip_rate` | receiver health |
| `airtime_fraction` (micro/hybrid) | pilot samples per interval / interval samples |

---

## 9. The synchronization schemes (CLI `--model ...`)

### 9.1 `sdr` — one-way loop (default)
Master transmits; slave measures, filters, corrects. Locks the *observable*
(oscillator + channel) phase: tracking RMSE 12.0 mrad, closed-loop residual
69.6 mrad RMS at defaults — but the oscillators themselves sit 2.92 rad apart
(the absorbed channel phase). Sufficient with user CSI; insufficient for
open-loop coherence.

### 9.2 `twoway` — reciprocal two-way loop (`ota_sync/coherent.py`)
Forward and reverse frames per interval over the same channel realization
(mirrored link shares taps + shadowing; the forward frame's intra-frame walk
folds into the state before the reverse frame). EKF consumes the
half-difference phase/frequency (R halved). Result: true oscillator residual
**81.8 mrad RMS**, 99.83% gain, 19.1% airtime. Includes the pi-calibration.

### 9.3 `micro` — two-tier micro-pilots (`ota_sync/microsync.py`)
Full two-way frame once per interval (acquisition machinery) + M reciprocal
ZC-255 phase-only micro-pilots at evenly spaced sub-intervals; corrections
every sub-interval; oscillators, EKF, and noise all run at sub-interval
resolution. Measured trade (M = 0/2/4/9): 71.5 / 35.3 / 27.9 / 22.8 mrad at
19.1 / 22.6 / 26.0 / 34.6 % airtime.

### 9.4 `hybrid` — one-way pilots + sparse two-way anchors (`hybrid_calibration/`)
3-state EKF over `[theta, omega, phi_c]`. One-way full frame each interval
and one-way micro-pilots at sub-intervals observe the *sum* theta+phi_c
(Jacobian row [1,0,1]; the direction theta-phi_c is unobservable from
one-way data; the process priors attribute innovations in proportion to
P_theta_theta vs P_cc). Two-way anchors every K intervals observe theta (the
half-difference) and phi_c (recovered as m_fwd - theta_hat, which equals the
half-sum and is uncorrelated with the half-difference), re-pinning the split.
NCO acts on [theta, omega] only. `channel_drift_std_rad` is the assumed
channel-phase drift per interval — the estimator's statement of channel
coherence time.

Measured: static channel — 31.8/32.9/34.0 mrad at K=1/5/20 (airtime
22.6/14.9/13.5%): reciprocity cadence decoupled from oscillator quality.
Moving channel — degrades with Doppler under the static prior (0.2 m/s,
K=20: 1752 mrad) and recovers with a matched prior sigma_c ~ 2*pi*f_D*T
(0.5 m/s, K=1: 128 -> 44 mrad). Known defect: the one-way frequency
observation is biased by the LOS Doppler (a 4th state would absorb it).

### 9.5 `dfpc` / `kfdfpc` — the paper's algorithms (`ota_sync/dfpc.py`)
Two levels. **Statistics level** (`run_consensus_stats`): faithful to their
Algorithms 1-2 — N nodes, random connected graph at connectivity c,
Metropolis-Hastings weights, their error statistics (ADEV law beta1 = beta2 =
5e-19, jitter 2.7 mrad i.i.d., CRLB estimation noise), including the Eq. 39
covariance mixing for KF-DFPC. Validated at their setup (N=20, c=0.2,
T=0.1 ms, 0 dB): DFPC 8.0 deg vs the Eq. 27 bound 5.9 deg; KF-DFPC 2.1 deg.
**Physical level** (`run_consensus_ota_simulation`): the two-node update rule
(each node retunes its own NCO by half its offset estimate) over the full
simulator. `reciprocal=False` = as published (raw one-way measurements):
bistable over a real channel — converges to relative phase 0 *or* pi
depending on whether |r0| + |phi_c| crosses pi (default realization captures:
3009 mrad, 0.68% gain). `reciprocal=True` (steelman, using the side channel
the paper already assumes): DFPC 153.0 mrad (raw 0.442 Hz frequency noise
feeds the NCOs), KF-DFPC 82.8 mrad — statistically identical to our two-way
EKF (80.8), demonstrating that filtered loops meet at the physical floor.

### 9.6 `compare`
Runs two-way EKF, micro, hybrid, DFPC naive, DFPC+reciprocity,
KF-DFPC+reciprocity under identical settings; prints the table; with
`--plot`, overlays all |residual| trajectories on one log-scale time axis.

### 9.7 CSI-aided evaluation (`--csi-gain`, with `--model sdr`)
Not a sync scheme — an application evaluation. With user CSI, combining gain
depends only on the differential station phase drift since the last CSI
epoch (all static biases cancel: `sum g_i^* h_i e^{j phi_i(t)} =
sum |h_i|^2 e^{j[phi_i(t)-phi_i(t0)]}`). Evaluated from the one-way loop's
oscillator residual plus per-epoch pilot noise: 100.00 / 99.88 / 99.79 /
99.72 / 99.56 % gain at CSI refresh every 1 / 2 / 5 / 10 / 20 intervals.

### 9.8 `ideal`
The original tone-plus-AWGN baseline (`ota_sync/core.py:run_simulation`);
kept for comparison only.

---

## 10. CLI reference (`simulation.py`)

```
--model {sdr,twoway,micro,hybrid,dfpc,kfdfpc,compare,ideal}   which loop (default sdr)
--iterations N        number of sync intervals (default 100)
--snr-db X            nominal link SNR (default 20)
--seed N              RNG seed (default 0); identical seed => byte-identical run
--device auto|cpu|cuda
--plot                loop-level plots (residual + coherent gain; overlay for compare)
--plot-all            full one-way diagnostics (6 panels)
--plot-iq             one capture at the IQ-sample level (waveform, constellation,
                      detection metric, matched-filter peak)
--sample-rate, --carrier-mhz, --cfo-hz, --tdl-model, --delay-spread-ns,
--speed-mps, --adc-bits                      physical knobs
--sfo-ppm X           force a fixed SFO (default: derived from carrier error)
--flicker-std-hz, --shadowing-std-db         noise knobs
--correction-latency N                        NCO load latency in intervals
--micro-pilots M      micro-pilots per interval (micro/hybrid; default 4)
--anchor-every K      intervals between two-way anchors (hybrid; default 5)
--csi-gain            with sdr: print JT gain vs CSI refresh cadence
--no-rf-impairments   zero SFO, phase noise, white PM, flicker, shadowing,
                      IQ imbalance, DC offset (ablation switch)
--pilot-length        ideal model only
```

---

## 11. Measured results (all reproducible; seed 0 unless noted)

**One-way ablation (100 intervals):**

| configuration | tracking RMSE | closed-loop residual RMS |
|---|---|---|
| full defaults | 12.0 mrad | 69.6 mrad |
| `--no-rf-impairments` (isolates latency) | 1.1 mrad | 34.3 mrad (predicted 38) |
| `--correction-latency 0` (isolates tracking) | 12.3 mrad | 12.3 mrad |

**Head-to-head (`--model compare`):**

| approach | residual | gain | airtime |
|---|---|---|---|
| two-tier micro-pilot (ours) | 28.1 mrad | 99.98% | 26.0% |
| hybrid 1-way + anchors (ours) | 33.8 mrad | 99.97% | 14.9% |
| two-way EKF (ours) | 80.8 mrad | 99.84% | 19.1% |
| KF-DFPC + reciprocity | 82.8 mrad | 99.83% | 19.1% |
| DFPC + reciprocity | 153.0 mrad | 99.42% | 19.1% |
| DFPC naive (as published) | 3009 mrad | 0.68% | 19.1% |

**Micro-pilot sweep:** M=0/2/4/9 -> 71.5/35.3/27.9/22.8 mrad at
19.1/22.6/26.0/34.6% airtime.

**Hybrid, static channel:** K=1/5/20 -> 31.8/32.9/34.0 mrad at
22.6/14.9/13.5% airtime (control 27.9 @ 26.0%).

**Hybrid under Doppler (60 intervals):**

| m/s | f_D | scheme | residual | gain |
|---|---|---|---|---|
| 0.2 | 0.61 | control (two-way) | 43 mrad | 99.95% |
| 0.2 | 0.61 | K=5, static prior | 446 mrad | 95.3% |
| 0.2 | 0.61 | K=5, matched prior | 235 mrad | 98.7% |
| 0.2 | 0.61 | K=20, static prior | 1752 mrad | 55.1% |
| 0.5 | 1.52 | control (two-way) | 33 mrad | 99.97% |
| 0.5 | 1.52 | K=1, static prior | 128 mrad | 99.6% |
| 0.5 | 1.52 | K=1, matched prior | 44 mrad | 99.95% |
| 0.5 | 1.52 | K=5, matched prior | 208 mrad | 98.9% |

**Independent verifications performed:** byte-identical rerun diffs; the
anti-phase capture reproduced by hand arithmetic (wrap(1.2+2.92) = -2.16,
wrap(-1.2+2.92) = +1.72, r1 = pi exactly) and by a four-line radio-free
iteration; seed sweep matching the capture condition |r0|+|phi_c| > pi;
error-budget cross-check (unfiltered consensus: 0.442 Hz measured frequency
noise predicts 146 mrad, observed 153).

---

## 12. The test suite (`tests/test_ota_sync.py`, 18 tests)

| test | asserts |
|---|---|
| `test_wrap_phase_uses_principal_interval` | wrap maps to [-pi, pi) |
| `test_pilot_receiver_recovers_phase_and_frequency_without_noise` | ideal-model estimator exact in noiseless case |
| `test_noiseless_ota_loop_synchronizes_slave` | ideal loop converges; covariance PSD |
| `test_seeded_sionna_awgn_run_is_reproducible` | identical seeds => identical results |
| `test_sdr_preamble_has_repeated_training_fields` | frame structure (STF repeats, total length) |
| `test_sdr_truth_reference_matches_measurement_when_clean` | oracle == measurement when impairments off at 100 dB SNR |
| `test_consensus_stats_converges_and_respects_eq27_bound` | stats-level DFPC collapses initial spread; residual <= 1.5x Eq. 27 |
| `test_consensus_stats_kalman_variant_reduces_residual` | KF-DFPC < DFPC in their model |
| `test_naive_consensus_ota_captures_at_anti_phase` | as-published DFPC over a real channel locks at pi (seed 0) |
| `test_reciprocal_consensus_ota_aligns_and_filtering_helps` | reciprocity fixes it; KF < unfiltered |
| `test_sdr_tdl_link_acquires_and_corrects_effective_ota_carrier` | one-way loop acquires; residual within walk floor |
| `test_sdr_delayed_corrections_converge_without_phase_noise` | forward-predicted delayed corrections converge |
| `test_two_way_sync_cancels_channel_phase_bias` | -2.92 rad bias removed; gain > 0.9 |
| `test_two_way_clean_loop_reaches_estimation_floor` | clean two-way < 0.1 rad, gain > 0.99 |
| `test_micro_pilot_loop_beats_plain_two_way` | micro < 0.5x two-way residual at < 35% airtime |
| `test_hybrid_calibration_matches_micro_at_much_lower_airtime` | hybrid < 0.6x two-way at < 16% airtime |
| `test_hybrid_doppler_requires_matched_channel_prior` | matched prior < 0.5x mismatched at 0.5 m/s |
| `test_csi_joint_transmission_gain_degrades_with_stale_csi` | JT gain ~1 fresh, monotonically lower stale |

Run: `python -m pytest` from the repo root.

---

## 13. File map

```
simulation.py                 CLI entry point, all plotting
ota_sync/core.py              Oscillator, PhaseFrequencyEKF, wrap helpers, ideal model
ota_sync/sdr.py               config, ZC frame, radio link + all impairments,
                              receiver DSP, R/Q construction, one-way loop, oracle
ota_sync/coherent.py          two-way loop, pi-calibration, CSI-gain evaluation
ota_sync/dfpc.py              Rashid-Nanzer reimplementations (stats + physical)
ota_sync/microsync.py         two-tier micro-pilot loop, micro estimator
hybrid_calibration/hybrid.py  3-state joint EKF, hybrid loop
tests/test_ota_sync.py        18 regression tests
requirements.txt              sionna-no-rt==2.0.1 (channel physics)
teaching_document.pdf/.tex    ground-up tutorial (concepts + derivations)
teaching_slides.pdf/.tex      59-slide lecture deck
simulation_design.pdf/.tex    architecture summary
paper_review_rashid_nanzer.pdf  summary/contributions/limitations of the paper
LITERATURE_REVIEW.md          verified 19-source prior-art pass
```

---

## 14. Known simplifications (honest limits)

- Flicker FM is a 4-component AR(1) surrogate, not true 1/f (fine at the
  simulated timescales; less trustworthy extrapolated to hours).
- Channel snapshots are per-interval (static within a capture); no blockage
  events or geometric consistency; TDL is statistical.
- AGC is instantaneous; IQ imbalance and DC offset are static.
- The one-way frequency observation in the hybrid is biased by LOS Doppler
  (4th state = planned fix).
- Two-way assumes the measurement exchange (side channel) is error-free, and
  chain asymmetry is a knob rather than an estimated quantity.
- **Everything is simulation. No hardware validation has been performed.**
```
