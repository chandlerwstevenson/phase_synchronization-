# How every synchronization method computes what it computes

The complete calculation chain, method by method, grounded in the source.
Companion to `CODE_REFERENCE.md` (what everything is) — this file is the
math (how everything is computed). Notation is plain ASCII; `wrap(x)`
always means reduction to [-pi, pi).

Conventions used throughout:

    theta      relative oscillator (crystal-to-crystal) phase, rad
    omega      relative angular frequency, rad/s (CFO = omega / 2*pi Hz)
    phi_c      propagation-channel phase of the link
    T          sync interval (s), default 0.05
    f_s        IQ sample rate (1e6), f_c carrier (915e6)
    sigma_pn   per-sample white-FM phase-walk std (oscillator quality)

---

## 1. The oscillator model (`ota_sync/core.py:Oscillator`)

Each node carries state x = [phi, omega]. Per interval:

    x <- F x + w,   F = [[1, T], [0, 1]],   w ~ N(0, diag(sig_phi^2, sig_om^2))

Four noise processes enter (`ota_sync/sdr.py`):

1. **White FM** — per received sample, phase does a random walk with std
   `phase_noise_std_rad`. Inside a capture it is applied as a cumulative
   sum on the samples; its end value plus an independent draw for the
   dead time (std = sigma_pn * sqrt(remainder_samples)) is folded back
   into the oscillator state (`carried_lo_walk`), so intra-frame and
   inter-frame noise form one continuous process. Accumulated phase
   variance over any span tau: sigma_pn^2 * f_s * tau.
2. **White PM** — i.i.d. N(0, sigma_wpm) phase jitter added per sample.
3. **Flicker FM** (`_FlickerFrequencyNoise`) — sum of 4 AR(1) processes
   with correlation times log-spaced from 2T to the run length; the sum
   approximates a 1/f frequency spectrum. Its per-step innovation
   variance is added to the EKF process covariance.
4. **Random-walk FM** — per-interval frequency step of std
   `frequency_process_std_hz` (aging/temperature).

**Oscillator profiles** (`ota_sync/oscillators.py`) convert datasheet
anchors to these knobs:

    sigma_pn        = ADEV_whiteFM(1s) * 2*pi*f_c / sqrt(f_s)
    flicker_std_hz  = ADEV_flicker_floor * f_c
    freq_walk/intvl = (Hz per sqrt-second rate) * sqrt(T)
    initial CFO     = f_c * accuracy_ppb * 1e-9

Derivation of the first: a phase walk of per-sample variance sigma_pn^2
accumulates sigma_pn^2*f_s*tau over tau seconds, so the fractional
frequency deviation averaged over tau is
sigma_y(tau) = sigma_pn*sqrt(f_s)/(2*pi*f_c*sqrt(tau)) — white FM's
tau^(-1/2) law; solve at tau = 1 s.

## 2. The pilot waveform (`ota_sync/sdr.py:make_sync_preamble`)

Zadoff-Chu sequences: x[n] = exp(-j*pi*u*n(n+1)/L) (odd L) or
exp(-j*pi*u*n^2/L) (even L), root u coprime to L. Frame =
STF (ZC-16 repeated 16x = 256 samples) + 2 x [128-sample cyclic prefix +
ZC-2047] = 4606 samples. Constant envelope; ideal periodic
autocorrelation (zero at all nonzero cyclic shifts).

## 3. What one capture does to the signal (`SDRRadioLink.capture`)

TX: place frame at start = guard + jitter + timing-drift step; modulate
by exp(j*(phi_m + omega_m*t)) (t referenced to frame center); soft-limit
(PA); 12-bit DAC. Channel: per-interval 3GPP TDL taps (Sionna), AR(1)
log-normal shadowing multiplying the taps, AWGN at a fixed thermal floor
set once from nominal power (so per-frame SNR genuinely fades). RX:
resample at the SFO with accumulated fractional delay; downconvert by
exp(-j*(phi_s + omega_s*t)); apply white-FM walk cumulatively + white PM
per sample; IQ imbalance; AGC to 0.25 RMS; DC offset; 12-bit ADC.

SFO is derived each interval from the physical (correction-free)
oscillators, because NCO corrections are digital and never touch the
crystal:

    sfo_ppm = (omega_s_physical - omega_m) / (2*pi*f_c) * 1e6
    omega_s_physical = omega_s_state - sum(applied frequency corrections)

The **oracle twin** repeats the identical capture (same frame, taps,
shadowing, PA/DAC, SFO/timing) with no AWGN/phase-noise/PM/IQ/AGC/DC/ADC;
running the same estimator on it defines ground truth.

## 4. Receiver estimation (`SDRSynchronizer.estimate`)

Given raw capture r[n]:

1. **DC removal**: r <- r - mean(r).
2. **Detection + coarse timing/CFO** (Schmidl-Cox on the STF, lag D=16):

       P(d) = sum_{m=0}^{W-1} conj(r[d+m]) * r[d+m+D],  W = 256 - D
       M(d) = |P(d)| / sqrt(E1(d) * E2(d))              (energy-normalized)

   argmax_d M(d) gives coarse timing; coarse CFO
   omega_coarse = angle(P) / (D * Ts), unambiguous to +-f_s/(2D)
   = +-31.25 kHz.
3. **Fine timing**: derotate by omega_coarse, cross-correlate against
   the full 4606-sample known frame, normalize by window energy; the ZC
   property gives one sharp peak. Normalized score < 0.25 => miss.
4. **Fine CFO** from the two identical LTF blocks separated by
   N_sep = 2175 samples:

       omega_fine = angle( sum conj(LTF1[n]) * LTF2[n] ) / (N_sep * Ts)

   Total CFO = coarse + fine. The block separation is the baseline; the
   +-pi ambiguity of the angle corresponds to +-f_s/(2*N_sep) = +-230 Hz,
   resolved by the coarse estimate.
5. **Phase**: remove total CFO (time-referenced to frame midpoint),
   matched-filter both LTFs against the known ZC; frame phase =
   angle( sum of the two correlation sums ).

**Micro-pilot estimator** (`microsync.py:_estimate_micro_phase`,
tracking only): derotate by the FILTER's current frequency estimate,
correlate the known ZC-255 at the expected arrival +-4 samples, take
the best normalized peak; its angle is the phase. Valid because after
lock, timing drifts ~0.016 samples per 10 ms and residual CFO (~0.2 Hz)
rotates only ~0.4 mrad across the 287-sample pilot.

## 5. The 2-state EKF (`core.py:PhaseFrequencyEKF`)

State x = [theta, omega]. Observation z = [cos m, sin m, omega_meas]
(the cos/sin embedding avoids the +-pi wrap in the innovation).

Predict:

    x <- F x
    P <- F P F^T + Q

Update (iterated Gauss-Newton, 6 relinearizations):

    h(x) = [cos theta, sin theta, omega]
    H(x) = [[-sin theta, 0], [cos theta, 0], [0, 1]]
    S    = H P- H^T + R
    K    = P- H^T S^-1
    x_i+1 = x- + K (z - h(x_i) - H (x- - x_i))     (iterate to convergence)
    P+   = (I - K H) P- (I - K H)^T + K R K^T      (Joseph form)

Relinearization matters because a one-shot EKF under-corrects large
offsets (sin(-1.4) != -1.4).

**Measurement covariance R** (diagonal, physical terms;
`sdr.py:_measurement_covariance`):

    phase var:  1/(2*SNR*L_tot)                    AWGN CRLB
              + sigma_pn^2 * (offset + span/3)     intra-frame walk
              + sigma_wpm^2 / L_tot                white PM
    freq var:   1/(SNR * L_b * T_sep^2)            AWGN
              + sigma_pn^2 * N_sep / T_sep^2       walk decorrelation
              + 2*sigma_wpm^2 / (L_b * T_sep^2)    white PM

**Process covariance Q**: both nodes' per-interval random walks (2x the
per-node variances) + the white-FM walk over one interval
(sigma_pn^2 * f_s * T) on phase + the flicker bank's innovation variance
on frequency.

## 6. Control: correction, latency, quantization (all loops)

On each detected frame the controller forward-predicts its own latency:

    correction = quantize( F^latency @ x_posterior )

quantized to 2*pi/2^16 in phase and 0.01 Hz in frequency, then queued;
it loads `correction_latency_intervals` later by adding to the slave's
state (NCO retune), and the EKF is reduced by the loaded value at load
time (predict -> reset_after_correction -> update ordering).

Resulting closed-loop error budget (verified by ablation):

    sigma^2 ~ sigma_pn^2*f_s*T            irreducible walk over one interval
            + (sigma_omega_post * L*T)^2  drift during the L-interval latency
            + tracking^2                  estimator error
    (45 mrad + 38 mrad + ~12 mrad at defaults -> ~70 mrad total RMS)

---

## 7. Method: one-way (`--model sdr`, `sdr.py:run_sdr_simulation`)

Master transmits; slave measures. The slave's observable is the SUM

    m = wrap(theta + phi_c) + noise

One equation, two unknowns: theta and phi_c are not separately
identifiable, so the EKF tracks the observable and the loop drives
theta + phi_c -> 0, parking the crystals at theta = -phi_c (~ -2.9 rad
at seed 0). Metrics: `ota_phase_error = wrap(true_observable[oracle] -
estimate)`; `post_correction_ota_phase` IS the observable at each
capture (with latency >= 1 every capture already includes all loaded
corrections). Sufficient for links and CSI-aided combining; open-loop
crystal gain is cos^2(theta/2) with theta stuck at -phi_c.

## 8. Method: two-way (`--model twoway`, `coherent.py`)

Both directions cross the SAME channel realization within an interval
(the mirrored link reuses taps + shadow state). Measurements:

    m_fwd = wrap(theta + phi_c),   m_rev = wrap(-theta + phi_c)

The channel is common-mode; the oscillator offset flips sign. The
half-difference cancels phi_c:

    theta_hat = wrap( wrap(m_fwd - m_rev)/2 + chain_bias
                      - omega_hat * tau_turnaround / 2 )

The last term: the reverse frame is measured one TDD turnaround tau
later, so the raw half-difference contains omega*tau/2 of drift; the
receiver removes the part its own CFO estimate predicts (residue =
CFO-estimate error x tau/2 + walk during the gap). During the gap both
oscillators advance deterministically (state += omega*tau) plus a walk
draw of std sigma_pn*sqrt(f_s*tau).

Frequency: omega_hat = (omega_fwd - omega_rev)/2. R is halved relative
to one-way (two measurements). EKF and control as in sections 5-6.

**The pi ambiguity**: the half-difference is defined only modulo pi
(adding pi to theta flips both directions' signs consistently). Branch
choice `_pick_half_phase`: nearest 0 at acquisition, nearest the filter
prediction afterwards. A wrong branch is self-consistent (loop parks at
pi, everything looks locked), so after 3 loaded corrections one external
combining check (1 bit: constructive/destructive) flips the slave NCO by
pi if needed; the filter is NOT reset (the measurement is pi-invariant).
Steady-state metrics mask to start after this calibration.

## 9. Method: two-tier micro-pilots (`--model micro`, `microsync.py`)

One full two-way frame per interval (keeps acquisition machinery), plus
M reciprocal ZC-255 micro-pilots at evenly spaced sub-intervals
T_sub = T/(M+1). Same half-difference math at every exchange; phase-only
micro measurements use the estimator of section 4's last paragraph;
oscillators, EKF, and corrections all run per sub-interval. Airtime:

    airtime = 2*(full_capture + M*micro_capture) / (T * f_s)

## 10. Method: hybrid (`--model hybrid`, `hybrid_calibration/hybrid.py`)

3-state EKF x = [theta, omega, phi_c] with F = [[1,T,0],[0,1,0],[0,0,1]]
and channel drift prior sigma_c (`channel_drift_std_rad`, the estimator's
statement of channel coherence time).

- **One-way observations** (full frame each interval + M one-way
  micro-pilots per interval) see only the sum: Jacobian phase row is
  [dh/dtheta, 0, dh/dphi_c] = [1, 0, 1] (in cos/sin embedding). The
  direction theta - phi_c is unobservable from these; innovations are
  attributed in proportion to the priors P_theta,theta vs P_cc.
- **Two-way anchors** every K intervals observe theta directly (the
  half-difference of section 8) and recover the channel as

      phi_c_meas = wrap(m_fwd - theta_hat)     (equals the half-sum)

  which is uncorrelated with the half-difference — the anchor re-pins
  the theta/phi_c split that one-way data cannot see.
- NCO corrections act on [theta, omega] only; at the pi-flip
  calibration the channel state is additionally shifted by -pi so the
  tracked sum stays consistent.

Known defect: the one-way frequency observation is biased by LOS
Doppler (a 4th state would absorb it). Airtime: like micro but with
one-way micro-pilots (half the pilot cost) plus anchors every K.

## 11. Method: DFPC / KF-DFPC (`--model dfpc|kfdfpc`, `dfpc.py`)

**Statistics level** (`run_consensus_stats`, faithful to the paper):
N nodes, random connected graph, Metropolis-Hastings mixing weights

    W[i,j] = 1/(1 + max(deg_i, deg_j)) for edges, W[i,i] = 1 - sum_j W[i,j]

Each iteration nodes consensus-average their (noisy) frequency and
phase estimates: x <- W x, with the paper's error statistics injected
(ADEV law beta1 = beta2 = 5e-19, 2.7 mrad i.i.d. jitter, CRLB estimation
noise). KF-DFPC additionally mixes covariances with squared weights
(their Eq. 39). Validated against their Eq. 27 residual bound.

**Physical level** (`run_consensus_ota_simulation`, N = 2): each node
estimates its offset to the other and retunes its own NCO by HALF the
estimate (symmetric update — both nodes move toward each other):

    naive (as published):  node i uses its raw one-way measurement,
                           which is wrap(+-theta + phi_c) — the channel
                           phase rides along. The symmetric wrapped
                           update then has two fixed points, theta = 0
                           and theta = pi, and captures at anti-phase
                           whenever |theta_0| + |phi_c| crosses pi.
    reciprocal (steelman): the nodes exchange measurements over the
                           paper's assumed side channel and use the
                           channel-free half-difference (section 8),
                           node a with +half, node b with -half.

KF-DFPC runs one 2-state EKF (section 5) per node on its observation;
each node forward-predicts its latency and the side channel carries the
applied corrections so each filter can reset by (own - other) at load.

## 12. N-station networks (`--stations N`, `ota_sync/network.py`)

Star topology: station 0 is the reference; each station k runs the
chosen pairwise scheme against it, TDMA on one channel.

Placement: uniform in a disc of radius R via r = R*sqrt(u1),
angle = 2*pi*u2, rejection-sampled to >= 10 m separation.

Link budget (log-distance path loss):

    SNR_k = SNR_ref - 10 * n * log10(d_k / d_ref),   capped at +50 dB

Array coherence with the reference as datum (theta_0 = 0):

    G(t) = | 1 + sum_k exp(j*theta_k(t)) |^2 / N^2

Total airtime = sum of per-link airtimes (the shared reference radio is
the bottleneck) = (N-1) x per-link fraction; >= 100% means the pilot
schedule physically cannot fit in the interval. Caveat: the pairwise
links are simulated independently, so the reference's own noise is
drawn per link instead of common-mode; the reported G is slightly
conservative.

## 13. Metric formulas (all methods)

    tracking RMSE      sqrt(mean over detected frames of
                       wrap(true - estimate)^2), truth from the oracle
    steady residual    RMS of the post-correction residual over frames
                       that are detected AND after the first loaded
                       correction AND (two-way family) after the
                       pi-calibration
    2-station gain     cos^2(residual/2)     ( = |1 + e^{j*res}|^2 / 4 )
    N-station gain     section 12 formula
    airtime fraction   pilot capture samples per interval / (T * f_s)
    detection rate     detected frames / all frames (min over links for
                       networks)
    free-running curve integrate physical_relative_frequency (the
                       correction-free relative oscillator frequency,
                       recorded every step) x step, then wrap; its gain
                       is cos^2(wrap(.)/2)

## 14. Where each computation lives

    ota_sync/core.py        Oscillator, wrap_phase, PhaseFrequencyEKF
    ota_sync/sdr.py         frame, capture chain, receiver DSP, R and Q,
                            one-way loop, oracle, quantization
    ota_sync/coherent.py    two-way loop, turnaround, pi-calibration,
                            CSI-gain evaluation
    ota_sync/microsync.py   micro estimator, two-tier loop
    hybrid_calibration/     3-state EKF, anchor updates
    ota_sync/dfpc.py        consensus (stats + physical)
    ota_sync/network.py     placement, path loss, array gain
    ota_sync/oscillators.py datasheet -> noise-knob conversions
