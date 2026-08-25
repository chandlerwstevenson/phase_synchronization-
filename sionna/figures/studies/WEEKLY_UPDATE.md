# Weekly research update — synchronization of distributed coherent arrays

*Prepared 2026-08-17. All results are from impairment-complete
simulation (framework and conditions in §2); no hardware validation
yet. Supporting detail: `THEORY.md` (derivations),
`EXPERIMENT_SUMMARY.md` (per-experiment protocol and results),
`FIGURES.md` (index of the 39 figures in this folder), `SNR_LAW.md`
(the SNR analysis in full). This document is self-contained.*

---

**Correction notice (2026-08-24).** Where this document attributes a
~100–153 mrad per-exchange noise floor to *multipath resampling*, the
magnitude is real but the mechanism is wrong: frozen-oscillator
controls show sample-timing jitter over frozen multipath contributes
< 0.05 mrad. That floor is unmodeled **oscillator** noise
(class-proportional ≈28/130/650 mrad for oven-controlled/
temperature-compensated/cheap oscillators; flicker frequency noise
modeled as white is the suspect). A genuine but smaller channel
mechanism does exist — clock-offset-induced *fractional* resampling,
10–25 mrad, conditionally white — with a validated zero-fit predictor.
No measured result in this document changes; the filter-overconfidence
finding stands with the corrected explanation. Detail:
`../../phase_sync_idea/RESULTS_DOMINANCE.md` and `resampling_law.py`.

## 1. Summary

This period produced four main results and one negative result:

1. **Self-sensing ("piggyback") synchronization.** The array's own
   sensing transmissions can serve as synchronization pilots, with
   sparse two-way exchanges retained only to pin the unknown static
   propagation phase. Dedicated sync airtime falls ~16× at
   equal-or-better residual, verified across 13 propagation
   environments (statistical and ray-traced, including a
   no-line-of-sight urban placement) and array sizes N = 2–30. The
   method's single failure boundary is environmental coherence time.
2. **Membership control.** Under sync contention, controlling which
   stations participate in the coherent sum dominates every metric
   we evaluated, at every array size tested (N = 6–20). The optimal
   gating rule is implementable with one feedback bit per station
   per interval (an exact identity, §4.3), and without membership
   control, enlarging the array *reduces* detection probability.
3. **An ex-ante synchronization budget theory.** The per-station
   sustainable coasting interval is predictable exactly (99.5% of
   ~26,000 measured service gaps) from oscillator datasheet
   specifications, the link budget, and the frame design, with zero
   fitted constants; a frozen two-parameter model built on it
   blind-predicts array coherence, spectral efficiency, and
   detection range, including extrapolation to array sizes outside
   its validation set (§4.1, §4.2).
4. **SNR dependence (the question raised last meeting).** Phase
   residual and sync airtime are nearly independent of link SNR over
   the operational range: the pilot's processing gain and an
   SNR-independent multipath "resampling" floor dominate thermal
   noise. SNR's operative roles are (i) the pilot detection
   threshold, which sets maximum station spacing, and (ii) enabling
   pilot shortening. Measured: the default pilot is ~6× longer than
   necessary; shortening it cuts sync airtime 19.1% → 3.0% at
   unchanged residual with *no* SNR trade required (§4.5).
5. **Negative result.** The hypothesized Doppler/channel-
   decorrelation term in the coasting law is not supported: the
   motivating observation was single-seed noise, and the measured
   reciprocity-bias structure function is flat in service gap at all
   tested speeds. The actual mechanism is white per-capture multipath
   resampling noise (~90–100 mrad, per-link, N-independent).

## 2. Methods and verification standards

**Simulation framework.** SDR-class physical layer over Sionna
channels at 915 MHz, 1 MHz bandwidth, 50 ms sync interval. Enabled
impairments (defaults, verified by audit): TDL-D line-of-sight fading
(100 ns delay spread; TDL-A/B/C/E and ray-traced scenes where
stated), log-distance path loss (exponent 2.7) over random
deployments, ±32-sample timing jitter, four-term oscillator noise
(white-PM, white-FM, flicker-FM, random-walk-FM) anchored to
datasheet Allan deviation of real parts (Stratum-3E OCXO, small-cell
TCXO, USRP-class TCXO), shadowing, IQ imbalance, DC offset, 12-bit
conversion, PA soft clipping, quantized corrections, one-interval
actuation latency, TDD turnaround, initial CFO with derived SFO.
Detection is counted Monte-Carlo (no closed-form shortcuts):
Swerling-1 target, thermal noise floor, matched filtering,
empirically calibrated thresholds at false-alarm rate 10⁻³,
recalibrated per receive-combiner variant.

**Verification standards applied to every claim.** (i) Multi-seed
replication (typically seeds 0–2; single-seed results are labeled
preliminary). (ii) Independent from-scratch re-runs of every
headline number — all reproduced; two within-noise deviations are
disclosed in `FIGURES.md`. (iii) Blind protocol for the theory:
predictions computed and printed before the corresponding
measurements, no post-hoc retuning; misses reported as misses.
(iv) 150+ regression tests green; no pre-existing code modified
(all studies are additive). (v) Adversarial literature checks
against 2025–26 publications for each claimed contribution (§6).

## 3. Core model

Each link tracks the state x = [θ, ω] (phase and angular-frequency
offset) with an extended Kalman filter:

    x_{t+1} = F x_t + w_t,   F = [[1, T],[0, 1]],   w ~ N(0, Q),

Q from both oscillators' datasheet noise over interval T; the
measurement is the two-way half-difference (reciprocity cancels the
channel phase) with covariance R set by the pilot's SNR·length
product. Two structural facts drive everything downstream: the
covariance recursion P → FPFᵀ + Q (coast) / Joseph update (service)
is deterministic and computable ex ante; and corrections are applied
L intervals late, so post-correction error is estimation error
integrated over the actuation horizon.

**Error floor.** The steady-state residual decomposes as

    σ² ≈ q_θ + (σ_ω⁺ · L·T)² + σ_track²,

drift + latency + tracking, with σ_ω⁺ the steady-state (Riccati)
frequency-posterior standard deviation. Verified by ablation.
A fourth, empirically identified term — white multipath resampling
noise σ_rs ≈ 153 mrad per capture on the default channel — explains
the filter's measured overconfidence: residual-at-service ÷ threshold
= ×1.45/×1.18/×1.12 at budgets 200/314/600 mrad, matching
√(b² + σ_rs²)/b with no fitted parameter.

## 4. Results

### 4.1 The coast-time law (posterior as predictor)

Setting the accumulated uncertainty equal to a phase budget b and
solving for the service interval,

    q_θ·(τ/T) + (σ_ω⁺·(τ + L·T))² = b²  →  τ_k,

with the exact form being the coast-cycle covariance fixed point
P* = update(predictᵐ(P*)). Validation across {OCXO, TCXO, SDR} ×
budgets {0.2, 0.314, 0.6 rad} × latencies {1, 2, 4} × seeds 0–2:
**25,972 of 26,094 individual coasting gaps predicted exactly
(99.5%)**, coast times spanning 1–111 intervals; exactness is
N-independent (99.2/99.5/99.6% at N = 6/10/14). Honest caveat: the
deployed scheduler thresholds on the same recursion, so exactness is
partly by construction; the substantive claim is that the recursion
is reconstructible from public specifications without simulation.

### 4.2 Supply, demand, and the airtime wall

Station k demands service at rate T/τ_k; a channel carrying C
exchanges/interval has supply C; ρ = C / Σ_k(T/τ_k) is the
supply/demand ratio. ρ alone orders 438 grid runs about twice as well
as the naive capacity/(N−1) axis (isotonic R² 0.577 vs 0.306) but is
not sufficient. The corrected **two-parameter model** adds a
feasibility gate (the §3 floor caps achievable gain independent of
capacity — SDR-class fleets floor near 1.1 rad and can never cohere
at 50 ms intervals) and prices demand at the budget line. Frozen and
blind-tested: **21/24 in the clean regime (latency 1, feasible
fleets: classification 8/8, plateaus 8/8 within 7 points, knees ±1),
20/24 extrapolating to N = 16/20 never seen in validation**; the
model also transfers to blind prediction of spectral efficiency
(9–10/12 per category) and detection range at comparable accuracy.
Its two failure modes are mechanistic findings: (a) one infeasible
station poisons worst-first scheduling (permanent maximal urgency,
fruitless capacity consumption) — i.e., *membership control is
required by the theory*; (b) at latency ≥ 2, measured plateaus fall
24–29 points below open-loop covariance growth — an unresolved
closed-loop amplification (delayed-PLL literature adjacent; open
problem).

Measured airtime walls (sync demand = 100% of frame): uniform
scheduling N ≈ 6.2; posterior scheduling N ≈ 14.1; genie N ≈ 17.9.
At N = 20 under contention, a uniformly synced array achieves 29% of
its ideal detection range; a posterior-scheduled one 93–98%.

### 4.3 Membership control (posterior as gatekeeper — and its limit)

Three rules compared at N = 10 (capacity 2 of 9) and scaled to
N ∈ {6, 10, 14, 20} with capacity ∝ links, five metrics (beam
quality; counted detection; mean and 95%-likely spectral efficiency;
detection range; net throughput = (1 − sync airtime) × rate):

- **Posterior gate** (bench if posterior σ exceeds threshold): loses
  mean beam quality (9.0% vs all-in 11.6% — a random-phase station
  still contributes +1/N² incoherent power on average) but removes
  the fade tail, winning reliability metrics (+21 detection points
  at capacity 2, positive in 39/40 grid cells; mechanism decomposed:
  ~88% of the lift is receive-side noise pruning; a power-matched
  control excludes power bookkeeping as the cause).
- **One-bit rule**: the oracle gate |wrap(θ)| ≤ π/2 is *identically*
  sign(cos θ) ≥ 0 — one feedback bit per station per interval
  (test-verified identity). It attains oracle membership: 26.4% gain
  vs the posterior gate's 9.0%; retains 68% of the oracle-minus-
  all-in gap at 10% bit-error rate; hysteresis degrades it
  (alignment windows are shorter than the smoothing). **At every N
  and under every metric it is first or effectively tied-first**;
  it collects 94% of the ideal N^(3/4) range growth at N = 20, vs
  55% for all-in.
- **Two-tier combining** ("demote, don't discard", after Qin et al.
  2024): S = |Σ_coh y|² + Σ_bench |y|² keeps benched stations' echo
  power without their phase; per-tier threshold recalibration.
  Dominates both hard benching and all-in in every regime tested.

Two scale results worth emphasis: at fixed power, **unmanaged array
growth reduces detection (95.5% → 90.6%, N = 10 → 20)** while managed
arrays reach ~99.7%; and in the clutter-limited regime the posterior
gate *inverts* (10% vs 69% detection; discarding receivers destroys
clutter discrimination) while the alignment bit survives (92/87%) —
currently single-seed, hardening scheduled (§8).

**Interpretation (the Kalman-posterior thread).** The posterior
schedules well and gates badly: it knows how *uncertain* a station is
but not which side of the phase line the station landed on, and a
lost station is an asset roughly half the time. The division of labor
supported by all of the above: posterior → airtime allocation;
measurement (1 bit) → beam membership; posterior covariance →
ex-ante prediction, with a known ×1.1–1.6 overconfidence correction
(§3).

### 4.4 Self-sensing (piggyback) synchronization

Observation model: any transmission from station i heard at j gives
φ_obs = wrap(θ_i − θ_j + φ_path) with φ_path constant while the
environment is static. A 3-state filter [θ, ω, φ_c] fuses per-frame
free observations (the sensing bursts the array already radiates;
one-way data observes only θ + φ_c) with sparse two-way anchors
every K intervals that re-pin the split. Results:

- N = 2: 42.2 ± 20.6 mrad at **0.48%** sync airtime vs 87.2 ± 4.3 at
  19.1% (two-way) and 168.4 ± 27.2 at 7.6% (micro-pilots).
- Real-waveform check: phase estimation from the actual OFDM sensing
  burst outperforms the dedicated Zadoff-Chu preamble (4.8 vs
  8.7 mrad/observation — shorter bursts accrue less intra-frame
  oscillator walk).
- Scaling: airtime advantage 16×/16×/16×/15×/13.9×/10.5× at
  N = 2/4/6/10/14/20; the two-way baseline saturates the frame at
  N = 20 (96% airtime, residuals → 636 mrad) while piggyback runs
  N = 30 at 13.9% airtime, 99.8% beam quality. An apparent error
  growth with N was root-caused to a configuration artifact
  (initial-CFO grid aliasing against the observation spacing;
  zero-offset control is flat in N at 50–61 mrad; mitigation:
  process-noise inflation).
- Environments: 13/13 (TDL D/E/A/B/C; delay spread 30–1000 ns —
  no effect; ray-traced two-ray, urban-LOS, urban-NLOS including a
  zero-direct-path placement synchronizing off six reflections at
  27 mrad). Weakened cell at N = 6: NLOS Rayleigh becomes comparable
  rather than better (290 ± 183 vs 246 ± 56 mrad), still at 16× less
  airtime.
- Boundaries with mechanisms: environmental motion ≥ 0.2 m/s breaks
  it (anchors must match the environment's coherence time; at K = 1
  the cost equals plain two-way). *Corrected 2026-08-18:* the
  previously reported interior-optimal observation rate was an
  acquisition transient in anchor-starved runs; in steady state,
  free observations are monotonically beneficial (≈ n^(−1/2) toward
  a ~37–45 mrad floor) and anchors can be arbitrarily sparse
  provided ~4 have occurred since acquisition.

### 4.5 SNR dependence and pilot length (the assigned question)

The full chain is derivable and validated (details and derivation in
`SNR_LAW.md`): SNR enters tracking only through the measurement
variance r(SNR) = 1/(4·SNR·L) + σ_rs². With L ≈ 2·10³, thermal phase
noise is ≤ 7 mrad even at 0 dB, so the operating range is *flat*:
measured residual 113 mrad ± ~1 from 10–30 dB against an 85 mrad
Riccati floor; collapse below the pilot detection threshold (222–292
mrad at 0–5 dB — a cliff, not a slope, located between 5–10 dB for
this preamble). Consequences, now all measured:

- Link budget spent on sync accuracy buys nothing in-regime.
- **Pilot shortening is the real lever and it is free:** sweeping
  pilot length 2047 → 127 at fixed 20 dB leaves residual unchanged
  (95–138 mrad, statistically flat) while sync airtime falls
  **19.1% → 3.0%** (floor set by cyclic prefix + guard, hence 6.4×
  rather than 16×). The fixed-SNR·L lever sweep gives the same
  curve — i.e., no SNR trade is needed at operating SNRs; SNR's role
  is detection margin as L shrinks.
- Over-long pilots actively hurt: 8191-sample pilots degrade both
  accuracy and variance (437 ± 252 mrad at 14 dB) via intra-frame
  oscillator walk — pilot length has an interior optimum and the
  default is ~6× past it.
- Since per-link SNR follows the path-loss law, the same chain gives
  residual/airtime *vs station spacing*: flat to the detection-cliff
  distance, then failure — a range wall.

## 5. Relation to prior work (adversarially checked, 2025–26)

Each contribution was checked by dedicated literature searches
instructed to find invalidating work; none was found, all claims were
narrowed to precise deltas. Key positioning: loop-delay penalties in
closed form exist (Wiener's discrete-PLL analysis, TCAS-II 2008) —
our delta is the posterior-explicit decomposition for OTA arrays;
interval-vs-oscillator-quality analysis exists (Mghabghab & Nanzer,
IEEE Access 2021) — our delta is exact per-station cadence,
heterogeneous fleets, blind protocol, and closure into a scheduling
theory; event-triggered sync exists reactively (Kramarev et al.,
MILCOM 2019) — ours is predictive; reflections-for-phase-sync now
exists receive-side, one-shot (Tong et al., arXiv:2603.13981, in
review) — ours is a closed transmit-oscillator loop with overhead
economics, and this window is **months**; 1-bit feedback beamforming
(Mudumbai et al.) does phase adjustment, not membership; Qin et al.
2024 gates on static offsets for communication only. Full citation
lists: `EXPERIMENT_SUMMARY.md` §§12–14.

## 6. Limitations

Simulation only; frozen or statistically-drawn channels (no
measured channel traces); no external interference; isotropic
antennas; no oscillator thermal transients (deterministic ramps
would violate the random-walk model underlying the coast law's
exactness); detection thresholds at Pfa 10⁻³; the clutter-regime
membership inversion is single-seed; the piggyback aliasing artifact
is mitigated, not mechanistically resolved in-capture.

## 7. Publication assessment (brief)

Three candidates survived the checks: membership + metric analysis
(novelty ~6.5/10, broadest evidence, recommended flagship), piggyback
sync (~6/10 and decaying — recommend staking quickly as a letter,
ideally with posterior-scheduled anchors as the unique mechanism),
and the ex-ante theory (~6.5/10, full-paper scope; any hardware
validation of a coast-time prediction would make it very strong).

## 8. Proposed next steps (ranked)

1. **Posterior-scheduled piggyback anchors** — fire two-way anchors
   from the filter's φ_c-uncertainty instead of a fixed K; unclaimed
   in the literature, fixes the misattribution bias, unifies §4.2
   and §4.4, and completes the letter.
2. **Multi-seed the clutter-regime inversion** (§4.3) — currently the
   weakest-evidenced load-bearing result.
3. **The latency-amplification theorem** (§4.2 failure mode b) —
   highest theory payoff; delayed-PLL literature is the entry point.
4. **Analytical near-optimality bound for the 1-bit rule** — short,
   pre-empts the obvious reviewer question.
5. **Pilot-length reduction in the default configuration** — a free
   ~6× sync-airtime win (§4.5), one config change plus regression.
6. **Online environment coherence-time estimation** from the filter's
   innovation sequence — converts the piggyback method's failure
   boundary into an adaptive anchor-rate controller.
7. **Hardware micro-demonstration** (2–3 SDRs, any single
   prediction) — upgrades all three papers; the field's 2026 output
   suggests hardware is becoming table stakes.

---

## Appendix: symbols

| symbol | meaning |
|---|---|
| θ, ω | phase and angular-frequency offset of a station vs reference |
| T | sync interval (50 ms) |
| L | actuation latency (intervals) — §4.5 reuses L for pilot length in samples, per context |
| q_θ | per-interval oscillator phase-walk variance (σ_pn²·f_s·T) |
| σ_ω⁺ | steady-state (Riccati/DARE) frequency-posterior std |
| b | phase budget (rad); 0.314 rad ≈ 90% beam quality |
| τ_k | station k's sustainable coasting interval |
| ρ | sync supply/demand ratio, C / Σ_k(T/τ_k) |
| σ_rs | per-capture multipath resampling noise (~153 mrad default channel) |
| r(SNR) | per-exchange phase measurement variance, 1/(4·SNR·L) + σ_rs² |
| G | beam quality (array gain), \|Σ_k e^{jθ_k}\|²/N² |
| S | two-tier detection statistic, \|Σ_coh y\|² + Σ_bench \|y\|² |
