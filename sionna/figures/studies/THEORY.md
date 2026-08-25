# Theory and mathematics behind the approaches

Companion to `EXPERIMENT_SUMMARY.md` (what was run and found) and
`FIGURES.md` (the figures), both in this directory. This file explains *why*
each approach works, with the math. Notation is defined at first use;
every formula here is the one actually implemented and validated in
the studies, not an idealization of it.

---

## 1. The physical problem

A distributed array is N radio stations, each with its own oscillator
(clock). To act as one coherent antenna — for beamforming, radar
detection, or communication — station k's carrier phase error
θ_k(t) relative to a common reference must stay small. Two facts make
this hard:

**Oscillators drift.** Each oscillator's phase follows a random walk
with a frequency that itself random-walks (white-FM plus
random-walk-FM noise, with flicker in between). Over an interval T,
the phase variance grows by

    σ_pn² · f_s · T          (white-FM phase random walk)

where σ_pn is the per-sample phase-walk standard deviation (derived
from the oscillator's datasheet Allan deviation via
σ_pn = ADEV(1s) · 2π f_c / √f_s, f_c the carrier frequency, f_s the
sample rate) plus slower terms from frequency wander. Better crystals
(oven-controlled OCXO, temperature-compensated TCXO, bare SDR
oscillators) differ by orders of magnitude in these constants.

**Measuring the drift costs radio time.** The standard cure is a
two-way exchange: stations A and B transmit sync waveforms to each
other; each one-way capture observes the oscillator offset *plus* the
propagation phase of the channel, but the channel phase is (nearly)
reciprocal, so the half-difference of the two directions,

    ½ · wrap(φ_AB − φ_BA)  =  θ_A − θ_B  + (asymmetry bias),

cancels the channel and isolates the clock offset. Each exchange
occupies the shared channel; at N stations a hub-and-spoke ("star")
needs N−1 exchanges per sync interval, and this airtime is the
central scarce resource of the whole project.

## 2. Tracking: the per-link Kalman filter

Each link runs an extended Kalman filter on the two-dimensional state
x = [phase offset θ, angular-frequency offset ω] with dynamics

    x_{t+1} = F x_t + w_t,     F = [[1, T], [0, 1]],
    w_t ~ N(0, Q),

where Q collects both oscillators' process noise (phase walk,
frequency walk, flicker innovation) over one interval T. The
measurement is the two-way half-difference, entering as
[cos θ, sin θ, ω] with covariance R set by the waveform's
signal-to-noise ratio and length (the phase-estimation
Cramér-Rao-style variance 1/(2·SNR·L) per component and a frequency
variance set by the pilot's time-energy). The filter's *covariance*
P — its own uncertainty about θ and ω — is the object every approach
below builds on.

Two properties matter:

- **The covariance recursion is deterministic.** P evolves by
  P → F P Fᵀ + Q on a coasting (no-measurement) interval and by the
  Joseph-form update on a serviced one. Given Q, R, and the schedule
  of services, P's entire trajectory is computable without running
  any signal through any channel. This is what makes the ex-ante
  theory possible.
- **Corrections arrive late.** A correction computed at interval t is
  applied at t+L (L = actuation latency in intervals). The controller
  forward-predicts through F^L, so what remains after a correction is
  the *estimation error* integrated over the horizon.

## 3. The error-floor decomposition

The steady-state phase residual of a closed-loop link separates into
three terms:

    σ_total²  ≈  σ_pn² f_s T        (drift: noise arriving after the
                                     correction was computed — no
                                     controller can remove it)
              +  (σ_ω⁺ · L · T)²    (latency: the frequency-estimate
                                     error σ_ω⁺ integrated over the
                                     actuation horizon)
              +  σ_track²           (estimation: how well the filter
                                     tracks at all, graded against an
                                     oracle capture)

σ_ω⁺ is the Kalman steady-state frequency-posterior standard
deviation — the solution of the discrete algebraic Riccati equation
(DARE) for (F, Q, R). The three terms were verified by ablation: kill
the RF impairments and only the latency term remains; kill the latency
and only tracking remains. The sharpest consequence: the loop can
*see* the phase to ~12 mrad yet only *hold* ~70 mrad, because
cadence + latency dominate. (Positioning note: classical delayed-PLL
theory prices loop delay into a minimum MSE — Wiener's discrete-PLL
loop-delay analysis — so the claim here is specifically the
posterior-explicit form with the DARE-computed σ_ω⁺, not that delay
matters at all.)

A fourth, initially unmodeled contributor was found by measurement:
**[SUPERSEDED 2026-08-24 — see the correction after this paragraph]**
multipath resampling noise — each capture's timing jitter
re-samples the frozen multipath composite, adding white per-exchange
phase noise of ~90–100 mrad (per link, independent of N and of the
service gap; confirmed by a flat structure function). It explains the
filter's measured overconfidence: the true residual at service exceeds
the filter's threshold by the factor √(threshold² + σ_rs²)/threshold
with σ_rs ≈ 153 mrad on this channel — with no fitted constants (a
1/(2K) Rice-factor estimate supplies σ_rs).

**Correction (2026-08-24).** The magnitude and flatness above are
measured facts; the mechanism is not. Frozen-oscillator controls show
sample-timing jitter over frozen multipath moves the two-way
measurement by < 0.05 mrad — the discrete channel is shift-invariant
and the correlator recovers integer shifts exactly. The correct
decomposition of that fourth contributor is:

    σ_extra²  =  σ_frac²(clock offset, delay-separated diffuse power)
                    ~ (10-25 mrad)², conditionally white
              +  σ_osc,unmodeled²(oscillator class)
                    ~ (28 / 130 / 650 mrad)² for ocxo / tcxo / sdr

The first term is a real, derived mechanism — the sample clock slides
*fractionally* through the multipath between exchanges whenever the
oscillators differ in rate — with an exact zero-fit predictor, but it
saturates at the *delay-separated* diffuse fraction (echoes co-located
with the line-of-sight peak move with it and cancel), giving 10–25 mrad
at 1 MHz bandwidth regardless of delay spread, and it is white only
when the per-exchange alignment step exceeds the waveform's ambiguity
width (otherwise strongly correlated, lag-1 up to +0.95). The second
term dominates and is *not* a channel effect at all: it is
class-proportional, channel-, signal-to-noise- and jitter-independent,
with flicker frequency noise modeled as white the prime suspect. The
overconfidence factor √(b² + σ²)/b therefore remains a useful
approximation (good at small phase budgets, degrading at mid budgets)
but its σ is oscillator-derived, not the Rice-factor estimate — that
numerical agreement was coincidental.

The candidate *motion* term did not survive: the proposed
channel-decorrelation contribution 2σ_c²(1 − J₀(2π f_D τ)) (Jakes
correlation at Doppler f_D over coast time τ) predicts a rising
structure function with service gap; the measured one is flat at all
tested speeds. The original supporting observation was single-seed
noise. Negative result, kept.

## 4. The coast-time law (ex-ante cadence)

Setting the error-floor expression equal to a phase budget b and
solving for the coast time τ gives station k's sustainable service
interval:

    σ_pn,k² f_s τ  +  (σ_ω⁺,k · (τ + L·T))²  =  b²   →   τ_k.

The closed form above is the intuition; the *exact* version — the one
that predicted 99.5% of ~26,000 individual coasting gaps precisely —
solves the actual covariance recursion self-consistently over a coast
cycle: find the fixed point

    P* = update( predictᵐ(P*) ),

i.e., the covariance that, after m coasting predicts and one
measurement update, returns to itself; the coast length m_k is the
largest m whose predicted phase standard deviation stays under the
trigger threshold. Every input (Q from datasheet Allan deviation, R
from the link budget and frame design) is known before any simulation.
The closed form runs 20–40% long for the best oscillators because it
truncates the cubic (m³/3) frequency-walk growth and the phase-
frequency cross-covariance; the fixed-point form does not. Exactness
is partly by construction — the deployed scheduler thresholds on the
same recursion — and the honest content is that the recursion is
reconstructible ex ante from public specifications.

## 5. Supply and demand: the two-parameter phase diagram

Each station demands service at rate T/τ_k (exchanges per interval).
A channel that carries C exchanges per interval has supply C. The
dimensionless ratio

    ρ = C / Σ_k (T/τ_k)

is the sync supply/demand ratio. ρ alone orders the data about twice
as well as the naive C/(N−1) axis, with its knee near 1, but it is
*not* sufficient — blind testing exposed two missing pieces, which
became the model:

1. **Feasibility gate.** Even at unlimited capacity, the error floor
   of Section 3 sets a minimum residual s_k^floor; if
   s_k^floor > b the station can never meet the budget (bare SDR
   oscillators at 50 ms intervals floor near 1.1 rad — unreachable).
   Gain must be predicted from the floor, not assumed to reach the
   budget.
2. **Pricing point.** Demand must be priced at the budget line
   τ_k(b), not the scheduler's early-trigger line τ_k(b/2): under
   contention the scheduler queues past the trigger without damage,
   so the sustainable knee tracks the budget. The round-2 model
   subsumes both with a single threshold multiplier κ ≥ 1 chosen so
   demand fits capacity: Σ_k 1/m_k(κ·f·b) ≤ C.

Predicted array gain then follows from the independent-phase
expectation: a station whose steady residual has standard deviation
s contributes expected phasor E[e^{jθ}] = e^{−s²/2}, and

    G_pred = [ (Σ_k w_k)² + Σ_k (1 − w_k²) ] / N²,   w_k = e^{−s_k²/2},

(the second term is the incoherent power of the phase spread). This
frozen model, with zero fitted constants, blind-predicted
reachability 8/8, plateaus 8/8, and knees within ±1 for feasible
low-latency fleets, including extrapolation to array sizes it never
saw. Its two failure modes are physics, not noise: a worst-first
scheduler gives an infeasible station permanent maximal urgency (it
eats capacity fruitlessly — membership control is *required* by the
theory), and at latency L ≥ 2 a closed-loop delayed-feedback
amplification depresses plateaus 24–29 points below the open-loop
prediction (open problem; the delayed-PLL literature is the adjacent
theory).

## 6. Scheduling: spending airtime where the posterior says

The scheduler ranks links by urgency = (predicted phase standard
deviation)/(budget) and services the worst offenders up to capacity,
forcing acquisition first. Variants: round-robin (uninformed
rotation), oracle (ranks by true residual — a genie bound), and a
myopic restless-bandit index (one-step growth of the budget-violation
probability 2Q(b/σ) — which *underperforms* the simple threshold under
severe contention because it chases already-blown links). Airtime is
counted as exchanges actually performed × the exchange's fraction of
the frame. The measured consequence of informed scheduling is the
airtime wall moving: uniform sync demand crosses 100% of the frame at
N ≈ 6, posterior scheduling at N ≈ 14, the genie at N ≈ 18.

## 7. Membership: who is allowed in the coherent sum

A station's contribution to the beam is its phasor e^{jθ_k}. Three
mathematically distinct membership rules:

- **Posterior gate.** Bench station k when its posterior standard
  deviation σ_k exceeds a threshold. Zero feedback cost. But a
  *random-phase* station still contributes +1/N² incoherent power on
  average (E|Σ|² picks up the variance term), so benching *lowers
  mean* gain; what benching buys is the removal of deep fades — a
  variance effect, visible in reliability metrics (counted detection,
  95%-likely throughput), not averages. And the posterior only knows
  *how uncertain* the filter is, not *where* the phase actually is.
- **1-bit alignment rule.** The oracle gate "bench if |wrap(θ_k)| >
  π/2" is *identically* the sign test sign(cos θ_k) ≥ 0 — one bit per
  station per interval, physically obtainable the same way the
  existing π-branch check works (a beacon comparison). With bit-error
  rate ε the bit is flipped i.i.d.; measured, ε = 0.1 still retains
  68% of the oracle-minus-all-in gain gap. Hysteresis (require two
  consecutive aligned bits) *hurts*, because a spinning station's
  alignment windows are shorter than the smoothing.
- **Two-tier combining ("demote, don't discard").** On receive,
  benched stations need not be discarded: form
  S = |Σ_coh y_j|² + Σ_bench |y_j|², i.e., coherent combining of the
  trusted tier plus square-law (noncoherent) addition of the rest.
  Noncoherent addition keeps each benched station's echo *power*
  while discarding its (unknown) phase, so S dominates both extremes:
  it equals the coherent statistic when nobody is benched and the
  noncoherent one when everybody is. The detection threshold must be
  recalibrated per membership rule because the null distribution of S
  changes with the tier split.

Regime dependence, mechanistically: when detection is limited by
thermal noise, dropping a benched station's receive stream removes
its noise from the combiner and lowers the threshold — most of the
detection lift is this receive-side pruning. When detection is
limited by ground clutter, the threshold is set by clutter, pruning
buys nothing, and discarding stations destroys degrees of freedom —
the uncertainty gate inverts, while the alignment bit (which keeps
momentarily-aligned stations) survives.

## 8. Clutter-referenced synchronization (piggyback)

A one-way capture of any transmission from station i at station j
observes

    φ_obs = wrap( θ_i − θ_j + φ_path ),

where φ_path is the propagation phase of the direct path plus all
multipath ("clutter") — *constant while the environment is static*.
The array transmits sensing bursts anyway (~49 per 50 ms interval),
so these observations are free: no marginal airtime. The estimation
structure is a 3-state Kalman filter on [θ, ω, φ_c]:

- one-way observations see only the *sum* θ + φ_c (an observability
  deficit — the split is unidentifiable from one-way data alone);
- sparse two-way anchors, every K intervals, measure θ alone (the
  half-difference cancels φ_c) and re-pin the split.

Because the free observations arrive several times per interval, the
loop's effective latency drops below the two-way loop's floor, which
is why residuals *improve* while airtime falls ~16×. The airtime is
anchors only: (N−1) · 2 · capture / (K · interval) — linear in N like
every two-way scheme, but divided by K; the free broadcasts are flat
in N (one transmission serves all listeners).

Boundaries, with their math: (i) the scheme inherits the environment's
coherence *time* — if the taps decorrelate (motion), φ_path is no
longer constant and anchors must come as fast as the environment
changes (at K = 1 the cost equals plain two-way); (ii) the θ/φ_c
split must *converge* before anchors can be sparse — the split
estimate needs roughly four anchors after acquisition, and until
then a high observation rate locks the sum tightly and exposes the
whole split error as a phase bias (an earlier version of this
section reported this as an interior-optimal observation rate; the
corrected steady-state measurement, `interior_optimum_study.py`, is
monotone: more free observations always help, ≈ n^(−1/2), and the
per-observation leakage into the channel state scales as
√(q_ψ/(n·r)), making the accumulated split wander n-independent —
the ideal filter predicts no interior optimum and the corrected
measurement agrees); (iii) the
observation is aliased against the substep spacing — a frequency
offset that is an integer number of cycles per substep looks
stationary (this produced the N-scaling artifact; the mitigation is
honest process-noise inflation, and the zero-offset control shows
per-station quality flat in N).

## 9. Detection mathematics (how "counted detection" works)

Array gain: G = |Σ_k e^{jθ_k}|² / N², computed from measured
residuals. Detection signal-to-noise of a passive target scales as

    SNR_N = N³ · G² · SNR_1

(transmit focusing contributes N²G on the target, coherent receive
combining another N·G — sync errors hurt both legs, hence G²). The
counted pipeline makes no Gaussian shortcut: every Monte-Carlo trial
draws one time-column of measured residuals, builds the bistatic echo
amplitudes through the radar equation with a Swerling-1 (complex
Gaussian) radar cross-section draw, adds thermal noise samples at
kT₀·F·f_s, matched-filters each receive stream at the hypothesized
delay, combines (coherently, weighted, or two-tier), and compares to
a threshold calibrated *empirically* from target-absent trials of the
same statistic at the stated false-alarm rate. Detection range comes
from inverting the link budget at the required Swerling-1
signal-to-noise for the target detection/false-alarm pair.

## 10. Communication metrics

Spectral efficiency to a user at distance d_k from station k:
per residual draw, received amplitude A = Σ_k w_k c_k e^{jθ_k}
(c_k the path amplitudes), signal-to-noise = |A|²·P / noise power, and

    SE = log₂(1 + SNR),

reported as the mean and the 95%-likely value (the rate exceeded 95%
of the time — the tail metric that punishes fades). Because log
compresses, strong links saturate: conditions far apart in G can be
identical in SE (why beam quality over-buys sync ~2–3× for nearby
users). The system-level metric is

    net throughput = (1 − sync airtime fraction) × mean SE,

which is what re-ranks the scheduling and piggyback families: sync
airtime is communication time not spent.

## 11. Why the metric inversions happen (one paragraph)

Averaging metrics (mean gain, mean SE, range) reward expected power,
and a random-phase station has positive expected power — so all-in
wins averages. Reliability metrics (counted detection at a threshold,
95%-likely SE) integrate the *lower tail*, which is dominated by the
occasional destructive alignment of stale stations — so membership
control wins tails. The two families of metrics are moments of the
same distribution pulled in opposite directions by the same
stations; no ranking can satisfy both unless a method removes the
tail without paying average power — which is exactly what the 1-bit
rule (keep the momentarily-aligned) and the two-tier combiner (keep
the power, drop the phase) are constructed to do, and why they are
the only metric-independent winners.

---

*Numbers quoted here (99.5% exact gaps, 16× airtime, wall positions,
etc.) are the measured values from `EXPERIMENT_SUMMARY.md`; formulas
are as implemented in `ota_sync/`, `coast_law.py`,
`phase_diagram_round2.py`, `clutter_sync_ofdm.py`, `metrics.py`, and
the study scripts. Everything is simulation-validated only.*
