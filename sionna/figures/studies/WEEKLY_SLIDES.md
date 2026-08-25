# Weekly update — slide outline

*One slide per section: a few bullets, one result, one equation.
Figure suggestions reference `figures/studies/` files. The full
written version is `WEEKLY_UPDATE.md`.*

---

## Slide 1 — Problem and setup

- Distributed array of N stations, each with its own oscillator;
  coherent operation requires per-station phase error held small
- Sync is bought with two-way exchanges that occupy the shared
  channel — airtime is the scarce resource
- Everything below: impairment-complete simulation, 915 MHz,
  datasheet oscillators, multi-seed, independently re-run

**Equation** (what a two-way exchange measures — reciprocity cancels
the channel):

    ½·wrap(φ_AB − φ_BA) = θ_A − θ_B

**Result:** N−1 exchanges per 50 ms interval → sync consumes 19% of
the channel at N=2 and stops fitting entirely near N≈6 (naive).

*Figure: scheduling_deployment_map.png*

---

## Slide 2 — The error floor (what a sync loop can hold)

- Steady residual decomposes into drift + latency + tracking
- Latency term: corrections arrive L intervals late, so the
  frequency-estimate error integrates over the horizon
- [CORRECTED 2026-08-24: the fourth term below is real in magnitude
  but is unmodeled *oscillator* noise, not multipath resampling —
  jitter over frozen multipath measures < 0.05 mrad; a genuine
  clock-offset-driven fractional-resampling term exists at 10–25 mrad.
  See WEEKLY_UPDATE.md's correction notice.]
- Verified by ablation; a fourth term (multipath resampling,
  ~153 mrad/capture, SNR-independent) found by measurement

**Equation:**

    σ² ≈ q_θ + (σ_ω⁺·L·T)² + σ_track²

**Result:** the loop *sees* phase to 12 mrad but *holds* only
~70 mrad — cadence + latency dominate, not estimation.

*Figure: theory_filter_overconfidence.png*

---

## Slide 3 — The coast-time law (Kalman posterior as predictor)

- The filter's covariance recursion is deterministic → each station's
  sustainable coasting interval is computable from datasheets + link
  budget, zero fitted constants
- Exact form: the coast-cycle covariance fixed point
  P* = update(predictᵐ(P*))

**Equation** (solve for τ):

    q_θ·(τ/T) + (σ_ω⁺·(τ + L·T))² = b²

**Result: 25,972 / 26,094 individual coast gaps predicted exactly
(99.5%)**, spanning 1–111 intervals; exactness independent of N.

*Figure: theory_coast_predicted_vs_measured.png*

---

## Slide 4 — Supply/demand and blind-tested prediction

- Demand per station = T/τ_k; supply = channel capacity C
- One ratio ρ orders everything ~2× better than naive normalization;
  add a feasibility gate (error floor caps gain) → two-parameter model
- Frozen model, predictions printed *before* measurement

**Equation:**

    ρ = C / Σ_k (T/τ_k),    G_pred from w_k = e^{−s_k²/2}

**Result: blind score 21/24 in the clean regime; 20/24 extrapolating
to N=16/20 never seen in validation.** Both failure modes are
findings (membership required; latency-≥2 amplification open).

*Figures: theory_collapse_master_curve.png, theory_blind_scorecard.png*

---

## Slide 5 — Scheduling moves the airtime wall

- Rank stations by posterior uncertainty ÷ budget, service the worst
- The wall (sync demand = 100% of frame) measured per policy

**Equation** (urgency rule):

    service k* = argmax_k  σ_k^pred / b_k

**Result: wall at N≈6 (uniform) → N≈14 (posterior-scheduled) →
N≈18 (genie).** At N=20 under contention: uniform gets 29% of ideal
detection range, scheduled gets 93–98%.

*Figures: fig_airtime_wall.png, fig_range_vs_N.png*

---

## Slide 6 — Membership: the posterior's limit

- Original idea: bench stations whose posterior uncertainty is high
- It wins reliability metrics (+21 detection points) but *loses* mean
  gain — a random-phase station still adds incoherent power
- The posterior knows how lost a station is, not which side of the
  phase line it landed on

**Equation** (why benching costs average power):

    E|Σe^{jθ}|² = (Σw_k)² + Σ(1−w_k²),  w_k = e^{−σ_k²/2}

**Result:** posterior gate 9.0% gain vs all-in 11.6% vs oracle
membership 26.1% — the gap is phase knowledge, not uncertainty.

*Figures: membership_gain_by_policy.png, dissociation_gap_vs_capacity.png*

---

## Slide 7 — One bit closes the gap

- The oracle rule is exactly a sign test → one feedback bit per
  station per interval implements optimal membership
- Robust: 68% of the oracle gap retained at 10% bit errors;
  hysteresis hurts (alignment windows shorter than smoothing)

**Equation** (exact identity, test-verified):

    |wrap(θ)| ≤ π/2   ⟺   sign(cos θ) ≥ 0

**Result: first or tied-first under every metric at every N (6–20).**
At fixed power, unmanaged growth *reduces* detection (95.5→90.6%,
N=10→20) while 1-bit-managed arrays reach ~99.7%.

*Figures: onebit_gain_vs_bit_error.png, scaling_detection_vs_N.png*

---

## Slide 8 — Demote, don't discard (two-tier combining)

- Benched stations keep echo power in a square-law tier; only their
  unknown phase is discarded; thresholds recalibrated per tier split
- Dominates both hard benching and all-in in every regime tested
- Caution: in clutter-limited detection the *uncertainty-only* gate
  inverts (10% vs 69%); the alignment bit survives (92/87%) —
  single-seed, hardening scheduled

**Equation:**

    S = |Σ_coh y_j|² + Σ_bench |y_j|²

**Result:** starved regime detection 92% (two-tier) vs 87% (discard)
vs 73% (all-in).

*Figure: hybrid_combiner_by_regime.png*

---

## Slide 9 — Self-sensing sync (the airtime result)

- Sensing bursts the array already transmits double as one-way sync
  observations: φ_obs = wrap(θ_i − θ_j + φ_path), φ_path constant in
  a static environment
- 3-state filter [θ, ω, φ_c]; sparse two-way anchors (every K
  intervals) re-pin the oscillator/channel split
- Verified on the real OFDM waveform (4.8 vs 8.7 mrad — better than
  the dedicated preamble), 13/13 environments incl. no-line-of-sight
  urban (syncs off reflections alone)

**Equation** (airtime, anchors only):

    A = (N−1)·2·L_cap / (K·f_s·T)

**Result: 42 mrad at 0.48% airtime vs 87 mrad at 19.1% (two-way) —
~16× cheaper at better accuracy; runs N=30 at 13.9% airtime where
two-way fails at N=20.** Breaks only if the environment moves.

*Figures: clutter_residual_vs_cadence.png,
piggyback_airtime_wall_vs_n.png, scene_urban_nlos_layout.png*

---

## Slide 10 — SNR: what it does and doesn't do (assigned question)

- SNR enters tracking only through the measurement variance; pilot
  processing gain (L≈2000) + resampling floor swamp thermal noise
- → residual and airtime *flat in SNR* over the operating range;
  a hard detection cliff at 5–10 dB sets max station spacing

**Equation:**

    r(SNR) = 1/(4·SNR·L) + σ_rs²

**Result:** measured residual 113 mrad flat from 10–30 dB; collapse
below 10 dB (222–292 mrad).

---

## Slide 11 — The pilot-length lever (measured this week)

- The real SNR→airtime conversion is pilot shortening — and at
  operating SNR it's *free* (no SNR trade needed)
- Over-long pilots actively hurt: intra-frame oscillator walk
  (8191 samples → 437±252 mrad)

**Equation** (why it's free: thermal term stays negligible):

    1/(4·SNR·L) ≪ σ_rs²  for all tested (SNR, L)

**Result: sync airtime 19.1% → 3.0% at unchanged residual by
shortening the pilot 2047 → 127 at fixed 20 dB.** The default
configuration over-spends ~6×.

---

## Slide 12 — Takeaways and next steps

- Division of labor: **posterior → who gets airtime; measurement
  (1 bit) → who's in the beam; posterior covariance → ex-ante
  prediction** (with a known ×1.1–1.6 overconfidence correction)
- Publication: membership = flagship; piggyback = letter, window is
  months (competing receive-side preprint in review); theory = full
  paper
- Next: (1) posterior-scheduled piggyback anchors (unclaimed,
  unifies the two best results), (2) multi-seed the clutter
  inversion, (3) latency-amplification theorem, (4) 1-bit optimality
  bound, (5) shorten the default pilot, (6) SDR hardware micro-demo

**Result to lead the discussion with:** growing an unmanaged array
makes it worse at its job; one feedback bit per station fixes it at
every size and under every metric we can compute.
