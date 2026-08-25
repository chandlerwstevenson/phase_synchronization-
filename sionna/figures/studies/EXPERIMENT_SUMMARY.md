# Experiment summary — claims, experiments, results

One entry per workstream, 2026-08-14. All experiments used new files only; nothing pre-existing was modified. The full test suite was green at close (111 tests).

All results are from simulation: impairment-complete SDR physical layer, Sionna TDL channels, datasheet oscillator models (Rakon OCXO/TCXO classes), and USRP B2xx models. Carrier frequency was 915 MHz with 1 MHz bandwidth. There is no hardware validation yet. Unless otherwise noted, all reported results were independently re-run from scratch and reproduced exactly during Phase 4 verification.

---

## 1. Posterior-gated membership (idea #7 of RESEARCH_IDEAS.md)

**Initial claim.** If a station's phase residual drifts beyond roughly 90°, it can reduce the coherent sum. The Kalman posterior should predict when this happens, so using the posterior to decide array membership should recover some of the gain lost under synchronization contention.

**Experiment.** `gating_study.py` + tests. The scheduled star was instrumented externally using an EKF subclass that records the filter state. The modified run was bit-identical to the unmodified run and was verified by the tests. Membership variants (all-in, posterior gate, soft (e^{-\sigma^2/2}), oracle, and greedy) were evaluated on the same synchronization runs. A weighted counted-detection pipeline was also used, with a separate H0 threshold for each membership policy. N=10 star, synchronization capacity of 2 out of 9 stations, seeds 0–2.

**Result.** The array-gain part of the claim did not hold. Posterior gating gave 9.0% mean array gain versus 11.6% for all-in. A station with a random phase does not necessarily reduce the average gain; it mainly adds incoherent power. Only stations that are consistently anti-phase actually subtract from the coherent sum, and the existing π-branch check already prevents that for acquired stations.

The detection result was different. Counted edge detection increased from 77.0% to 97.6% for seed 0 and from 82.4% to 96.2% for seed 1, despite the lower mean array gain.

## 2. Detection/gain dissociation

**Initial claim.** The increase in detection probability despite the decrease in mean array gain is systematic and is not just an artifact of how transmit power is counted.

**Experiment.** `gating_dissociation_study.py` + tests. Synchronization capacities {1–4} and seeds {0–4} were evaluated over 40 waypoint-cells. The study separated transmit-only, receive-only, and combined gating. A power-matched all-in control scaled transmit power by ×0.924 to match the mean beam power.

**Result.** The detection-probability difference (gate − all-in) was positive in 39/40 cells. The average improvement was +17.8±10.5 percentage points at capacity 1 and +21.0±10.5 points at capacity 2. The difference approached zero as synchronization contention disappeared. Mean array gain was lower for the gated case throughout.

The main mechanism was receive-side stream pruning. Receive gating alone accounted for +18.9 of the +21.4 point improvement, about 88%. Removing transmit fades contributed another +9.7 points. Matching the mean transmit power did not remove the detection advantage.

The result reproduced exactly on re-run and the same gain/detection dissociation was observed with ray-traced propagation in Phase 4.

## 3. 1-bit opportunistic membership

**Initial claim.** The oracle membership policy gives 26–31% gain compared with about 9% for the posterior gate. If the useful information is mainly whether a station is currently aligned or anti-aligned, perhaps a single feedback bit is enough to capture most of the oracle benefit.

**Experiment.** `opportunistic_membership_study.py` + tests. Policies using 1–2 feedback bits per station per interval were evaluated with bit-error rates (\epsilon\in{0,0.1,0.2}). A hysteresis variant was also tested. Counted detection was evaluated at seed 0.

**Result.** The oracle membership rule is exactly equivalent to a 1-bit sign decision:

[
\operatorname{sign}(\cos\theta)
\quad\Longleftrightarrow\quad
|\theta|\leq\pi/2.
]

At (\epsilon=0.1), the 1-bit policy captured 68% of the oracle-minus-all-in gain gap, giving 21.7% gain. Detection matched the oracle at 98.3% for both policies.

Hysteresis made the result worse: the gap fraction dropped from 1.00 to 0.41. In this setup, spinning stations benefit from a memoryless decision rather than hysteresis.

Relevant prior work includes Mudumbai's 1-bit feedback beamforming work and Qin et al. 2024.

## 4. Doppler channel-decorrelation coast term

**Initial claim.** Scheduled coasting should become worse as channel Doppler increases. The original measurements suggested phase error increasing from 145 to 227 mrad at 3 m/s for scheduled synchronization while uniform synchronization stayed roughly constant. The hypothesis was that the error-floor model was missing a channel-decorrelation term that could be derived in the same way as the latency term.

**Experiment.** `doppler_coast_study.py` + tests. Doppler was swept from 0–3 m/s over seeds 0–2. A reciprocity-bias structure-function diagnostic was used, along with two candidate corrections to the error-floor model.

**Result.** The original observation was caused by noise in seed 0. Across seeds 0–2, scheduled synchronization increased from 192 to 208 mrad while uniform synchronization increased from 117 to 128 mrad; both changes were only about 8–9%.

**CORRECTION (2026-08-24): the attribution in this paragraph is wrong — see the correction block at the end of this section.** The dominant bias was white per-capture multipath resampling noise, about 100 mrad RMS, rather than Jakes channel decorrelation. The structure function was flat, and the Jakes prediction of 145 mrad at a one-interval gap was not observed; the measured value was 101 mrad.

The useful result is that resampling noise, rather than oscillator drift, appears to dominate the star's error floor under these conditions. This gives a possible additional term for the error-floor analysis. On NLOS TDL-A, the trend also reverses: motion can improve uniform synchronization.

**CORRECTION (2026-08-24).** The two paragraphs above are right that the ~100 mrad per-exchange excess is real, flat in service gap, and not Jakes decorrelation. Their *attribution* to multipath resampling is wrong, established by controls run later:

- With oscillators frozen, toggling sample-timing jitter over frozen multipath changes the two-way measurement by **< 0.05 mrad**. The discrete channel is shift-invariant and the correlator recovers integer sample shifts exactly, so the proposed mechanism cannot produce the observed magnitude at any jitter setting.
- A related mechanism does exist and was derived and validated: *clock-offset-induced fractional* resampling, worth **10–25 mrad** at 1 MHz bandwidth (independent of delay spread over 30–1000 ns), with saturation set by the delay-*separated* diffuse power rather than the total diffuse fraction. It is **conditionally** white: strongly correlated (lag-1 up to +0.95) at realistic clock offsets, white only once the per-exchange alignment step exceeds the waveform's ambiguity width.
- The ~100–130 mrad excess is instead **unmodeled oscillator noise**: it scales with oscillator class (≈28 / 130 / 650 mrad for oven-controlled / temperature-compensated / cheap oscillators) and is independent of channel model, signal-to-noise ratio, and jitter. Flicker frequency noise being modeled as white is the prime suspect. The apparent agreement between 153 mrad and the channel's diffuse-power estimate was a numerical coincidence.

Consequences elsewhere in this document: the overconfidence factor in §5 is real and measured, but its *explanation* is the oscillator-noise misspecification above (the √(b²+σ²)/b form remains a good approximation at small phase budgets and degrades at mid budgets); the flat-in-signal-to-noise conclusion in `SNR_LAW.md` is unaffected in substance, since the term that swamps thermal noise is now the oscillator floor rather than a channel floor. Sources: `../../phase_sync_idea/RESULTS_DOMINANCE.md`, `resampling_law.py`, `RESULTS_reversal.md`.

## 5. Ex-ante coast-time law

**Initial claim.** The error-floor model can be used to predict, for each station, how long it can coast between synchronization events using only oscillator ADEV, link budget, and frame parameters. The prediction should not require simulation or fitted constants.

**Experiment.** `coast_law.py` + tests. Three prediction methods were evaluated:

1. closed-form prediction;
2. per-interval DARE reconstruction;
3. a self-consistent coast-cycle fixed point,

[
P=\operatorname{update}(\operatorname{predict}^m(P)).
]

The validation grid used OCXO, TCXO, and SDR oscillator classes; budgets {0.2, 0.314, 0.6}; latencies {1, 2, 4}; seeds 0–2; and N=6. Every measured service gap was compared against the prediction.

**Result.** 25,972 of 26,094 coast gaps (99.5%) were predicted exactly. Coast times ranged from 1 to 111 intervals. The DARE reconstruction matched the running filter to 1.000.

There is an important caveat: the scheduler itself uses the same recursion to set its thresholds, so part of the agreement is by construction. The useful point is that the recursion can be reconstructed ahead of time from the system parameters.

The filter was also overconfident. The true residual at service exceeded the filter threshold by factors of 1.45, 1.18, and 1.12 for the TCXO at budgets 200, 314, and 600 mrad. This matches

[
\frac{\sqrt{\mathrm{threshold}^2+153^2}}{\mathrm{threshold}}
]

when the unmodeled resampling noise is included, without fitting an additional parameter.

## 6. Single-number universality

**Initial claim.** Synchronization contention might be summarized by a single dimensionless supply/demand ratio,

[
\rho =
\frac{\text{capacity}}
{\sum_k T/\tau_k},
]

with different operating conditions collapsing onto one master curve and a transition near (\rho\approx1).

**Experiment.** `coherence_collapse_study.py` + tests, using a 438-run cached grid. N={6,10,14}, five oscillator fleets, all synchronization capacities, latency and budget sweeps, and seeds 0–2. An isotonic master curve was fit against both the proposed (\rho) axis and the simpler capacity/(N−1) axis. Residuals and transition width versus N were also examined.

**Result.** The single-number model does not explain the data by itself. The proposed (\rho) axis performs better than the naive capacity/(N−1) axis (R² 0.577 versus 0.306; `frac_met` 0.519 versus 0.139), and the knee occurs in roughly the expected range, but the conditions do not collapse onto one curve.

The residuals have structure. Latency contributes about −16 points, tight synchronization budgets about +19, and fleet feasibility contributes roughly ±10–19 points. There was also no clear increase in transition sharpness with N.

These residuals motivated the two-parameter model used in the next experiment.

## 7. Blind prediction rounds

**Initial claim.** The coast-time model should be able to predict, without fitting to the test data, critical synchronization capacities, the shift between scheduled and uniform synchronization, and demand curves for heterogeneous oscillator fleets.

**Experiment.** `wall_prediction_study.py` followed by `phase_diagram_round2.py`, with tests and 219 fresh runs. Predictions were calculated and recorded before the measurements were run. No parameters were retuned after seeing the results.

For round 2, the model was frozen with three changes: κ-threshold demand, a feasibility gate based on the error-floor ceiling, and expected-phasor gain. The model was then tested on new N, synchronization budgets, latencies, oscillator fleets, and random seeds.

**Result.** Round 1 got 1/10 predictions right. Two systematic failures were identified. First, the model had no feasibility gate: the SDR fleet had an 1118 mrad floor and therefore could not achieve coherence regardless of synchronization capacity. Second, demand was being priced at the trigger line even when the observed knees tracked the synchronization budget line instead.

After these changes, round 2 achieved 22/32 overall and 21/24 in the clean regime (L=1, feasible fleets). Classification was correct in 8/8 cases, and the predicted plateaus were within 7 points in 8/8 cases. No fitted constants were used. The pooled 657-run model-axis R² increased from 0.577 to 0.746.

Two remaining error clusters are useful for the model. First, an infeasible station can poison the fleet under worst-first scheduling, which means the theory needs membership gating. This independently leads back to the membership results from Sections 1–3. Second, for L≥2, observed plateaus were 24–29 points below the open-loop prediction. This appears to involve delayed feedback and possibly delayed-PLL behavior and remains unresolved.

The clean-regime result was also robust to channel changes: the difference was ≤0.2 points under NLOS TDL-A and ≤0.5 m/s motion at the tested condition.

## 8. Clutter-referenced synchronization

**Initial claim.** Static environmental clutter, or sensing transmissions already being sent by the array, might provide phase observations without requiring additional dedicated synchronization airtime.

**Experiment.** `clutter_sync_study.py`, followed by `clutter_sync_ofdm.py` + tests. One-way observations were piggybacked onto the roughly 49 sensing bursts already transmitted in each interval, so they incurred no additional airtime. Only sparse two-way anchors, with cadence K, were charged as synchronization overhead.

The second round replaced the ZC preamble with a matched-filter estimator operating directly on the random-QPSK OFDM sensing burst, without genie equalization. The study was extended to an N=6 star and varied reference SNR, environmental motion, and observation rate.

**Result.** For N=2, the clutter-based method produced 42.2±20.6 mrad at 0.48% synchronization airtime, compared with 87.2 mrad at 19.1% for two-way synchronization. This is roughly 40× less airtime with lower residual error. Micro-pilots produced 168.4 mrad at 7.6% airtime.

The OFDM estimator also performed better than the ZC estimator on phase error: 4.8 versus 8.7 mrad per observation.

For N=6, the worst-station residual was 63.2±14 mrad, with 99.86% gain at 2.39% synchronization airtime. Scheduled two-way synchronization gave 191.7 mrad at 38.25% airtime, so the clutter-based method used about 16× less airtime.

There are clear limits. The reference needs to be within roughly 10–15 dB of the direct path. In a moving environment at 0.2–0.5 m/s, the benefit disappears and K=1 anchors are needed, bringing the cost back to the full two-way case.

**Correction (2026-08-18, `interior_optimum_study.py`):** an earlier version of this section reported an interior-optimal observation rate (a ~205 mrad misattribution bias at n=10 observations per interval). Rigorous re-measurement with run lengths scaled to ≥4 anchor cycles per cell showed that result was an acquisition transient in anchor-starved runs (60 intervals gave K=40 only two anchors ever). In steady state, free observations are monotonically beneficial (residual improves ≈ n^(−1/2) toward a ~37–45 mrad floor over the full 126-cell grid), and anchors can be arbitrarily sparse (bias ~11 mrad even at one anchor per 8 s) provided roughly four anchors have occurred since acquisition. The design rule is a convergence requirement on anchor count, not a ceiling on observation rate.

## 9. Hybrid two-tier combiner ("demote, don't discard")

**Initial claim.** A station that is not trusted for coherent combining does not necessarily need to be discarded. It may be better to put it into a noncoherent detection tier, preserving its signal energy without allowing its uncertain phase to reduce the coherent sum.

The proposed statistic is

[
S =
\left|\sum_{\mathrm{coh}}x_i\right|^2
+
\sum_{\mathrm{bench}}|x_i|^2.
]

The expectation was that this could outperform both all-in coherent combining and pure posterior gating.

**Experiment.** `hybrid_combiner_study.py` + tests. The two-tier statistic above was evaluated with separate H0 calibration for each variant. Transmit-side participation remained all-in. Five operating regimes were tested, ranging from synchronization-starved/unsaturated to regimes where coherent combining was strongly favored. Seeds 0–2 were used.

**Result.** The posterior-based hybrid combiner was at least as good as the better of all-in and gate-and-discard in every tested regime/waypoint.

For the starved-unsaturated case (capacity 2, 0.05 W), detection was:

* hybrid-posterior: 91.8/87.5%
* gate-and-discard: 87.2/78.8%
* all-in: 73.2/63.2%

The hybrid method was the only non-dominated combiner across the tested regimes. Pure noncoherent combining can match it under severe synchronization starvation, but loses when coherent combining becomes useful.

Interestingly, the hybrid-oracle version was slightly worse than the hybrid-posterior version at low power. The likely explanation is that the oracle membership changes too frequently, while the posterior-based membership keeps stations in the bench tier more persistently.

The resulting interpretation is different from the original selection idea: the synchronization posterior may be more useful for **partitioning** receivers into coherent and noncoherent tiers than for deciding whether they should be used at all. The result reproduced exactly on re-run and also survived the ray-traced tests.

## 10. Environment dependence of clutter-referenced synchronization

**Initial claim (challenge).** The clutter-referenced results might be specific to the one environment they were developed in (frozen TDL-D LOS, 100 ns delay spread), since "clutter" there is a statistical multipath composite rather than actual geometry.

**Experiment.** `environment_dependence_study.py` + tests. The N=2 headline comparison (K ∈ {5, 40}, static, seeds 0–2) was re-run across: all five TDL letters (D/E LOS Rician, A/B/C NLOS Rayleigh); a delay-spread sweep at TDL-D {30, 100, 300, 1000} ns; and three ray-traced geometric scenes via sionna-rt (two-ray, urban-LOS, urban-NLOS), including a placement with no direct path at all (6 reflected paths only) and a variant with the NLOS excess loss charged to the link budget.

**Result.** All 13 environments work: the piggyback scheme at K=40 (0.48% airtime) matches or beats the paid two-way baseline (19.1% airtime) in every one. The zero-direct-path urban placement synchronizes off reflections alone at 27 mrad. Delay spread has no effect (per-observation noise flat at ~8.7 mrad, 30–1000 ns). NLOS Rayleigh degrades both methods together, and deep-fade variance is the weak spot there (TDL-C: 91±103 mrad piggyback vs 140±106 two-way).

The governing parameter is environmental coherence **time**, not environment type: across every test in this project, only motion (0.2–0.5 m/s tap decorrelation) breaks the method, and it breaks it in every environment.

**Scaling with array size** (`piggyback_scaling_study.py`, then `piggyback_largen_study.py`, seeds 0–2): the airtime advantage holds at 16×/16×/16×/15×/14×/10.5× for N = 2/4/6/10/14/20 stations, because the free observations are broadcasts (flat in N) and only the rare per-station handshakes grow linearly — the same linear growth the standard method pays every interval. The standard method reaches 87% of all airtime at N=14 and effectively fails at N=20 (96% airtime, residuals exploding to 636 mrad); the piggyback scheme runs at 13.9% airtime with 99.8% beam quality even at N=30, and does not hit the airtime wall until beyond N≈100 (~0.46% per station).

The clock-error creep originally observed between N=6 and N=14 (63 → 212 mrad) turned out to be a simulation artifact, not a scaling effect: the configured initial frequency offsets form a grid (1500·s/(N−1) Hz), and stations whose offset is an integer multiple of 100 Hz alias cleanly against the 10 ms observation spacing while fractional ones wander — small arrays were accidentally all-integer. With a zero-offset control the per-station error is flat in N (50/53/61 mrad at N = 10/14/20), and a filter-side mitigation (inflating the modeled process noise) bounds the realistic case at a ~150 mrad plateau. The exact in-capture mechanism of the aliasing remains an open item. One environmental qualification at N=6: in the no-line-of-sight statistical channel the method becomes comparable to (no longer strictly better than) the paid baseline — still at 16× less airtime — while the line-of-sight and ray-traced urban scenes remain fully robust.

## 11. Multi-metric comparison

**Initial claim (question).** Do the method rankings change depending on the metric — probability of detection, spectral efficiency (communication throughput, mean and 95%-likely), beam quality, detection range, or net throughput (throughput × the fraction of airtime not spent synchronizing)?

**Experiment.** A shared metric layer (`metrics.py` + tests) scored three families on all metrics: membership methods (`metrics_membership_study.py`), scheduling policies and synchronization schemes (`multi_metric_study.py`), and the prediction model's transfer to the new metrics (`metric_theory_study.py`). Seeds 0–2 throughout; all runs re-executed, not re-scored from caches, except where noted in the logs.

**Result.** Rankings do invert, and the inversions follow two rules. Within the membership family, metrics that average (beam quality, mean throughput, range) reward keeping every station in, while metrics that count bad moments (detection, 95%-likely throughput) reward benching stale stations — the all-in array is even third on mean throughput but last on guaranteed throughput. Within the scheduling family, the quality metrics crown a configuration (round-robin at capacity 8) that demands 153% of the frame's airtime and therefore cannot exist; net throughput exposes it (zero) and instead crowns informed scheduling at low capacity.

Two methods turn out to be metric-independent winners. The 1-bit alignment-feedback membership is at or near the top of every metric (best beam quality, best mean and guaranteed throughput, best range, within 3 points of best detection — even with 10% feedback errors). And the clutter-referenced piggyback scheme wins every metric in its family outright: 99.9% beam quality, tied-best detection, and 58% more delivered data than the paid two-way baseline (net rate 8.30 vs 5.24 bits/s/Hz), because it combines the best residuals with 16× less synchronization overhead.

The prediction model also transfers: it blind-predicts throughput and detection-range plateaus and knees at essentially the same accuracy as it predicted beam quality, with every miss confined to the two known failure modes. One system-level warning from the transfer study: beam quality correlates only moderately (~0.5–0.6) with the metrics users experience, and *negatively* (−0.23) with net throughput — tuning a system to beam quality over-buys synchronization by 2–3× for most users.

## 12. Array-size scaling of everything

**Initial claim (challenge).** Most results were measured at a single array size (N=10 for membership, N=6 for the coast law, N=2 for the environment sweep); they might not survive scaling.

**Experiment.** Four studies (`membership_scaling_study.py`, `scheduling_scaling_study.py`, `piggyback_largen_study.py`, `theory_nscaling_study.py` + tests and caches) re-ran each family at N = 6, 10, 14, 20 (piggyback to 30; theory blind-extrapolated to 16 and 20), with sync capacity scaled proportionally to keep contention comparable and detection power matched across N where cross-size comparison required it.

**Result.** Everything survives, and most results strengthen with size:

- *Membership:* the 1-bit method stays the best-on-every-metric winner at all four sizes, its detection advantage growing from +12 to +38 points. Sharpest finding: at fixed power, adding stations to an unmanaged contended array makes detection *worse* (95.5% → 90.6% from 10 to 20 stations) while managed arrays climb to ~99.7%; only the 1-bit method collects the growing-array prize (94% of the ideal range growth at N=20 vs 55% for all-in).
- *Scheduling:* the airtime wall measured per policy — uniform synchronization stops fitting the frame at ~6 stations, posterior-driven scheduling at ~14, perfect information at ~18 (a 2.3× wall shift). A uniformly-synced 20-station array achieves 29% of its ideal detection range; a scheduled one 93–98%. The quality-metric pathology worsens with size: at N=20 the best-beam-quality configuration demands 104% of the frame.
- *Theory:* the coast-time law stays exact at every size (99.2/99.5/99.6% of gaps at N=6/10/14); the frozen model blind-extrapolates to 16- and 20-station arrays at 20/24, with all four misses being the known conservative demand-overpricing; the per-exchange multipath noise floor is flat in N (98 → 90 mrad).
- *Piggyback:* covered in Section 10's scaling paragraph (artifact root-caused; per-station quality flat in N; the standard method fails at N=20 while piggyback runs N=30 at 13.9% airtime).

## 13. Final novelty verdicts and paper choice (2026-08-15)

Three adversarial literature checks (2025–2026 sweep) scored the three paper candidates. None was killed:

- *Clutter/self-sensing sync:* 6/10 and falling — a March 2026 preprint (Tong et al., arXiv:2603.13981, in review at IEEE TWC) already uses reflections for receive-side phase synchronization, and two other 2025–26 works cover the Kalman-in-frame-with-overhead-math and hardware sensing-assisted-calibration pillars. The composite (closed transmit-clock loop + zero-marginal-airtime observations + overhead economics) is still unclaimed, but the window is months. Strongest additions: a small hardware demo, or posterior-scheduled anchors (untouched by anyone).
- *Membership + metric inversions:* 6.5/10 — the central claims (membership as a control variable; one alignment bit ≈ oracle; unmanaged growth backfires; the metric-inversion analysis) have no equivalent found. Must wall off: over-the-air federated-learning age-based device selection (arXiv:2501.01828), hybrid coherent+noncoherent fusion (IEEE 2025), Qin et al. 2024, and the Mudumbai 1-bit line. Known fixes: multi-seed the clutter-inversion result; add a short near-optimality argument for the 1-bit rule.
- *Ex-ante sync supply/demand theory:* 6.5/10, full-paper scope — Mghabghab & Nanzer (IEEE Access 2021) own the "gain vs update interval vs oscillator quality" concept, so the paper lives on its composition: exact per-station cadence, zero fitted constants, blind-prediction protocol (absent from the subfield), feasibility gate, and measured scheduling walls.

**Recommendation:** the membership paper is the flagship (best evidence base, clearest unclaimed headline, slowest-moving competition); stake the clutter-sync claim quickly as a letter because its novelty is decaying; the theory work is the third, full-length paper, made near-unanswerable by hardware validation of even one coast-time prediction. Note the twist that frames the flagship: the winning membership mechanism is the one that does *not* use the Kalman posterior — the posterior keeps its job as the scheduler (who gets sync airtime) and loses its job as the bouncer (who is in the beam), because only a measurement knows which side of the phase line a station actually landed on.

## 14. Literature risk checks

The initial claims that needed literature checking were:

1. the latency contribution to the synchronization error floor;
2. uncertainty-driven pilot scheduling;
3. clutter-referenced synchronization.

The literature search was done specifically to look for work that would invalidate these claims.

The claims were narrowed rather than eliminated.

For the latency term, delay-to-variance effects are already treated in PLL theory, including Wiener discrete-PLL loop-delay analysis and work by Wang et al. in *Metrologia* (2015). The narrower remaining claim is the posterior-explicit Kalman/DARE decomposition for OTA distributed arrays. Any paper using this result should explicitly distinguish it from the existing PLL literature.

For uncertainty-driven pilot scheduling, the closest work found was Kramarev et al., MILCOM 2019, which is event-triggered but reactive rather than predictive, and AoCSI pilot scheduling (arXiv:2503.13866), which considers a single link.

For clutter-referenced synchronization, the closest work found was distributed-ISAC OTA synchronization using scatterers (arXiv:2503.08920). That work focuses on TO/CFO rather than phase tracking and does not use the same transmit-side loop. There is also LoS-tracking drift compensation (arXiv:2510.13442), which is receive-only, and clutter-observation DCAR work in *Chinese Journal of Aeronautics* (2022), which uses coherent-on-receive batch processing.

Qin et al. 2024 is relevant to the 1-bit membership result and motivated the hybrid-combiner experiment, but it does not invalidate the measured results here.

---

## Standing caveats

All results are from impairment-complete simulation, not hardware measurements. The default channel is frozen TDL-D LOS; NLOS and motion were tested where noted. There is no external interference, antennas are isotropic, and oscillator thermal transients are not modeled.

The detection pipeline uses an empirical (P_{FA}=10^{-3}) threshold. The carrier is 915 MHz with 1 MHz bandwidth.

The ray-traced clutter-limited detection variant (RT "Part C": CPI pipeline with ground clutter at 65 dB CNR) produced an important qualification of Section 2: in that regime the posterior gate *inverts* (10.0% Pd vs 69.2% all-in — discarding receivers destroys clutter discrimination once the threshold is clutter-set rather than noise-set), while oracle/1-bit membership still wins decisively (91.7/86.7%). This result is preliminary (120 trials, single seed, window-Pfa 10⁻²) but changes the recommendation: under clutter, membership must be driven by alignment (the 1-bit rule) or use the hybrid two-tier combiner — uncertainty-only gating is a noise-limited-regime result. All other reported results have been independently re-run and reproduced.
