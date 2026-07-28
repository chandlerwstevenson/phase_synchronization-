# Literature review: closed-loop error floor of one-way OTA carrier phase sync

Scope: prior art for a prospective letter deriving a closed-form steady-state
closed-loop phase residual, sigma^2 ~ (white-FM walk rate)·T_sync +
(Kalman frequency-posterior std × correction latency)^2 + measurement terms,
with a white-PM/white-FM/flicker-FM oscillator model, validated by the
impairment-complete SDR simulation in this repository.

Method: multi-agent deep-research pass (2026-07-28), 5 search angles, 19
primary sources fetched, 95 claims extracted, top 25 adversarially verified
with 3-vote panels (25 confirmed, 0 refuted). Negative claims are verified
against the inspected full texts only, not the entire field.

## Verdict

No verified paper kills the contribution. One near-miss paper constrains the
framing and must be differentiated explicitly: **Rashid & Nanzer 2022**. The
specific latency-coupled closed-loop term appears unpublished in the
distributed-array sync literature through Dec 2025.

## Closest prior art (must-cite, in order of proximity)

1. **Rashid & Nanzer, IEEE TWC 2022 (arXiv:2201.08931)** — THE near-miss.
   Derives a closed-form steady-state residual phase error for
   Kalman-plus-consensus distributed sync (KF-DFPC) as a five-term
   root-sum-of-squares (Eq. 27): oscillator frequency-drift phase error
   (sigma_phi^f = 2π·sigma_f·T), frequency-estimation error, intra-interval
   drift, phase-estimation error, and phase jitter. Two-term Allan-deviation
   oscillator model (white FM + random-walk FM; flicker only qualitative).
   Measurement noise is per-measurement CRLB, not Kalman steady-state
   posterior. Corrections are applied instantaneously each iteration —
   "latency" appears once, qualitatively. Verified 3-0 against full text.
   Its first and third terms effectively reproduce two terms of our
   decomposition; the letter must state its delta against Eq. 27 explicitly.

2. **Mghabghab, Schlegel & Nanzer, IEEE Access 2021 (10.1109/ACCESS.2021.3071637)**
   — direct prior art for the open-loop drift term: phase-error growth
   between periodic wireless sync updates vs. VCO/PLL power-law phase noise
   and update interval, including an optimal-update-interval result, for
   N = 2–100 arrays. Zero occurrences of "Kalman" or "latency" (verified).
   Open-loop coherence analysis only.

3. **David & Brown, IEEE Aerospace 2015 (SPINLab)** — Allan-variance-to-Kalman
   parameter identification for two-radio carrier tracking (two-state
   [phase, frequency], Q(T) from q1/q2 white-FM/RW-FM terms; no flicker).
   Steady-state characterization is numerical/empirical only (error vs.
   Kalman ECM), open-loop prediction-error growth, no closed form, no
   latency. USRP N210 experiments were over COAX, not OTA (< 25° RMS at
   T0 = 50 ms at 900 MHz) — do not cite as OTA; our OTA validation is a
   delta against this benchmark.

4. **Brown, Prince & McNeill, SPAWC 2005** — continuous hardware-PLL
   master-beacon carrier sync; steady-state analysis is final-value-theorem
   algebra for deterministic ramps; no stochastic oscillator model.

5. **Nanzer-group experimental benchmarks** (architecture prior art, no
   competing analysis): Mghabghab & Nanzer TMTT 2020/2021 (open-loop two-node
   beamforming, 1.5 GHz, > 90% ideal gain under motion, continuous analog
   frequency transfer); Mghabghab/Schlegel/Nanzer IEEE TAP 2021
   (arXiv:2009.05127; 90 m outdoor, 7 days, coherence to 3 GHz, PI-controlled
   ranging); Ellison et al. TMTT 2020 (4.5 GHz ranging + frequency transfer,
   3.35 mm, < 18° phase error, CRLB bounds only).

6. **Merlo, Wagner, Lancaster & Nanzer, IEEE TMTT Dec 2025 (arXiv:2506.07267)**
   — state of the art: 60–70 ps timing, > 99% coherent gain via two-way time
   transfer (channel reciprocity; different topology from one-way).
   First-order finite-difference frequency estimator with CRLB bounds; no
   Kalman filter anywhere; resync-interval effect treated qualitatively.
   Confirms the gap is still open in the most recent flagship work.

## Firsthand read of Rashid & Nanzer (2026-07-28, full text in this directory)

Published version: IEEE TWC vol. 22 no. 4, Apr. 2023, pp. 2789--2802,
DOI 10.1109/TWC.2022.3213788. Confirms the verified claims and sharpens them:

- Eq. (27) is the DFPC (pure consensus) residual; for the Kalman variant
  (KF-DFPC) there is NO closed-form steady state -- performance is shown by
  simulation against the DFPC bound (Figs. 10-11). The Kalman steady-state
  (DARE) closed form is genuinely absent.
- Updates are instantaneous consensus state assignments each iteration; the
  single "latency" mention (Sec. IV) is qualitative, about CRLB observation
  windows. No actuation-delay parameter anywhere.
- Phase jitter is drawn i.i.d. per interval from integrated phase-noise
  power (Eq. 5, A = -53.46 dB, sigma = 2.7 mrad); it is not a continuous
  process. Oscillator model is the two-term ADEV law (Eq. 2), no flicker.
- The measurement model (Eq. 15) observes each node's oscillator signal
  directly in AWGN -- no propagation channel phase, no multipath, no packet
  detection, no SFO/timing. All channel/geometry phases are lumped into a
  uniform constant theta_e assumed handled elsewhere; identifiability is
  never addressed.
- Their "simulation" is statistics-level Monte Carlo: Gaussian draws from
  the assumed error distributions (Algorithm 1 injects N(0, sigma^2)
  directly). The theory is validated against its own assumptions, not
  against an independent waveform-level model -- a strong differentiation
  point for our impairment-complete IQ validation.
- Useful anchors to reuse: the 18-degree total-phase-error threshold for
  >90% coherent gain (their ref [4]); Fig. 10's optimal update interval
  (T = 20 ms at their parameters) -- the same design-chart shape our letter
  would extend with the latency axis; their refs [19]-[22] (Ouassal,
  Mghabghab) for the decentralized/open-loop lineage.
- Topology note: they treat N-node consensus; ours is the pairwise link.
  Frame the letter as the closed-loop building-block analysis (master-slave
  link), which their consensus layer would sit on top of.

## The available novelty delta (framing for the letter)

1. **The latency term**: (Kalman steady-state frequency-posterior std ×
   correction/actuation latency)^2, with latency a parameter DISTINCT from
   the sync interval. Caveat from verification (2-1 vote): Rashid & Nanzer's
   sigma_phi^m = 2π·f_c·sigma_fm·T already couples frequency-estimation
   uncertainty (CRLB) to the update interval T — so the claim must be
   precisely "actuation latency as a separate design parameter, coupled to
   the closed-loop steady-state posterior," or reviewers may read it as
   incremental.
2. **Kalman steady-state (Riccati/DARE) algebra** in the decomposition,
   replacing per-measurement CRLB bounds and purely numerical ECM curves.
3. **Analytical flicker-FM term** in the power-law oscillator model (absent
   from both closest analytical models; verified).
4. **Impairment-complete one-way OTA closed-loop validation** (prior closest
   experiments: cabled, continuous-analog, or two-way).

## Venues and activity

Closest works: IEEE TWC (2022), TMTT (2020, 2021, 2025), IEEE Access (2021),
TAP (2021), IEEE Aerospace (2015), SPAWC (2005); heavy arXiv posting. Field
active 2020–2026, dominated by the Nanzer group (Michigan State), with
Brown/SPINLab as the Kalman-sync foundation. A letter fits IEEE WCL or
LMWT/TMTT; adjacent context: IEEE JSAC 2023 cell-free massive MIMO OTA phase
sync prototype (measured > 20° LO drift per 7.5 ms between COTS RRUs —
usable as motivation), Ericsson 5G sync requirements (3GPP budgets are
microsecond-level time alignment, not carrier-phase coherence — motivates
why carrier-level analysis is not covered by standards work).

## Kill risks / required checks before submission

- **Classical delayed-feedback PLL / ADPLL loop-delay jitter literature**
  (Gardner-style) is the most plausible hiding place for an equivalent
  (frequency-uncertainty × loop-delay)^2 variance term under different
  vocabulary. NOT covered by this review — must be searched before writing.
- **Rashid & Nanzer 2023+ follow-ons** on distributed Kalman filtering were
  mentioned by verifiers but not examined — check for Eq. 27 extensions with
  actuation/broadcast latency.
- White Rabbit / IEEE 1588 servo phase-noise budgets and 3GPP/O-RAN sync
  specs: unexamined for servo-latency-coupled residual terms.
- Mudumbai/Madhow/Barriac foundational consensus-beamforming line and
  Quitin/Rahman distributed-MIMO frequency sync: produced no surviving
  verified claims — re-search directly before submission.
- Field publishes continuously (latest verified Dec 2025): re-run the search
  immediately before submission.

## Primary sources verified

- arXiv:2201.08931 (Rashid & Nanzer, TWC 2022)
- ieeexplore.ieee.org/document/9398687 (Mghabghab et al., Access 2021)
- spinlab.wpi.edu/pubs/David_AEROCONF_2015.pdf
- spinlab.wpi.edu/pubs/Brown_SPAWC_2005.pdf
- arXiv:2009.05127 (TAP 2021), TMTT 2020/2021 open-loop beamforming,
  Ellison et al. TMTT 2020
- arXiv:2506.07267 (Merlo et al., TMTT 2025)
- arXiv:2405.18384 (Shandi et al., consensus TWTT, ps-level)
- IEEE JSAC 2023 (10.1109/JSAC.2023.3276057), Ericsson 5G sync review,
  White Rabbit clock characteristics (CERN), MDPI Remote Sensing 17(3):497
  (diffusion KF radar network)
