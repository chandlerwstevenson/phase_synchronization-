# Potentially publishable results — with the math and the numbers

Each entry: the claim, the derivation, the measured evidence from this
repository, and how to reproduce it. Prior-art status is in
`RESEARCH_IDEAS.md`. Last literature check: 2026-08-05.

---

## 1. The latency-coupled closed-loop error floor (strongest)

**Claim.** The steady-state phase residual of a closed-loop OTA sync
link decomposes as

    sigma_total^2  ~=  sigma_pn^2 * f_s * T          (drift term)
                    +  (sigma_omega+ * L * T)^2       (LATENCY term)
                    +  sigma_track^2                  (estimation term)

where T is the sync interval, L the correction latency in intervals,
sigma_pn the per-sample white-FM phase-walk std, f_s the sample rate,
sigma_omega+ the Kalman steady-state frequency-posterior std, and
sigma_track the oracle-graded estimator error. The middle term — the
actuation latency L as a design parameter distinct from T, coupled to
the Kalman posterior — is the unpublished piece.

**Derivation of each term.**

*Drift term:* the oscillator pair's white-FM phase random walk has
per-sample variance sigma_pn^2; between two corrections (T seconds =
f_s*T samples) it accumulates variance sigma_pn^2 * f_s * T. No
controller can remove noise that arrives after its correction was
computed, so this is an irreducible floor set by cadence alone.
At defaults (sigma_pn = 2e-4 rad, f_s = 1e6, T = 0.05 s):
sqrt((2e-4)^2 * 5e4) = 44.7 mrad.

*Latency term:* the correction computed at interval k loads at k+L.
The controller forward-predicts through F^L using its own frequency
estimate, so what remains is the estimate ERROR integrated over the
horizon: phase error = omega_error * L * T, with omega_error ~
N(0, sigma_omega+^2). At defaults the filter's frequency posterior is
~0.12 Hz, L = 1, T = 0.05 s: 2*pi*0.12*0.05 = 38 mrad.

*Estimation term:* wrap(true - estimate) against the oracle capture
(ground truth = same capture with all noise/impairments removed):
12.0 mrad at defaults.

**Measured evidence (ablation, `--model sdr`, seed 0, 100 intervals).**
Each knob isolates one term:

    configuration                  predicted        measured residual
    full defaults                  sqrt(45^2+38^2+12^2) = 60    69.6 mrad
    --no-rf-impairments            latency only ~ 38            34.3 mrad
    --correction-latency 0         tracking only ~ 12           12.3 mrad

The three-way ablation confirms the terms separate as claimed (the
full-default gap 60 vs 69.6 is flicker + shadowing cross-terms).

**Sharpest single demonstration:** the loop SEES the phase to 12 mrad
but HOLDS only 70 mrad — a factor ~6 lost entirely to cadence+latency,
invisible in any analysis that assumes instantaneous corrections
(Rashid & Nanzer TWC 2022 Eq. 27 has drift*T and CRLB terms but no
actuation latency anywhere).

**Reproduce:** `.venv/bin/python simulation.py --plot-story`, then the
two ablation flags above. Oscillator-class dependence:
`--oscillator ocxo|tcxo|sdr` scales sigma_pn from datasheet ADEV via
sigma_pn = ADEV(1s) * 2*pi*f_c / sqrt(f_s).

**Open check before writing:** classical delayed-PLL/ADPLL literature
could hide an equivalent (frequency-uncertainty x delay)^2 term;
searched 2026-08-05, inconclusive (patents only).

---

## 2. What statistics-level validation hid about consensus sync
(two findings, one section)

### 2a. As published, DFPC over a real channel is bistable and can
lock the array into anti-phase

**Math.** Two nodes, relative oscillator phase theta = phi_a - phi_b,
common channel phase phi_c. Raw one-way observations (what the
published algorithm consenses on — it assumes no channel):

    o_a = wrap(-theta + phi_c),      o_b = wrap(theta + phi_c)

Symmetric update (each node retunes by half its observation):

    theta' = theta + (o_a - o_b)/2

Without wrapping, phi_c cancels and theta' = 0 in one step. With
wrapping, whenever |theta| + |phi_c| > pi exactly one of the two
observations wraps, adding +-pi to the difference, and the update maps
theta to +-pi instead of 0. And theta* = pi IS a fixed point: there
o_a = o_b = wrap(phi_c - pi), so the update is zero and every internal
statistic looks locked — two transmitters steadily canceling.

**Measured.** Seed 0 (channel phase -2.92 rad, initial theta 1.2):
hand arithmetic wrap(1.2+2.92) = -2.16, wrap(-1.2+2.92) = +1.72 gives
theta_1 = pi exactly; the full waveform simulation captures at
3009 mrad residual, 0.68% coherent gain. Seed sweep matches the
capture condition |theta_0| + |phi_c| > pi. With the reciprocity side
channel (half-difference cancels phi_c) the same algorithm reaches
153 mrad (unfiltered) / 83 mrad (Kalman).
Reproduce: `--model compare` (row "DFPC naive (as published)").

### 2b. Simultaneous (Jacobi) consensus updates cost 20-25 points of
array gain; turn-taking recovers it

**Mechanism.** On a shared node of degree d, simultaneous corrections
from d edges conflict; stability then requires degree weighting
(each endpoint applies c/(2*deg)), so only a fraction

    f_e = 1/(2*deg_p) + 1/(2*deg_q)   (= 1/2 on interior chain edges)

of each edge's correction lands per step, while the OTHER edges'
corrections arrive as disturbances between an edge's own updates.
Turn-taking (edges update on alternating steps — Gauss-Seidel) makes
simultaneous conflict impossible, so full-strength corrections are
stable with no weighting. An elected-root tree (each node retunes
fully toward its parent, PTP-BMCA style) removes the conflict
entirely.

**Measured (N-node mesh over the full physical layer — to our
knowledge the first waveform-level N-node consensus numbers; the DFPC
paper's own N-node results are statistics-level draws from its assumed
error distributions). Identical deployment/seed/airtime per column:**

    control law                N=4 gain    N=6 gain    N=6 edge residuals
    symmetric (as published)     74.0%       80.0%      314-1005 mrad
    alternating (masterless)     99.9%       95.8%      112-221 mrad
    directed (elected root)      99.9%       99.8%       24-76 mrad
    centralized star (ref.)      99.96%      ~99.9%     27-44 mrad/station

    array gain formula: G = |sum_i exp(j*phi_i)|^2 / N^2

DFPC itself in the same mesh harness: 61.5% (N=4) / 46.3% (N=6) at
MORE airtime (57.4% / 95.6% vs hybrid's 44.7% / 74.6%).

**Reproduce:** `--model dhybrid --stations 4 --mesh-control
symmetric|alternating|directed`; DFPC mesh via
`hybrid_calibration.mesh.run_dfpc_mesh`.

**Honesty note:** the turn-taking mechanism is textbook (asynchronous
gossip is standard in WSN time sync); the publishable part is the
measured COST in coherent gain for carrier-phase arrays, which that
subfield has never computed because it validates without a channel.

---

## 3. Uncertainty-driven pilot scheduling — BUILT AND MEASURED
(2026-08-05, `ota_sync/scheduled.py`, `smart_sync_study.py`)

**Idea.** Every measured deployment shows stations are heterogeneous
(path-loss SNR spread 15-23 dB; oscillator classes differ), yet every
method polls every station at the same fixed cadence. The filter
already knows each station's posterior variance; sync a station only
when its predicted uncertainty approaches the coherence limit.

**The scheduling rule falls out of result 1:** station k can coast
for tau_k, the solution of

    sigma_pn,k^2 * f_s * tau  +  (sigma_omega+,k * (tau + L*T))^2
        = theta_max^2            (theta_max = 314 mrad for 90% gain)

Pilot rate per station ~ 1/tau_k; total airtime sum_k C/tau_k instead
of the fixed-rate N*C/T. Stations with good crystals/links coast far
longer than T, so the airtime wall (which binds at N ~ 6-27 in every
sweep we ran) moves right by roughly the harmonic-mean ratio of
tau_k to T.

**Status:** intersection verified empty in the literature
(event-triggered Kalman filtering exists in control theory; fixed
cadence is universal in the sync literature). ~A day to build: the
scheduler loop plus fixed-vs-adaptive comparison on `--sweep-stations`.

**MEASURED RESULTS (N=6 star, 60 intervals, seed 0, counted detection
at two 1.2 km edge waypoints):**

    policy                  sync airtime   array gain   edge Pd
    uniform                     95.6%        99.4%      99.8 / 99.1%
    scheduled (flat 314mrad)    38.2%        98.7%      99.8 / 99.1%
    scheduled (task-aware)      44.0%        98.0%      99.8 / 99.1%

Headline: the scheduler returns ~55-60% of the channel at IDENTICAL
counted detection performance. Budget steering verifiably works
(200-mrad-budget stations hold ~130-140 mrad; 600-mrad stations ride
at 215-299), though at this uncontended operating point task-aware
does not beat flat on airtime — its value case is a contended channel
(more stations than capacity) and should be demonstrated there.

**Two failure modes discovered by simulation (paper material):**
(1) coasting during acquisition is fatal — residual CFO turns one
skipped interval into an unrecoverable drift, so the scheduler must
force service until settled; (2) coasting can drift a station across
the pi branch of the two-way half-difference — the 1-bit combining
check must be PERIODIC, not one-shot (independently discovered in the
decentralized mesh: same mechanism, two architectures).

**Original task-aware framing:** The clutter-limited
detection runs showed detection is a cliff: inside ~1 km even the
UNSYNCHRONIZED array detects (100 vs 95%); sync only earns its keep in
the coverage-edge annulus. So allocate pilots by (station uncertainty)
x (detection utility of that station's coherence for the current
coverage annulus / target hypothesis), priced in airtime. Headline:
same detection coverage at a fraction of the sync airtime; the airtime
wall moves right accordingly. All ingredients exist in the repo:
per-station Kalman uncertainty, the coasting-time rule from result 1,
airtime accounting, and the counted-Pd pipeline as the utility oracle.
Prior-art caveat: ISAC resource allocation is a hot field for POWER
and BEAMS - sync-as-allocated-resource appears open, but run a
dedicated literature pass before writing.

---

## Application stakes: what each result is worth in detection range
(drone-detection viability layer, 2026-08-05)

Detection SNR of a passive target = N^3 * G^2 x single station, with G
each method's measured array gain (transmit focusing N^2*G, receive
combining N*G — sync errors cost BOTH legs, hence squared). Swerling-1,
Pd >= 0.9 @ Pfa = 1e-6, -15 dBsm quadcopter, 1 W/station, N = 6:

    perfect sync         2900 m        (the N^3 prize: x3.83 range)
    star methods         2896-2899 m   (sync effectively free)
    mesh alternating     2839 m
    mesh symmetric       2594 m        <- result 2b costs 245-305 m
    KF-DFPC / DFPC       2185 / 1972 m <- published algorithm: -0.9 km
    free-running         1184 m        (no coherent prize at all)

This converts every sync result into operational terms:
  - result 1 (error floor): the star's 70 mrad residual costs ~0 m of
    range at N=6 — but the floor grows with cadence, and the sweep
    machinery can price any (interval, oscillator class) choice in
    meters (G ~ e^(-sigma^2), range ~ G^(1/2... via N^3G^2 ^ 1/4)).
  - result 2 (consensus): the published DFPC structure gives up ~930 m
    of detection range vs the star at identical hardware and airtime;
    turn-taking recovers ~870 m of it.
Reproduce: `.venv/bin/python detection_study.py`. Scope: link budget +
detection statistics over measured residuals; no waveform-level echo/
clutter/micro-Doppler yet.

## Detection phase (2026-08-05): NOT novel — supporting material

The waveform/RT/clutter detection layers reproduce known results:
sync-error impact on coherent MIMO radar detection, coherent-vs-
noncoherent tradeoffs, and "noncoherent fusion is robust without
sync" are all established (Godrich/Haimovich sensitivity line, MIMO
radar detection under sync errors, IEEE 2009-2021+). The one usable
sliver: that literature assumes Gaussian sync errors; our pipeline
feeds detection with residuals from an actual physical sync loop
(latency floor included) and counts detections at waveform level over
RT propagation with clutter/direct-path/micro-Doppler. Use it as the
OPERATIONAL-IMPACT section of papers 1-2 ("what 70 mrad costs in
meters and Pd"), never as a claim.

## Supporting material only (do not present as novel)

Hybrid's accuracy-per-airtime dominance (27 mrad @ 26% vs 83 @ 19% at
N=2; wall at N=27+ vs 6 at 200 ms); N-station scaling walls (concept:
Mudumbai 2007); oscillator profiles / TDD turnaround / Monte Carlo /
oracle grading — these make results 1-3 defensible, they are not
claims. Everything is simulation; no hardware validation exists.
