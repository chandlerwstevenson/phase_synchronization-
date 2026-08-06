# Figure index for `figures/deck/` — self-contained reference

This document describes every figure in this folder. It is written to
be understandable with NO other context: a reader (human or LLM) who
has never seen this project should be able to describe, caption, or
reason about any figure from this file alone. Read the Background,
Glossary, and "Standard scoreboard format" sections first; the
per-figure entries rely on them.

---

## Background: what this project is

This is a **simulation study of carrier-phase synchronization for
distributed antenna arrays**, with drone detection as the driving
application. The setting: several radio base stations, spread over a
region hundreds of meters across, want to act together as ONE antenna
array. If all stations transmit with their radio carriers exactly in
phase at a target, their electromagnetic fields add constructively:
N stations deliver N-squared times one station's power at the target
(transmit focusing), and combining their received echoes coherently
adds another factor of N — so radar detection signal-to-noise ratio
scales as N-cubed. This only works if the stations' phases agree to
roughly 18 degrees (314 milliradians) at the carrier frequency.

The obstacle: each station generates its 915 MHz carrier from its own
imperfect crystal oscillator. Two free-running stations disagree in
frequency by tens to thousands of hertz and their phase difference
drifts continuously, wrapping around the full circle many times per
second. Without active synchronization, the coherent combining gain
does not exist. The project simulates and compares algorithms that
keep the stations phase-locked over the air (using radio pilot
signals between the stations), and then measures what each
algorithm's residual phase error costs in actual drone-detection
performance.

**How the simulation works:** everything is simulated at the level of
raw complex IQ samples (complex baseband, 1 megasample per second) in
PyTorch, with the multipath radio channel and noise supplied by the
Sionna library (3GPP TDL-D channel model). The 915 MHz carrier is
never synthesized explicitly; it enters through its effects (phase
and frequency offsets between oscillators, phase noise). A long list
of hardware impairments is always enabled: power-amplifier clipping,
12-bit converters, IQ imbalance, DC offset, automatic gain control,
timing jitter, sample-clock offset, oscillator phase noise of several
types, and slow shadow fading. All runs use random seed 0 unless
stated, making every number exactly reproducible. Estimation accuracy
is always graded against an "oracle" copy of the same signal with the
noise and impairments removed — ground truth, not self-assessment.

**The synchronization methods compared** (each has its own figure
group below):

1. **One-way**: a master station transmits a pilot; the slave
   measures it and retunes itself. Fails for coherent combining
   because a one-way measurement can only see the SUM of the
   oscillator phase difference and the radio channel's phase, so the
   loop unknowingly absorbs the channel phase into the oscillator.
2. **Two-way**: both stations exchange pilots across the same channel
   in both directions within each interval; half the DIFFERENCE of
   the two measurements cancels the channel phase exactly (channel
   reciprocity), so the actual oscillators align. The workhorse.
3. **Micro**: two-way, plus several very short "micro-pilots"
   (phase-only measurements) inserted between the full exchanges —
   corrects drift more often for better accuracy at higher channel
   cost.
4. **Hybrid**: mostly cheap one-way pilots tracked by a three-state
   filter that models the channel phase as its own state; an
   occasional two-way "anchor" exchange re-separates oscillator from
   channel. Nearly two-way accuracy at much lower channel cost.
5. **Decentralized hybrid (dhybrid)**: hybrid's estimation with
   symmetric control — each node retunes halfway toward the other,
   so there is no master station. At more than two stations it runs
   as a true mesh over a nearest-neighbor chain.
6. **DFPC and KF-DFPC**: a published consensus algorithm from the
   literature (Rashid and Nanzer, IEEE Transactions on Wireless
   Communications), reimplemented here over a full physical layer.
   Run exactly as published it is bistable over a real channel and
   can lock at anti-phase; granted the channel-free measurement
   exchange its publication assumes, it works, and its filtered
   variant matches this project's two-way method.

**The detection layers** (later figure groups): each method's
measured phase errors are fed into progressively more realistic
drone-detection simulations — a closed-form radar-equation layer, a
waveform-level Monte Carlo with ray-traced propagation (counting
actual detections of synthesized echoes), and a clutter-limited layer
with ground clutter, direct-path interference, and range–Doppler
processing.

**The scheduling layer** (final figure groups): synchronization
pilots consume the shared radio channel ("airtime"), which caps how
many stations one channel can keep coherent. The project's research
frontier treats sync airtime as a scarce resource: each station is
serviced with a pilot only when its Kalman filter's predicted
uncertainty approaches a per-station phase budget, and budgets can
follow the sensing task. These figures compare scheduling policies
under normal load, under channel overload, and with budgets that
track a moving drone.

## Glossary

- **Phase residual** (or just "residual"): the remaining phase error
  between two stations' oscillators after synchronization corrections
  are applied, in milliradians (mrad). 1000 mrad = 1 radian ≈ 57.3
  degrees. Lower is better.
- **314 mrad / 18 degrees**: the community threshold for "coherent
  enough" — total phase error at or below this keeps at least 90
  percent of the ideal combining gain in a large array. Drawn as a
  red line on most residual plots.
- **Coherent gain / array gain**: the fraction of the ideal combined
  power the array actually delivers, from 0 to 100 percent. For two
  stations it equals cos²(residual/2). 100 percent = perfectly in
  phase; roughly 0 percent = anti-phase (fields cancel); an
  unsynchronized pair averages 50 percent over time.
- **Airtime**: the fraction of the shared radio channel's time spent
  transmitting synchronization pilots instead of useful payload. A
  full two-way exchange costs 19.1 percent of the channel at the
  default 50-millisecond cadence; airtime scales with the number of
  stations because they share one channel.
- **Pilot**: a known reference waveform transmitted so the receiver
  can measure timing, frequency, and phase. Here pilots are
  Zadoff–Chu sequences (constant-envelope waveforms used in LTE/5G),
  about 4.6 milliseconds long; micro-pilots are 0.287 milliseconds.
- **Free-running**: no synchronization at all; the oscillators drift
  as their physics dictates. Shown in red on many plots as the
  "before" condition.
- **EKF**: extended Kalman filter — the estimator each link runs to
  track phase and frequency from noisy pilot measurements.
- **Sync interval / cadence**: how often the loop measures and
  corrects; default 50 milliseconds.
- **Correction latency**: corrections computed from a measurement are
  applied one interval late (as in real hardware). This latency turns
  out to dominate the achievable accuracy.
- **Oracle**: a noise-free, impairment-free copy of the same
  simulated capture, used as ground truth for grading estimates.
- **TDL-D**: the 3GPP standard tapped-delay-line multipath channel
  model, line-of-sight variant, 100 nanosecond delay spread.
- **Pd / Pfa**: probability of detection / probability of false
  alarm, for the drone-detection experiments.
- **RCS**: radar cross-section; the drone's is -15 dBsm (decibels
  relative to one square meter), representing a small quadcopter.
- **Waypoint**: one position along the drone's simulated flight path
  at which detection statistics are gathered.
- **N**: the number of base stations in the array.
- **Seed 0**: the fixed random seed; identical runs give identical
  numbers.

## Where to find the deployment geometry (2-D system layouts)

Four figures are top-down 2-D plan views of the base-station
configuration, useful as system-model illustrations: `net6_02.png`
(the six-station star with per-link distances and SNRs),
`mesh4_02.png` (the four-station masterless mesh with its chain
topology), `wavescene_02.png` (stations plus the drone's flight path
for the detection scenario), and `realistic_02.png` (the same for
the clutter-limited scenario). Two more figures one directory up
overlay the station positions on 2-D field maps
(`radiation_per_station.png` and `array_pattern_wide.png`). Note
that no hand-drawn schematic "system model" diagram exists; all of
these are generated from the simulation itself.

## Standard experimental conditions (unless an entry says otherwise)

- Carrier 915 MHz; complex-baseband simulation at 1 MS/s.
- Sync interval 50 ms; correction latency 1 interval; 100 iterations
  (5 seconds simulated) for single-link runs.
- Link SNR 20 dB (single-link runs); initial frequency offset 1500
  Hz; all hardware impairments enabled.
- Multi-station runs: stations placed uniformly at random in a
  500-meter-radius disc; per-link SNR follows log-distance path loss.
- Detection: N = 6 stations, 1 watt transmit power each, drone RCS
  -15 dBsm, drone altitude 60 m, station masts 15 m.

## The standard "scoreboard" figure format

The per-method figure groups (oneway, twoway, micro, hybrid, dhybrid,
dfpc, kfdfpc) all use the SAME three-panel layout with identical
axes, units, and thresholds, so methods can be compared side by side.
The combined `_01` file stacks all three panels with explanatory
titles; files `_02`–`_04` are clean single-panel copies in this
order:

- **Panel 1 (`_02`) — phase on the circle.** X-axis: time in seconds
  (0 to 5). Y-axis: phase offset in radians, wrapped to ±π. Red dots:
  the free-running phase, which rolls over constantly (thousands of
  wraps in 5 seconds appear as dense diagonal streaks). Blue: the
  same physics with the synchronization loop running, holding the
  phase near zero. Shows the problem and the fix at a glance.
- **Panel 2 (`_03`) — residual magnitude.** X-axis: time in seconds.
  Y-axis: absolute phase residual in milliradians on a LOGARITHMIC
  scale. A red dashed line marks 314 mrad (the 18-degree coherence
  threshold). An annotation box states the airtime cost. The initial
  transient (acquisition) decays into a steady-state band whose RMS
  is the method's headline accuracy number.
- **Panel 3 (`_04`) — two-station coherent gain.** X-axis: time in
  seconds. Y-axis: combining gain in percent (0–100), with a
  threshold line at 90 percent. Blue: the gain with the loop running.
  Red dots: the same gain reconstructed with NO synchronization —
  it sweeps the full range from 100 percent down to about 0 as the
  free-running phase drifts through anti-phase, making the
  before/after contrast vivid.

---

# Figure entries

## Group `story` — the one-way walkthrough
Command: `python simulation.py --plot-story`. Single link (one master,
one slave), one-way method, defaults as above. This is the guided
"understand the problem" figure rather than the standard scoreboard.

- **story_01.png** — Combined three-panel walkthrough. Panel 1: phase
  offset (radians, ±π) versus time (seconds) for two free-running
  radios — the phase wraps roughly 7,500 times in 5 seconds (dense
  red streaks) — overlaid with the same link held near zero by the
  closed loop (blue). Panel 2: the held residual's magnitude in
  milliradians (log scale) versus time, settling to 69.6 mrad RMS,
  below the 314 mrad threshold line. Panel 3: the estimator's error
  versus time — the Kalman filter tracks the phase to 12.0 mrad RMS.
  The gap between what the filter measures (12 mrad) and what the
  loop holds (70 mrad) is the project's central finding: accuracy is
  lost to correction cadence and latency, not to estimation.
- **story_02.png** — Clean copy of panel 1 (free-running rollover in
  red versus the closed loop in blue; radians versus seconds).
- **story_03.png** — Clean copy of panel 2 (residual magnitude, mrad,
  log scale, versus seconds, with the 314 mrad line).
- **story_04.png** — Clean copy of panel 3 (estimator tracking error,
  mrad, versus seconds; 12.0 mrad RMS).

## Group `iq` — one pilot at the raw-sample level
Command: `python simulation.py --plot-iq`. Shows a single pilot
capture end to end, at the level of individual IQ samples — the best
figures for explaining what a "pilot" physically is.

- **iq_01.png** — Combined six-panel figure containing all of the
  panels below.
- **iq_02.png** — The transmitted pilot frame: real part of the
  complex baseband signal versus sample index. Structure: a short
  training field (a 16-sample Zadoff–Chu sequence repeated 16 times,
  256 samples, used for detection and coarse frequency estimation)
  followed by two long training fields (2047-sample Zadoff–Chu
  sequences with 128-sample cyclic prefixes, used for fine frequency
  and phase). Total about 4.6 ms at 1 MS/s.
- **iq_03.png** — The transmitted signal's envelope (magnitude versus
  sample index): a flat line, because Zadoff–Chu sequences have
  constant envelope — the property that makes them friendly to power
  amplifiers and why LTE/5G use this waveform family.
- **iq_04.png** — The received frame after the multipath channel,
  noise, and all hardware impairments (versus sample index), plotted
  alongside its "oracle" twin — the same capture with noise and
  impairments removed — which the simulation uses as ground truth
  for grading estimates.
- **iq_05.png** — The received constellation (imaginary versus real
  part of the samples): a thickened ring rather than discrete
  points. Zadoff–Chu is a polyphase sequence, so its ideal
  constellation is a circle; noise and impairments thicken it.
- **iq_06.png** — The detection metric versus sample index: the
  short-training-field self-correlation (correlating the signal with
  itself 16 samples later), which plateaus when the repeating
  preamble is present. This is how the receiver decides a pilot has
  arrived.
- **iq_07.png** — The matched-filter (cross-correlation) output
  versus delay: a single sharp peak at the true arrival sample. This
  gives sample-level timing through multipath — something a pure
  tone could not provide.

## Group `oneway` — one-way method scoreboard
Command: `python simulation.py --plot` (one-way is the default
model). Standard scoreboard format (see above). Key numbers:
observable residual held at 69.6 mrad; TRUE oscillator-to-oscillator
offset 2906 mrad; coherent gain 2.2 percent; airtime 9.6 percent;
pilot detection 100 percent.

- **oneway_01.png** — Combined scoreboard. The crucial subtlety
  visible here: panel 2 shows TWO traces — the "observable" residual
  (oscillator plus channel phase, orange) held at about 70 mrad, and
  the true crystal-to-crystal offset (blue) parked at 2906 mrad,
  ABOVE the 314 mrad line. A one-way measurement cannot separate
  oscillator phase from channel phase, so the loop absorbed the
  channel phase into its correction: the link looks locked, but the
  actual oscillators are far apart, and panel 3 shows the
  consequence — a coherent gain of only 2.2 percent (the pair is
  near anti-phase). Lesson: one-way sync is fine for serving users
  (where the channel phase is part of what you want anyway) but
  useless for open-loop coherent combining toward a third point.
- **oneway_02.png** — Clean panel 1: free-running rollover versus
  the held observable.
- **oneway_03.png** — Clean panel 2: the observable residual (~70
  mrad) versus the true crystal offset (2906 mrad), mrad, log scale.
- **oneway_04.png** — Clean panel 3: coherent gain about 2.2 percent
  despite the "locked" link; free-running gain sweeping 100 to 0
  percent in red.

## Group `twoway` — two-way method scoreboard
Command: `python simulation.py --model twoway --plot`. Standard
scoreboard. Key numbers: true residual 83.5 mrad RMS; coherent gain
99.83 percent; airtime 19.1 percent (two ~4.8 ms captures per 50 ms
interval); detection 100 percent.

- **twoway_01.png** — Combined scoreboard. Both stations exchange
  pilots over the same channel each interval; half the difference of
  the two measured phases cancels the channel phase (reciprocity),
  so the TRUE oscillator offset is driven to 83.5 mrad — this time
  the residual plotted in panel 2 is the real crystal-to-crystal
  error, safely under 314 mrad, and panel 3 shows 99.83 percent
  coherent gain (3.99 times of the 4.0-times ideal for two
  stations).
- **twoway_02.png** — Clean panel 1 (rollover versus held phase).
- **twoway_03.png** — Clean panel 2 (83.5 mrad steady residual, log
  scale, 19.1 percent airtime annotated).
- **twoway_04.png** — Clean panel 3 (99.83 percent gain versus the
  free-running sweep).

## Group `micro` — micro-pilot method scoreboard
Command: `python simulation.py --model micro --plot`. Standard
scoreboard. Two-way, plus 4 short reciprocal phase-only micro-pilots
(287 samples = 0.287 ms each, versus 4.6 ms full frames) inserted
between the full exchanges; the receiver rides its tracking state
(derotates with its current frequency estimate, correlates at the
expected arrival) so micro-pilots need no detection preamble. Key
numbers: 28.1 mrad; 99.98 percent gain; 26.0 percent airtime.

- **micro_01.png** — Combined scoreboard: correcting five times per
  interval instead of once leaves less time for drift to accumulate,
  cutting the residual roughly threefold versus two-way (28.1 versus
  83.5 mrad) at the cost of 7 more points of channel airtime.
- **micro_02.png** — Clean panel 1.
- **micro_03.png** — Clean panel 2 (28.1 mrad steady band).
- **micro_04.png** — Clean panel 3 (99.98 percent gain).

## Group `hybrid` — hybrid method scoreboard
Command: `python simulation.py --model hybrid --plot`. Standard
scoreboard. A three-state Kalman filter tracks [oscillator phase,
frequency, channel phase]; cheap ONE-WAY micro-pilots observe only
the sum of oscillator and channel phase (enough to follow drift),
and a full TWO-WAY anchor exchange every 5 intervals re-separates
the two states. Key numbers: 33.8 mrad; 99.97 percent gain; 14.9
percent airtime — the best accuracy per unit of airtime of any
method tested.

- **hybrid_01.png** — Combined scoreboard: nearly two-way's accuracy
  at three-quarters of its channel cost, by letting a smarter
  observer extract information from cheaper pilots.
- **hybrid_02.png** — Clean panel 1.
- **hybrid_03.png** — Clean panel 2 (33.8 mrad steady band).
- **hybrid_04.png** — Clean panel 3 (99.97 percent gain).

## Group `dhybrid` — decentralized hybrid scoreboard
Command: `python simulation.py --model dhybrid --plot`. Standard
scoreboard. Same estimation as hybrid, but symmetric control: each
node retunes HALFWAY toward the other, so neither is the reference
and the pair meets at its average clock. Key numbers: 33.4 mrad;
99.97 percent gain; 14.9 percent airtime — statistically identical
to the centralized hybrid.

- **dhybrid_01.png** — Combined scoreboard: decentralizing the
  CONTROL is free at two stations. Gained: no master, no single
  point of failure. Lost: a fixed reference — the pair's common
  clock drifts freely (no station holds an absolute datum).
- **dhybrid_02.png** — Clean panel 1.
- **dhybrid_03.png** — Clean panel 2 (33.4 mrad steady band).
- **dhybrid_04.png** — Clean panel 3 (99.97 percent gain).

## Group `dfpc` — published consensus algorithm scoreboard
Command: `python simulation.py --model dfpc --plot`. Standard
scoreboard. DFPC is the consensus algorithm from Rashid and Nanzer
(IEEE Transactions on Wireless Communications): no master; each node
measures the other and retunes itself by half its estimated offset.
IMPORTANT: this standalone run grants the algorithm the channel-free
measurement exchange its publication assumes (the "reciprocity
steelman"). Key numbers under that assumption: 153.0 mrad; 99.42
percent gain; 19.1 percent airtime. Run WITHOUT that assumption (raw
one-way measurements over the real channel), the naive update is
bistable and locks at anti-phase — 3009 mrad and 0.68 percent gain —
which is visible only in the `compare` overlay figure, not here.

- **dfpc_01.png** — Combined scoreboard for DFPC with the channel
  cancelled for it: converges correctly to 153 mrad.
- **dfpc_02.png** — Clean panel 1.
- **dfpc_03.png** — Clean panel 2 (153 mrad steady band).
- **dfpc_04.png** — Clean panel 3 (99.42 percent gain).

## Group `kfdfpc` — Kalman-filtered consensus scoreboard
Command: `python simulation.py --model kfdfpc --plot`. Standard
scoreboard, same reciprocity assumption as `dfpc`. KF-DFPC adds a
Kalman filter to the consensus estimates. Key numbers: 82.8 mrad;
99.83 percent gain; 19.1 percent airtime — statistically the same
floor as this project's two-way method, showing the physical layer
(not the algorithm brand) sets the limit once the channel is handled.

- **kfdfpc_01.png** — Combined scoreboard (82.8 mrad).
- **kfdfpc_02.png** — Clean panel 1.
- **kfdfpc_03.png** — Clean panel 2.
- **kfdfpc_04.png** — Clean panel 3.

## Group `mesh4` — four-station decentralized mesh
Command: `python simulation.py --model dhybrid --stations 4 --plot`.
NOT the standard scoreboard: this is a network figure. Four stations
with no reference station synchronize over a nearest-neighbor chain
(a tree topology, which avoids phase-winding states), using the
"alternating" control law (edges take turns applying corrections —
still fully masterless). Periodic one-bit checks fix the sign
ambiguity of the two-way half-difference. Key numbers: per-edge
residuals 39.4 / 40.3 / 35.5 mrad on edges 0–1 / 1–2 / 2–3; array
coherent gain 99.91 percent; total airtime 44.7 percent; detection
100 percent.

- **mesh4_01.png** — Combined figure: deployment map with the chain
  topology drawn, per-edge residual trajectories (mrad, log scale,
  versus time), and the four-station array coherent gain versus time.
  Context: with the naive simultaneous-update control law used by
  the published consensus literature, the same mesh only reaches
  74 percent gain — the difference is purely the update schedule.
- **mesh4_02.png** — Clean deployment map: four station positions in
  the plane (meters) with the nearest-neighbor chain edges.
- **mesh4_03.png** — Clean per-edge residuals versus time (three
  traces, 35–40 mrad steady).
- **mesh4_04.png** — Clean array coherent gain versus time (99.91
  percent steady mean).

## Group `ideal` — the toy baseline
Command: `python simulation.py --model ideal --plot`. The project's
original simplified model, kept only for comparison: a pure tone in
white noise, NO multipath channel, NO hardware impairments, ideal
measurements into the same Kalman filter and correction loop. Key
numbers: phase error 1.6 mrad RMS; frequency error 0.33 rad/s RMS.
The gap between this 1.6 mrad and the realistic loop's 70 mrad is
the price of the physical layer. Four panels, not the scoreboard.

- **ideal_01.png** — Combined two-by-two figure of the four panels
  below.
- **ideal_02.png** — Pre-correction relative phase versus iteration:
  the true phase difference and the Kalman filter's estimate lying
  on top of it.
- **ideal_03.png** — Pre-correction relative frequency versus
  iteration: true versus estimated angular frequency.
- **ideal_04.png** — Phase residual after corrections are applied
  (radians versus iteration): the closed-loop error.
- **ideal_05.png** — Frequency residual after corrections (radians
  per second versus iteration).

## Group `diagnostics` — the receiver's full debugging view
Command: `python simulation.py --plot-all`. Six diagnostic panels for
a one-way run (same run as group `oneway`): everything the receiver
estimates internally. Header numbers: detection 100 percent, phase
tracking error 12.0 mrad RMS, frequency error 0.196 Hz RMS.

- **diagnostics_01.png** — Combined six-panel figure of the panels
  below.
- **diagnostics_02.png** — The observable phase (oscillator plus
  channel) versus time, with the filter's track over it.
- **diagnostics_03.png** — Decomposition of the measured phase into
  its oscillator component and its channel component versus time —
  showing what the one-way loop cannot separate on its own.
- **diagnostics_04.png** — Carrier-frequency-offset acquisition
  versus time on a symmetric-log axis: the coarse estimate (from the
  short training field, ±31 kHz range) and the fine estimate (from
  the long fields, sub-hertz) converging on the true 1500 Hz offset.
- **diagnostics_05.png** — The post-correction phase residual versus
  time (the same quantity as scoreboard panel 2).
- **diagnostics_06.png** — Pilot detection confidence per interval
  (the correlation metric against its threshold; 100 percent
  detected here).
- **diagnostics_07.png** — Packet timing: estimated arrival sample
  index per interval, showing the timing estimator following clock
  drift.

## Group `compare` — all methods head to head
Command: `python simulation.py --model compare --plot`. Runs SEVEN
configurations under identical physical conditions (same seed, same
channel realizations) and overlays their residual trajectories.

- **compare_01.png** — One axis: absolute phase residual in
  milliradians (log scale, roughly 10 to 3000) versus time in
  seconds, one colored trace per method, red dashed line at 314
  mrad. Six traces settle below the line: micro 28.1, decentralized
  hybrid 33.4, hybrid 33.8, KF-DFPC-with-reciprocity 82.8, two-way
  83.5, DFPC-with-reciprocity 153.0 mrad. One trace — naive DFPC
  exactly as published, measuring over the real channel — sits flat
  near 3009 mrad: it has CONVERGED, but onto the anti-phase fixed
  point of its wrapped symmetric update (coherent gain 0.68
  percent). This figure is the visual evidence that
  statistics-level validation (no channel simulated) could not have
  caught the failure.

## Group `arc` — the five-beat story in one figure
Command: `python narrative_arc_figure.py`. Not a simulation run: a
summary figure assembled from the measured seed-0 numbers documented
elsewhere in this file (single link, 50 ms cadence unless noted),
built to carry the presentation's five-step argument on one slide.
Method colors follow the deck (two-way blue, micro orange, hybrid
green — stepped to a colorblind-safe teal-green); one-way is gray
because it is disqualified, not a contender.

- **arc_01.png** — Three deliberately word-free panels covering all
  six methods (one-way, two-way, micro, hybrid, DFPC, KF-DFPC); the
  five-beat narration belongs in the caption or slide text. Panel 1
  (beats 1–3), "True residual": bar chart of TRUE oscillator
  residual per method on a log scale against the red 314 mrad line —
  one-way's bar towers at 2906 mrad (the loop *believes* it holds 70
  mrad, but the channel phase was absorbed; 2.2 percent gain);
  two-way 84, micro 28, hybrid 34; the published consensus baselines
  DFPC (153) and KF-DFPC (83) are their reciprocity-steelman numbers
  (run exactly as published over the real channel, DFPC instead
  locks anti-phase at 3009 mrad — see `compare_01.png`). Panel 2
  (beat 4), "Residual vs airtime": scatter — hybrid (34 mrad at 14.9
  percent) is ringed as the Pareto pick; KF-DFPC sits on top of
  two-way (same 19.1 percent airtime, 83 versus 84 mrad) so the two
  share a label; one-way appears as a gray X (invalid). Panel 3
  (beat 5), "Airtime vs array size": total pilot airtime versus
  station count N (per-link cost times N-1 links) against the
  100-percent-of-channel line — two-way, DFPC, and KF-DFPC share one
  line (identical 19.1 percent per-link cost); open markers show the
  largest N that fits at 50 ms (micro 4, two-way/DFPC/KF-DFPC 6,
  hybrid 7); one-way is a gray dashed line, cheapest to scale but
  invalid because its array never actually combines. Even the
  winning method hits the airtime wall, motivating the scheduling
  work.
- **arc_02.png** — Standalone bar chart of pilot airtime per link
  (percent of the shared channel) for the same six methods in the
  same colors: one-way 9.6, two-way 19.1, micro 26.0, hybrid 14.9,
  DFPC 19.1, KF-DFPC 19.1. Linear scale, value labels on the bars,
  no annotations.

## Group `net6` — a six-station network
Command: `python simulation.py --model twoway --stations 6 --plot`.
Six stations placed uniformly at random in a 500-meter-radius disc;
station 0 is the reference and each other station runs a two-way
link to it, time-multiplexed on ONE shared channel; per-link SNR
follows log-distance path loss. Key numbers: per-station residuals
66.2 / 76.5 / 72.5 / 70.0 / 68.2 mrad at distances 414–782 m; array
coherent gain 99.65 percent; total pilot airtime 95.6 percent of the
channel; detection 100 percent everywhere.

- **net6_01.png** — Combined three-panel figure: deployment map,
  per-station residuals, array gain. The headline: accuracy is fine
  at every station, but five two-way links already consume 95.6
  percent of the shared channel — airtime, not accuracy, is the
  scaling bottleneck.
- **net6_02.png** — Clean deployment map: station positions in
  meters, reference marked, each link annotated with its distance
  and SNR (22.2 dB at 414 m down to 14.8 dB at 782 m).
- **net6_03.png** — Clean per-station residual trajectories (mrad,
  log scale) versus time, all settling at 66–77 mrad.
- **net6_04.png** — Clean six-station array coherent gain versus
  time (99.65 percent steady mean, 90 percent threshold marked).

## Group `sweepT` — the cadence trade
Command: `python simulation.py --model twoway --sweep-interval
--plot`. Re-runs the two-way link at sync intervals of 10, 20, 50,
100, 200, and 500 ms. Measured: 32.8 / 48.6 / 83.5 / 137.1 / 243.5 /
1035.9 mrad residual and 95.6 / 47.8 / 19.1 / 9.6 / 4.8 / 1.9
percent airtime respectively.

- **sweepT_01.png** — Combined two-panel figure: accuracy versus
  interval and airtime versus interval (log x-axis in milliseconds),
  the 50 ms default marked. Longer intervals mean more drift between
  corrections but fewer pilots. Residual crosses the 314 mrad
  threshold between 200 and 500 ms. Shows that 50 ms is one point on
  a trade curve, not a special value.
- **sweepT_02.png** — Clean accuracy panel: steady residual (mrad)
  versus sync interval (ms, log scale).
- **sweepT_03.png** — Clean cost panel: pilot airtime (percent)
  versus sync interval.

## Group `sweepN` — the station-count trade
Command: `python simulation.py --model hybrid --sweep-stations
--plot`. Re-runs the hybrid network at N = 2, 4, 8, 12 stations.
Measured: array gain 99.98 / 99.95 / 99.91 / 99.90 percent; worst
station 26.1 / 30.7 / 45.6 / 45.6 mrad; total airtime 14.9 / 44.7 /
104.4 / 164.1 percent — the last two DO NOT FIT the channel.

- **sweepN_01.png** — Combined three-panel figure versus N: array
  gain (flat, near 100 percent), worst-station residual (rising but
  far below threshold), and total airtime rising LINEARLY through
  the 100-percent-of-channel line between N = 4 and N = 8. The
  "airtime wall": pilots share one channel, so beyond a certain N
  the pilots alone would consume more than the whole channel. This
  is the motivation for the scheduling work.
- **sweepN_02.png** — Clean array gain versus N.
- **sweepN_03.png** — Clean worst-station residual versus N.
- **sweepN_04.png** — Clean total airtime versus N with the
  100-percent line marked.

## Group `scalability` — the full two-dimensional verdict
Command: `python scalability_sweep.py`. Sweeps cadence (25, 50, 100,
200 ms) AND array size together for three methods (two-way, micro,
hybrid); for each method-cadence cell it computes the largest N whose
pilots fit the channel and then actually runs the network at that N.
Measured maximum fitting N per cadence — two-way: 3 / 6 / 11 / 21+;
micro: 2 / 4 / 8 / 16; hybrid: 4 / 7 / 14 / 27+. At 200 ms hybrid
holds 20 stations at 99.71 percent gain (worst station 75 mrad) on
70.8 percent airtime, while two-way at the same cadence has degraded
to 96.14 percent gain with its worst station at 255.8 mrad.

- **scalability_01.png** — Combined three-panel figure versus
  cadence, one line per method: maximum N that fits the channel;
  array gain at that maximum N; worst-station residual at that
  maximum N. The verdict: hybrid dominates the entire grid (largest
  arrays at equal or better accuracy); micro is the most accurate
  per link but hits the wall soonest.
- **scalability_02.png** — Clean maximum-fitting-N versus cadence,
  per method.
- **scalability_03.png** — Clean array gain at the maximum N versus
  cadence, per method.
- **scalability_04.png** — Clean worst-station residual at the
  maximum N versus cadence, per method.

## Group `detrange` — analytic detection ranges
Command: `python detection_study.py`. The quick closed-form detection
layer: each sync method's MEASURED array gain G is plugged into the
radar equation (detection SNR = N³ G² times one station;
Swerling-1 target statistics; Pd ≥ 0.9 at Pfa = 1e-6; N = 6, 1 W per
station, -15 dBsm drone). Reference points: one station alone
detects to 757 m; perfect six-station coherence reaches 2900 m (a
23.3 dB SNR advantage, 3.83 times the range). Measured ranges:
two-way 2896 m; micro and hybrid 2899 m; decentralized-hybrid
directed 2897 m; alternating 2839 m; symmetric 2594 m; KF-DFPC
2185 m; DFPC 1972 m; free-running 1184 m.

- **detrange_01.png** — Combined figure: probability-of-detection
  versus range curves (one per method, showing where each falls
  below Pd = 0.9) above a bar chart of the resulting detection
  ranges. Interpretation: the star methods are within 0.1 percent of
  the physical bound — synchronization is effectively free — while
  the published consensus algorithm gives up about 930 meters and
  free-running loses the coherent prize entirely.
- **detrange_02.png** — Clean Pd-versus-range curves per method.
- **detrange_03.png** — Clean detection-range bar chart per method.

## Group `wavescene` — counted waveform-level detection
Command: `python waveform_detection_study.py`. The rigorous detection
test: NO detection formulas anywhere. A drone flies a straight
2.4-km path across the six-station deployment at 60 m altitude; at
each of 12 waypoints, 1500 Monte Carlo trials synthesize the actual
echo sample streams (ray-traced station-to-drone propagation
including the ground bounce, random Swerling-1 radar cross-section,
the sync residuals taken from each method's ACTUAL measured run,
thermal noise), matched-filter them, combine coherently, and compare
to an EMPIRICAL threshold calibrated on target-absent trials
(achieved false-alarm rate re-measured: 1.00e-3 against a 1e-3
target). The transmitted sensing waveform is a 5G-style OFDM burst
(64 QPSK subcarriers, cyclic prefix, 1.04 ms). Measured combining
loss and Pd along the path: two-way and hybrid 0.00 dB loss, Pd
84–100 percent, IDENTICAL to perfect sync waypoint-for-waypoint;
decentralized alternating -0.33 dB; symmetric -1.08 dB;
free-running -9.24 dB with Pd as low as 19 percent.

- **wavescene_01.png** — Combined figure: (top) the scene map —
  station positions, the drone path with each waypoint colored by
  its measured probability of detection, and snapshots of the
  array's focusing direction as it re-steers to follow the drone;
  (middle) Pd versus waypoint for every method; (bottom) each
  method's waveform-measured combining loss in dB.
- **wavescene_02.png** — Clean scene map: a top-down 2-D plan
  view (axes in meters) of the six base-station positions and the
  drone's straight 2.4-km flight path, with each waypoint colored by
  its measured probability of detection, plus indicators of the
  direction the array focuses at three snapshots as it re-steers to
  follow the drone. Doubles as the system-configuration figure for
  the detection scenario.
- **wavescene_03.png** — Clean Pd-versus-waypoint curves per method.
- **wavescene_04.png** — Clean combining-loss bar chart per method.

## Group `realistic` — clutter-limited detection
Command: `python realistic_detection_study.py`. The field-realism
layer on top of the waveform test: gamma-distributed ground clutter
with internal motion injected into the raw streams (peak
clutter-to-noise 70.1 dB — the problem becomes clutter-limited,
matching real UHF drone radar); direct-path self-interference at
true delays removed by least-squares cancellation (46.1 dB above
noise before, 6.6 dB after); an aspect-dependent drone body RCS plus
rotor-blade micro-Doppler; a 64-pulse coherent processing interval
(38 ms) with range–Doppler processing; and cell-averaging CFAR
detection with the zero-Doppler clutter ridge notched, calibrated
empirically on target-absent trials that CONTAIN clutter. The drone
flies at 15 m/s. Result: all synchronized methods 100 percent Pd at
every waypoint; even free-running reaches 97–100 percent at this
geometry, because Doppler separation carries detection inside about
a kilometer — an honest finding that sync earns its keep at longer
range and at the coverage edge, not up close.

- **realistic_01.png** — Combined figure: the scene map; one example
  fused range–Doppler map after processing (x-axis Doppler, y-axis
  range; the bright vertical ridge at zero Doppler is ground
  clutter, notched out by the detector; the drone appears as a
  marked cell offset in Doppler by its motion); and Pd along the
  path per method.
- **realistic_02.png** — Clean scene map: a top-down 2-D plan
  view (axes in meters) of the six station positions and the drone's
  flight path for the clutter-limited scenario, in the same style as
  `wavescene_02.png`.
- **realistic_03.png** — Clean range–Doppler map with the clutter
  ridge and the detected drone cell marked.
- **realistic_04.png** — Clean Pd-along-path panel per method.

## Group `smart` — uncertainty-driven pilot scheduling
Command: `python smart_sync_study.py`. The scheduling idea: instead
of polling every station every interval (uniform), service a station
ONLY when its Kalman filter's PREDICTED phase uncertainty approaches
that station's phase budget; "task-aware" budgets are tight (200
mrad) for stations whose ray-traced propagation matters most at the
detection coverage edge and loose (600 mrad) elsewhere. Six-station
star, 60 intervals. Measured: uniform 95.6 percent airtime at 99.42
percent gain; scheduled with flat 314 mrad budgets 38.2 percent
airtime at 98.72 percent; task-aware 44.0 percent at 98.00 percent —
and counted edge detection is IDENTICAL for all three (99.8 / 99.1
percent at the two 1.2-km edge waypoints). Headline: half to
two-thirds of the channel returned at no detection cost.

- **smart_01.png** — Combined three-panel figure: (top) each
  station's residual sawing upward while coasting and dropping at
  each correction, plotted against its budget, with tick marks at
  the pilots actually transmitted — the whitespace between ticks is
  the reclaimed channel; (middle) horizontal bars of airtime used
  per policy, annotated with the array gain each held; (bottom)
  counted edge-detection bars per policy, visually identical.
- **smart_02.png** — Clean residual-sawtooth panel with budgets and
  pilot ticks.
- **smart_03.png** — Clean airtime-per-policy bars.
- **smart_04.png** — Clean edge-detection-per-policy bars.

## Group `contention` — scheduling on an overloaded channel
Command: `python contention_study.py`. Ten stations (nine links) with
the channel CAPPED below demand — at capacity 2, only 2 of the 9
desired exchanges fit per interval. Five policies compared: uniform
(always the same first links — the honest failure of fixed
allocation: it permanently starves the rest), round-robin (rotation),
scheduled (the uncertainty-driven rule), whittle (an index policy
from restless-bandit theory), and a genie "oracle" that ranks links
by their TRUE current error, which no real policy can observe.
Measured at capacity 2: uniform 10.5 percent array gain (starved
stations never even acquire), round-robin 84.8, scheduled 94.0,
whittle 64.2, oracle 97.8 percent. A noteworthy negative result: the
myopic whittle index UNDERPERFORMS the simple threshold rule under
overload because it keeps chasing links that are already too far
gone.

- **contention_01.png** — Combined two-panel figure versus channel
  capacity (2, 4, 8 exchanges per interval), one line per policy:
  (top) array coherent gain — uniform collapses, scheduled tracks
  the oracle; (bottom) mean counted edge detection.
- **contention_02.png** — Clean array-gain versus capacity panel.
- **contention_03.png** — Clean edge-detection versus capacity
  panel.

## Group `wall` — the array-size wall, per scheduling policy
Command: `python airtime_wall_study.py`. The channel is pinned at its
physical capacity (5 full exchanges per 50 ms interval) and the
array size is swept: N = 4, 6, 8, 10, 12. Measured array gain
(policy: N=6 / 8 / 10 / 12): uniform 99.4 / 58.9 / 27.0 / 31.2
percent — it holds until EXACTLY N = 6 (matching the arithmetic that
five 19.1-percent exchanges fill the channel) then collapses;
round-robin 99.4 / 99.1 / 98.4 / 98.6; scheduled 98.6 / 98.7 / 98.2
/ 98.4 while using only 38.6 / 52.0 / 65.0 / 77.3 percent airtime.
The largest N holding at least 90 percent gain: uniform 6; scheduled,
round-robin, and whittle all beyond 12 (past the end of the sweep).

- **wall_01.png** — Combined two-panel figure versus N, one line per
  policy: (top) array gain — the uniform line falls off a cliff
  after N = 6 while the others stay above 92 percent; (bottom)
  airtime used — uniform and round-robin saturate at 95.6 percent
  while scheduled climbs gradually, showing the headroom that moves
  the wall.
- **wall_02.png** — Clean array-gain versus N panel.
- **wall_03.png** — Clean airtime versus N panel.

## Group `sensloop` — budgets that follow the sensing target
Command: `python sensing_loop_study.py`. Closes the sensing-to-sync
loop: a drone crosses the deployment in 6 track segments (12
intervals each), and at each segment boundary the per-station phase
budgets are RE-ASSIGNED using the same ray-traced propagation legs
the detector uses — the stations most important for detecting the
drone at its CURRENT position get tight 200 mrad budgets, the rest
get loose 600 mrad budgets. Three policies: uniform, static-edge
(budgets set once for the coverage edge), and target-tracking
(budgets follow the drone). Measured: uniform 95.6 percent airtime
at 99.47 percent gain; static-edge 43.0 percent at 98.25; target-
tracking 42.2 percent at 98.20 — with counted detection at every one
of six waypoints matching across all three policies (99.3–100
percent). 56 percent of the channel returned. (At this uncontended
operating point tracking ties static budgets; the differentiating
experiment is tracking budgets on an overloaded channel.)

- **sensloop_01.png** — Combined three-panel figure: (top) each
  station's budget stepping up and down as the drone passes through
  the six segments, with that station's residual plotted beneath its
  budget line; (middle) counted detection probability at each of
  the six waypoints, grouped bars per policy; (bottom) airtime bars
  per policy annotated with array gain.
- **sensloop_02.png** — Clean stepped-budgets panel with residuals.
- **sensloop_03.png** — Clean per-waypoint detection bars per
  policy.
- **sensloop_04.png** — Clean airtime-per-policy bars.

---

## Related figures one directory up (`figures/`)

Not part of this folder but referenced by the same presentation:
`scene_overview.png`, `scene_rays.png`, `scene_side_profile.png`,
`scene_drone_closeup.png`, `scene_station_closeup.png` (3-D renders
of the ray-tracing scene: six 15-m masts, the drone path at 60 m
altitude, and the solver-traced line-of-sight plus ground-bounce
rays), `radiation_per_station.png` (six top-down 2-D maps, one per
station, of that station's illumination at drone altitude with the
mast position marked — the rings are two-ray ground-bounce
interference, since each station has a single omnidirectional
antenna), and
`array_pattern_wide.png` / `array_pattern_zoom.png` (top-down 2-D
maps of the coherent array's focusing pattern with the six station
positions, the flight path, and the focus marked: no far-field beam,
but a ~25 cm focal spot at 0 dB in speckle whose median is -8.2 dB,
matching the 1/N sparse-array prediction of -7.8 dB). Generated by `render_scene.py`
and `radiation_maps.py`.

## Regeneration

All figures in this folder were produced by running the repository's
plotting commands through the headless capture wrapper:

```
.venv/bin/python capture_figures.py <prefix> <script.py> [args...]
```

which saves every window the interactive scripts would open as
`<prefix>_NN.png`. The exact command for each group is given in its
section header. All commands use the project virtual environment
(`ota_sync/.venv`) and default to random seed 0.
