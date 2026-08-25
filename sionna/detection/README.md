# The drone detection and tracking code, explained

This document explains what the drone code does, file by file, in
plain language. The code lives in two places:

- `sionna/detection/` — the four-file package that does the actual
  radar work (this folder).
- `sionna/*_detection_study.py` and `sionna/sensing_loop_study.py` —
  the runnable scripts that drive the package and make the figures.

## What this code is for

The synchronization work measures how well a group of base stations
can align their transmit phases. This code answers the follow-up
question: **does that alignment actually buy you anything?** The test
case is detecting a small drone (a quadcopter, roughly the radar size
of a bird) flying across the deployment. The stations transmit
together, the drone reflects a tiny fraction of the energy, and every
station listens for the echo.

The payoff from synchronization is large and specific. With N
stations, perfect alignment multiplies the echo signal-to-noise ratio
by N³ compared to one station alone: N² from the transmissions adding
up in phase at the drone, and another N from adding the received
echoes in phase. Phase errors cut into both legs, so the measured
alignment quality G (0 to 1) enters **squared**:

    signal-to-noise ratio = N³ × G² × (single-station value)

Every experiment in this folder takes G from a *measured*
synchronization run — the phase errors are the real residuals the
sync simulators produce, not an assumed number.

The "tracking" part: the drone flies a straight path across the
deployment. The path is cut into waypoints, and detection is tested
at every waypoint, so the output is detection probability along the
whole flight. One study (`sensing_loop_study.py`) closes the loop:
as the drone moves, the synchronization schedule itself re-targets
which stations get tightly synchronized.

## The four layers, from simplest to most realistic

The package is built as four fidelity layers. Each layer keeps
everything the previous one had and removes one idealization. Every
layer states its remaining gaps in its own docstring.

### Layer 1 — `viability.py`: the link budget (math only, no waveforms)

Pure closed-form radar math. It takes a station count, a measured
alignment quality G, and radar assumptions (1 watt per station,
915 MHz, a drone radar cross-section of 0.03 square meters, 50 ms of
integration), and computes:

- the signal-to-noise ratio at any range (the standard radar range
  equation),
- the probability of detecting the drone, using the standard model
  for a slowly fluctuating small target (Swerling case 1),
- the maximum range at which detection probability stays above 90%.

This layer is for fast tables and scaling curves. It contains no
simulation — its honesty rests on the G values being measured.

### Layer 2 — `waveform.py`: real samples, counted detections

This layer throws away the closed-form detection formulas and
simulates the receiver. Per trial:

1. Every station transmits the same probe burst — a 5G-style
   multicarrier communication signal (random data on 64 subcarriers).
   Station positions are known, so the transmissions are pre-steered
   at the target; the only phase error left is each station's
   measured synchronization residual, drawn from a real sync run.
2. The fields add up at the drone. The drone re-radiates with a
   random reflection strength each trial (the fluctuating-target
   model).
3. Every station receives the echo buried in thermal noise —
   actual complex noise samples at the physical noise floor.
4. The receiver matched-filters each station's sample stream,
   adds the stations coherently, and compares the power to a
   threshold.
5. The threshold is not taken from a formula. It is calibrated by
   running the same pipeline thousands of times with no target and
   picking the value that gives the requested false-alarm rate. The
   achieved false-alarm rate is then re-measured and reported.

Detection probability is the counted fraction of target-present
trials that cross the threshold. Nothing in the loop is a formula.

### Layer 3 — `rt_echo.py`: ray-traced propagation

This layer replaces the free-space propagation in layer 2 with
Sionna's ray tracer: a 3-D scene with station masts at 15 m, the
drone at 60 m, a physically modeled ground plane, and optionally
concrete buildings. The dominant new physics is the ground bounce —
at these frequencies and heights, the direct ray and the
ground-reflected ray interfere and produce deep ripples in signal
strength along the flight path that beam steering cannot remove.

One implementation note, because it matters: the ray tracer cannot
find rays that bounce off a 0.3 m drone hundreds of meters away (the
chance of a random ray hitting it is about 1 in 100 million — we
verified it finds zero). So the code uses the standard radar-and-
communications coupling instead: trace the exact propagation legs
from each station to the drone's *position* (a point probe, which the
tracer handles analytically), then apply the drone's reflection
analytically at that point, and use reciprocity for the return trip.
This reproduces the textbook bistatic radar equation exactly in free
space (validated to float precision) while keeping the ray-traced
ground and building effects.

### Layer 4 — `realistic.py`: clutter, interference, and a real drone signature

The field-realism layer. Three big additions:

- **Ground clutter.** The ground itself reflects, and that return is
  vastly stronger than the drone. The ground is gridded into cells;
  each cell's reflected power (a standard reflectivity model for
  rural terrain) lands in the correct delay bin for every
  transmit–receive pair, and wind-blown vegetation gives the clutter
  a slow random motion. The clutter is injected into the raw sample
  streams, so it leaks between range bins through the waveform's own
  sidelobes, exactly as in a real receiver. The drone survives
  because it moves: its echo shifts in Doppler frequency while the
  clutter stays near zero, so this layer processes a train of 64
  pulses and forms a full range–Doppler map per receiver.
- **Direct-path self-interference.** Every receiver hears the other
  stations' transmissions directly, about 80 dB louder than the
  drone echo. The code adds these signals at their true delays and
  then removes them the way a real system would — least-squares
  subtraction of the known transmitted templates. The cancellation
  depth is whatever the algebra actually achieves on the samples; it
  is not assumed.
- **A real drone signature.** Instead of a single random reflection
  number, the drone has an aspect-dependent body return (stronger
  broadside, weaker nose-on) plus rotating rotor blades, each of
  which phase-modulates its small return at the rotor rate. The
  blade modulation spreads energy into the sidebands that real
  drone-detection radars key on.

Detection here is a genuine search, not a cued check: a
constant-false-alarm-rate detector sweeps the range–Doppler map with
the zero-Doppler clutter ridge notched out, and its threshold is
again calibrated empirically on target-absent trials that still
contain clutter, interference, and noise.

## The runnable scripts

- **`detection_study.py`** — layer 1. Runs every synchronization
  method (star two-way, mesh, hybrid, and so on), takes each one's
  measured alignment quality, and prints/plots detection range and
  detection-probability-versus-range per method. Fast.
- **`waveform_detection_study.py`** — layers 2–3. The drone flies a
  straight path; at each waypoint the counted Monte-Carlo detection
  runs with each sync method's measured residuals. Produces the
  scene figure with the flight path colored by detection
  probability.
- **`realistic_detection_study.py`** — layer 4. Same flight, now
  clutter-limited with the full range–Doppler search. Reports
  detection probability per waypoint, the achieved false-alarm rate,
  the clutter-to-noise level, and the direct-path suppression
  achieved.
- **`sensing_loop_study.py`** — the tracking loop. Compares three
  synchronization scheduling policies on identical physics: a fixed
  uniform schedule, a schedule set once for the coverage edge, and a
  schedule that re-targets as the drone moves (stations that
  dominate the current target position get tight sync budgets, the
  rest coast). Detection at each waypoint uses only the sync
  residuals from the time interval when the drone was actually
  there — so the claimed benefit of tracking-aware scheduling is
  verified by counted detections, not by the schedule's own
  bookkeeping.

Each script's command-line options are in its docstring; all run
from `sionna/` with `.venv/bin/python <script>.py`.

Figures land in `sionna/figures/deck/`, one prefix per script:
`detrange_*` (link-budget study), `wavescene_*` (waveform flight-path
study), `realistic_*` (clutter-limited study), `sensloop_*` (tracking
loop). `sionna/figures/deck/FIGURES.md` describes them; the 3-D scene
renders are in `sionna/figures/`.

## Honest limits

Everything is simulation; no hardware. Layer 4 still has no discrete
clutter (buildings affect the drone's echo path but not the clutter
map), no receiver nonlinearities, and processes one pulse train at a
time — there is no track filter stitching detections across the
flight (each waypoint is detected independently). The clutter and
drone-signature parameters are standard published values, not
measurements of a specific site or aircraft.
