"""Part II of the coding guide: realistic base-station parameters,
short/long training fields built from the ground up, single-station
monostatic sensing in Sionna (line-of-sight and blocked), OFDM
versus single-tone waveforms, four-station localization against the
Cramer-Rao bound, and -- with the target meshed into the scene as
real geometry -- clutter, multipath and ghost targets.

New code, written for this guide and tested; every measured number
quoted on the slides comes from running this file:

    python3 build_sensing.py
"""

from __future__ import annotations

import itertools
import math
import os
import tempfile
from dataclasses import dataclass

import numpy as np

BOLTZMANN_T0 = 1.380649e-23 * 290.0
SPEED_OF_LIGHT = 299792458.0

rng = np.random.default_rng(0)


# %% SECTION: Realistic base-station parameters
# %% NOTE: One dataclass models a mid-band 5G small cell: 3.5\,GHz
# %% NOTE: carrier (band n78), 20\,MHz of bandwidth sampled at the
# %% NOTE: bandwidth (complex baseband), 10\,W transmit power, an
# %% NOTE: 8\,dBi antenna, a 7\,dB receiver noise figure. The one
# %% NOTE: derived number that drives everything: the thermal noise
# %% NOTE: power $kT_0 \cdot F \cdot B$. At 20\,MHz that is
# %% NOTE: $-94$\,dBm -- every detection question below is a contest
# %% NOTE: between an echo and this floor.
@dataclass(frozen=True)
class BaseStationParams:
    carrier_frequency_hz: float = 3.5e9     # 5G band n78
    bandwidth_hz: float = 20e6              # sample rate = bandwidth
    tx_power_w: float = 10.0                # small-cell class
    antenna_gain_dbi: float = 8.0
    noise_figure_db: float = 7.0
    subcarriers: int = 64                   # OFDM grid over the band
    cyclic_prefix: int = 16

    @property
    def wavelength_m(self) -> float:
        return SPEED_OF_LIGHT / self.carrier_frequency_hz

    @property
    def sample_period_s(self) -> float:
        return 1.0 / self.bandwidth_hz

    @property
    def noise_power_w(self) -> float:
        return (BOLTZMANN_T0
                * 10.0 ** (self.noise_figure_db / 10.0)
                * self.bandwidth_hz)


# %% SECTION: The short and long training fields, from the ground up
# %% NOTE: The same two-field idea as the sync preamble, now built
# %% NOTE: bare-handed at base-station parameters (the 802.11/LTE
# %% NOTE: pattern). \textbf{Short field}: one 16-sample
# %% NOTE: constant-power sequence tiled ten times -- its job is
# %% NOTE: coarse timing and coarse frequency. \textbf{Long field}:
# %% NOTE: a known 64-subcarrier BPSK symbol, sent twice behind one
# %% NOTE: 32-sample cyclic prefix -- its job is fine frequency and
# %% NOTE: the phase. At 20\,MHz the 16-sample repetition makes
# %% NOTE: coarse frequency unambiguous to $\pm 625$\,kHz; a
# %% NOTE: $\pm$10\,ppm oscillator at 3.5\,GHz is only
# %% NOTE: $\pm 35$\,kHz, so acquisition always succeeds.
STF_LEN = 16
STF_REPS = 10
LTF_LEN = 64
LTF_CP = 32


def make_training_fields(params: BaseStationParams, seed: int = 7):
    """Return (waveform, ltf_time) -- the transmitted preamble and
    the known long-field symbol the receiver matches against."""
    field_rng = np.random.default_rng(seed)
    # Short field: constant-modulus random-QPSK 16-sample sequence.
    stf = np.exp(1j * (np.pi / 2.0)
                 * field_rng.integers(0, 4, STF_LEN)
                 + 1j * np.pi / 4.0)
    # Long field: known BPSK on all 64 subcarriers, to time domain.
    bpsk = 2.0 * field_rng.integers(0, 2, LTF_LEN) - 1.0
    ltf_time = np.fft.ifft(bpsk) * math.sqrt(LTF_LEN)
    waveform = np.concatenate([
        np.tile(stf, STF_REPS),
        ltf_time[-LTF_CP:],          # cyclic prefix
        ltf_time,
        ltf_time,                    # the repeat that measures CFO
    ])
    return waveform, ltf_time


# %% SECTION: The receiver: timing, frequency, phase
# %% NOTE: Three stages, each earning one quantity. (1)
# %% NOTE: \textbf{Coarse (Schmidl--Cox)}: correlate the signal with
# %% NOTE: itself 16 samples later; the correlation magnitude peaks
# %% NOTE: where the repeating short field coherently fills the
# %% NOTE: window (coarse timing) and its angle is the frequency
# %% NOTE: offset accumulated over 16 samples. (Deliberately the
# %% NOTE: \emph{unnormalized} peak: the classic normalized ratio
# %% NOTE: spikes on noise-only windows -- found the hard way while
# %% NOTE: testing this.) (2) \textbf{Fine timing}: after coarse
# %% NOTE: derotation, cross-correlate with the known long symbol --
# %% NOTE: a single sharp peak. (3) \textbf{Fine frequency and
# %% NOTE: phase}: the two long symbols are identical and 64 samples
# %% NOTE: apart; their angle refines the frequency, and the matched
# %% NOTE: filter of both, after derotation, reads the carrier
# %% NOTE: phase.
def estimate_preamble(rx, ltf_time, params: BaseStationParams):
    """Return (timing_index, cfo_hz, phase_rad)."""
    ts = params.sample_period_s
    stf_span = STF_LEN * STF_REPS

    # (1) coarse: lag-16 self-correlation over the short field.
    lag = STF_LEN
    width = stf_span - lag
    prod = np.conj(rx[:-lag]) * rx[lag:]
    csum = np.cumsum(np.concatenate([[0.0 + 0j], prod]))
    corr = csum[width:] - csum[:-width]
    # Unnormalized peak: the ratio metric spikes on noise-only
    # windows (tiny numerator over tiny denominator); the raw
    # correlation magnitude only peaks where the repeats coherently
    # fill the window.
    coarse = int(np.argmax(np.abs(corr)))
    cfo = np.angle(corr[coarse]) / (lag * ts) / (2.0 * np.pi)  # Hz

    # (2) fine timing: matched filter of the known long symbol.
    # The long field repeats, so the filter has TWO equal peaks, 64
    # samples apart -- score candidate positions by the pair
    # |mf[i]| + |mf[i+64]|, which only the FIRST symbol maximizes.
    n = np.arange(rx.size)
    derot = rx * np.exp(-2j * np.pi * cfo * n * ts)
    mf = np.abs(np.correlate(derot, ltf_time, mode="valid"))
    # Search a window around where the long field should start,
    # with a margin on BOTH sides -- the coarse estimate can land a
    # sample or two late, which would otherwise push the true peak
    # just outside the window.
    start = max(coarse + stf_span + LTF_CP - 8, 0)
    window = mf[start : min(start + 4 * LTF_LEN, mf.size)]
    pair = window[:-LTF_LEN] + window[LTF_LEN:]
    first_sym = start + int(np.argmax(pair))
    first_sym = min(first_sym, rx.size - 2 * LTF_LEN)

    # (3) fine frequency from the symbol repeat, then the phase.
    a = derot[first_sym : first_sym + LTF_LEN]
    b = derot[first_sym + LTF_LEN : first_sym + 2 * LTF_LEN]
    fine = np.angle(np.sum(np.conj(a) * b)) / (LTF_LEN * ts) \
        / (2.0 * np.pi)
    cfo += fine
    derot = rx * np.exp(-2j * np.pi * cfo * n * ts)
    both = derot[first_sym : first_sym + 2 * LTF_LEN]
    phase = np.angle(np.sum(np.conj(np.tile(ltf_time, 2)) * both))
    return first_sym, cfo, phase


# %% SECTION: Test the receiver at realistic impairments
# %% NOTE: The test: random arrival time, a 20\,kHz frequency offset
# %% NOTE: ($\approx$\,6\,ppm at 3.5\,GHz), a fixed 3-tap multipath
# %% NOTE: channel, 15\,dB signal-to-noise ratio, 500 trials.
# %% NOTE: Measured: timing exact in \textbf{500/500} trials;
# %% NOTE: frequency error \textbf{1.03\,kHz rms} (0.29\,ppm of the
# %% NOTE: carrier); phase repeatable to \textbf{0.107\,rad rms}.
# %% NOTE: The kHz-level frequency error is not a flaw -- it is
# %% NOTE: physics: frequency precision is bought with observation
# %% NOTE: \emph{time}, and this whole preamble lasts 16\,$\mu$s.
# %% NOTE: Part I's synchronization preamble reached $\sim$0.1\,Hz
# %% NOTE: because it observes for 4\,ms, at 250$\times$ less
# %% NOTE: bandwidth. Time buys frequency; bandwidth buys delay --
# %% NOTE: hold that thought for the waveform comparison.
def test_training_fields(params: BaseStationParams, trials=500,
                         snr_db=15.0, cfo_hz=20e3):
    waveform, ltf_time = make_training_fields(params)
    ts = params.sample_period_s
    # One fixed multipath channel across trials, so the phase spread
    # measures the ESTIMATOR, not a re-rolled channel.
    taps = np.array([1.0,
                     0.4 * np.exp(2j * np.pi * 0.3),
                     0.2 * np.exp(2j * np.pi * 0.7)])
    timing_hits, cfo_errs, phases = 0, [], []
    for _ in range(trials):
        delay = int(rng.integers(20, 120))
        tx = np.zeros(delay + waveform.size + 96, complex)
        tx[delay : delay + waveform.size] = waveform
        rx = (np.convolve(tx, taps)[: tx.size]
              * np.exp(2j * np.pi * cfo_hz
                       * np.arange(tx.size) * ts))
        snr = 10.0 ** (snr_db / 10.0)
        sigma = math.sqrt(np.mean(np.abs(waveform) ** 2)
                          / (2.0 * snr))
        rx = rx + sigma * (rng.standard_normal(rx.size)
                           + 1j * rng.standard_normal(rx.size))
        sym, cfo, phase = estimate_preamble(rx, ltf_time, params)
        expected = delay + STF_LEN * STF_REPS + LTF_CP
        timing_hits += int(sym == expected)
        cfo_errs.append(cfo - cfo_hz)
        phases.append(phase)
    # Circular spread: length of the mean phasor.
    resultant = np.abs(np.mean(np.exp(1j * np.array(phases))))
    phase_std = math.sqrt(max(-2.0 * math.log(resultant), 0.0))
    print(f"  timing exact      : {timing_hits}/{trials}")
    print(f"  cfo error rms     : {np.std(cfo_errs):.1f} Hz "
          f"({np.std(cfo_errs)/params.carrier_frequency_hz*1e6:.3f} ppm)")
    print(f"  phase spread rms  : {phase_std:.4f} rad")


# %% SECTION: The ray-tracing model
# %% NOTE: What Sionna's ray tracer actually computes, because Part
# %% NOTE: II leans on it. A \textbf{scene} is a set of triangle
# %% NOTE: meshes, each wearing a \textbf{radio material}: a complex
# %% NOTE: permittivity (its real part $\varepsilon_r$ and
# %% NOTE: conductivity $\sigma_c$ set, at the carrier frequency,
# %% NOTE: how much of an incident wave reflects and with what phase
# %% NOTE: -- the Fresnel coefficients) and a thickness. The
# %% NOTE: \textbf{path solver} then finds every geometric route
# %% NOTE: from a transmitter to a receiver up to a bounce budget
# %% NOTE: (\texttt{max\_depth}): the direct line if unobstructed,
# %% NOTE: and specular (mirror-like) reflections found by the
# %% NOTE: image method -- reflect the endpoint across a face,
# %% NOTE: draw the straight line, keep it if nothing blocks it.
# %% NOTE: Every found path is returned as one complex gain $a_p$
# %% NOTE: (spreading loss $\times$ the Fresnel losses of its
# %% NOTE: bounces, phase = electrical length) and one delay
# %% NOTE: $\tau_p$ -- exactly the $(a_p, \tau_p)$ pairs the echo
# %% NOTE: model consumes. Occlusion is decided by the actual
# %% NOTE: triangles: put a building in the way and the direct path
# %% NOTE: simply does not appear in the list. The two helpers
# %% NOTE: below write scene geometry (a ground plane, a box
# %% NOTE: building) as PLY mesh files the scene loader accepts.
def _ply_box(x, y, w, d, h, z0=0.0):
    x0, x1, y0, y1 = x - w / 2, x + w / 2, y - d / 2, y + d / 2
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z0 + h), (x1, y0, z0 + h), (x1, y1, z0 + h),
         (x0, y1, z0 + h)]
    f = [(0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7),
         (2, 7, 6), (3, 0, 4), (3, 4, 7), (4, 5, 6), (4, 6, 7)]
    if z0 > 0.0:
        # A box sitting on the ground needs no floor, and giving it
        # one would put a face coplanar with the ground plane -- an
        # invitation to z-fighting in the renderer and to degenerate
        # intersections in the solver. A FLOATING box (a drone body)
        # does need its underside, or it renders hollow from below.
        f += [(0, 2, 1), (0, 3, 2)]
    lines = ["ply", "format ascii 1.0", f"element vertex {len(v)}",
             "property float x", "property float y",
             "property float z", f"element face {len(f)}",
             "property list uchar int vertex_indices", "end_header"]
    lines += [f"{a} {b} {c}" for a, b, c in v]
    lines += [f"3 {a} {b} {c}" for a, b, c in f]
    return "\n".join(lines) + "\n"


def _ply_ground(half):
    return ("ply\nformat ascii 1.0\nelement vertex 4\n"
            "property float x\nproperty float y\nproperty float z\n"
            "element face 2\n"
            "property list uchar int vertex_indices\nend_header\n"
            f"-{half} -{half} 0\n{half} -{half} 0\n"
            f"{half} {half} 0\n-{half} {half} 0\n"
            "3 0 1 2\n3 0 2 3\n")


def _add_mesh(rt, scene, name, ply, material):
    """Write a PLY string to a temp file and add it to the scene."""
    handle = tempfile.NamedTemporaryFile("w", suffix=".ply",
                                         delete=False)
    handle.write(ply)
    handle.close()
    try:
        scene.edit(add=rt.SceneObject(fname=handle.name, name=name,
                                      radio_material=material))
    finally:
        os.unlink(handle.name)


def _materials(rt):
    """Ground and concrete, colored so renders are readable."""
    ground = rt.RadioMaterial("ground", thickness=10.0,
                              relative_permittivity=15.0,
                              conductivity=0.035,
                              color=(0.55, 0.65, 0.45))
    concrete = rt.RadioMaterial("concrete", thickness=0.3,
                                relative_permittivity=5.24,
                                conductivity=0.123,
                                color=(0.72, 0.70, 0.66))
    return ground, concrete


def add_visual_markers(rt, scene, stations, targets, scale=12.0):
    """Draw the radios as objects. VISUALIZATION ONLY.

    Point transmitters and receivers render as bare colored dots with
    no size, so a picture of them says nothing about the geometry --
    how high the mast is, how far up the target sits, how big it is.
    These meshes fix that: a pole under each antenna, and a body plus
    a cross-arm at each target, drawn ENLARGED (a real drone is ~0.3 m
    and would be one pixel at these ranges).

    Call this AFTER the path solve. The markers are not physics: the
    target enters the echo model as a point probe plus an analytic
    radar cross-section, and a solid box sitting on the probe would
    block the very paths the picture is meant to show.
    """
    steel = rt.RadioMaterial("mast-material", thickness=0.05,
                             relative_permittivity=1.0,
                             conductivity=1e7,
                             color=(0.20, 0.25, 0.55))
    red = rt.RadioMaterial("target-material", thickness=0.003,
                           relative_permittivity=3.0,
                           conductivity=1e-4,
                           color=(0.85, 0.10, 0.10))
    for index, station in enumerate(np.atleast_2d(stations)):
        x, y, z = (float(v) for v in station)
        _add_mesh(rt, scene, f"mast-{index}",
                  _ply_box(x, y, 0.25 * scale, 0.25 * scale, z), steel)
        _add_mesh(rt, scene, f"head-{index}",
                  _ply_box(x, y, 0.6 * scale, 0.6 * scale,
                           0.25 * scale, z0=z), steel)
    for index, target in enumerate(np.atleast_2d(targets)):
        x, y, z = (float(v) for v in target)
        base = z - scale / 12.0
        _add_mesh(rt, scene, f"target-body-{index}",
                  _ply_box(x, y, scale, scale / 4.0, scale / 6.0,
                           z0=base), red)
        _add_mesh(rt, scene, f"target-cross-{index}",
                  _ply_box(x, y, scale / 4.0, scale, scale / 6.0,
                           z0=base), red)


# %% SECTION: Monostatic sensing: three scenes
# %% NOTE: One base station (mast at 15\,m), one passive target
# %% NOTE: (40\,m altitude, 300\,m down-range), no cooperation: the
# %% NOTE: station transmits and listens for its own echo. The ray
# %% NOTE: tracer supplies the station\,$\to$\,target leg with a
# %% NOTE: point probe receiver at the target position (a ray tracer
# %% NOTE: cannot hit a target-sized object at range, so the
# %% NOTE: standard coupling applies the radar cross-section
# %% NOTE: analytically and reuses the leg for the round trip).
# %% NOTE: Three scenes on the same geometry: \textbf{line-of-sight}
# %% NOTE: (ground plane only); \textbf{blocked} (a
# %% NOTE: $40{\times}40{\times}60$\,m concrete building midway);
# %% NOTE: \textbf{blocked + reflector} (same blocker, plus a long
# %% NOTE: building row 80\,m to the side whose wall offers the
# %% NOTE: solver a specular detour around the blocker -- the urban
# %% NOTE: canyon situation).
STATION = np.array([0.0, 0.0, 15.0])      # the mast
TARGET = np.array([300.0, 0.0, 40.0])     # the passive target


def monostatic_scene(params: BaseStationParams, scenario: str):
    """Build one of the three scenes; returns (rt, scene)."""
    import sionna.rt as rt

    scene = rt.load_scene()
    scene.frequency = params.carrier_frequency_hz
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                    pattern="iso", polarization="V")
    scene.rx_array = scene.tx_array

    ground, concrete = _materials(rt)
    _add_mesh(rt, scene, "ground-plane", _ply_ground(800.0), ground)
    if scenario in ("blocked", "reflector"):
        _add_mesh(rt, scene, "blocker",
                  _ply_box(150.0, 0.0, 40.0, 40.0, 60.0), concrete)
    if scenario == "reflector":
        # A long building row whose street-side wall (y = 80 m) can
        # mirror the signal around the blocker.
        _add_mesh(rt, scene, "reflector",
                  _ply_box(150.0, 90.0, 200.0, 20.0, 60.0), concrete)

    scene.add(rt.Transmitter("bs", position=STATION.tolist()))
    scene.add(rt.Receiver("target-probe", position=TARGET.tolist()))
    return rt, scene


def monostatic_legs(rt, scene):
    """Solve the scene: returns (paths, gains, delays)."""
    paths = rt.PathSolver()(scene, max_depth=3, los=True,
                            specular_reflection=True,
                            diffuse_reflection=False,
                            refraction=False)
    a, tau = paths.cir(normalize_delays=False, out_type="numpy")
    if a.shape[-2] == 0:
        # The solver found no way from station to target at all --
        # complete blockage. Zero paths means zero echo.
        return paths, np.zeros(0, complex), np.zeros(0)
    # a: [rx, rx_ant, tx, tx_ant, path, time]; tau may or may not
    # keep the antenna axes depending on the array setup.
    return (paths, a[0, 0, 0, 0, :, 0],
            tau.reshape(-1, tau.shape[-1])[0])


# %% SECTION: Monostatic sensing: echo, matched filter, range
# %% NOTE: The echo is assembled from the traced paths: going out on
# %% NOTE: path $p$ and back on path $q$ contributes amplitude
# %% NOTE: $a_p a_q \sqrt{4\pi\sigma}/\lambda$ at delay
# %% NOTE: $\tau_p + \tau_q$ (radar cross-section $\sigma = 1$\,m$^2$,
# %% NOTE: a small vehicle). Fractional delays are applied exactly,
# %% NOTE: as a phase ramp in the frequency domain. Thermal noise at
# %% NOTE: the $-94$\,dBm floor, matched filter against the
# %% NOTE: transmitted burst, parabolic interpolation around the
# %% NOTE: peak, and range $= c\,\hat\tau/2$. Note what detection
# %% NOTE: will hinge on: the echo arrives \emph{below} the noise
# %% NOTE: floor per sample; only the matched filter's processing
# %% NOTE: gain ($10\log_{10} 1600 = 32$\,dB over the burst) lifts
# %% NOTE: it out.
def fractional_delay(signal, delay_samples, total_len):
    padded = np.zeros(total_len, complex)
    padded[: signal.size] = signal
    freq = np.fft.fftfreq(total_len)
    return np.fft.ifft(np.fft.fft(padded)
                       * np.exp(-2j * np.pi * freq * delay_samples))


def monostatic_range(params: BaseStationParams, burst,
                     gains, delays, trials=100):
    """Counted range estimates; returns (detect_rate, errors_m,
    echo power in watts)."""
    ts = params.sample_period_s
    amp_gain = 10.0 ** (params.antenna_gain_dbi / 20.0)
    rcs_factor = math.sqrt(4.0 * math.pi * 1.0) \
        / params.wavelength_m
    total = burst.size + 4096
    echo = np.zeros(total, complex)
    for p in range(gains.size):
        for q in range(gains.size):
            amplitude = (math.sqrt(params.tx_power_w) * amp_gain**2
                         * gains[p] * gains[q] * rcs_factor)
            delay_s = (delays[p] + delays[q]) / ts
            echo += amplitude * fractional_delay(
                burst, delay_s, total)
    echo_power = float(np.mean(np.abs(echo) ** 2)
                       * total / burst.size)
    sigma = math.sqrt(params.noise_power_w / 2.0)
    true_range = math.sqrt(300.0**2 + 25.0**2)

    detections, errors = 0, []
    for _ in range(trials):
        rx = echo + sigma * (rng.standard_normal(total)
                             + 1j * rng.standard_normal(total))
        mf = np.abs(np.correlate(rx, burst, mode="valid")) ** 2
        peak = int(np.argmax(mf))
        floor = np.median(mf)
        if mf[peak] < 30.0 * floor:      # ~15 dB above the floor
            continue
        detections += 1
        if 0 < peak < mf.size - 1:       # parabolic refinement
            num = mf[peak - 1] - mf[peak + 1]
            den = mf[peak - 1] - 2 * mf[peak] + mf[peak + 1]
            peak = peak + 0.5 * num / den
        range_m = peak * ts * SPEED_OF_LIGHT / 2.0
        errors.append(range_m - true_range)
    return detections / trials, np.array(errors), echo_power


# %% SECTION: Render the scenes, with the traced rays
# %% NOTE: The ray tracer can draw exactly what it solved:
# %% NOTE: \texttt{render\_to\_file} paints the scene geometry from
# %% NOTE: a camera and overlays the found propagation paths. The
# %% NOTE: three renders on the next slides are the three scenarios,
# %% NOTE: with every path the solver found drawn in. Line-of-sight:
# %% NOTE: the direct ray and the ground bounce. Blocked: the
# %% NOTE: building swallows everything -- no rays to draw. Blocked
# %% NOTE: + reflector: the detour around the blocker via the
# %% NOTE: building row's wall, plus its ground-bounce variants.
# %% NOTE: \textbf{Order matters in this function.} The solve runs
# %% NOTE: first, on the bare radio scene; only then are the mast
# %% NOTE: and target meshes added and the picture drawn. Adding
# %% NOTE: them first would put a solid box on top of the probe
# %% NOTE: receiver and delete the very paths the figure exists to
# %% NOTE: show. \texttt{show\_devices=False} then hides the
# %% NOTE: default red/green endpoint dots, because the meshes now
# %% NOTE: say the same thing with actual geometry: a 15\,m mast,
# %% NOTE: and a target 300\,m down-range at 40\,m altitude. The
# %% NOTE: target is drawn far larger than life -- at 0.4\,m per
# %% NOTE: pixel a real vehicle is a smudge and a real drone is
# %% NOTE: invisible -- so read it as a position marker, not a size.
# %% IMAGE: figures/scene_los.png | Line-of-sight: the direct ray and the ground bounce reach the target.
# %% IMAGE: figures/scene_blocked.png | Blocked: the solver finds no path at all -- there are no rays to draw.
# %% IMAGE: figures/scene_reflector.png | Blocked + reflector: the wall of the building row mirrors the signal around the blocker (three paths, all longer than the straight line).
def render_scenes(params: BaseStationParams, out_dir="figures"):
    os.makedirs(out_dir, exist_ok=True)
    for scenario in ("los", "blocked", "reflector"):
        rt, scene = monostatic_scene(params, scenario)
        paths, gains, _ = monostatic_legs(rt, scene)   # SOLVE FIRST
        add_visual_markers(rt, scene, STATION, TARGET)
        camera = rt.Camera(position=[150.0, -340.0, 300.0],
                           look_at=[150.0, 20.0, 20.0])
        scene.render_to_file(
            camera=camera,
            filename=os.path.join(out_dir,
                                  f"scene_{scenario}.png"),
            # The renderer cannot overlay an empty path list.
            paths=paths if gains.size else None,
            show_devices=False,
            resolution=(900, 500))


# %% SECTION: The range profiles, side by side
# %% NOTE: One noisy trial per scenario, matched filter output
# %% NOTE: against range. Line-of-sight: one clean peak at the true
# %% NOTE: 301\,m. Blocked: noise, nothing to find. Blocked +
# %% NOTE: reflector: a real peak -- but at the detour's length, not
# %% NOTE: the target's range. The echo survives the blockage and
# %% NOTE: the detection is genuine; the \emph{range} is a lie
# %% NOTE: (about $+40$\,m here). A single station cannot tell a
# %% NOTE: detour from a distance; resolving that ambiguity takes
# %% NOTE: more geometry -- stations that look from somewhere else.
# %% IMAGE: figures/range_profiles.png | Matched-filter range profiles, one noisy trial per scenario; the vertical line is the true range.
def range_profile_figure(params: BaseStationParams,
                         out_dir="figures"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    ts = params.sample_period_s
    burst = make_ofdm_burst(params)
    sigma = math.sqrt(params.noise_power_w / 2.0)
    true_range = math.sqrt(300.0**2 + 25.0**2)
    fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    for axis, scenario in zip(axes,
                              ("los", "blocked", "reflector")):
        rt, scene = monostatic_scene(params, scenario)
        _, gains, delays = monostatic_legs(rt, scene)
        total = burst.size + 4096
        echo = np.zeros(total, complex)
        amp_gain = 10.0 ** (params.antenna_gain_dbi / 20.0)
        rcs_factor = math.sqrt(4.0 * math.pi) / params.wavelength_m
        for p in range(gains.size):
            for q in range(gains.size):
                amplitude = (math.sqrt(params.tx_power_w)
                             * amp_gain**2 * gains[p] * gains[q]
                             * rcs_factor)
                echo += amplitude * fractional_delay(
                    burst, (delays[p] + delays[q]) / ts, total)
        rx = echo + sigma * (rng.standard_normal(total)
                             + 1j * rng.standard_normal(total))
        mf = np.abs(np.correlate(rx, burst, mode="valid")) ** 2
        ranges = (np.arange(mf.size) * ts * SPEED_OF_LIGHT / 2.0)
        keep = ranges <= 600.0
        axis.plot(ranges[keep],
                  10.0 * np.log10(mf[keep] / np.max(mf)))
        axis.axvline(true_range, color="C3", linewidth=0.8)
        axis.set_ylabel(f"{scenario}\n(dB)")
    axes[-1].set_xlabel("range (m)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "range_profiles.png"),
                dpi=150)


# %% SECTION: OFDM versus a single tone
# %% NOTE: Same duration, same power, two waveforms. The
# %% NOTE: \textbf{OFDM burst} spreads energy over the whole 20\,MHz;
# %% NOTE: the \textbf{single tone} concentrates it at one
# %% NOTE: frequency. For \emph{frequency and phase} the tone is
# %% NOTE: fine -- fit a phase ramp. For \emph{range} the tone is
# %% NOTE: nearly blind: range lives in \emph{delay}, delay
# %% NOTE: precision comes from \emph{bandwidth} ($c/2B = 7.5$\,m at
# %% NOTE: 20\,MHz), and the carrier itself carries none. What
# %% NOTE: little timing a real tone burst has lives in its slow
# %% NOTE: power envelope -- which is why the tone here gets a
# %% NOTE: smooth ramp: a rectangular edge would smuggle wideband
# %% NOTE: timing into a ``narrowband'' waveform. This is the
# %% NOTE: cleanest statement of why sensing wants wideband
# %% NOTE: waveforms while synchronization alone can live on a
# %% NOTE: tone.
def make_ofdm_burst(params: BaseStationParams, symbols=20, seed=3):
    burst_rng = np.random.default_rng(seed)
    angles = (burst_rng.integers(0, 4,
                                 (symbols, params.subcarriers))
              * (np.pi / 2.0) + np.pi / 4.0)
    time_symbols = np.fft.ifft(np.exp(1j * angles), axis=1) \
        * math.sqrt(params.subcarriers)
    with_cp = np.concatenate(
        [time_symbols[:, -params.cyclic_prefix:], time_symbols],
        axis=1).reshape(-1)
    return with_cp / math.sqrt(np.mean(np.abs(with_cp) ** 2))


def compare_waveforms(params: BaseStationParams, trials=300,
                      snr_db=10.0, cfo_hz=5e3):
    ts = params.sample_period_s
    ofdm = make_ofdm_burst(params)
    # A single tone at one subcarrier frequency. A real transmitter
    # ramps power smoothly, so the tone gets a smooth (Hann)
    # envelope -- a rectangular edge would smuggle wideband timing
    # information into a "narrowband" waveform.
    tone = (np.exp(2j * np.pi * 3.125e6
                   * np.arange(ofdm.size) * ts)
            * np.hanning(ofdm.size))
    tone = tone / math.sqrt(np.mean(np.abs(tone) ** 2))
    snr = 10.0 ** (snr_db / 10.0)
    sigma = math.sqrt(1.0 / (2.0 * snr))
    results = {}
    for name, burst in (("ofdm", ofdm), ("tone", tone)):
        cfo_errs, delay_errs = [], []
        for _ in range(trials):
            delay = int(rng.integers(0, 200))
            rx = fractional_delay(burst, delay, burst.size + 512)
            rx = rx * np.exp(2j * np.pi * cfo_hz
                             * np.arange(rx.size) * ts)
            rx = rx + sigma * (rng.standard_normal(rx.size)
                               + 1j * rng.standard_normal(rx.size))
            # frequency: split-correlation, identical for both
            half = burst.size // 2
            seg = rx[delay : delay + burst.size]
            r = np.sum(np.conj(burst[:half] * seg[half:half+half])
                       * (burst[half:half+half] * np.conj(seg[:half])))
            est = -np.angle(r) / (half * ts) / (2.0 * np.pi)
            cfo_errs.append(est - cfo_hz)
            # delay: matched filter, no help given
            mf = np.abs(np.correlate(rx, burst, mode="valid"))
            delay_errs.append(int(np.argmax(mf)) - delay)
        results[name] = (np.std(cfo_errs),
                         np.sqrt(np.mean(np.square(delay_errs))))
    for name, (cfo_rms, delay_rms) in results.items():
        print(f"  {name:5s} cfo rms {cfo_rms:8.1f} Hz | "
              f"delay rms {delay_rms:8.1f} samples "
              f"({delay_rms * ts * SPEED_OF_LIGHT / 2.0:.0f} m)")


# %% SECTION: Localization: from ranges to a position
# %% NOTE: The ghost-range result begs the next step: more stations.
# %% NOTE: Four base stations at the corners of the area each run
# %% NOTE: the monostatic measurement on the same \textbf{drone}
# %% NOTE: (radar cross-section 0.03\,m$^2$, hovering at 60\,m).
# %% NOTE: Each station's round-trip delay gives one range -- a
# %% NOTE: sphere the drone must sit on -- and four spheres pin the
# %% NOTE: position. One ray-tracer solve supplies all four legs
# %% NOTE: (ground bounce included); the position solver is
# %% NOTE: Gauss--Newton: linearize each range around the current
# %% NOTE: guess (the gradient of $\|p - s_i\|$ is the unit vector
# %% NOTE: from station to drone), solve the little least-squares
# %% NOTE: problem, step, repeat.
DRONE = np.array([170.0, 90.0, 60.0])
DRONE_RCS_M2 = 0.03
STATIONS = np.array([[0.0, 0.0, 15.0], [300.0, 0.0, 15.0],
                     [0.0, 200.0, 15.0], [300.0, 200.0, 15.0]])


def localization_legs(params: BaseStationParams):
    """One RT solve, four stations -> drone: [(gains, delays)]."""
    import sionna.rt as rt

    scene = rt.load_scene()
    scene.frequency = params.carrier_frequency_hz
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                    pattern="iso", polarization="V")
    scene.rx_array = scene.tx_array
    handle = tempfile.NamedTemporaryFile("w", suffix=".ply",
                                         delete=False)
    handle.write(_ply_ground(800.0))
    handle.close()
    try:
        scene.edit(add=rt.SceneObject(
            fname=handle.name, name="ground-plane",
            radio_material=rt.RadioMaterial(
                "ground", thickness=10.0,
                relative_permittivity=15.0, conductivity=0.035)))
    finally:
        os.unlink(handle.name)
    for index, station in enumerate(STATIONS):
        scene.add(rt.Transmitter(f"bs-{index}",
                                 position=station.tolist()))
    scene.add(rt.Receiver("drone", position=DRONE.tolist()))
    paths = rt.PathSolver()(scene, max_depth=3, los=True,
                            specular_reflection=True,
                            diffuse_reflection=False,
                            refraction=False)
    a, tau = paths.cir(normalize_delays=False, out_type="numpy")
    legs = []
    for index in range(len(STATIONS)):
        gains = a[0, 0, index, 0, :, 0]
        delays = tau[0, index, :]
        keep = delays >= 0.0          # padded slots carry tau = -1
        legs.append((gains[keep], delays[keep]))
    return legs


def measure_range(params: BaseStationParams, burst,
                  gains, delays, upsample=16):
    """One noisy monostatic range measurement from one station.

    The peak search runs on a 16x frequency-domain upsampled
    cross-correlation: the coarse three-point parabola on the raw
    sample grid leaves a bias worth several times the noise floor,
    and an estimator meant to be compared against the Cramer-Rao
    bound has to earn it.
    """
    ts = params.sample_period_s
    amp_gain = 10.0 ** (params.antenna_gain_dbi / 20.0)
    rcs_factor = (math.sqrt(4.0 * math.pi * DRONE_RCS_M2)
                  / params.wavelength_m)
    total = burst.size + 2048
    echo = np.zeros(total, complex)
    for p in range(gains.size):
        for q in range(gains.size):
            amplitude = (math.sqrt(params.tx_power_w) * amp_gain**2
                         * gains[p] * gains[q] * rcs_factor)
            echo += amplitude * fractional_delay(
                burst, (delays[p] + delays[q]) / ts, total)
    sigma = math.sqrt(params.noise_power_w / 2.0)
    rx = echo + sigma * (rng.standard_normal(total)
                         + 1j * rng.standard_normal(total))
    spectrum = np.fft.fft(rx) * np.conj(np.fft.fft(burst, total))
    padded = np.zeros(total * upsample, complex)
    padded[: total // 2] = spectrum[: total // 2]
    padded[-total // 2 :] = spectrum[-total // 2 :]
    cc = np.abs(np.fft.ifft(padded)) ** 2
    search = cc[: (total - burst.size) * upsample]
    peak = int(np.argmax(search))
    num = cc[peak - 1] - cc[peak + 1]
    den = cc[peak - 1] - 2 * cc[peak] + cc[peak + 1]
    peak = peak + 0.5 * num / den
    return peak / upsample * ts * SPEED_OF_LIGHT / 2.0, echo


def solve_position(ranges, stations, initial):
    """Gauss-Newton: least-squares position from station ranges.

    The initial guess must sit OFF the stations' common plane:
    every look direction's vertical component is zero there, so the
    plane is a stationary point the iteration can never leave --
    found the hard way (the solver returned z = 15 m forever).
    """
    position = initial.copy()
    for _ in range(20):
        vectors = position - stations
        distances = np.linalg.norm(vectors, axis=1)
        jacobian = vectors / distances[:, None]
        residual = ranges - distances
        step, *_ = np.linalg.lstsq(jacobian, residual, rcond=None)
        position = position + step
        if np.linalg.norm(step) < 1e-6:
            break
    return position


# %% SECTION: The Cramer-Rao bound
# %% NOTE: How good could ANY unbiased estimator possibly be? That
# %% NOTE: is the Cramer--Rao bound, and both layers of it are
# %% NOTE: computable from things already in hand. \textbf{Per
# %% NOTE: station (delay):} $\mathrm{var}(\hat\tau) \geq 1 / (8
# %% NOTE: \pi^2 \beta^2 \cdot E/N_0)$, where $\beta$ is the
# %% NOTE: waveform's rms bandwidth and $E/N_0$ the received echo
# %% NOTE: energy over the noise density -- bandwidth and energy are
# %% NOTE: the only two currencies delay accuracy trades in (the
# %% NOTE: single tone fails exactly here: $\beta \approx 0$).
# %% NOTE: Monostatic ranging halves the delay error into range:
# %% NOTE: $r = c\tau/2$. \textbf{Position:} each station
# %% NOTE: contributes information only ALONG its look direction
# %% NOTE: $u_i$; the Fisher information matrix is $J = \sum_i u_i
# %% NOTE: u_i^T / \sigma_{r_i}^2$ and $J^{-1}$ lower-bounds the
# %% NOTE: position covariance. Geometry enters through the $u_i$:
# %% NOTE: directions the stations do not span are directions the
# %% NOTE: network cannot measure -- watch the vertical axis.
def rms_bandwidth_hz(burst, sample_rate):
    spectrum = np.abs(np.fft.fft(burst)) ** 2
    freq = np.fft.fftfreq(burst.size, 1.0 / sample_rate)
    center = np.sum(freq * spectrum) / np.sum(spectrum)
    return math.sqrt(np.sum((freq - center) ** 2 * spectrum)
                     / np.sum(spectrum))


def range_crb_m(params: BaseStationParams, burst, echo):
    """Per-station range standard-deviation bound (meters)."""
    beta = rms_bandwidth_hz(burst, params.bandwidth_hz)
    # E/N0: echo energy over noise density, in sample units.
    energy_over_n0 = np.sum(np.abs(echo) ** 2) / params.noise_power_w
    var_tau = 1.0 / (8.0 * math.pi**2 * beta**2 * energy_over_n0)
    return (SPEED_OF_LIGHT / 2.0) * math.sqrt(var_tau)


def position_bound(stations, drone, range_sigmas):
    """Cramer-Rao covariance for the position (3x3)."""
    vectors = drone - stations
    units = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    fisher = np.zeros((3, 3))
    for u, sigma in zip(units, range_sigmas):
        fisher += np.outer(u, u) / sigma**2
    return np.linalg.inv(fisher)


# %% SECTION: Localize the drone, and check against the bound
# %% NOTE: 200 full trials: four noisy matched-filter ranges, one
# %% NOTE: Gauss--Newton solve each; then measured error next to
# %% NOTE: the bound, axis by axis. Measured: per-station range
# %% NOTE: bound 21\,cm; position \textbf{spread} x/y/z =
# %% NOTE: 0.16/0.25/0.49\,m against bounds 0.14/0.18/0.43\,m --
# %% NOTE: the estimator runs within 20--40\% of the theoretical
# %% NOTE: limit on every axis. Two lessons in those numbers.
# %% NOTE: \textbf{Geometry}: the vertical bound is 3$\times$ the
# %% NOTE: horizontal ones before a single trial runs -- all four
# %% NOTE: stations sit below the drone in nearly a plane, so their
# %% NOTE: look directions barely span the vertical. \textbf{Bias
# %% NOTE: is not spread}: on top of the spread sits a systematic
# %% NOTE: error (x $-0.34$, y $+0.24$, z $+3.4$\,m) from the
# %% NOTE: ground-bounce path fusing with the direct one inside the
# %% NOTE: resolution cell. The bound only governs an unbiased
# %% NOTE: estimator's spread; multipath is a modeling error, it
# %% NOTE: does not average away, and the weak axis amplifies it --
# %% NOTE: the same lesson as the ghost target, in miniature.
# %% IMAGE: figures/localization_scatter.png | 200 position estimates (horizontal plane), the true position (+), and the bound's 3-sigma ellipse. The cloud's SIZE matches the ellipse -- the spread is near the bound -- but the cloud sits displaced from the truth: that offset is the multipath bias, and no amount of averaging shrinks it.
def localization_study(params: BaseStationParams, trials=200,
                       out_dir="figures"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    burst = make_ofdm_burst(params)
    legs = localization_legs(params)
    initial = np.array([150.0, 100.0, 120.0])  # off the plane!

    # Per-station bounds from one clean echo each.
    sigmas = []
    for gains, delays in legs:
        _, echo = measure_range(params, burst, gains, delays)
        sigmas.append(range_crb_m(params, burst, echo))
    bound = position_bound(STATIONS, DRONE, sigmas)

    estimates = []
    for _ in range(trials):
        ranges = np.array([
            measure_range(params, burst, gains, delays)[0]
            for gains, delays in legs])
        estimates.append(solve_position(ranges, STATIONS, initial))
    estimates = np.array(estimates)
    errors = estimates - DRONE

    print("localization of the drone (200 trials, 4 stations):")
    print(f"  per-station range bound : "
          f"{np.mean(sigmas)*100:.1f} cm")
    # The bound limits the SPREAD of an unbiased estimator; the
    # multipath bias is a separate, systematic effect -- report both.
    for axis, name in enumerate("xyz"):
        print(f"  {name}: bias {np.mean(errors[:, axis]):+7.3f} m"
              f"  spread {np.std(errors[:, axis]):6.3f} m"
              f"  | bound {math.sqrt(bound[axis, axis]):6.3f} m")

    # Scatter in the horizontal plane with the 3-sigma ellipse.
    os.makedirs(out_dir, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(estimates[:, 0], estimates[:, 1], s=6)
    values, directions = np.linalg.eigh(bound[:2, :2])
    angle = np.linspace(0.0, 2.0 * np.pi, 200)
    circle = np.stack([np.cos(angle), np.sin(angle)])
    ellipse = (directions
               @ np.diag(3.0 * np.sqrt(values)) @ circle)
    axis.plot(DRONE[0] + ellipse[0], DRONE[1] + ellipse[1], "C3")
    axis.plot(DRONE[0], DRONE[1], "k+")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "localization_scatter.png"),
                dpi=150)
    return sigmas, bound, errors


# %% SECTION: Ghosts: put the target IN the scene
# %% NOTE: Everything so far modeled the target as a point probe
# %% NOTE: plus an analytic radar cross-section. That is the right
# %% NOTE: tool for a link budget, and it is wrong for this
# %% NOTE: question, because it can only ever return the target.
# %% NOTE: A sensing receiver does not get a clean target return --
# %% NOTE: it gets everything in the scene that sends energy back,
# %% NOTE: and the hard part of the job is that most of those
# %% NOTE: returns are not the object. So here the target becomes
# %% NOTE: \textbf{geometry}: a $4{\times}2{\times}1.5$\,m metal
# %% NOTE: box at the target position, meshed into the scene like
# %% NOTE: any building. Each station gets a receiver co-located
# %% NOTE: with its transmitter, and the solver is asked for the
# %% NOTE: round trips directly -- station\,$\to$\,anything\,$\to$
# %% NOTE: \,same station. No radar cross-section is applied and no
# %% NOTE: legs are paired by hand: whatever comes back, comes back.
TARGET_BOX = (4.0, 2.0, 1.5)          # w, d, h -- vehicle class
GHOST_BLOCKER = (250.0, 160.0, 36.0, 36.0, 60.0)   # x, y, w, d, h
GHOST_WALL = (150.0, 250.0, 400.0, 10.0, 60.0)
LEAKAGE_GATE_M = 30.0   # ignore returns closer than this


def multipath_scene(params: BaseStationParams, scenario: str,
                    with_target=True):
    """Four stations + a MESHED target in a built-up scene.

    with_target=False builds the identical scene minus the target --
    that is the clutter reference the study subtracts.
    """
    import sionna.rt as rt

    scene = rt.load_scene()
    scene.frequency = params.carrier_frequency_hz
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                    pattern="iso", polarization="V")
    scene.rx_array = scene.tx_array
    ground, concrete = _materials(rt)
    _add_mesh(rt, scene, "ground-plane", _ply_ground(800.0), ground)
    if scenario in ("multipath", "nlos"):
        _add_mesh(rt, scene, "building-row",
                  _ply_box(*GHOST_WALL), concrete)
    if scenario == "nlos":
        _add_mesh(rt, scene, "blocker",
                  _ply_box(*GHOST_BLOCKER), concrete)

    # The target itself, as an object. Metal: a conductor reflects
    # essentially everything, which is the point -- this is the one
    # thing in the scene we WANT to come back.
    if with_target:
        metal = rt.RadioMaterial("target-metal", thickness=0.002,
                                 relative_permittivity=1.0,
                                 conductivity=1e7,
                                 color=(0.85, 0.10, 0.10))
        width, depth, height = TARGET_BOX
        _add_mesh(rt, scene, "target",
                  _ply_box(DRONE[0], DRONE[1], width, depth, height,
                           z0=DRONE[2] - height / 2.0), metal)

    # Monostatic means the receiver IS the transmitter, so every
    # station gets both. They are separated by half a meter rather
    # than placed on the same point: a zero-length transmitter-to-
    # receiver path is a degenerate case for a path solver, and at
    # 200 m range half a meter changes no delay that matters (it is
    # 1/15th of a resolution cell).
    for index, station in enumerate(STATIONS):
        scene.add(rt.Transmitter(f"bs-{index}",
                                 position=station.tolist()))
        receiver = station + np.array([0.0, 0.0, 0.5])
        scene.add(rt.Receiver(f"rx-{index}",
                              position=receiver.tolist()))
    return rt, scene


# %% SECTION: What actually comes back
# %% NOTE: One solve gives every station its own round trips. Two
# %% NOTE: pieces of bookkeeping. \textbf{Leakage}: a receiver
# %% NOTE: sitting on its own transmitter sees a zero-length path
# %% NOTE: -- in hardware this is the coupling that saturates the
# %% NOTE: front end, and in both cases it is gated out by range.
# %% NOTE: \textbf{Everything else is a return, and the solver does
# %% NOTE: not label them.} Nothing in the output says ``this one
# %% NOTE: is the target.'' The station gets the target echo, the
# %% NOTE: ground under its own feet, the wall of the building row
# %% NOTE: 245\,m away, the target-then-wall double bounce -- all
# %% NOTE: as one list of (gain, delay) pairs, exactly as a real
# %% NOTE: receiver gets them summed into one waveform.
def monostatic_returns(rt, scene, min_range_m=LEAKAGE_GATE_M):
    """One solve -> [(gains, delays)] of round trips per station."""
    paths = rt.PathSolver()(scene, max_depth=3, los=True,
                            specular_reflection=True,
                            diffuse_reflection=False,
                            refraction=False)
    a, tau = paths.cir(normalize_delays=False, out_type="numpy")
    # a: [rx, rx_ant, tx, tx_ant, path, time]; tau: [rx, tx, path].
    # Monostatic = the diagonal: station i transmitting, station i
    # listening. The off-diagonal entries are the bistatic pairs --
    # real, useful, and a bigger study than this one.
    minimum_delay = 2.0 * min_range_m / SPEED_OF_LIGHT
    returns = []
    for index in range(len(STATIONS)):
        gains = a[index, 0, index, 0, :, 0]
        delays = tau[index, index, :]
        keep = delays >= minimum_delay      # drops padding AND leakage
        returns.append((gains[keep], delays[keep]))
    return paths, returns


def roundtrip_echo(params: BaseStationParams, burst, gains, delays,
                   pad=4096):
    """Sum the traced round trips into one received waveform.

    Contrast this with the probe model earlier: there, one-way legs
    had to be paired (p out, q back) and a radar cross-section
    supplied by hand. Here each entry already IS a round trip, so
    the echo is a plain sum -- and it contains the clutter too.
    """
    ts = params.sample_period_s
    amp_gain = 10.0 ** (params.antenna_gain_dbi / 20.0)
    total = burst.size + pad
    echo = np.zeros(total, complex)
    for p in range(gains.size):
        amplitude = (math.sqrt(params.tx_power_w) * amp_gain**2
                     * gains[p])
        echo += amplitude * fractional_delay(
            burst, delays[p] / ts, total)
    return echo


# %% SECTION: Every peak is a candidate range
# %% NOTE: The matched filter now returns a \emph{comb}, and the
# %% NOTE: detector has no way to know which peak is the target --
# %% NOTE: so take them all. The obvious way to peel them off is to
# %% NOTE: find the largest, blank a resolution cell around it, and
# %% NOTE: repeat; do that and the detector immediately reports
# %% NOTE: three targets where there is one, because a burst's
# %% NOTE: autocorrelation has \textbf{sidelobes} and the first
# %% NOTE: pair sits about 2.5 cells out -- outside the blanked
# %% NOTE: window and far above the noise floor. Sidelobes are not
# %% NOTE: noise; they are a known, deterministic feature of the
# %% NOTE: waveform, so they can be removed exactly. That is
# %% NOTE: \textbf{CLEAN}: each time a peak is accepted, subtract a
# %% NOTE: correctly scaled and shifted copy of the burst's own
# %% NOTE: autocorrelation -- the full shape a single return makes,
# %% NOTE: sidelobes included -- and look again at what is left.
# %% NOTE: The correlation is also upsampled 16$\times$, for the
# %% NOTE: reason \texttt{measure\_range} gives: the coarse
# %% NOTE: parabola biases a peak by meters, and the residual test
# %% NOTE: below cannot tell a wrong association from a biased
# %% NOTE: right one if the right one is already off by half a cell.
def _upsampled(spectrum, total, upsample):
    padded = np.zeros(total * upsample, complex)
    padded[: total // 2] = spectrum[: total // 2]
    padded[-total // 2:] = spectrum[-total // 2:]
    return np.fft.ifft(padded)


def range_peaks(params: BaseStationParams, burst, echo,
                max_peaks=4, upsample=16, snr_gate=30.0):
    """All candidate ranges in one echo; (ranges_m, profile, grid)."""
    ts = params.sample_period_s
    if echo.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    total = echo.size
    sigma = math.sqrt(params.noise_power_w / 2.0)
    rx = echo + sigma * (rng.standard_normal(total)
                         + 1j * rng.standard_normal(total))
    burst_spectrum = np.fft.fft(burst, total)
    correlation = _upsampled(np.fft.fft(rx) * np.conj(burst_spectrum),
                             total, upsample)
    # The response a SINGLE return produces: the burst correlated
    # with itself, at the same upsampling. This is what CLEAN
    # subtracts, and why sidelobes come out with the peak.
    template = _upsampled(np.abs(burst_spectrum) ** 2, total,
                          upsample)
    search = (total - burst.size) * upsample
    profile = np.abs(correlation[:search]) ** 2
    grid = (np.arange(search) / upsample * ts
            * SPEED_OF_LIGHT / 2.0)

    floor = np.median(profile)
    ranges = []
    while len(ranges) < max_peaks:
        power = np.abs(correlation[:search]) ** 2
        index = int(np.argmax(power))
        if power[index] < snr_gate * floor:
            break
        refined = float(index)
        if 0 < index < search - 1:
            num = power[index - 1] - power[index + 1]
            den = (power[index - 1] - 2 * power[index]
                   + power[index + 1])
            if den != 0.0:
                refined = index + 0.5 * num / den
        ranges.append(refined / upsample * ts * SPEED_OF_LIGHT / 2.0)
        amplitude = correlation[index] / template[0]
        correlation = correlation - amplitude * np.roll(template,
                                                        index)
    return np.array(sorted(ranges)), profile, grid


# %% SECTION: Association: where the ghosts come from
# %% NOTE: Now the network has to decide which peak at station $i$
# %% NOTE: belongs to the same object as which peak at station $j$
# %% NOTE: -- the \textbf{association} problem, and it is
# %% NOTE: combinatorial: with four candidates at each of four
# %% NOTE: stations there are $4^4 = 256$ ways to pick one range
# %% NOTE: each, and 255 of them are wrong. Each combination is a
# %% NOTE: perfectly well-posed least-squares problem returning a
# %% NOTE: perfectly plausible position. Those are the
# %% NOTE: \textbf{ghost targets}. What saves the network is that
# %% NOTE: four ranges determine three coordinates, leaving one
# %% NOTE: degree of freedom: a wrong combination generally cannot
# %% NOTE: be fitted by ANY position, so its residual is large and
# %% NOTE: it can be gated away. Note the corollary, which is the
# %% NOTE: real argument for the fourth station: with three
# %% NOTE: stations the fit is exact, the residual is identically
# %% NOTE: zero, and \emph{no} ghost can ever be rejected.
def associate(candidates, stations, initial):
    """Every one-peak-per-station combination -> a fix.

    Returns [(position, residual_rms_m)], one per combination.
    """
    fixes = []
    for combination in itertools.product(*candidates):
        ranges = np.array(combination)
        position = solve_position(ranges, stations, initial)
        if not np.all(np.isfinite(position)):
            continue
        residual = ranges - np.linalg.norm(position - stations,
                                           axis=1)
        fixes.append((position,
                      float(np.sqrt(np.mean(residual**2)))))
    return fixes


# %% SECTION: Clutter cancellation, and what it cannot remove
# %% NOTE: Run the scene as built and the target is invisible --
# %% NOTE: not marginal, invisible. A building wall reflects
# %% NOTE: \emph{specularly}, so its return behaves like a mirror
# %% NOTE: image of the transmitter and falls off as one-way
# %% NOTE: spreading over the folded path; the target's return is a
# %% NOTE: radar return and falls off as $r^{-4}$. The wall wins by
# %% NOTE: tens of dB, and every peak the detector reports is the
# %% NOTE: building. This is the real first problem in sensing, and
# %% NOTE: it has a standard answer: the clutter does not move.
# %% NOTE: Solve the identical scene \emph{without} the target,
# %% NOTE: subtract that echo, and the static returns cancel. (In
# %% NOTE: hardware the same cancellation is done in Doppler --
# %% NOTE: same idea, and the reference is measured rather than
# %% NOTE: simulated.) Now the important part: what survives the
# %% NOTE: subtraction is everything that \emph{touched} the
# %% NOTE: target, and that includes station\,$\to$\,target\,$\to$
# %% NOTE: \,wall\,$\to$\,station. Clutter cancellation removes the
# %% NOTE: scene; it cannot remove the target's own multipath. The
# %% NOTE: ghosts come from what is left.


# %% SECTION: The ghost study, and the picture of it
# %% NOTE: One run per scenario: solve the scene twice (with and
# %% NOTE: without the target), cancel, pull every peak, enumerate
# %% NOTE: the associations, gate on the residual, report what
# %% NOTE: survives. Read the map as three claims. \texttt{open}:
# %% NOTE: nothing but ground and target, association is nearly
# %% NOTE: unique. \texttt{multipath}: the target is also seen
# %% NOTE: around the building row, so each station reports extra
# %% NOTE: ranges and a lattice of ghost fixes appears; the
# %% NOTE: residual gate kills most, and the survivors are the
# %% NOTE: dangerous ones -- consistent positions where nothing is
# %% NOTE: flying. \texttt{nlos}: the blocker removes the north-east
# %% NOTE: station's direct view, so its only target-bearing return
# %% NOTE: is the detour and the best association is confidently
# %% NOTE: wrong. That is the failure worth remembering: not a
# %% NOTE: large error announcing itself, but a small residual on
# %% NOTE: the wrong answer.
def ghost_study(params: BaseStationParams, gate_m=2.0,
                out_dir="figures"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    burst = make_ofdm_burst(params)
    initial = np.array([150.0, 100.0, 120.0])   # off the plane!
    true_ranges = np.linalg.norm(DRONE - STATIONS, axis=1)
    scenarios = ("open", "multipath", "nlos")
    fig_map, map_axes = plt.subplots(1, 3, figsize=(13, 4.6),
                                     sharex=True, sharey=True)
    fig_pro, pro_axes = plt.subplots(len(STATIONS), 3,
                                     figsize=(13, 8), sharex=True)
    for column, scenario in enumerate(scenarios):
        rt, scene = multipath_scene(params, scenario)
        _, returns = monostatic_returns(rt, scene)
        rt, empty_scene = multipath_scene(params, scenario,
                                          with_target=False)
        _, clutter = monostatic_returns(rt, empty_scene)
        candidates, profiles, grids, raw_profiles = [], [], [], []
        for (gains, delays), (cg, cd) in zip(returns, clutter):
            echo = roundtrip_echo(params, burst, gains, delays)
            reference = roundtrip_echo(params, burst, cg, cd)
            # Noise is added once, inside range_peaks: the reference
            # is a stored clutter map, not a second noisy capture.
            ranges, profile, grid = range_peaks(params, burst,
                                                echo - reference)
            candidates.append(ranges)
            profiles.append(profile)
            grids.append(grid)
            raw_profiles.append(range_peaks(params, burst, echo)[1])

        # Did the ray tracer actually find the target? A peak within
        # one resolution cell of the true range says yes. This is
        # the check that decides whether the run means anything.
        cell = SPEED_OF_LIGHT / (2.0 * params.bandwidth_hz)
        found = [bool(np.any(np.abs(c - t) < cell))
                 for c, t in zip(candidates, true_ranges)]
        counts = [len(c) for c in candidates]
        print(f"  {scenario:9s}: returns/station "
              f"{[g.size for g, _ in returns]}, peaks {counts}, "
              f"target seen by {sum(found)}/{len(found)} stations")

        if min(counts) == 0:
            fixes, kept = [], []
            print("             a station heard nothing at all -- "
                  "no fix is possible")
        else:
            fixes = associate(candidates, STATIONS, initial)
            kept = [f for f in fixes if f[1] <= gate_m]
            print(f"             {len(fixes)} associations, "
                  f"{len(kept)} survive the {gate_m:.0f} m gate")
        if kept:
            errors = [np.linalg.norm(p - DRONE) for p, _ in kept]
            best = min(kept, key=lambda f: f[1])
            # Two separate questions, and they have different
            # answers: is the truth still in the surviving set, and
            # does the survivor the network would PICK happen to be
            # it? A ghost can fit better than the target.
            truth_survives = min(errors) < 5.0
            print(f"             {sum(e > 5.0 for e in errors)} of "
                  f"{len(kept)} survivors are ghosts; truth "
                  f"{'survives' if truth_survives else 'is GONE'} "
                  f"(closest {min(errors):.1f} m)")
            print(f"             lowest-residual fix is "
                  f"{np.linalg.norm(best[0] - DRONE):6.1f} m from "
                  f"the target (residual {best[1]:.2f} m)")

        axis = map_axes[column]
        buildings = []
        if scenario in ("multipath", "nlos"):
            buildings.append(GHOST_WALL)
        if scenario == "nlos":
            buildings.append(GHOST_BLOCKER)
        for bx, by, bw, bd, _ in buildings:
            axis.add_patch(plt.Rectangle((bx - bw / 2, by - bd / 2),
                                         bw, bd, color="0.75"))
        rejected = np.array([p for p, r in fixes if r > gate_m])
        survived = np.array([p for p, r in kept])
        if rejected.size:
            axis.scatter(rejected[:, 0], rejected[:, 1], s=18,
                         facecolors="none", edgecolors="0.6",
                         label="rejected by residual")
        if survived.size:
            axis.scatter(survived[:, 0], survived[:, 1], s=26,
                         color="C3", label="survives the gate")
        axis.scatter(STATIONS[:, 0], STATIONS[:, 1], marker="s",
                     s=45, color="C0", label="base stations")
        axis.plot(DRONE[0], DRONE[1], "k+", markersize=14,
                  markeredgewidth=2, label="true target")
        axis.set_title(scenario)
        axis.set_xlabel("x (m)")
        axis.set_aspect("equal")
        if column == 0:
            axis.set_ylabel("y (m)")
            axis.legend(fontsize=7, loc="upper left")

        for row in range(len(STATIONS)):
            paxis = pro_axes[row, column]
            profile, grid = profiles[row], grids[row]
            if profile.size:
                keep = grid <= 500.0
                paxis.semilogy(grid[keep], raw_profiles[row][keep],
                               lw=0.7, color="0.7",
                               label="before cancellation")
                paxis.semilogy(grid[keep], profile[keep], lw=0.8,
                               color="C0", label="after")
                for candidate in candidates[row]:
                    paxis.axvline(candidate, color="C3", lw=0.8,
                                  ls=":")
            paxis.axvline(true_ranges[row], color="k", lw=1.0)
            if row == 0 and column == 0:
                paxis.legend(fontsize=6)
            if column == 0:
                paxis.set_ylabel(f"bs-{row}")
            if row == 0:
                paxis.set_title(scenario)
            if row == len(STATIONS) - 1:
                paxis.set_xlabel("range (m)")

    fig_map.tight_layout()
    fig_map.savefig(os.path.join(out_dir, "ghost_map.png"), dpi=150)
    fig_pro.tight_layout()
    fig_pro.savefig(os.path.join(out_dir, "ghost_profiles.png"),
                    dpi=150)


# %% IMAGE: figures/ghost_profiles.png | Matched-filter range profiles, one row per station, one column per scenario. Solid line = the target's true range; dotted lines = the peaks the detector actually reports. Every peak that is not on the solid line is a return from something that is not the target.
# %% IMAGE: figures/ghost_map.png | Every association, solved. Hollow circles are combinations the residual gate rejects; filled circles survive. Open ground gives essentially one answer; the building row breeds a lattice of ghosts and some are consistent enough to survive; under blockage the surviving cluster no longer contains the truth.
# %% IMAGE: figures/scene_ghost.png | The NLOS scene as the ray tracer sees it, with every traced round trip drawn: the blocker cutting the north-east station off, and the building row returning the detour that replaces it. The target IS in this scene as a 4 m metal box -- the enlarged red marker around it is drawn after the solve, so the picture can show where it is.
def render_ghost_scene(params: BaseStationParams,
                       out_dir="figures"):
    os.makedirs(out_dir, exist_ok=True)
    rt, scene = multipath_scene(params, "nlos")
    paths, _ = monostatic_returns(rt, scene)      # SOLVE FIRST
    add_visual_markers(rt, scene, STATIONS, DRONE, scale=16.0)
    camera = rt.Camera(position=[150.0, -430.0, 420.0],
                       look_at=[150.0, 120.0, 30.0])
    scene.render_to_file(
        camera=camera,
        filename=os.path.join(out_dir, "scene_ghost.png"),
        paths=paths, show_devices=False, resolution=(900, 560))


# %% SECTION: Run Part II
# %% NOTE: Everything measured in one run (exact output below the
# %% NOTE: code). Training fields: 500/500 exact timing, 1.03\,kHz
# %% NOTE: frequency error, 0.107\,rad phase spread. Monostatic
# %% NOTE: sensing, line-of-sight: 2 paths (direct + ground bounce),
# %% NOTE: echo $-97.5$\,dBm -- below the $-94$\,dBm floor, saved by
# %% NOTE: the matched filter's 32\,dB -- detection 100\%, range
# %% NOTE: error $-1.2 \pm 0.03$\,m (the small bias is the ground
# %% NOTE: bounce's slightly longer path pulling the peak). Blocked:
# %% NOTE: the solver finds \textbf{zero} paths -- no echo exists at
# %% NOTE: all, detection 0\%; at these scales blockage is total,
# %% NOTE: not a few dB. Blocked + reflector: the detour restores an
# %% NOTE: echo ($-109.2$\,dBm, detection 54\%) but the measured
# %% NOTE: range is \textbf{$+37.4 \pm 1.2$\,m wrong} -- a ghost
# %% NOTE: target at the detour's length. One station cannot tell a
# %% NOTE: detour from a distance: the concrete argument for a
# %% NOTE: \emph{distributed} array that looks from several places
# %% NOTE: at once. Waveforms at 10\,dB: OFDM delay error 0.0
# %% NOTE: samples rms; the smoothed tone 2.8 samples (21\,m), all
# %% NOTE: of it from the envelope; frequency: both at the few-kHz
# %% NOTE: level set by the 40\,$\mu$s burst duration.
if __name__ == "__main__":
    params = BaseStationParams()
    print(f"noise floor: "
          f"{10*math.log10(params.noise_power_w)+30:.1f} dBm")

    print("training fields (500 trials, 15 dB, 20 kHz offset):")
    test_training_fields(params)

    burst = make_ofdm_burst(params)
    for scenario in ("los", "blocked", "reflector"):
        rt, scene = monostatic_scene(params, scenario)
        _, gains, delays = monostatic_legs(rt, scene)
        rate, errors, echo_power = monostatic_range(
            params, burst, gains, delays)
        echo_dbm = (10 * math.log10(echo_power) + 30
                    if echo_power > 0 else float("-inf"))
        line = (f"monostatic {scenario:10s}: {gains.size} paths, "
                f"echo {echo_dbm:6.1f} dBm, "
                f"detection {rate*100:3.0f} %")
        if errors.size:
            line += (f", range error {np.mean(errors):+.1f} "
                     f"+- {np.std(errors):.1f} m")
        print(line)

    print("waveforms (300 trials, 10 dB, 5 kHz offset):")
    compare_waveforms(params)

    localization_study(params)

    print("ghosts (4 stations, built-up scene):")
    ghost_study(params)

    print("rendering scenes and range profiles into figures/ ...")
    render_scenes(params)
    range_profile_figure(params)
    render_ghost_scene(params)
