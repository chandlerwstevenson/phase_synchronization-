"""Part II of the coding guide: realistic base-station parameters,
short/long training fields built from the ground up, single-station
monostatic sensing in Sionna (line-of-sight and blocked), and OFDM
versus single-tone waveforms.

New code, written for this guide and tested; every measured number
quoted on the slides comes from running this file:

    python3 build_sensing.py
"""

from __future__ import annotations

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
def _ply_box(x, y, w, d, h):
    x0, x1, y0, y1 = x - w / 2, x + w / 2, y - d / 2, y + d / 2
    v = [(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0),
         (x0, y0, h), (x1, y0, h), (x1, y1, h), (x0, y1, h)]
    f = [(0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7),
         (2, 7, 6), (3, 0, 4), (3, 4, 7), (4, 5, 6), (4, 6, 7)]
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
def monostatic_scene(params: BaseStationParams, scenario: str):
    """Build one of the three scenes; returns (rt, scene)."""
    import sionna.rt as rt

    scene = rt.load_scene()
    scene.frequency = params.carrier_frequency_hz
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                    pattern="iso", polarization="V")
    scene.rx_array = scene.tx_array

    def add(name, ply, material):
        handle = tempfile.NamedTemporaryFile("w", suffix=".ply",
                                             delete=False)
        handle.write(ply)
        handle.close()
        try:
            scene.edit(add=rt.SceneObject(
                fname=handle.name, name=name, radio_material=material))
        finally:
            os.unlink(handle.name)

    ground = rt.RadioMaterial("ground", thickness=10.0,
                              relative_permittivity=15.0,
                              conductivity=0.035)
    add("ground-plane", _ply_ground(800.0), ground)
    concrete = rt.RadioMaterial("concrete", thickness=0.3,
                                relative_permittivity=5.24,
                                conductivity=0.123)
    if scenario in ("blocked", "reflector"):
        add("blocker", _ply_box(150.0, 0.0, 40.0, 40.0, 60.0),
            concrete)
    if scenario == "reflector":
        # A long building row whose street-side wall (y = 80 m) can
        # mirror the signal around the blocker.
        add("reflector", _ply_box(150.0, 90.0, 200.0, 20.0, 60.0),
            concrete)

    scene.add(rt.Transmitter("bs", position=[0.0, 0.0, 15.0]))
    scene.add(rt.Receiver("target-probe",
                          position=[300.0, 0.0, 40.0]))
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
# %% IMAGE: figures/scene_los.png | Line-of-sight: the direct ray and the ground bounce reach the target.
# %% IMAGE: figures/scene_blocked.png | Blocked: the solver finds no path at all -- there are no rays to draw.
# %% IMAGE: figures/scene_reflector.png | Blocked + reflector: the wall of the building row mirrors the signal around the blocker (three paths, all longer than the straight line).
def render_scenes(params: BaseStationParams, out_dir="figures"):
    os.makedirs(out_dir, exist_ok=True)
    for scenario in ("los", "blocked", "reflector"):
        rt, scene = monostatic_scene(params, scenario)
        paths, gains, _ = monostatic_legs(rt, scene)
        camera = rt.Camera(position=[150.0, -340.0, 300.0],
                           look_at=[150.0, 20.0, 20.0])
        scene.render_to_file(
            camera=camera,
            filename=os.path.join(out_dir,
                                  f"scene_{scenario}.png"),
            # The renderer cannot overlay an empty path list.
            paths=paths if gains.size else None,
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

    print("rendering scenes and range profiles into figures/ ...")
    render_scenes(params)
    range_profile_figure(params)
