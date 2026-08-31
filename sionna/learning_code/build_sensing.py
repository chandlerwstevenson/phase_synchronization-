from __future__ import annotations
import math
import os
import tempfile
from dataclasses import dataclass
import numpy as np
BOLTZMANN_T0 = 1.380649e-23 * 290.0
SPEED_OF_LIGHT = 299792458.0
rng = np.random.default_rng(0) 


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


