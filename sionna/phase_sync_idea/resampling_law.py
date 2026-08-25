"""The multipath resampling law: derivation and predictors.

THE MODEL. A sync capture is the preamble x(t) through a static
multipath channel, r(t) = sum_l h_l x(t - tau_l), sampled on the
receiver's grid with an alignment offset delta (samples) relative to
the transmitter's grid. The matched-filter estimator picks the integer
lag k* = argmax |C(k)| and reads the phase of

    C(k* + delta) = sum_l h_l R_x(k* + delta - tau_l),

with R_x the waveform's (band-limited) autocorrelation. Split the
channel into specular + diffuse, h = s + d (Rice factor
K = |s|^2 / E|d|^2):

    C = s R_x(delta) + D(delta),   D(delta) = sum_diffuse d_l R_x(delta - tau_l).

|R_x| is flat to second order at its peak, so the SPECULAR phase is
insensitive to delta. The DIFFUSE sum is a band-limited random process
in delta whose normalized autocovariance rho(Delta) has width set by
the waveform correlation width (~1 sample at critical sampling) and
the delay spread. Reading the composite at two alignments delta_1,
delta_2 changes the measured phase by the re-drawn part of D:

    var[phi(delta_1) - phi(delta_2)] ~= (1/K) * (1 - rho(delta_1 - delta_2)),

using the small-perturbation phase variance Im{D/sR}^2 -> (1/2K) per
fully independent draw and the factor 2(1-rho) for a partial re-draw.

REGIMES (one-way capture):
  - |Delta| << 1 sample:  sigma_mp^2 ~= (1/K) * (1 - rho(Delta))
                          ~ perturbative, quadratic in Delta.
  - |Delta| >= 1 sample (integer lag change, or fractional shifts on a
    waveform whose ambiguity decorrelates within a sample — the
    Zadoff-Chu preamble's near-ideal correlation makes adjacent lags
    nearly independent reads): full re-draw,
                          sigma_mp^2 -> 1/(2K)  (saturation).
    TDL-D's Rice factor is 13.3 dB (K = 21.4): sqrt(1/2K) = 153 mrad
    one-way; the two-way half-difference of two INDEPENDENT alignments
    is (1/4)(2 * 1/(2K)) = 1/(4K) -> 108 mrad. This is the ~100 mrad
    floor previously measured.

WHAT MOVES delta BETWEEN CAPTURES (the mechanism inventory measured by
resampling_law_study.py):
  M1  Integer insertion jitter (the `timing_jitter_samples` draw).
      The discrete channel is shift-invariant and the correlator's
      argmax tracks integer shifts exactly, so BY ITSELF this predicts
      NO resampling noise. (The study tests this directly.)
  M2  Sample-clock (SFO) carry: the fractional part of the accumulated
      clock drift is applied as a genuine fractional resample of the
      composite; it sweeps deterministically (sawtooth), producing a
      COLORED error with the carry period.
  M3  Noise-driven argmax toggling: when the true peak sits near a
      half-sample boundary (fractional carry near 0.5), thermal noise
      flips the detected integer lag capture-to-capture; each flip is
      a full re-draw of the diffuse read => WHITE error with variance
      gated by the fractional position and SNR.
  M4  Thermal phase noise: 1/(2 SNR L) per leg, white, small.

The observed "white ~100 mrad per exchange" is the composition of M2
and M3 over the operating distribution of fractional carries, with
amplitude bounded by the saturation value sqrt(1/2K) (one-way).

WHITENESS CONDITIONS (derived, tested in the study):
  successive one-way errors are independent iff the alignment offsets
  of successive captures are separated by more than the ambiguity
  decorrelation width (>= 1 sample for ZC-class waveforms), OR the
  argmax toggles independently (M3 with fresh noise). Deterministic
  sub-sample carry (M2 alone) violates this: errors are then a
  low-frequency sawtooth, NOT white. i.i.d. insertion jitter (M1)
  contributes nothing and therefore neither whiteness nor color.

This module provides numeric predictors with zero fitted constants:
  - alignment_phase_profile(): the noiseless measured phase phi(delta)
    for the ACTUAL channel taps and ACTUAL estimator, by direct
    synthesis (the exact law, evaluated on the realization).
  - rho_x(): the waveform ambiguity decorrelation rho(Delta).
  - saturation_sigma(): the ensemble closed forms 1/(2K), 1/(4K).
"""

from __future__ import annotations

import math

import torch

from ota_sync.core import REAL_DTYPE, Oscillator, resolve_device, wrap_phase
from ota_sync.sdr import (
    SDRRadioLink,
    SDRSimulationConfig,
    SDRSynchronizer,
    make_sync_preamble,
)

# TDL-D line-of-sight Rice factor (3GPP TR 38.901): 13.3 dB.
TDL_D_RICE_DB = 13.3


def rice_k_linear(model: str) -> float | None:
    if model == "D":
        return 10.0 ** (TDL_D_RICE_DB / 10.0)
    if model == "E":
        return 10.0 ** (22.0 / 10.0)  # TDL-E: 22 dB (TR 38.901)
    return None  # A/B/C are NLOS (no specular term; law saturates hard)


def saturation_sigma(model: str) -> tuple[float, float]:
    """(one-way, two-way half-difference) full-redraw sigma in rad."""

    k = rice_k_linear(model)
    if k is None:
        return float("nan"), float("nan")
    return math.sqrt(1.0 / (2.0 * k)), math.sqrt(1.0 / (4.0 * k))


def _quiet_settings(base: SDRSimulationConfig, **overrides) -> SDRSimulationConfig:
    """Copy of settings with every oscillator/LO noise term off and the
    deterministic two-way biases removed, keeping the RF chain."""

    fields = {name: getattr(base, name) for name in base.__dataclass_fields__}
    fields.update(
        dict(
            phase_noise_std_rad=0.0,
            phase_noise_white_pm_std_rad=0.0,
            flicker_frequency_std_hz=0.0,
            phase_process_std_rad=0.0,
            frequency_process_std_hz=0.0,
            shadowing_std_db=0.0,
            twoway_chain_asymmetry_deg=0.0,
            tdd_turnaround_s=0.0,
        )
    )
    fields.update(overrides)
    return SDRSimulationConfig(**fields)


def alignment_phase_profile(
    settings: SDRSimulationConfig,
    deltas: list[float],
    seed: int = 0,
    snr_db: float = 200.0,
):
    """Noiseless measured phase phi(delta) for the actual taps and the
    actual estimator: synthesize one link, force the receiver alignment
    to each delta (integer part enters the insertion point, fractional
    part the resampler), estimate, return wrapped phases.

    This IS the law sigma_mp = std(phi(delta)) evaluated exactly on the
    channel realization, with zero fitted constants."""

    device = resolve_device("cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 1)
    quiet = _quiet_settings(
        settings, snr_db=snr_db, timing_jitter_samples=0, num_iterations=2
    )
    preamble = make_sync_preamble(quiet, device)
    link = SDRRadioLink(quiet, preamble, device, generator)
    synchronizer = SDRSynchronizer(quiet, preamble)
    zero_cov = torch.zeros(2, 2, dtype=REAL_DTYPE, device=device)
    master = Oscillator(0.3, 0.0, quiet.sync_interval, zero_cov, device, generator)
    slave = Oscillator(-0.5, 0.0, quiet.sync_interval, zero_cov, device, generator)

    phases = []
    for delta in deltas:
        link._timing_carry = float(delta)
        capture = link.capture(master, slave, 0, 0.0)
        est = synchronizer.estimate(capture.samples)
        phases.append(float(wrap_phase(est.phase).item()) if est.detected else float("nan"))
    return phases


def rho_x(settings: SDRSimulationConfig, max_shift: float = 2.0, points: int = 41):
    """Waveform ambiguity decorrelation: normalized correlation between
    the matched filter's diffuse read at alignment 0 and alignment
    Delta, computed from the actual preamble via FFT fractional shifts.
    Returns (shifts, rho)."""

    device = resolve_device("cpu")
    quiet = _quiet_settings(settings)
    preamble = make_sync_preamble(quiet, device)
    x = preamble.long_sequence.to(torch.complex128)
    n = x.numel()
    spectrum = torch.fft.fft(x)
    freqs = torch.fft.fftfreq(n, d=1.0)
    shifts = torch.linspace(-max_shift, max_shift, points)
    base = torch.fft.ifft(spectrum)  # = x
    rho = []
    for shift in shifts:
        shifted = torch.fft.ifft(
            spectrum * torch.exp(-2j * math.pi * freqs * float(shift))
        )
        num = torch.abs(torch.sum(torch.conj(base) * shifted))
        den = torch.sqrt(
            torch.sum(torch.abs(base) ** 2) * torch.sum(torch.abs(shifted) ** 2)
        )
        rho.append(float((num / den).item()))
    return [float(s) for s in shifts], rho


def perturbative_sigma(model: str, rho_at_delta: float) -> float:
    """One-way sigma for a partial re-draw with ambiguity correlation
    rho: sigma^2 = (1/K)(1 - rho) (perturbative regime; saturates at
    1/(2K) when rho -> 0 would overshoot, so cap there)."""

    k = rice_k_linear(model)
    if k is None:
        return float("nan")
    return math.sqrt(min((1.0 / k) * (1.0 - rho_at_delta), 1.0 / (2.0 * k)))
