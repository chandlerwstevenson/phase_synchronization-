"""Common metrics layer: score any synchronization/membership method on
several system-level metrics from the same measured residuals.

Every function takes the same raw material the studies already produce
— a residual matrix (stations x time samples, row 0 the reference), a
membership-weight matrix of the same shape, station positions, and the
run's synchronization airtime fraction — and returns one metric:

  probability_of_detection   counted fraction of Monte-Carlo trials in
                             which the radar detects the target (wraps
                             the existing waveform pipelines; supports
                             both the discard combiner and the two-tier
                             coherent + noncoherent combiner)
  spectral_efficiency        communication throughput to a user
                             terminal in bits per second per hertz,
                             log2(1 + signal-to-noise ratio) per time
                             draw; reports the mean and the 95%-likely
                             value (the throughput exceeded 95% of the
                             time — the convention of Qin et al. 2024)
  mean_array_gain            beam quality: |sum of weighted phasors|^2
                             normalized to the perfect full array
  detection_range_m          the range (meters) at which the array
                             still meets the configured detection
                             requirement, from the link-budget layer
  net_throughput             (1 - sync airtime fraction) x mean
                             spectral efficiency — the honest
                             system-level number, since every second
                             spent synchronizing is a second not spent
                             communicating

Conventions (documented, deliberate):
- For communication, the membership weights scale TRANSMIT amplitude
  (a benched station does not send the user's data). An optional
  noncoherent tier models demoted stations Qin-style as a second data
  stream decoded with successive interference cancellation: stream 1
  (coherent group) is decoded first with the demoted group's signal
  counted as interference, then cancelled, then stream 2 is decoded
  against the noise floor alone. Passing no tier gives the plain
  single-stream formula.
- The user terminal antenna is 0 dBi; stations use the configured
  antenna gain; free-space (Friis) path amplitudes at the configured
  carrier; thermal noise floor kT0 x noise figure x losses x bandwidth
  (same noise model as the detection pipeline).
- Detection wrappers do not re-implement anything: they call
  gating_study.run_gated_waveform_detection (weights on both transmit
  and the receive combiner) or
  hybrid_combiner_study.run_hybrid_waveform_detection (transmit all-in,
  receive-side two-tier combiner). Pick the convention explicitly via
  ``combiner``.

Nothing existing is modified; this module only imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from detection import DetectionParams
from detection.viability import (
    BOLTZMANN_T0,
    SPEED_OF_LIGHT,
    detection_range_m as _viability_range_m,
)
from gating_study import run_gated_waveform_detection, weighted_gain
from hybrid_combiner_study import run_hybrid_waveform_detection


# ---------------------------------------------------------------------
# Spectral efficiency
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class SpectralEfficiencyResult:
    """Throughput to one user terminal, in bits per second per hertz."""

    mean_bps_hz: float
    likely95_bps_hz: float  # value exceeded 95% of the time
    mean_snr_db: float
    per_draw_bps_hz: torch.Tensor  # (time draws,)


def _path_amplitudes(
    positions: np.ndarray,
    user_m: np.ndarray,
    antenna_gain_dbi: float,
    user_gain_dbi: float,
    carrier_frequency_hz: float,
) -> torch.Tensor:
    """Free-space (Friis) amplitude gain of each station->user path."""

    distances = np.linalg.norm(
        np.asarray(positions, dtype=float) - np.asarray(user_m, dtype=float),
        axis=1,
    )
    distances = np.maximum(distances, 1.0)
    wavelength = SPEED_OF_LIGHT / carrier_frequency_hz
    gain_product = 10.0 ** ((antenna_gain_dbi + user_gain_dbi) / 20.0)
    return torch.tensor(
        gain_product * wavelength / (4.0 * math.pi * distances),
        dtype=torch.float64,
    )


def spectral_efficiency(
    residual_phases: torch.Tensor,
    weights: torch.Tensor,
    positions: np.ndarray,
    user_m: np.ndarray,
    tx_power_w: float,
    noncoherent_weights: torch.Tensor | None = None,
    antenna_gain_dbi: float = 6.0,
    user_gain_dbi: float = 0.0,
    carrier_frequency_hz: float = 915e6,
    bandwidth_hz: float = 1e6,
    noise_figure_db: float = 6.0,
    losses_db: float = 3.0,
) -> SpectralEfficiencyResult:
    """Downlink throughput to a user terminal at ``user_m``.

    ``weights`` scale each station's transmit amplitude for the
    coherent data stream. ``noncoherent_weights`` (optional, same
    shape) scale amplitudes of a second, demoted group carrying its own
    stream, decoded by successive interference cancellation: stream 1
    sees stream 2 as interference, stream 2 is decoded after stream 1
    is cancelled. Per time draw the channel (residual phases) is known,
    so both group amplitudes are formed from the same phasor draw.
    """

    if weights.shape != residual_phases.shape:
        raise ValueError("weights must align with residual_phases")
    if noncoherent_weights is not None and (
        noncoherent_weights.shape != residual_phases.shape
    ):
        raise ValueError("noncoherent_weights must align with residual_phases")

    amplitudes = _path_amplitudes(
        positions, user_m, antenna_gain_dbi, user_gain_dbi,
        carrier_frequency_hz,
    )
    phasors = torch.exp(1j * residual_phases.to(torch.complex128))
    scaled = amplitudes.unsqueeze(1).to(torch.complex128) * phasors

    noise_w = (
        BOLTZMANN_T0
        * 10.0 ** (noise_figure_db / 10.0)
        * 10.0 ** (losses_db / 10.0)
        * bandwidth_hz
    )

    coherent_amp2 = (
        torch.abs(
            torch.sum(weights.to(torch.complex128) * scaled, dim=0)
        )
        ** 2
    ) * tx_power_w
    if noncoherent_weights is None:
        snr = coherent_amp2 / noise_w
        per_draw = torch.log2(1.0 + snr)
    else:
        demoted_amp2 = (
            torch.abs(
                torch.sum(
                    noncoherent_weights.to(torch.complex128) * scaled, dim=0
                )
            )
            ** 2
        ) * tx_power_w
        sinr_stream1 = coherent_amp2 / (noise_w + demoted_amp2)
        snr_stream2 = demoted_amp2 / noise_w
        per_draw = torch.log2(1.0 + sinr_stream1) + torch.log2(
            1.0 + snr_stream2
        )
        snr = (coherent_amp2 + demoted_amp2) / noise_w  # for reporting

    mean = torch.mean(per_draw).item()
    likely95 = torch.quantile(per_draw, 0.05).item()
    mean_snr_db = 10.0 * math.log10(max(torch.mean(snr).item(), 1e-30))
    return SpectralEfficiencyResult(
        mean_bps_hz=mean,
        likely95_bps_hz=likely95,
        mean_snr_db=mean_snr_db,
        per_draw_bps_hz=per_draw,
    )


# ---------------------------------------------------------------------
# Probability of detection
# ---------------------------------------------------------------------

DETECTION_COMBINERS = ("gated", "two-tier-discard", "two-tier-noncoherent")


def probability_of_detection(
    label: str,
    positions: np.ndarray,
    residual_phases: torch.Tensor,
    weights: torch.Tensor,
    targets_m: np.ndarray,
    combiner: str = "two-tier-discard",
    params: DetectionParams = DetectionParams(),
    trials: int = 400,
    h0_trials: int = 15000,
    seed: int = 0,
    **kwargs,
):
    """Counted detection under a stated combiner convention.

    combiner="gated": weights scale transmit AND receive (the
    gating_study convention). "two-tier-discard": transmit all-in,
    weighted receivers only (combiner-only benching).
    "two-tier-noncoherent": transmit all-in, benched receivers added
    square-law (the demote-don't-discard combiner).
    """

    if combiner not in DETECTION_COMBINERS:
        raise ValueError(f"combiner must be one of {DETECTION_COMBINERS}")
    if combiner == "gated":
        return run_gated_waveform_detection(
            label, positions, residual_phases, weights, targets_m,
            params=params, trials=trials, h0_trials=h0_trials, seed=seed,
            **kwargs,
        )
    mode = "discard" if combiner == "two-tier-discard" else "noncoherent"
    return run_hybrid_waveform_detection(
        label, positions, residual_phases, weights, mode, targets_m,
        params=params, trials=trials, h0_trials=h0_trials, seed=seed,
        **kwargs,
    )


# ---------------------------------------------------------------------
# Beam quality, range, net throughput
# ---------------------------------------------------------------------

def mean_array_gain(
    residual_phases: torch.Tensor, weights: torch.Tensor
) -> float:
    """Mean beam quality relative to the perfect full array."""

    return torch.mean(weighted_gain(residual_phases, weights)).item()


def detection_range_m(
    num_stations: int,
    sync_gain: float,
    params: DetectionParams = DetectionParams(),
) -> float:
    """Range (meters) at which the array still meets the configured
    detection requirement, treating the measured gain as acting on both
    the transmit and receive legs (the link-budget approximation of
    detection/viability.py)."""

    return _viability_range_m(num_stations, sync_gain, params)


def net_throughput(
    mean_se_bps_hz: float, sync_airtime_fraction: float
) -> float:
    """(1 - sync airtime) x mean spectral efficiency: throughput after
    paying the synchronization overhead."""

    airtime = min(max(sync_airtime_fraction, 0.0), 1.0)
    return (1.0 - airtime) * mean_se_bps_hz
