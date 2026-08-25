"""Waveform-level coherent detection of a passive target - real software
testing, not detection statistics.

What is actually simulated, per Monte-Carlo trial:

1. Every station transmits the same probe burst — by default a
   5G-style OFDM frame (random QPSK subcarriers + cyclic prefix), i.e.
   a communication-type waveform as an ISAC deployment would radiate;
   Zadoff-Chu remains the BS-to-BS SYNC waveform and is available here
   only as an option. The burst is pre-steered with the KNOWN
   station->target geometry (positions are known; the target cell is
   the hypothesis under test). What geometry pre-steering cannot remove is each
   station's synchronization residual theta_k - and those are taken
   directly from a measured sync run, one random steady-state time
   sample per trial.
2. The fields superpose at the target: complex amplitude
   sum_k sqrt(P)*c_k*exp(j*theta_k), with c_k the per-path spreading
   loss lambda/( (4*pi)^(3/2) )-consistent bistatic factors below. The
   target reradiates with a Swerling-1 fluctuating RCS (complex
   Gaussian draw per trial).
3. Every station receives the echo: delayed pulse, per-receiver
   carrier phase (geometry part hypothesized away, sync residual
   theta_j remains), plus REAL complex AWGN sample streams at the
   thermal floor kT0*F*fs.
4. Receive processing on the raw samples: matched filter each
   receiver's stream at the hypothesized delay gate, coherently sum
   across receivers, detect |sum|^2 against a threshold.
5. The threshold is EMPIRICAL: calibrated from target-absent trials of
   the same pipeline at the requested false-alarm rate, and the
   achieved Pfa is re-measured and reported.

P_d is then the counted fraction of target-present trials above
threshold. No Swerling closed forms anywhere in the loop.

Honest scope (the remaining gaps to full radar fidelity): single
hypothesized range gate (cued detection - no CFAR sweep), point target
(no micro-Doppler), no ground clutter or direct-path self-interference
(assumed time-gated), target static within the pulse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .viability import BOLTZMANN_T0, SPEED_OF_LIGHT, DetectionParams


def _zadoff_chu(length: int, root: int) -> torch.Tensor:
    n = torch.arange(length, dtype=torch.float64)
    if length % 2 == 0:
        phase = -math.pi * root * n * n / length
    else:
        phase = -math.pi * root * n * (n + 1) / length
    return torch.exp(1j * phase)


def _ofdm_burst(
    num_samples: int,
    generator: torch.Generator,
    subcarriers: int = 64,
    cyclic_prefix: int = 16,
) -> torch.Tensor:
    """5G-style OFDM burst: random QPSK on every subcarrier, IFFT + CP.

    This is the DETECTION waveform — a communication-type signal, as an
    ISAC deployment would transmit — while Zadoff-Chu remains what the
    base stations use among themselves for synchronization. The
    receiver knows the transmitted frame (it is its own reference), so
    matched filtering applies unchanged. Normalized to unit average
    power; note OFDM's higher PAPR (PA effects on the detection
    transmit chain are not modeled at this layer).
    """

    symbol_length = subcarriers + cyclic_prefix
    num_symbols = max(1, num_samples // symbol_length)
    angles = (
        torch.randint(
            0, 4, (num_symbols, subcarriers), generator=generator
        ).to(torch.float64)
        * (math.pi / 2.0)
        + math.pi / 4.0
    )
    frequency_domain = torch.exp(1j * angles.to(torch.complex128))
    time_domain = torch.fft.ifft(frequency_domain, dim=1) * math.sqrt(
        float(subcarriers)
    )
    with_cp = torch.cat(
        (time_domain[:, -cyclic_prefix:], time_domain), dim=1
    ).reshape(-1)
    return with_cp / torch.sqrt(torch.mean(torch.abs(with_cp) ** 2))


@dataclass(frozen=True)
class WaveformDetectionResult:
    """Counted detection performance of one method at a set of ranges."""

    label: str
    num_stations: int
    ranges_m: list[float]
    pd_measured: list[float]
    measured_pfa: float
    threshold_pfa: float
    trials_per_range: int
    combining_loss_db: list[float]


def run_waveform_detection(
    label: str,
    positions: np.ndarray,
    residual_phases: torch.Tensor,
    targets_m: np.ndarray,
    params: DetectionParams = DetectionParams(),
    pulse_length: int = 1023,
    trials: int = 2000,
    h0_trials: int = 60000,
    threshold_pfa: float = 1e-3,
    seed: int = 0,
    waveform: str = "ofdm",
    leg_gains: np.ndarray | None = None,
) -> WaveformDetectionResult:
    """Monte-Carlo waveform detection with measured sync residuals.

    ``residual_phases``: (num_stations, num_steady_samples) tensor of
    per-station phase residuals (station 0's row is typically zeros -
    the reference/datum). Each trial draws one time column at random,
    so P_d is averaged over the measured residual process.

    ``threshold_pfa`` is deliberately larger than an operational 1e-6:
    an empirical threshold at 1e-6 would need >1e7 target-absent
    trials. The comparison across methods is unaffected (same
    threshold pipeline for all).
    """

    generator = torch.Generator().manual_seed(seed)
    num_stations = positions.shape[0]
    if waveform == "ofdm":
        pulse = _ofdm_burst(pulse_length, generator)
    elif waveform == "zc":
        pulse = _zadoff_chu(pulse_length, 25)
    else:
        raise ValueError("waveform must be 'ofdm' or 'zc'")
    pulse_length = pulse.shape[0]

    # Thermal noise per complex sample at the full sample rate. The
    # pulse bandwidth equals fs, so this is the matched noise floor.
    sample_rate = 1e6
    noise_power = (
        BOLTZMANN_T0
        * 10.0 ** (params.noise_figure_db / 10.0)
        * 10.0 ** (params.losses_db / 10.0)
        * sample_rate
    )
    noise_std = math.sqrt(noise_power / 2.0)
    antenna_gain = 10.0 ** (params.antenna_gain_dbi / 10.0)
    wavelength = params.wavelength_m

    # ---- Empirical threshold from target-absent trials -------------
    # Matched-filter each receiver's noise-only stream at the gate and
    # combine; the statistic is |sum_j <n_j, s>|^2 computed on real
    # sample draws (window = pulse length; the MF at a single gate
    # only sees these samples). Batched to bound memory.
    batch = 2000
    h0_values = []
    remaining = h0_trials
    while remaining > 0:
        count = min(batch, remaining)
        remaining -= count
        noise = (
            torch.randn(
                count,
                num_stations,
                pulse_length,
                2,
                dtype=torch.float64,
                generator=generator,
            )
            * noise_std
        )
        streams = torch.view_as_complex(noise.contiguous())
        mf = torch.einsum("tjk,k->tj", streams, torch.conj(pulse))
        h0_values.append(torch.abs(torch.sum(mf, dim=1)) ** 2)
    h0_stat = torch.cat(h0_values)
    threshold = torch.quantile(h0_stat, 1.0 - threshold_pfa).item()
    measured_pfa = torch.mean((h0_stat > threshold).to(torch.float64)).item()

    # ---- Target-present trials per range ----------------------------
    # The number of coherently integrated pulses reproduces the
    # configured integration time (pulse repetition assumed back to
    # back at the same residual within one sync interval).
    pulses_per_cpi = max(
        1, int(round(params.integration_time_s * sample_rate / pulse_length))
    )
    steady_columns = residual_phases.shape[1]

    pd_measured: list[float] = []
    combining_loss_db: list[float] = []
    ranges_m: list[float] = []
    centroid = positions.mean(axis=0)
    target_list = np.atleast_2d(np.asarray(targets_m, dtype=float))
    for target_index, target in enumerate(target_list):
        ranges_m.append(float(np.linalg.norm(target - centroid)))
        if leg_gains is not None:
            # Ray-traced steered legs (antenna gain and propagation
            # already inside; see detection/rt_echo.py). The bistatic
            # echo is leg_k * leg_j * sqrt(4*pi*sigma)/lambda; sigma is
            # carried by the Swerling draw below.
            base_amplitude = math.sqrt(
                params.tx_power_w * 4.0 * math.pi
            ) / wavelength
            inverse_distance = torch.tensor(
                leg_gains[target_index], dtype=torch.complex128
            )
        else:
            distances = np.linalg.norm(positions - target, axis=1)
            distances = np.maximum(distances, 1.0)
            # Bistatic pair (k transmit, j receive) amplitude so that
            # the monostatic single-station case reproduces the radar
            # equation P*G^2*lambda^2*sigma / ((4pi)^3 d^4) exactly:
            #   A_kj = sqrt(P*G^2*lambda^2/(4pi)^3) * chi*sqrt(sigma)
            #          / (d_k * d_j)
            base_amplitude = math.sqrt(
                params.tx_power_w
                * antenna_gain**2
                * wavelength**2
                / (4.0 * math.pi) ** 3
            )
            inverse_distance = torch.tensor(
                1.0 / distances, dtype=torch.float64
            ).to(torch.complex128)

        columns = torch.randint(
            0, steady_columns, (trials,), generator=generator
        )
        theta = residual_phases[:, columns].T  # (trials, stations)
        phasors = torch.exp(1j * theta.to(torch.complex128))

        # Field at the target: geometry pre-steered away, sync residual
        # and per-path spreading remain. Swerling-1 RCS draw per trial.
        tx_field = torch.einsum("tk,k->t", phasors, inverse_distance)
        rcs_draw = (
            torch.randn(trials, 2, dtype=torch.float64, generator=generator)
            / math.sqrt(2.0)
        )
        rcs_amp = torch.view_as_complex(rcs_draw.contiguous()) * math.sqrt(
            params.rcs_m2
        )

        # Echo amplitude arriving at each receiver j.
        echo = (
            base_amplitude
            * tx_field.unsqueeze(1)
            * rcs_amp.unsqueeze(1)
            * inverse_distance.unsqueeze(0)
            * phasors
        )  # (trials, stations)

        # Build the actual received sample streams at the gate and run
        # the actual matched filter, one CPI pulse at a time. Batched
        # over trials to bound memory.
        hits = 0
        start = 0
        while start < trials:
            stop = min(start + 500, trials)
            echo_batch = echo[start:stop]
            cpi_sum = torch.zeros(stop - start, dtype=torch.complex128)
            for _ in range(pulses_per_cpi):
                noise = (
                    torch.randn(
                        stop - start,
                        num_stations,
                        pulse_length,
                        2,
                        dtype=torch.float64,
                        generator=generator,
                    )
                    * noise_std
                )
                streams = torch.view_as_complex(noise.contiguous())
                streams = streams + echo_batch.unsqueeze(-1) * pulse.unsqueeze(
                    0
                ).unsqueeze(0)
                mf = torch.einsum("tjk,k->tj", streams, torch.conj(pulse))
                cpi_sum = cpi_sum + torch.sum(mf, dim=1)
            # CPI-coherent statistic normalized to the single-pulse
            # threshold scale.
            statistic = torch.abs(cpi_sum) ** 2 / pulses_per_cpi
            hits += int(torch.sum(statistic > threshold).item())
            start = stop
        pd_measured.append(hits / trials)

        # Waveform-measured combining loss vs perfect sync: ratio of
        # the mean coherent echo power to the same with theta = 0.
        perfect_power = (
            base_amplitude
            * torch.abs(torch.sum(inverse_distance)) ** 2
        ) ** 2
        actual_power = torch.mean(
            torch.abs(
                torch.sum(
                    base_amplitude
                    * tx_field.unsqueeze(1)
                    * inverse_distance.unsqueeze(0)
                    * phasors,
                    dim=1,
                )
            )
            ** 2
        )
        combining_loss_db.append(
            10.0
            * math.log10(
                max(actual_power.item() / perfect_power.item(), 1e-12)
            )
        )

    return WaveformDetectionResult(
        label=label,
        num_stations=num_stations,
        ranges_m=list(ranges_m),
        pd_measured=pd_measured,
        measured_pfa=measured_pfa,
        threshold_pfa=threshold_pfa,
        trials_per_range=trials,
        combining_loss_db=combining_loss_db,
    )
