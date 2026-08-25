"""From synchronization residuals to drone-detection viability.

The bridge between the sync simulators and radar detection:

  Transmit focusing: N stations transmitting the same waveform with
  per-station phase errors theta_k put field amplitude sum_k e^{j
  theta_k} on the target, i.e. illuminating power N^2 * G * P where

      G = | sum_k e^{j theta_k} |^2 / N^2

  is exactly the array coherent gain every sync result in this
  repository measures. Coherent receive combining (station positions
  known, target cell hypothesized, so only sync error decorrelates the
  sum) contributes another factor N * G. Relative to one station:

      SNR_N = N^3 * G^2 * SNR_1

  Perfect sync:  G = 1        ->  N^3        (the coherent prize)
  Free running:  G ~= 1/N     ->  N^3/N^2 = N (incoherent - no prize)
  Synchronized:  G measured   ->  in between; squared, because sync
                                  errors hurt on transmit AND receive.

Link budget (coherent integration over T_int, energy form - the
bandwidth cancels):

      SNR_1 = P_t * G_a^2 * lambda^2 * sigma_rcs * T_int
              / ( (4*pi)^3 * R^4 * k*T0*F * L )

Detection: Swerling-1 (slowly fluctuating target, appropriate for a
small drone), square-law detector:

      P_d = P_fa^( 1 / (1 + SNR) )
      SNR_required(P_d, P_fa) = ln(P_fa)/ln(P_d) - 1
      (P_d = 0.9, P_fa = 1e-6  ->  SNR_req ~= 130 ~= 21.2 dB)

Default target: small quadcopter, RCS 0.03 m^2 (~ -15 dBsm; published
UHF/L-band measurements of DJI-class drones span roughly -20..-10
dBsm). Default integration time: one sync interval (50 ms) - within a
CPI the loop holds the residuals the sync results report, and drone
maneuvering limits useful CPIs to this order anyway.

Honest scope: this is a link-budget + detection-statistics layer on
top of the measured sync residuals. It does not simulate target
echoes at the waveform level (no micro-Doppler, no clutter, no
direct-path interference); those are the next fidelity step and are
listed as such wherever results are reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

BOLTZMANN_T0 = 1.380649e-23 * 290.0
SPEED_OF_LIGHT = 299792458.0


@dataclass(frozen=True)
class DetectionParams:
    """Radar-side assumptions for the viability study."""

    tx_power_w: float = 1.0
    antenna_gain_dbi: float = 6.0
    carrier_frequency_hz: float = 915e6
    rcs_m2: float = 0.03
    integration_time_s: float = 0.05
    noise_figure_db: float = 6.0
    losses_db: float = 3.0
    pfa: float = 1e-6
    pd_target: float = 0.9

    @property
    def wavelength_m(self) -> float:
        return SPEED_OF_LIGHT / self.carrier_frequency_hz


def required_snr(pfa: float, pd: float) -> float:
    """Swerling-1 single-look SNR needed for (pd, pfa)."""

    return math.log(pfa) / math.log(pd) - 1.0


def probability_of_detection(snr: float, pfa: float) -> float:
    """Swerling-1 square-law detector."""

    if snr <= 0.0:
        return pfa
    return pfa ** (1.0 / (1.0 + snr))


def single_node_snr(range_m: float, params: DetectionParams) -> float:
    """Monostatic single-station coherent-integration SNR at range R."""

    gain = 10.0 ** (params.antenna_gain_dbi / 10.0)
    noise = (
        BOLTZMANN_T0
        * 10.0 ** (params.noise_figure_db / 10.0)
        * 10.0 ** (params.losses_db / 10.0)
    )
    numerator = (
        params.tx_power_w
        * gain**2
        * params.wavelength_m**2
        * params.rcs_m2
        * params.integration_time_s
    )
    return numerator / ((4.0 * math.pi) ** 3 * range_m**4 * noise)


def coherent_snr_factor(num_stations: int, sync_gain: float) -> float:
    """SNR multiple over a single station: N^3 * G^2.

    N^2 * G from transmit focusing, N * G from coherent receive
    combining; the sync error enters on both legs, hence G squared.
    """

    return float(num_stations) ** 3 * sync_gain**2


def detection_range_m(
    num_stations: int, sync_gain: float, params: DetectionParams
) -> float:
    """Largest range with P_d >= pd_target at pfa (Swerling 1)."""

    snr_needed = required_snr(params.pfa, params.pd_target)
    # SNR_1(R) * N^3 G^2 = snr_needed, SNR_1 ~ R^-4.
    reference = single_node_snr(1.0, params)  # SNR at 1 m
    factor = coherent_snr_factor(num_stations, sync_gain)
    return (reference * factor / snr_needed) ** 0.25


@dataclass(frozen=True)
class MethodViability:
    """Detection viability of one synchronization method."""

    label: str
    num_stations: int
    sync_gain: float
    params: DetectionParams

    @property
    def snr_factor_db(self) -> float:
        return 10.0 * math.log10(
            coherent_snr_factor(self.num_stations, self.sync_gain)
        )

    @property
    def range_m(self) -> float:
        return detection_range_m(self.num_stations, self.sync_gain, self.params)

    @property
    def range_vs_single(self) -> float:
        return self.range_m / detection_range_m(1, 1.0, self.params)

    @property
    def range_vs_perfect(self) -> float:
        return self.range_m / detection_range_m(
            self.num_stations, 1.0, self.params
        )

    def pd_at(self, range_m: float) -> float:
        snr = single_node_snr(range_m, self.params) * coherent_snr_factor(
            self.num_stations, self.sync_gain
        )
        return probability_of_detection(snr, self.params.pfa)


def evaluate_method(
    label: str,
    num_stations: int,
    sync_gain: float,
    params: DetectionParams = DetectionParams(),
) -> MethodViability:
    """Package one method's measured array gain as detection viability."""

    return MethodViability(
        label=label,
        num_stations=num_stations,
        sync_gain=sync_gain,
        params=params,
    )
