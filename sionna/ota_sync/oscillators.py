"""Named oscillator noise profiles anchored to real, purchasable parts.

Each profile stores fractional-frequency stability (Allan deviation) and
phase-jitter anchors taken from datasheets and standards, then converts
them into the simulator's per-sample / per-interval noise knobs for a
given carrier frequency and sample rate. The conversion identities:

  white FM   sigma_y(1 s) -> per-sample phase-walk std
             sigma_pn = sigma_y(1s) * 2*pi*f_c / sqrt(f_s)
             (a phase random walk of per-sample variance sigma_pn^2
             accumulates sigma_pn^2 * f_s * tau over tau seconds, giving
             sigma_y(tau) = sigma_pn * sqrt(f_s) / (2*pi*f_c*sqrt(tau)))
  flicker FM ADEV flicker floor sigma_y -> RMS frequency deviation
             flicker_std_hz = sigma_y * f_c
  RW FM      frequency wander (aging + temperature), expressed as a
             Hz-per-sqrt-second walk rate at the carrier
  white PM   integrated synthesizer/PLL jitter at the carrier, expressed
             directly as an RMS phase per sample

Sources for the anchors:
  - 3GPP TS 38.104: base-station air-interface frequency error limits,
    +-50 ppb wide area, +-100 ppb local area / small cell.
  - Rakon "Stratum 3E OCXO Product Brief" (ROM1490E class, the family
    marketed for G.8263 base-station timing): 10 ppb pk-pk stability
    over temperature, ageing < +-1 ppb/day, noise floor down to
    -158 dBc/Hz at 100 kHz offset on the 10-40 MHz output.
  - Telecom Stratum 3E OCXO class ADEV(1 s) ~ 1e-11 (premium parts such
    as the CTS OX-049 reach 2e-13; 1e-11 is a conservative class value).
  - Small-cell / ultra-stable TCXO class (Rakon small-cell family):
    ADEV(1 s) ~ 1e-10, +-50..100 ppb over temperature.
  - Bench-SDR TCXO class (USRP B2xx style, +-2 ppm accuracy):
    ADEV(1 s) ~ 3e-10..1e-9 (GPS-receiver-grade TCXO); the profile uses
    5e-10.

The phase noise these produce is referred to the 915 MHz carrier: a
reference oscillator multiplied to the carrier keeps its fractional
stability (ADEV) while its phase noise scales with the multiplication,
which is exactly how the conversion above treats it. The synthesizer's
own additive jitter is folded into the white-PM anchor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class OscillatorProfile:
    """Datasheet-level description of one oscillator class."""

    name: str
    description: str
    adev_1s_white_fm: float
    adev_flicker_floor: float
    frequency_walk_hz_per_root_s: float
    white_pm_std_rad: float
    accuracy_ppb: float

    def noise_settings(
        self,
        carrier_frequency_hz: float,
        sample_rate: float,
        sync_interval: float,
    ) -> dict[str, float]:
        """Convert the datasheet anchors into simulator noise knobs."""

        return {
            "phase_noise_std_rad": (
                self.adev_1s_white_fm
                * 2.0
                * math.pi
                * carrier_frequency_hz
                / math.sqrt(sample_rate)
            ),
            "flicker_frequency_std_hz": (
                self.adev_flicker_floor * carrier_frequency_hz
            ),
            "frequency_process_std_hz": (
                self.frequency_walk_hz_per_root_s * math.sqrt(sync_interval)
            ),
            "phase_noise_white_pm_std_rad": self.white_pm_std_rad,
            # The per-interval oscillator-state walk is already covered by
            # the white-FM term; keep only a token allocation.
            "phase_process_std_rad": 1e-4,
        }

    def expected_cfo_hz(self, carrier_frequency_hz: float) -> float:
        """Initial frequency error implied by the part's accuracy."""

        return carrier_frequency_hz * self.accuracy_ppb * 1e-9


OSCILLATOR_PROFILES: dict[str, OscillatorProfile] = {
    # Macro base station: Stratum 3E OCXO (Rakon ROM1490E class),
    # GNSS/SyncE-disciplined to the 3GPP wide-area +-50 ppb limit.
    "ocxo": OscillatorProfile(
        name="ocxo",
        description=(
            "macro base station Stratum 3E OCXO (Rakon ROM1490E class), "
            "disciplined to the 3GPP wide-area +-50 ppb limit"
        ),
        adev_1s_white_fm=1e-11,
        adev_flicker_floor=3e-12,
        frequency_walk_hz_per_root_s=4.5e-3,
        white_pm_std_rad=1e-3,
        accuracy_ppb=50.0,
    ),
    # Small cell: ultra-stable TCXO (Rakon small-cell family), 3GPP
    # local-area +-100 ppb limit.
    "tcxo": OscillatorProfile(
        name="tcxo",
        description=(
            "small-cell ultra-stable TCXO (Rakon small-cell class), "
            "3GPP local-area +-100 ppb limit"
        ),
        adev_1s_white_fm=1e-10,
        adev_flicker_floor=5e-11,
        frequency_walk_hz_per_root_s=0.45,
        white_pm_std_rad=3e-3,
        accuracy_ppb=100.0,
    ),
    # Bench SDR without GPSDO: USRP B2xx-style +-2 ppm TCXO. Carrier
    # phase wanders ~radians per 50 ms at 915 MHz - this class is why
    # real SDR sync demos add a GPSDO or shorten the pilot cadence.
    "sdr": OscillatorProfile(
        name="sdr",
        description=(
            "bench SDR TCXO without GPSDO (USRP B2xx class, +-2 ppm)"
        ),
        adev_1s_white_fm=5e-10,
        adev_flicker_floor=2e-10,
        frequency_walk_hz_per_root_s=1.5,
        white_pm_std_rad=5e-3,
        accuracy_ppb=2000.0,
    ),
}


# The repository's original hand-tuned values, kept as the default so
# every previously published number stays reproducible. Equivalent to
# sigma_y(1 s) = 3.5e-11 white FM with a 5.5e-11 flicker floor - between
# the ocxo and tcxo classes above.
LEGACY_PROFILE_NAME = "custom"


def resolve_oscillator_noise(
    profile_name: str,
    carrier_frequency_hz: float,
    sample_rate: float,
    sync_interval: float,
) -> tuple[dict[str, float], float | None]:
    """Return (noise-settings dict, expected initial CFO in Hz).

    For the legacy "custom" profile the dict is empty and the CFO is
    None, meaning: keep the SDRSimulationConfig defaults.
    """

    if profile_name == LEGACY_PROFILE_NAME:
        return {}, None
    profile = OSCILLATOR_PROFILES[profile_name]
    return (
        profile.noise_settings(carrier_frequency_hz, sample_rate, sync_interval),
        profile.expected_cfo_hz(carrier_frequency_hz),
    )
