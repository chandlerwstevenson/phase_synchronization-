"""The pilot-shortening lever, measured (SNR_LAW.md's one unrun
experiment).

Claim under test: at fixed SNR x pilot-length product, the residual
holds constant while sync airtime falls in proportion to the pilot
length - i.e. every 3 dB of link budget can halve the sync airtime.
Sweep: (SNR, long-sequence length) pairs sharing one product, anchored
at the default (20 dB, 2047). Control rows: same lengths at FIXED
20 dB SNR (product falling), to show any residual growth is the
product's doing, not the length's.

N=2 star, uniform policy (serviced every interval), 60 intervals,
seeds 0-2. Airtime is the star's own accounting.
"""

from __future__ import annotations

import math

from ota_sync import SDRSimulationConfig
from ota_sync.scheduled import run_scheduled_star

SEEDS = [0, 1, 2]
BASE_SNR = 20.0
BASE_LENGTH = 2047
# (snr_db, long_sequence_length): halve length per +3 dB
LEVER = [
    (14.0, 8191),
    (17.0, 4095),
    (20.0, 2047),
    (23.0, 1023),
    (26.0, 511),
    (29.0, 255),
    (32.0, 127),
]


def run_point(snr_db: float, length: int) -> tuple[float, float]:
    residuals, airtimes = [], []
    cp = min(128, max(8, length // 4))
    for seed in SEEDS:
        settings = SDRSimulationConfig(
            num_iterations=60,
            seed=seed,
            device="cpu",
            snr_db=snr_db,
            long_sequence_length=length,
            long_cp_length=cp,
        )
        result = run_scheduled_star(settings, num_stations=2, policy="uniform")
        rms = result.station_steady_rms[0]
        if rms == rms:
            residuals.append(1e3 * rms)
        airtimes.append(100.0 * result.airtime_used_fraction)
    mean_r = sum(residuals) / len(residuals) if residuals else float("nan")
    std_r = (
        math.sqrt(sum((r - mean_r) ** 2 for r in residuals) / len(residuals))
        if len(residuals) > 1
        else 0.0
    )
    return mean_r, std_r, sum(airtimes) / len(airtimes)


def main() -> None:
    print("Lever sweep: fixed SNR x length product "
          f"(anchor {BASE_SNR:.0f} dB x {BASE_LENGTH})")
    print(f"{'SNR dB':>7} {'length':>7} {'residual mrad':>16} "
          f"{'airtime %':>10}")
    for snr, length in LEVER:
        mean_r, std_r, air = run_point(snr, length)
        print(f"{snr:>7.0f} {length:>7} {mean_r:>10.1f}±{std_r:<5.1f} "
              f"{air:>9.2f}")

    print(f"\nControl: fixed {BASE_SNR:.0f} dB SNR, falling product")
    print(f"{'SNR dB':>7} {'length':>7} {'residual mrad':>16} "
          f"{'airtime %':>10}")
    for _, length in LEVER:
        mean_r, std_r, air = run_point(BASE_SNR, length)
        print(f"{BASE_SNR:>7.0f} {length:>7} {mean_r:>10.1f}±{std_r:<5.1f} "
              f"{air:>9.2f}")


if __name__ == "__main__":
    main()
