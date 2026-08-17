"""Score the scheduling-policy family and the clutter-referenced-sync
family under five metrics, and see which rankings flip.

The five metrics, in plain words:

  probability of detection   fraction of trials in which the array
                             actually detects the drone at the coverage
                             edge (counted waveform pipeline, same as
                             every detection study in this repo).
  mean spectral efficiency   average communication rate to a user
                             terminal at the coverage edge, in bits per
                             second per hertz: log2(1 + received
                             signal-to-noise ratio), averaged over the
                             measured residual draws.
  95%-likely spectral eff.   the rate the user gets in at least 95% of
                             time slots (5th percentile of the same
                             draws) - the fairness/outage view used by
                             the cell-free literature (Qin et al.).
  array gain                 beam quality: |sum of station phasors|^2
                             normalized to the perfect N^2.
  net throughput             mean spectral efficiency times the
                             fraction of airtime NOT spent on
                             synchronization. Sync overhead steals
                             communication time; this is the metric
                             that prices that theft.

Link-budget convention for the user terminal: the repo's own
inter-station convention (20 dB single-station signal-to-noise ratio at
500 m, path-loss exponent 2.7), applied station-by-station to the
user's location. Stated shortcut: net throughput assumes every unit of
airtime not spent on sync carries data.

NOTE: a sibling workstream is building a shared metrics.py; it was not
on disk when this study was written, so the metric functions live here
inline. Reconcile later; definitions above are the contract.

Usage:
    .venv/bin/python multi_metric_study.py            # full (~20 min)
    .venv/bin/python multi_metric_study.py --quick    # 1 seed, small
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from clutter_sync_ofdm import run_piggyback_star
from detection import DetectionParams
from detection.viability import detection_range_m
from detection.waveform import run_waveform_detection
from ota_sync import SDRSimulationConfig
from ota_sync.network import place_stations
from ota_sync.scheduled import run_scheduled_star

REFERENCE_SNR_DB = 20.0  # single station at the reference distance
REFERENCE_DISTANCE_M = 500.0
PATH_LOSS_EXPONENT = 2.7


# ---------------------------------------------------------------------
# Metric implementations (inline; see NOTE in module docstring)
# ---------------------------------------------------------------------

def user_amplitudes(positions: np.ndarray, user_m: np.ndarray) -> torch.Tensor:
    """Per-station received-amplitude coefficients at the user, in
    units where the noise amplitude is 1: a_k = 10^(SNRref/20) *
    (d_k/500)^(-eta/2)."""

    distances = np.maximum(np.linalg.norm(positions - user_m, axis=1), 1.0)
    amplitude = 10.0 ** (REFERENCE_SNR_DB / 20.0) * (
        distances / REFERENCE_DISTANCE_M
    ) ** (-PATH_LOSS_EXPONENT / 2.0)
    return torch.tensor(amplitude, dtype=torch.float64)


def spectral_efficiency_draws(
    positions: np.ndarray,
    residual_matrix: torch.Tensor,
    users_m: np.ndarray,
) -> torch.Tensor:
    """Per-draw spectral efficiency (bits/s/Hz), averaged over the user
    locations, one value per residual-matrix column."""

    phasors = torch.exp(1j * residual_matrix.to(torch.complex128))
    values = torch.zeros(residual_matrix.shape[1], dtype=torch.float64)
    for user in np.atleast_2d(users_m):
        amps = user_amplitudes(positions, user).to(torch.complex128)
        field = torch.einsum("k,kt->t", amps, phasors)
        snr = torch.abs(field) ** 2
        values = values + torch.log2(1.0 + snr)
    return values / len(np.atleast_2d(users_m))


def summarize_se(draws: torch.Tensor) -> tuple[float, float]:
    """(mean spectral efficiency, 95%-likely spectral efficiency)."""

    return (
        torch.mean(draws).item(),
        torch.quantile(draws, 0.05).item(),
    )


def net_throughput(mean_se: float, sync_airtime: float) -> float:
    """Mean spectral efficiency times the airtime left after sync.
    Airtime above 100% means the sync demand does not physically fit
    the frame; net throughput is then zero (and the row is flagged)."""

    return max(0.0, 1.0 - sync_airtime) * mean_se


def star_residual_matrix(result) -> torch.Tensor:
    """(stations, samples) with row 0 the reference - steady window or
    tail quarter fallback, same convention as contention_study."""

    matrix = result.residual_matrix()
    if matrix.shape[1] > 0:
        return matrix
    intervals = result.residuals.shape[1]
    tail = slice(max(0, intervals - max(1, intervals // 4)), intervals)
    rows = [torch.zeros(tail.stop - tail.start, dtype=torch.float64)]
    for row in result.residuals:
        rows.append(row[tail])
    return torch.stack(rows)


def piggyback_residual_matrix(result) -> torch.Tensor:
    """(stations, samples) for the piggyback star: columns where every
    station has a valid (post-acquisition) residual."""

    valid = result.all_valid
    rows = [torch.zeros(int(valid.sum().item()), dtype=torch.float64)]
    for row in result.station_residuals:
        rows.append(row[valid].to(torch.float64))
    return torch.stack(rows)


def mean_gain_from_matrix(matrix: torch.Tensor) -> float:
    phasors = torch.exp(1j * matrix.to(torch.complex128))
    gain = torch.abs(torch.sum(phasors, dim=0)) ** 2 / matrix.shape[0] ** 2
    return torch.mean(gain.real).item()


# ---------------------------------------------------------------------
# Scoring one configuration
# ---------------------------------------------------------------------

def edge_targets(positions: np.ndarray) -> np.ndarray:
    centroid = positions.mean(axis=0)
    return np.array(
        [centroid + [1200.0, 150.0], centroid + [-1200.0, 150.0]]
    )


def score(
    label: str,
    per_seed: list[tuple[np.ndarray, torch.Tensor]],  # (positions, matrix)
    sync_airtime: float,
    params: DetectionParams,
    trials: int,
    h0_trials: int,
) -> dict:
    """All five metrics. Detection is counted on the first seed's
    deployment (the convention every detection study here uses);
    spectral efficiency and gain pool all seeds."""

    se_all = []
    gains = []
    for positions, matrix in per_seed:
        se_all.append(
            spectral_efficiency_draws(
                positions, matrix, edge_targets(positions)
            )
        )
        gains.append(mean_gain_from_matrix(matrix))
    draws = torch.cat(se_all)
    mean_se, likely_se = summarize_se(draws)
    gain = float(np.mean(gains))
    num_stations = per_seed[0][1].shape[0]

    positions0, matrix0 = per_seed[0]
    detect = run_waveform_detection(
        label,
        positions0,
        matrix0,
        edge_targets(positions0),
        params=params,
        trials=trials,
        h0_trials=h0_trials,
        seed=0,
    )
    pd_mean = float(np.mean(detect.pd_measured))

    return {
        "label": label,
        "pd": pd_mean,
        "mean_se": mean_se,
        "likely_se": likely_se,
        "gain": gain,
        "airtime": sync_airtime,
        "net": net_throughput(mean_se, sync_airtime),
        "range_m": detection_range_m(num_stations, gain, params),
        "overfit": sync_airtime > 1.0,
    }


def print_table(rows: list[dict], title: str) -> None:
    print(f"\n=== {title} ===")
    print(
        f"  {'method':<26} {'detect%':>8} {'rate':>6} {'95%rate':>8} "
        f"{'beam%':>6} {'sync-air%':>9} {'net-rate':>8} {'range m':>8}"
    )
    for r in rows:
        flag = "*" if r["overfit"] else " "
        print(
            f"  {r['label']:<26} {100 * r['pd']:8.1f} {r['mean_se']:6.2f} "
            f"{r['likely_se']:8.2f} {100 * r['gain']:6.1f} "
            f"{100 * r['airtime']:8.1f}{flag} {r['net']:8.2f} "
            f"{r['range_m']:8.0f}"
        )


def best_per_metric(rows: list[dict]) -> None:
    print("\n  best method per metric:")
    for key, name in (
        ("pd", "probability of detection"),
        ("mean_se", "mean spectral efficiency"),
        ("likely_se", "95%-likely spectral efficiency"),
        ("gain", "array gain (beam quality)"),
        ("net", "net throughput"),
        ("range_m", "detection range"),
    ):
        top = max(rows, key=lambda r: r[key])
        print(f"    {name:<32} -> {top['label']}")


# ---------------------------------------------------------------------
# The study
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="five-metric comparison: scheduling and clutter families"
    )
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--h0-trials", type=int, default=12000)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    seeds = [0] if args.quick else [int(s) for s in args.seeds.split(",")]
    iterations = 20 if args.quick else args.iterations
    trials = 100 if args.quick else args.trials
    h0 = 4000 if args.quick else args.h0_trials
    params = DetectionParams(tx_power_w=0.5)

    print(
        "Five-metric comparison (detection counted on seed "
        f"{seeds[0]}'s deployment; rate metrics pool seeds {seeds}; "
        f"{iterations} intervals)"
    )
    print(
        "rate = mean spectral efficiency, bits/s/Hz at the coverage-edge "
        "user; net-rate = rate x (1 - sync airtime); * = sync demand "
        "exceeds the frame (not realizable)"
    )

    # ---- family 1: scheduling policies, N=10 -----------------------
    rows1 = []
    for capacity in (2, 4, 8):
        for policy in ("uniform", "roundrobin", "scheduled", "oracle"):
            per_seed = []
            airtimes = []
            for seed in seeds:
                settings = SDRSimulationConfig(
                    num_iterations=iterations, seed=seed, device="cpu"
                )
                result = run_scheduled_star(
                    settings, num_stations=10, policy=policy,
                    max_exchanges_per_interval=capacity,
                )
                per_seed.append(
                    (result.positions, star_residual_matrix(result))
                )
                airtimes.append(result.airtime_used_fraction)
            rows1.append(
                score(
                    f"{policy} @cap{capacity}", per_seed,
                    float(np.mean(airtimes)), params, trials, h0,
                )
            )
            print(f"  scored {rows1[-1]['label']}")
    # cheap-pilot reference: micro pilots under the scheduled policy
    per_seed = []
    airtimes = []
    for seed in seeds:
        settings = SDRSimulationConfig(
            num_iterations=iterations, seed=seed, device="cpu"
        )
        result = run_scheduled_star(
            settings, num_stations=10, policy="scheduled",
            max_exchanges_per_interval=4, multi_fidelity=True,
        )
        per_seed.append((result.positions, star_residual_matrix(result)))
        airtimes.append(result.airtime_used_fraction)
    rows1.append(
        score(
            "scheduled+micro @cap4", per_seed, float(np.mean(airtimes)),
            params, trials, h0,
        )
    )
    print_table(rows1, "scheduling policies, 10-station array")
    best_per_metric(rows1)

    # ---- family 2: how the array pays for sync, N=6 ----------------
    rows2 = []
    for label, runner in (
        (
            "two-way scheduled",
            lambda s: run_scheduled_star(s, num_stations=6, policy="scheduled"),
        ),
        (
            "micro-pilot star",
            lambda s: run_scheduled_star(
                s, num_stations=6, policy="scheduled", multi_fidelity=True
            ),
        ),
    ):
        per_seed = []
        airtimes = []
        for seed in seeds:
            settings = SDRSimulationConfig(
                num_iterations=iterations, seed=seed, device="cpu"
            )
            result = runner(settings)
            per_seed.append((result.positions, star_residual_matrix(result)))
            airtimes.append(result.airtime_used_fraction)
        rows2.append(
            score(label, per_seed, float(np.mean(airtimes)), params, trials, h0)
        )
        print(f"  scored {label}")
    for cadence in (5, 40):
        per_seed = []
        airtimes = []
        for seed in seeds:
            settings = SDRSimulationConfig(
                num_iterations=iterations, seed=seed, device="cpu"
            )
            result = run_piggyback_star(
                settings, num_stations=6,
                anchor_every_intervals=cadence, waveform="ofdm",
            )
            positions = place_stations(6, 500.0, seed)
            per_seed.append((positions, piggyback_residual_matrix(result)))
            airtimes.append(result.piggyback_airtime)
        rows2.append(
            score(
                f"piggyback-clutter K={cadence}", per_seed,
                float(np.mean(airtimes)), params, trials, h0,
            )
        )
        print(f"  scored piggyback K={cadence}")
    print_table(rows2, "paying for sync, 6-station array")
    best_per_metric(rows2)

    # ---- combined view ---------------------------------------------
    print_table(rows1 + rows2, "combined (note: families differ in array size)")
    print(
        "\n(10-station and 6-station rows are not directly comparable on "
        "absolute numbers; compare within a family, and compare the "
        "piggyback rows against their own two-way/micro baselines)"
    )


if __name__ == "__main__":
    main()
