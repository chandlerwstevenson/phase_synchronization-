"""Ex-ante per-station coast-time law for scheduled OTA phase sync.

The claim under test: how long station k can coast between sync pilots
is predictable BEFORE any simulation (and, in deployment, before any
airtime is spent), from datasheet oscillator anchors, the link budget,
and the sync frame design alone - no fitted constants. The coast time
tau_k then defines each station's sync demand T/tau_k, and the array's
operating point collapses to one dimensionless supply/demand ratio

    rho = capacity / sum_k (T / tau_k)

which downstream studies use as the control parameter.

Three predictor layers, all strictly ex ante:

  closed form   solve  a*tau + (sigma_w*(tau + L*T))^2 + c0 = B^2
                for tau, with a the pair's white-FM phase-variance rate
                (datasheet ADEV -> sigma_pn, a = 0.5*(s_ref^2+s_slv^2)*fs),
                sigma_w the steady-state Kalman frequency-posterior std,
                c0 the posterior phase variance. This is the error-floor
                formula rearranged (METHOD_CALCULATIONS.md result 1).
  DARE          sigma_w and c0 from the every-interval Riccati fixed
                point of the SAME (F, Q, R) matrices run_scheduled_star
                builds for the link - reconstructed here from the
                oscillator profile and link SNR without running anything.
  cycle         the every-interval DARE understates the posterior of a
                filter that coasts: at steady state the filter runs
                m predicts then ONE update. The cycle fixed point solves
                P = update(predict^m(P)), m = first crossing of the
                trigger threshold, self-consistently. This is the
                discrete predictor a scheduler would actually deploy,
                and it prices the m^3/3 frequency-random-walk growth and
                the P01 cross term the closed form truncates away.

Threshold semantics: run_scheduled_star's "scheduled" policy services a
link when its POST-PREDICT phase std crosses trigger_fraction * budget;
the trigger does not look ahead through the correction latency. So the
CADENCE predictor uses threshold B = trigger_fraction * budget with
L = 0, while the L-inclusive closed form is the SAFE-coast rule (keeps
the residual under budget through the late correction landing) and is
reported separately. Validation measures both cadence (service gaps)
and physical calibration (true residual at service time vs threshold).

Measurement-noise honesty: detection work in this repository
(doppler_coast_study.py) measured ~100 mrad rms of white per-exchange
multipath-resampling noise in the two-way half-difference - an order of
magnitude above the modeled measurement covariance's phase entry at
default SNR. The predictor exposes `extra_phase_measurement_var` plus
an ex-ante estimate from the TDL model's Rice factor
(`resampling_phase_variance`), and the CLI reports the grid with and
without it. Nothing is fitted to residuals.

Usage:
    .venv/bin/python coast_law.py                # full validation grid
    .venv/bin/python coast_law.py --quick        # 1 seed, 2 budgets
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace

import numpy as np
import torch

from ota_sync import SDRSimulationConfig
from ota_sync.core import REAL_DTYPE
from ota_sync.network import MAX_LINK_SNR_DB, place_stations
from ota_sync.oscillators import resolve_oscillator_noise
from ota_sync.sdr import (
    _FlickerFrequencyNoise,
    _measurement_covariance,
    make_sync_preamble,
)
import ota_sync.scheduled as scheduled_module

_DEVICE = torch.device("cpu")

# 3GPP TR 38.901 Table 7.7.2-4: TDL-D Rice factor (dB) of the
# LOS-dominated profile the simulator instantiates. Used ONLY for the
# ex-ante resampling-noise estimate; not fitted.
TDL_D_RICE_FACTOR_DB = 13.3


# ---------------------------------------------------------------------
# Ex-ante reconstruction of the link's filter matrices
# ---------------------------------------------------------------------

def profiled_settings(
    settings: SDRSimulationConfig, profile: str, snr_db: float
) -> SDRSimulationConfig:
    """The per-link SDRSimulationConfig run_scheduled_star would build
    for a station of this oscillator class at this link SNR."""

    noise, _ = resolve_oscillator_noise(
        profile,
        settings.carrier_frequency_hz,
        settings.sample_rate,
        settings.sync_interval,
    )
    return replace(settings, snr_db=min(snr_db, MAX_LINK_SNR_DB), **noise)


def station_snr_db(
    settings: SDRSimulationConfig,
    num_stations: int,
    station: int,
    radius_m: float = 500.0,
    path_loss_exponent: float = 2.7,
    reference_distance_m: float = 500.0,
) -> float:
    """Replicates run_scheduled_star's per-station link budget (the
    deployment geometry is known ex ante)."""

    positions = place_stations(num_stations, radius_m, settings.seed)
    distance = max(
        float(np.linalg.norm(positions[station] - positions[0])), 1.0
    )
    return min(
        settings.snr_db
        - 10.0 * path_loss_exponent
        * math.log10(distance / reference_distance_m),
        MAX_LINK_SNR_DB,
    )


@dataclass(frozen=True)
class LinkMatrices:
    """(F, Q, R_eff) of one link's EKF plus the drift-rate ingredients."""

    transition: torch.Tensor  # 2x2
    process: torch.Tensor  # 2x2
    measurement: torch.Tensor  # 2x2 effective (phase, angular frequency)
    white_fm_rate: float  # rad^2 per second (the closed form's `a`)


def link_matrices(
    settings: SDRSimulationConfig,
    profile: str,
    snr_db: float,
    horizon_s: float,
    reference_profile: str | None = None,
    extra_phase_measurement_var: float = 0.0,
) -> LinkMatrices:
    """Reconstruct the exact (F, Q, R) run_scheduled_star gives this
    link's EKF - from datasheet anchors and the link budget only.

    The construction mirrors ota_sync/scheduled.py lines 263-409:
    process = reference oscillator covariance + slave oscillator
    covariance + diag(white-FM capture walk, flicker innovation);
    measurement = 0.5 * two-way covariance. The effective 2x2
    measurement covariance takes the (sin, frequency) rows of the
    3-entry [cos, sin, omega] form - the linearization at zero phase
    error, which is where a locked loop lives.
    """

    if reference_profile is None:
        reference_profile = profile
    ref = profiled_settings(settings, reference_profile, settings.snr_db)
    slave = profiled_settings(settings, profile, snr_db)

    def oscillator_covariance(cfg: SDRSimulationConfig) -> torch.Tensor:
        return torch.diag(
            torch.tensor(
                [
                    cfg.phase_process_std_rad**2,
                    (2.0 * math.pi * cfg.frequency_process_std_hz) ** 2,
                ],
                dtype=REAL_DTYPE,
                device=_DEVICE,
            )
        )

    interval_samples = int(
        round(settings.sync_interval * settings.sample_rate)
    )
    white_fm_phase_variance = (
        0.5
        * (slave.phase_noise_std_rad**2 + ref.phase_noise_std_rad**2)
        * interval_samples
    )
    # innovation_variance is a deterministic function of (std, interval,
    # horizon); the generator seeds only the state draw, which we do not
    # use.
    flicker = _FlickerFrequencyNoise(
        ref.flicker_frequency_std_hz,
        settings.sync_interval,
        horizon_s,
        _DEVICE,
        torch.Generator(device=_DEVICE).manual_seed(0),
    )
    process = (
        oscillator_covariance(ref)
        + oscillator_covariance(slave)
        + torch.diag(
            torch.tensor(
                [white_fm_phase_variance, flicker.innovation_variance],
                dtype=REAL_DTYPE,
                device=_DEVICE,
            )
        )
    )
    preamble = make_sync_preamble(slave, _DEVICE)
    full = 0.5 * _measurement_covariance(slave, preamble, _DEVICE)
    measurement = torch.diag(
        torch.stack(
            (full[1, 1] + extra_phase_measurement_var, full[2, 2])
        )
    )
    transition = torch.tensor(
        [[1.0, settings.sync_interval], [0.0, 1.0]],
        dtype=REAL_DTYPE,
        device=_DEVICE,
    )
    white_fm_rate = (
        0.5
        * (slave.phase_noise_std_rad**2 + ref.phase_noise_std_rad**2)
        * settings.sample_rate
    )
    return LinkMatrices(transition, process, measurement, white_fm_rate)


def resampling_phase_variance(
    rice_factor_db: float = TDL_D_RICE_FACTOR_DB,
) -> float:
    """Ex-ante estimate of the per-exchange multipath-resampling phase
    variance in the two-way half-difference.

    A Ricean composite of K-factor K has small-signal phase spread
    var ~ 1/(2*K) around the LOS phase. Independent timing jitter in
    the forward and reverse captures re-samples the diffuse part
    independently, and the half-difference averages the two legs, so
    the leaked variance is ~ 0.5 * (2 * 1/(2K)) = 1/(2K). For TDL-D
    (K = 13.3 dB) this gives ~(153 mrad)^2 - the right order against
    the ~100 mrad rms measured in doppler_coast_study.py. An estimate
    from the channel spec, not a fit.
    """

    k_linear = 10.0 ** (rice_factor_db / 10.0)
    return 1.0 / (2.0 * k_linear)


# ---------------------------------------------------------------------
# Riccati fixed points
# ---------------------------------------------------------------------

def _update(P: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    gain = torch.linalg.solve((P + R).T, P.T).T
    eye = torch.eye(2, dtype=REAL_DTYPE, device=_DEVICE)
    residual_map = eye - gain
    return residual_map @ P @ residual_map.T + gain @ R @ gain.T


def dare_posterior(
    matrices: LinkMatrices,
    iterations: int = 4000,
    tolerance: float = 1e-14,
) -> torch.Tensor:
    """Every-interval-update posterior covariance fixed point."""

    F, Q, R = matrices.transition, matrices.process, matrices.measurement
    P = torch.diag(
        torch.tensor(
            [math.pi**2, (2.0 * math.pi * 50e3) ** 2],
            dtype=REAL_DTYPE,
            device=_DEVICE,
        )
    )
    for _ in range(iterations):
        predicted = F @ P @ F.T + Q
        posterior = _update(predicted, R)
        if torch.max(torch.abs(posterior - P)).item() < tolerance:
            return posterior
        P = posterior
    return P


def _coast_crossing(
    posterior: torch.Tensor,
    matrices: LinkMatrices,
    threshold_rad: float,
    max_intervals: int,
) -> int:
    """First m >= 1 with sqrt(P00) >= threshold after m predicts - the
    scheduler's own trigger semantics."""

    F, Q = matrices.transition, matrices.process
    P = posterior
    for m in range(1, max_intervals + 1):
        P = F @ P @ F.T + Q
        if math.sqrt(max(P[0, 0].item(), 0.0)) >= threshold_rad:
            return m
    return max_intervals + 1  # censored: never crosses in horizon


def cycle_posterior(
    matrices: LinkMatrices,
    threshold_rad: float,
    max_intervals: int = 100000,
    max_outer: int = 200,
) -> tuple[torch.Tensor, int]:
    """Self-consistent (posterior, coast length) of a filter that
    coasts to the trigger and takes ONE update: P = upd(pred^m(P)),
    m = crossing(P). Starts from the every-interval DARE point."""

    F, Q, R = matrices.transition, matrices.process, matrices.measurement
    P = dare_posterior(matrices)
    previous_m = -1
    for _ in range(max_outer):
        m = _coast_crossing(P, matrices, threshold_rad, max_intervals)
        if m > max_intervals:
            return P, m  # censored at this posterior
        predicted = P
        for _ in range(m):
            predicted = F @ predicted @ F.T + Q
        P = _update(predicted, R)
        if m == previous_m:
            return P, m
        previous_m = m
    return P, m


# ---------------------------------------------------------------------
# The predictor
# ---------------------------------------------------------------------

def predict_coast_time(
    oscillator_profile: str,
    link_snr_db: float,
    latency_intervals: int,
    sync_interval: float,
    budget_rad: float,
    settings: SDRSimulationConfig = SDRSimulationConfig(),
    trigger_fraction: float = 1.0,
    horizon_s: float | None = None,
    extra_phase_measurement_var: float = 0.0,
    mode: str = "cycle",
    include_latency: bool = False,
) -> float:
    """Ex-ante coast time tau (seconds) for one station.

    mode="cycle": discrete self-consistent predictor (deployable rule,
    matches run_scheduled_star's trigger). mode="closed": the error-
    floor closed form a*tau + (sigma_w*(tau+L*T))^2 + c0 = B^2 with
    sigma_w, c0 from the cycle posterior; include_latency=False drops
    L from the horizon (cadence semantics - the scheduler's trigger
    does not look through the latency), True keeps it (safe-coast
    rule). B = trigger_fraction * budget_rad throughout.
    """

    if sync_interval != settings.sync_interval:
        settings = replace(settings, sync_interval=sync_interval)
    if latency_intervals != settings.correction_latency_intervals:
        settings = replace(
            settings, correction_latency_intervals=latency_intervals
        )
    if horizon_s is None:
        horizon_s = settings.num_iterations * settings.sync_interval
    matrices = link_matrices(
        settings,
        oscillator_profile,
        link_snr_db,
        horizon_s,
        extra_phase_measurement_var=extra_phase_measurement_var,
    )
    threshold = trigger_fraction * budget_rad
    posterior, m_cycle = cycle_posterior(matrices, threshold)
    if mode == "cycle":
        return m_cycle * settings.sync_interval
    if mode != "closed":
        raise ValueError("mode must be 'cycle' or 'closed'")

    a = matrices.white_fm_rate
    sigma_w = math.sqrt(max(posterior[1, 1].item(), 0.0))
    c0 = max(posterior[0, 0].item(), 0.0)
    horizon_offset = (
        latency_intervals * settings.sync_interval if include_latency else 0.0
    )
    # sigma_w^2*(tau+h)^2 + a*tau + c0 - B^2 = 0
    A = sigma_w**2
    Bq = 2.0 * A * horizon_offset + a
    Cq = A * horizon_offset**2 + c0 - threshold**2
    if Cq >= 0.0:
        return 0.0  # already over budget before coasting at all
    if A == 0.0:
        return -Cq / Bq
    tau = (-Bq + math.sqrt(Bq**2 - 4.0 * A * Cq)) / (2.0 * A)
    return max(tau, 0.0)


def supply_demand_ratio(
    fleet_profiles: list[str],
    snrs_db: list[float],
    capacity_exchanges_per_interval: float,
    latency_intervals: int,
    sync_interval: float,
    budget_rad: float,
    settings: SDRSimulationConfig = SDRSimulationConfig(),
    trigger_fraction: float = 1.0,
    extra_phase_measurement_var: float = 0.0,
) -> float:
    """rho = capacity / sum_k (T / tau_k), tau_k from the cycle
    predictor. fleet_profiles/snrs cover the NON-reference stations
    (one entry each); capacity is in two-way exchanges per sync
    interval, matching run_scheduled_star's max_exchanges_per_interval.
    rho > 1: the channel can carry the fleet's steady-state sync
    demand; rho < 1: it cannot, and stations must go stale.
    """

    if len(fleet_profiles) != len(snrs_db):
        raise ValueError("one SNR per fleet station required")
    demand = 0.0
    for profile, snr in zip(fleet_profiles, snrs_db):
        tau = predict_coast_time(
            profile,
            snr,
            latency_intervals,
            sync_interval,
            budget_rad,
            settings=settings,
            trigger_fraction=trigger_fraction,
            extra_phase_measurement_var=extra_phase_measurement_var,
            mode="cycle",
        )
        demand += sync_interval / max(tau, 1e-12)
    return capacity_exchanges_per_interval / demand


# ---------------------------------------------------------------------
# Validation against the running scheduler
# ---------------------------------------------------------------------

SETTLING_SERVICES = 8  # forced acquisition/settling services to drop


def measured_coast_gaps(result, station_row: int) -> list[int]:
    """Service-to-service gaps (intervals) after settling."""

    serviced = torch.nonzero(result.serviced[station_row]).flatten()
    if serviced.numel() <= SETTLING_SERVICES + 1:
        return []
    kept = serviced[SETTLING_SERVICES:]
    return torch.diff(kept).tolist()


def residuals_at_service(result, station_row: int) -> list[float]:
    serviced = torch.nonzero(result.serviced[station_row]).flatten()
    kept = serviced[SETTLING_SERVICES:]
    return [
        abs(result.residuals[station_row, t].item()) for t in kept
    ]


def run_validation_cell(
    profile: str,
    budget: float,
    latency: int,
    seed: int,
    num_stations: int = 6,
    iterations: int = 150,
    extra_phase_measurement_var: float = 0.0,
):
    settings = SDRSimulationConfig(
        num_iterations=iterations,
        seed=seed,
        device="cpu",
        correction_latency_intervals=latency,
    )
    result = scheduled_module.run_scheduled_star(
        settings,
        num_stations=num_stations,
        policy="scheduled",
        trigger_fraction=1.0,
        budgets_rad=[budget] * (num_stations - 1),
        max_exchanges_per_interval=num_stations - 1,
        oscillator_profiles=[profile] * num_stations,
    )
    horizon_s = iterations * settings.sync_interval
    rows = []
    for station in range(1, num_stations):
        snr = station_snr_db(settings, num_stations, station)
        shared = dict(
            oscillator_profile=profile,
            link_snr_db=snr,
            latency_intervals=latency,
            sync_interval=settings.sync_interval,
            budget_rad=budget,
            settings=settings,
            trigger_fraction=1.0,
            horizon_s=horizon_s,
            extra_phase_measurement_var=extra_phase_measurement_var,
        )
        predicted_cycle = predict_coast_time(**shared, mode="cycle")
        predicted_closed = predict_coast_time(
            **shared, mode="closed", include_latency=False
        )
        predicted_safe = predict_coast_time(
            **shared, mode="closed", include_latency=True
        )
        gaps = measured_coast_gaps(result, station - 1)
        service_residuals = residuals_at_service(result, station - 1)
        rows.append(
            {
                "station": station,
                "snr_db": snr,
                "predicted_cycle_intervals":
                    predicted_cycle / settings.sync_interval,
                "predicted_closed_intervals":
                    predicted_closed / settings.sync_interval,
                "predicted_safe_intervals":
                    predicted_safe / settings.sync_interval,
                "measured_gaps": gaps,
                "service_residuals": service_residuals,
            }
        )
    return result, rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="validate the ex-ante coast-time predictor"
    )
    parser.add_argument("--classes", type=str, default="ocxo,tcxo,sdr")
    parser.add_argument("--budgets", type=str, default="0.2,0.314,0.6")
    parser.add_argument("--latencies", type=str, default="1,2,4")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--stations", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument(
        "--resampling-noise", action="store_true",
        help="add the ex-ante TDL-D resampling phase variance to the "
        "predictor's measurement covariance (predictor side only; the "
        "running filter is untouched)",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    classes = args.classes.split(",")
    budgets = [float(b) for b in args.budgets.split(",")]
    latencies = [int(v) for v in args.latencies.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.quick:
        budgets, latencies, seeds = budgets[:2], latencies[:1], seeds[:1]
    extra = (
        resampling_phase_variance() if args.resampling_noise else 0.0
    )

    print(
        f"Coast-law validation: N={args.stations} star, scheduled "
        f"policy, trigger=1.0, {args.iterations} intervals, "
        f"classes {classes}, budgets {budgets}, latencies {latencies}, "
        f"seeds {seeds}"
        + (
            f", predictor +resampling var ({1e3 * math.sqrt(extra):.0f} "
            "mrad)" if extra else ""
        )
    )
    print(
        f"\n{'class':<6}{'budget':>7}{'L':>3} | "
        f"{'pred cyc':>9}{'pred clsd':>10} | "
        f"{'measured med':>13}{'n':>5} | "
        f"{'ratio med':>10}{'IQR':>13} | {'resid@svc':>10}{'/thr':>6}"
    )
    all_ratios: list[float] = []
    ratios_by_latency: dict[int, list[float]] = {v: [] for v in latencies}
    ratios_by_class: dict[str, list[float]] = {c: [] for c in classes}
    ratios_by_budget: dict[float, list[float]] = {b: [] for b in budgets}
    exact_matches = 0
    total_gaps = 0
    for profile in classes:
        # ocxo coasts for tens of intervals; give it enough horizon to
        # complete several coast cycles past settling.
        class_iterations = (
            max(args.iterations, 600) if profile == "ocxo"
            else args.iterations
        )
        for budget in budgets:
            for latency in latencies:
                cell_ratios: list[float] = []
                cell_gaps: list[float] = []
                cell_pred_cycle: list[float] = []
                cell_pred_closed: list[float] = []
                cell_residuals: list[float] = []
                censored = 0
                for seed in seeds:
                    _, rows = run_validation_cell(
                        profile, budget, latency, seed,
                        num_stations=args.stations,
                        iterations=class_iterations,
                        extra_phase_measurement_var=extra,
                    )
                    for row in rows:
                        cell_pred_cycle.append(
                            row["predicted_cycle_intervals"]
                        )
                        cell_pred_closed.append(
                            row["predicted_closed_intervals"]
                        )
                        cell_residuals.extend(row["service_residuals"])
                        if not row["measured_gaps"]:
                            censored += 1
                            continue
                        for gap in row["measured_gaps"]:
                            cell_gaps.append(gap)
                            cell_ratios.append(
                                gap / row["predicted_cycle_intervals"]
                            )
                            total_gaps += 1
                            if gap == round(
                                row["predicted_cycle_intervals"]
                            ):
                                exact_matches += 1
                pred_cycle = float(np.median(cell_pred_cycle))
                pred_closed = float(np.median(cell_pred_closed))
                if cell_ratios:
                    ratio_median = float(np.median(cell_ratios))
                    q1, q3 = np.percentile(cell_ratios, [25, 75])
                    gap_median = float(np.median(cell_gaps))
                    all_ratios.extend(cell_ratios)
                    ratios_by_latency[latency].extend(cell_ratios)
                    ratios_by_class[profile].extend(cell_ratios)
                    ratios_by_budget[budget].extend(cell_ratios)
                    ratio_text = (
                        f"{ratio_median:10.2f}"
                        f"  [{q1:4.2f},{q3:4.2f}]"
                    )
                    gap_text = f"{gap_median:13.1f}{len(cell_gaps):5d}"
                else:
                    ratio_text = f"{'censored':>10}{'':>13}"
                    gap_text = f"{'-':>13}{0:5d}"
                resid = (
                    float(
                        np.sqrt(np.mean(np.square(cell_residuals)))
                    )
                    if cell_residuals
                    else float("nan")
                )
                print(
                    f"{profile:<6}{budget:7.3f}{latency:3d} | "
                    f"{pred_cycle:9.1f}{pred_closed:10.1f} | "
                    f"{gap_text} | {ratio_text} | "
                    f"{1e3 * resid:8.0f}mr{resid / budget:6.2f}"
                    + (f"  ({censored} censored)" if censored else "")
                )

    if all_ratios:
        print(
            f"\nOVERALL measured/predicted (cycle): median "
            f"{float(np.median(all_ratios)):.2f}, IQR "
            f"[{np.percentile(all_ratios, 25):.2f}, "
            f"{np.percentile(all_ratios, 75):.2f}], n={len(all_ratios)}"
        )
        print(
            f"  gaps matching the cycle prediction EXACTLY: "
            f"{exact_matches}/{total_gaps} "
            f"({100.0 * exact_matches / total_gaps:.1f}%)"
        )
        for name, groups in (
            ("class", ratios_by_class),
            ("budget", ratios_by_budget),
            ("latency", ratios_by_latency),
        ):
            parts = [
                f"{key}: {float(np.median(vals)):.2f} (n={len(vals)})"
                for key, vals in groups.items()
                if vals
            ]
            print(f"  by {name}:  " + "   ".join(parts))


if __name__ == "__main__":
    main()
