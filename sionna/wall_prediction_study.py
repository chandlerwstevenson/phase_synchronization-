"""Blind ex-ante prediction of the sync airtime wall.

The claim under falsification-style test: where an array's coherence
collapses under sync contention is computable BEFORE running anything,
from the coast-time formula alone - per-link demand rate 1/tau_k with
tau_k the coast time out of the filter's own model, no fitted
constants. Three distinct quantitative predictions, each printed
before its measurement runs:

  P1  critical capacity C* = sum_k T/tau_k per fleet (all-sdr,
      all-tcxo, mixed) at N=10: the measured >90%-gain knee of the
      scheduled policy must land at ceil(C*).
  P2  the wall shift at fixed capacity 3 (sdr fleet): predicted
      largest sustainable N per policy - uniform demands (N-1)
      exchanges/interval so N*_uni = capacity+1; scheduled demands
      sum_k 1/m_k so N*_sched = max N with demand <= capacity.
  P3  heterogeneous dividend: demand D(p) for OCXO fleets with a
      TCXO fraction p among the slaves - a specific nonlinear curve
      in p - and hence the minimum capacity per p.

Ex-ante decisions, fixed BEFORE any measurement and never revisited
(rules of engagement - misses are reported as misses):

  * Demand is priced at the TRIGGER line, sigma = trigger_fraction *
    budget (default 0.5 * 0.314 rad), not at the budget: the deployed
    scheduler in ota_sync/scheduled.py services a link when its
    predicted std crosses the trigger, so the channel must carry the
    trigger-rate. tau(budget) would describe an idealized scheduler
    that is not the code under test.
  * The primary tau_k is the EXACT coast-step count of the EKF's own
    covariance recursion (steady post-update covariance from the
    Riccati iteration, then predict-only steps until the trigger) -
    every ingredient is constructed ex ante from oscillator profiles
    and the link-budget geometry exactly as run_scheduled_star builds
    its filters; nothing is fitted to data. The closed-form
    sigma_pn^2*fs*tau + (sigma_omega*(tau+L*T))^2 = target^2 is
    reported alongside as the interpretable approximation.
  * Knee criterion: first capacity whose 3-seed mean effective gain
    (steady mean, else tail-quarter mean) is >= 90%.
  * Measurements use 100 intervals: at capacity 1-2 the scheduler
    serializes acquisition across ~9 links, and 50 intervals would
    confound the knee with acquisition transients.
  * Measurement windows are centered on the predicted knee (that is
    design economy, not peeking - the prediction is already fixed);
    if the window's top capacity still fails 90%, it auto-extends
    upward so an unbracketed knee is reported as a miss, not hidden.

Nothing in ota_sync/ is modified; the demand model REBUILDS the same
matrices run_scheduled_star constructs, by importing the same
helpers.

Usage:
    .venv/bin/python wall_prediction_study.py --part predict   # no sims
    .venv/bin/python wall_prediction_study.py --part 1|2|3
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from ota_sync import SDRSimulationConfig
from ota_sync.network import MAX_LINK_SNR_DB, place_stations
from ota_sync.oscillators import resolve_oscillator_noise
from ota_sync.scheduled import run_scheduled_star
from ota_sync.sdr import (
    _FlickerFrequencyNoise,
    _measurement_covariance,
    make_sync_preamble,
)

BUDGET_RAD = 0.314
TRIGGER_FRACTION = 0.5  # run_scheduled_star default
GAIN_KNEE = 0.90
MEASURE_INTERVALS = 100
SEEDS = (0, 1, 2)


# ---------------------------------------------------------------------
# Ex-ante demand model (no simulation, no fitted constants)
# ---------------------------------------------------------------------

def _noise_fields(settings: SDRSimulationConfig, profile: str | None):
    """The per-station noise fields exactly as run_scheduled_star
    resolves them (profile None = the settings' own class)."""

    if profile is None:
        return {}, None
    return resolve_oscillator_noise(
        profile,
        settings.carrier_frequency_hz,
        settings.sample_rate,
        settings.sync_interval,
    )


def link_matrices(
    settings: SDRSimulationConfig,
    num_stations: int,
    profiles: list[str] | None,
    seed: int,
    radius_m: float = 500.0,
    path_loss_exponent: float = 2.7,
    reference_distance_m: float = 500.0,
):
    """Per-link (F, Q, R) rebuilt from the same construction
    run_scheduled_star uses - returns a list of dicts, one per
    non-reference station."""

    positions = place_stations(num_stations, radius_m, seed)
    device = torch.device("cpu")
    preamble = make_sync_preamble(settings, device)
    interval_samples = int(
        round(settings.sync_interval * settings.sample_rate)
    )

    reference_noise, _ = _noise_fields(
        settings, None if profiles is None else profiles[0]
    )
    reference_phase_walk = reference_noise.get(
        "phase_noise_std_rad", settings.phase_noise_std_rad
    )
    reference_covariance = np.diag(
        [
            reference_noise.get(
                "phase_process_std_rad", settings.phase_process_std_rad
            )
            ** 2,
            (
                2.0
                * math.pi
                * reference_noise.get(
                    "frequency_process_std_hz",
                    settings.frequency_process_std_hz,
                )
            )
            ** 2,
        ]
    )
    generator = torch.Generator()
    generator.manual_seed(0)
    flicker = _FlickerFrequencyNoise(
        reference_noise.get(
            "flicker_frequency_std_hz", settings.flicker_frequency_std_hz
        ),
        settings.sync_interval,
        settings.num_iterations * settings.sync_interval,
        device,
        generator,
    )

    transition = np.array([[1.0, settings.sync_interval], [0.0, 1.0]])
    links = []
    for station in range(1, num_stations):
        distance = max(
            float(np.linalg.norm(positions[station] - positions[0])), 1.0
        )
        snr_db = min(
            settings.snr_db
            - 10.0
            * path_loss_exponent
            * math.log10(distance / reference_distance_m),
            MAX_LINK_SNR_DB,
        )
        station_noise, _ = _noise_fields(
            settings, None if profiles is None else profiles[station]
        )
        fields = {
            field: getattr(settings, field)
            for field in settings.__dataclass_fields__
        }
        fields.update(station_noise)
        fields["snr_db"] = snr_db
        link_settings = SDRSimulationConfig(**fields)

        slave_covariance = np.diag(
            [
                link_settings.phase_process_std_rad**2,
                (2.0 * math.pi * link_settings.frequency_process_std_hz)
                ** 2,
            ]
        )
        white_fm_phase_variance = (
            0.5
            * (
                reference_phase_walk**2
                + link_settings.phase_noise_std_rad**2
            )
            * interval_samples
        )
        process = (
            reference_covariance
            + slave_covariance
            + np.diag(
                [white_fm_phase_variance, float(flicker.innovation_variance)]
            )
        )
        measurement = (
            0.5
            * _measurement_covariance(link_settings, preamble, device)
            .cpu()
            .numpy()
        )
        links.append(
            {
                "station": station,
                "F": transition,
                "Q": process,
                "R": measurement,
                "snr_db": snr_db,
            }
        )
    return links


def steady_posterior(F, Q, R, iterations: int = 400) -> np.ndarray:
    """Steady-state post-update covariance of the link EKF, linearized
    at zero phase (H rows: cos-row uninformative, sin-row = phase,
    third row = frequency)."""

    H = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    P = np.diag([math.pi**2, (2.0 * math.pi * 50e3) ** 2])
    identity = np.eye(2)
    for _ in range(iterations):
        P = F @ P @ F.T + Q
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        residual = identity - K @ H
        P = residual @ P @ residual.T + K @ R @ K.T
    return P


def coast_steps(
    F, Q, posterior, sigma_target: float, max_steps: int = 100000
) -> int:
    """Predict-only intervals from the steady posterior until the
    predicted phase std crosses sigma_target (the scheduler's
    service point)."""

    P = posterior.copy()
    for step in range(1, max_steps + 1):
        P = F @ P @ F.T + Q
        if math.sqrt(max(P[0, 0], 0.0)) >= sigma_target:
            return step
    return max_steps


def closed_form_coast(
    q_phase: float,
    sigma_omega: float,
    sigma_target: float,
    sync_interval: float,
    latency_intervals: int,
) -> float:
    """Solve q_phase*(tau/T) + (sigma_omega*(tau+L*T))^2 = target^2
    for tau (seconds) by bisection."""

    def excess(tau: float) -> float:
        return (
            q_phase * (tau / sync_interval)
            + (
                sigma_omega
                * (tau + latency_intervals * sync_interval)
            )
            ** 2
            - sigma_target**2
        )

    low, high = 0.0, sync_interval
    while excess(high) < 0.0 and high < 1e6 * sync_interval:
        high *= 2.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if excess(mid) > 0.0:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def predict_demand(
    settings: SDRSimulationConfig,
    num_stations: int,
    profiles: list[str] | None,
    seed: int,
    budget_rad: float = BUDGET_RAD,
    trigger_fraction: float = TRIGGER_FRACTION,
):
    """Ex-ante per-link coast steps and total demand (exchanges per
    interval). Returns (demand, [per-link coast steps], [closed-form
    coast steps])."""

    sigma_target = trigger_fraction * budget_rad
    demand = 0.0
    steps_exact: list[int] = []
    steps_closed: list[float] = []
    for link in link_matrices(settings, num_stations, profiles, seed):
        posterior = steady_posterior(link["F"], link["Q"], link["R"])
        m_exact = coast_steps(link["F"], link["Q"], posterior, sigma_target)
        tau_closed = closed_form_coast(
            link["Q"][0, 0],
            math.sqrt(max(posterior[1, 1], 0.0)),
            sigma_target,
            settings.sync_interval,
            settings.correction_latency_intervals,
        )
        steps_exact.append(m_exact)
        steps_closed.append(tau_closed / settings.sync_interval)
        demand += 1.0 / m_exact
    return demand, steps_exact, steps_closed


# ---------------------------------------------------------------------
# Measurement (the repo's own simulator, untouched)
# ---------------------------------------------------------------------

def effective_gain(result) -> float:
    gain = result.mean_array_gain
    if gain == gain:
        return gain
    tail = result.array_gain[-max(1, result.array_gain.numel() // 4):]
    return float(torch.mean(tail))


def measure_gain(
    num_stations: int,
    policy: str,
    capacity: int,
    profiles: list[str] | None,
    seeds=SEEDS,
    intervals: int = MEASURE_INTERVALS,
) -> float:
    gains = []
    for seed in seeds:
        settings = SDRSimulationConfig(
            num_iterations=intervals, seed=seed, device="cpu"
        )
        result = run_scheduled_star(
            settings,
            num_stations=num_stations,
            policy=policy,
            budgets_rad=[BUDGET_RAD] * (num_stations - 1),
            max_exchanges_per_interval=capacity,
            oscillator_profiles=profiles,
        )
        gains.append(effective_gain(result))
    return float(np.mean(gains))


def measured_knee(
    num_stations: int,
    profiles: list[str] | None,
    predicted_capacity: int,
    policy: str = "scheduled",
) -> tuple[int | None, dict[int, float]]:
    """Sweep a window centered on the predicted knee; extend upward if
    the window top still fails, so an unbracketed knee is a reported
    miss rather than a hidden one. Returns (knee or None, gains)."""

    low = max(1, predicted_capacity - 2)
    high = min(num_stations - 1, predicted_capacity + 2)
    gains: dict[int, float] = {}
    capacity = low
    while capacity <= high:
        gains[capacity] = measure_gain(
            num_stations, policy, capacity, profiles
        )
        capacity += 1
        if capacity > high and gains[high] < GAIN_KNEE:
            new_high = min(num_stations - 1, high + 2)
            if new_high > high:
                high = new_high
    # Also probe below the window if its bottom already passes, so the
    # knee (FIRST passing capacity) is not overstated.
    while low > 1 and gains[low] >= GAIN_KNEE:
        low -= 1
        gains[low] = measure_gain(num_stations, policy, low, profiles)
    knee = None
    for capacity in sorted(gains):
        if gains[capacity] >= GAIN_KNEE:
            knee = capacity
            break
    return knee, gains


# ---------------------------------------------------------------------
# The three predictions
# ---------------------------------------------------------------------

P1_FLEETS = {
    "all-sdr": ["sdr"] * 10,
    "all-tcxo": ["tcxo"] * 10,
    "mixed": ["tcxo"] * 5 + ["sdr"] * 5,
}
P2_CAPACITY = 3
P2_SWEEP = (4, 6, 8, 10, 12, 14, 16)
P3_TCXO_COUNTS = (0, 2, 5, 7, 9)  # among the 9 slaves; reference ocxo


def p3_profiles(tcxo_count: int) -> list[str]:
    return (
        ["ocxo"]
        + ["tcxo"] * tcxo_count
        + ["ocxo"] * (9 - tcxo_count)
    )


def predict_all():
    """Every prediction, computed and returned before any simulation
    is allowed to run."""

    settings = SDRSimulationConfig(
        num_iterations=MEASURE_INTERVALS, seed=0, device="cpu"
    )
    predictions: dict = {}

    p1 = {}
    for name, profiles in P1_FLEETS.items():
        per_seed = [
            predict_demand(settings, 10, profiles, seed)[0]
            for seed in SEEDS
        ]
        demand = float(np.mean(per_seed))
        p1[name] = {
            "demand": demand,
            "demand_per_seed": per_seed,
            "knee": int(math.ceil(demand - 1e-9)),
        }
    predictions["p1"] = p1

    p2_demand = {}
    for n in P2_SWEEP:
        per_seed = [
            predict_demand(settings, n, ["sdr"] * n, seed)[0]
            for seed in SEEDS
        ]
        p2_demand[n] = float(np.mean(per_seed))
    n_sched = max(
        (n for n in P2_SWEEP if p2_demand[n] <= P2_CAPACITY),
        default=None,
    )
    predictions["p2"] = {
        "demand": p2_demand,
        "n_uniform": P2_CAPACITY + 1,
        "n_scheduled": n_sched,
    }

    p3 = {}
    for count in P3_TCXO_COUNTS:
        per_seed = [
            predict_demand(settings, 10, p3_profiles(count), seed)[0]
            for seed in SEEDS
        ]
        demand = float(np.mean(per_seed))
        p3[count] = {
            "demand": demand,
            "knee": int(math.ceil(demand - 1e-9)),
        }
    predictions["p3"] = p3
    return predictions


def print_predictions(predictions) -> None:
    print("=" * 68)
    print("EX-ANTE PREDICTIONS (fixed before any measurement below)")
    print(f"demand priced at trigger = {TRIGGER_FRACTION} x "
          f"{BUDGET_RAD} rad = {TRIGGER_FRACTION * BUDGET_RAD:.4f} rad")
    print("=" * 68)
    print("\nP1  critical capacity, N=10 scheduled:")
    for name, entry in predictions["p1"].items():
        per_seed = ", ".join(f"{d:.2f}" for d in entry["demand_per_seed"])
        print(
            f"  {name:<9} demand C* = {entry['demand']:.2f} "
            f"exchanges/interval (seeds: {per_seed}) "
            f"-> predicted knee capacity {entry['knee']}"
        )
    p2 = predictions["p2"]
    print(f"\nP2  wall shift at capacity {P2_CAPACITY}, sdr fleet:")
    print(
        "  predicted N*_uniform = "
        f"{p2['n_uniform']}  (uniform demands N-1)"
    )
    demand_string = ", ".join(
        f"N={n}:{d:.2f}" for n, d in p2["demand"].items()
    )
    print(f"  scheduled demand: {demand_string}")
    print(
        f"  predicted N*_scheduled = {p2['n_scheduled']}  "
        "(largest swept N with demand <= capacity)"
    )
    if p2["n_scheduled"]:
        print(
            "  predicted wall-shift ratio = "
            f"{p2['n_scheduled'] / p2['n_uniform']:.2f}"
        )
    print("\nP3  heterogeneous dividend, N=10 (reference ocxo, k tcxo "
          "slaves):")
    for count, entry in predictions["p3"].items():
        print(
            f"  k={count}  demand D = {entry['demand']:.2f} "
            f"-> predicted knee capacity {entry['knee']}"
        )
    print()


# ---------------------------------------------------------------------
# Measurement parts
# ---------------------------------------------------------------------

def run_part1(predictions) -> None:
    print("--- P1 measurements (scheduled, N=10, 3-seed mean gain) ---")
    for name, profiles in P1_FLEETS.items():
        predicted = predictions["p1"][name]["knee"]
        knee, gains = measured_knee(10, profiles, predicted)
        curve = "  ".join(
            f"cap{c}:{100 * g:5.1f}%" for c, g in sorted(gains.items())
        )
        demand = predictions["p1"][name]["demand"]
        if knee is None:
            verdict = "MISS (no capacity in window reached 90%)"
        elif knee - 1 < demand <= knee:
            verdict = f"HIT (C*={demand:.2f} in ({knee - 1},{knee}])"
        else:
            verdict = (
                f"MISS (knee {knee}, C*={demand:.2f}, "
                f"error {knee - demand:+.2f} exchanges/interval)"
            )
        print(f"  {name:<9} predicted {predicted}  measured knee "
              f"{knee}  ->  {verdict}")
        print(f"            {curve}")


def run_part2(predictions) -> None:
    print(f"--- P2 measurements (capacity {P2_CAPACITY}, sdr fleet, "
          "3-seed mean gain) ---")
    measured = {}
    for policy in ("uniform", "scheduled"):
        for n in P2_SWEEP:
            measured[(policy, n)] = measure_gain(
                n, policy, P2_CAPACITY, ["sdr"] * n
            )
        curve = "  ".join(
            f"N={n}:{100 * measured[(policy, n)]:5.1f}%" for n in P2_SWEEP
        )
        print(f"  {policy:<10} {curve}")
    for policy, predicted in (
        ("uniform", predictions["p2"]["n_uniform"]),
        ("scheduled", predictions["p2"]["n_scheduled"]),
    ):
        passing = [
            n for n in P2_SWEEP if measured[(policy, n)] >= GAIN_KNEE
        ]
        star = max(passing) if passing else None
        verdict = "HIT" if star == predicted else (
            f"MISS (error {star - predicted:+d} in N)"
            if star is not None and predicted is not None
            else "MISS (no sustainable N found)"
        )
        print(
            f"  {policy:<10} predicted N* = {predicted}, measured N* = "
            f"{star}  ->  {verdict}"
        )
    uniform_star = max(
        (n for n in P2_SWEEP if measured[("uniform", n)] >= GAIN_KNEE),
        default=None,
    )
    scheduled_star = max(
        (n for n in P2_SWEEP if measured[("scheduled", n)] >= GAIN_KNEE),
        default=None,
    )
    if uniform_star and scheduled_star:
        print(
            "  measured wall-shift ratio = "
            f"{scheduled_star / uniform_star:.2f} (predicted "
            f"{predictions['p2']['n_scheduled'] / predictions['p2']['n_uniform']:.2f})"
        )


def run_part3(predictions) -> None:
    print("--- P3 measurements (scheduled, N=10, 3-seed mean gain) ---")
    for count in P3_TCXO_COUNTS:
        entry = predictions["p3"][count]
        knee, gains = measured_knee(10, p3_profiles(count), entry["knee"])
        curve = "  ".join(
            f"cap{c}:{100 * g:5.1f}%" for c, g in sorted(gains.items())
        )
        if knee is None:
            verdict = "MISS (no capacity in window reached 90%)"
        elif knee - 1 < entry["demand"] <= knee:
            verdict = f"HIT (D={entry['demand']:.2f} in ({knee - 1},{knee}])"
        else:
            verdict = (
                f"MISS (knee {knee}, D={entry['demand']:.2f}, "
                f"error {knee - entry['demand']:+.2f})"
            )
        print(
            f"  k={count}  predicted {entry['knee']}  measured knee "
            f"{knee}  ->  {verdict}"
        )
        print(f"        {curve}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="blind ex-ante prediction of the sync airtime wall"
    )
    parser.add_argument(
        "--part", type=str, default="all",
        choices=("predict", "1", "2", "3", "all"),
    )
    args = parser.parse_args()

    predictions = predict_all()
    print_predictions(predictions)
    if args.part == "predict":
        return
    if args.part in ("1", "all"):
        run_part1(predictions)
    if args.part in ("2", "all"):
        run_part2(predictions)
    if args.part in ("3", "all"):
        run_part3(predictions)


if __name__ == "__main__":
    main()
