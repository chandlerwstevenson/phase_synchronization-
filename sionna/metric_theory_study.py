"""Does the frozen phase-diagram theory transfer across metrics?

Round 2 (phase_diagram_round2.py) blind-tested the frozen two-parameter
model on ARRAY GAIN — the fraction of ideal coherent beam power the
array holds. But nobody deploys an array to maximize array gain; they
deploy it to detect targets or to carry data. This study pushes the
SAME frozen model (zero refits, zero new constants) through to two
deployment metrics and re-scores the round-2 blind test on each:

  spectral efficiency   log2(1 + SNR) at a single-antenna user the
                        array beamforms one data stream to, in bits
                        per second per hertz. Reported as the mean
                        over residual draws and as the "95%-likely"
                        value (the level exceeded 95% of the time —
                        the cell-free literature's edge-user metric).
                        Evaluated at three users:
                          edge    fixed per-station SNR of -20 dB
                                  (weak link: sync errors matter)
                          strong  fixed per-station SNR of +10 dB
                                  (saturated: log compresses errors)
                          phys    a user 1.2 km from the array
                                  centroid under the repo's own link
                                  budget (turns out deeply saturated)
  detection range       largest range at which the array detects a
                        -15 dBsm drone 90% of the time (Swerling 1,
                        false-alarm 1e-6), via the repo's N^3 G^2
                        viability budget. Scales as G^(1/2), so it
                        compresses gain differences too, just less
                        than the logarithm does.
  net throughput        (1 - sync airtime) x mean edge spectral
                        efficiency: what the channel actually delivers
                        after paying for synchronization.

Concavity is handled by full Monte Carlo propagation: the model's
per-station residual spreads define independent Gaussian phase draws,
and every metric is averaged over draws of log2(1 + SNR(theta)), never
computed from the mean SNR (Jensen's inequality makes the naive
version an overestimate; the tests pin the ordering).

Blind protocol, with one honest caveat: predictions are printed before
any measurement, and the model is byte-identical to the round-2
freeze — but the round-2 GAIN measurements already exist, so only the
spectral-efficiency and range scorecards are fully fresh; the gain
column is reproduced for reference, not re-claimed as blind.

Nothing in any existing file is modified. The sibling metrics.py did
not exist when this study ran, so the metric definitions above are
implemented here (importable for later reconciliation).

Usage:
    .venv/bin/python metric_theory_study.py           # full study
    .venv/bin/python metric_theory_study.py --quick   # 2 curves, 1 seed
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from detection.viability import DetectionParams, detection_range_m
from detection.waveform import BOLTZMANN_T0
from gating_study import evaluation_mask, phase_matrix
from ota_sync import SDRSimulationConfig
from ota_sync.network import place_stations
from ota_sync.scheduled import run_scheduled_star
from phase_diagram_round2 import CURVES, SEEDS, fleet_r2, predict_condition

ITERATIONS = 60
EDGE_SNR_PER_STATION = 10.0 ** (-20.0 / 10.0)   # -20 dB: weak link
STRONG_SNR_PER_STATION = 10.0 ** (10.0 / 10.0)  # +10 dB: saturated
USER_OFFSET_M = np.array([1200.0, 150.0])       # the detection studies' edge
COMM_BANDWIDTH_HZ = 1e6
MC_DRAWS = 4000
RUNS_PATH = os.path.join(
    os.path.dirname(__file__), "metric_theory_runs.json"
)
PARAMS = DetectionParams(tx_power_w=0.5)  # matches the detection studies


# ---------------------------------------------------------------------
# Metric definitions (shared wording with the phase-5 plan)
# ---------------------------------------------------------------------

def physical_user_amplitudes(n: int, seed: int) -> np.ndarray:
    """Per-station received amplitude sqrt(SNR_1) at the physical user:
    one-way Friis link at the repo's 915 MHz / 1 MHz budget, stations
    from the same deterministic deployment the simulator uses."""

    positions = place_stations(n, 500.0, seed)
    user = positions.mean(axis=0) + USER_OFFSET_M
    distances = np.maximum(np.linalg.norm(positions - user, axis=1), 1.0)
    antenna_gain = 10.0 ** (PARAMS.antenna_gain_dbi / 10.0)
    noise_w = (
        BOLTZMANN_T0
        * 10.0 ** (PARAMS.noise_figure_db / 10.0)
        * 10.0 ** (PARAMS.losses_db / 10.0)
        * COMM_BANDWIDTH_HZ
    )
    received_w = (
        PARAMS.tx_power_w
        * antenna_gain**2
        * PARAMS.wavelength_m**2
        / (4.0 * math.pi * distances) ** 2
    )
    return np.sqrt(received_w / noise_w)


def metric_draws(phases: np.ndarray, amplitudes: np.ndarray) -> np.ndarray:
    """Per-draw spectral efficiency log2(1+SNR), SNR the coherent
    beamforming sum |sum_k a_k e^{j theta_k}|^2. phases: (draws, N)."""

    field = np.sum(
        amplitudes[None, :] * np.exp(1j * phases), axis=1
    )
    return np.log2(1.0 + np.abs(field) ** 2)


def score_metrics(
    phases: np.ndarray, n: int, seed: int, airtime: float
) -> dict:
    """All metrics from a (draws, N) phase matrix (radians, station 0
    the reference datum). Draws are steady intervals (measured) or
    Monte Carlo samples (predicted) — same estimator either way."""

    gain_draws = (
        np.abs(np.sum(np.exp(1j * phases), axis=1)) ** 2 / n**2
    )
    ones = np.ones(n)
    edge = metric_draws(phases, math.sqrt(EDGE_SNR_PER_STATION) * ones)
    strong = metric_draws(
        phases, math.sqrt(STRONG_SNR_PER_STATION) * ones
    )
    physical = metric_draws(phases, physical_user_amplitudes(n, seed))
    return {
        "gain": float(np.mean(gain_draws)),
        "se_edge": float(np.mean(edge)),
        "se_edge_95": float(np.quantile(edge, 0.05)),
        "se_strong": float(np.mean(strong)),
        "se_phys": float(np.mean(physical)),
        "range_m": detection_range_m(
            n, float(np.mean(gain_draws)), PARAMS
        ),
        "net_throughput": (1.0 - airtime) * float(np.mean(edge)),
    }


def perfect_metrics(n: int, seed: int) -> dict:
    """Every metric at exact synchronization (the 100% reference)."""

    zero = np.zeros((1, n))
    return score_metrics(zero, n, seed, airtime=0.0)


# ---------------------------------------------------------------------
# Predicted metrics from the frozen round-2 model (no refits)
# ---------------------------------------------------------------------

def predicted_metrics(curve: dict, capacity: int, seed: int) -> dict:
    """Monte Carlo the frozen model's per-station residual spreads
    through every metric. Airtime prediction: serviced cadence from the
    model's coast intervals, priced at the star's per-exchange cost."""

    prediction = predict_condition(
        curve["n"], fleet_r2(curve["fleet"], curve["n"]), seed,
        curve["latency"], curve["budget"], capacity, ITERATIONS,
    )
    spreads = np.array(prediction["s_mrad"]) / 1e3  # radians
    rng = np.random.default_rng(9000 + 7 * seed + capacity)
    phases = np.zeros((MC_DRAWS, curve["n"]))
    phases[:, 1:] = rng.normal(0.0, spreads[None, :], (MC_DRAWS, curve["n"] - 1))
    # Ex-ante airtime: exchanges per interval at the serviced cadence,
    # capped at capacity, priced at the star's measured per-exchange
    # cost (one full two-way exchange occupies ~19.1% of an interval).
    demand = sum(1.0 / m for m in prediction["m_intervals"])
    airtime = min(min(demand, float(capacity)) * 0.1912, 1.0)
    return score_metrics(phases, curve["n"], seed, airtime)


# ---------------------------------------------------------------------
# Measured metrics (fresh runs; cached, resumable)
# ---------------------------------------------------------------------

def measure_condition(curve: dict, capacity: int, seed: int) -> dict:
    settings = SDRSimulationConfig(
        num_iterations=ITERATIONS, seed=seed, device="cpu",
        correction_latency_intervals=curve["latency"],
    )
    result = run_scheduled_star(
        settings,
        num_stations=curve["n"],
        policy="scheduled",
        budgets_rad=[curve["budget"]] * (curve["n"] - 1),
        max_exchanges_per_interval=capacity,
        oscillator_profiles=fleet_r2(curve["fleet"], curve["n"]),
    )
    mask = evaluation_mask(result)
    phases = phase_matrix(result)[:, mask].numpy().T  # (draws, N)
    return score_metrics(
        phases, curve["n"], seed, result.airtime_used_fraction
    )


# ---------------------------------------------------------------------
# Scoring (round-2 conventions, bands stated before measuring)
# ---------------------------------------------------------------------

METRIC_BANDS = {
    # metric key -> (plateau band as fraction of the perfect value,
    #                knee rule: first capacity >= 90% of perfect)
    "se_edge": 0.15,
    "range_m": 0.10,
}


def curve_summary(per_capacity: dict, perfect: float) -> tuple:
    plateau = per_capacity[max(per_capacity)]
    knee = None
    for capacity in sorted(per_capacity):
        if per_capacity[capacity] >= 0.90 * perfect:
            knee = capacity
            break
    return plateau, knee


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ranks_a = np.argsort(np.argsort(a)).astype(float)
    ranks_b = np.argsort(np.argsort(b)).astype(float)
    ca, cb = ranks_a - ranks_a.mean(), ranks_b - ranks_b.mean()
    denominator = math.sqrt(float(np.sum(ca**2) * np.sum(cb**2)))
    return float(np.sum(ca * cb) / denominator) if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="frozen theory scored on deployment metrics"
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    curves = []
    for n, fleet, budget, latency in (CURVES[:2] if args.quick else CURVES):
        curves.append(
            {"n": n, "fleet": fleet, "budget": budget, "latency": latency}
        )
    seeds = SEEDS[:1] if args.quick else SEEDS

    with open("phase_diagram_round2_runs.json") as handle:
        capacity_lists: dict = {}
        for record in json.load(handle):
            key = (record["n"], record["fleet"], record["latency"],
                   round(record["budget"], 6))
            capacity_lists.setdefault(key, set()).add(record["capacity"])

    # ---- STEP 1: predictions first --------------------------------
    print("=== STEP 1: FROZEN-MODEL METRIC PREDICTIONS (printed before "
          "any measurement) ===")
    print("bands, fixed now: edge spectral efficiency plateau within "
          "15% of prediction; detection-range plateau within 10%; knee "
          "(first capacity reaching 90% of the perfect-sync value) "
          "within +-1; classification = plateau reaches 90% of perfect. "
          "Gain was already measured in round 2, so only the spectral-"
          "efficiency and range scorecards are fresh.")
    predictions = {}
    for curve in curves:
        key = (curve["n"], curve["fleet"], curve["latency"],
               round(curve["budget"], 6))
        capacities = sorted(capacity_lists.get(key, set()))
        if not capacities:
            capacities = sorted({1, 2, 3, curve["n"] - 1})
        per_metric: dict = {}
        for capacity in capacities:
            seed_scores = [
                predicted_metrics(curve, capacity, seed) for seed in seeds
            ]
            for metric in seed_scores[0]:
                per_metric.setdefault(metric, {})[capacity] = float(
                    np.mean([s[metric] for s in seed_scores])
                )
        perfect = perfect_metrics(curve["n"], seeds[0])
        summary = {}
        for metric in ("se_edge", "range_m"):
            plateau, knee = curve_summary(per_metric[metric], perfect[metric])
            summary[metric] = {
                "plateau": plateau, "knee": knee,
                "perfect": perfect[metric],
                "reachable": plateau >= 0.90 * perfect[metric],
            }
        predictions[key] = {
            "curve": curve, "capacities": capacities,
            "per_metric": per_metric, "summary": summary,
        }
        se, rng_ = summary["se_edge"], summary["range_m"]
        print(
            f"  N={curve['n']:<3}{curve['fleet']:<6} "
            f"B={curve['budget']:.2f} L={curve['latency']}  "
            f"edge-SE plateau {se['plateau']:5.2f}/"
            f"{se['perfect']:4.2f} b/s/Hz "
            f"knee={'-' if se['knee'] is None else se['knee']:<3} "
            f"{'REACH' if se['reachable'] else 'UNREACH':<7} | "
            f"range plateau {rng_['plateau']:6.0f}/"
            f"{rng_['perfect']:5.0f} m "
            f"knee={'-' if rng_['knee'] is None else rng_['knee']}"
        )

    # ---- STEP 2: measurement --------------------------------------
    if os.path.exists(RUNS_PATH):
        with open(RUNS_PATH) as handle:
            measured_runs = json.load(handle)
    else:
        measured_runs = []
    done = {
        (r["n"], r["fleet"], r["latency"], round(r["budget"], 6),
         r["capacity"], r["seed"])
        for r in measured_runs
    }
    todo = []
    for key, entry in predictions.items():
        for capacity in entry["capacities"]:
            for seed in seeds:
                if key + (capacity, seed) not in done:
                    todo.append((key, entry["curve"], capacity, seed))
    print(f"\n=== STEP 2: measuring {len(todo)} runs "
          f"({len(measured_runs)} cached) ===", flush=True)
    started = time.time()
    for index, (key, curve, capacity, seed) in enumerate(todo):
        scores = measure_condition(curve, capacity, seed)
        measured_runs.append({
            "n": curve["n"], "fleet": curve["fleet"],
            "latency": curve["latency"], "budget": curve["budget"],
            "capacity": capacity, "seed": seed, **scores,
        })
        with open(RUNS_PATH, "w") as handle:
            json.dump(measured_runs, handle)
        if (index + 1) % 20 == 0 or index + 1 == len(todo):
            print(f"  [{index + 1}/{len(todo)}] "
                  f"{(time.time() - started) / 60.0:.1f} min", flush=True)

    def measured(key, metric):
        values: dict = {}
        for r in measured_runs:
            if (r["n"], r["fleet"], r["latency"],
                    round(r["budget"], 6)) == key:
                values.setdefault(r["capacity"], []).append(r[metric])
        return {c: float(np.mean(v)) for c, v in values.items()}

    # ---- STEP 3: scorecards ---------------------------------------
    print("\n=== STEP 3: BLIND SCORECARDS ===")
    totals = {}
    for metric, band in METRIC_BANDS.items():
        name = ("edge spectral efficiency" if metric == "se_edge"
                else "detection range")
        print(f"\n--- {name} ---")
        hits = {"class": 0, "plateau": 0, "knee": 0}
        counts = {"class": 0, "plateau": 0, "knee": 0}
        for key, entry in predictions.items():
            pred = entry["summary"][metric]
            meas_curve = measured(key, metric)
            if not meas_curve:
                continue
            perfect = pred["perfect"]
            plateau_meas, knee_meas = curve_summary(meas_curve, perfect)
            class_hit = (
                (plateau_meas >= 0.90 * perfect) == pred["reachable"]
            )
            plateau_hit = (
                abs(plateau_meas - pred["plateau"])
                <= band * max(perfect, 1e-9)
            )
            knee_hit = (
                pred["knee"] is not None and knee_meas is not None
                and abs(pred["knee"] - knee_meas) <= 1
            ) or (pred["knee"] is None and knee_meas is None)
            for label, hit in (
                ("class", class_hit), ("plateau", plateau_hit),
                ("knee", knee_hit),
            ):
                counts[label] += 1
                hits[label] += int(hit)
            curve = entry["curve"]
            print(
                f"  N={curve['n']:<3}{curve['fleet']:<6} "
                f"B={curve['budget']:.2f} L={curve['latency']}  "
                f"plateau pred {pred['plateau']:8.2f} "
                f"meas {plateau_meas:8.2f}  "
                f"knee pred {'-' if pred['knee'] is None else pred['knee']} "
                f"meas {'-' if knee_meas is None else knee_meas}  "
                f"[{'Y' if class_hit else 'n'}"
                f"{'Y' if plateau_hit else 'n'}"
                f"{'Y' if knee_hit else 'n'}]"
            )
        totals[metric] = (hits, counts)
        print(f"  score: classification {hits['class']}/{counts['class']}, "
              f"plateau {hits['plateau']}/{counts['plateau']}, "
              f"knee {hits['knee']}/{counts['knee']}")

    # ---- STEP 4: does the metric change the conclusion? -----------
    print("\n=== STEP 4: RANK AGREEMENT AND SATURATION ===")
    metric_keys = ["se_edge", "se_edge_95", "se_strong", "se_phys",
                   "range_m", "net_throughput"]
    gains = np.array([r["gain"] for r in measured_runs])
    print("rank correlation of each metric with array gain over all "
          f"{len(measured_runs)} measured runs (1.0 = same ordering):")
    for metric in metric_keys:
        values = np.array([r[metric] for r in measured_runs])
        print(f"  {metric:<15} {spearman(gains, values):+.3f}")
    # knee shift per curve: gain knee vs each metric's knee
    print("\nknee (first capacity at 90% of perfect) by metric, "
          "per curve — where the metric choice moves the operating "
          "point:")
    for key, entry in predictions.items():
        curve = entry["curve"]
        knees = []
        for metric in ("gain", "se_edge", "se_strong", "range_m",
                       "net_throughput"):
            meas_curve = measured(key, metric)
            if metric == "net_throughput":
                # net throughput has no perfect=1 anchor; report argmax
                best = max(meas_curve, key=meas_curve.get)
                knees.append(f"nt*={best}")
                continue
            perfect = (1.0 if metric == "gain"
                       else entry["summary"].get(metric, {}).get("perfect"))
            if perfect is None:
                perfect = perfect_metrics(curve["n"], seeds[0])[metric]
            _, knee = curve_summary(meas_curve, perfect)
            knees.append(f"{metric}={'-' if knee is None else knee}")
        print(f"  N={curve['n']:<3}{curve['fleet']:<6} "
              f"B={curve['budget']:.2f} L={curve['latency']}  "
              + "  ".join(knees))


if __name__ == "__main__":
    main()
