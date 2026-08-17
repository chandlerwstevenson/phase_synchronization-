"""Round-2 blind test of the two-parameter coherence phase diagram.

Round 1 (wall_prediction_study.py, coherence_collapse_study.py) killed
single-number universality (1/10 blind hits; isotonic R^2 0.577) but
left two one-signed, mechanistic diagnoses: (a) a missing feasibility
gate - the error floor caps achievable gain independent of capacity -
and (b) demand priced at the trigger line when sustainable capacity
tracks the budget line. This module freezes the CORRECTED model and
blind-tests it on conditions round 1 never ran.

The frozen model (ex ante, zero fitted constants):

  1. Per station k, reconstruct the link's exact EKF matrices from
     datasheet anchors + deployment geometry (coast_law.link_matrices,
     reference profile = fleet[0]).
  2. Cadence: the deployed scheduler services worst-first when the
     posterior phase std crosses f*B (f = trigger fraction 0.5). Under
     contention, worst-first equalizes urgency, so every station coasts
     to a COMMON threshold multiple kappa*f*B, with kappa >= 1 the
     smallest value whose total demand fits the channel:
         sum_k 1/m_k(kappa*f*B) <= capacity,   threshold capped at pi.
     m_k(theta) is the self-consistent coast cycle P = upd(pred^m(P))
     (coast_law's exact cadence layer, reimplemented in closed form for
     speed and cross-checked against coast_law in the tests).
  3. Residual: the truth the beam sees is the filter's coasting phase
     variance evaluated THROUGH the correction latency, plus the
     ex-ante multipath-resampling variance the filter does not model
     (coast_law.resampling_phase_variance, TDL-D Rice factor - an
     estimate from the channel spec, not a fit):
         s_k^2 = mean_{j=1..m_k} P00(j + L) + var_resamp.
     This carries BOTH round-1 diagnoses: the feasibility gate is
     s_k(m=1) (nothing can hold tighter than one-interval service) and
     the latency term lowers the plateau through the j+L horizon.
  4. Gain: stations contribute their expected phasor w_k = e^{-s_k^2/2}
     (reference w=1); independent-phase expectation of the coherent sum
         G_pred = [ (sum_k w_k)^2 + sum_k (1 - w_k^2) ] / N^2
     (the second term is the incoherent power of what has decohered -
     round 1's starved stations taught us it is not zero).

  Reported phase-diagram coordinates: rho_budget = capacity /
  sum_k 1/m_k(B), and phi_k = s_k(m=1)/B (feasibility).

Blind protocol: all predictions for the fresh grid are computed and
printed BEFORE any measurement runs; hits and misses are scored
against pre-stated bands and reported as-is.

Nothing in ota_sync/ or detection/ (or any existing file) is modified.

Usage:
    .venv/bin/python phase_diagram_round2.py             # full round 2
    .venv/bin/python phase_diagram_round2.py --reconcile # step 0 only
    .venv/bin/python phase_diagram_round2.py --pooled    # step 3 only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from coast_law import (
    LinkMatrices,
    link_matrices,
    predict_coast_time,
    resampling_phase_variance,
    station_snr_db,
    supply_demand_ratio,
)
from coherence_collapse_study import (
    CACHE_PATH,
    FLEETS,
    isotonic_fit,
    load_cache,
)
from gating_study import evaluation_mask
from ota_sync import SDRSimulationConfig
from ota_sync.scheduled import run_scheduled_star

TRIGGER_FRACTION = 0.5  # run_scheduled_star's deployed default
SYNC_INTERVAL = 0.05
MAX_COAST_INTERVALS = 60000
KAPPA_GRID_POINTS = 25
RESAMP_VAR = resampling_phase_variance()

ROUND2_PATH = os.path.join(
    os.path.dirname(__file__), "phase_diagram_round2_runs.json"
)


# ---------------------------------------------------------------------
# Fast exact coast-cycle machinery (closed-form covariance propagation;
# cross-checked against coast_law.cycle_posterior in the tests)
# ---------------------------------------------------------------------

def _p00_trajectory(P: np.ndarray, q00: float, q11: float, T: float,
                    steps: int) -> np.ndarray:
    """P00 after j predicts, j = 1..steps, in closed form.

    The predict recursion with constant F = [[1,T],[0,1]] and diagonal
    Q is linear, so each entry is polynomial in j:
      p11(j) = p11 + j q11
      p01(j) = p01 + T [ j p11 + q11 j(j-1)/2 ]
      p00(j) = p00 + sum_{i=0}^{j-1} ( 2T p01(i) + T^2 p11(i) + q00 )
    """

    p00, p01, p11 = float(P[0, 0]), float(P[0, 1]), float(P[1, 1])
    j = np.arange(steps, dtype=np.float64)  # i = 0..steps-1
    p11_i = p11 + j * q11
    p01_i = p01 + T * (j * p11 + q11 * j * (j - 1) / 2.0)
    increments = 2.0 * T * p01_i + T * T * p11_i + q00
    return p00 + np.cumsum(increments)


def _predict_m(P: np.ndarray, q00: float, q11: float, T: float,
               m: int) -> np.ndarray:
    """P after m predicts (exact, closed form)."""

    p00_final = _p00_trajectory(P, q00, q11, T, m)[-1]
    p00, p01, p11 = float(P[0, 0]), float(P[0, 1]), float(P[1, 1])
    p11_m = p11 + m * q11
    p01_m = p01 + T * (m * p11 + q11 * m * (m - 1) / 2.0)
    return np.array([[p00_final, p01_m], [p01_m, p11_m]])


def _update_np(P: np.ndarray, R: np.ndarray) -> np.ndarray:
    gain = np.linalg.solve((P + R).T, P.T).T
    residual_map = np.eye(2) - gain
    return residual_map @ P @ residual_map.T + gain @ R @ gain.T


def fast_cycle(matrices: LinkMatrices, threshold_rad: float,
               max_intervals: int = MAX_COAST_INTERVALS,
               max_outer: int = 60) -> tuple[np.ndarray, int]:
    """Self-consistent (posterior, coast length): P = upd(pred^m(P)),
    m = first crossing of threshold. Mirrors coast_law.cycle_posterior
    exactly (same recursion, same trigger semantics); censored coasts
    return m = max_intervals + 1."""

    Q = matrices.process.numpy()
    R = matrices.measurement.numpy()
    T = float(matrices.transition[0, 1])
    q00, q11 = float(Q[0, 0]), float(Q[1, 1])

    # every-interval DARE start, iterated with the same machinery
    P = np.diag([math.pi**2, (2.0 * math.pi * 50e3) ** 2])
    for _ in range(4000):
        posterior = _update_np(_predict_m(P, q00, q11, T, 1), R)
        if np.max(np.abs(posterior - P)) < 1e-14:
            P = posterior
            break
        P = posterior

    previous_m = -1
    threshold_var = threshold_rad**2
    for _ in range(max_outer):
        trajectory = _p00_trajectory(P, q00, q11, T, max_intervals)
        crossed = np.nonzero(trajectory >= threshold_var)[0]
        if crossed.size == 0:
            return P, max_intervals + 1
        m = int(crossed[0]) + 1
        P_new = _update_np(_predict_m(P, q00, q11, T, m), R)
        if m == previous_m:
            return P_new, m
        P, previous_m = P_new, m
    return P, m


# ---------------------------------------------------------------------
# The frozen predictive model
# ---------------------------------------------------------------------

_MATRIX_CACHE: dict[tuple, LinkMatrices] = {}
_CYCLE_CACHE: dict[tuple, tuple[np.ndarray, int]] = {}


def _station_matrices(profile: str, reference: str, snr_db: float,
                      horizon_s: float) -> LinkMatrices:
    key = (profile, reference, round(snr_db, 4), round(horizon_s, 4))
    if key not in _MATRIX_CACHE:
        settings = SDRSimulationConfig(device="cpu")
        _MATRIX_CACHE[key] = link_matrices(
            settings, profile, snr_db, horizon_s,
            reference_profile=reference,
        )
    return _MATRIX_CACHE[key]


def _cycle(matrices_key: tuple, matrices: LinkMatrices,
           threshold: float) -> tuple[np.ndarray, int]:
    key = (matrices_key, round(threshold, 9))
    if key not in _CYCLE_CACHE:
        _CYCLE_CACHE[key] = fast_cycle(matrices, threshold)
    return _CYCLE_CACHE[key]


def predict_condition(
    n: int,
    profiles: list[str],
    seed: int,
    latency: int,
    budget: float,
    capacity: int,
    iterations: int = 60,
) -> dict:
    """Frozen ex-ante prediction for one (condition, capacity, seed).
    Calls nothing from the simulator's run path."""

    settings = SDRSimulationConfig(
        num_iterations=iterations, seed=seed, device="cpu",
        correction_latency_intervals=latency,
    )
    horizon_s = iterations * SYNC_INTERVAL
    stations = []
    for station in range(1, n):
        snr = station_snr_db(settings, n, station)
        matrices = _station_matrices(
            profiles[station], profiles[0], snr, horizon_s
        )
        key = (profiles[station], profiles[0], round(snr, 4),
               round(horizon_s, 4))
        stations.append((key, matrices))

    def demand_at(threshold: float) -> tuple[float, list[int]]:
        total, ms = 0.0, []
        for key, matrices in stations:
            _, m = _cycle(key, matrices, threshold)
            ms.append(m)
            total += 1.0 / m
        return total, ms

    # kappa: smallest common threshold multiple that fits the channel
    base = TRIGGER_FRACTION * budget
    kappa_max = math.pi / base
    grid = np.geomspace(1.0, kappa_max, KAPPA_GRID_POINTS)
    kappa = grid[-1]
    for candidate in grid:
        total, _ = demand_at(candidate * base)
        if total <= capacity + 1e-12:
            kappa = candidate
            break
    threshold = kappa * base
    _, ms = demand_at(threshold)

    weights = [1.0]  # reference
    s_list = []
    for (key, matrices), m in zip(stations, ms):
        posterior, m_cycle = _cycle(key, matrices, threshold)
        m_use = min(m_cycle, MAX_COAST_INTERVALS)
        Q = matrices.process.numpy()
        q00, q11 = float(Q[0, 0]), float(Q[1, 1])
        trajectory = _p00_trajectory(
            posterior, q00, q11, SYNC_INTERVAL, m_use + latency
        )
        visible = trajectory[latency:latency + m_use]
        s_sq = float(np.mean(visible)) + RESAMP_VAR
        s_list.append(math.sqrt(s_sq))
        weights.append(math.exp(-0.5 * min(s_sq, 60.0)))

    # feasibility floor per station: forced m=1 cycle (= DARE), through
    # latency, plus resampling
    phis = []
    for key, matrices in stations:
        Q = matrices.process.numpy()
        R = matrices.measurement.numpy()
        q00, q11 = float(Q[0, 0]), float(Q[1, 1])
        P = np.diag([math.pi**2, (2.0 * math.pi * 50e3) ** 2])
        for _ in range(4000):
            posterior = _update_np(
                _predict_m(P, q00, q11, SYNC_INTERVAL, 1), R
            )
            if np.max(np.abs(posterior - P)) < 1e-14:
                P = posterior
                break
            P = posterior
        floor_traj = _p00_trajectory(
            P, q00, q11, SYNC_INTERVAL, 1 + latency
        )
        floor_sq = float(floor_traj[latency]) + RESAMP_VAR
        phis.append(math.sqrt(floor_sq) / budget)

    w = np.array(weights)
    gain = (np.sum(w) ** 2 + np.sum(1.0 - w**2)) / n**2

    # rho at the budget line (round-1 diagnosis (b))
    demand_budget, _ = demand_at(budget)
    return {
        "gain_pred": float(gain),
        "kappa": float(kappa),
        "rho_budget": capacity / demand_budget
        if demand_budget > 0 else float("inf"),
        "phi_max": float(max(phis)),
        "s_mrad": [1e3 * s for s in s_list],
        "m_intervals": ms,
    }


def predict_curve(n, profiles, seeds, latency, budget, capacities,
                  iterations=60):
    """Seed-averaged predicted gain per capacity, plateau, knee."""

    per_capacity = {}
    for capacity in capacities:
        gains = [
            predict_condition(
                n, profiles, seed, latency, budget, capacity, iterations
            )["gain_pred"]
            for seed in seeds
        ]
        per_capacity[capacity] = float(np.mean(gains))
    plateau = per_capacity[max(capacities)]
    knee = None
    for capacity in sorted(capacities):
        if per_capacity[capacity] >= 0.90:
            knee = capacity
            break
    return per_capacity, plateau, knee


# ---------------------------------------------------------------------
# Step 0: reconciliation of the collapse cache's local rho against
# coast_law's canonical supply_demand_ratio
# ---------------------------------------------------------------------

def reconcile(records, sample=24) -> dict:
    families = {}
    for rec in records:
        key = (rec["n"], rec["fleet"], rec["seed"], rec["latency"],
               round(rec["budget"], 6))
        families.setdefault(key, rec)
    keys = sorted(families)[:: max(1, len(families) // sample)][:sample]
    ratios = []
    print(f"\n=== STEP 0: reconciliation on {len(keys)} config families "
          "(cache rho_trig vs coast_law supply_demand_ratio, "
          f"trigger={TRIGGER_FRACTION}) ===")
    for key in keys:
        n, fleet, seed, latency, budget = key
        rec = families[key]
        settings = SDRSimulationConfig(
            num_iterations=50, seed=seed, device="cpu",
            correction_latency_intervals=latency,
        )
        profiles = FLEETS[fleet](n)
        snrs = [station_snr_db(settings, n, s) for s in range(1, n)]
        rho_canonical = supply_demand_ratio(
            profiles[1:], snrs, rec["capacity"], latency,
            SYNC_INTERVAL, budget, settings=settings,
            trigger_fraction=TRIGGER_FRACTION,
        )
        ratio = rec["rho_trig"] / rho_canonical
        ratios.append(ratio)
        print(
            f"  N={n} {fleet:<6} seed={seed} L={latency} B={budget:.3f} "
            f"cap={rec['capacity']}: cache {rec['rho_trig']:6.3f}  "
            f"coast_law {rho_canonical:6.3f}  ratio {ratio:5.2f}"
        )
    summary = {
        "median": float(np.median(ratios)),
        "iqr": [float(np.percentile(ratios, 25)),
                float(np.percentile(ratios, 75))],
    }
    print(
        f"  cache/canonical ratio: median {summary['median']:.3f}, "
        f"IQR [{summary['iqr'][0]:.3f}, {summary['iqr'][1]:.3f}]"
    )
    return summary


# ---------------------------------------------------------------------
# Steps 1-2: frozen predictions on fresh conditions, then measurement
# ---------------------------------------------------------------------

def fleet_r2(name: str, n: int) -> list[str]:
    if name == "tcxo":
        return ["tcxo"] * n
    if name == "ocxo":
        return ["ocxo"] * n
    if name == "mix13":  # 1/3 ocxo, 2/3 tcxo, ocxo reference
        return ["ocxo" if i % 3 == 0 else "tcxo" for i in range(n)]
    if name == "sdr1":  # one sdr slave in an ocxo fleet
        profiles = ["ocxo"] * n
        profiles[1] = "sdr"
        return profiles
    raise ValueError(name)


CURVES = [
    # (n, fleet, budget, latency)
    (8, "tcxo", 0.25, 1),
    (8, "tcxo", 0.45, 1),
    (12, "tcxo", 0.25, 1),
    (12, "tcxo", 0.45, 1),
    (8, "ocxo", 0.25, 1),
    (12, "ocxo", 0.45, 1),
    (8, "mix13", 0.25, 1),
    (12, "mix13", 0.45, 1),
    (8, "sdr1", 0.25, 1),
    (12, "sdr1", 0.45, 1),
    (8, "tcxo", 0.25, 3),
    (8, "ocxo", 0.25, 3),
]
SEEDS = (3, 4, 5)
ITERATIONS = 60


def curve_capacities(n, knee):
    if knee is None:
        base = {1, 2, 3, max(1, (n - 1) // 2), n - 1}
    else:
        base = set(range(max(1, knee - 3), min(n - 1, knee + 3) + 1))
        base.add(n - 1)
    return sorted(base)


def assert_fresh(records):
    have = {
        (r["n"], r["fleet"], r["capacity"], r["seed"], r["latency"],
         round(r["budget"], 6))
        for r in records
    }
    for n, fleet, budget, latency in CURVES:
        for seed in SEEDS:
            assert not any(
                k[0] == n and k[3] == seed for k in have
            ), f"condition N={n} seed={seed} already in round-1 cache"


def freeze_predictions():
    """Compute and print every prediction BEFORE any measurement."""

    frozen = []
    print("\n=== STEP 1: FROZEN PREDICTIONS (printed before any "
          "measurement; bands stated now) ===")
    print("scoring bands, fixed now: plateau +-10 pts; knee +-1 "
          "capacity; reachable = plateau >= 90%")
    for n, fleet, budget, latency in CURVES:
        profiles = fleet_r2(fleet, n)
        # provisional capacities from a coarse knee scan
        scan = sorted(set(range(1, n)) )
        per_cap, plateau, knee = predict_curve(
            n, profiles, SEEDS, latency, budget, scan, ITERATIONS
        )
        capacities = curve_capacities(n, knee)
        diagnostics = predict_condition(
            n, profiles, SEEDS[0], latency, budget, max(capacities),
            ITERATIONS,
        )
        frozen.append({
            "n": n, "fleet": fleet, "budget": budget, "latency": latency,
            "capacities": capacities,
            "pred_gain": {c: per_cap[c] for c in capacities},
            "plateau": plateau,
            "knee": knee,
            "reachable": plateau >= 0.90,
            "phi_max": diagnostics["phi_max"],
        })
        gain_text = " ".join(
            f"{c}:{100 * per_cap[c]:.0f}" for c in capacities
        )
        print(
            f"  N={n:<3}{fleet:<6} B={budget:.2f} L={latency}  "
            f"knee={'-' if knee is None else knee:<3} "
            f"plateau={100 * plateau:5.1f}% "
            f"{'REACHABLE' if plateau >= 0.90 else 'UNREACHABLE':<12} "
            f"phi_max={diagnostics['phi_max']:.2f}  gains[{gain_text}]"
        )
    return frozen


def measure_curves(frozen):
    if os.path.exists(ROUND2_PATH):
        with open(ROUND2_PATH) as handle:
            runs = json.load(handle)
    else:
        runs = []
    have = {
        (r["n"], r["fleet"], r["capacity"], r["seed"], r["latency"],
         round(r["budget"], 6))
        for r in runs
    }
    todo = []
    for curve in frozen:
        for capacity in curve["capacities"]:
            for seed in SEEDS:
                key = (curve["n"], curve["fleet"], capacity, seed,
                       curve["latency"], round(curve["budget"], 6))
                if key not in have:
                    todo.append((curve, capacity, seed))
    print(f"\n=== STEP 2: measuring {len(todo)} fresh runs "
          f"({len(runs)} cached) ===")
    started = time.time()
    for index, (curve, capacity, seed) in enumerate(todo):
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
        gain = torch.mean(result.array_gain[mask]).item()
        runs.append({
            "n": curve["n"], "fleet": curve["fleet"],
            "capacity": capacity, "seed": seed,
            "latency": curve["latency"], "budget": curve["budget"],
            "gain": gain,
        })
        with open(ROUND2_PATH, "w") as handle:
            json.dump(runs, handle)
        if (index + 1) % 20 == 0 or index + 1 == len(todo):
            print(
                f"  [{index + 1}/{len(todo)}] "
                f"{(time.time() - started) / 60.0:.1f} min", flush=True
            )
    return runs


def score(frozen, runs):
    def measured_gain(curve, capacity):
        values = [
            r["gain"] for r in runs
            if r["n"] == curve["n"] and r["fleet"] == curve["fleet"]
            and r["capacity"] == capacity
            and r["latency"] == curve["latency"]
            and abs(r["budget"] - curve["budget"]) < 1e-9
        ]
        return float(np.mean(values)) if values else float("nan")

    print("\n=== SCORE: frozen predictions vs measurement ===")
    hits, total = 0, 0
    rows = []
    for curve in frozen:
        capacities = curve["capacities"]
        measured = {c: measured_gain(curve, c) for c in capacities}
        measured_plateau = measured[max(capacities)]
        measured_reachable = any(
            g == g and g >= 0.90 for g in measured.values()
        )
        measured_knee = None
        for capacity in sorted(capacities):
            if measured[capacity] == measured[capacity] and \
                    measured[capacity] >= 0.90:
                measured_knee = capacity
                break
        verdicts = []
        # classification
        total += 1
        class_hit = curve["reachable"] == measured_reachable
        hits += class_hit
        verdicts.append(("class", class_hit,
                         f"pred {'R' if curve['reachable'] else 'U'} "
                         f"meas {'R' if measured_reachable else 'U'}"))
        # plateau
        total += 1
        plateau_err = abs(curve["plateau"] - measured_plateau)
        plateau_hit = plateau_err <= 0.10
        hits += plateau_hit
        verdicts.append(("plateau", plateau_hit,
                         f"pred {100 * curve['plateau']:.0f} "
                         f"meas {100 * measured_plateau:.0f} "
                         f"(err {100 * plateau_err:.0f})"))
        # knee, only when both sides call it reachable
        if curve["reachable"] and measured_reachable:
            total += 1
            knee_hit = (
                curve["knee"] is not None
                and measured_knee is not None
                and abs(curve["knee"] - measured_knee) <= 1
            )
            hits += knee_hit
            verdicts.append(("knee", knee_hit,
                             f"pred {curve['knee']} meas {measured_knee}"))
        rows.append((curve, verdicts, measured))
        text = "  ".join(
            f"{name}:{'HIT ' if hit else 'MISS'} ({detail})"
            for name, hit, detail in verdicts
        )
        print(
            f"  N={curve['n']:<3}{curve['fleet']:<6} "
            f"B={curve['budget']:.2f} L={curve['latency']}  {text}"
        )
        curve_gains = "  ".join(
            f"{c}: {100 * curve['pred_gain'][c]:3.0f}/"
            f"{100 * measured[c]:3.0f}"
            for c in curve["capacities"]
        )
        print(f"      pred/meas per capacity: {curve_gains}")
    print(f"\n  BLIND SCORE: {hits}/{total} hits")
    return hits, total, rows


# ---------------------------------------------------------------------
# Step 3: pooled identity-line collapse (round 1 cache + round 2 runs)
# ---------------------------------------------------------------------

def pooled_collapse(records, runs):
    print("\n=== STEP 3: pooled model-vs-measurement collapse ===")
    pooled = []
    for rec in records:
        pooled.append((
            rec["n"], FLEETS[rec["fleet"]](rec["n"]), rec["fleet"],
            rec["seed"], rec["latency"], rec["budget"],
            rec["capacity"], rec["gain"], 50,
        ))
    for rec in runs:
        pooled.append((
            rec["n"], fleet_r2(rec["fleet"], rec["n"]), rec["fleet"],
            rec["seed"], rec["latency"], rec["budget"],
            rec["capacity"], rec["gain"], ITERATIONS,
        ))
    predicted, measured, meta = [], [], []
    started = time.time()
    for index, (n, profiles, fleet, seed, latency, budget, capacity,
                gain, iterations) in enumerate(pooled):
        prediction = predict_condition(
            n, profiles, seed, latency, budget, capacity, iterations
        )
        predicted.append(prediction["gain_pred"])
        measured.append(gain)
        meta.append((n, fleet, latency, budget))
        if (index + 1) % 100 == 0:
            print(f"  [{index + 1}/{len(pooled)}] "
                  f"{(time.time() - started) / 60.0:.1f} min", flush=True)
    predicted = np.array(predicted)
    measured = np.array(measured)
    identity_r2 = 1.0 - np.sum((measured - predicted) ** 2) / np.sum(
        (measured - np.mean(measured)) ** 2
    )
    _, isotonic_r2 = isotonic_fit(predicted, measured)
    print(
        f"\n  identity-line R^2 (G_meas vs G_pred, no fit): "
        f"{identity_r2:.4f}   [round-1 rho-only isotonic: 0.577]"
    )
    print(f"  isotonic R^2 on the G_pred axis: {isotonic_r2:.4f}")
    residuals = measured - predicted
    print(f"  residual: mean {100 * np.mean(residuals):+.1f} pts, "
          f"rms {100 * np.sqrt(np.mean(residuals ** 2)):.1f} pts")
    print("\n  -- residual structure (measured - predicted, mean) --")
    for axis, getter in (
        ("N", lambda m: m[0]),
        ("fleet", lambda m: m[1]),
        ("latency", lambda m: m[2]),
        ("budget", lambda m: m[3]),
    ):
        levels = sorted({getter(m) for m in meta}, key=str)
        parts = []
        for level in levels:
            idx = [i for i, m in enumerate(meta) if getter(m) == level]
            parts.append(
                f"{level}: {100 * float(np.mean(residuals[idx])):+5.1f}"
                f" (n={len(idx)})"
            )
        print(f"  {axis:<8} " + "   ".join(parts))
    return identity_r2, isotonic_r2


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reconcile", action="store_true")
    parser.add_argument("--pooled", action="store_true")
    args = parser.parse_args()

    records = load_cache()
    print(f"round-1 cache: {len(records)} runs ({CACHE_PATH})")

    if args.reconcile:
        reconcile(records)
        return
    if args.pooled:
        runs = (
            json.load(open(ROUND2_PATH)) if os.path.exists(ROUND2_PATH)
            else []
        )
        pooled_collapse(records, runs)
        return

    reconcile(records)
    assert_fresh(records)
    frozen = freeze_predictions()
    runs = measure_curves(frozen)
    score(frozen, runs)
    pooled_collapse(records, runs)


if __name__ == "__main__":
    main()
