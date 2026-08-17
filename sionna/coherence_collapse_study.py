"""Data-collapse test: is array coherence under sync contention governed
by ONE dimensionless number?

The candidate control parameter is the supply/demand ratio

    rho = capacity / sum_k min(1, T / tau_k)

where capacity is the channel's two-way exchanges per interval and
tau_k is station k's coast time from the error-floor formula

    0.5*(sigma_pn,ref^2 + sigma_pn,k^2) * f_s * tau
        + ( sigma_omega+_k * (tau + L*T) )^2  =  budget^2

with sigma_pn from the oscillator profile's datasheet ADEV conversion
and sigma_omega+_k the steady-state Kalman frequency-posterior std of
link k's own filter (computed ex ante by iterating the filter's exact
covariance recursion - the same Q and R the simulator builds - to its
fixed point; no fitted constants anywhere). A second, deployable
variant tau_trig uses the scheduler's actual service rule: coast until
the filter's own predicted phase std crosses the trigger threshold
(trigger_fraction * budget); it is likewise ex ante.

If the law holds, gain-vs-capacity curves that scatter across N, fleet
composition, correction latency, and budget must collapse onto a single
master curve in rho, with the knee at rho ~= 1. The test is quantified:
an isotonic (monotone, shape-free) master curve is fitted to ALL points
and its R^2 is compared against the same fit on the naive axis
capacity/(N-1); residuals are examined per swept axis - any axis whose
residuals stay structured did not collapse.

The rho computation is meant to live in `coast_law.py` (a sibling
deliverable); until that file exists this module carries its own
implementation of the same formula and marks results "provisional".

Nothing in ota_sync/ or detection/ is modified.

Usage:
    .venv/bin/python coherence_collapse_study.py --grid      # run/extend the grid (cached)
    .venv/bin/python coherence_collapse_study.py --analyze   # collapse analysis from cache
    .venv/bin/python coherence_collapse_study.py --quick     # micro smoke grid
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import replace

import numpy as np
import torch

from gating_study import evaluation_mask
from ota_sync import SDRSimulationConfig
from ota_sync.network import MAX_LINK_SNR_DB, place_stations
from ota_sync.oscillators import resolve_oscillator_noise
from ota_sync.scheduled import run_scheduled_star
from ota_sync.sdr import (
    _FlickerFrequencyNoise,
    _measurement_covariance,
    make_sync_preamble,
)

CACHE_PATH = os.path.join(os.path.dirname(__file__), "coherence_collapse_cache.json")

FLEETS = {
    # "custom" is the legacy default noise class (the repo's
    # best-characterized regime, ~45 mrad/interval pair drift);
    # "sdr" (bench-SDR TCXO, ADEV 5e-10) is demand-SATURATED at a
    # 314 mrad budget - its coast time is zero, so it pins per-station
    # demand at 1 and stresses the rho axis at the congested end.
    "custom": lambda n: ["custom"] * n,
    "sdr": lambda n: ["sdr"] * n,
    "tcxo": lambda n: ["tcxo"] * n,
    "ocxo": lambda n: ["ocxo"] * n,
    "mixed": lambda n: ["ocxo"]
    + [("ocxo", "tcxo", "sdr")[i % 3] for i in range(n - 1)],
}


# ---------------------------------------------------------------------
# Ex-ante coast time and supply/demand ratio (no fitted constants).
# Preferred source is the sibling coast_law.py; local fallback below
# implements the identical formula and is flagged provisional.
# ---------------------------------------------------------------------

try:  # pragma: no cover - exercised only once coast_law.py lands
    from coast_law import predict_coast_time, supply_demand_ratio  # noqa: F401

    COAST_LAW_SOURCE = "coast_law.py"
except Exception:
    COAST_LAW_SOURCE = "local (provisional until coast_law.py lands)"
    predict_coast_time = None
    supply_demand_ratio = None


def _link_noise_model(
    settings: SDRSimulationConfig,
    profiles: list[str],
    num_stations: int,
    station: int,
    device: torch.device,
    preamble,
):
    """Reconstruct link `station`'s exact EKF Q, R, and pair drift rate
    from the same expressions run_scheduled_star uses (scheduled.py) -
    datasheet profiles in, matrices out, no simulation involved."""

    ref_noise, _ = resolve_oscillator_noise(
        profiles[0],
        settings.carrier_frequency_hz,
        settings.sample_rate,
        settings.sync_interval,
    )
    slave_noise, _ = resolve_oscillator_noise(
        profiles[station],
        settings.carrier_frequency_hz,
        settings.sample_rate,
        settings.sync_interval,
    )
    ref = {**{f: getattr(settings, f) for f in (
        "phase_noise_std_rad", "phase_process_std_rad",
        "frequency_process_std_hz", "flicker_frequency_std_hz",
    )}, **ref_noise}
    slave = {**{f: getattr(settings, f) for f in (
        "phase_noise_std_rad", "phase_process_std_rad",
        "frequency_process_std_hz", "flicker_frequency_std_hz",
    )}, **slave_noise}

    positions = place_stations(num_stations, 500.0, settings.seed)
    distance = max(
        float(np.linalg.norm(positions[station] - positions[0])), 1.0
    )
    snr_db = min(
        settings.snr_db - 10.0 * 2.7 * math.log10(distance / 500.0),
        MAX_LINK_SNR_DB,
    )
    link_settings = replace(settings, snr_db=snr_db, **slave_noise)

    interval_samples = int(
        round(settings.sync_interval * settings.sample_rate)
    )
    white_fm = (
        0.5
        * (ref["phase_noise_std_rad"] ** 2 + slave["phase_noise_std_rad"] ** 2)
        * interval_samples
    )
    flicker = _FlickerFrequencyNoise(
        ref["flicker_frequency_std_hz"],
        settings.sync_interval,
        settings.num_iterations * settings.sync_interval,
        device,
        torch.Generator(device=device).manual_seed(0),
    )
    q_phase = (
        ref["phase_process_std_rad"] ** 2
        + slave["phase_process_std_rad"] ** 2
        + white_fm
    )
    q_freq = (
        (2.0 * math.pi * ref["frequency_process_std_hz"]) ** 2
        + (2.0 * math.pi * slave["frequency_process_std_hz"]) ** 2
        + flicker.innovation_variance
    )
    process = np.diag([q_phase, q_freq])
    measurement = (
        0.5 * _measurement_covariance(link_settings, preamble, device)
    ).numpy()
    drift_rate = (
        0.5
        * (ref["phase_noise_std_rad"] ** 2 + slave["phase_noise_std_rad"] ** 2)
        * settings.sample_rate
    )  # rad^2 per second of pair coast, half-difference geometry
    return process, measurement, drift_rate


def _steady_posterior(process, measurement, sync_interval):
    """Fixed point of the filter's own covariance recursion (predict +
    linearized update at phi=0) - the DARE steady state, iterated."""

    transition = np.array([[1.0, sync_interval], [0.0, 1.0]])
    jac = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    covariance = np.diag([math.pi**2, (2.0 * math.pi * 50e3) ** 2])
    for _ in range(2000):
        prior = transition @ covariance @ transition.T + process
        innovation = jac @ prior @ jac.T + measurement
        gain = np.linalg.solve(innovation, jac @ prior).T
        residual_map = np.eye(2) - gain @ jac
        posterior = (
            residual_map @ prior @ residual_map.T
            + gain @ measurement @ gain.T
        )
        if np.max(np.abs(posterior - covariance)) < 1e-16:
            covariance = posterior
            break
        covariance = posterior
    return covariance


def _coast_time_physical(drift_rate, sigma_omega, latency_s, budget):
    """Positive root of 0.5-drift + (sigma_omega*(tau+L*T))^2 = budget^2."""

    a = drift_rate
    b = sigma_omega
    if b <= 0.0:
        return max((budget**2) / max(a, 1e-30), 0.0)
    quad_a = b**2
    quad_b = a + 2.0 * b**2 * latency_s
    quad_c = (b * latency_s) ** 2 - budget**2
    if quad_c >= 0.0:
        return 0.0  # cannot hold the budget even with zero coast
    disc = quad_b**2 - 4.0 * quad_a * quad_c
    return (-quad_b + math.sqrt(disc)) / (2.0 * quad_a)


def _coast_time_trigger(
    process, posterior, sync_interval, trigger_std
):
    """Intervals the filter itself would coast: propagate its posterior
    predict-only until predicted phase std crosses the trigger."""

    transition = np.array([[1.0, sync_interval], [0.0, 1.0]])
    covariance = posterior.copy()
    for step in range(1, 10001):
        covariance = transition @ covariance @ transition.T + process
        if math.sqrt(max(covariance[0, 0], 0.0)) >= trigger_std:
            return step * sync_interval
    return 10000 * sync_interval


def compute_rhos(
    num_stations: int,
    fleet: str,
    seed: int,
    latency_intervals: int,
    budget: float,
    num_iterations: int = 50,
    trigger_fraction: float = 0.5,
):
    """Both ex-ante supply/demand ratios for one configuration, per unit
    capacity: multiply by the channel capacity to get rho."""

    settings = SDRSimulationConfig(
        num_iterations=num_iterations,
        seed=seed,
        device="cpu",
        correction_latency_intervals=latency_intervals,
    )
    device = torch.device("cpu")
    preamble = make_sync_preamble(settings, device)
    profiles = FLEETS[fleet](num_stations)
    interval = settings.sync_interval
    latency_s = latency_intervals * interval
    demand_phys = 0.0
    demand_trig = 0.0
    taus = []
    for station in range(1, num_stations):
        process, measurement, drift = _link_noise_model(
            settings, profiles, num_stations, station, device, preamble
        )
        posterior = _steady_posterior(process, measurement, interval)
        sigma_omega = math.sqrt(max(posterior[1, 1], 0.0))
        tau_phys = _coast_time_physical(drift, sigma_omega, latency_s, budget)
        tau_trig = _coast_time_trigger(
            process, posterior, interval, trigger_fraction * budget
        )
        taus.append((tau_phys, tau_trig, sigma_omega))
        demand_phys += min(1.0, interval / max(tau_phys, 1e-9))
        demand_trig += min(1.0, interval / max(tau_trig, 1e-9))
    return demand_phys, demand_trig, taus


# ---------------------------------------------------------------------
# Grid runner with incremental cache
# ---------------------------------------------------------------------

def _record_key(rec):
    return (
        rec["n"], rec["fleet"], rec["capacity"], rec["seed"],
        rec["latency"], round(rec["budget"], 6),
    )


def load_cache(path=CACHE_PATH):
    if os.path.exists(path):
        with open(path) as handle:
            return json.load(handle)
    return []


def save_cache(records, path=CACHE_PATH):
    with open(path, "w") as handle:
        json.dump(records, handle)


def run_cell(n, fleet, capacity, seed, latency, budget, iterations=50):
    settings = SDRSimulationConfig(
        num_iterations=iterations,
        seed=seed,
        device="cpu",
        correction_latency_intervals=latency,
    )
    result = run_scheduled_star(
        settings,
        num_stations=n,
        policy="scheduled",
        budgets_rad=[budget] * (n - 1),
        max_exchanges_per_interval=capacity,
        oscillator_profiles=FLEETS[fleet](n),
    )
    mask = evaluation_mask(result)
    gain = torch.mean(result.array_gain[mask]).item()
    rms = [
        torch.sqrt(torch.mean(row[mask].square())).item()
        for row in result.residuals
    ]
    demand_phys, demand_trig, taus = compute_rhos(
        n, fleet, seed, latency, budget, iterations
    )
    return {
        "n": n,
        "fleet": fleet,
        "capacity": capacity,
        "seed": seed,
        "latency": latency,
        "budget": budget,
        "gain": gain,
        "worst_rms": max(rms),
        "frac_met": float(np.mean([v <= budget for v in rms])),
        "airtime": result.airtime_used_fraction,
        "steady": bool(torch.any(result.steady)),
        "demand_phys": demand_phys,
        "demand_trig": demand_trig,
        "rho_phys": capacity / demand_phys if demand_phys > 0 else float("inf"),
        "rho_trig": capacity / demand_trig if demand_trig > 0 else float("inf"),
        "naive": capacity / (n - 1),
        "tau_phys_s": [t[0] for t in taus],
        "tau_trig_s": [t[1] for t in taus],
    }


def grid_cells(quick=False):
    cells = []
    if quick:
        for capacity in (1, 2):
            cells.append((3, "sdr", capacity, 0, 1, 0.314, 10))
        return cells
    capacities_by_n = {
        6: list(range(1, 6)),
        10: list(range(1, 10)),
        14: [1, 2, 3, 5, 7, 9, 11, 13],
    }
    seeds = (0, 1, 2)
    for n, caps in capacities_by_n.items():
        for fleet in FLEETS:
            for capacity in caps:
                for seed in seeds:
                    cells.append((n, fleet, capacity, seed, 1, 0.314, 50))
    for latency in (2, 4):  # latency axis, one fleet with a transition
        for capacity in range(1, 10):
            for seed in seeds:
                cells.append((10, "custom", capacity, seed, latency, 0.314, 50))
    for budget in (0.2, 0.6):  # budget axis, same fleet
        for capacity in range(1, 10):
            for seed in seeds:
                cells.append((10, "custom", capacity, seed, 1, budget, 50))
    return cells


def run_grid(quick=False, time_limit_s=None):
    records = load_cache()
    have = {_record_key(rec) for rec in records}
    cells = grid_cells(quick)
    todo = [
        cell for cell in cells
        if (cell[0], cell[1], cell[2], cell[3], cell[4], round(cell[5], 6))
        not in have
    ]
    print(
        f"grid: {len(cells)} cells, {len(cells) - len(todo)} cached, "
        f"{len(todo)} to run  (rho source: {COAST_LAW_SOURCE})"
    )
    started = time.time()
    for index, (n, fleet, capacity, seed, latency, budget, iters) in enumerate(
        todo
    ):
        if time_limit_s is not None and time.time() - started > time_limit_s:
            print(
                f"time limit reached with {len(todo) - index} cells left; "
                "re-run --grid to continue (cache resumes)"
            )
            break
        cell_start = time.time()
        rec = run_cell(n, fleet, capacity, seed, latency, budget, iters)
        records.append(rec)
        save_cache(records)
        print(
            f"[{index + 1}/{len(todo)}] N={n} {fleet:<5} cap={capacity:<2} "
            f"seed={seed} L={latency} budget={budget:.3f}  "
            f"gain={100 * rec['gain']:5.1f}%  rho_phys={rec['rho_phys']:5.2f} "
            f"rho_trig={rec['rho_trig']:5.2f}  ({time.time() - cell_start:.1f}s, "
            f"total {(time.time() - started) / 60.0:.1f} min)",
            flush=True,
        )
    return records


# ---------------------------------------------------------------------
# Collapse analysis
# ---------------------------------------------------------------------

def isotonic_fit(x, y):
    """Increasing isotonic regression via pool-adjacent-violators.
    Returns fitted values aligned with the sort order of x, plus R^2."""

    order = np.argsort(x, kind="stable")
    sorted_y = np.asarray(y, dtype=float)[order]
    values: list[float] = []
    weights: list[int] = []
    for value in sorted_y:
        values.append(float(value))
        weights.append(1)
        while len(values) > 1 and values[-2] > values[-1]:
            merged = weights[-1] + weights[-2]
            values[-2] = (
                values[-1] * weights[-1] + values[-2] * weights[-2]
            ) / merged
            weights[-2] = merged
            values.pop()
            weights.pop()
    fitted_sorted = np.concatenate(
        [np.full(w, v) for v, w in zip(values, weights)]
    )
    fitted = np.empty_like(fitted_sorted)
    fitted[order] = fitted_sorted
    total = np.sum((np.asarray(y) - np.mean(y)) ** 2)
    residual = np.sum((np.asarray(y) - fitted) ** 2)
    r_squared = 1.0 - residual / total if total > 0 else float("nan")
    return fitted, r_squared


def analyze(records):
    finite = [
        r for r in records
        if np.isfinite(r["rho_phys"]) and np.isfinite(r["rho_trig"])
    ]
    print(
        f"\n=== collapse analysis over {len(finite)} runs "
        f"(rho source: {COAST_LAW_SOURCE}) ==="
    )
    gain = np.array([r["gain"] for r in finite])
    axes = {
        "rho_phys": np.array([r["rho_phys"] for r in finite]),
        "rho_trig": np.array([r["rho_trig"] for r in finite]),
        "naive c/(N-1)": np.array([r["naive"] for r in finite]),
    }
    fits = {}
    print("\n-- master-curve R^2 per control parameter (isotonic fit) --")
    for name, axis in axes.items():
        fitted, r2 = isotonic_fit(axis, gain)
        fits[name] = (fitted, r2)
        print(f"  {name:<14} R^2 = {r2:.4f}")

    # Residual structure per swept axis, for the best rho variant.
    best = max(("rho_phys", "rho_trig"), key=lambda k: fits[k][1])
    residuals = gain - fits[best][0]
    print(f"\n-- residual structure of '{best}' fit (mean residual, pts) --")
    for axis_name, getter in (
        ("N", lambda r: r["n"]),
        ("fleet", lambda r: r["fleet"]),
        ("latency", lambda r: r["latency"]),
        ("budget", lambda r: r["budget"]),
    ):
        levels = sorted({getter(r) for r in finite}, key=str)
        parts = []
        for level in levels:
            idx = [i for i, r in enumerate(finite) if getter(r) == level]
            parts.append(
                f"{level}: {100 * float(np.mean(residuals[idx])):+5.1f}%"
                f" (n={len(idx)})"
            )
        print(f"  {axis_name:<8} " + "   ".join(parts))

    # Master-curve table on the best axis.
    edges = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 1e9]
    axis = axes[best]
    print(f"\n-- master curve: gain vs {best} (bin mean +- across-run std) --")
    plateau_points = gain[axis > 2.0]
    plateau = float(np.mean(plateau_points)) if plateau_points.size else float("nan")
    knee = None
    for low, high in zip(edges[:-1], edges[1:]):
        idx = (axis >= low) & (axis < high)
        if not np.any(idx):
            continue
        mean_gain = float(np.mean(gain[idx]))
        label = f"[{low:.2f},{high if high < 1e9 else float('inf'):.2f})"
        print(
            f"  rho {label:<14} gain {100 * mean_gain:5.1f}% "
            f"+- {100 * float(np.std(gain[idx])):4.1f}  (n={int(idx.sum())})"
        )
        if knee is None and plateau == plateau and mean_gain >= 0.9 * plateau:
            knee = 0.5 * (low + min(high, 10.0))
    print(
        f"  plateau (rho>2): {100 * plateau:.1f}%   "
        f"knee (first bin >= 90% of plateau): rho ~ {knee}"
    )

    # Transition sharpness vs N on the best axis (main grid only).
    print(f"\n-- transition width vs N ({best}, 20%->80% of rise) --")
    for n in sorted({r["n"] for r in finite}):
        widths = []
        for fleet in FLEETS:
            rows = [
                r for r in finite
                if r["n"] == n and r["fleet"] == fleet
                and r["latency"] == 1 and abs(r["budget"] - 0.314) < 1e-9
            ]
            if len({r["capacity"] for r in rows}) < 4:
                continue
            caps = sorted({r["capacity"] for r in rows})
            xs, ys = [], []
            for cap in caps:
                sub = [r for r in rows if r["capacity"] == cap]
                xs.append(float(np.mean([r[best] for r in sub])))
                ys.append(float(np.mean([r["gain"] for r in sub])))
            xs, ys = np.array(xs), np.array(ys)
            low_g, high_g = ys.min(), ys.max()
            if high_g - low_g < 0.05:
                continue  # no transition inside the swept range
            targets = [low_g + f * (high_g - low_g) for f in (0.2, 0.8)]
            crossings = []
            for target in targets:
                crossing = None
                for i in range(len(xs) - 1):
                    y0, y1 = ys[i], ys[i + 1]
                    if (y0 - target) * (y1 - target) <= 0 and y1 != y0:
                        crossing = xs[i] + (target - y0) * (
                            xs[i + 1] - xs[i]
                        ) / (y1 - y0)
                        break
                crossings.append(crossing)
            if None not in crossings:
                widths.append(crossings[1] - crossings[0])
        if widths:
            print(
                f"  N={n:<3} width {float(np.mean(widths)):.3f} "
                f"+- {float(np.std(widths)):.3f}  (curves: {len(widths)})"
            )
        else:
            print(f"  N={n:<3} no measurable transition in swept range")
    return fits


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--time-limit", type=float, default=None,
                        help="stop the grid cleanly after this many seconds")
    args = parser.parse_args()
    if args.quick:
        records = run_grid(quick=True)
        print(json.dumps(records[-2:], indent=1)[:800])
        return
    if args.grid:
        run_grid(time_limit_s=args.time_limit)
    if args.analyze or not args.grid:
        analyze(load_cache())


if __name__ == "__main__":
    main()
