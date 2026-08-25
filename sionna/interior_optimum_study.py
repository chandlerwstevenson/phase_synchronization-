"""The interior-optimal free-observation rate: mechanism, surface,
scaling (external review item: derive the optimum, don't just report
it).

Competing-terms model
---------------------
Free observations see only the SUM s = theta + psi (oscillator +
channel phase); anchors every K intervals see the split. Two error
channels compete as the observation rate n (per interval) grows:

  tracking term (falls with n): the sum is a random walk with
    per-interval process q_s observed n times per interval with
    per-observation noise r, and the loop actuates once per detected
    observation, so both the effective measurement variance per
    interval (r/n) and the actuation delay (T/n) shrink. Steady-state
    walk-observed-in-noise tracking variance scales as
        sigma_track^2 ~ sqrt(q_s * r / n)        [derived]
  split-wander term (grows with n*K in practice): the theta/psi split
    is unobserved between anchors. Idealized linear analysis says the
    per-free-update leakage into psi-hat has gain g_psi ~
    sqrt(q_psi/(n r)) and accumulates over m = n*K updates to
    K*q_psi - i.e. n-INDEPENDENT. The measured n-dependence therefore
    comes from what the linear model omits: each of the n*K quantized
    corrections between anchors re-injects actuation error that the
    sum observation immediately re-splits, and the wrapped-phase
    nonlinearity rectifies part of it into a LOCKED bias. We
    therefore fit the empirical exponents:
        sigma_split^2 ~ b^2 * (n*K)^gamma        [empirical exponent]
  total:  sigma^2(n, K) = a^2 * (q_s r)^(1/2) n^(-1/2) + b^2 (nK)^gamma
  optimum: d/dn = 0  ->  n* proportional to K^(-gamma/(gamma + 1/2))
    (for gamma = 1: n* ~ K^(-2/3); gamma = 1/2: n* ~ K^(-1/2)).

What is derived vs approximated: the n^(-1/2) tracking scaling is
standard random-walk-in-noise steady state (derived); the split term's
gamma is fit to the measured surface because the ideal linear
filter predicts no n-dependence at all (stated honestly - the
n-dependence is a nonlinear/actuation effect).

Measurements: residual rms, locked bias (circular mean of the tail),
and wander (circular std) per (n, K, seed); the bias-vs-K mechanism
check at fixed n above optimum.

Usage:
    .venv/bin/python interior_optimum_study.py --ks 10,20
    .venv/bin/python interior_optimum_study.py --report   # cache only
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from clutter_sync_ofdm import run_piggyback_star
from ota_sync import SDRSimulationConfig
from ota_sync.oscillators import resolve_oscillator_noise

CACHE = Path(__file__).resolve().parent / "interior_optimum_cache.json"
N_GRID = [1, 2, 3, 5, 8, 12, 20]
K_GRID = [10, 20, 40, 80, 160]
SEEDS = [0, 1, 2]


def intervals_for(k: int) -> int:
    """Every cell sees at least ~4 anchor cycles; with a flat 60
    intervals, any K > 59 degenerates to acquire-once-never-re-anchor
    and all such K are the same run (caught in the first sweep)."""

    return max(60, 4 * k)


def profiled(settings: SDRSimulationConfig, profile: str | None):
    if profile is None:
        return settings
    noise, _ = resolve_oscillator_noise(
        profile,
        settings.carrier_frequency_hz,
        settings.sample_rate,
        settings.sync_interval,
    )
    from dataclasses import replace

    return replace(settings, **noise)


def tail_stats(result) -> tuple[float, float, float]:
    """(rms, |circular-mean bias|, circular std) in mrad over the
    valid tail half of the run's residuals (single link, N=2)."""

    residuals = result.station_residuals[0]
    valid = result.station_valid[0]
    idx = torch.nonzero(valid).flatten()
    if idx.numel() < 4:
        return float("nan"), float("nan"), float("nan")
    tail = residuals[idx[idx.numel() // 2:]]
    rms = torch.sqrt(torch.mean(tail.square())).item()
    z = torch.exp(1j * tail.to(torch.complex128))
    mean_z = torch.mean(z)
    bias = abs(torch.angle(mean_z).item())
    resultant = min(max(torch.abs(mean_z).item(), 1e-12), 1.0)
    circ_std = math.sqrt(max(-2.0 * math.log(resultant), 0.0))
    return 1e3 * rms, 1e3 * bias, 1e3 * circ_std


def run_cell(n: int, k: int, seed: int, profile: str | None) -> dict:
    settings = profiled(
        SDRSimulationConfig(
            num_iterations=intervals_for(k), seed=seed, device="cpu"
        ),
        profile,
    )
    result = run_piggyback_star(
        settings,
        num_stations=2,
        anchor_every_intervals=k,
        obs_per_interval=n,
    )
    rms, bias, wander = tail_stats(result)
    return {
        "rms_mrad": rms,
        "bias_mrad": bias,
        "wander_mrad": wander,
        "detection": result.detection_rate,
    }


def load_cache() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return {}


def key(n: int, k: int, seed: int, profile: str | None) -> str:
    base = f"n{n}_K{k}_s{seed}_{profile or 'sdr-default'}"
    iv = intervals_for(k)
    return base if iv == 60 else f"{base}_i{iv}"


def fill(cache: dict, cells, profile=None) -> None:
    for n, k in cells:
        for seed in SEEDS:
            cell_key = key(n, k, seed, profile)
            if cell_key in cache:
                continue
            cache[cell_key] = run_cell(n, k, seed, profile)
            CACHE.write_text(json.dumps(cache, indent=1))
            print(
                f"  {cell_key}: rms {cache[cell_key]['rms_mrad']:.1f} "
                f"bias {cache[cell_key]['bias_mrad']:.1f} mrad",
                flush=True,
            )


def seed_mean(cache: dict, n: int, k: int, profile=None, field="rms_mrad"):
    values = [
        cache[key(n, k, seed, profile)][field]
        for seed in SEEDS
        if key(n, k, seed, profile) in cache
        and cache[key(n, k, seed, profile)][field] == cache[
            key(n, k, seed, profile)
        ][field]
    ]
    if not values:
        return float("nan")
    return sum(values) / len(values)


def fit_and_report(cache: dict) -> None:
    print("\n=== rms surface, mrad (rows n, cols K; mean over seeds) ===")
    header = f"{'n':>4} " + "".join(f"{k:>9}" for k in K_GRID)
    print(header)
    for n in N_GRID:
        row = "".join(
            f"{seed_mean(cache, n, k):>9.1f}" for k in K_GRID
        )
        print(f"{n:>4} {row}")

    print("\nmeasured optimum n*(K) (argmin of the seed-mean rms):")
    optima = {}
    for k in K_GRID:
        values = [(seed_mean(cache, n, k), n) for n in N_GRID]
        values = [(v, n) for v, n in values if v == v]
        if values:
            optima[k] = min(values)[1]
            print(f"  K={k:<4} n* = {optima[k]}")

    # Two-term fit: sigma^2 = A n^(-1/2) + B (nK)^gamma, log-space
    # grid search over gamma, closed-form A,B per gamma.
    best = None
    for gamma in [x / 20.0 for x in range(4, 41)]:
        rows = []
        targets = []
        for n in N_GRID:
            for k in K_GRID:
                v = seed_mean(cache, n, k)
                if v != v:
                    continue
                rows.append((n ** -0.5, (n * k) ** gamma))
                targets.append((v / 1e3) ** 2)
        X = torch.tensor(rows, dtype=torch.float64)
        y = torch.tensor(targets, dtype=torch.float64)
        sol = torch.linalg.lstsq(X, y.unsqueeze(1)).solution.flatten()
        pred = X @ sol
        ss_res = torch.sum((y - pred) ** 2).item()
        ss_tot = torch.sum((y - y.mean()) ** 2).item()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        if sol.min().item() > 0 and (best is None or r2 > best[0]):
            best = (r2, gamma, sol[0].item(), sol[1].item())
    if best:
        r2, gamma, a2, b2 = best
        print(
            f"\ntwo-term fit sigma^2 = A n^-0.5 + B (nK)^gamma: "
            f"gamma = {gamma:.2f}, R^2 = {r2:.3f}"
        )
        exponent = -gamma / (gamma + 0.5)
        print(
            f"implied optimum scaling n* ~ K^({exponent:.2f}); "
            f"predicted n*(K): "
            + ", ".join(
                f"K={k}: {((a2 / (2 * gamma * b2 * k ** gamma)) ** (1 / (gamma + 0.5))):.1f}"
                for k in K_GRID
            )
        )

    print("\n=== mechanism check: bias vs K at n=12 (locked offset) ===")
    for k in K_GRID:
        bias = seed_mean(cache, 12, k, field="bias_mrad")
        wander = seed_mean(cache, 12, k, field="wander_mrad")
        print(f"  K={k:<4} bias {bias:7.1f} mrad   wander {wander:7.1f}")

    print("\n=== oscillator-class axis: tcxo row at K=40 ===")
    for n in N_GRID:
        v = seed_mean(cache, n, 40, profile="tcxo")
        if v == v:
            print(f"  n={n:<3} rms {v:7.1f} mrad")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ks", type=str, default="")
    parser.add_argument("--tcxo", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    cache = load_cache()
    if args.report:
        fit_and_report(cache)
        return
    if args.tcxo:
        fill(cache, [(n, 40) for n in N_GRID], profile="tcxo")
    elif args.ks:
        ks = [int(v) for v in args.ks.split(",")]
        fill(cache, [(n, k) for k in ks for n in N_GRID])
    else:
        fill(cache, [(n, k) for k in K_GRID for n in N_GRID])
    fit_and_report(cache)


if __name__ == "__main__":
    main()
