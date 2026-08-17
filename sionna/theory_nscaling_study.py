"""N-scaling of the theory stack.

Three questions, one per part:

  part 2 (this file's main job): does the frozen round-2 phase-diagram
     model EXTRAPOLATE in array size? Blind predictions for N in
     {16, 20} - beyond every N the model was ever tested on - in the
     clean regime (latency 1, feasible fleets), scored with round-2's
     own bands. Predictions printed before any measurement; fresh
     seeds; runs cached in a separate file so the round-2 cache is
     never touched.

  part 3: is the ~100 mrad per-exchange multipath-resampling noise a
     per-LINK constant (as the floor accounting assumes) or does it
     grow with N? Reuses doppler_coast_study's outside-in
     instrumentation at N in {6, 10, 14}, static, dense service.

  (part 1, coast-law exactness vs N, needs no new code: coast_law.py
  already takes --stations; run its CLI at N = 6/10/14 on the same
  sub-grid and compare the exact-match lines.)

Usage:
    .venv/bin/python theory_nscaling_study.py --part 2
    .venv/bin/python theory_nscaling_study.py --part 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from gating_study import evaluation_mask
from ota_sync import SDRSimulationConfig
from ota_sync.scheduled import run_scheduled_star
from phase_diagram_round2 import (
    ITERATIONS,
    curve_capacities,
    fleet_r2,
    predict_condition,
    predict_curve,
    score,
)

NS_PATH = os.path.join(
    os.path.dirname(__file__), "theory_nscaling_runs.json"
)
SEEDS = (3, 4, 5)

# Clean regime only (latency 1, feasible fleets): the two known
# failure modes (sdr1 poisoning, latency >= 2) are already
# characterized and would only obscure the N question.
CURVES = [
    (16, "tcxo", 0.25, 1),
    (16, "tcxo", 0.45, 1),
    (16, "ocxo", 0.25, 1),
    (16, "ocxo", 0.45, 1),
    (20, "tcxo", 0.25, 1),
    (20, "tcxo", 0.45, 1),
    (20, "ocxo", 0.25, 1),
    (20, "ocxo", 0.45, 1),
]


def freeze_nscaling_predictions(curves=CURVES, seeds=SEEDS):
    """Round-2 freeze protocol on the N-extension curves."""

    frozen = []
    print(
        "\n=== STEP 1: FROZEN PREDICTIONS for the N extension "
        "(printed before any measurement) ==="
    )
    print(
        "bands, fixed now (round-2's): plateau +-10 pts; knee +-1 "
        "capacity; reachable = plateau >= 90%"
    )
    for n, fleet, budget, latency in curves:
        profiles = fleet_r2(fleet, n)
        scan = sorted(set(range(1, n)))
        per_cap, plateau, knee = predict_curve(
            n, profiles, seeds, latency, budget, scan, ITERATIONS
        )
        capacities = curve_capacities(n, knee)
        diagnostics = predict_condition(
            n, profiles, seeds[0], latency, budget, max(capacities),
            ITERATIONS,
        )
        frozen.append({
            "n": n, "fleet": fleet, "budget": budget,
            "latency": latency, "capacities": capacities,
            "pred_gain": {c: per_cap[c] for c in capacities},
            "plateau": plateau, "knee": knee,
            "reachable": plateau >= 0.90,
            "phi_max": diagnostics["phi_max"],
        })
        gains = " ".join(
            f"{c}:{100 * per_cap[c]:.0f}" for c in capacities
        )
        print(
            f"  N={n:<3}{fleet:<6} B={budget:.2f} L={latency}  "
            f"knee={'-' if knee is None else knee:<3} "
            f"plateau={100 * plateau:5.1f}% "
            f"{'REACHABLE' if plateau >= 0.90 else 'UNREACHABLE':<12} "
            f"phi_max={diagnostics['phi_max']:.2f}  gains[{gains}]"
        )
    return frozen


def load_runs(path=NS_PATH):
    if os.path.exists(path):
        with open(path) as handle:
            return json.load(handle)
    return []


def measure_nscaling(frozen, seeds=SEEDS, path=NS_PATH):
    runs = load_runs(path)
    have = {
        (r["n"], r["fleet"], r["capacity"], r["seed"], r["latency"],
         round(r["budget"], 6))
        for r in runs
    }
    todo = []
    for curve in frozen:
        for capacity in curve["capacities"]:
            for seed in seeds:
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
        with open(path, "w") as handle:
            json.dump(runs, handle)
        if (index + 1) % 15 == 0 or index + 1 == len(todo):
            print(
                f"  [{index + 1}/{len(todo)}] "
                f"{(time.time() - started) / 60.0:.1f} min",
                flush=True,
            )
    return runs


def run_part2() -> None:
    frozen = freeze_nscaling_predictions()
    runs = measure_nscaling(frozen)
    score(frozen, runs)


def run_part3(station_counts=(6, 10, 14), seeds=(0, 1),
              iterations=60) -> None:
    """Per-exchange reciprocity (resampling) noise vs array size:
    dense service (every link every interval), static channel."""

    from doppler_coast_study import (
        bias_structure,
        reciprocity_bias_series,
        run_star_instrumented,
    )

    print(
        "\n=== resampling noise vs N (static, dense service, "
        f"{iterations} intervals, seeds {list(seeds)}) ==="
    )
    print(
        f"{'N':>3} {'per-link sigma_b (mrad)':>24} "
        f"{'structure D(1)':>15} {'D(2)':>8} {'D(3)':>8}"
    )
    for n in station_counts:
        sigmas, d1s, d2s, d3s = [], [], [], []
        for seed in seeds:
            settings = SDRSimulationConfig(
                num_iterations=iterations, seed=seed, device="cpu"
            )
            result, tape = run_star_instrumented(
                settings,
                num_stations=n,
                policy="uniform",
                max_exchanges_per_interval=n - 1,
            )
            series = reciprocity_bias_series(result, tape, settings)
            sigma_b, structure = bias_structure(series)
            sigmas.append(sigma_b)
            # structure values are variances (rad^2); store their
            # square roots so everything below is in radians
            d1s.append(math.sqrt(structure.get(1, float("nan"))))
            d2s.append(math.sqrt(structure.get(2, float("nan"))))
            d3s.append(math.sqrt(structure.get(3, float("nan"))))

        def rms_mrad(values):
            values = [v for v in values if v == v]
            if not values:
                return float("nan")
            return 1e3 * float(np.sqrt(np.mean(np.square(values))))

        print(
            f"{n:>3} {rms_mrad(sigmas):>20.1f}     "
            f"{rms_mrad(d1s):>11.1f} mr {rms_mrad(d2s):>5.1f} mr "
            f"{rms_mrad(d3s):>5.1f} mr"
        )
    print(
        "(sigma_b = per-link std of the two-way measurement's channel "
        "bias, pooled; D(gap) = half the mean squared bias change "
        "between services that many intervals apart, in mrad after "
        "square root - flat in gap = white per-capture noise)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="N-scaling of the theory stack"
    )
    parser.add_argument("--part", type=int, choices=(2, 3), default=2)
    parser.add_argument("--iterations", type=int, default=60,
                        help="part 3 only")
    args = parser.parse_args()
    if args.part == 2:
        run_part2()
    else:
        run_part3(iterations=args.iterations)


if __name__ == "__main__":
    main()
