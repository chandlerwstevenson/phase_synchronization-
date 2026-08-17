"""Opportunistic membership from 1-bit alignment feedback.

gating_study.py showed a large gap between what the Kalman POSTERIOR
can gate (9% mean gain under uniform starvation) and what per-interval
TRUE-phase membership earns (oracle/greedy 26-31%): the missing
information is instantaneous ALIGNMENT, which no posterior can supply
for a starved station. This study asks how much of that gap survives
when the alignment information is reduced to what a deployed array
could actually feed back: ONE BIT per station per interval.

The bit is sign(cos(theta_k)) - "is my carrier within 90 degrees of
the reference right now" - i.e. exactly the observable the repo's
existing periodic pi-branch combining check already models from true
state (same fidelity standard: a beacon/feedback measurement the
array needs anyway for branch calibration). The bit PIGGYBACKS on
that beacon; its cost is 1 bit per station per interval on a side
channel, not a sync exchange. We charge it honestly by reporting it
next to the airtime a two-way exchange costs.

Membership policies (bookkeeping on the same recorded run; nothing in
ota_sync/ or detection/ is modified):

  all-in      status quo
  posterior   gating_study's sigma > pi/2 posterior gate
  oracle      gating_study's true-|theta| > pi/2 gate
  greedy      per-interval genie subset (upper bound)
  onebit      in iff the alignment bit says aligned; bit flips i.i.d.
              with probability eps (feedback error)
  onebit-hys  same bit, but a benched station needs 2 consecutive
              aligned bits to re-enter (anti-chatter hysteresis)
  quant2      2-bit quadrant of |theta|, mapped to quantized cosine
              amplitude weights cos(pi/8), cos(3pi/8), 0, 0 - the
              2-bit version of ideal cosine weighting

With eps = 0 the onebit rule IS the oracle gate (sign(cos theta) >= 0
iff |wrap(theta)| <= pi/2) - verified exactly in the tests. So the
"oracle" of gating_study.py is not a genie at all: it is implementable
with one feedback bit.

Usage:
    .venv/bin/python opportunistic_membership_study.py
    .venv/bin/python opportunistic_membership_study.py --no-detect
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from detection import DetectionParams
from gating_study import (
    evaluation_mask,
    greedy_oracle_weights,
    oracle_gate_weights,
    phase_matrix,
    posterior_gate_weights,
    run_gated_waveform_detection,
    run_star_with_posteriors,
    summarize,
    weighted_gain,
)
from ota_sync import SDRSimulationConfig
from ota_sync.core import wrap_phase

# 2-bit quadrant -> amplitude weight: the quantized cosine of the
# quadrant midpoint, clipped at zero (a station past 90 degrees is
# benched, not phase-inverted - amplitude-only control).
QUADRANT_WEIGHTS = (
    math.cos(math.pi / 8.0),
    math.cos(3.0 * math.pi / 8.0),
    0.0,
    0.0,
)


def alignment_bits(
    phases: torch.Tensor, eps: float, generator: torch.Generator | None
) -> torch.Tensor:
    """Per-station/interval 1-bit feedback: True = 'within 90 deg of
    the reference'. eps flips each bit i.i.d. (feedback error). Row 0
    (the reference datum) is always aligned."""

    aligned = torch.abs(wrap_phase(phases)) <= math.pi / 2.0
    if eps > 0.0:
        if generator is None:
            raise ValueError("bit errors need a generator")
        flips = (
            torch.rand(
                phases.shape, generator=generator, dtype=torch.float64
            )
            < eps
        )
        aligned = aligned ^ flips
    aligned[0] = True
    return aligned


def onebit_weights(bits: torch.Tensor) -> torch.Tensor:
    weights = bits.to(torch.float64)
    weights[0] = 1.0
    return weights


def onebit_hysteresis_weights(
    bits: torch.Tensor, reentry: int = 2
) -> torch.Tensor:
    """Bench on the first misaligned bit; re-enter only after
    ``reentry`` consecutive aligned bits."""

    num_stations, intervals = bits.shape
    weights = torch.ones(num_stations, intervals, dtype=torch.float64)
    for station in range(1, num_stations):
        member = True
        streak = 0
        for t in range(intervals):
            streak = streak + 1 if bool(bits[station, t]) else 0
            if member and not bool(bits[station, t]):
                member = False
            elif not member and streak >= reentry:
                member = True
            weights[station, t] = 1.0 if member else 0.0
    return weights


def quantized2_weights(phases: torch.Tensor) -> torch.Tensor:
    """2-bit |theta| quadrant -> quantized cosine amplitude weight."""

    quadrant = torch.clamp(
        (torch.abs(wrap_phase(phases)) / (math.pi / 4.0)).floor(),
        max=3.0,
    ).to(torch.int64)
    table = torch.tensor(QUADRANT_WEIGHTS, dtype=torch.float64)
    weights = table[quadrant]
    weights[0] = 1.0
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(
        description="1-bit opportunistic membership on the contended star"
    )
    parser.add_argument("--stations", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--capacity", type=int, default=2)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--eps", type=str, default="0,0.1,0.2")
    parser.add_argument("--gate", type=float, default=math.pi / 2.0)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--h0-trials", type=int, default=15000)
    parser.add_argument("--no-detect", action="store_true")
    args = parser.parse_args()

    n = args.stations
    seeds = [int(s) for s in args.seeds.split(",")]
    eps_grid = [float(e) for e in args.eps.split(",")]

    print(
        f"1-bit opportunistic membership, N={n} star, uniform policy, "
        f"capacity {args.capacity}/{n - 1}, {args.iterations} intervals, "
        f"seeds {seeds}"
    )

    runs = {}
    for seed in seeds:
        settings = SDRSimulationConfig(
            num_iterations=args.iterations, seed=seed, device="cpu"
        )
        runs[seed] = run_star_with_posteriors(
            settings,
            num_stations=n,
            policy="uniform",
            budgets_rad=[0.314] * (n - 1),
            max_exchanges_per_interval=args.capacity,
        )

    # feedback-vs-airtime cost statement
    result0, _ = runs[seeds[0]]
    exchange_airtime = result0.airtime_uniform_fraction / (n - 1)
    print(
        f"feedback cost: {n - 1} bits/interval total (piggybacks on the "
        "branch-calibration beacon the array already requires); one full "
        f"two-way sync exchange costs {100 * exchange_airtime:.2f}% "
        "airtime per interval - the bit is not free but it is not an "
        "exchange either"
    )

    policies: list[tuple[str, float | None]] = [
        ("all-in", None),
        ("posterior", None),
        ("greedy", None),
        ("oracle", None),
    ]
    for eps in eps_grid:
        policies.append(("onebit", eps))
        policies.append(("onebit-hys", eps))
    policies.append(("quant2", None))

    def weights_for(policy, eps, phases, sigma, seed):
        generator = (
            torch.Generator().manual_seed(1000 + 17 * seed)
            if eps and eps > 0.0
            else None
        )
        if policy == "all-in":
            return torch.ones_like(phases)
        if policy == "posterior":
            return posterior_gate_weights(sigma, phases.shape[0], args.gate)
        if policy == "oracle":
            return oracle_gate_weights(phases, math.pi / 2.0)
        if policy == "greedy":
            return greedy_oracle_weights(phases)
        if policy == "onebit":
            return onebit_weights(alignment_bits(phases, eps or 0.0, generator))
        if policy == "onebit-hys":
            return onebit_hysteresis_weights(
                alignment_bits(phases, eps or 0.0, generator)
            )
        if policy == "quant2":
            return quantized2_weights(phases)
        raise ValueError(policy)

    stats = {}
    for policy, eps in policies:
        per_seed = []
        for seed in seeds:
            result, sigma = runs[seed]
            mask = evaluation_mask(result)
            phases = phase_matrix(result)[:, mask]
            weights = weights_for(policy, eps, phases, sigma[:, mask], seed)
            per_seed.append(
                summarize(weighted_gain(phases, weights))
            )
        stats[(policy, eps)] = np.array(per_seed).mean(axis=0)

    allin_gain = stats[("all-in", None)][0]
    oracle_gain = stats[("oracle", None)][0]
    gap = oracle_gain - allin_gain

    print(
        f"\n=== mean over seeds {seeds} (evaluation window = tail "
        "quarter; gap fraction = share of the oracle-minus-all-in mean-"
        "gain gap captured) ==="
    )
    print(
        f"{'policy':<16} {'G%':>7} {'G²%':>7} {'p5%':>7} {'gap frac':>9}"
    )
    for policy, eps in policies:
        mean_gain, mean_sq, p5 = stats[(policy, eps)]
        name = policy if eps is None else f"{policy}@{eps:g}"
        fraction = (mean_gain - allin_gain) / gap if gap > 0 else float("nan")
        print(
            f"{name:<16} {100 * mean_gain:7.1f} {100 * mean_sq:7.1f} "
            f"{100 * p5:7.1f} {fraction:9.2f}"
        )

    if args.no_detect:
        return

    # ---- counted edge detection, seed 0 ----------------------------
    detect_seed = seeds[0]
    result, sigma = runs[detect_seed]
    mask = evaluation_mask(result)
    phases = phase_matrix(result)[:, mask]
    sig = sigma[:, mask]
    positions = result.positions
    centroid = positions.mean(axis=0)
    edge_targets = np.array(
        [centroid + [1200.0, 150.0], centroid + [-1200.0, 150.0]]
    )
    params = DetectionParams(tx_power_w=0.5)
    detect_policies = [
        ("all-in", None),
        ("posterior", None),
        ("oracle", None),
        ("onebit", 0.1),
        ("onebit-hys", 0.1),
        ("quant2", None),
    ]
    print(
        f"\n=== counted edge detection (seed {detect_seed}, "
        f"{args.trials} trials/target, per-variant H0 threshold) ==="
    )
    print(f"{'policy':<16} {'edge Pd':>16}")
    for policy, eps in detect_policies:
        weights = weights_for(policy, eps, phases, sig, detect_seed)
        detect = run_gated_waveform_detection(
            f"uniform/{policy}",
            positions,
            phases,
            weights,
            edge_targets,
            params=params,
            trials=args.trials,
            h0_trials=args.h0_trials,
            seed=detect_seed,
        )
        name = policy if eps is None else f"{policy}@{eps:g}"
        print(
            f"{name:<16} "
            + " ".join(f"{100 * pd:6.1f}%" for pd in detect.pd_measured)
        )


if __name__ == "__main__":
    main()
