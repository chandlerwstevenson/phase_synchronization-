"""Score the membership family on five system metrics and compare.

The question: does the RANKING of membership methods change with the
metric you score by? Metrics (see metrics.py): probability of
detection, spectral efficiency (mean and 95%-likely), beam quality
(mean array gain), detection range in meters, and net throughput
((1 - sync airtime) x mean spectral efficiency).

Methods, all scored on the SAME synchronization runs:
  all-in       every station participates everywhere
  post-gate    bench stations whose phase-uncertainty posterior
               exceeds 90 degrees
  1-bit        bench on the per-station alignment feedback bit
               (error-free bit = the oracle gate, an exact identity);
               also scored with 10% bit errors
  hybrid       coherent core = posterior members; benched stations
               demoted to a second tier (square-law receive fusion for
               detection; second data stream with successive
               interference cancellation for communication)

Conventions: detection uses the receive-combiner comparison (transmit
all-in for every method — the combiner-only convention of
hybrid_combiner_study.py). Communication weights scale transmit (a
benched station does not send the user's data; hybrid's demoted group
sends a second stream). Detection range uses each method's
transmit-weighted coherent gain (conservative for hybrid: its
noncoherent tier is not credited).

Usage:
    .venv/bin/python metrics_membership_study.py
    .venv/bin/python metrics_membership_study.py --quick
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from detection import DetectionParams
from gating_study import (
    evaluation_mask,
    phase_matrix,
    posterior_gate_weights,
    run_star_with_posteriors,
)
from metrics import (
    detection_range_m,
    mean_array_gain,
    net_throughput,
    probability_of_detection,
    spectral_efficiency,
)
from opportunistic_membership_study import alignment_bits, onebit_weights

METHODS = ("all-in", "post-gate", "1-bit", "1-bit-10%err", "hybrid")


def method_weights(
    name: str,
    phases: torch.Tensor,
    sigma: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(coherent weights, optional noncoherent-tier weights)."""

    if name == "all-in":
        return torch.ones_like(phases), None
    if name == "post-gate":
        return (
            posterior_gate_weights(sigma, phases.shape[0], math.pi / 2.0),
            None,
        )
    if name == "1-bit":
        return onebit_weights(alignment_bits(phases, 0.0, None)), None
    if name == "1-bit-10%err":
        generator = torch.Generator().manual_seed(seed + 777)
        return (
            onebit_weights(alignment_bits(phases, 0.1, generator)),
            None,
        )
    if name == "hybrid":
        coherent = posterior_gate_weights(
            sigma, phases.shape[0], math.pi / 2.0
        )
        tier = 1.0 - coherent
        tier[0] = 0.0  # the reference is always in the coherent core
        return coherent, tier
    raise ValueError(f"unknown method '{name}'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="membership family scored on five system metrics"
    )
    parser.add_argument("--stations", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--capacity", type=int, default=2)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--tx-power", type=float, default=0.5)
    parser.add_argument(
        "--detect-powers", type=str, default="0.5,0.05",
        help="per-station transmit powers (W) for the detection metric; "
        "0.5 W saturates at the edge targets, 0.05 W separates methods",
    )
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--h0-trials", type=int, default=12000)
    parser.add_argument("--quick", action="store_true",
                        help="1 seed, fewer trials")
    args = parser.parse_args()

    n = args.stations
    seeds = [0] if args.quick else [int(s) for s in args.seeds.split(",")]
    trials = 120 if args.quick else args.trials
    h0_trials = 4000 if args.quick else args.h0_trials
    params = DetectionParams(tx_power_w=args.tx_power)
    detect_powers = [float(p) for p in args.detect_powers.split(",")]

    # policy label -> extra run_scheduled_star kwargs
    policies = {
        "uniform-contended": dict(
            policy="uniform", max_exchanges_per_interval=args.capacity
        ),
        "scheduled-control": dict(policy="scheduled"),
    }

    for policy_label, policy_kwargs in policies.items():
        # metric accumulators: method -> list over seeds
        acc: dict[str, dict[str, list]] = {
            m: {
                "gain": [], "se_edge": [], "se_edge95": [],
                "se_near": [], "se_near95": [], "range": [], "net": [],
                **{f"pd@{p}": [] for p in detect_powers},
            }
            for m in METHODS
        }
        airtimes = []
        for seed in seeds:
            from ota_sync import SDRSimulationConfig

            settings = SDRSimulationConfig(
                num_iterations=args.iterations, seed=seed, device="cpu"
            )
            result, sigma_full = run_star_with_posteriors(
                settings, num_stations=n, **policy_kwargs
            )
            airtimes.append(result.airtime_used_fraction)
            mask = evaluation_mask(result)
            phases = phase_matrix(result)[:, mask]
            sigma = sigma_full[:, mask]
            positions = result.positions
            centroid = positions.mean(axis=0)
            edge_targets = np.array(
                [centroid + [1200.0, 150.0], centroid + [-1200.0, 150.0]]
            )
            near_user = centroid + np.array([400.0, 0.0])

            detection_cache: dict[bytes, list[float]] = {}
            for name in METHODS:
                weights, tier = method_weights(name, phases, sigma, seed)

                gain = mean_array_gain(phases, weights)
                acc[name]["gain"].append(gain)
                acc[name]["range"].append(
                    detection_range_m(n, gain, params)
                )

                # Detection: transmit all-in, per-method receive combiner.
                combiner = (
                    "two-tier-noncoherent" if tier is not None
                    else "two-tier-discard"
                )
                for power in detect_powers:
                    key = (
                        f"{combiner}@{power}".encode()
                        + weights.to(torch.uint8).numpy().tobytes()
                    )
                    if key not in detection_cache:
                        detect = probability_of_detection(
                            f"{policy_label}/{name}@s{seed}/{power}W",
                            positions, phases, weights, edge_targets,
                            combiner=combiner,
                            params=DetectionParams(tx_power_w=power),
                            trials=trials, h0_trials=h0_trials, seed=seed,
                        )
                        detection_cache[key] = list(detect.pd_measured)
                    acc[name][f"pd@{power}"].append(detection_cache[key])

                # Communication: weights on transmit; hybrid uses the
                # second-stream tier with successive interference
                # cancellation.
                se_edge = spectral_efficiency(
                    phases, weights, positions, edge_targets[0],
                    args.tx_power, noncoherent_weights=tier,
                )
                se_near = spectral_efficiency(
                    phases, weights, positions, near_user,
                    args.tx_power, noncoherent_weights=tier,
                )
                acc[name]["se_edge"].append(se_edge.mean_bps_hz)
                acc[name]["se_edge95"].append(se_edge.likely95_bps_hz)
                acc[name]["se_near"].append(se_near.mean_bps_hz)
                acc[name]["se_near95"].append(se_near.likely95_bps_hz)
                acc[name]["net"].append(
                    net_throughput(
                        se_near.mean_bps_hz, result.airtime_used_fraction
                    )
                )

        airtime = float(np.mean(airtimes))
        print(
            f"\n=== {policy_label}: N={n}, "
            f"{'capacity ' + str(args.capacity) + '/' + str(n - 1) if 'uniform' in policy_label else 'demand-driven'}, "
            f"seeds {seeds}, sync airtime {100 * airtime:.1f}%, "
            f"transmit power {args.tx_power} W ==="
        )
        detect_headers = "".join(
            f"{'detect@' + str(p) + 'W':>16}" for p in detect_powers
        )
        header = (
            f"{'method':<13}{'beam quality':>13}{detect_headers}"
            f"{'thru edge':>11}{'thru edge 95%':>15}{'thru near':>11}"
            f"{'range (m)':>11}{'net thru':>10}"
        )
        print(header)
        for name in METHODS:
            gain = np.mean(acc[name]["gain"])
            detect_cells = ""
            for power in detect_powers:
                pd = np.array(acc[name][f"pd@{power}"]).mean(axis=0)
                detect_cells += f"{100 * pd[0]:>9.1f}/{100 * pd[1]:.1f}%"
            se_e = np.mean(acc[name]["se_edge"])
            se_e95 = np.mean(acc[name]["se_edge95"])
            se_n = np.mean(acc[name]["se_near"])
            rng = np.mean(acc[name]["range"])
            net = np.mean(acc[name]["net"])
            print(
                f"{name:<13}{100 * gain:>12.1f}%{detect_cells}"
                f"{se_e:>11.2f}{se_e95:>15.2f}{se_n:>11.2f}"
                f"{rng:>11.0f}{net:>10.2f}"
            )
        print(
            "(beam quality = mean array gain vs perfect; detect = "
            "probability of detection at the two edge targets, "
            "receive-combiner convention; thru = spectral efficiency in "
            "bits/s/Hz, mean and 95%-likely, at the 1.2 km edge and the "
            "400 m user; range = detection range from the link budget; "
            "net thru = (1 - sync airtime) x near-user throughput)"
        )


if __name__ == "__main__":
    main()
