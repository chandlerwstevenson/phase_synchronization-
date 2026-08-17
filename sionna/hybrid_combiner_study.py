"""Hybrid coherent/noncoherent combining for posterior-benched stations.

Motivated by the cell-free massive MIMO hybrid-transmission idea (Qin
et al. 2024: misaligned transmitters are demoted to non-coherent
operation instead of being dropped): our posterior gate wins counted
detection by discarding stale receivers, but it throws their echo
power away with the fade risk. Here the benched stations keep
contributing NON-coherently:

    S = |sum_{j in coherent set} cpi_j|^2 / P
        + sum_{j benched} |cpi_j|^2 / P          (square-law fusion)

with per-station CPI sums cpi_j (a station is coherent with ITSELF
over one CPI, so per-station pulse integration stays coherent; only
the cross-station combining of benched stations is noncoherent). With
nobody benched this is exactly the coherent pipeline; with everybody
benched it is plain noncoherent fusion.

Transmit side is ALL-IN for every variant: benching is a receive-side
combiner decision here (the stations radiate their sensing bursts
regardless), so this study isolates the COMBINER — numbers are not
directly comparable to gating_study.py's headline, where the gate also
scaled transmit amplitudes.

The H0 threshold is recalibrated per variant for the SAME two-tier
statistic and membership distribution (the noncoherent terms change
the H0 law). Single-pulse calibration remains valid: for noise,
|sum_P n|^2 / P is distributed exactly as a single-pulse |n|^2, per
tier.

Nothing existing is modified; membership and star machinery come from
gating_study.py.

Usage:
    .venv/bin/python hybrid_combiner_study.py                # capacity 2
    .venv/bin/python hybrid_combiner_study.py --capacity 3 \
        --variants all-in,gate-discard,hybrid-post
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from detection import DetectionParams
from detection.viability import BOLTZMANN_T0
from detection.waveform import _ofdm_burst, _zadoff_chu
from gating_study import (
    SimpleDetectionResult,
    evaluation_mask,
    oracle_gate_weights,
    phase_matrix,
    posterior_gate_weights,
    run_star_with_posteriors,
)
from ota_sync import SDRSimulationConfig

BENCH_MODES = ("discard", "noncoherent")

# variant name -> (membership source, bench mode)
VARIANTS = {
    "all-in": ("ones", "discard"),
    "gate-discard": ("posterior", "discard"),
    "hybrid-post": ("posterior", "noncoherent"),
    "hybrid-oracle": ("oracle", "noncoherent"),
    "noncoh-all": ("zeros", "noncoherent"),
}


def run_hybrid_waveform_detection(
    label: str,
    positions: np.ndarray,
    residual_phases: torch.Tensor,
    rx_weights: torch.Tensor,
    bench_mode: str,
    targets_m: np.ndarray,
    params: DetectionParams = DetectionParams(),
    pulse_length: int = 1023,
    trials: int = 2000,
    h0_trials: int = 60000,
    threshold_pfa: float = 1e-3,
    seed: int = 0,
    waveform: str = "ofdm",
    leg_gains: np.ndarray | None = None,
) -> SimpleDetectionResult:
    """Counted detection with a two-tier receive combiner.

    ``rx_weights`` (stations, columns) in {0,1}: 1 = coherent set,
    0 = benched. ``bench_mode``: "discard" drops benched streams
    entirely (the pure gate); "noncoherent" adds them square-law.
    Transmit is all-in regardless. Draw order mirrors
    gating_study.run_gated_waveform_detection so that unit weights +
    "discard" reproduces the original pipeline exactly.
    """

    if bench_mode not in BENCH_MODES:
        raise ValueError(f"bench_mode must be one of {BENCH_MODES}")
    if rx_weights.shape != residual_phases.shape:
        raise ValueError("rx_weights must align with residual_phases")

    generator = torch.Generator().manual_seed(seed)
    weight_generator = torch.Generator().manual_seed(seed + 987654321)
    num_stations = positions.shape[0]
    if waveform == "ofdm":
        pulse = _ofdm_burst(pulse_length, generator)
    elif waveform == "zc":
        pulse = _zadoff_chu(pulse_length, 25)
    else:
        raise ValueError("waveform must be 'ofdm' or 'zc'")
    pulse_length = pulse.shape[0]

    sample_rate = 1e6
    noise_power = (
        BOLTZMANN_T0
        * 10.0 ** (params.noise_figure_db / 10.0)
        * 10.0 ** (params.losses_db / 10.0)
        * sample_rate
    )
    noise_std = math.sqrt(noise_power / 2.0)
    antenna_gain = 10.0 ** (params.antenna_gain_dbi / 10.0)
    wavelength = params.wavelength_m
    steady_columns = residual_phases.shape[1]

    def statistic(mf_by_station: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Two-tier statistic from per-station (already CPI-normalized)
        matched-filter sums, shape (trials, stations)."""

        coherent = (
            torch.abs(
                torch.einsum("tj,tj->t", w.to(torch.complex128), mf_by_station)
            )
            ** 2
        )
        if bench_mode == "discard":
            return coherent
        benched = torch.sum(
            (1.0 - w) * torch.abs(mf_by_station) ** 2, dim=1
        )
        return coherent + benched

    # ---- H0 threshold for the same statistic/membership law --------
    batch = 2000
    h0_values = []
    remaining = h0_trials
    while remaining > 0:
        count = min(batch, remaining)
        remaining -= count
        noise = (
            torch.randn(
                count,
                num_stations,
                pulse_length,
                2,
                dtype=torch.float64,
                generator=generator,
            )
            * noise_std
        )
        streams = torch.view_as_complex(noise.contiguous())
        mf = torch.einsum("tjk,k->tj", streams, torch.conj(pulse))
        h0_columns = torch.randint(
            0, steady_columns, (count,), generator=weight_generator
        )
        h0_weights = rx_weights[:, h0_columns].T
        h0_values.append(statistic(mf, h0_weights))
    h0_stat = torch.cat(h0_values)
    threshold = torch.quantile(h0_stat, 1.0 - threshold_pfa).item()
    measured_pfa = torch.mean((h0_stat > threshold).to(torch.float64)).item()

    pulses_per_cpi = max(
        1, int(round(params.integration_time_s * sample_rate / pulse_length))
    )

    pd_measured: list[float] = []
    combining_loss_db: list[float] = []
    ranges_m: list[float] = []
    centroid = positions.mean(axis=0)
    target_list = np.atleast_2d(np.asarray(targets_m, dtype=float))
    for target_index, target in enumerate(target_list):
        ranges_m.append(float(np.linalg.norm(target - centroid)))
        if leg_gains is not None:
            base_amplitude = math.sqrt(
                params.tx_power_w * 4.0 * math.pi
            ) / wavelength
            inverse_distance = torch.tensor(
                leg_gains[target_index], dtype=torch.complex128
            )
        else:
            distances = np.linalg.norm(positions - target, axis=1)
            distances = np.maximum(distances, 1.0)
            base_amplitude = math.sqrt(
                params.tx_power_w
                * antenna_gain**2
                * wavelength**2
                / (4.0 * math.pi) ** 3
            )
            inverse_distance = torch.tensor(
                1.0 / distances, dtype=torch.float64
            ).to(torch.complex128)

        columns = torch.randint(
            0, steady_columns, (trials,), generator=generator
        )
        theta = residual_phases[:, columns].T  # (trials, stations)
        phasors = torch.exp(1j * theta.to(torch.complex128))
        trial_weights = rx_weights[:, columns].T  # (trials, stations)

        # Transmit leg: ALL-IN — benched stations still radiate.
        tx_field = torch.einsum("tk,k->t", phasors, inverse_distance)
        rcs_draw = (
            torch.randn(trials, 2, dtype=torch.float64, generator=generator)
            / math.sqrt(2.0)
        )
        rcs_amp = torch.view_as_complex(rcs_draw.contiguous()) * math.sqrt(
            params.rcs_m2
        )
        echo = (
            base_amplitude
            * tx_field.unsqueeze(1)
            * rcs_amp.unsqueeze(1)
            * inverse_distance.unsqueeze(0)
            * phasors
        )  # (trials, stations)

        hits = 0
        start = 0
        while start < trials:
            stop = min(start + 500, trials)
            echo_batch = echo[start:stop]
            weight_batch = trial_weights[start:stop]
            cpi = torch.zeros(
                stop - start, num_stations, dtype=torch.complex128
            )
            for _ in range(pulses_per_cpi):
                noise = (
                    torch.randn(
                        stop - start,
                        num_stations,
                        pulse_length,
                        2,
                        dtype=torch.float64,
                        generator=generator,
                    )
                    * noise_std
                )
                streams = torch.view_as_complex(noise.contiguous())
                streams = streams + echo_batch.unsqueeze(-1) * pulse.unsqueeze(
                    0
                ).unsqueeze(0)
                cpi = cpi + torch.einsum(
                    "tjk,k->tj", streams, torch.conj(pulse)
                )
            values = statistic(cpi / math.sqrt(pulses_per_cpi), weight_batch)
            hits += int(torch.sum(values > threshold).item())
            start = stop
        pd_measured.append(hits / trials)

        perfect_field = torch.abs(torch.sum(inverse_distance)) ** 2
        actual_field = torch.mean(torch.abs(tx_field) ** 2)
        combining_loss_db.append(
            10.0
            * math.log10(
                max(actual_field.item() / perfect_field.item(), 1e-12)
            )
        )

    return SimpleDetectionResult(
        label=label,
        num_stations=num_stations,
        ranges_m=ranges_m,
        pd_measured=pd_measured,
        measured_pfa=measured_pfa,
        threshold_pfa=threshold_pfa,
        trials_per_range=trials,
        combining_loss_db=combining_loss_db,
    )


def variant_weights(
    source: str, phases: torch.Tensor, sigma: torch.Tensor, gate_rad: float
) -> torch.Tensor:
    if source == "ones":
        return torch.ones_like(phases)
    if source == "zeros":
        return torch.zeros_like(phases)
    if source == "posterior":
        return posterior_gate_weights(sigma, phases.shape[0], gate_rad)
    if source == "oracle":
        return oracle_gate_weights(phases, gate_rad)
    raise ValueError(f"unknown membership source '{source}'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="hybrid coherent/noncoherent combining of benched stations"
    )
    parser.add_argument("--stations", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--capacity", type=int, default=2)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--gate", type=float, default=math.pi / 2.0)
    parser.add_argument(
        "--variants", type=str, default=",".join(VARIANTS),
        help=f"comma list from {tuple(VARIANTS)}",
    )
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--h0-trials", type=int, default=15000)
    parser.add_argument(
        "--tx-power", type=float, default=0.5,
        help="per-station transmit power, W (lower it to unsaturate "
        "the comparison: at 0.5 W even pure noncoherent fusion can "
        "hit the Pd ceiling at the edge waypoints)",
    )
    args = parser.parse_args()

    n = args.stations
    seeds = [int(s) for s in args.seeds.split(",")]
    variants = [v.strip() for v in args.variants.split(",")]
    for v in variants:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant '{v}'")
    params = DetectionParams(tx_power_w=args.tx_power)

    print(
        f"Hybrid combining, N={n} star, uniform policy, capacity "
        f"{args.capacity}/{n - 1}, {args.iterations} intervals, seeds "
        f"{seeds}, gate {args.gate:.3f} rad, tx {args.tx_power} W, "
        "transmit ALL-IN for every variant (combiner-only comparison)"
    )
    print(f"{'seed':>4} {'variant':<14} {'edge Pd':>16} {'meas Pfa':>12}")
    collected: dict[str, list[list[float]]] = {v: [] for v in variants}
    for seed in seeds:
        settings = SDRSimulationConfig(
            num_iterations=args.iterations, seed=seed, device="cpu"
        )
        result, sigma = run_star_with_posteriors(
            settings,
            num_stations=n,
            policy="uniform",
            max_exchanges_per_interval=args.capacity,
        )
        mask = evaluation_mask(result)
        phases = phase_matrix(result)[:, mask]
        sig = sigma[:, mask]
        positions = result.positions
        centroid = positions.mean(axis=0)
        edge_targets = np.array(
            [centroid + [1200.0, 150.0], centroid + [-1200.0, 150.0]]
        )
        for name in variants:
            source, mode = VARIANTS[name]
            weights = variant_weights(source, phases, sig, args.gate)
            detect = run_hybrid_waveform_detection(
                f"{name}@s{seed}",
                positions,
                phases,
                weights,
                mode,
                edge_targets,
                params=params,
                trials=args.trials,
                h0_trials=args.h0_trials,
                seed=seed,
            )
            collected[name].append(list(detect.pd_measured))
            print(
                f"{seed:>4} {name:<14} "
                + " ".join(f"{100 * pd:6.1f}%" for pd in detect.pd_measured)
                + f"  {detect.measured_pfa:12.2e}"
            )

    print("\n=== mean ± std over seeds (per waypoint) ===")
    for name in variants:
        stack = np.array(collected[name])  # (seeds, waypoints)
        means = stack.mean(axis=0)
        stds = stack.std(axis=0)
        print(
            f"{name:<14} "
            + "  ".join(
                f"{100 * m:5.1f}±{100 * s:4.1f}%"
                for m, s in zip(means, stds)
            )
        )


if __name__ == "__main__":
    main()
