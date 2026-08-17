"""Hardening the gating dissociation result (gating_study.py follow-on).

gating_study.py found that under uniform-scheduling starvation the
posterior membership gate RAISES counted edge detection (77 -> 98%)
while LOWERING mean array gain. This study makes that single point
defensible:

1. Robustness sweep: capacity x seed grid, uniform policy. The claim
   needs the Pd gap (gate - all-in) large and consistently positive in
   the starved regime and vanishing as capacity approaches demand.
2. Mechanism decomposition: the gate acts on BOTH legs - benched
   stations stop transmitting (fade removal at the target) and leave
   the receive combiner (their noise no longer raises the empirical
   threshold). Applying the gate tx-only / rx-only / both splits the
   lift between the two mechanisms.
3. Power-matched counter-check: scale the all-in transmit weights so
   its MEAN beam power at the target equals the gated array's. If the
   gate still wins, the win is fade structure, not power bookkeeping
   or a threshold artifact.

No existing file is modified: the split-weight detection pipeline is
adapted here from gating_study.run_gated_waveform_detection (which is
itself the unit-weight-exact adaptation of detection/waveform.py);
with tx_weights == rx_weights it reproduces that function EXACTLY
(same generator discipline - regression-tested in
tests/test_gating_dissociation.py).

Usage:
    .venv/bin/python gating_dissociation_study.py
    .venv/bin/python gating_dissociation_study.py --quick   # smoke run
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
    membership_weights,
    phase_matrix,
    run_star_with_posteriors,
    weighted_gain,
)
from ota_sync import SDRSimulationConfig


# ---------------------------------------------------------------------
# Split-weight counted detection: independent transmit and receive
# membership. tx_weights scale each station's radiated amplitude;
# rx_weights scale its stream in the combiner AND the H0 threshold
# calibration (a benched receiver's noise must leave both).
# ---------------------------------------------------------------------

def run_split_waveform_detection(
    label: str,
    positions: np.ndarray,
    residual_phases: torch.Tensor,
    tx_weights: torch.Tensor,
    rx_weights: torch.Tensor,
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
    if tx_weights.shape != residual_phases.shape:
        raise ValueError("tx_weights must align with residual_phases")
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

    # ---- H0 threshold under the receive-membership distribution ----
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
        h0_weights = rx_weights[:, h0_columns].T.to(torch.complex128)
        h0_values.append(
            torch.abs(torch.einsum("tj,tj->t", h0_weights, mf)) ** 2
        )
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
        trial_tx = tx_weights[:, columns].T.to(torch.complex128)
        trial_rx = rx_weights[:, columns].T.to(torch.complex128)

        tx_field = torch.einsum(
            "tk,k->t", trial_tx * phasors, inverse_distance
        )
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
            rx_batch = trial_rx[start:stop]
            cpi_sum = torch.zeros(stop - start, dtype=torch.complex128)
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
                mf = torch.einsum("tjk,k->tj", streams, torch.conj(pulse))
                cpi_sum = cpi_sum + torch.einsum("tj,tj->t", rx_batch, mf)
            statistic = torch.abs(cpi_sum) ** 2 / pulses_per_cpi
            hits += int(torch.sum(statistic > threshold).item())
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


# ---------------------------------------------------------------------
# The study
# ---------------------------------------------------------------------

def _edge_targets(positions: np.ndarray) -> np.ndarray:
    centroid = positions.mean(axis=0)
    return np.array(
        [centroid + [1200.0, 150.0], centroid + [-1200.0, 150.0]]
    )


def _prepared(runs, policy_key):
    """(phases, sigma, positions) restricted to the evaluation window."""

    result, sigma = runs[policy_key]
    mask = evaluation_mask(result)
    return phase_matrix(result)[:, mask], sigma[:, mask], result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="capacity/seed robustness + mechanism split of the "
        "membership-gating detection result"
    )
    parser.add_argument("--stations", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--capacities", type=str, default="1,2,3,4")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--mech-seeds", type=str, default="0,1,2")
    parser.add_argument("--mech-capacity", type=int, default=2)
    parser.add_argument("--gate", type=float, default=math.pi / 2.0)
    parser.add_argument("--flat-budget", type=float, default=0.314)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--h0-trials", type=int, default=15000)
    parser.add_argument("--quick", action="store_true",
                        help="tiny smoke run (capacity 2, seed 0 only)")
    args = parser.parse_args()

    if args.quick:
        args.capacities, args.seeds, args.mech_seeds = "2", "0", "0"
        args.trials, args.h0_trials = 100, 4000

    n = args.stations
    capacities = [int(c) for c in args.capacities.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    mech_seeds = [int(s) for s in args.mech_seeds.split(",")]
    params = DetectionParams(tx_power_w=0.5)

    print(
        f"Gating dissociation robustness, N={n} star (demand {n - 1}), "
        f"uniform policy, capacities {capacities}, seeds {seeds}, "
        f"gate {args.gate:.3f} rad, {args.trials} trials/target"
    )

    # ---- one sync run per (capacity, seed); membership = bookkeeping
    runs = {}
    for capacity in capacities:
        for seed in seeds:
            settings = SDRSimulationConfig(
                num_iterations=args.iterations, seed=seed, device="cpu"
            )
            runs[(capacity, seed)] = run_star_with_posteriors(
                settings,
                num_stations=n,
                policy="uniform",
                budgets_rad=[args.flat_budget] * (n - 1),
                max_exchanges_per_interval=capacity,
            )

    # ---- Part 1: capacity x seed sweep -----------------------------
    print("\n=== Part 1: capacity x seed sweep (uniform policy) ===")
    print(
        f"{'cap':>3} {'seed':>4} {'member':<7} {'gain%':>7} "
        f"{'Pd wp1':>7} {'Pd wp2':>7}"
    )
    sweep: dict[tuple[int, int, str], tuple[float, list[float]]] = {}
    for capacity in capacities:
        for seed in seeds:
            phases, sigma, result = _prepared(runs, (capacity, seed))
            targets = _edge_targets(result.positions)
            for member in ("all-in", "gate", "oracle"):
                weights = membership_weights(
                    member, phases, sigma, args.gate
                )
                gain = torch.mean(weighted_gain(phases, weights)).item()
                detect = run_split_waveform_detection(
                    f"{member}@c{capacity}s{seed}",
                    result.positions,
                    phases,
                    weights,
                    weights,
                    targets,
                    params=params,
                    trials=args.trials,
                    h0_trials=args.h0_trials,
                    seed=seed,
                )
                sweep[(capacity, seed, member)] = (
                    gain, detect.pd_measured
                )
                print(
                    f"{capacity:>3} {seed:>4} {member:<7} "
                    f"{100 * gain:7.1f} "
                    + " ".join(
                        f"{100 * pd:6.1f}%" for pd in detect.pd_measured
                    )
                )

    print("\n--- Pd gap (gate - all-in), percentage points, per capacity ---")
    print(f"{'cap':>3} {'mean gap':>9} {'std':>6} {'min':>7} {'max':>7} "
          f"{'gain gap (G% gate - all-in)':>28}")
    for capacity in capacities:
        gaps, gain_gaps = [], []
        for seed in seeds:
            gain_a, pd_a = sweep[(capacity, seed, "all-in")]
            gain_g, pd_g = sweep[(capacity, seed, "gate")]
            gaps.extend(
                100.0 * (g - a) for g, a in zip(pd_g, pd_a)
            )
            gain_gaps.append(100.0 * (gain_g - gain_a))
        gaps = np.array(gaps)
        print(
            f"{capacity:>3} {gaps.mean():9.1f} {gaps.std():6.1f} "
            f"{gaps.min():7.1f} {gaps.max():7.1f} "
            f"{np.mean(gain_gaps):28.1f}"
        )

    # ---- Part 2: mechanism decomposition ---------------------------
    print(
        f"\n=== Part 2: mechanism split (capacity {args.mech_capacity}, "
        f"seeds {mech_seeds}; posterior gate) ==="
    )
    print(
        f"{'seed':>4} {'mode':<9} {'Pd wp1':>7} {'Pd wp2':>7}   "
        "(tx-only: fades removed, all-in threshold; rx-only: noise "
        "pruned, gated threshold)"
    )
    mech: dict[tuple[int, str], list[float]] = {}
    for seed in mech_seeds:
        phases, sigma, result = _prepared(runs, (args.mech_capacity, seed))
        targets = _edge_targets(result.positions)
        ones = torch.ones_like(phases)
        gate_w = membership_weights("gate", phases, sigma, args.gate)
        modes = {
            "all-in": (ones, ones),
            "tx-only": (gate_w, ones),
            "rx-only": (ones, gate_w),
            "both": (gate_w, gate_w),
        }
        for mode, (tx_w, rx_w) in modes.items():
            detect = run_split_waveform_detection(
                f"{mode}@s{seed}",
                result.positions,
                phases,
                tx_w,
                rx_w,
                targets,
                params=params,
                trials=args.trials,
                h0_trials=args.h0_trials,
                seed=seed,
            )
            mech[(seed, mode)] = detect.pd_measured
            print(
                f"{seed:>4} {mode:<9} "
                + " ".join(
                    f"{100 * pd:6.1f}%" for pd in detect.pd_measured
                )
            )
    lifts = {mode: [] for mode in ("tx-only", "rx-only", "both")}
    for seed in mech_seeds:
        base = mech[(seed, "all-in")]
        for mode in lifts:
            lifts[mode].extend(
                100.0 * (pd - b)
                for pd, b in zip(mech[(seed, mode)], base)
            )
    print("--- mean Pd lift over all-in (percentage points) ---")
    for mode, values in lifts.items():
        print(f"  {mode:<9} {np.mean(values):6.1f}")

    # ---- Part 3: power-matched counter-check -----------------------
    seed = mech_seeds[0]
    print(
        f"\n=== Part 3: power-matched all-in (capacity "
        f"{args.mech_capacity}, seed {seed}) ==="
    )
    phases, sigma, result = _prepared(runs, (args.mech_capacity, seed))
    targets = _edge_targets(result.positions)
    ones = torch.ones_like(phases)
    gate_w = membership_weights("gate", phases, sigma, args.gate)
    gain_allin = torch.mean(weighted_gain(phases, ones)).item()
    gain_gate = torch.mean(weighted_gain(phases, gate_w)).item()
    scale = math.sqrt(gain_gate / gain_allin)
    matched = ones * scale
    print(
        f"mean gain: all-in {100 * gain_allin:.1f}%, gate "
        f"{100 * gain_gate:.1f}% -> tx scale {scale:.3f} "
        f"(matched mean beam power)"
    )
    for label, tx_w, rx_w in (
        ("all-in", ones, ones),
        ("matched", matched, ones),
        ("gate", gate_w, gate_w),
    ):
        detect = run_split_waveform_detection(
            f"pm-{label}",
            result.positions,
            phases,
            tx_w,
            rx_w,
            targets,
            params=params,
            trials=args.trials,
            h0_trials=args.h0_trials,
            seed=seed,
        )
        print(
            f"  {label:<8} Pd "
            + " ".join(f"{100 * pd:6.1f}%" for pd in detect.pd_measured)
        )
    print(
        "(if gate > matched, the win is fade structure, not beam-power "
        "bookkeeping)"
    )


if __name__ == "__main__":
    main()
