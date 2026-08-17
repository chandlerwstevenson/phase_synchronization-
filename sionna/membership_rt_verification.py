"""Membership results under the most realistic propagation available.

Every membership/detection headline so far used isotropic 1/d legs.
This script re-runs the headline comparisons under:

  A. Ray-traced steered legs (detection/rt_echo.py): 3-D geometry,
     station masts, ground-bounce two-ray lobing over ITU-class ground.
     Same single-pulse cued pipeline, per-variant empirical thresholds.
  B. The same RT legs for the hybrid two-tier combiner regimes,
     including the starved-unsaturated power point.
  C. The CPI-level clutter/interference pipeline (detection/realistic.py):
     constant-gamma ground clutter with internal motion, direct-path
     self-interference with LS cancellation, drone body + rotor
     micro-Doppler, range-Doppler CFAR — adapted (in this file, no
     existing file modified) to carry per-interval membership weights:
     a benched station neither radiates the probe (its direct-path
     interference disappears too) nor contributes its stream to the
     noncoherent fusion. CFAR scale recalibrated per membership variant.

Usage:
    .venv/bin/python membership_rt_verification.py --part a
    .venv/bin/python membership_rt_verification.py --part b
    .venv/bin/python membership_rt_verification.py --part c
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch

from detection import DetectionParams
from detection.realistic import (
    RealisticDetectionConfig,
    _clutter_gate_power,
    _range_gates,
)
from detection.rt_echo import rt_steered_legs
from detection.viability import BOLTZMANN_T0, SPEED_OF_LIGHT
from detection.waveform import _ofdm_burst
from gating_study import (
    evaluation_mask,
    membership_weights,
    phase_matrix,
    run_gated_waveform_detection,
    run_star_with_posteriors,
    weighted_gain,
)
from hybrid_combiner_study import VARIANTS, run_hybrid_waveform_detection
from ota_sync import SDRSimulationConfig

def sync_run(seed: int, capacity: int = 2, stations: int = 10,
             iterations: int = 50):
    settings = SDRSimulationConfig(
        num_iterations=iterations, seed=seed, device="cpu"
    )
    result, sigma = run_star_with_posteriors(
        settings,
        num_stations=stations,
        policy="uniform",
        budgets_rad=[0.314] * (stations - 1),
        max_exchanges_per_interval=capacity,
    )
    mask = evaluation_mask(result)
    phases = phase_matrix(result)[:, mask]
    return result, phases, sigma[:, mask]


# ---------------------------------------------------------------------
# Part C: CPI-level clutter pipeline with membership weights.
# Adapted from detection/realistic.py:run_realistic_detection (which
# has no membership concept). Changes: (1) per-trial weights drawn with
# the residual column gate a station's transmit (probe + its direct-
# path interference at every other receiver) AND its receive stream in
# the noncoherent fusion; (2) CFAR scale calibrated per membership
# variant under the same weight distribution; (3) example-map
# bookkeeping dropped.
# ---------------------------------------------------------------------

def run_realistic_membership_detection(
    label: str,
    positions: np.ndarray,
    residual_phases: torch.Tensor,
    weights: torch.Tensor,
    waypoints: np.ndarray,
    leg_gains: np.ndarray,
    params: DetectionParams = DetectionParams(),
    config: RealisticDetectionConfig = RealisticDetectionConfig(),
    trials: int = 120,
    h0_trials: int = 400,
    seed: int = 0,
    rx_weights: torch.Tensor | None = None,
):
    if weights.shape != residual_phases.shape:
        raise ValueError("weights must align with residual_phases")
    if rx_weights is None:
        rx_weights = weights
    if rx_weights.shape != residual_phases.shape:
        raise ValueError("rx_weights must align with residual_phases")
    generator = torch.Generator().manual_seed(seed)
    sample_rate = 1e6
    n = positions.shape[0]
    pulse = _ofdm_burst(config.pulse_length, generator).to(torch.complex64)
    pulse_len = pulse.shape[0]
    wavelength = params.wavelength_m
    gain_amp = math.sqrt(10.0 ** (params.antenna_gain_dbi / 10.0))

    gate_samples, gate_lo = _range_gates(
        positions, waypoints, config, sample_rate
    )
    num_gates = len(gate_samples)
    stream_len = num_gates + pulse_len - 1
    num_pulses = config.num_pulses

    noise_power = (
        BOLTZMANN_T0
        * 10.0 ** (params.noise_figure_db / 10.0)
        * 10.0 ** (params.losses_db / 10.0)
        * sample_rate
    )
    noise_std = math.sqrt(noise_power / 2.0)

    clutter_power = torch.tensor(
        _clutter_gate_power(
            positions, gate_samples, params, config, sample_rate
        ),
        dtype=torch.float64,
    )
    clutter_to_noise_db = 10.0 * math.log10(
        max(clutter_power.max().item(), 1e-30) / noise_power
    )
    sigma_f = 2.0 * config.clutter_motion_std_mps / wavelength
    doppler_axis = torch.fft.fftfreq(num_pulses, d=config.pri_s)
    spectrum = torch.exp(-0.5 * (doppler_axis / max(sigma_f, 1e-3)) ** 2)
    spectrum = spectrum / torch.sqrt(torch.mean(spectrum**2))

    stations3 = np.column_stack(
        (positions, np.full(n, config.station_height_m))
    )
    inter = np.linalg.norm(
        stations3[:, None, :] - stations3[None, :, :], axis=2
    )
    np.fill_diagonal(inter, np.inf)
    direct_amp = (
        gain_amp**2
        * wavelength
        / (4.0 * math.pi * inter)
        * math.sqrt(params.tx_power_w)
        * 10.0 ** (-config.analog_isolation_db / 20.0)
    )
    finite_inter = np.where(np.isfinite(inter), inter, 0.0)
    direct_gate = (
        np.round(finite_inter / SPEED_OF_LIGHT * sample_rate).astype(int)
        - gate_lo
    )
    np.fill_diagonal(direct_gate, -1)

    window = torch.hann_window(num_pulses, dtype=torch.float64)
    notch = config.clutter_notch_bins
    steady_columns = residual_phases.shape[1]

    def make_streams(batch, theta, w, target_terms):
        streams = (
            torch.randn(
                batch, n, num_pulses, stream_len, 2,
                dtype=torch.float32, generator=generator,
            )
            * noise_std
        )
        streams = torch.view_as_complex(streams.contiguous())
        phasors = torch.exp(1j * theta.to(torch.complex128)).to(
            torch.complex64
        )  # (b, n)
        tx_phasors = phasors * w.to(torch.complex64)  # benched: no probe

        white = torch.randn(
            batch, n, num_gates, num_pulses, 2,
            dtype=torch.float32, generator=generator,
        )
        white = torch.view_as_complex(white.contiguous()).to(torch.complex64)
        shaped = torch.fft.ifft(
            torch.fft.fft(white, dim=-1) * spectrum.to(torch.complex64),
            dim=-1,
        )
        amplitude = torch.sqrt(clutter_power / 2.0).to(torch.complex64)
        gate_signal = shaped * amplitude.unsqueeze(0).unsqueeze(-1)
        gate_signal = gate_signal.permute(0, 1, 3, 2)
        clutter_streams = torch.nn.functional.conv1d(
            gate_signal.reshape(-1, 1, num_gates),
            pulse.flip(0).unsqueeze(0).unsqueeze(0),
            padding=pulse_len - 1,
        ).reshape(batch, n, num_pulses, -1)[..., :stream_len]
        streams = streams + clutter_streams

        for rx in range(n):
            for tx in range(n):
                if tx == rx:
                    continue
                gate = direct_gate[tx, rx]
                if gate < 0 or gate >= num_gates:
                    continue
                coefficient = float(direct_amp[tx, rx]) * tx_phasors[:, tx]
                streams[:, rx, :, gate : gate + pulse_len] += (
                    coefficient.unsqueeze(-1).unsqueeze(-1) * pulse
                )

        if target_terms is not None:
            echo_base, doppler_kj, signature, gate_t = target_terms
            echo_kj = (
                echo_base.unsqueeze(0)
                * tx_phasors.unsqueeze(2)
                * phasors.unsqueeze(1)
            )  # (b, tx, rx): tx gated, rx LO physical
            pulse_idx = torch.arange(num_pulses, dtype=torch.float64)
            for rx in range(n):
                ramp = torch.exp(
                    1j
                    * 2.0
                    * math.pi
                    * doppler_kj[:, rx].unsqueeze(-1)
                    * pulse_idx
                    * config.pri_s
                ).to(torch.complex64)
                combined = torch.einsum(
                    "bk,kp->bp", echo_kj[:, :, rx], ramp
                )
                combined = combined * signature
                streams[:, rx, :, gate_t : gate_t + pulse_len] += (
                    combined.unsqueeze(-1) * pulse
                )
        return streams

    def process(streams, w):
        batch = streams.shape[0]
        for rx in range(n):
            gates = sorted(
                {
                    int(direct_gate[tx, rx])
                    for tx in range(n)
                    if tx != rx
                    and 0 <= direct_gate[tx, rx] < num_gates
                }
            )
            if not gates:
                continue
            templates = torch.zeros(
                len(gates), stream_len, dtype=torch.complex64
            )
            for row, gate in enumerate(gates):
                templates[row, gate : gate + pulse_len] = pulse
            gram = templates @ templates.conj().T
            gram_inverse = torch.linalg.inv(gram)
            flat = streams[:, rx].reshape(batch * num_pulses, stream_len)
            coefficients = flat @ templates.conj().T @ gram_inverse.T
            flat = flat - coefficients @ templates
            streams[:, rx] = flat.reshape(batch, num_pulses, stream_len)

        flat = streams.reshape(-1, 1, stream_len)
        mf = torch.nn.functional.conv1d(
            flat, pulse.conj().unsqueeze(0).unsqueeze(0).resolve_conj()
        ).reshape(streams.shape[0], n, num_pulses, -1)[..., :num_gates]
        rd = torch.fft.fft(
            mf * window.to(torch.complex64).reshape(1, 1, -1, 1), dim=2
        )
        # Membership-weighted noncoherent fusion: benched receive
        # streams leave the map (their clutter/noise/direct residue
        # with them).
        fused = torch.einsum(
            "bn,bnpg->bpg", w.to(torch.float32), torch.abs(rd) ** 2
        )
        return fused.permute(0, 2, 1)  # (b, gates, doppler)

    def cfar_normalize(maps):
        guard, train = config.cfar_guard_cells, config.cfar_train_cells
        kernel_size = 2 * (guard + train) + 1
        kernel = torch.zeros(kernel_size, dtype=torch.float32)
        kernel[:train] = 1.0
        kernel[-train:] = 1.0
        kernel = kernel / kernel.sum()
        padded = torch.nn.functional.conv1d(
            maps.permute(0, 2, 1).reshape(-1, 1, maps.shape[1]).float(),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=guard + train,
        ).reshape(maps.shape[0], maps.shape[2], maps.shape[1]).permute(0, 2, 1)
        return maps / torch.clamp(padded, min=1e-30)

    def detection_statistic(maps, gate_t):
        normalized = cfar_normalize(maps)
        bins = torch.ones(num_pulses, dtype=torch.bool)
        bins[: notch + 1] = False
        bins[num_pulses - notch :] = False
        lo_g = max(gate_t - 1, 0)
        hi_g = min(gate_t + 2, normalized.shape[1])
        region = normalized[:, lo_g:hi_g][:, :, bins]
        return region.reshape(region.shape[0], -1).max(dim=1).values

    mid = waypoints[len(waypoints) // 2]
    mid3 = np.array([mid[0], mid[1], config.target_height_m])
    d_mid = np.linalg.norm(stations3 - mid3, axis=1)
    gate_mid = int(
        round((d_mid.mean() * 2.0) / SPEED_OF_LIGHT * sample_rate)
    ) - gate_lo
    gate_mid = int(np.clip(gate_mid, 1, num_gates - 2))

    h0_values = []
    batch = 20
    done = 0
    while done < h0_trials:
        count = min(batch, h0_trials - done)
        done += count
        columns = torch.randint(
            0, steady_columns, (count,), generator=generator
        )
        theta = residual_phases[:, columns].T
        w_tx = weights[:, columns].T
        w_rx = rx_weights[:, columns].T
        streams = make_streams(count, theta, w_tx, None)
        maps = process(streams, w_rx)
        h0_values.append(detection_statistic(maps, gate_mid))
    h0_stat = torch.cat(h0_values)
    scale = torch.quantile(h0_stat.double(), 1.0 - config.window_pfa).item()
    measured_pfa = torch.mean((h0_stat > scale).to(torch.float64)).item()

    pd_measured = []
    body_max = 10.0 ** (config.body_rcs_broadside_dbsm / 10.0)
    body_min = 10.0 ** (config.body_rcs_nose_dbsm / 10.0)
    blade_amp = math.sqrt(10.0 ** (config.blade_rcs_dbsm / 10.0))
    velocity = np.array([config.drone_speed_mps, 0.0, 0.0])

    for w_index, waypoint in enumerate(waypoints):
        target3 = np.array(
            [waypoint[0], waypoint[1], config.target_height_m]
        )
        vectors = target3 - stations3
        d = np.linalg.norm(vectors, axis=1)
        unit = vectors / d[:, None]
        radial = unit @ velocity
        doppler_kj = torch.tensor(
            (radial[:, None] + radial[None, :]) / wavelength,
            dtype=torch.float64,
        )
        bisector = unit[:, None, :] + unit[None, :, :]
        bisector = bisector / np.maximum(
            np.linalg.norm(bisector, axis=2, keepdims=True), 1e-9
        )
        cos_aspect = np.abs(
            bisector @ (velocity / max(config.drone_speed_mps, 1e-9))
        )
        body_rcs = body_min + (body_max - body_min) * (1.0 - cos_aspect**2)
        gate_t = int(
            round(
                (d[:, None] + d[None, :]).mean()
                / SPEED_OF_LIGHT
                * sample_rate
            )
        ) - gate_lo
        gate_t = int(np.clip(gate_t, 1, num_gates - 2))

        legs = torch.tensor(leg_gains[w_index], dtype=torch.complex128)
        echo_base = (
            math.sqrt(params.tx_power_w * 4.0 * math.pi)
            / wavelength
            * legs[:, None]
            * legs[None, :]
            * torch.tensor(np.sqrt(body_rcs), dtype=torch.complex128)
        ).to(torch.complex64)

        hits = 0
        remaining = trials
        while remaining > 0:
            count = min(batch, remaining)
            remaining -= count
            columns = torch.randint(
                0, steady_columns, (count,), generator=generator
            )
            theta = residual_phases[:, columns].T
            w_tx = weights[:, columns].T
            w_rx = rx_weights[:, columns].T
            pulse_times = (
                torch.arange(num_pulses, dtype=torch.float64) * config.pri_s
            )
            signature = torch.ones(
                count, num_pulses, dtype=torch.complex128
            )
            for rotor in range(config.num_rotors):
                rate = config.rotor_rate_hz * (1.0 + 0.05 * rotor)
                for blade in range(config.blades_per_rotor):
                    phase0 = (
                        torch.rand(count, 1, generator=generator)
                        * 2.0
                        * math.pi
                    )
                    beta = 4.0 * math.pi * config.blade_length_m / wavelength
                    signature = signature + (
                        blade_amp
                        / math.sqrt(body_max)
                        * torch.exp(
                            1j
                            * beta
                            * torch.sin(
                                2.0 * math.pi * rate * pulse_times
                                + phase0
                                + blade * math.pi
                            )
                        )
                    )
            signature = signature.to(torch.complex64)
            streams = make_streams(
                count, theta, w_tx,
                (echo_base, doppler_kj, signature, gate_t),
            )
            maps = process(streams, w_rx)
            stat = detection_statistic(maps, gate_t)
            hits += int(torch.sum(stat > scale).item())
        pd_measured.append(hits / trials)

    return {
        "label": label,
        "pd": pd_measured,
        "pfa": measured_pfa,
        "cnr_db": clutter_to_noise_db,
    }


# ---------------------------------------------------------------------


def edge_targets_for(positions: np.ndarray) -> np.ndarray:
    centroid = positions.mean(axis=0)
    return np.array(
        [centroid + [1200.0, 150.0], centroid + [-1200.0, 150.0]]
    )


def part_a(args) -> None:
    result, phases, sigma = sync_run(args.seed)
    targets = edge_targets_for(result.positions)
    legs = legs_for_seed(args.seed, result.positions, targets)
    params = DetectionParams(tx_power_w=args.tx_power)
    print(
        f"\n=== A: gating variants, RT ground-bounce legs "
        f"(uniform cap 2, seed {args.seed}, {args.tx_power} W, "
        f"{args.trials} trials) ==="
    )
    print(f"{'membership':<10} {'edge Pd':>16} {'meas Pfa':>10} "
          f"{'mean gain':>10}")
    for variant in ("all-in", "gate", "soft", "oracle"):
        weights = membership_weights(
            variant, phases, sigma, math.pi / 2.0
        )
        gain = torch.mean(weighted_gain(phases, weights)).item()
        detect = run_gated_waveform_detection(
            f"rt/{variant}", result.positions, phases, weights, targets,
            params=params, trials=args.trials, h0_trials=args.h0_trials,
            seed=args.seed, leg_gains=legs,
        )
        print(
            f"{variant:<10} "
            + " ".join(f"{100 * pd:6.1f}%" for pd in detect.pd_measured)
            + f"  {detect.measured_pfa:10.2e} {100 * gain:9.1f}%"
        )


def legs_for_seed(seed: int, positions: np.ndarray,
                  targets: np.ndarray) -> np.ndarray:
    cache = f"membership_rt_legs_seed{seed}.npz"
    if os.path.exists(cache):
        data = np.load(cache)
        if (
            np.allclose(data["positions"], positions)
            and np.allclose(data["targets"], targets)
        ):
            return data["legs"]
    print(f"ray tracing legs for seed {seed}...")
    legs = rt_steered_legs(positions, targets)
    np.savez(cache, positions=positions, targets=targets, legs=legs)
    return legs


def part_b(args) -> None:
    for tx_power, seeds in ((0.5, [args.seed]), (0.05, [0, 1, 2])):
        params = DetectionParams(tx_power_w=tx_power)
        rows: dict[str, list[list[float]]] = {}
        for seed in seeds:
            result, phases, sigma = sync_run(seed)
            seed_targets = edge_targets_for(result.positions)
            legs = legs_for_seed(seed, result.positions, seed_targets)
            for variant, (source, mode) in VARIANTS.items():
                if source == "ones":
                    weights = torch.ones_like(phases)
                elif source == "zeros":
                    weights = torch.zeros_like(phases)
                    weights[0] = 1.0
                elif source == "posterior":
                    weights = membership_weights(
                        "gate", phases, sigma, math.pi / 2.0
                    )
                else:
                    weights = membership_weights(
                        "oracle", phases, sigma, math.pi / 2.0
                    )
                detect = run_hybrid_waveform_detection(
                    f"rt/{variant}@{tx_power}W", result.positions, phases,
                    weights, mode, seed_targets, params=params,
                    trials=args.trials, h0_trials=args.h0_trials,
                    seed=seed, leg_gains=legs,
                )
                rows.setdefault(variant, []).append(detect.pd_measured)
        print(
            f"\n=== B: hybrid combiner, RT legs, {tx_power} W, "
            f"seeds {seeds} (mean Pd %) ==="
        )
        for variant, values in rows.items():
            arr = 100.0 * np.array(values)
            cells = " / ".join(
                f"{m:5.1f}±{s:4.1f}" for m, s in zip(
                    arr.mean(axis=0), arr.std(axis=0)
                )
            )
            print(f"{variant:<13} {cells}")


def part_c(args) -> None:
    result, phases, sigma = sync_run(args.seed)
    targets = edge_targets_for(result.positions)
    legs = legs_for_seed(args.seed, result.positions, targets)
    params = DetectionParams(tx_power_w=args.tx_power)
    config = RealisticDetectionConfig()
    print(
        f"\n=== C: CPI clutter/interference pipeline, RT legs "
        f"(uniform cap 2, seed {args.seed}, {args.tx_power} W, "
        f"{args.c_trials} trials, h0 {args.c_h0}) ==="
    )
    ones = torch.ones_like(phases)
    cases = (
        ("all-in", ones, ones),
        ("gate", membership_weights("gate", phases, sigma, math.pi / 2.0),
         None),
        ("gate-rx-only", ones,
         membership_weights("gate", phases, sigma, math.pi / 2.0)),
        ("oracle", membership_weights("oracle", phases, sigma, math.pi / 2.0),
         None),
    )
    for name, tx_w, rx_w in cases:
        out = run_realistic_membership_detection(
            f"clutter/{name}", result.positions, phases, tx_w,
            targets, legs, params=params, config=config,
            trials=args.c_trials, h0_trials=args.c_h0, seed=args.seed,
            rx_weights=rx_w,
        )
        print(
            f"{name:<13} edge Pd "
            + " ".join(f"{100 * pd:6.1f}%" for pd in out["pd"])
            + f"  window-Pfa {out['pfa']:.3f}  CNR {out['cnr_db']:5.1f} dB"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="membership results under RT + clutter realism"
    )
    parser.add_argument("--part", type=str, default="all",
                        choices=("a", "b", "c", "all"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tx-power", type=float, default=0.5)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--h0-trials", type=int, default=12000)
    parser.add_argument("--c-trials", type=int, default=120)
    parser.add_argument("--c-h0", type=int, default=400)
    args = parser.parse_args()
    if args.part in ("a", "all"):
        part_a(args)
    if args.part in ("b", "all"):
        part_b(args)
    if args.part in ("c", "all"):
        part_c(args)


if __name__ == "__main__":
    main()
