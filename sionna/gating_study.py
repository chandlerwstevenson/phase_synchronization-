"""Posterior-gated array membership (RESEARCH_IDEAS.md idea #7).

A station whose sync residual has drifted far enough does not merely
stop helping the coherent sum - past 90 degrees it subtracts from it.
The scheduler's own Kalman posterior predicts when a coasting station
crosses that line, so membership in the coherent sum can be gated (or
soft-weighted) from the same state that already drives pilot
scheduling. This study asks whether that buys anything real, in array
gain and in counted waveform detection, at the contended operating
point where stations genuinely go stale (contention_study.py regime).

Nothing in ota_sync/ or detection/ is modified. The star is
instrumented from the outside: a recording EKF subclass is swapped
into ota_sync.scheduled's namespace for the duration of one run. The
subclass only appends to a list after predict(), so the run's random
draw order - and therefore every number - is bit-identical to the
unpatched run (regression-tested in tests/test_gating_study.py).

Membership policies, all evaluated on the SAME sync run:

  all-in   every station transmits and is combined every interval
           (the status quo in every study in this repository)
  gate     bench station k at interval t when its predicted phase
           posterior std sigma_{k,t} exceeds a threshold (default
           pi/2 - the asset-to-liability line of the abstract)
  soft     weight station k by its expected phasor under the
           posterior, E[e^{j theta}] = exp(-sigma^2/2) (the
           robust-beamforming weighting, but driven by the
           scheduler's filter state, not a separate estimator)
  oracle   bench on the TRUE residual |theta| > pi/2 - the genie a
           deployed system cannot be
  greedy   per-interval genie subset: greedily drop stations while
           the coherent sum improves. Upper bound of ANY per-station
           membership rule (gain bookkeeping only; no detection run)

Gain is always normalized to the full-array perfect ideal N^2, so a
benched station's lost transmit power is charged to the gate - no
free lunch from shrinking the array.

Detection is the counted waveform pipeline with two changes: benched/
weighted stations scale their transmit amplitude AND their receive
combining weight, and the empirical H0 threshold is recalibrated per
membership policy with the same weight distribution (dropping noisy
receive streams narrows the noise statistic; using the all-in
threshold would flatter the gate).

Usage:
    .venv/bin/python gating_study.py                    # full study
    .venv/bin/python gating_study.py --no-detect        # gain only
    .venv/bin/python gating_study.py --stations 10 --capacity 2
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
import torch

from detection import DetectionParams
from detection.viability import BOLTZMANN_T0
from detection.waveform import _ofdm_burst, _zadoff_chu
import ota_sync.scheduled as scheduled_module
from ota_sync import SDRSimulationConfig
from ota_sync.core import PhaseFrequencyEKF, wrap_phase


# ---------------------------------------------------------------------
# Outside-in instrumentation of run_scheduled_star
# ---------------------------------------------------------------------

class RecordingEKF(PhaseFrequencyEKF):
    """PhaseFrequencyEKF that logs its predicted phase std.

    predict() is called exactly once per link per interval inside
    run_scheduled_star, so each instance's log has one entry per
    interval - the same predicted-std quantity the "scheduled" policy
    ranks on. Recording happens after the covariance step and touches
    no random state.
    """

    instances: list["RecordingEKF"] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predicted_phase_std: list[float] = []
        RecordingEKF.instances.append(self)

    def predict(self) -> None:
        super().predict()
        self.predicted_phase_std.append(
            math.sqrt(max(self.covariance[0, 0].item(), 0.0))
        )


def run_star_with_posteriors(settings: SDRSimulationConfig, **kwargs):
    """run_scheduled_star, plus the per-station posterior-std matrix.

    Returns (result, sigma) with sigma of shape (stations-1,
    intervals): sigma[k, t] is link k+1's predicted phase std at
    interval t, i.e. what the filter believed BEFORE any service that
    interval - the quantity a deployed gate would act on.
    """

    RecordingEKF.instances = []
    original = scheduled_module.PhaseFrequencyEKF
    scheduled_module.PhaseFrequencyEKF = RecordingEKF
    try:
        result = scheduled_module.run_scheduled_star(settings, **kwargs)
    finally:
        scheduled_module.PhaseFrequencyEKF = original
    sigma = torch.tensor(
        [ekf.predicted_phase_std for ekf in RecordingEKF.instances],
        dtype=torch.float64,
    )
    expected = (result.num_stations - 1, settings.num_iterations)
    if tuple(sigma.shape) != expected:
        raise RuntimeError(
            f"posterior recording shape {tuple(sigma.shape)} != {expected}"
        )
    return result, sigma


# ---------------------------------------------------------------------
# Membership bookkeeping on the recorded matrices
# ---------------------------------------------------------------------

def evaluation_mask(result) -> torch.Tensor:
    """Steady intervals, falling back to the tail quarter when the run
    never reaches all-stations steady (uniform under contention never
    does) - same convention as contention_study.py."""

    if torch.any(result.steady):
        return result.steady.clone()
    intervals = result.residuals.shape[1]
    mask = torch.zeros(intervals, dtype=torch.bool)
    mask[max(0, intervals - max(1, intervals // 4)):] = True
    return mask


def phase_matrix(result) -> torch.Tensor:
    """(stations, intervals) phases with row 0 the reference datum."""

    reference = torch.zeros(
        1, result.residuals.shape[1], dtype=torch.float64
    )
    return torch.cat((reference, result.residuals.to(torch.float64)))


def weighted_gain(phases: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Per-interval coherent gain |sum_k w_k e^{j phi_k}|^2 / N^2."""

    phasors = weights.to(torch.complex128) * torch.exp(
        1j * phases.to(torch.complex128)
    )
    return (
        torch.abs(torch.sum(phasors, dim=0)) ** 2 / phases.shape[0] ** 2
    ).to(torch.float64)


def posterior_gate_weights(
    sigma: torch.Tensor, num_stations: int, gate_rad: float
) -> torch.Tensor:
    weights = torch.ones(num_stations, sigma.shape[1], dtype=torch.float64)
    weights[1:] = (sigma <= gate_rad).to(torch.float64)
    return weights


def posterior_soft_weights(
    sigma: torch.Tensor, num_stations: int
) -> torch.Tensor:
    weights = torch.ones(num_stations, sigma.shape[1], dtype=torch.float64)
    weights[1:] = torch.exp(-0.5 * sigma.square())
    return weights


def oracle_gate_weights(
    phases: torch.Tensor, gate_rad: float
) -> torch.Tensor:
    weights = (torch.abs(wrap_phase(phases)) <= gate_rad).to(torch.float64)
    weights[0] = 1.0
    return weights


def greedy_oracle_weights(phases: torch.Tensor) -> torch.Tensor:
    """Per-interval genie subset by greedy removal (never below the
    all-in sum, by construction). Reference stays in as the datum."""

    num_stations, intervals = phases.shape
    weights = torch.ones(num_stations, intervals, dtype=torch.float64)
    for t in range(intervals):
        phasors = torch.exp(1j * phases[:, t].to(torch.complex128))
        active = torch.ones(num_stations, dtype=torch.bool)
        total = torch.sum(phasors)
        while True:
            power = torch.abs(total) ** 2
            best_power, best_station = power, -1
            for k in range(1, num_stations):
                if not active[k]:
                    continue
                without = torch.abs(total - phasors[k]) ** 2
                if without > best_power + 1e-12:
                    best_power, best_station = without, k
            if best_station < 0:
                break
            active[best_station] = False
            total = total - phasors[best_station]
        weights[:, t] = active.to(torch.float64)
    return weights


@dataclass(frozen=True)
class SimpleDetectionResult:
    label: str
    num_stations: int
    ranges_m: list[float]
    pd_measured: list[float]
    measured_pfa: float
    threshold_pfa: float
    trials_per_range: int
    combining_loss_db: list[float]


# ---------------------------------------------------------------------
# Counted waveform detection with membership weights
# ---------------------------------------------------------------------
# Adapted from detection/waveform.py:run_waveform_detection (kept
# byte-compatible where weights are all ones: with unit weights the
# main generator's draw sequence is identical, so it reproduces the
# unweighted pipeline's numbers exactly - regression-tested). The two
# additions: (1) station weights scale the transmit amplitude and the
# receive combining, (2) the empirical H0 threshold is calibrated
# under the same weight distribution (weight columns for H0 come from
# a SEPARATE generator so the main draw sequence is untouched).

def run_gated_waveform_detection(
    label: str,
    positions: np.ndarray,
    residual_phases: torch.Tensor,
    weights: torch.Tensor,
    targets_m: np.ndarray,
    params: DetectionParams = DetectionParams(),
    pulse_length: int = 1023,
    trials: int = 2000,
    h0_trials: int = 60000,
    threshold_pfa: float = 1e-3,
    seed: int = 0,
    waveform: str = "ofdm",
    leg_gains: np.ndarray | None = None,
):
    if weights.shape != residual_phases.shape:
        raise ValueError("weights must align with residual_phases")

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

    # ---- H0 threshold under the same membership distribution --------
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
        h0_weights = weights[:, h0_columns].T.to(torch.complex128)
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
        trial_weights = weights[:, columns].T.to(torch.complex128)

        # Transmit leg: a benched/weighted station radiates scaled
        # amplitude. Receive leg residual stays physical on the echo;
        # the weight is applied in the combiner below.
        tx_field = torch.einsum(
            "tk,k->t", trial_weights * phasors, inverse_distance
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
            weight_batch = trial_weights[start:stop]
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
                cpi_sum = cpi_sum + torch.einsum(
                    "tj,tj->t", weight_batch, mf
                )
            statistic = torch.abs(cpi_sum) ** 2 / pulses_per_cpi
            hits += int(torch.sum(statistic > threshold).item())
            start = stop
        pd_measured.append(hits / trials)

        # Transmit-leg combining loss vs the full perfect array.
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

MEMBERSHIP_VARIANTS = ("all-in", "gate", "soft", "oracle", "greedy")


def membership_weights(
    variant: str,
    phases: torch.Tensor,
    sigma: torch.Tensor,
    gate_rad: float,
) -> torch.Tensor:
    num_stations = phases.shape[0]
    if variant == "all-in":
        return torch.ones_like(phases)
    if variant == "gate":
        return posterior_gate_weights(sigma, num_stations, gate_rad)
    if variant == "soft":
        return posterior_soft_weights(sigma, num_stations)
    if variant == "oracle":
        return oracle_gate_weights(phases, math.pi / 2.0)
    if variant == "greedy":
        return greedy_oracle_weights(phases)
    raise ValueError(f"unknown membership variant '{variant}'")


def summarize(gain: torch.Tensor) -> tuple[float, float, float]:
    """(mean gain, mean squared gain, 5th-percentile gain) over the
    evaluation window. Mean G^2 is the detection-relevant moment: sync
    errors hit transmit AND receive, so echo SNR scales with G^2 of
    the SAME draw."""

    return (
        torch.mean(gain).item(),
        torch.mean(gain.square()).item(),
        torch.quantile(gain, 0.05).item(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="posterior-gated membership on a contended sync channel"
    )
    parser.add_argument("--stations", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--capacity", type=int, default=2)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument(
        "--policies", type=str,
        default="uniform,roundrobin,scheduled,oracle",
        help="SCHEDULING policies (who gets pilots); membership gating "
        "is evaluated on top of each",
    )
    parser.add_argument("--flat-budget", type=float, default=0.314)
    parser.add_argument("--gate", type=float, default=math.pi / 2.0,
                        help="posterior-std bench threshold, rad")
    parser.add_argument(
        "--gate-sweep", type=str,
        default="0.6,0.9,1.2,1.571,2.0,2.5",
        help="thresholds swept for the gate-tuning table",
    )
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--h0-trials", type=int, default=20000)
    parser.add_argument("--no-detect", action="store_true")
    args = parser.parse_args()

    n = args.stations
    seeds = [int(s) for s in args.seeds.split(",")]
    policies = [p.strip() for p in args.policies.split(",")]
    gate_sweep = [float(g) for g in args.gate_sweep.split(",")]

    print(
        f"Posterior-gated membership, N={n} star, capacity "
        f"{args.capacity}/{n - 1} exchanges/interval, "
        f"{args.iterations} intervals, seeds {seeds}, "
        f"gate {args.gate:.3f} rad"
    )

    # ---- gain bookkeeping over seeds -----------------------------
    runs: dict[tuple[str, int], tuple] = {}
    for policy in policies:
        for seed in seeds:
            settings = SDRSimulationConfig(
                num_iterations=args.iterations, seed=seed, device="cpu"
            )
            result, sigma = run_star_with_posteriors(
                settings,
                num_stations=n,
                policy=policy,
                budgets_rad=[args.flat_budget] * (n - 1),
                max_exchanges_per_interval=args.capacity,
            )
            runs[(policy, seed)] = (result, sigma)

    print(
        "\n=== array gain by membership policy (mean over seeds; "
        "evaluation window = steady or tail quarter) ==="
    )
    header = (
        f"{'sched policy':<12} {'airtime':>8} "
        + "".join(f"{v:>21}" for v in MEMBERSHIP_VARIANTS)
    )
    print(header)
    print(f"{'':<12} {'':>8} " + "".join(
        f"{'G% / G²% / p5%':>21}" for _ in MEMBERSHIP_VARIANTS
    ))
    summary: dict[tuple[str, str], list[tuple[float, float, float]]] = {}
    for policy in policies:
        airtimes = []
        for seed in seeds:
            result, sigma = runs[(policy, seed)]
            airtimes.append(result.airtime_used_fraction)
            mask = evaluation_mask(result)
            phases = phase_matrix(result)[:, mask]
            sig = sigma[:, mask]
            for variant in MEMBERSHIP_VARIANTS:
                weights = membership_weights(
                    variant, phases, sig, args.gate
                )
                gain = weighted_gain(phases, weights)
                summary.setdefault((policy, variant), []).append(
                    summarize(gain)
                )
        cells = []
        for variant in MEMBERSHIP_VARIANTS:
            stats = np.array(summary[(policy, variant)])
            mean_gain, mean_sq, p5 = stats.mean(axis=0)
            cells.append(
                f"{100 * mean_gain:6.1f}/{100 * mean_sq:6.1f}/{100 * p5:5.1f}"
            )
        print(
            f"{policy:<12} {100 * float(np.mean(airtimes)):7.1f}% "
            + "".join(f"{c:>21}" for c in cells)
        )
    print(
        "(G% mean gain, G²% mean squared gain - the detection-relevant "
        "moment, p5% 5th-percentile interval)"
    )

    # ---- gate-threshold sweep (posterior gate, all policies) ------
    print("\n=== posterior gate threshold sweep (mean gain % over seeds) ===")
    print(f"{'sched policy':<12} " + "".join(
        f"{g:>8.2f}" for g in gate_sweep
    ) + f"{'all-in':>9}")
    for policy in policies:
        cells = []
        for gate in gate_sweep:
            gains = []
            for seed in seeds:
                result, sigma = runs[(policy, seed)]
                mask = evaluation_mask(result)
                phases = phase_matrix(result)[:, mask]
                weights = posterior_gate_weights(
                    sigma[:, mask], phases.shape[0], gate
                )
                gains.append(
                    torch.mean(weighted_gain(phases, weights)).item()
                )
            cells.append(f"{100 * float(np.mean(gains)):8.1f}")
        allin = np.mean([
            s[0] for s in summary[(policy, "all-in")]
        ])
        print(f"{policy:<12} " + "".join(cells) + f"{100 * allin:9.1f}")

    if args.no_detect:
        return

    # ---- counted waveform detection, seed 0 ------------------------
    detect_seed = seeds[0]
    params = DetectionParams(tx_power_w=0.5)
    result0, _ = runs[(policies[0], detect_seed)]
    positions = result0.positions
    centroid = positions.mean(axis=0)
    edge_targets = np.array(
        [centroid + [1200.0, 150.0], centroid + [-1200.0, 150.0]]
    )
    print(
        f"\n=== counted edge detection (seed {detect_seed}, "
        f"{args.trials} trials/target, isotropic 1/d legs, per-variant "
        "H0 threshold) ==="
    )
    print(
        f"{'sched policy':<12} {'membership':<10} {'edge Pd':>16} "
        f"{'meas Pfa':>10}"
    )
    for policy in policies:
        result, sigma = runs[(policy, detect_seed)]
        mask = evaluation_mask(result)
        phases = phase_matrix(result)[:, mask]
        sig = sigma[:, mask]
        for variant in ("all-in", "gate", "soft", "oracle"):
            weights = membership_weights(variant, phases, sig, args.gate)
            detect = run_gated_waveform_detection(
                f"{policy}/{variant}",
                positions,
                phases,
                weights,
                edge_targets,
                params=params,
                trials=args.trials,
                h0_trials=args.h0_trials,
                seed=detect_seed,
            )
            print(
                f"{policy:<12} {variant:<10} "
                + " ".join(
                    f"{100 * pd:6.1f}%" for pd in detect.pd_measured
                )
                + f"  {detect.measured_pfa:10.2e}"
            )


if __name__ == "__main__":
    main()
