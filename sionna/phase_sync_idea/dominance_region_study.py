"""Dominance-region map for the "multipath resampling" term - and a
mechanism check that the term's story must survive first.

Reviewer's demand: "dominant term" needs a sweep over oscillator
quality x timing jitter x SNR x channel; and the claimed mechanism
(timing jitter re-samples the static multipath composite) must be
established, not asserted.

FINDING THAT RESHAPED THIS STUDY (part 1): with oscillators frozen,
integer-sample timing jitter over the frozen multipath composite
produces ~0.2-0.3 mrad of half-difference noise - three orders of
magnitude below the ~100-150 mrad attributed to "resampling".
The synchronizer's correlation-based timing recovery removes integer
sample shifts, so jitter x static multipath is NOT the mechanism of
the loop's excess noise in this simulator. Part 1b therefore
diagnoses what the excess DOES come from (intra-capture LO walk,
channel dependence), and parts 2-4 build the budget map from the
LOOP-INFERRED excess (mechanism-agnostic), not the refuted number.

Terms, all in half-difference measurement units:
  drift     2 * sigma_pn^2 * f_s * T  (pair walk per interval, ex ante)
  thermal   0.5 / (2 * SNR * L * reps)  (ex ante)
  excess    inferred per cell: the extra measurement variance e that
            makes the steady-state Kalman prediction match the
            measured loop residual (one number per cell, inverted by
            bisection - NOT fitted across budgets)

Part 4 then tests the overconfidence signature sqrt(b^2 + e)/b
across five budgets using each corner's inferred e - an
out-of-sample test of whether e behaves as white measurement noise.
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from coast_law import dare_posterior, link_matrices, station_snr_db
from ota_sync import SDRSimulationConfig
from ota_sync.core import Oscillator, wrap_phase, REAL_DTYPE, resolve_device
from ota_sync.oscillators import resolve_oscillator_noise
from ota_sync.scheduled import run_scheduled_star
from ota_sync.sdr import SDRRadioLink, SDRSynchronizer, make_sync_preamble

CLASSES = ["ocxo", "tcxo", "sdr"]
JITTERS = [2, 8, 32]
SNRS = [10.0, 20.0, 30.0]
CHANNELS = ["E", "D", "A"]
T = 0.05
FS = 1e6


def pair_halfdiff_error_var(
    tdl_model: str,
    jitter: int,
    snr_db: float,
    seed: int,
    reps: int = 240,
    lo_walk_std: float = 0.0,
    delay_spread_s: float = 100e-9,
) -> float:
    """Variance of (two-way half-difference minus true oscillator
    offset) with oscillator PROCESS noise frozen. lo_walk_std > 0
    enables only the intra-capture LO random walk."""

    settings = SDRSimulationConfig(
        seed=seed,
        device="cpu",
        num_iterations=reps,
        snr_db=snr_db,
        tdl_model=tdl_model,
        delay_spread_s=delay_spread_s,
        timing_jitter_samples=jitter,
        phase_noise_std_rad=lo_walk_std,
        phase_process_std_rad=0.0,
        frequency_process_std_hz=0.0,
        flicker_frequency_std_hz=0.0,
        slave_initial_frequency_hz=0.0,
        slave_initial_phase=0.7,
    )
    device = resolve_device("cpu")
    torch.manual_seed(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 1)
    preamble = make_sync_preamble(settings, device)
    zero_cov = torch.zeros((2, 2), dtype=REAL_DTYPE, device=device)
    reference = Oscillator(0.0, 0.0, T, zero_cov, device, generator)
    slave = Oscillator(0.7, 0.0, T, zero_cov, device, generator)
    forward = SDRRadioLink(settings, preamble, device, generator)
    reverse = SDRRadioLink(
        settings, preamble, device, generator, mirror_of=forward
    )
    synchronizer = SDRSynchronizer(settings, preamble)
    errors = []
    for iteration in range(reps):
        truth = wrap_phase(reference.state[0] - slave.state[0])
        cap_f = forward.capture(reference, slave, iteration, 0.0)
        reference.state[0] = wrap_phase(
            reference.state[0] + cap_f.lo_walk_end
        )
        cap_r = reverse.capture(slave, reference, iteration, 0.0)
        slave.state[0] = wrap_phase(slave.state[0] + cap_r.lo_walk_end)
        est_f = synchronizer.estimate(cap_f.samples)
        est_r = synchronizer.estimate(cap_r.samples)
        if not (est_f.detected and est_r.detected):
            continue
        half = wrap_phase(wrap_phase(est_f.phase - est_r.phase) / 2.0)
        errors.append(wrap_phase(half - truth).item())
    if len(errors) < reps // 2:
        return float("nan")
    tensor = torch.tensor(errors, dtype=torch.float64)
    centered = wrap_phase(tensor - tensor.mean())
    return centered.var().item()


def drift_var(profile: str) -> float:
    noise, _ = resolve_oscillator_noise(profile, 915e6, FS, T)
    return 2.0 * noise["phase_noise_std_rad"] ** 2 * FS * T


def lo_walk_of(profile: str) -> float:
    noise, _ = resolve_oscillator_noise(profile, 915e6, FS, T)
    return noise["phase_noise_std_rad"]


def thermal_var(snr_db: float, settings: SDRSimulationConfig) -> float:
    snr = 10.0 ** (snr_db / 10.0)
    return 0.5 / (
        2.0 * snr * settings.long_sequence_length * settings.long_repetitions
    )


def predicted_prior_std(
    settings: SDRSimulationConfig,
    profile: str,
    extra_var: float,
) -> float:
    link_snr = station_snr_db(settings, 2, 1)
    matrices = link_matrices(
        settings, profile, link_snr, 60 * T,
        reference_profile=profile,
        extra_phase_measurement_var=extra_var,
    )
    posterior = dare_posterior(matrices)
    prior = (
        matrices.transition @ posterior @ matrices.transition.T
        + matrices.process
    )
    return math.sqrt(max(float(prior[0, 0]), 0.0))


def infer_excess(
    settings: SDRSimulationConfig, profile: str, measured: float
) -> float:
    """Bisect the extra measurement variance that reproduces the
    measured steady residual; 0 if the plain model already covers it."""

    if predicted_prior_std(settings, profile, 0.0) >= measured:
        return 0.0
    low, high = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (low + high)
        if predicted_prior_std(settings, profile, mid) < measured:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def measured_loop_rms(
    profile: str, jitter: int, snr: float, channel: str, seeds=range(3)
) -> float:
    values = []
    for seed in seeds:
        settings = SDRSimulationConfig(
            num_iterations=60, seed=seed, device="cpu",
            snr_db=snr, tdl_model=channel,
            timing_jitter_samples=jitter,
        )
        result = run_scheduled_star(
            settings, num_stations=2, policy="uniform",
            oscillator_profiles=[profile, profile],
        )
        rms = result.station_steady_rms[0]
        if rms == rms:
            values.append(rms)
    return sum(values) / len(values)


def main() -> None:
    base = SDRSimulationConfig(device="cpu")

    print("PART 1 - mechanism check: frozen oscillators, integer-sample "
          "jitter over frozen multipath (seeds 0-1 pooled, error vs truth)")
    print(f"{'channel':>8} {'jitter':>7} {'error std (mrad)':>17}")
    for channel in CHANNELS:
        for jitter in (0, 32):
            var = sum(
                pair_halfdiff_error_var(channel, jitter, 30.0, seed)
                for seed in (0, 1)
            ) / 2.0
            print(f"{channel:>8} {jitter:>7} {1e3 * math.sqrt(var):>17.2f}")

    print("\nPART 1b - what does produce capture-level noise: intra-capture "
          "LO walk on/off x channel x jitter (tcxo walk, seeds 0-1)")
    walk = lo_walk_of("tcxo")
    print(f"  (tcxo per-sample LO walk std = {1e6 * walk:.1f} urad)")
    print(f"{'config':>44} {'error std (mrad)':>17}")
    for label, kwargs in [
        ("walk ON, TDL-D, jitter 32", dict(lo_walk_std=walk)),
        ("walk ON, TDL-D, jitter 0",
         dict(lo_walk_std=walk, jitter_override=0)),
        ("walk ON, ~no multipath (1 ns), jitter 32",
         dict(lo_walk_std=walk, delay_spread_s=1e-9)),
        ("walk OFF, TDL-D, jitter 32", dict()),
    ]:
        jitter = kwargs.pop("jitter_override", 32)
        var = sum(
            pair_halfdiff_error_var(
                "D", jitter, 30.0, seed, **kwargs
            )
            for seed in (0, 1)
        ) / 2.0
        print(f"{label:>44} {1e3 * math.sqrt(var):>17.2f}")

    print("\nPART 3 - loop-level excess, inferred per corner cell "
          "(N=2 serviced-every-interval, 60 intervals, seeds 0-2)")
    corners = [
        ("ocxo", 32, 20.0, "D"), ("tcxo", 32, 20.0, "D"),
        ("sdr", 32, 20.0, "D"),
        ("tcxo", 2, 20.0, "D"), ("tcxo", 8, 20.0, "D"),
        ("tcxo", 32, 20.0, "E"), ("tcxo", 32, 20.0, "A"),
        ("tcxo", 32, 10.0, "D"), ("tcxo", 32, 30.0, "D"),
        ("ocxo", 2, 30.0, "E"), ("sdr", 32, 10.0, "A"),
        ("ocxo", 8, 20.0, "D"),
    ]
    print(f"{'cell':>20} {'measured':>9} {'pred(no excess)':>15} "
          f"{'inferred excess std':>20}")
    excess = {}
    for profile, jitter, snr, channel in corners:
        measured = measured_loop_rms(profile, jitter, snr, channel)
        settings = SDRSimulationConfig(
            device="cpu", snr_db=snr, tdl_model=channel,
            timing_jitter_samples=jitter,
        )
        plain = predicted_prior_std(settings, profile, 0.0)
        e_var = infer_excess(settings, profile, measured)
        excess[(profile, jitter, snr, channel)] = e_var
        print(f"{profile:>6}/J{jitter:<2}/S{int(snr):<2}/{channel}"
              f"{'':>3} {1e3 * measured:>8.1f} {1e3 * plain:>15.1f} "
              f"{1e3 * math.sqrt(e_var):>20.1f}")

    print("\nPART 2 - dominance map (81 cells, ex-ante drift/thermal + "
          "excess interpolated from the corner contrasts)")
    # Scaling observed in part 3 drives how excess extends across the
    # grid; use the tcxo/J*/S20/D jitter row and the channel row, and
    # assume class-independence unless part 3 contradicts it (checked
    # in the printout above). Nearest-measured-cell assignment:
    def nearest_excess(profile, jitter, snr, channel):
        keys = list(excess)
        def distance(key):
            kp, kj, ks, kc = key
            return (
                (kp != profile) * 1
                + abs(math.log2(max(kj, 1) / max(jitter, 1)))
                + abs(ks - snr) / 10.0
                + (kc != channel) * 1
            )
        return excess[min(keys, key=distance)]

    print(f"{'class':>6} {'chan':>5}  " + "  ".join(
        f"J{jitter}/S{int(snr)}" for jitter in JITTERS for snr in SNRS
    ))
    for profile in CLASSES:
        d_var = drift_var(profile)
        for channel in CHANNELS:
            winners = []
            for jitter in JITTERS:
                for snr in SNRS:
                    terms = {
                        "D": d_var,
                        "T": thermal_var(snr, base),
                        "X": nearest_excess(profile, jitter, snr, channel),
                    }
                    winner = max(terms, key=terms.get)
                    runner = max(
                        v for k, v in terms.items() if k != winner
                    )
                    margin = terms[winner] / max(runner, 1e-12)
                    winners.append(f"{winner}{min(margin, 99):4.1f}x")
            print(f"{profile:>6} {channel:>5}  " + "  ".join(
                f"{w:>7}" for w in winners
            ))
    for profile in CLASSES:
        print(f"  drift std {profile}: "
              f"{1e3 * math.sqrt(drift_var(profile)):6.1f} mrad/interval")

    print("\nPART 4 - overconfidence signature sqrt(b^2+e)/b across "
          "budgets at three corners, using each corner's inferred e "
          "(out-of-sample: e inferred from uniform-service runs, "
          "tested on scheduled-coasting runs)")
    signature_corners = [
        ("tcxo", 32, 20.0, "D"),
        ("tcxo", 32, 20.0, "E"),
        ("tcxo", 2, 20.0, "D"),
    ]
    budgets = [0.15, 0.2, 0.314, 0.45, 0.6]
    for profile, jitter, snr, channel in signature_corners:
        e_var = excess[(profile, jitter, snr, channel)]
        print(f"  corner {profile}/J{jitter}/{channel}: inferred excess "
              f"std {1e3 * math.sqrt(e_var):.0f} mrad")
        print(f"    {'budget':>7} {'meas ratio':>11} {'pred ratio':>11}")
        for budget in budgets:
            ratios = []
            for seed in range(3):
                settings = SDRSimulationConfig(
                    num_iterations=150, seed=seed, device="cpu",
                    snr_db=snr, tdl_model=channel,
                    timing_jitter_samples=jitter,
                )
                result = run_scheduled_star(
                    settings, num_stations=2, policy="scheduled",
                    trigger_fraction=1.0,
                    budgets_rad=[budget],
                    oscillator_profiles=[profile, profile],
                )
                serviced = torch.nonzero(result.serviced[0]).flatten()
                kept = serviced[6:]
                if kept.numel() < 5:
                    continue
                values = [
                    abs(result.residuals[0, t].item()) for t in kept
                ]
                rms = math.sqrt(sum(v * v for v in values) / len(values))
                ratios.append(rms / budget)
            if not ratios:
                continue
            print(f"    {budget:>7.3f} "
                  f"{sum(ratios) / len(ratios):>11.2f} "
                  f"{math.sqrt(budget**2 + e_var) / budget:>11.2f}")


if __name__ == "__main__":
    main()
