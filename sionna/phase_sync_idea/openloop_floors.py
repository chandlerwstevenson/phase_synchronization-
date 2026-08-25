"""The floor budget of one open-loop two-way RF carrier-sync link.

Every term is derived ex ante (zero fitted constants), measured in
isolation with a discriminating control, then the assembled budget is
checked for additivity against all-on runs. Frozen oscillator STATES
throughout (the harness of resampling_law_study): every number below
is per-exchange MEASUREMENT floor, not tracking/drift error.

Notation: f_c = 915 MHz carrier, f_s = 1 MHz sample rate, T = 50 ms
exchange interval, dt = TDD turnaround (gap between the two directions
of one exchange), f_D = v f_c / c the max Doppler at environment speed
v, L = long-field length used by the phase estimator, sigma_pn =
per-sample white-FM walk std of the (pair) LO process, sigma_wpm =
white-PM per-sample std, K_eff = specular-to-total-diffuse power ratio
of the channel composite.

TERM 1 - CHANNEL NONRECIPROCITY (the new one). The two directions of
an exchange sample the channel dt apart. The half-difference cancels
only the COMMON channel phase; the part that changed in dt survives at
half weight:

    residual = [phi_c(t) - phi_c(t+dt)] / 2.

Split the composite into specular + diffuse.
  (a) Specular ramp -> BIAS. Sionna's TDL rotates the line-of-sight
      component at f_LOS = f_D cos(pi/4) (los_angle_of_arrival =
      pi/4, verified). So
          bias = pi * f_D * cos(pi/4) * dt        [rad]
      constant across exchanges (same dt every time): a pure bias.
  (b) Diffuse decorrelation -> VARIANCE. Diffuse taps evolve with
      Jakes autocorrelation rho_J = J0(2 pi f_D dt). One-way diffuse
      phase variance is 1/(2 K_eff) (small-perturbation), so
          var[phi_d(t) - phi_d(t+dt)] = (1/K_eff) (1 - rho_J)
      and the half-difference divides variance by 4:
          sigma_nr^2 = (1 - J0(2 pi f_D dt)) / (4 K_eff).
      K_eff is measured from the generated taps (specular power over
      total diffuse power, all taps) - a channel property, not a fit.
  Control: same machinery with the reverse direction reading the SAME
  tap index (dt = 0 in the channel while everything else unchanged)
  must give zero; v = 0 likewise.
  Scope note: the stock simulator holds taps static across the gap
  (documented simplification in coherent.py); this module builds the
  finer-grained channel to measure what that simplification hides.
  Within-capture Doppler remains unmodeled (taps static per frame).

TERM 2 - FRACTIONAL RESAMPLING vs EXCHANGE CADENCE. From
resampling_law.py: the sample-clock carry moves the receiver's
fractional alignment delta by
    step(M) = M * |sfo| * 1e-6 * f_s * T   samples per exchange,
for exchanges every M intervals at fractional clock offset sfo (ppm).
The one-way error from reading the composite at alignment delta has
ensemble variance (1/K_eff)(1 - rho_x(delta - delta')) saturating at
1/(2 K_eff); rho_x is the waveform ambiguity correlation (width ~1
sample). Two-way, two independent-ish reads, half weight:
    sigma_fr^2(M) = min( (1/K_eff)(1 - rho_x(step(M))), 1/(2 K_eff) ) / 2.
Whiteness is cadence-dependent: colored (sawtooth) while step(M) << 1
sample, white once step(M) crosses the ambiguity width. A floor whose
magnitude AND color depend on how often you synchronize.

TERM 3 - INTRA-CAPTURE LO WALK. The capture applies one white-FM walk
(per-sample std sigma_pn, the pair process) across the frame; the
phase estimator reads roughly the walk averaged over the long field,
    var_oneway ~= sigma_pn^2 * (a + W/3)
(window of length W starting a samples into the walk; approximation -
the frequency fit re-weights this, so the module also provides the
numerically exact component value by running the actual estimator on
synthetic walk-only captures: still zero fitted constants). Two-way:
    sigma_ic^2 = 2 * var_oneway / 4.
White-PM adds sigma_wpm^2 / (2 L) two-way (negligible here).

TERM 4 - TURNAROUND WALK. Both oscillators walk across the gap dt
(coherent.py applies walk std sigma_pn sqrt(f_s dt) to each); only the
reverse capture sees it, at half weight:
    sigma_tw^2 = 2 * sigma_pn^2 * f_s * dt / 4.

TERM 5 - CFO-CORRECTION RESIDUE. The receiver removes the
deterministic oscillator advance across dt using its MEASURED
frequency (coherent.py); what remains is the frequency-estimate error
times the gap:
    sigma_cc^2 = (sigma_f_rad^2 / 2) * (dt / 2)^2
with sigma_f_rad the one-way angular-frequency estimate std (from the
ex-ante measurement covariance of the estimator).

TERM 6 - THERMAL. sigma_th^2 = 1 / (4 SNR L) two-way (each direction
1/(2 SNR L), half weight, two directions).

ASSEMBLED: bias = term 1a (+ wrap); variance = quadrature sum of
1b + 2 + 3 + 4 + 5 + 6, checked against all-on runs at grid points
spanning class x speed x turnaround x cadence x SNR.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

from ota_sync.core import REAL_DTYPE, Oscillator, resolve_device, wrap_phase
from ota_sync.oscillators import resolve_oscillator_noise
from ota_sync.sdr import (
    SDRRadioLink,
    SDRSimulationConfig,
    SDRSynchronizer,
    _measurement_covariance,
    make_sync_preamble,
)
from sionna.phy import config as sionna_config
from sionna.phy.channel import cir_to_time_channel
from sionna.phy.channel.tr38901 import TDL

from resampling_law import _quiet_settings

F_CARRIER = 915e6
C_LIGHT = 299792458.0
RESULTS = Path(__file__).resolve().parent / "openloop_floors_results.json"


def bessel_j0(x: float) -> float:
    return float(torch.special.bessel_j0(torch.tensor(float(x), dtype=torch.float64)))


# ------------------------------------------------------------------
# channel with two-time-scale taps: exchanges T apart, directions dt
# apart inside each exchange
# ------------------------------------------------------------------

class FineTapsLink(SDRRadioLink):
    """SDRRadioLink whose per-frame taps come from a finer time grid:
    frame k reads fine index k * fine_stride + fine_offset."""

    fine_taps: torch.Tensor
    fine_stride: int = 1
    fine_offset: int = 0

    def _channel_for_frame(self, iteration: int) -> torch.Tensor:
        output_length = self.input_length + self.l_tot - 1
        index = self.fine_offset + iteration * self.fine_stride
        index = min(index, self.fine_taps.shape[-2] - 1)
        taps = self.fine_taps[..., index, :].unsqueeze(-2)
        return taps.expand(*taps.shape[:-2], output_length, self.l_tot)


def make_fine_taps(
    settings: SDRSimulationConfig,
    link: SDRRadioLink,
    speed_mps: float,
    dt_s: float,
    exchanges: int,
    seed: int,
):
    """Taps on a grid of spacing dt covering `exchanges` intervals,
    plus the measured composite K_eff (specular over total diffuse
    power) of the realization."""

    steps_per_interval = max(1, int(round(settings.sync_interval / dt_s)))
    num_steps = exchanges * steps_per_interval + 2
    torch.manual_seed(seed + 7)
    sionna_config.seed = seed + 7
    tdl = TDL(
        model=settings.tdl_model,
        delay_spread=settings.delay_spread_s,
        carrier_frequency=settings.carrier_frequency_hz,
        min_speed=speed_mps,
        max_speed=speed_mps,
        precision="double",
        device="cpu",
    )
    coefficients, delays = tdl(
        batch_size=1,
        num_time_steps=num_steps,
        sampling_frequency=1.0 / dt_s,
    )
    taps = cir_to_time_channel(
        settings.sample_rate, coefficients, delays, link.l_min, link.l_max,
        normalize=True,
    )
    # composite K_eff from the tap-generation model: specular power of
    # the first cluster over total diffuse power (time-averaged).
    powers = coefficients.abs().square().mean(dim=-1).reshape(-1)
    if bool(tdl.los):
        # Sionna splits the first tap into LOS + diffuse components?
        # It does not expose them separately here; use the spec values:
        # first-tap K factor and per-tap powers give
        # P_los = k/(1+k) * P_tap0, P_diff = P_tap0/(1+k) + sum(rest).
        k_first = float(tdl.k_factor.reshape(-1)[0])
        p0 = float(powers[0])
        p_rest = float(powers.sum()) - p0
        p_los = p0 * k_first / (1.0 + k_first)
        p_diff = p0 / (1.0 + k_first) + p_rest
        k_eff = p_los / p_diff
    else:
        k_eff = float("nan")
    return taps, steps_per_interval, k_eff


# ------------------------------------------------------------------
# the generalized frozen-state two-way harness
# ------------------------------------------------------------------

def run_floor_exchanges(
    seed: int = 0,
    exchanges: int = 120,
    snr_db: float = 40.0,
    cfo_hz: float = 0.0,
    speed_mps: float = 0.0,
    dt_s: float = 0.0,
    cadence_m: int = 1,
    lo_walk_std: float = 0.0,
    lo_wpm_std: float = 0.0,
    turnaround_walk: bool = False,
    cfo_correction: bool = False,
    reverse_same_taps: bool = False,
    tdl_model: str = "D",
):
    """Two-way exchanges, frozen oscillator states. Returns
    (errors, bias-relevant diagnostics, k_eff)."""

    device = resolve_device("cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    base = SDRSimulationConfig(num_iterations=4, seed=seed, device="cpu")
    quiet = _quiet_settings(
        base,
        snr_db=snr_db,
        timing_jitter_samples=0,
        tdl_model=tdl_model,
        num_iterations=4,
        phase_noise_std_rad=lo_walk_std,
        phase_noise_white_pm_std_rad=lo_wpm_std,
    )
    preamble = make_sync_preamble(quiet, device)
    forward = FineTapsLink(quiet, preamble, device, generator)
    reverse = FineTapsLink(quiet, preamble, device, generator, mirror_of=forward)
    synchronizer = SDRSynchronizer(quiet, preamble)

    use_fine = speed_mps > 0.0 and dt_s > 0.0
    if use_fine:
        taps, stride, k_eff = make_fine_taps(
            quiet, forward, speed_mps, dt_s, exchanges, seed
        )
    else:
        # static: one tap set for every frame
        torch.manual_seed(seed + 7)
        sionna_config.seed = seed + 7
        tdl = TDL(
            model=quiet.tdl_model,
            delay_spread=quiet.delay_spread_s,
            carrier_frequency=quiet.carrier_frequency_hz,
            min_speed=0.0,
            max_speed=0.0,
            precision="double",
            device="cpu",
        )
        coefficients, delays = tdl(
            batch_size=1, num_time_steps=1, sampling_frequency=1.0
        )
        taps = cir_to_time_channel(
            quiet.sample_rate, coefficients, delays, forward.l_min,
            forward.l_max, normalize=True,
        )
        stride = 0
        if bool(tdl.los):
            k_first = float(tdl.k_factor.reshape(-1)[0])
            powers = coefficients.abs().square().mean(dim=-1).reshape(-1)
            p0 = float(powers[0])
            p_rest = float(powers.sum()) - p0
            p_los = p0 * k_first / (1.0 + k_first)
            k_eff = p_los / (p0 / (1.0 + k_first) + p_rest)
        else:
            k_eff = float("nan")
    for link, offset in ((forward, 0), (reverse, 0 if reverse_same_taps else 1)):
        link.fine_taps = taps
        link.fine_stride = stride
        link.fine_offset = offset if use_fine else 0

    zero_cov = torch.zeros(2, 2, dtype=REAL_DTYPE, device=device)
    master = Oscillator(0.3, 0.0, quiet.sync_interval, zero_cov, device, generator)
    slave = Oscillator(
        -0.5, 2.0 * math.pi * cfo_hz, quiet.sync_interval, zero_cov, device,
        generator,
    )
    truth = wrap_phase(master.state[0] - slave.state[0]).item()
    sfo = cfo_hz / F_CARRIER * 1e6
    extra_carry = (cadence_m - 1) * sfo * 1e-6 * forward.interval_samples
    gap_walk_std = lo_walk_std * math.sqrt(quiet.sample_rate * dt_s)

    errors = []
    misses = 0
    for iteration in range(exchanges):
        forward._timing_carry += extra_carry
        reverse._timing_carry += -extra_carry
        capture_f = forward.capture(master, slave, iteration, sfo)

        saved_master = master.state.clone()
        saved_slave = slave.state.clone()
        if dt_s > 0.0:
            for osc in (master, slave):
                wander = 0.0
                if turnaround_walk and gap_walk_std > 0.0:
                    wander = float(
                        torch.randn((), dtype=REAL_DTYPE, generator=generator)
                        * gap_walk_std
                    )
                osc.state[0] = wrap_phase(
                    osc.state[0] + osc.state[1] * dt_s + wander
                )
        capture_r = reverse.capture(slave, master, iteration, -sfo)
        master.state = saved_master
        slave.state = saved_slave

        est_f = synchronizer.estimate(capture_f.samples)
        est_r = synchronizer.estimate(capture_r.samples)
        if not (est_f.detected and est_r.detected):
            misses += 1
            continue
        half = wrap_phase((est_f.phase - est_r.phase) / 2.0)
        if cfo_correction and dt_s > 0.0:
            combined_frequency = (est_f.frequency - est_r.frequency) / 2.0
            half = wrap_phase(half - combined_frequency * dt_s / 2.0)
        errors.append(math.remainder(float(half.item()) - truth, math.pi))
    return errors, misses, k_eff


def stats(errors):
    if len(errors) < 8:
        return dict(mean=float("nan"), std=float("nan"), lag1=float("nan"))
    t = torch.tensor(errors, dtype=torch.float64)
    mean = t.mean()
    centered = t - mean
    var = centered.square().mean()
    lag1 = float((centered[:-1] * centered[1:]).mean() / var) if var > 0 else 0.0
    return dict(mean=float(mean), std=float(var.sqrt()), lag1=lag1)


def pooled(rows):
    means = [r["mean"] for r in rows]
    stds = [r["std"] for r in rows]
    lag1s = [r["lag1"] for r in rows]
    n = len(rows)
    return (
        sum(means) / n,
        sum(stds) / n,
        (max(means) - min(means)) / 2.0,
        sum(lag1s) / n,
    )


# ------------------------------------------------------------------
# ex-ante predictors (zero fitted constants)
# ------------------------------------------------------------------

def f_doppler(speed_mps: float) -> float:
    return speed_mps * F_CARRIER / C_LIGHT


def predict_nonreciprocity(speed_mps: float, dt_s: float, k_eff: float):
    fd = f_doppler(speed_mps)
    bias = math.pi * fd * math.cos(math.pi / 4.0) * dt_s
    x = 2.0 * math.pi * fd * dt_s
    var = (1.0 - bessel_j0(x)) / (4.0 * k_eff) if k_eff == k_eff else float("nan")
    return bias, math.sqrt(var) if var == var else float("nan")


def predict_nonreciprocity_exact(
    settings: SDRSimulationConfig,
    speed_mps: float,
    dt_s: float,
    exchanges: int,
    seed: int,
):
    """Realization-exact predictor: the residual computed directly from
    the same generated tap sequence the harness uses (delay spread <<
    one sample here, so the correlator's read is the tap sum). Zero
    fitted constants; captures the realization's actual diffuse
    weighting that the ensemble closed form averages away."""

    device = resolve_device("cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    quiet = _quiet_settings(settings, timing_jitter_samples=0, num_iterations=4)
    preamble = make_sync_preamble(quiet, device)
    link = SDRRadioLink(quiet, preamble, device, generator)
    taps, stride, _ = make_fine_taps(quiet, link, speed_mps, dt_s, exchanges, seed)
    flat = taps.reshape(-1, taps.shape[-2], taps.shape[-1])[0]
    phases = torch.angle(flat.sum(dim=-1))
    idx = torch.arange(exchanges) * stride
    resid = 0.5 * (
        torch.remainder(phases[idx] - phases[idx + 1] + math.pi, 2 * math.pi)
        - math.pi
    )
    return float(resid.mean()), float(resid.std())


def carry_sequence(step: float, exchanges: int):
    """The fractional alignments the receiver actually visits: the
    capture adds `step` to its carry, then re-centers on the nearest
    integer (drift absorbed by the correlator's argmax), leaving the
    fractional residue as the resampler input."""

    carry = 0.0
    visited = []
    for _ in range(exchanges):
        carry += step
        carry -= round(carry)
        visited.append(carry)
    return visited


def predict_fractional_exact(
    cadence_m: int,
    cfo_hz: float,
    exchanges: int,
    seed: int,
    settings: SDRSimulationConfig,
):
    """Realization-exact fractional-resampling predictor: evaluate the
    noiseless alignment-phase profile of the SAME static channel draw
    the harness uses (taps seeded seed+7), at the exact carry sequence
    the two directions visit, and compose the half-difference. Zero
    fitted constants; the estimator itself defines the profile."""

    device = resolve_device("cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    base = SDRSimulationConfig(num_iterations=4, seed=seed, device="cpu")
    quiet = _quiet_settings(
        base, snr_db=200.0, timing_jitter_samples=0, num_iterations=4
    )
    preamble = make_sync_preamble(quiet, device)
    link = FineTapsLink(quiet, preamble, device, generator)
    torch.manual_seed(seed + 7)
    sionna_config.seed = seed + 7
    tdl = TDL(
        model=quiet.tdl_model,
        delay_spread=quiet.delay_spread_s,
        carrier_frequency=quiet.carrier_frequency_hz,
        min_speed=0.0, max_speed=0.0, precision="double", device="cpu",
    )
    coefficients, delays = tdl(batch_size=1, num_time_steps=1,
                               sampling_frequency=1.0)
    link.fine_taps = cir_to_time_channel(
        quiet.sample_rate, coefficients, delays, link.l_min, link.l_max,
        normalize=True,
    )
    link.fine_stride = 0
    link.fine_offset = 0
    synchronizer = SDRSynchronizer(quiet, preamble)
    zero_cov = torch.zeros(2, 2, dtype=REAL_DTYPE, device=device)
    master = Oscillator(0.3, 0.0, quiet.sync_interval, zero_cov, device,
                        generator)
    slave = Oscillator(-0.5, 0.0, quiet.sync_interval, zero_cov, device,
                       generator)

    sfo = cfo_hz / F_CARRIER * 1e6
    step = cadence_m * sfo * 1e-6 * link.interval_samples

    def profile(deltas):
        phases = []
        for delta in deltas:
            link._timing_carry = float(delta)
            capture = link.capture(master, slave, 0, 0.0)
            est = synchronizer.estimate(capture.samples)
            phases.append(float(est.phase.item()) if est.detected
                          else float("nan"))
        return phases

    forward_p = profile(carry_sequence(step, exchanges))
    reverse_p = profile(carry_sequence(-step, exchanges))
    half = torch.tensor(
        [0.5 * (f - r) for f, r in zip(forward_p, reverse_p)],
        dtype=torch.float64,
    )
    half = half[~torch.isnan(half)]
    centered = half - half.mean()
    var = centered.square().mean()
    lag1 = float((centered[:-1] * centered[1:]).mean() / var) if var > 0 else 0.0
    return float(var.sqrt()), abs(step), lag1


def predict_intracapture_mc(lo_walk_std: float, lo_wpm_std: float,
                            draws: int = 60, seed: int = 123):
    """Numerically exact one-way component: actual estimator on quiet
    captures with only the LO processes enabled (static channel, no
    thermal: SNR 200 dB)."""

    errors, _, _ = run_floor_exchanges(
        seed=seed, exchanges=draws, snr_db=200.0, cfo_hz=0.0,
        lo_walk_std=lo_walk_std, lo_wpm_std=lo_wpm_std,
    )
    return stats(errors)["std"]


def predict_intracapture_analytic(lo_walk_std: float, preamble_len: int):
    return lo_walk_std * math.sqrt(preamble_len / 3.0) * math.sqrt(2.0) / 2.0


def predict_turnaround_walk(lo_walk_std: float, dt_s: float,
                            sample_rate: float = 1e6):
    return math.sqrt(2.0 * lo_walk_std**2 * sample_rate * dt_s / 4.0)


def predict_cfo_correction(dt_s: float, settings, preamble, device):
    cov = _measurement_covariance(settings, preamble, device)
    sigma_f = math.sqrt(float(cov[2, 2]))  # rad/s, one direction
    return (sigma_f / math.sqrt(2.0)) * dt_s / 2.0


def predict_thermal(snr_db: float, settings: SDRSimulationConfig):
    snr = 10.0 ** (snr_db / 10.0)
    long_total = settings.long_sequence_length * settings.long_repetitions
    return math.sqrt(1.0 / (4.0 * snr * long_total))


# ------------------------------------------------------------------
# the study
# ------------------------------------------------------------------

def main() -> None:
    quick = "--quick" in sys.argv
    seeds = [0] if quick else [0, 1, 2]
    exchanges = 60 if quick else 120
    results = {}
    base = SDRSimulationConfig(device="cpu")
    device = resolve_device("cpu")
    quiet_for_pred = _quiet_settings(base, timing_jitter_samples=0, num_iterations=4)
    preamble = make_sync_preamble(quiet_for_pred, device)
    interval_samples = int(round(base.sync_interval * base.sample_rate))
    classes = {
        name: resolve_oscillator_noise(
            name, base.carrier_frequency_hz, base.sample_rate, base.sync_interval
        )[0]
        for name in ("ocxo", "tcxo", "sdr")
    }

    # ---------------- PART 1: channel nonreciprocity ----------------
    print("=== PART 1: channel nonreciprocity floor ===")
    print("predictions FIRST (bias = pi cos(pi/4) f_D dt; "
          "std = sqrt((1-J0(2 pi f_D dt))/(4 K_eff)); K_eff from taps)")
    grid1 = [
        (1.0, 1e-3), (3.0, 1e-3), (10.0, 1e-3), (30.0, 1e-3),
        (3.0, 1e-4), (3.0, 1e-2), (1.0, 1e-2),
    ]
    # K_eff is realization-independent in expectation; take it from one draw
    _, _, k_probe = run_floor_exchanges(seed=0, exchanges=8, speed_mps=1.0,
                                        dt_s=1e-3)
    print(f"measured composite K_eff (TDL-D): {k_probe:.1f} "
          f"(= {10*math.log10(k_probe):.1f} dB)")
    predictions1 = {}
    print(f"{'v m/s':>6} {'dt ms':>6} | {'pred bias mrad':>14} {'pred std mrad':>13}")
    for v, dt in grid1:
        bias, std = predict_nonreciprocity(v, dt, k_probe)
        predictions1[(v, dt)] = (bias, std)
        print(f"{v:>6.0f} {dt*1e3:>6.1f} | {1e3*bias:>14.1f} {1e3*std:>13.1f}")

    print("\nmeasurements (frozen oscillators, CFO 0, SNR 40 dB, "
          f"{exchanges} exchanges x {len(seeds)} seeds; exact = per-seed "
          "realization predictor from the tap sequence itself):")
    print(f"{'v m/s':>6} {'dt ms':>6} | {'meas bias':>10} {'meas std':>9} "
          f"{'pred bias':>10} {'pred std':>9} {'exact bias':>10} "
          f"{'exact std':>9} {'lag1':>6}")
    for v, dt in grid1:
        rows = [stats(run_floor_exchanges(
            seed=s, exchanges=exchanges, speed_mps=v, dt_s=dt)[0])
            for s in seeds]
        exact = [predict_nonreciprocity_exact(base, v, dt, exchanges, s)
                 for s in seeds]
        exact_bias = sum(e[0] for e in exact) / len(exact)
        exact_std = sum(e[1] for e in exact) / len(exact)
        mean, std, spread, lag1 = pooled(rows)
        pb, ps = predictions1[(v, dt)]
        results[f"p1_v{v}_dt{dt}"] = dict(
            meas_bias=1e3*mean, meas_std=1e3*std, pred_bias=1e3*pb,
            pred_std=1e3*ps, exact_bias=1e3*exact_bias,
            exact_std=1e3*exact_std, lag1=lag1)
        print(f"{v:>6.0f} {dt*1e3:>6.1f} | {1e3*mean:>10.1f} {1e3*std:>9.1f} "
              f"{-1e3*pb:>10.1f} {1e3*ps:>9.1f} {1e3*exact_bias:>10.1f} "
              f"{1e3*exact_std:>9.1f} {lag1:>6.2f}")

    print("\ncontrols:")
    for label, kwargs in (
        ("v=0, dt=1ms (machinery null)", dict(speed_mps=0.0, dt_s=1e-3)),
        ("v=3, dt=1ms, SAME taps to reverse", dict(
            speed_mps=3.0, dt_s=1e-3, reverse_same_taps=True)),
    ):
        rows = [stats(run_floor_exchanges(
            seed=s, exchanges=exchanges, **kwargs)[0]) for s in seeds]
        mean, std, _, _ = pooled(rows)
        results[f"p1_ctrl_{label[:12]}"] = dict(mean=1e3*mean, std=1e3*std)
        print(f"  {label:<38} bias {1e3*mean:>7.2f}  std {1e3*std:>6.2f} mrad")

    # ---------------- PART 2: fractional resampling vs cadence -------
    print("\n=== PART 2: fractional-resampling floor vs exchange cadence ===")
    cfo = 1500.0
    print("predictions FIRST (realization-exact: noiseless alignment "
          "profile of the same channel draw, at the exact visited carry "
          "sequence per direction)")
    grid2 = [1, 2, 4, 8, 16, 32]
    predictions2 = {}
    print(f"{'M':>4} {'step smp':>9} | {'pred std mrad':>13} {'pred lag1':>9}")
    for m in grid2:
        per_seed = [predict_fractional_exact(m, cfo, exchanges, s,
                                             quiet_for_pred)
                    for s in seeds]
        sigma = sum(p[0] for p in per_seed) / len(per_seed)
        step = per_seed[0][1]
        plag = sum(p[2] for p in per_seed) / len(per_seed)
        predictions2[m] = (sigma, step, plag)
        print(f"{m:>4} {step:>9.2f} | {1e3*sigma:>13.1f} {plag:>9.2f}")
    print(f"\nmeasurements (frozen oscillators, static channel, CFO {cfo:.0f} Hz,"
          " SNR 40 dB):")
    print(f"{'M':>4} | {'meas std':>9} {'pred std':>9} {'meas lag1':>9} "
          f"{'pred lag1':>9}")
    for m in grid2:
        rows = [stats(run_floor_exchanges(
            seed=s, exchanges=exchanges, cfo_hz=cfo, cadence_m=m,
            snr_db=40.0)[0]) for s in seeds]
        _, std, _, lag1 = pooled(rows)
        sigma, step, plag = predictions2[m]
        results[f"p2_m{m}"] = dict(meas_std=1e3*std, pred_std=1e3*sigma,
                                   lag1=lag1, pred_lag1=plag, step=step)
        print(f"{m:>4} | {1e3*std:>9.1f} {1e3*sigma:>9.1f} {lag1:>9.2f} "
              f"{plag:>9.2f}")

    # ---------------- PART 3: intra-capture walk ---------------------
    print("\n=== PART 3: intra-capture LO-walk floor per oscillator class ===")
    print("predictions FIRST (analytic approx + numerically exact component)")
    predictions3 = {}
    print(f"{'class':>6} {'sigma_pn':>9} | {'analytic mrad':>13} {'exact mrad':>10}")
    for name, noise in classes.items():
        pn = noise["phase_noise_std_rad"]
        wpm = noise["phase_noise_white_pm_std_rad"]
        analytic = predict_intracapture_analytic(pn, preamble.length)
        exact = predict_intracapture_mc(pn, wpm, draws=40 if quick else 80)
        predictions3[name] = exact
        print(f"{name:>6} {pn:>9.1e} | {1e3*analytic:>13.1f} {1e3*exact:>10.1f}")
    print("\nmeasurements (fresh seeds, thermal on at 40 dB, static, CFO 0):")
    print(f"{'class':>6} | {'meas std':>9} {'pred std (exact+thermal)':>24}")
    thermal40 = predict_thermal(40.0, base)
    for name, noise in classes.items():
        rows = [stats(run_floor_exchanges(
            seed=s + 50, exchanges=exchanges, snr_db=40.0,
            lo_walk_std=noise["phase_noise_std_rad"],
            lo_wpm_std=noise["phase_noise_white_pm_std_rad"])[0])
            for s in seeds]
        _, std, _, _ = pooled(rows)
        pred = math.sqrt(predictions3[name]**2 + thermal40**2)
        results[f"p3_{name}"] = dict(meas_std=1e3*std, pred_std=1e3*pred)
        print(f"{name:>6} | {1e3*std:>9.1f} {1e3*pred:>24.1f}")

    # ---------------- PART 4: assembled budget -----------------------
    print("\n=== PART 4: assembled budget at grid points ===")
    grid4 = [
        ("tcxo", 0.0, 1e-3, 1, 20.0),
        ("tcxo", 3.0, 1e-3, 1, 40.0),
        ("tcxo", 3.0, 1e-2, 1, 40.0),
        ("tcxo", 0.0, 1e-3, 8, 20.0),
        ("ocxo", 3.0, 1e-2, 1, 40.0),
        ("sdr", 0.0, 1e-3, 1, 20.0),
        ("tcxo", 3.0, 1e-2, 8, 20.0),
        ("ocxo", 0.0, 1e-4, 1, 40.0),
    ]
    print("predictions FIRST (nonreciprocity + fractional terms use the "
          "seed-matched realization-exact predictors):")
    header = (f"{'class':>5} {'v':>3} {'dt ms':>6} {'M':>3} {'SNR':>4} | "
              f"{'bias':>7} {'std':>7}   (all mrad)")
    print(header)
    predictions4 = {}
    for name, v, dt, m, snr in grid4:
        noise = classes[name]
        pn = noise["phase_noise_std_rad"]
        if v > 0:
            b_nr, _ = predict_nonreciprocity(v, dt, k_probe)
            exact = [predict_nonreciprocity_exact(base, v, dt, exchanges,
                                                  s + 100)
                     for s in seeds]
            s_nr = sum(e[1] for e in exact) / len(exact)
        else:
            b_nr, s_nr = 0.0, 0.0
        fr = [predict_fractional_exact(m, cfo, exchanges, s + 100,
                                       quiet_for_pred) for s in seeds]
        s_fr = sum(p[0] for p in fr) / len(fr)
        s_ic = predictions3[name]
        s_tw = predict_turnaround_walk(pn, dt)
        s_cc = predict_cfo_correction(dt, quiet_for_pred, preamble, device)
        s_th = predict_thermal(snr, base)
        total_std = math.sqrt(s_nr**2 + s_fr**2 + s_ic**2 + s_tw**2
                              + s_cc**2 + s_th**2)
        predictions4[(name, v, dt, m, snr)] = (b_nr, total_std)
        print(f"{name:>5} {v:>3.0f} {dt*1e3:>6.1f} {m:>3} {snr:>4.0f} | "
              f"{-1e3*b_nr:>7.1f} {1e3*total_std:>7.1f}")
    print("\nmeasurements (everything on: fine taps, CFO 1500, turnaround "
          "advance + walk + CFO correction, class LO noise):")
    print(f"{'class':>5} {'v':>3} {'dt ms':>6} {'M':>3} {'SNR':>4} | "
          f"{'meas bias':>9} {'meas std':>8} {'pred bias':>9} {'pred std':>8} "
          f"{'ratio':>6}")
    for name, v, dt, m, snr in grid4:
        noise = classes[name]
        rows = [stats(run_floor_exchanges(
            seed=s + 100, exchanges=exchanges, snr_db=snr, cfo_hz=cfo,
            speed_mps=v, dt_s=dt, cadence_m=m,
            lo_walk_std=noise["phase_noise_std_rad"],
            lo_wpm_std=noise["phase_noise_white_pm_std_rad"],
            turnaround_walk=True, cfo_correction=True)[0])
            for s in seeds]
        mean, std, _, _ = pooled(rows)
        pb, ps = predictions4[(name, v, dt, m, snr)]
        ratio = std / ps if ps > 0 else float("nan")
        results[f"p4_{name}_v{v}_dt{dt}_m{m}_snr{snr}"] = dict(
            meas_bias=1e3*mean, meas_std=1e3*std, pred_bias=-1e3*pb,
            pred_std=1e3*ps, ratio=ratio)
        print(f"{name:>5} {v:>3.0f} {dt*1e3:>6.1f} {m:>3} {snr:>4.0f} | "
              f"{1e3*mean:>9.1f} {1e3*std:>8.1f} {-1e3*pb:>9.1f} "
              f"{1e3*ps:>8.1f} {ratio:>6.2f}")

    RESULTS.write_text(json.dumps(results, indent=1))
    print(f"\nsaved {RESULTS.name}")


if __name__ == "__main__":
    main()
