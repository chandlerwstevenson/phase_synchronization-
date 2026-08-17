"""Why scheduled coasting breaks under channel Doppler - and the
missing coast-rule term (RESEARCH_IDEAS.md idea #2, "motion" bullet).

heterogeneous_fleet_study.py part 2 measured the symptom: at 3 m/s the
scheduled policy's residuals blow past budget (145 -> 227+ mrad) while
uniform holds. This study finds the mechanism and derives/validates
the missing term, WITHOUT modifying any existing file (recording
subclasses are swapped into ota_sync.scheduled's namespace for the
duration of a run, exactly like gating_study.py's RecordingEKF; the
default path is regression-tested bit-identical).

Mechanism candidates the instrumentation separates:
  M1  fade outages: a serviced exchange fails detection, the station
      coasts an extra interval per miss (airtime spent, no update).
  M2  fade over-trust: the EKF's measurement covariance is FIXED at
      the link's nominal SNR, so a pilot taken in a fade is trusted
      as if it were clean; the corrupted frequency estimate is then
      integrated over the whole coast horizon h = tau + L*T. The
      believed posterior lies exactly when coasting relies on it.

The candidate missing term (M2, fit-free counterfactual): re-run each
recorded EKF update with the TRUE fade-scaled measurement covariance
R(gamma) = R_nom * (gamma_nom / gamma_hat), where gamma_hat comes
from the detector's own normalized correlation metric m via
gamma_hat = m^2 / (1 - m^2). The missing variance over a coast window
of horizon h is

    Delta sigma^2(h) ~= (P_true[0,0] - P_bel[0,0])
                      + h^2 * (P_true[1,1] - P_bel[1,1])

i.e. the error-floor formula's tracking + latency terms with the
posterior evaluated under the true fade SNR instead of the nominal
one. No fitted constants anywhere: R scaling comes from the metric,
the Kalman gain from the recorded update.

The corrected rule follows immediately: make R SNR-adaptive from the
same metric (a quantity every real preamble detector already
computes). The posterior then widens honestly after a faded pilot, the
existing predicted-std trigger re-services the station sooner, and the
coast rule needs no other change.

Usage:
    .venv/bin/python doppler_coast_study.py                # full study
    .venv/bin/python doppler_coast_study.py --speeds 0,3 --seeds 0
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

import ota_sync.scheduled as scheduled_module
from ota_sync import SDRSimulationConfig
from ota_sync.core import PhaseFrequencyEKF
from ota_sync.sdr import SDRRadioLink, SDRSynchronizer


# ---------------------------------------------------------------------
# Outside-in instrumentation
# ---------------------------------------------------------------------

class RecordingSynchronizer(SDRSynchronizer):
    """Logs every SDRMeasurement; one instance per link, forward and
    reverse estimates of one exchange arrive as consecutive events."""

    instances: list["RecordingSynchronizer"] = []

    def __init__(self, settings, preamble):
        super().__init__(settings, preamble)
        self.events = []
        RecordingSynchronizer.instances.append(self)

    def estimate(self, samples):
        measurement = super().estimate(samples)
        self.events.append(measurement)
        _FADE_BLACKBOARD.append(
            (measurement.detection_metric, self.settings.snr_db)
        )
        return measurement


class RecordingRadioLink(SDRRadioLink):
    """Registers instances so channel taps are readable post-run, and
    logs the true oscillator phases at each capture so the channel-
    induced measurement bias is computable offline. Creation order
    inside run_scheduled_star: forward, reverse per link - forward
    links are instances 0, 2, 4, ..."""

    instances: list["RecordingRadioLink"] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capture_truth: list[tuple[int, float, float]] = []
        RecordingRadioLink.instances.append(self)

    def capture(self, master, slave, iteration, sfo_ppm):
        self.capture_truth.append(
            (iteration, master.state[0].item(), slave.state[0].item())
        )
        return super().capture(master, slave, iteration, sfo_ppm)


def _metric_to_snr(metric: float) -> float:
    """Per-sample SNR estimate from the detector's normalized
    correlation, gamma = m^2/(1-m^2)."""

    m = min(max(metric, 1e-6), 0.999999)
    return m * m / (1.0 - m * m)


_FADE_BLACKBOARD: list[tuple[float, float]] = []


class RecordingEKF(PhaseFrequencyEKF):
    """Logs post-predict stds and, per update, everything needed to
    replay the update under a different measurement covariance.

    extra_phase_process_var, when nonzero, is added to the phase entry
    of the process covariance: the channel-decorrelation term. The
    posterior then grows at the rate the reciprocity bias actually
    wanders, so the existing predicted-std trigger re-services coasting
    stations before the stale bias eats the budget."""

    instances: list["RecordingEKF"] = []
    fade_aware: bool = False
    max_scale: float = 1000.0
    extra_phase_process_var: float = 0.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if RecordingEKF.extra_phase_process_var > 0.0:
            bump = torch.zeros_like(self.process_covariance)
            bump[0, 0] = RecordingEKF.extra_phase_process_var
            self.process_covariance = self.process_covariance + bump
        self.base_measurement_covariance = self.measurement_covariance.clone()
        self.predicted_phase_std: list[float] = []
        self.predicted_freq_std: list[float] = []
        self.updates: list[dict] = []
        RecordingEKF.instances.append(self)

    def predict(self) -> None:
        super().predict()
        self.predicted_phase_std.append(
            math.sqrt(max(self.covariance[0, 0].item(), 0.0))
        )
        self.predicted_freq_std.append(
            math.sqrt(max(self.covariance[1, 1].item(), 0.0))
        )

    def _fade_scale(self) -> float:
        # The forward and reverse metrics of this exchange are the two
        # most recent blackboard entries (both directions share the
        # reciprocal taps, so their fades agree up to noise).
        if len(_FADE_BLACKBOARD) < 2:
            return 1.0
        scales = []
        for metric, snr_db in _FADE_BLACKBOARD[-2:]:
            nominal = 10.0 ** (snr_db / 10.0)
            estimated = _metric_to_snr(metric)
            scales.append(
                min(max(nominal / max(estimated, 1e-9), 1.0),
                    RecordingEKF.max_scale)
            )
        return float(np.mean(scales))

    def update(self, measurement) -> None:
        scale = self._fade_scale()
        prior_covariance = self.covariance.clone()
        if RecordingEKF.fade_aware:
            self.measurement_covariance = (
                self.base_measurement_covariance * scale
            )
        try:
            super().update(measurement)
        finally:
            if RecordingEKF.fade_aware:
                self.measurement_covariance = (
                    self.base_measurement_covariance.clone()
                )
        self.updates.append(
            {
                "interval": len(self.predicted_phase_std) - 1,
                "prior_covariance": prior_covariance,
                "posterior_covariance": self.covariance.clone(),
                "posterior_state": self.state.clone(),
                "fade_scale": scale,
            }
        )


def replay_update_with_true_noise(
    ekf: RecordingEKF, record: dict
) -> torch.Tensor:
    """Joseph-form posterior of a recorded update, replayed with the
    fade-scaled (true) measurement covariance but the same gain the
    fixed-R filter actually applied. Returns the true posterior
    covariance the believed one should have been."""

    phi = record["posterior_state"][0]
    zero = torch.zeros((), dtype=phi.dtype)
    one = torch.ones((), dtype=phi.dtype)
    jacobian = torch.stack(
        (
            torch.stack((-torch.sin(phi), zero)),
            torch.stack((torch.cos(phi), zero)),
            torch.stack((zero, one)),
        )
    )
    prior = record["prior_covariance"]
    r_believed = ekf.base_measurement_covariance
    innovation = jacobian @ prior @ jacobian.T + r_believed
    gain = torch.linalg.solve(innovation, jacobian @ prior).T
    identity = torch.eye(2, dtype=prior.dtype)
    residual_map = identity - gain @ jacobian
    r_true = r_believed * record["fade_scale"]
    return (
        residual_map @ prior @ residual_map.T
        + gain @ r_true @ gain.T
    )


def run_star_instrumented(
    settings: SDRSimulationConfig,
    fade_aware: bool = False,
    channel_process_var: float = 0.0,
    **kwargs,
):
    """run_scheduled_star with recording synchronizers/EKFs/links
    swapped in from the outside. All knobs off changes nothing
    (regression-tested). fade_aware scales the EKF measurement
    covariance by the metric-estimated fade of each exchange (tested,
    rejected - the metric conflates multipath with noise).
    channel_process_var adds the channel-decorrelation term to the
    EKF's phase process noise (the corrected coast rule)."""

    RecordingSynchronizer.instances = []
    RecordingRadioLink.instances = []
    RecordingEKF.instances = []
    RecordingEKF.fade_aware = fade_aware
    RecordingEKF.extra_phase_process_var = channel_process_var
    _FADE_BLACKBOARD.clear()
    originals = (
        scheduled_module.PhaseFrequencyEKF,
        scheduled_module.SDRSynchronizer,
        scheduled_module.SDRRadioLink,
    )
    scheduled_module.PhaseFrequencyEKF = RecordingEKF
    scheduled_module.SDRSynchronizer = RecordingSynchronizer
    scheduled_module.SDRRadioLink = RecordingRadioLink
    try:
        result = scheduled_module.run_scheduled_star(settings, **kwargs)
    finally:
        (
            scheduled_module.PhaseFrequencyEKF,
            scheduled_module.SDRSynchronizer,
            scheduled_module.SDRRadioLink,
        ) = originals
        RecordingEKF.fade_aware = False
        RecordingEKF.extra_phase_process_var = 0.0
    return result, {
        "synchronizers": list(RecordingSynchronizer.instances),
        "ekfs": list(RecordingEKF.instances),
        "forward_links": RecordingRadioLink.instances[0::2],
    }


# ---------------------------------------------------------------------
# Per-run analysis
# ---------------------------------------------------------------------

def channel_energy(link) -> torch.Tensor:
    """Per-interval channel energy sum_l |h_l(t)|^2 of a forward link
    (mirror shares the taps, so this is the exchange's fade state)."""

    taps = link.channel_taps
    energy = torch.sum(torch.abs(taps) ** 2, dim=-1)
    return energy.reshape(-1).to(torch.float64)


def service_records(result, tape) -> list[list[dict]]:
    """Align each link's serviced intervals with its synchronizer's
    forward/reverse event pairs."""

    per_link = []
    for k, synchronizer in enumerate(tape["synchronizers"]):
        events = synchronizer.events
        serviced_intervals = torch.nonzero(result.serviced[k]).flatten()
        if len(events) != 2 * len(serviced_intervals):
            raise RuntimeError(
                f"link {k}: {len(events)} events for "
                f"{len(serviced_intervals)} services"
            )
        records = []
        for index, interval in enumerate(serviced_intervals.tolist()):
            forward = events[2 * index]
            reverse = events[2 * index + 1]
            records.append(
                {
                    "interval": interval,
                    "detected": bool(forward.detected and reverse.detected),
                    "metric_min": min(
                        forward.detection_metric, reverse.detection_metric
                    ),
                }
            )
        per_link.append(records)
    return per_link


def run_summary(result, tape) -> dict:
    services = service_records(result, tape)
    total = sum(len(records) for records in services)
    misses = sum(
        1 for records in services for r in records if not r["detected"]
    )
    metrics = [
        r["metric_min"] for records in services for r in records
    ]
    worst = max(
        (v for v in result.station_steady_rms if v == v),
        default=float("nan"),
    )
    return {
        "gain": result.mean_array_gain,
        "worst_rms": worst,
        "mean_rms": float(
            np.nanmean([v for v in result.station_steady_rms])
        ),
        "airtime": result.airtime_used_fraction,
        "miss_rate": misses / max(total, 1),
        "metric_p10": float(np.percentile(metrics, 10)) if metrics else float("nan"),
    }


def _fold_half_branch(value: float) -> float:
    """The two-way half-difference is defined modulo pi; fold to
    (-pi/2, pi/2] so bias statistics are branch-agnostic."""

    folded = math.remainder(value, 2.0 * math.pi)
    if folded > math.pi / 2.0:
        folded -= math.pi
    elif folded <= -math.pi / 2.0:
        folded += math.pi
    return folded


def reciprocity_bias_series(result, tape, settings) -> list[list[tuple]]:
    """Per link: (interval, bias) at every successful service, where
    bias = measured two-way half-difference minus the TRUE relative
    oscillator phase at the forward capture. This is the channel's
    un-cancelled contribution to the measurement: taps are shared by
    the two directions, but each receiver samples the composite
    response with its own timing/SFO/AGC state, so a residual
    composite-phase asymmetry survives the half-difference - constant
    when the taps are static, wandering when they move."""

    tdd = settings.tdd_turnaround_s
    per_link = []
    for k, synchronizer in enumerate(tape["synchronizers"]):
        events = synchronizer.events
        truths = tape["forward_links"][k].capture_truth
        serviced_intervals = torch.nonzero(result.serviced[k]).flatten()
        series = []
        for index, interval in enumerate(serviced_intervals.tolist()):
            forward = events[2 * index]
            reverse = events[2 * index + 1]
            if not (forward.detected and reverse.detected):
                continue
            combined_frequency = (
                forward.frequency.item() - reverse.frequency.item()
            ) / 2.0
            half = math.remainder(
                math.remainder(
                    forward.phase.item() - reverse.phase.item(),
                    2.0 * math.pi,
                )
                / 2.0
                - combined_frequency * tdd / 2.0,
                2.0 * math.pi,
            )
            _, ref_phase, slave_phase = truths[index]
            truth = ref_phase - slave_phase
            series.append(
                (interval, _fold_half_branch(half - truth))
            )
        per_link.append(series)
    return per_link


def bias_structure(all_series: list[list[tuple]], max_gap: int = 6):
    """sigma_b (per-link std, pooled rms) and the structure function
    D(gap) = E[(b(t+gap) - b(t))^2] / 2 over ALL service pairs at that
    interval separation (consecutive-only pairs are selection-biased:
    in a dense schedule a gap-2 pair only exists after a missed
    detection, i.e. conditioned on a deep fade)."""

    stds = []
    gap_sums: dict[int, list[float]] = {}
    for series in all_series:
        if len(series) < 3:
            continue
        values = np.array([b for _, b in series])
        stds.append(float(np.std(values)))
        for i in range(len(series)):
            for j in range(i + 1, len(series)):
                gap = series[j][0] - series[i][0]
                if gap > max_gap:
                    break
                gap_sums.setdefault(gap, []).append(
                    0.5 * (series[j][1] - series[i][1]) ** 2
                )
    sigma_b = float(np.sqrt(np.mean(np.square(stds)))) if stds else float("nan")
    structure = {
        gap: float(np.mean(values)) for gap, values in sorted(gap_sums.items())
    }
    return sigma_b, structure


def window_bias_correlation(result, tape, settings) -> tuple[float, int]:
    """The smoking gun: if servicing steers the oscillator to cancel
    the measurement bias b(t_service), then the mean SIGNED residual
    of the following coast window should equal -(b_service - mean b).
    Returns (Pearson r, number of windows) pooled over links."""

    series = reciprocity_bias_series(result, tape, settings)
    latency = settings.correction_latency_intervals
    xs, ys = [], []
    for k, link_series in enumerate(series):
        if len(link_series) < 3:
            continue
        mean_bias = float(np.mean([b for _, b in link_series]))
        residual_row = result.residuals[k]
        for index, (interval, bias) in enumerate(link_series):
            start = interval + latency
            end = (
                link_series[index + 1][0] + latency
                if index + 1 < len(link_series)
                else residual_row.shape[0]
            )
            end = min(end, residual_row.shape[0])
            if end <= start or interval < 10:
                continue
            window_mean = torch.mean(residual_row[start:end]).item()
            xs.append(-(bias - mean_bias))
            ys.append(window_mean)
    if len(xs) < 8:
        return float("nan"), len(xs)
    return float(np.corrcoef(xs, ys)[0, 1]), len(xs)


def jakes_structure(sigma_b: float, f_doppler: float, gap_s: float) -> float:
    """Model D(gap) = sigma_b^2 * (1 - J0(2 pi f_D gap)): the wander a
    Jakes-correlated composite-phase bias accumulates over the gap."""

    j0 = float(
        torch.special.bessel_j0(
            torch.tensor(2.0 * math.pi * f_doppler * gap_s)
        )
    )
    return sigma_b**2 * (1.0 - j0)


def missing_term_validation(
    result, tape, settings, f_doppler: float, sigma_b: float
) -> dict:
    """Per coast window following a successful update: the measured
    mean-square residual, the BELIEVED prediction from the fixed-R
    posterior (drift + tracking + latency terms - the existing coast
    rule), and the channel term 2 sigma_b^2 (1 - J0(2 pi f_D h))
    averaged over the window's horizons h. If the reciprocity-bias
    wander is the missing mechanism, believed + channel tracks the
    measurement at every speed with NO fitted constants beyond the
    independently measured sigma_b."""

    interval = settings.sync_interval
    interval_samples = settings.sync_interval * settings.sample_rate
    drift_var_per_interval = (
        settings.phase_noise_std_rad**2 * interval_samples
    )
    latency = settings.correction_latency_intervals
    rows = []
    for k, ekf in enumerate(tape["ekfs"]):
        residual_row = result.residuals[k]
        updates = [u for u in ekf.updates]
        for index, record in enumerate(updates):
            start = record["interval"]
            end = (
                updates[index + 1]["interval"]
                if index + 1 < len(updates)
                else result.residuals.shape[1]
            )
            # The correction from this update loads at start+latency;
            # the window it governs runs from there to the next
            # update's loading point.
            window = residual_row[
                start + latency: end + latency
                if end + latency <= residual_row.shape[0]
                else residual_row.shape[0]
            ]
            if window.numel() == 0 or start < 10:  # skip acquisition
                continue
            believed = record["posterior_covariance"]
            measured_ms = torch.mean(window.square()).item()
            horizons = interval * (
                latency + 1 + torch.arange(window.numel(), dtype=torch.float64)
            )
            believed_ms = torch.mean(
                believed[0, 0]
                + horizons.square() * believed[1, 1]
                + 2.0 * horizons * believed[0, 1]
                + drift_var_per_interval * (horizons / interval)
            ).item()
            channel_ms = float(
                np.mean(
                    [
                        2.0 * jakes_structure(sigma_b, f_doppler, h)
                        for h in horizons.tolist()
                    ]
                )
            )
            rows.append(
                {
                    "station": k,
                    "measured_ms": measured_ms,
                    "believed_ms": believed_ms,
                    "channel_ms": channel_ms,
                }
            )
    return {"windows": rows}


def print_validation_row(label: str, rows: list[dict]) -> None:
    if not rows:
        print(f"  {label:<10} (no coast windows)")
        return
    measured = math.sqrt(np.mean([r["measured_ms"] for r in rows]))
    believed = math.sqrt(np.mean([r["believed_ms"] for r in rows]))
    combined = math.sqrt(
        np.mean([r["believed_ms"] + r["channel_ms"] for r in rows])
    )
    print(
        f"  {label:<10} {len(rows):>8} "
        f"{1e3 * measured:>9.0f} "
        f"{1e3 * believed:>10.0f} ({measured / believed:4.2f}x) "
        f"{1e3 * combined:>10.0f} ({measured / combined:4.2f}x)"
    )


# ---------------------------------------------------------------------
# The study
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Doppler coasting: mechanism, missing term, fix"
    )
    parser.add_argument("--stations", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--speeds", type=str, default="0,1,3,5")
    parser.add_argument("--flat-budget", type=float, default=0.314)
    args = parser.parse_args()

    n = args.stations
    seeds = [int(s) for s in args.seeds.split(",")]
    speeds = [float(v) for v in args.speeds.split(",")]
    budgets = [args.flat_budget] * (n - 1)

    print(
        f"Doppler coasting study, N={n} star, {args.iterations} "
        f"intervals, budget {1e3 * args.flat_budget:.0f} mrad, "
        f"speeds {speeds} m/s, seeds {seeds}"
    )
    carrier = SDRSimulationConfig().carrier_frequency_hz
    for speed in speeds:
        fd = speed * carrier / 299792458.0
        print(
            f"  {speed:g} m/s -> f_D {fd:5.1f} Hz, "
            f"J0(2 pi f_D T) = "
            f"{float(torch.special.bessel_j0(torch.tensor(2 * math.pi * fd * 0.05))):+.2f}"
        )

    interval = SDRSimulationConfig().sync_interval

    print(
        "\n=== grid: policy x speed (mean over seeds) "
        "===\n  rows: gain% / worst-rms mrad / airtime% / miss%"
    )
    grid_runs: dict[tuple, tuple] = {}
    print(f"  {'policy':<12}" + "".join(f"{v:>22g} m/s" for v in speeds))
    for policy in ("uniform", "scheduled"):
        cells = []
        for speed in speeds:
            stats = []
            for seed in seeds:
                settings = SDRSimulationConfig(
                    num_iterations=args.iterations,
                    seed=seed,
                    device="cpu",
                    channel_speed_mps=speed,
                )
                result, tape = run_star_instrumented(
                    settings,
                    num_stations=n,
                    policy=policy,
                    budgets_rad=budgets,
                )
                grid_runs[(policy, speed, seed)] = (result, tape, settings)
                stats.append(run_summary(result, tape))
            mean = {
                key: float(np.mean([s[key] for s in stats]))
                for key in stats[0]
            }
            cells.append(
                f"{100 * mean['gain']:5.1f}/{1e3 * mean['worst_rms']:4.0f}"
                f"/{100 * mean['airtime']:3.0f}/{100 * mean['miss_rate']:3.0f}"
            )
        print(f"  {policy:<12}" + "".join(f"{c:>26}" for c in cells))

    # ---- the reciprocity bias process, measured from uniform runs --
    print(
        "\n=== reciprocity bias b(t): measured wander vs the Jakes "
        "model sigma_b^2 (1 - J0(2 pi f_D dt)) ===\n"
        "  (b = measured two-way half-difference minus TRUE relative "
        "oscillator phase;\n   constant bias is harmless - its WANDER "
        "is what a coasting station eats)"
    )
    sigma_b_by_speed: dict[float, float] = {}
    structure_by_speed: dict[float, dict] = {}
    for speed in speeds:
        all_series = []
        for seed in seeds:
            result, tape, settings = grid_runs[("uniform", speed, seed)]
            all_series.extend(
                reciprocity_bias_series(result, tape, settings)
            )
        sigma_b, structure = bias_structure(all_series)
        sigma_b_by_speed[speed] = sigma_b
        structure_by_speed[speed] = structure
    # White nugget: per-capture re-sampling of the composite response
    # (timing jitter, AGC, estimator noise) moves b even over a frozen
    # channel; the gap-1 wander at 0 m/s measures it.
    nugget_var = (
        structure_by_speed[0.0].get(1, 0.0) if 0.0 in structure_by_speed
        else 0.0
    )
    print(f"  white nugget sigma_n = {1e3 * math.sqrt(nugget_var):.0f} mrad "
          "(gap-1 wander at 0 m/s)")

    def channel_var(speed: float) -> float:
        return max(sigma_b_by_speed[speed] ** 2 - nugget_var, 0.0)

    print(
        f"  {'speed':<8} {'sigma_b':>9} {'sigma_c':>9}  "
        + "".join(f"{'D(' + str(g) + 'T)':>10}" for g in (1, 2, 3, 4))
        + "   [measured/model rms mrad; model = nugget + "
        "sigma_c^2(1-J0)]"
    )
    for speed in speeds:
        f_doppler = speed * carrier / 299792458.0
        sigma_c2 = channel_var(speed)
        cells = []
        for gap in (1, 2, 3, 4):
            measured = structure_by_speed[speed].get(gap, float("nan"))
            model = nugget_var + jakes_structure(
                math.sqrt(sigma_c2), f_doppler, gap * interval
            )
            cells.append(
                f"{1e3 * math.sqrt(max(measured, 0.0)):4.0f}/"
                f"{1e3 * math.sqrt(model):4.0f}"
            )
        print(
            f"  {speed:<8g} {1e3 * sigma_b_by_speed[speed]:>7.0f}mr "
            f"{1e3 * math.sqrt(sigma_c2):>7.0f}mr  "
            + "".join(f"{c:>10}" for c in cells)
        )

    # ---- smoking gun: coast windows hold the stale service bias ----
    print(
        "\n=== smoking gun: corr(mean window residual, -(b_service - "
        "mean b)) ===\n  (if servicing steers the oscillator onto the "
        "measurement bias, coast windows should carry it verbatim)"
    )
    print(f"  {'speed':<8} {'scheduled':>12} {'uniform':>12}")
    for speed in speeds:
        row = []
        for policy in ("scheduled", "uniform"):
            correlations, counts = [], 0
            for seed in seeds:
                result, tape, settings = grid_runs[(policy, speed, seed)]
                corr, count = window_bias_correlation(
                    result, tape, settings
                )
                if corr == corr:
                    correlations.append(corr)
                counts += count
            row.append(
                f"{np.mean(correlations):+5.2f} (n={counts})"
                if correlations else "   n/a"
            )
        print(f"  {speed:<8g} {row[0]:>12} {row[1]:>12}")

    # ---- does believed + channel term close the prediction gap? ----
    print(
        "\n=== coast-window validation (scheduled runs): measured vs "
        "believed vs believed + channel term ===\n"
        "  windows grouped per speed; rms mrad, ratio measured/pred "
        "in parens (1.00x = the rule prices the coast correctly)"
    )
    print(
        f"  {'speed':<10} {'windows':>8} {'measured':>9} "
        f"{'believed':>17} {'bel+channel':>17}"
    )
    for speed in speeds:
        f_doppler = speed * carrier / 299792458.0
        rows = []
        for seed in seeds:
            result, tape, settings = grid_runs[("scheduled", speed, seed)]
            rows.extend(
                missing_term_validation(
                    result, tape, settings, f_doppler,
                    math.sqrt(channel_var(speed)),
                )["windows"]
            )
        print_validation_row(f"{speed:g} m/s", rows)

    # ---- corrected coast rule: channel term in the process noise ---
    print(
        "\n=== corrected rule: add q_chan = 2 sigma_b^2 "
        "(1 - J0(2 pi f_D T)) to the EKF phase process noise ===\n"
        "  (sigma_b identified from the same environment's dense-"
        "service runs; scheduler machinery otherwise untouched)"
    )
    print(
        f"  {'speed':<8} {'fixed worst-rms':>16} {'corrected':>11} "
        f"{'fixed air':>10} {'corr air':>9}  {'fixed gain':>11} "
        f"{'corr gain':>10}"
    )
    for speed in speeds:
        f_doppler = speed * carrier / 299792458.0
        q_chan = 2.0 * jakes_structure(
            math.sqrt(channel_var(speed)), f_doppler, interval
        )
        fixed_stats, aware_stats = [], []
        for seed in seeds:
            fixed_stats.append(
                run_summary(*grid_runs[("scheduled", speed, seed)][:2])
            )
            settings = SDRSimulationConfig(
                num_iterations=args.iterations,
                seed=seed,
                device="cpu",
                channel_speed_mps=speed,
            )
            result, tape = run_star_instrumented(
                settings,
                channel_process_var=q_chan,
                num_stations=n,
                policy="scheduled",
                budgets_rad=budgets,
            )
            aware_stats.append(run_summary(result, tape))
        print(
            f"  {speed:<8g} "
            f"{1e3 * np.mean([r['worst_rms'] for r in fixed_stats]):>14.0f}mr "
            f"{1e3 * np.mean([r['worst_rms'] for r in aware_stats]):>9.0f}mr "
            f"{100 * np.mean([r['airtime'] for r in fixed_stats]):>9.1f}% "
            f"{100 * np.mean([r['airtime'] for r in aware_stats]):>8.1f}% "
            f"{100 * np.mean([r['gain'] for r in fixed_stats]):>10.2f}% "
            f"{100 * np.mean([r['gain'] for r in aware_stats]):>9.2f}%"
        )


if __name__ == "__main__":
    main()
