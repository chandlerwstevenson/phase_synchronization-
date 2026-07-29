"""Rashid & Nanzer's DFPC / KF-DFPC algorithms, and an OTA adaptation.

Two layers:

1. A faithful statistics-level implementation of the algorithms in
   Rashid & Nanzer, IEEE TWC 22(4):2789-2802, 2023 (DOI
   10.1109/TWC.2022.3213788): N nodes on a random connectivity graph,
   Metropolis-Hastings consensus mixing, per-iteration error injection from
   their oscillator/estimation statistics (their Algorithms 1 and 2), plus
   their Eq. 27 closed-form residual bound. Errors are injected as Gaussian
   draws exactly as in the paper -- no waveform is generated.

2. An over-the-air adaptation at N = 2 that runs the consensus update rule
   through this repository's sampled-IQ physical layer: each node measures
   the other's frame, then retunes its own NCO by half its measured offset
   (the two-node Metropolis-Hastings weight), subject to the same correction
   latency and quantization as our EKF loops. With a reciprocal channel and
   simultaneous symmetric updates, the channel phase enters both directions
   identically and cancels in the relative coordinate. The KF variant gives
   each node a local EKF, mirroring the paper's per-node filter; the
   exchange of applied corrections (the paper's error-free side channel) is
   assumed ideal.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from sionna.phy import config as sionna_config

from .coherent import _pick_half_phase
from .core import (
    REAL_DTYPE,
    Oscillator,
    PhaseFrequencyEKF,
    resolve_device,
    wrap_phase,
)
from .sdr import (
    SDRRadioLink,
    SDRSimulationConfig,
    SDRSynchronizer,
    _FlickerFrequencyNoise,
    _measurement_covariance,
    _quantize_correction,
    make_sync_preamble,
)


# ---------------------------------------------------------------------------
# Layer 1: statistics-level DFPC / KF-DFPC, faithful to the paper.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsensusStatsConfig:
    """Parameters of the paper's statistical model (their Section V)."""

    num_nodes: int = 20
    connectivity: float = 0.2
    num_iterations: int = 200
    carrier_frequency_hz: float = 1e9
    sample_rate: float = 10e6
    update_interval: float = 1e-4
    snr_db: float = 0.0
    adev_beta1: float = 5e-19
    adev_beta2: float = 5e-19
    integrated_phase_noise_dbc: float = -53.46
    initial_frequency_std_hz: float = 1e-4 * 1e9
    algorithm: str = "kf-dfpc"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.num_nodes < 2:
            raise ValueError("need at least two nodes")
        if not 0.0 < self.connectivity <= 1.0:
            raise ValueError("connectivity must be in (0, 1]")
        if self.algorithm not in {"dfpc", "kf-dfpc"}:
            raise ValueError("algorithm must be 'dfpc' or 'kf-dfpc'")


@dataclass(frozen=True)
class ConsensusStatsResult:
    total_phase_error_std: torch.Tensor
    frequency_spread_hz: torch.Tensor
    eq27_bound_rad: float

    @property
    def final_phase_error_std(self) -> float:
        return self.total_phase_error_std[-1].item()


def _error_statistics(settings: ConsensusStatsConfig) -> dict[str, float]:
    """Their Eqs. (2), (5), (13), (16), (17) evaluated at the settings."""

    t = settings.update_interval
    snr = 10.0 ** (settings.snr_db / 10.0)
    samples = int(round(t * settings.sample_rate))
    sigma_f = settings.carrier_frequency_hz * math.sqrt(
        settings.adev_beta1 / t + settings.adev_beta2 * t
    )
    sigma_jitter = math.sqrt(
        2.0 * 10.0 ** (settings.integrated_phase_noise_dbc / 10.0)
    )
    sigma_f_meas = (
        math.sqrt(6.0 / ((2.0 * math.pi) ** 2 * samples**3 * snr))
        * settings.sample_rate
    )
    sigma_theta_meas = math.sqrt(2.0 / (samples * snr))
    return {
        "sigma_f": sigma_f,
        "sigma_theta_drift": math.pi * t * sigma_f,
        "sigma_jitter": sigma_jitter,
        "sigma_f_meas": sigma_f_meas,
        "sigma_theta_meas": sigma_theta_meas,
    }


def dfpc_total_phase_error_bound(settings: ConsensusStatsConfig) -> float:
    """Their Eq. 27: sparse-connectivity residual total phase error (rad)."""

    stats = _error_statistics(settings)
    t = settings.update_interval
    return math.sqrt(
        (2.0 * math.pi * stats["sigma_f"] * t) ** 2
        + (2.0 * math.pi * stats["sigma_f_meas"] * t) ** 2
        + stats["sigma_theta_drift"] ** 2
        + stats["sigma_theta_meas"] ** 2
        + stats["sigma_jitter"] ** 2
    )


def _random_connected_graph(
    num_nodes: int, connectivity: float, generator: torch.Generator
) -> torch.Tensor:
    """Random undirected graph with the requested edge fraction, connected."""

    pairs = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]
    num_edges = max(num_nodes - 1, int(round(connectivity * len(pairs))))
    for _ in range(200):
        order = torch.randperm(len(pairs), generator=generator)
        adjacency = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
        for index in order[:num_edges]:
            i, j = pairs[index]
            adjacency[i, j] = adjacency[j, i] = True
        reached = torch.zeros(num_nodes, dtype=torch.bool)
        frontier = [0]
        reached[0] = True
        while frontier:
            node = frontier.pop()
            for neighbor in torch.nonzero(adjacency[node]).flatten().tolist():
                if not reached[neighbor]:
                    reached[neighbor] = True
                    frontier.append(neighbor)
        if bool(reached.all()):
            return adjacency
    raise RuntimeError("could not sample a connected graph")


def _metropolis_weights(adjacency: torch.Tensor) -> torch.Tensor:
    """Their Eq. 9: Metropolis-Hastings doubly stochastic mixing matrix."""

    degrees = adjacency.sum(dim=1)
    n = adjacency.shape[0]
    weights = torch.zeros(n, n, dtype=REAL_DTYPE)
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j]:
                weights[i, j] = weights[j, i] = 1.0 / (
                    1.0 + max(int(degrees[i]), int(degrees[j]))
                )
    weights += torch.diag(1.0 - weights.sum(dim=1))
    return weights


def run_consensus_stats(
    settings: ConsensusStatsConfig = ConsensusStatsConfig(),
) -> ConsensusStatsResult:
    """Run their Algorithm 1 (DFPC) or Algorithm 2 (KF-DFPC), verbatim model."""

    generator = torch.Generator()
    generator.manual_seed(settings.seed)
    stats = _error_statistics(settings)
    t = settings.update_interval
    n = settings.num_nodes
    mixing = _metropolis_weights(
        _random_connected_graph(n, settings.connectivity, generator)
    )

    frequencies = (
        torch.randn(n, dtype=REAL_DTYPE, generator=generator)
        * settings.initial_frequency_std_hz
    )
    phases = (
        torch.rand(n, dtype=REAL_DTYPE, generator=generator) * 2.0 * math.pi
    )

    use_kf = settings.algorithm == "kf-dfpc"
    if use_kf:
        # Their Eq. 29 process covariance and Eq. 31 measurement covariance.
        q = torch.tensor(
            [
                [stats["sigma_f"] ** 2, -math.pi * t * stats["sigma_f"] ** 2],
                [
                    -math.pi * t * stats["sigma_f"] ** 2,
                    (math.pi * t * stats["sigma_f"]) ** 2
                    + stats["sigma_jitter"] ** 2,
                ],
            ],
            dtype=REAL_DTYPE,
        )
        r = torch.diag(
            torch.tensor(
                [stats["sigma_f_meas"] ** 2, stats["sigma_theta_meas"] ** 2],
                dtype=REAL_DTYPE,
            )
        )
        covariances = (
            torch.diag(
                torch.tensor(
                    [
                        settings.initial_frequency_std_hz**2,
                        4.0 * math.pi**2 / 12.0,
                    ],
                    dtype=REAL_DTYPE,
                )
            )
            .unsqueeze(0)
            .repeat(n, 1, 1)
        )

    history_phase, history_freq = [], []
    for _ in range(settings.num_iterations):
        # Truth evolution: their Algorithm 1 steps 1-2.
        frequencies = frequencies + torch.randn(
            n, dtype=REAL_DTYPE, generator=generator
        ) * stats["sigma_f"]
        phases = (
            phases
            + torch.randn(n, dtype=REAL_DTYPE, generator=generator)
            * stats["sigma_theta_drift"]
            + torch.randn(n, dtype=REAL_DTYPE, generator=generator)
            * stats["sigma_jitter"]
        )
        # Estimation errors: step 3.
        observed_f = frequencies + torch.randn(
            n, dtype=REAL_DTYPE, generator=generator
        ) * stats["sigma_f_meas"]
        observed_theta = phases + torch.randn(
            n, dtype=REAL_DTYPE, generator=generator
        ) * stats["sigma_theta_meas"]

        if use_kf:
            posterior_f = torch.empty(n, dtype=REAL_DTYPE)
            posterior_theta = torch.empty(n, dtype=REAL_DTYPE)
            for node in range(n):
                prior_mean = torch.stack((frequencies[node], phases[node]))
                prior_cov = covariances[node] + q
                observation = torch.stack(
                    (observed_f[node], observed_theta[node])
                )
                gain = torch.linalg.solve(prior_cov + r, prior_cov).T
                posterior = prior_mean + gain @ (observation - prior_mean)
                covariances[node] = (
                    torch.eye(2, dtype=REAL_DTYPE) - gain
                ) @ prior_cov
                posterior_f[node] = posterior[0]
                posterior_theta[node] = posterior[1]
            frequencies = mixing @ posterior_f
            phases = mixing @ posterior_theta
            # Their Eq. 39: mix the covariance entries with squared weights.
            mixed = torch.empty_like(covariances)
            squared = mixing.square()
            for a in range(2):
                for b in range(2):
                    mixed[:, a, b] = squared @ covariances[:, a, b]
            covariances = mixed
        else:
            frequencies = mixing @ observed_f
            phases = mixing @ observed_theta

        # Their Eq. 10 metric: spread of total phase errors around the mean.
        total_phase = 2.0 * math.pi * (frequencies - frequencies.mean()) * t + (
            phases - phases.mean()
        )
        history_phase.append(torch.std(total_phase))
        history_freq.append(torch.std(frequencies))

    return ConsensusStatsResult(
        total_phase_error_std=torch.stack(history_phase),
        frequency_spread_hz=torch.stack(history_freq),
        eq27_bound_rad=dfpc_total_phase_error_bound(settings),
    )


# ---------------------------------------------------------------------------
# Layer 2: their consensus update rule over our physical layer (N = 2).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsensusOTAResult:
    """Metrics from a two-node consensus run over the sampled-IQ link."""

    true_phase: torch.Tensor
    post_correction_phase: torch.Tensor
    post_correction_frequency: torch.Tensor
    coherent_gain: torch.Tensor
    detected: torch.Tensor
    correction_active: torch.Tensor
    device: torch.device

    @property
    def detection_rate(self) -> float:
        return torch.mean(self.detected.to(torch.float64)).item()

    @property
    def steady_state_phase_rms(self) -> float:
        valid = self.detected & self.correction_active
        if not torch.any(valid):
            return float("nan")
        return torch.sqrt(
            torch.mean(self.post_correction_phase[valid].square())
        ).item()

    @property
    def mean_coherent_gain(self) -> float:
        valid = self.detected & self.correction_active
        if not torch.any(valid):
            return float("nan")
        return torch.mean(self.coherent_gain[valid]).item()

    @property
    def final_phase_error(self) -> float:
        return self.post_correction_phase[-1].item()

    @property
    def final_frequency_error_hz(self) -> float:
        return self.post_correction_frequency[-1].item() / (2.0 * math.pi)


def run_consensus_ota_simulation(
    settings: SDRSimulationConfig = SDRSimulationConfig(),
    algorithm: str = "dfpc",
    reciprocal: bool = True,
) -> ConsensusOTAResult:
    """Run DFPC or KF-DFPC between two nodes over the sampled-IQ link.

    Each interval both directions are exchanged over the reciprocal channel
    and each node retunes its own NCO by half its offset estimate toward the
    other (the two-node Metropolis-Hastings weight). 'kf-dfpc' gives each
    node a local EKF, mirroring the paper's per-node filter; applied
    corrections travel over the paper's assumed ideal side channel.

    With ``reciprocal=False`` each node consenses on its raw directional
    measurement, which is the paper's stated assumption (channel phase
    handled elsewhere). Over a real channel that measurement contains the
    channel phase, and because the updates are symmetric and measurements
    wrap, the pair becomes bistable: it converges to relative phase 0 or to
    the anti-phase fixed point pi, depending on whether channel phase plus
    offset crosses the +/- pi wrap during convergence -- a physical-layer
    failure mode invisible to the paper's channel-free model. With
    ``reciprocal=True`` (the steelman) nodes exchange measurements over the
    side channel and consense on the channel-free half-difference,
    inheriting the two-way pi ambiguity, resolved as in
    ``run_two_way_simulation``.
    """

    if algorithm not in {"dfpc", "kf-dfpc"}:
        raise ValueError("algorithm must be 'dfpc' or 'kf-dfpc'")
    use_kf = algorithm == "kf-dfpc"

    device = resolve_device(settings.device)
    torch.manual_seed(settings.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(settings.seed)
    sionna_config.seed = settings.seed
    generator = torch.Generator(device=device)
    generator.manual_seed(settings.seed + 1)

    frequency_process_std = 2.0 * math.pi * settings.frequency_process_std_hz
    oscillator_covariance = torch.diag(
        torch.tensor(
            [settings.phase_process_std_rad**2, frequency_process_std**2],
            dtype=REAL_DTYPE,
            device=device,
        )
    )
    node_a = Oscillator(
        settings.master_initial_phase,
        2.0 * math.pi * settings.master_initial_frequency_hz,
        settings.sync_interval,
        oscillator_covariance,
        device,
        generator,
    )
    node_b = Oscillator(
        settings.slave_initial_phase,
        2.0 * math.pi * settings.slave_initial_frequency_hz,
        settings.sync_interval,
        oscillator_covariance,
        device,
        generator,
    )
    preamble = make_sync_preamble(settings, device)
    link_ab = SDRRadioLink(settings, preamble, device, generator)
    link_ba = SDRRadioLink(settings, preamble, device, generator, mirror_of=link_ab)
    synchronizer = SDRSynchronizer(settings, preamble)

    interval_samples = int(round(settings.sync_interval * settings.sample_rate))
    white_fm_phase_variance = settings.phase_noise_std_rad**2 * interval_samples
    flicker = _FlickerFrequencyNoise(
        settings.flicker_frequency_std_hz,
        settings.sync_interval,
        settings.num_iterations * settings.sync_interval,
        device,
        generator,
    )
    if use_kf:
        measurement_noise = _measurement_covariance(settings, preamble, device)
        if reciprocal:
            # The half-difference averages two independent measurements.
            measurement_noise = 0.5 * measurement_noise
        process = 2.0 * oscillator_covariance + torch.diag(
            torch.tensor(
                [white_fm_phase_variance, flicker.innovation_variance],
                dtype=REAL_DTYPE,
                device=device,
            )
        )
        initial = torch.diag(
            torch.tensor(
                [math.pi**2, (2.0 * math.pi * 50e3) ** 2],
                dtype=REAL_DTYPE,
                device=device,
            )
        )
        filters = {
            "a": PhaseFrequencyEKF(
                settings.sync_interval, process, measurement_noise, device, initial
            ),
            "b": PhaseFrequencyEKF(
                settings.sync_interval, process, measurement_noise, device, initial
            ),
        }
        acquired = {"a": False, "b": False}

    capture_samples = link_ab.input_length + link_ab.l_tot - 1
    remainder_samples = max(0, interval_samples - 2 * capture_samples)
    pending: dict[int, dict[str, torch.Tensor]] = {}
    carried_lo_walk = torch.zeros((), dtype=REAL_DTYPE, device=device)
    flicker_previous = torch.zeros((), dtype=REAL_DTYPE, device=device)
    corrections_total = {
        "a": torch.zeros((), dtype=REAL_DTYPE, device=device),
        "b": torch.zeros((), dtype=REAL_DTYPE, device=device),
    }
    # Ambiguity-resolution references for the filterless reciprocal variant.
    last_offsets = {
        "a": torch.zeros((), dtype=REAL_DTYPE, device=device),
        "b": torch.zeros((), dtype=REAL_DTYPE, device=device),
    }
    correction_has_loaded = False

    history: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "true_phase",
            "post_correction_phase",
            "post_correction_frequency",
            "coherent_gain",
            "detected",
            "correction_active",
        )
    }

    for iteration in range(settings.num_iterations):
        node_a.step()
        node_b.step()
        node_a.state[0] = wrap_phase(node_a.state[0] + carried_lo_walk)
        flicker_now = flicker.step()
        node_a.state[1] = node_a.state[1] + (flicker_now - flicker_previous)
        flicker_previous = flicker_now

        due = pending.pop(iteration, None)
        if due is not None:
            node_a.apply_correction(due["a"])
            node_b.apply_correction(due["b"])
            corrections_total["a"] = corrections_total["a"] + due["a"][1]
            corrections_total["b"] = corrections_total["b"] + due["b"][1]
            correction_has_loaded = True

        physical_a = node_a.state[1] - corrections_total["a"]
        physical_b = node_b.state[1] - corrections_total["b"]
        if settings.sample_clock_offset_ppm is not None:
            sfo_forward = settings.sample_clock_offset_ppm
        else:
            sfo_forward = float(
                (physical_b - physical_a).item()
                / (2.0 * math.pi * settings.carrier_frequency_hz)
                * 1e6
            )

        relative_state = node_a.state - node_b.state
        capture_ab = link_ab.capture(node_a, node_b, iteration, sfo_forward)
        node_a.state[0] = wrap_phase(node_a.state[0] + capture_ab.lo_walk_end)
        capture_ba = link_ba.capture(node_b, node_a, iteration, -sfo_forward)
        node_b.state[0] = wrap_phase(node_b.state[0] + capture_ba.lo_walk_end)
        if settings.phase_noise_std_rad > 0.0 and remainder_samples > 0:
            carried_lo_walk = torch.randn(
                (), dtype=REAL_DTYPE, device=device, generator=generator
            ) * (settings.phase_noise_std_rad * math.sqrt(remainder_samples))
        else:
            carried_lo_walk = torch.zeros((), dtype=REAL_DTYPE, device=device)

        measured_b = synchronizer.estimate(capture_ab.samples)  # at node B
        measured_a = synchronizer.estimate(capture_ba.samples)  # at node A
        detected = measured_a.detected and measured_b.detected

        if use_kf:
            filters["a"].predict()
            filters["b"].predict()
            if due is not None:
                # The paper's side channel: each node learns the other's
                # applied correction and shifts its filter accordingly.
                filters["a"].reset_after_correction(due["a"] - due["b"])
                filters["b"].reset_after_correction(due["b"] - due["a"])

        corrections: dict[str, torch.Tensor] | None = None
        if detected:
            # Each node's estimate of (other - self), either raw (naive) or
            # channel-free via the exchanged half-difference (reciprocal).
            if reciprocal:
                half = wrap_phase(
                    wrap_phase(measured_a.phase - measured_b.phase) / 2.0
                )
                node_observations = {
                    "a": (half, (measured_a.frequency - measured_b.frequency) / 2.0),
                    "b": (
                        wrap_phase(-half),
                        (measured_b.frequency - measured_a.frequency) / 2.0,
                    ),
                }
            else:
                node_observations = {
                    "a": (measured_a.phase, measured_a.frequency),
                    "b": (measured_b.phase, measured_b.frequency),
                }

            offsets = {}
            for name in ("a", "b"):
                phase_obs, frequency_obs = node_observations[name]
                if use_kf:
                    ekf = filters[name]
                    if reciprocal:
                        reference = (
                            wrap_phase(ekf.state[0])
                            if acquired[name]
                            else torch.zeros_like(phase_obs)
                        )
                        phase_obs = _pick_half_phase(phase_obs, reference)
                    if not acquired[name]:
                        ekf.state = torch.stack((phase_obs, frequency_obs))
                        ekf.covariance = torch.diag(
                            torch.stack(
                                (
                                    measurement_noise[0, 0],
                                    measurement_noise[2, 2],
                                )
                            )
                        )
                        acquired[name] = True
                    else:
                        ekf.update(
                            torch.stack(
                                (
                                    torch.cos(phase_obs),
                                    torch.sin(phase_obs),
                                    frequency_obs,
                                )
                            )
                        )
                    predicted = ekf.state.clone()
                    for _ in range(settings.correction_latency_intervals):
                        predicted = ekf.transition @ predicted
                    offsets[name] = predicted
                else:
                    if reciprocal:
                        phase_obs = _pick_half_phase(
                            phase_obs, last_offsets[name]
                        )
                        last_offsets[name] = phase_obs.clone()
                    offsets[name] = torch.stack((phase_obs, frequency_obs))
            corrections = {
                name: _quantize_correction(offsets[name] / 2.0, settings)
                for name in ("a", "b")
            }

        if corrections is not None:
            if settings.correction_latency_intervals == 0:
                node_a.apply_correction(corrections["a"])
                node_b.apply_correction(corrections["b"])
                corrections_total["a"] = corrections_total["a"] + corrections["a"][1]
                corrections_total["b"] = corrections_total["b"] + corrections["b"][1]
                correction_has_loaded = True
                if use_kf:
                    filters["a"].reset_after_correction(
                        corrections["a"] - corrections["b"]
                    )
                    filters["b"].reset_after_correction(
                        corrections["b"] - corrections["a"]
                    )
            else:
                pending[
                    iteration + settings.correction_latency_intervals
                ] = corrections

        history["true_phase"].append(wrap_phase(relative_state[0]).clone())
        history["detected"].append(
            torch.tensor(detected, dtype=torch.bool, device=device)
        )
        history["correction_active"].append(
            torch.tensor(correction_has_loaded, dtype=torch.bool, device=device)
        )
        residual_state = node_a.state - node_b.state
        residual_phase = wrap_phase(residual_state[0])
        history["post_correction_phase"].append(residual_phase.clone())
        history["post_correction_frequency"].append(residual_state[1].clone())
        history["coherent_gain"].append(torch.cos(residual_phase / 2.0).square())

    return ConsensusOTAResult(
        **{
            name: torch.stack(values).detach().cpu() for name, values in history.items()
        },
        device=device,
    )
