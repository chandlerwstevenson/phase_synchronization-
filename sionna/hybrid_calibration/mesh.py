"""Decentralized hybrid calibration on an N-node mesh (chain topology).

This is the N > 2 test of decentralized hybrid: N oscillators shared
across all links (so corrections on one edge genuinely disturb the
neighboring edges), per-edge hybrid estimation (one-way micro-pilots +
sparse two-way anchors, the 3-state EKF from ``hybrid.py``), and
symmetric consensus control - each node retunes by a degree-weighted
share of every incident edge's correction, so NO node is a master and
the array converges to a floating average clock.

Topology: a chain (spanning tree) over randomly placed stations,
greedily ordered nearest-neighbor from station 0's position. A tree has
no cycles, which rules out the winding (vortex) equilibria that plague
decentralized phase sync on loopy graphs; pi-ambiguity flips propagate
to the whole downstream subtree so every other edge's relative phase is
untouched.

Control law per loaded correction c_e on edge e = (p, q):

    node p applies -c_e / (2 * deg_p),  node q applies +c_e / (2 * deg_q)

and every node broadcasts what it applied (the same side channel the
consensus literature assumes), so each edge's filter is reset by the
NET relative correction its endpoints actually received - including
disturbances from adjacent edges. Interior edges are therefore
under-relaxed (a fraction of the correction lands per step), which
trades convergence speed for stability where edges share nodes.

Honest simplifications versus the 2-node loops: each edge's
intra-capture LO walk is drawn independently (edge-correlated walk via
shared nodes is not modeled); dead-time walk is applied per node at
1/sqrt(2) of the pair-relative rate; airtime is charged per edge with
no broadcast amortization (conservative for the one-way tier).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import torch
from sionna.phy import config as sionna_config

from ota_sync.coherent import _pick_half_phase
from ota_sync.core import REAL_DTYPE, Oscillator, resolve_device, wrap_phase
from ota_sync.microsync import _estimate_micro_phase, _make_micro_preamble
from ota_sync.network import MAX_LINK_SNR_DB, place_stations
from ota_sync.sdr import (
    SDRRadioLink,
    SDRSimulationConfig,
    SDRSynchronizer,
    _FlickerFrequencyNoise,
    _measurement_covariance,
    _quantize_correction,
    make_sync_preamble,
)

from .hybrid import (
    _JointPhaseChannelEKF,
    _jacobian_anchor,
    _jacobian_sum_only,
    _jacobian_sum_with_frequency,
    _observe_anchor,
    _observe_sum_only,
    _observe_sum_with_frequency,
)


def _chain_order(positions: np.ndarray) -> list[int]:
    """Greedy nearest-neighbor path starting from station 0."""

    remaining = list(range(1, positions.shape[0]))
    order = [0]
    while remaining:
        last = positions[order[-1]]
        nearest = min(
            remaining,
            key=lambda index: float(np.linalg.norm(positions[index] - last)),
        )
        order.append(nearest)
        remaining.remove(nearest)
    return order


@dataclass(frozen=True)
class MeshSyncResult:
    """Metrics from an N-node decentralized hybrid chain."""

    positions: np.ndarray
    chain: list[int]
    edge_distances_m: list[float]
    edge_snrs_db: list[float]
    edge_residuals: torch.Tensor  # (num_edges, num_substeps)
    array_gain: torch.Tensor
    detected: torch.Tensor
    steady: torch.Tensor
    airtime_fraction: float
    device: torch.device

    @property
    def num_nodes(self) -> int:
        return len(self.chain)

    @property
    def detection_rate(self) -> float:
        return torch.mean(self.detected.to(torch.float64)).item()

    @property
    def edge_steady_rms(self) -> list[float]:
        values = []
        for row in self.edge_residuals:
            if torch.any(self.steady):
                values.append(
                    torch.sqrt(torch.mean(row[self.steady].square())).item()
                )
            else:
                values.append(float("nan"))
        return values

    @property
    def worst_edge_rms(self) -> float:
        clean = [v for v in self.edge_steady_rms if v == v]
        return max(clean) if clean else float("nan")

    @property
    def mean_array_gain(self) -> float:
        if not torch.any(self.steady):
            return float("nan")
        return torch.mean(self.array_gain[self.steady]).item()


def run_decentralized_hybrid_mesh(
    settings: SDRSimulationConfig = SDRSimulationConfig(),
    num_nodes: int = 4,
    micro_pilots_per_interval: int = 4,
    anchor_every_intervals: int = 5,
    micro_sequence_length: int = 255,
    micro_cp_length: int = 32,
    channel_drift_std_rad: float = 0.01,
    radius_m: float = 500.0,
    path_loss_exponent: float = 2.7,
    reference_distance_m: float = 500.0,
    control: str = "symmetric",
) -> MeshSyncResult:
    """N-node decentralized hybrid over a random deployment.

    ``control`` selects the consensus control law:
      "symmetric"    every edge corrects every substep; endpoints apply
                     degree-weighted halves (Jacobi-style; the baseline
                     that exhibits the consensus tax)
      "alternating"  edges take turns by parity (Gauss-Seidel-style):
                     no node receives two corrections at once, so each
                     correction applies at full half/half strength
      "directed"     spanning-tree control with an elected root: each
                     node retunes FULLY toward its chain parent.
                     Decentralized in fault structure (the root is
                     elected, not designated; re-election on failure is
                     assumed out of band), centralized in control.
    """

    if num_nodes < 2:
        raise ValueError("the mesh needs at least 2 nodes")
    if control not in ("symmetric", "alternating", "directed"):
        raise ValueError("control must be symmetric, alternating, or directed")
    substeps = micro_pilots_per_interval + 1
    dt = settings.sync_interval / substeps
    dt_samples = int(round(dt * settings.sample_rate))

    device = resolve_device(settings.device)
    torch.manual_seed(settings.seed)
    sionna_config.seed = settings.seed
    generator = torch.Generator(device=device)
    generator.manual_seed(settings.seed + 1)

    positions = place_stations(num_nodes, radius_m, settings.seed)
    chain = _chain_order(positions)
    edges = [(chain[k], chain[k + 1]) for k in range(num_nodes - 1)]
    degree = {node: 0 for node in chain}
    for p, q in edges:
        degree[p] += 1
        degree[q] += 1

    # Oscillators: node 0 starts at the master values, the others spread
    # linearly up to the configured slave offset (deterministic per seed).
    frequency_process_std = 2.0 * math.pi * settings.frequency_process_std_hz
    substep_covariance = torch.diag(
        torch.tensor(
            [
                settings.phase_process_std_rad**2 / substeps,
                frequency_process_std**2 / substeps,
            ],
            dtype=REAL_DTYPE,
            device=device,
        )
    )
    oscillators = []
    for index in range(num_nodes):
        fraction = index / max(num_nodes - 1, 1)
        oscillators.append(
            Oscillator(
                settings.master_initial_phase
                + fraction
                * (settings.slave_initial_phase - settings.master_initial_phase),
                2.0
                * math.pi
                * (
                    settings.master_initial_frequency_hz
                    + fraction
                    * (
                        settings.slave_initial_frequency_hz
                        - settings.master_initial_frequency_hz
                    )
                ),
                dt,
                substep_covariance,
                device,
                generator,
            )
        )
    flickers = [
        _FlickerFrequencyNoise(
            settings.flicker_frequency_std_hz,
            dt,
            settings.num_iterations * settings.sync_interval,
            device,
            generator,
        )
        for _ in range(num_nodes)
    ]
    flicker_previous = [
        torch.zeros((), dtype=REAL_DTYPE, device=device) for _ in range(num_nodes)
    ]

    # Per-edge machinery: link budget from geometry, radio links, EKF.
    full_preamble = make_sync_preamble(settings, device)
    micro_preamble = _make_micro_preamble(
        micro_sequence_length, micro_cp_length, device
    )
    edge_state = []
    total_airtime = 0.0
    for p, q in edges:
        distance = max(float(np.linalg.norm(positions[p] - positions[q])), 1.0)
        snr_db = min(
            settings.snr_db
            - 10.0
            * path_loss_exponent
            * math.log10(distance / reference_distance_m),
            MAX_LINK_SNR_DB,
        )
        edge_settings = replace(settings, snr_db=snr_db)
        micro_settings = replace(edge_settings, timing_jitter_samples=0)
        link_fwd = SDRRadioLink(edge_settings, full_preamble, device, generator)
        link_rev = SDRRadioLink(
            edge_settings, full_preamble, device, generator, mirror_of=link_fwd
        )
        micro_link = SDRRadioLink(
            micro_settings,
            micro_preamble,
            device,
            generator,
            mirror_of=link_fwd,
            captures_per_interval=substeps,
        )
        synchronizer = SDRSynchronizer(edge_settings, full_preamble)
        oneway_noise = _measurement_covariance(edge_settings, full_preamble, device)
        snr = 10.0 ** (edge_settings.snr_db / 10.0)
        micro_var = 1.0 / (2.0 * snr * micro_sequence_length)
        micro_frame = micro_cp_length + micro_sequence_length
        micro_var += settings.phase_noise_std_rad**2 * (
            micro_cp_length + micro_frame / 3.0
        )
        micro_var += (
            settings.phase_noise_white_pm_std_rad**2 / micro_sequence_length
        )
        micro_noise = torch.diag(
            torch.tensor([micro_var, micro_var], dtype=REAL_DTYPE, device=device)
        )
        anchor_noise = torch.diag(
            torch.stack(
                (
                    0.5 * oneway_noise[0, 0],
                    0.5 * oneway_noise[1, 1],
                    0.5 * oneway_noise[2, 2],
                    0.5 * oneway_noise[0, 0],
                    0.5 * oneway_noise[1, 1],
                )
            )
        )
        white_fm_substep = settings.phase_noise_std_rad**2 * dt_samples
        flicker_innovation = flickers[0].innovation_variance
        process = torch.diag(
            torch.tensor(
                [
                    2.0 * substep_covariance[0, 0].item() + white_fm_substep,
                    2.0 * substep_covariance[1, 1].item()
                    + 2.0 * flicker_innovation,
                    channel_drift_std_rad**2 / substeps,
                ],
                dtype=REAL_DTYPE,
                device=device,
            )
        )
        full_capture = link_fwd.input_length + link_fwd.l_tot - 1
        micro_capture = micro_link.input_length + micro_link.l_tot - 1
        interval_samples = int(
            round(settings.sync_interval * settings.sample_rate)
        )
        total_airtime += (
            full_capture * (1.0 + 1.0 / anchor_every_intervals)
            + micro_pilots_per_interval * micro_capture
        ) / interval_samples
        edge_state.append(
            {
                "index": len(edge_state),
                "pair": (p, q),
                "distance": distance,
                "snr_db": snr_db,
                "link_fwd": link_fwd,
                "link_rev": link_rev,
                "micro_link": micro_link,
                "synchronizer": synchronizer,
                "oneway_noise": oneway_noise,
                "micro_noise": micro_noise,
                "anchor_noise": anchor_noise,
                "ekf": _JointPhaseChannelEKF(dt, process, device),
                "acquired": False,
                "settled": 0,
                "calibrated": False,
                "corrections_loaded": False,
                "pending": {},
                "corr_freq": {
                    p: torch.zeros((), dtype=REAL_DTYPE, device=device),
                    q: torch.zeros((), dtype=REAL_DTYPE, device=device),
                },
            }
        )

    node_corr_freq = [
        torch.zeros((), dtype=REAL_DTYPE, device=device) for _ in range(num_nodes)
    ]
    micro_expected_start = (
        settings.capture_guard_samples
        - edge_state[0]["micro_link"].l_min
        + micro_cp_length
    )
    chain_bias = math.radians(settings.twoway_chain_asymmetry_deg)
    per_node_walk = settings.phase_noise_std_rad / math.sqrt(2.0)

    edge_residual_rows: list[list[torch.Tensor]] = [[] for _ in edges]
    gain_history: list[torch.Tensor] = []
    detected_history: list[bool] = []
    steady_history: list[bool] = []

    total_substeps = settings.num_iterations * substeps
    for substep in range(total_substeps):
        iteration = substep // substeps
        at_frame_slot = substep % substeps == 0
        is_anchor = at_frame_slot and iteration % anchor_every_intervals == 0

        # Advance every oscillator and its flicker process.
        for index in range(num_nodes):
            oscillators[index].step()
            now = flickers[index].step()
            oscillators[index].state[1] = (
                oscillators[index].state[1] + (now - flicker_previous[index])
            )
            flicker_previous[index] = now

        # Load due corrections: gather each node's net applied delta, then
        # tell every edge's filter what its endpoints actually did (the
        # side channel).
        node_delta = [
            torch.zeros(2, dtype=REAL_DTYPE, device=device)
            for _ in range(num_nodes)
        ]
        any_loaded = False
        for edge in edge_state:
            due = edge["pending"].pop(substep, None)
            if due is None:
                continue
            p, q = edge["pair"]
            if control == "directed":
                # Child retunes fully toward its parent (elected-root tree).
                node_delta[q] = node_delta[q] + due
            elif control == "alternating":
                # Parity scheduling guarantees no simultaneous corrections
                # on a shared node, so full half/half strength is safe.
                node_delta[p] = node_delta[p] - due / 2.0
                node_delta[q] = node_delta[q] + due / 2.0
            else:
                node_delta[p] = node_delta[p] - due / (2.0 * degree[p])
                node_delta[q] = node_delta[q] + due / (2.0 * degree[q])
            edge["corrections_loaded"] = True
            any_loaded = True
        if any_loaded:
            for index in range(num_nodes):
                if torch.any(node_delta[index] != 0.0):
                    oscillators[index].apply_correction(node_delta[index])
                    node_corr_freq[index] = (
                        node_corr_freq[index] + node_delta[index][1]
                    )
            for edge in edge_state:
                p, q = edge["pair"]
                relative_applied = node_delta[p] - node_delta[q]
                if torch.any(relative_applied != 0.0):
                    edge["ekf"].reset_after_correction(-relative_applied)

        # Pi-ambiguity calibration per edge. Unlike the pairwise loops
        # (one-shot after convergence), the mesh converges slowly on
        # interior edges (under-relaxed, cross-coupled corrections), so a
        # single early check can pass and the edge can later drift to the
        # pi fixed point. Model instead a periodic 1-bit combining check
        # at the anchor cadence, with hysteresis (flip only when clearly
        # destructive) to prevent chatter near +-pi/2.
        if is_anchor:
            for position, edge in enumerate(edge_state):
                if not edge["corrections_loaded"]:
                    continue
                edge["settled"] += 1
                p, q = edge["pair"]
                if torch.cos(
                    oscillators[p].state[0] - oscillators[q].state[0]
                ) < -0.2:
                    flip = torch.tensor(
                        [math.pi, 0.0], dtype=REAL_DTYPE, device=device
                    )
                    for downstream in chain[position + 1 :]:
                        oscillators[downstream].apply_correction(flip)
                    edge["ekf"].state[2] = wrap_phase(
                        edge["ekf"].state[2] - math.pi
                    )
                if edge["settled"] >= 2:
                    edge["calibrated"] = True

        # Every edge exchanges its pilot for this substep (TDMA).
        all_detected = True
        for row, edge in zip(edge_residual_rows, edge_state):
            p, q = edge["pair"]
            ekf = edge["ekf"]
            physical_p = oscillators[p].state[1] - node_corr_freq[p]
            physical_q = oscillators[q].state[1] - node_corr_freq[q]
            if settings.sample_clock_offset_ppm is not None:
                sfo = settings.sample_clock_offset_ppm
            else:
                sfo = float(
                    (physical_q - physical_p).item()
                    / (2.0 * math.pi * settings.carrier_frequency_hz)
                    * 1e6
                )

            if at_frame_slot:
                capture = edge["link_fwd"].capture(
                    oscillators[p], oscillators[q], iteration, sfo
                )
                oscillators[p].state[0] = wrap_phase(
                    oscillators[p].state[0] + capture.lo_walk_end
                )
                forward = edge["synchronizer"].estimate(capture.samples)
                if is_anchor:
                    capture_rev = edge["link_rev"].capture(
                        oscillators[q], oscillators[p], iteration, -sfo
                    )
                    oscillators[q].state[0] = wrap_phase(
                        oscillators[q].state[0] + capture_rev.lo_walk_end
                    )
                    reverse = edge["synchronizer"].estimate(capture_rev.samples)
                    detected = forward.detected and reverse.detected
                else:
                    detected = forward.detected and edge["acquired"]
            else:
                capture = edge["micro_link"].capture(
                    oscillators[p], oscillators[q], iteration, sfo
                )
                oscillators[p].state[0] = wrap_phase(
                    oscillators[p].state[0] + capture.lo_walk_end
                )
                found, micro_phase = _estimate_micro_phase(
                    capture.samples,
                    micro_preamble.long_sequence,
                    micro_expected_start,
                    ekf.state[1],
                    settings.sample_period,
                )
                detected = found and edge["acquired"]

            ekf.predict()
            if detected:
                if is_anchor:
                    combined_half = wrap_phase(
                        wrap_phase(forward.phase - reverse.phase) / 2.0
                        + chain_bias
                    )
                    if not edge["acquired"]:
                        theta_obs = _pick_half_phase(
                            combined_half, torch.zeros_like(combined_half)
                        )
                        frequency_obs = (
                            forward.frequency - reverse.frequency
                        ) / 2.0
                        channel_obs = wrap_phase(forward.phase - theta_obs)
                        ekf.state = torch.stack(
                            (theta_obs, frequency_obs, channel_obs)
                        )
                        ekf.covariance = torch.diag(
                            torch.stack(
                                (
                                    edge["anchor_noise"][0, 0],
                                    edge["anchor_noise"][2, 2],
                                    edge["anchor_noise"][3, 3],
                                )
                            )
                        )
                        edge["acquired"] = True
                    else:
                        theta_obs = _pick_half_phase(
                            combined_half, wrap_phase(ekf.state[0])
                        )
                        frequency_obs = (
                            forward.frequency - reverse.frequency
                        ) / 2.0
                        channel_obs = wrap_phase(forward.phase - theta_obs)
                        ekf.update(
                            torch.stack(
                                (
                                    torch.cos(theta_obs),
                                    torch.sin(theta_obs),
                                    frequency_obs,
                                    torch.cos(channel_obs),
                                    torch.sin(channel_obs),
                                )
                            ),
                            edge["anchor_noise"],
                            _observe_anchor,
                            _jacobian_anchor,
                        )
                elif at_frame_slot:
                    ekf.update(
                        torch.stack(
                            (
                                torch.cos(forward.phase),
                                torch.sin(forward.phase),
                                forward.frequency,
                            )
                        ),
                        edge["oneway_noise"],
                        _observe_sum_with_frequency,
                        _jacobian_sum_with_frequency,
                    )
                else:
                    ekf.update(
                        torch.stack(
                            (torch.cos(micro_phase), torch.sin(micro_phase))
                        ),
                        edge["micro_noise"],
                        _observe_sum_only,
                        _jacobian_sum_only,
                    )
                if edge["acquired"] and (
                    control != "alternating"
                    or (substep + 1) % 2 == edge["index"] % 2
                ):
                    predicted = ekf.transition @ ekf.state
                    edge["pending"][substep + 1] = _quantize_correction(
                        predicted[:2], settings
                    )
            all_detected = all_detected and detected
            row.append(
                wrap_phase(
                    oscillators[p].state[0] - oscillators[q].state[0]
                ).clone()
            )

        # Dead-time LO walk per node (per-node rate, see module docstring):
        # only the part of the substep not already covered by this
        # substep's capture.
        if at_frame_slot:
            used_samples = edge_state[0]["link_fwd"].input_length
        else:
            used_samples = edge_state[0]["micro_link"].input_length
        remainder_samples = max(0, dt_samples - used_samples)
        if per_node_walk > 0.0 and remainder_samples > 0:
            for index in range(num_nodes):
                oscillators[index].state[0] = wrap_phase(
                    oscillators[index].state[0]
                    + torch.randn(
                        (), dtype=REAL_DTYPE, device=device, generator=generator
                    )
                    * per_node_walk
                    * math.sqrt(remainder_samples)
                )

        phases = torch.stack(
            [oscillators[index].state[0] for index in range(num_nodes)]
        )
        phasors = torch.exp(1j * phases.to(torch.complex128))
        gain_history.append(
            (torch.abs(torch.sum(phasors)) ** 2 / num_nodes**2).real.to(
                torch.float64
            )
        )
        detected_history.append(all_detected)
        steady_history.append(
            all_detected
            and all(edge["corrections_loaded"] for edge in edge_state)
            and all(edge["calibrated"] for edge in edge_state)
        )

    return MeshSyncResult(
        positions=positions,
        chain=chain,
        edge_distances_m=[edge["distance"] for edge in edge_state],
        edge_snrs_db=[edge["snr_db"] for edge in edge_state],
        edge_residuals=torch.stack(
            [torch.stack(row).detach().cpu() for row in edge_residual_rows]
        ),
        array_gain=torch.stack(gain_history).detach().cpu(),
        detected=torch.tensor(detected_history, dtype=torch.bool),
        steady=torch.tensor(steady_history, dtype=torch.bool),
        airtime_fraction=total_airtime,
        device=device,
    )


def run_dfpc_mesh(
    settings: SDRSimulationConfig = SDRSimulationConfig(),
    num_nodes: int = 4,
    use_kf: bool = False,
    radius_m: float = 500.0,
    path_loss_exponent: float = 2.7,
    reference_distance_m: float = 500.0,
) -> MeshSyncResult:
    """N-node DFPC / KF-DFPC over the actual channel, same mesh harness.

    Identical deployment, chain topology, shared oscillators, symmetric
    degree-weighted control, side-channel correction reporting, and
    periodic 1-bit branch checks as ``run_decentralized_hybrid_mesh`` -
    only the measurement/estimation layer differs: each edge performs
    one reciprocal two-way exchange per sync interval (the DFPC cadence)
    and the correction is the raw half-difference observation
    (``use_kf=False``, DFPC) or a 2-state Kalman posterior per edge
    (``use_kf=True``, KF-DFPC). No micro-pilot tier, no channel state -
    faithful to the paper's structure, steelmanned with the reciprocity
    side channel it already assumes.
    """

    from ota_sync.core import PhaseFrequencyEKF

    if num_nodes < 2:
        raise ValueError("the mesh needs at least 2 nodes")
    dt = settings.sync_interval
    dt_samples = int(round(dt * settings.sample_rate))

    device = resolve_device(settings.device)
    torch.manual_seed(settings.seed)
    sionna_config.seed = settings.seed
    generator = torch.Generator(device=device)
    generator.manual_seed(settings.seed + 1)

    positions = place_stations(num_nodes, radius_m, settings.seed)
    chain = _chain_order(positions)
    edges = [(chain[k], chain[k + 1]) for k in range(num_nodes - 1)]
    degree = {node: 0 for node in chain}
    for p, q in edges:
        degree[p] += 1
        degree[q] += 1

    frequency_process_std = 2.0 * math.pi * settings.frequency_process_std_hz
    interval_covariance = torch.diag(
        torch.tensor(
            [
                settings.phase_process_std_rad**2,
                frequency_process_std**2,
            ],
            dtype=REAL_DTYPE,
            device=device,
        )
    )
    oscillators = []
    for index in range(num_nodes):
        fraction = index / max(num_nodes - 1, 1)
        oscillators.append(
            Oscillator(
                settings.master_initial_phase
                + fraction
                * (settings.slave_initial_phase - settings.master_initial_phase),
                2.0
                * math.pi
                * (
                    settings.master_initial_frequency_hz
                    + fraction
                    * (
                        settings.slave_initial_frequency_hz
                        - settings.master_initial_frequency_hz
                    )
                ),
                dt,
                interval_covariance,
                device,
                generator,
            )
        )
    flickers = [
        _FlickerFrequencyNoise(
            settings.flicker_frequency_std_hz,
            dt,
            settings.num_iterations * settings.sync_interval,
            device,
            generator,
        )
        for _ in range(num_nodes)
    ]
    flicker_previous = [
        torch.zeros((), dtype=REAL_DTYPE, device=device) for _ in range(num_nodes)
    ]

    full_preamble = make_sync_preamble(settings, device)
    white_fm_interval = settings.phase_noise_std_rad**2 * dt_samples
    edge_state = []
    total_airtime = 0.0
    for p, q in edges:
        distance = max(float(np.linalg.norm(positions[p] - positions[q])), 1.0)
        snr_db = min(
            settings.snr_db
            - 10.0
            * path_loss_exponent
            * math.log10(distance / reference_distance_m),
            MAX_LINK_SNR_DB,
        )
        edge_settings = replace(settings, snr_db=snr_db)
        link_fwd = SDRRadioLink(edge_settings, full_preamble, device, generator)
        link_rev = SDRRadioLink(
            edge_settings, full_preamble, device, generator, mirror_of=link_fwd
        )
        synchronizer = SDRSynchronizer(edge_settings, full_preamble)
        measurement_noise = 0.5 * _measurement_covariance(
            edge_settings, full_preamble, device
        )
        ekf = None
        if use_kf:
            ekf = PhaseFrequencyEKF(
                dt,
                2.0 * interval_covariance
                + torch.diag(
                    torch.tensor(
                        [
                            white_fm_interval,
                            2.0 * flickers[0].innovation_variance,
                        ],
                        dtype=REAL_DTYPE,
                        device=device,
                    )
                ),
                measurement_noise,
                device,
                initial_covariance=torch.diag(
                    torch.tensor(
                        [math.pi**2, (2.0 * math.pi * 50e3) ** 2],
                        dtype=REAL_DTYPE,
                        device=device,
                    )
                ),
            )
        full_capture = link_fwd.input_length + link_fwd.l_tot - 1
        interval_samples = int(
            round(settings.sync_interval * settings.sample_rate)
        )
        total_airtime += 2.0 * full_capture / interval_samples
        edge_state.append(
            {
                "pair": (p, q),
                "distance": distance,
                "snr_db": snr_db,
                "link_fwd": link_fwd,
                "link_rev": link_rev,
                "synchronizer": synchronizer,
                "noise": measurement_noise,
                "ekf": ekf,
                "acquired": False,
                "settled": 0,
                "calibrated": False,
                "corrections_loaded": False,
                "pending": {},
                "last_offset": torch.zeros((), dtype=REAL_DTYPE, device=device),
            }
        )

    node_corr_freq = [
        torch.zeros((), dtype=REAL_DTYPE, device=device) for _ in range(num_nodes)
    ]
    chain_bias = math.radians(settings.twoway_chain_asymmetry_deg)
    per_node_walk = settings.phase_noise_std_rad / math.sqrt(2.0)
    full_capture_samples = (
        edge_state[0]["link_fwd"].input_length + edge_state[0]["link_fwd"].l_tot - 1
    )

    edge_residual_rows: list[list[torch.Tensor]] = [[] for _ in edges]
    gain_history: list[torch.Tensor] = []
    detected_history: list[bool] = []
    steady_history: list[bool] = []

    for iteration in range(settings.num_iterations):
        for index in range(num_nodes):
            oscillators[index].step()
            now = flickers[index].step()
            oscillators[index].state[1] = (
                oscillators[index].state[1] + (now - flicker_previous[index])
            )
            flicker_previous[index] = now

        node_delta = [
            torch.zeros(2, dtype=REAL_DTYPE, device=device)
            for _ in range(num_nodes)
        ]
        any_loaded = False
        for edge in edge_state:
            due = edge["pending"].pop(iteration, None)
            if due is None:
                continue
            p, q = edge["pair"]
            node_delta[p] = node_delta[p] - due / (2.0 * degree[p])
            node_delta[q] = node_delta[q] + due / (2.0 * degree[q])
            edge["corrections_loaded"] = True
            any_loaded = True
        if any_loaded:
            for index in range(num_nodes):
                if torch.any(node_delta[index] != 0.0):
                    oscillators[index].apply_correction(node_delta[index])
                    node_corr_freq[index] = (
                        node_corr_freq[index] + node_delta[index][1]
                    )
            for edge in edge_state:
                if edge["ekf"] is not None:
                    p, q = edge["pair"]
                    relative_applied = node_delta[p] - node_delta[q]
                    if torch.any(relative_applied != 0.0):
                        edge["ekf"].reset_after_correction(-relative_applied)

        # Periodic 1-bit branch check, same policy as the hybrid mesh.
        for position, edge in enumerate(edge_state):
            if not edge["corrections_loaded"]:
                continue
            edge["settled"] += 1
            p, q = edge["pair"]
            if torch.cos(
                oscillators[p].state[0] - oscillators[q].state[0]
            ) < -0.2:
                flip = torch.tensor(
                    [math.pi, 0.0], dtype=REAL_DTYPE, device=device
                )
                for downstream in chain[position + 1 :]:
                    oscillators[downstream].apply_correction(flip)
            if edge["settled"] >= 2:
                edge["calibrated"] = True

        all_detected = True
        for row, edge in zip(edge_residual_rows, edge_state):
            p, q = edge["pair"]
            physical_p = oscillators[p].state[1] - node_corr_freq[p]
            physical_q = oscillators[q].state[1] - node_corr_freq[q]
            if settings.sample_clock_offset_ppm is not None:
                sfo = settings.sample_clock_offset_ppm
            else:
                sfo = float(
                    (physical_q - physical_p).item()
                    / (2.0 * math.pi * settings.carrier_frequency_hz)
                    * 1e6
                )
            capture_fwd = edge["link_fwd"].capture(
                oscillators[p], oscillators[q], iteration, sfo
            )
            oscillators[p].state[0] = wrap_phase(
                oscillators[p].state[0] + capture_fwd.lo_walk_end
            )
            capture_rev = edge["link_rev"].capture(
                oscillators[q], oscillators[p], iteration, -sfo
            )
            oscillators[q].state[0] = wrap_phase(
                oscillators[q].state[0] + capture_rev.lo_walk_end
            )
            forward = edge["synchronizer"].estimate(capture_fwd.samples)
            reverse = edge["synchronizer"].estimate(capture_rev.samples)
            detected = forward.detected and reverse.detected

            if edge["ekf"] is not None:
                edge["ekf"].predict()
            if detected:
                combined_half = wrap_phase(
                    wrap_phase(forward.phase - reverse.phase) / 2.0 + chain_bias
                )
                frequency_obs = (forward.frequency - reverse.frequency) / 2.0
                if edge["ekf"] is not None:
                    if not edge["acquired"]:
                        theta_obs = _pick_half_phase(
                            combined_half, torch.zeros_like(combined_half)
                        )
                        edge["ekf"].state = torch.stack(
                            (theta_obs, frequency_obs)
                        )
                        edge["ekf"].covariance = torch.diag(
                            torch.stack(
                                (edge["noise"][0, 0], edge["noise"][2, 2])
                            )
                        )
                        edge["acquired"] = True
                    else:
                        theta_obs = _pick_half_phase(
                            combined_half, wrap_phase(edge["ekf"].state[0])
                        )
                        edge["ekf"].update(
                            torch.stack(
                                (
                                    torch.cos(theta_obs),
                                    torch.sin(theta_obs),
                                    frequency_obs,
                                )
                            )
                        )
                    predicted = edge["ekf"].transition @ edge["ekf"].state
                    edge["pending"][iteration + 1] = _quantize_correction(
                        predicted, settings
                    )
                else:
                    theta_obs = _pick_half_phase(
                        combined_half, edge["last_offset"]
                    )
                    edge["last_offset"] = theta_obs.clone()
                    edge["acquired"] = True
                    edge["pending"][iteration + 1] = _quantize_correction(
                        torch.stack((theta_obs, frequency_obs)), settings
                    )
            all_detected = all_detected and detected
            row.append(
                wrap_phase(
                    oscillators[p].state[0] - oscillators[q].state[0]
                ).clone()
            )

        remainder_samples = max(0, dt_samples - 2 * full_capture_samples)
        if per_node_walk > 0.0 and remainder_samples > 0:
            for index in range(num_nodes):
                oscillators[index].state[0] = wrap_phase(
                    oscillators[index].state[0]
                    + torch.randn(
                        (), dtype=REAL_DTYPE, device=device, generator=generator
                    )
                    * per_node_walk
                    * math.sqrt(remainder_samples)
                )

        phases = torch.stack(
            [oscillators[index].state[0] for index in range(num_nodes)]
        )
        phasors = torch.exp(1j * phases.to(torch.complex128))
        gain_history.append(
            (torch.abs(torch.sum(phasors)) ** 2 / num_nodes**2).real.to(
                torch.float64
            )
        )
        detected_history.append(all_detected)
        steady_history.append(
            all_detected
            and all(edge["corrections_loaded"] for edge in edge_state)
            and all(edge["calibrated"] for edge in edge_state)
        )

    return MeshSyncResult(
        positions=positions,
        chain=chain,
        edge_distances_m=[edge["distance"] for edge in edge_state],
        edge_snrs_db=[edge["snr_db"] for edge in edge_state],
        edge_residuals=torch.stack(
            [torch.stack(row).detach().cpu() for row in edge_residual_rows]
        ),
        array_gain=torch.stack(gain_history).detach().cpu(),
        detected=torch.tensor(detected_history, dtype=torch.bool),
        steady=torch.tensor(steady_history, dtype=torch.bool),
        airtime_fraction=total_airtime,
        device=device,
    )
