"""Cadence-aware variant of the sandbox open-loop graph testbed.

Adapted copy of phase_sync_idea/openloop_topology_study.run_openloop_graph
with ONE physics change, needed because direction A services edges far
below once-per-interval: every edge's Kalman filter runs its predict
step EVERY interval, whether or not the edge is serviced. The original
predicts only at service, so at a service gap of m intervals it
propagates one interval of dynamics across an m-interval coast - the
residual pair frequency times the unmodeled gap mis-predicts the phase,
the mod-pi branch pick goes wrong, and the loop flip-storms. Controls
that established this (run before this file was written, recorded in
RESULTS_A.md): storms persist at zero initial frequency offset, at
every pilot length 255..2047, and soften only as the service gap
shrinks - all consistent with gap-blind prediction, none with the
alternative hypotheses.

The sibling file is left untouched (it is in use by other directions
and is correct for its own every-interval campaigns).

Everything else is verbatim from the original: randomized initial
conditions, per-edge SNR from path loss, half-difference measurement,
degree accounting, parity-staggered correction issuance, periodic
subtree branch check with flip-storm re-acquisition escape, dead-time
walk, and the same return structure.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "phase_sync_idea"))

from sionna.phy import config as sionna_config  # noqa: E402

import openloop_topology_study as topo  # noqa: E402
from ota_sync.coherent import _pick_half_phase  # noqa: E402
from ota_sync.core import (  # noqa: E402
    REAL_DTYPE,
    Oscillator,
    PhaseFrequencyEKF,
    resolve_device,
    wrap_phase,
)
from ota_sync.network import MAX_LINK_SNR_DB  # noqa: E402
from ota_sync.sdr import (  # noqa: E402
    SDRRadioLink,
    SDRSimulationConfig,
    SDRSynchronizer,
    _FlickerFrequencyNoise,
    _measurement_covariance,
    _quantize_correction,
    make_sync_preamble,
)


def run_openloop_graph_cadence(
    settings: SDRSimulationConfig,
    num_nodes: int,
    edge_spec: list[tuple[int, int, str]],
    *,
    budget_edges_per_interval: int | None = None,
    acquisition_intervals: int = 10,
    branch_check: bool = True,
    init_phase_span: float = 0.5,
    init_cfo_span_hz: float = 1500.0,
    radius_m: float = 500.0,
    path_loss_exponent: float = 2.7,
    reference_distance_m: float = 500.0,
):
    t0 = time.time()
    device = resolve_device(settings.device)
    torch.manual_seed(settings.seed)
    sionna_config.seed = settings.seed
    generator = torch.Generator(device=device)
    generator.manual_seed(settings.seed + 1)
    init_rng = np.random.default_rng(settings.seed + 977)

    positions = topo.place_stations(num_nodes, radius_m, settings.seed)
    dt = settings.sync_interval
    dt_samples = int(round(dt * settings.sample_rate))

    frequency_process_std = 2.0 * math.pi * settings.frequency_process_std_hz
    interval_covariance = torch.diag(
        torch.tensor(
            [settings.phase_process_std_rad**2, frequency_process_std**2],
            dtype=REAL_DTYPE,
            device=device,
        )
    )
    oscillators = []
    for index in range(num_nodes):
        phase0 = float(init_rng.uniform(-init_phase_span, init_phase_span))
        cfo0 = float(init_rng.uniform(-init_cfo_span_hz, init_cfo_span_hz))
        oscillators.append(
            Oscillator(
                phase0,
                2.0 * math.pi * cfo0,
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
        torch.zeros((), dtype=REAL_DTYPE, device=device)
        for _ in range(num_nodes)
    ]

    preamble = make_sync_preamble(settings, device)
    white_fm_interval = settings.phase_noise_std_rad**2 * dt_samples
    edges = []
    for p, q, kind in edge_spec:
        distance = max(float(np.linalg.norm(positions[p] - positions[q])), 1.0)
        snr_db = min(
            settings.snr_db
            - 10.0 * path_loss_exponent
            * math.log10(distance / reference_distance_m),
            MAX_LINK_SNR_DB,
        )
        edge_settings = replace(settings, snr_db=snr_db)
        link_fwd = SDRRadioLink(edge_settings, preamble, device, generator)
        link_rev = SDRRadioLink(
            edge_settings, preamble, device, generator, mirror_of=link_fwd
        )
        oneway_noise = _measurement_covariance(edge_settings, preamble, device)
        measurement_noise = (
            0.5 * oneway_noise if kind == "two" else oneway_noise
        )
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
        edges.append(
            {
                "pair": (p, q),
                "kind": kind,
                "snr_db": snr_db,
                "link_fwd": link_fwd,
                "link_rev": link_rev,
                "synchronizer": SDRSynchronizer(edge_settings, preamble),
                "ekf": ekf,
                "acquired": False,
                "pending": {},
                "loaded": False,
            }
        )

    node_corr_freq = [
        torch.zeros((), dtype=REAL_DTYPE, device=device)
        for _ in range(num_nodes)
    ]
    chain_bias = math.radians(settings.twoway_chain_asymmetry_deg)
    per_node_walk = settings.phase_noise_std_rad / math.sqrt(2.0)
    capture_samples = (
        edges[0]["link_fwd"].input_length + edges[0]["link_fwd"].l_tot - 1
        if edges else 0
    )
    tree_pairs, _, subtree_of = topo.bfs_subtrees(num_nodes, edge_spec)
    flip_count = 0
    realign_count = 0

    node_rows: list[list[torch.Tensor]] = [[] for _ in range(num_nodes)]
    serviced_count = 0
    detected_count = 0
    rr_pointer = 0

    for iteration in range(settings.num_iterations):
        for index in range(num_nodes):
            oscillators[index].step()
            now = flickers[index].step()
            oscillators[index].state[1] = (
                oscillators[index].state[1] + (now - flicker_previous[index])
            )
            flicker_previous[index] = now

        # THE CADENCE FIX: every edge's filter predicts every interval.
        for edge in edges:
            if edge["acquired"]:
                edge["ekf"].predict()

        node_delta = [
            torch.zeros(2, dtype=REAL_DTYPE, device=device)
            for _ in range(num_nodes)
        ]
        any_loaded = False
        for edge in edges:
            due = edge["pending"].pop(iteration, None)
            if due is None:
                continue
            p, q = edge["pair"]
            node_delta[p] = node_delta[p] - due / 2.0
            node_delta[q] = node_delta[q] + due / 2.0
            edge["loaded"] = True
            any_loaded = True
        if any_loaded:
            for index in range(num_nodes):
                if torch.any(node_delta[index] != 0.0):
                    oscillators[index].apply_correction(node_delta[index])
                    node_corr_freq[index] = (
                        node_corr_freq[index] + node_delta[index][1]
                    )
            for edge in edges:
                p, q = edge["pair"]
                relative = node_delta[p] - node_delta[q]
                if torch.any(relative != 0.0):
                    edge["ekf"].reset_after_correction(-relative)

        if (
            branch_check
            and iteration >= acquisition_intervals
            and iteration % 4 == 0
        ):
            for edge in edges:
                if edge["kind"] != "two" or not edge["loaded"]:
                    continue
                p, q = edge["pair"]
                if (min(p, q), max(p, q)) not in tree_pairs:
                    continue
                if torch.cos(
                    oscillators[p].state[0] - oscillators[q].state[0]
                ) < -0.2:
                    child = q if q in subtree_of and p not in subtree_of.get(
                        q, set()
                    ) else p
                    flip = torch.tensor(
                        [math.pi, 0.0], dtype=REAL_DTYPE, device=device
                    )
                    for member in subtree_of.get(child, {child}):
                        oscillators[member].apply_correction(flip)
                    flip_count += 1
                    last = edge.get("last_flip", -10)
                    if iteration - last <= 4:
                        edge["acquired"] = False
                        edge["ekf"].state = torch.zeros(
                            2, dtype=REAL_DTYPE, device=device
                        )
                        realign_count += 1
                    edge["last_flip"] = iteration

        if (
            budget_edges_per_interval is None
            or iteration < acquisition_intervals
        ):
            serviced = list(range(len(edges)))
        else:
            serviced = [
                (rr_pointer + k) % len(edges)
                for k in range(budget_edges_per_interval)
            ]
            rr_pointer = (
                rr_pointer + budget_edges_per_interval
            ) % len(edges)

        for edge_index in serviced:
            edge = edges[edge_index]
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
            forward = edge["synchronizer"].estimate(capture_fwd.samples)
            if edge["kind"] == "two":
                capture_rev = edge["link_rev"].capture(
                    oscillators[q], oscillators[p], iteration, -sfo
                )
                oscillators[q].state[0] = wrap_phase(
                    oscillators[q].state[0] + capture_rev.lo_walk_end
                )
                reverse = edge["synchronizer"].estimate(capture_rev.samples)
                detected = forward.detected and reverse.detected
            else:
                detected = bool(forward.detected)
            serviced_count += 1
            detected_count += int(detected)
            if not detected:
                continue
            ekf = edge["ekf"]
            if edge["kind"] == "two":
                combined_half = wrap_phase(
                    wrap_phase(forward.phase - reverse.phase) / 2.0
                    + chain_bias
                )
                frequency_obs = (forward.frequency - reverse.frequency) / 2.0
                if not edge["acquired"]:
                    theta_obs = _pick_half_phase(
                        combined_half, torch.zeros_like(combined_half)
                    )
                    ekf.state = torch.stack((theta_obs, frequency_obs))
                    ekf.covariance = torch.diag(
                        torch.stack(
                            (
                                ekf.measurement_covariance[0, 0],
                                ekf.measurement_covariance[2, 2],
                            )
                        )
                    )
                    edge["acquired"] = True
                else:
                    theta_obs = _pick_half_phase(
                        combined_half, wrap_phase(ekf.state[0])
                    )
                    ekf.update(
                        torch.stack(
                            (
                                torch.cos(theta_obs),
                                torch.sin(theta_obs),
                                frequency_obs,
                            )
                        )
                    )
            else:
                theta_obs = wrap_phase(forward.phase)
                frequency_obs = forward.frequency
                if not edge["acquired"]:
                    ekf.state = torch.stack((theta_obs, frequency_obs))
                    ekf.covariance = torch.diag(
                        torch.stack(
                            (
                                ekf.measurement_covariance[0, 0],
                                ekf.measurement_covariance[2, 2],
                            )
                        )
                    )
                    edge["acquired"] = True
                else:
                    ekf.update(
                        torch.stack(
                            (
                                torch.cos(theta_obs),
                                torch.sin(theta_obs),
                                frequency_obs,
                            )
                        )
                    )
            # Parity-stagger issuance whenever edges are (or may be)
            # serviced every interval - the sibling's stagger engaged
            # only for budget=None, leaving budget >= |E| in the
            # simultaneous-Jacobi storm it was built to prevent.
            stagger = (
                budget_edges_per_interval is None
                or budget_edges_per_interval >= len(edges)
            )
            if (not stagger) or (iteration + 1) % 2 == edge_index % 2:
                predicted = ekf.transition @ ekf.state
                edge["pending"][iteration + 1] = _quantize_correction(
                    predicted, settings
                )

        used = capture_samples * (2 if edges else 0)
        remainder = max(0, dt_samples - used)
        if per_node_walk > 0.0 and remainder > 0:
            for index in range(num_nodes):
                oscillators[index].state[0] = wrap_phase(
                    oscillators[index].state[0]
                    + torch.randn(
                        (), dtype=REAL_DTYPE, device=device,
                        generator=generator,
                    )
                    * per_node_walk
                    * math.sqrt(remainder)
                )

        for index in range(num_nodes):
            node_rows[index].append(
                wrap_phase(
                    oscillators[index].state[0] - oscillators[0].state[0]
                ).clone()
            )

    node_traces = torch.stack(
        [torch.stack(row).detach().cpu() for row in node_rows]
    )
    edge_means = []
    steady = slice(settings.num_iterations // 2, settings.num_iterations)
    for edge in edges:
        p, q = edge["pair"]
        diff = wrap_phase(node_traces[p, steady] - node_traces[q, steady])
        mean, std = topo.circ_mean_std(diff)
        edge_means.append({"pair": [p, q], "kind": edge["kind"],
                           "mean": mean, "std": std})
    return {
        "node_traces": node_traces,
        "edge_means": edge_means,
        "detect_rate": detected_count / max(serviced_count, 1),
        "flips": flip_count,
        "realigns": realign_count,
        "wall_s": time.time() - t0,
    }
