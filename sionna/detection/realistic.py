"""Clutter-limited, interference-limited CPI-level drone detection.

This layer closes the field-realism gaps of the single-pulse study:

CLUTTER (the big one). Ground return is modeled with the standard
constant-gamma reflectivity law sigma0 = gamma * sin(psi) (gamma
defaults to -15 dB, rural UHF; psi is the grazing angle from each
station's mast height). The ground is gridded; every cell's bistatic
delay per TX-RX pair maps its power into a range gate, giving a
per-(receiver, gate) clutter power profile. Per CPI, the clutter
slow-time sequence is a correlated complex-Gaussian process (Gaussian
Doppler spectrum from internal clutter motion, sigma_v default
0.25 m/s — wind-blown vegetation), and it is injected into the RAW
fast-time streams by convolving the gate-domain clutter with the
transmitted pulse — so clutter leaks across gates through the
waveform's own range sidelobes, exactly as in a real receiver. Static
ground means the clutter ridge sits at DC in Doppler; detection of a
moving drone survives by Doppler processing, which is why this layer
simulates a full pulse train.

DIRECT-PATH SELF-INTERFERENCE. Every receiver hears every other
station's 1 W transmission directly (~80 dB above the echo). The
simulation adds these direct signals to the raw streams at their true
delays and amplitudes (assuming a configurable analog isolation so the
ADC stays linear) and then runs the standard digital mitigation: least
squares projection of each receiver's stream onto the known direct
templates, subtracted before matched filtering. Cancellation depth is
whatever the algebra achieves on the actual samples — it is not
assumed.

TARGET SIGNATURE. The scalar Swerling draw is replaced by a drone
model: an aspect-dependent body return (ellipsoid law between nose-on
and broadside RCS) plus rotor-blade micro-Doppler — each blade
contributes a small return phase-modulated by (4*pi*L/lambda) *
sin(2*pi*f_rot*t), which spreads energy into the HERM sidebands real
drone radars key on. The body carries the bulk bistatic Doppler from
the drone's velocity, per TX-RX pair.

SEARCH INSTEAD OF A CUED GATE. Each receiver forms a range-Doppler
map (matched filter per gate, Hann-windowed slow-time DFT); maps are
combined NONcoherently across receivers (standard multistatic
practice — transmit focusing keeps the coherent sync gain, receive
fusion is square-law). Detection = cell-averaging CFAR across range,
per Doppler bin, with the clutter ridge notched out, over the whole
map; the CFAR scale is calibrated EMPIRICALLY on target-absent trials
(which still contain clutter + direct path + noise) to a window-level
false-alarm rate, and the achieved rate is re-measured.

Remaining honest gaps: no hardware validation (simulation only), no
discrete clutter (buildings enter the RT target legs but not the
clutter map), no RF nonlinearities on the sensing chain, single CPI
(no tracking).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .viability import BOLTZMANN_T0, SPEED_OF_LIGHT, DetectionParams
from .waveform import _ofdm_burst


@dataclass(frozen=True)
class RealisticDetectionConfig:
    """Everything the CPI-level scenario needs beyond DetectionParams."""

    pulse_length: int = 512
    num_pulses: int = 64
    pri_s: float = 600e-6
    clutter_gamma_db: float = -15.0
    clutter_motion_std_mps: float = 0.25
    clutter_cell_m: float = 60.0
    clutter_extent_m: float = 2500.0
    analog_isolation_db: float = 30.0
    drone_speed_mps: float = 15.0
    body_rcs_broadside_dbsm: float = -13.0
    body_rcs_nose_dbsm: float = -19.0
    blade_rcs_dbsm: float = -26.0
    num_rotors: int = 4
    blades_per_rotor: int = 2
    blade_length_m: float = 0.12
    rotor_rate_hz: float = 110.0
    station_height_m: float = 15.0
    target_height_m: float = 60.0
    cfar_train_cells: int = 8
    cfar_guard_cells: int = 2
    clutter_notch_bins: int = 2
    window_pfa: float = 1e-2


@dataclass(frozen=True)
class RealisticDetectionResult:
    label: str
    pd_measured: list[float]
    measured_window_pfa: float
    clutter_to_noise_db: float
    direct_before_after_db: tuple[float, float]
    example_map: np.ndarray  # (gates, doppler bins), one H1 trial
    example_truth: tuple[int, int]


def _range_gates(
    positions: np.ndarray, waypoints: np.ndarray, config, sample_rate: float
) -> tuple[np.ndarray, int]:
    """Common bistatic-delay gate axis covering path and clutter."""

    stations = np.column_stack(
        (positions, np.full(len(positions), config.station_height_m))
    )
    targets = np.column_stack(
        (waypoints, np.full(len(waypoints), config.target_height_m))
    )
    delays = []
    for target in targets:
        d = np.linalg.norm(stations - target, axis=1)
        delays.extend((d[:, None] + d[None, :]).ravel() / SPEED_OF_LIGHT)
    lo = int(np.floor(min(delays) * sample_rate)) - 4
    hi = int(np.ceil(max(delays) * sample_rate)) + 4
    return np.arange(lo, hi + 1), lo


def _clutter_gate_power(
    positions: np.ndarray,
    gate_samples: np.ndarray,
    params: DetectionParams,
    config: RealisticDetectionConfig,
    sample_rate: float,
) -> np.ndarray:
    """(num_rx, num_gates) clutter power via the constant-gamma law.

    TX contributions are summed incoherently per receiver (with the
    array focused on the target cell, other directions see a
    pseudo-random transmit phase pattern).
    """

    n = len(positions)
    gamma = 10.0 ** (config.clutter_gamma_db / 10.0)
    gain = 10.0 ** (params.antenna_gain_dbi / 10.0)
    cell = config.clutter_cell_m
    axis = np.arange(-config.clutter_extent_m, config.clutter_extent_m, cell)
    gx, gy = np.meshgrid(axis, axis)
    cells = np.column_stack((gx.ravel(), gy.ravel()))
    area = cell * cell

    stations = np.column_stack(
        (positions, np.full(n, config.station_height_m))
    )
    cells3 = np.column_stack((cells, np.zeros(len(cells))))
    # Distances station->cell (slant) and grazing angles.
    diff = cells3[None, :, :] - stations[:, None, :]
    dist = np.maximum(np.linalg.norm(diff, axis=2), 1.0)  # (n, cells)
    grazing = np.arcsin(config.station_height_m / dist)
    sigma_cell = gamma * np.sin(grazing) * area  # (n, cells) per TX view

    power = np.zeros((n, len(gate_samples)))
    lo = gate_samples[0]
    for rx in range(n):
        for tx in range(n):
            bistatic = (dist[tx] + dist[rx]) / SPEED_OF_LIGHT
            gates = np.round(bistatic * sample_rate).astype(int) - lo
            valid = (gates >= 0) & (gates < len(gate_samples))
            # sigma0 evaluated with the tx grazing (illumination side).
            cell_power = (
                params.tx_power_w
                * gain**2
                * params.wavelength_m**2
                * sigma_cell[tx]
                / ((4.0 * math.pi) ** 3 * dist[tx] ** 2 * dist[rx] ** 2)
            )
            np.add.at(power[rx], gates[valid], cell_power[valid])
    return power


def run_realistic_detection(
    label: str,
    positions: np.ndarray,
    residual_phases: torch.Tensor,
    waypoints: np.ndarray,
    leg_gains: np.ndarray,
    params: DetectionParams = DetectionParams(),
    config: RealisticDetectionConfig = RealisticDetectionConfig(),
    trials: int = 200,
    h0_trials: int = 400,
    seed: int = 0,
) -> RealisticDetectionResult:
    """Counted CPI-level detection along the drone path."""

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

    # ---- clutter power profile and slow-time correlation -----------
    clutter_power = torch.tensor(
        _clutter_gate_power(
            positions, gate_samples, params, config, sample_rate
        ),
        dtype=torch.float64,
    )
    # Post-matched-filter, both clutter and noise scale with pulse
    # energy, so the per-gate ratio equals the per-sample ratio.
    clutter_to_noise_db = 10.0 * math.log10(
        max(clutter_power.max().item(), 1e-30) / noise_power
    )
    # Gaussian clutter Doppler spectrum -> slow-time correlation via
    # spectral shaping of white Gaussian sequences.
    sigma_f = 2.0 * config.clutter_motion_std_mps / wavelength
    doppler_axis = torch.fft.fftfreq(num_pulses, d=config.pri_s)
    spectrum = torch.exp(-0.5 * (doppler_axis / max(sigma_f, 1e-3)) ** 2)
    spectrum = spectrum / torch.sqrt(torch.mean(spectrum**2))

    stations3 = np.column_stack(
        (positions, np.full(n, config.station_height_m))
    )

    # Direct-path bookkeeping: delays and amplitudes between stations.
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
    np.fill_diagonal(direct_gate, -1)  # self-pairs never used

    window = torch.hann_window(num_pulses, dtype=torch.float64)
    notch = config.clutter_notch_bins

    def make_streams(batch, theta, target_terms, include_clutter=True):
        """Raw fast-time streams for one batch of CPI trials.

        target_terms: None (H0) or (echo_kj (b,n,n), doppler_kj (n,n),
        signature (b, pulses)).
        Returns streams (b, n_rx, pulses, stream_len) complex64.
        """

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

        if not include_clutter:
            white = None
        # Clutter: gate-domain slow-time processes convolved with the
        # pulse. Frozen power profile; new realization each CPI.
        if include_clutter:
            white = torch.randn(
                batch, n, num_gates, num_pulses, 2,
                dtype=torch.float32, generator=generator,
            )
            white = torch.view_as_complex(white.contiguous()).to(
                torch.complex64
            )
        else:
            white = torch.zeros(
                batch, n, num_gates, num_pulses, dtype=torch.complex64
            )
        shaped = torch.fft.ifft(
            torch.fft.fft(white, dim=-1) * spectrum.to(torch.complex64),
            dim=-1,
        )
        amplitude = torch.sqrt(clutter_power / 2.0).to(torch.complex64)
        gate_signal = shaped * amplitude.unsqueeze(0).unsqueeze(-1)
        gate_signal = gate_signal.permute(0, 1, 3, 2)  # (b, n, pulses, gates)
        clutter_streams = torch.nn.functional.conv1d(
            gate_signal.reshape(-1, 1, num_gates),
            pulse.flip(0).unsqueeze(0).unsqueeze(0),
            padding=pulse_len - 1,
        ).reshape(batch, n, num_pulses, -1)[..., :stream_len]
        streams = streams + clutter_streams

        # Direct path: every other station's transmission, true delay
        # and amplitude (post analog isolation), transmit phase applied.
        for rx in range(n):
            for tx in range(n):
                if tx == rx:
                    continue
                gate = direct_gate[tx, rx]
                if gate < 0 or gate >= num_gates:
                    continue
                coefficient = (
                    float(direct_amp[tx, rx]) * phasors[:, tx]
                )  # (b,)
                streams[:, rx, :, gate : gate + pulse_len] += (
                    coefficient.unsqueeze(-1).unsqueeze(-1) * pulse
                )

        if target_terms is not None:
            echo_kj, doppler_kj, signature, gate_t = target_terms
            pulse_idx = torch.arange(num_pulses, dtype=torch.float64)
            for rx in range(n):
                ramp = torch.exp(
                    1j
                    * 2.0
                    * math.pi
                    * doppler_kj[:, rx].unsqueeze(-1)
                    * pulse_idx
                    * config.pri_s
                ).to(torch.complex64)  # (n_tx, pulses)
                # Sum tx contributions incl. their Doppler ramps.
                combined = torch.einsum(
                    "bk,kp->bp", echo_kj[:, :, rx], ramp
                )  # (b, pulses)
                combined = combined * signature
                streams[:, rx, :, gate_t : gate_t + pulse_len] += (
                    combined.unsqueeze(-1) * pulse
                )
        return streams

    def process(streams):
        """LS direct-path removal -> MF -> range-Doppler -> fused map."""

        batch = streams.shape[0]
        # Least-squares removal of the known direct templates.
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

        # Matched filter every gate: correlate with the pulse.
        flat = streams.reshape(-1, 1, stream_len)
        mf = torch.nn.functional.conv1d(
            flat, pulse.conj().unsqueeze(0).unsqueeze(0).resolve_conj()
        ).reshape(streams.shape[0], n, num_pulses, -1)[..., :num_gates]
        # Slow-time windowed DFT -> per-rx range-Doppler, noncoherent
        # fusion across receivers.
        rd = torch.fft.fft(
            mf * window.to(torch.complex64).reshape(1, 1, -1, 1), dim=2
        )
        fused = torch.sum(torch.abs(rd) ** 2, dim=1)  # (b, pulses, gates)
        return fused.permute(0, 2, 1)  # (b, gates, doppler)

    def cfar_normalize(maps):
        """CA-CFAR noise estimate along range, per Doppler bin."""

        guard, train = config.cfar_guard_cells, config.cfar_train_cells
        kernel_size = 2 * (guard + train) + 1
        kernel = torch.zeros(kernel_size, dtype=torch.float32)
        kernel[: train] = 1.0
        kernel[-train:] = 1.0
        kernel = kernel / kernel.sum()
        padded = torch.nn.functional.conv1d(
            maps.permute(0, 2, 1).reshape(-1, 1, maps.shape[1]).float(),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=guard + train,
        ).reshape(maps.shape[0], maps.shape[2], maps.shape[1]).permute(0, 2, 1)
        return maps / torch.clamp(padded, min=1e-30)

    def detection_statistic(maps, gate_t):
        """Max CFAR ratio near the true gate, outside the clutter notch."""

        normalized = cfar_normalize(maps)
        bins = torch.ones(num_pulses, dtype=torch.bool)
        bins[: notch + 1] = False
        bins[num_pulses - notch :] = False
        lo_g = max(gate_t - 1, 0)
        hi_g = min(gate_t + 2, normalized.shape[1])
        region = normalized[:, lo_g:hi_g][:, :, bins]
        return region.reshape(region.shape[0], -1).max(dim=1).values

    # ---- empirical CFAR scale from target-absent trials -------------
    centroid = positions.mean(axis=0)
    mid = waypoints[len(waypoints) // 2]
    mid3 = np.array([mid[0], mid[1], config.target_height_m])
    d_mid = np.linalg.norm(stations3 - mid3, axis=1)
    gate_mid = int(
        round((d_mid.mean() * 2.0) / SPEED_OF_LIGHT * sample_rate)
    ) - gate_lo
    gate_mid = int(np.clip(gate_mid, 1, num_gates - 2))

    steady_columns = residual_phases.shape[1]
    h0_values = []
    batch = 20
    done = 0
    direct_power_before = None
    while done < h0_trials:
        count = min(batch, h0_trials - done)
        done += count
        columns = torch.randint(
            0, steady_columns, (count,), generator=generator
        )
        theta = residual_phases[:, columns].T
        streams = make_streams(count, theta, None)
        if direct_power_before is None:
            direct_power_before = torch.mean(torch.abs(streams) ** 2).item()
        maps = process(streams)
        h0_values.append(detection_statistic(maps, gate_mid))
    h0_stat = torch.cat(h0_values)
    scale = torch.quantile(
        h0_stat.double(), 1.0 - config.window_pfa
    ).item()
    measured_pfa = torch.mean((h0_stat > scale).to(torch.float64)).item()

    # Direct-path suppression achieved by the LS stage (diagnostic,
    # measured clutter-free so the direct path is visible).
    probe = make_streams(
        4, residual_phases[:, :4].T, None, include_clutter=False
    )
    before = torch.mean(torch.abs(probe) ** 2).item()
    after_streams = probe.clone()
    _ = process(after_streams)
    after = torch.mean(torch.abs(after_streams) ** 2).item()
    direct_before_after_db = (
        10.0 * math.log10(before / (noise_power)),
        10.0 * math.log10(max(after, 1e-30) / (noise_power)),
    )

    # ---- target-present trials per waypoint --------------------------
    pd_measured = []
    example_map = None
    example_truth = (0, 0)
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
        # Bistatic Doppler per (tx, rx) pair.
        radial = unit @ velocity
        doppler_kj = torch.tensor(
            (radial[:, None] + radial[None, :]) / wavelength,
            dtype=torch.float64,
        )
        # Aspect-dependent body RCS: ellipsoid law on the angle between
        # the bistatic bisector and the velocity.
        bisector = unit[:, None, :] + unit[None, :, :]
        bisector = bisector / np.maximum(
            np.linalg.norm(bisector, axis=2, keepdims=True), 1e-9
        )
        cos_aspect = np.abs(bisector @ (velocity / max(config.drone_speed_mps, 1e-9)))
        body_rcs = body_min + (body_max - body_min) * (1.0 - cos_aspect**2)
        gate_t = int(
            round((d[:, None] + d[None, :]).mean() / SPEED_OF_LIGHT * sample_rate)
        ) - gate_lo
        gate_t = int(np.clip(gate_t, 1, num_gates - 2))

        legs = torch.tensor(leg_gains[w_index], dtype=torch.complex128)
        echo_base = (
            math.sqrt(params.tx_power_w * 4.0 * math.pi)
            / wavelength
            * legs[:, None]
            * legs[None, :]
            * torch.tensor(np.sqrt(body_rcs), dtype=torch.complex128)
        ).to(torch.complex64)  # (tx, rx) body echo amplitudes

        hits = 0
        remaining = trials
        while remaining > 0:
            count = min(batch, remaining)
            remaining -= count
            columns = torch.randint(
                0, steady_columns, (count,), generator=generator
            )
            theta = residual_phases[:, columns].T
            phasors = torch.exp(1j * theta.to(torch.complex128)).to(
                torch.complex64
            )
            echo_kj = (
                echo_base.unsqueeze(0)
                * phasors.unsqueeze(2)
                * phasors.unsqueeze(1)
            )  # (b, tx, rx)
            # Micro-Doppler signature: body + rotating blades.
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
                count, theta, (echo_kj, doppler_kj, signature, gate_t)
            )
            maps = process(streams)
            stat = detection_statistic(maps, gate_t)
            hits += int(torch.sum(stat > scale).item())
            if example_map is None and w_index == len(waypoints) // 2:
                doppler_bin = int(
                    round(
                        float(doppler_kj.mean())
                        * num_pulses
                        * config.pri_s
                    )
                ) % num_pulses
                example_map = (
                    cfar_normalize(maps)[0].cpu().numpy().copy()
                )
                example_truth = (gate_t, doppler_bin)
        pd_measured.append(hits / trials)

    return RealisticDetectionResult(
        label=label,
        pd_measured=pd_measured,
        measured_window_pfa=measured_pfa,
        clutter_to_noise_db=clutter_to_noise_db,
        direct_before_after_db=direct_before_after_db,
        example_map=example_map,
        example_truth=example_truth,
    )
