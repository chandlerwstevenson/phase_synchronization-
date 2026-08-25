"""Observability / identifiability of the oscillator-channel phase split.

THE QUESTION. The piggyback architecture tracks x = [theta, omega, psi]
(oscillator phase offset, oscillator frequency offset, propagation
phase) from two observation types:

  one-way (free, rate n per interval):   z1 = theta + psi + v1
  two-way anchor (paid, every K intervals): z2 = (theta, omega, psi) + v2

"psi is constant while the environment is static" is NOT sufficient for
the split to be identifiable - the state DYNAMICS have to separate it.
This module derives when they do.

--------------------------------------------------------------------
(i) THE UNOBSERVABLE DIRECTION UNDER ONE-WAY DATA ALONE

Dynamics x_{k+1} = F x_k + w, F = [[1, dt, 0], [0, 1, 0], [0, 0, 1]];
one-way observation rows H1 = [1, 0, 1] (phase; at frame slots also
[0, 1, 0], frequency). The observability Gramian of (F, H1) is rank 2:
a state perturbation

    v = (1, 0, -1) / sqrt(2)        (move theta up, psi down equally)

satisfies H1 F^k v = [1,0,1] . (1,0,-1) = 0 for every k (F leaves the
theta/psi split of v unchanged because omega's coupling into theta is
the v-orthogonal part), and the frequency row is orthogonal to v too.
So the DIFFERENCE theta - psi is invisible to any amount of one-way
data; only the SUM theta + psi and omega are observable. The proof is
computational in `gramian_rank_and_null` (rank 2, null vector v); with
one anchor row [1, 0, 0] added the Gramian becomes rank 3.

(ii) HOW THE HIDDEN SPLIT DRIFTS

Write d = (theta - psi)/sqrt(2), the null coordinate. Its process
noise per step has variance (q_theta + q_psi)/2 - both the oscillator
walk and the channel walk leak into it, and no one-way update removes
it (the Kalman gain corrects only the directions the innovation is
correlated with; along v the innovation carries nothing). Between
anchors the split variance therefore grows LINEARLY:

    var(d) at m substeps after an anchor
        >= var(d | anchor) + m * (q_theta_sub + q_psi_sub) / 2

With a frozen channel (q_psi = 0) the growth is set by the oscillator
walk alone; with channel motion it accelerates by the Jakes-consistent
per-interval innovation

    q_psi(v) = 2 sigma_c^2 (1 - J0(2 pi f_D T)),   f_D = v f_c / c,

sigma_c^2 = 1/(2 K_rice) the diffuse fraction of the Rician composite
(TDL-D: sigma_c ~ 0.153 rad - the same 153 mrad the resampling studies
measured, because it is the same diffuse power).

The residual theta error inherits half the split error (theta =
(s + d)/sqrt(2) with s observed): the identifiability condition for a
phase budget b over an anchor cycle of K intervals x n substeps is

    P_d(anchor) + K*n*(q_theta_sub + q_psi_sub)/2  <~  2 b^2,

i.e. anchors must arrive before the oscillator+channel walk moves the
hidden coordinate by ~b. Everything on the left is ex ante.

(iii) MISATTRIBUTION (why more free observations can hurt)

The deployed filter assumes a FIXED channel prior q_hat_psi
(0.01 rad / interval in the repo), independent of the true q_psi(v).
When q_psi_true > q_hat_psi the filter is mis-modeled: its gains,
computed from the believed covariance, attribute real channel movement
to the oscillator states, which the loop then actuates on - a phase
error that persists as long as the channel realization does (the
per-seed "locked offset" the studies measured). The right tool is the
MISMATCHED-KALMAN recursion: gains L_k from the believed Riccati
(F, Q_hat, R); true error covariance propagated with the TRUE noise:

    P_true <- (I - L H)(F P_true F' + Q_true)(I - L H)' + L R_true L'

Iterating one full anchor cycle (anchor update, then K*n - 1 one-way
updates, channel innovation added once per interval) to its fixed
point gives the believed and TRUE steady split/theta uncertainty as a
function of (q_theta, q_omega, q_psi_true, q_hat_psi, r1, n, K, r2).
`split_uncertainty_cycle` implements exactly that; `predict_residual`
converts it to a predicted steady residual (prior theta variance,
one-substep actuation latency included) and a predicted quasi-static
split component (the bias-like part).

All parameters are reconstructed from the same expressions
run_piggyback_star uses (ota_sync defaults + calibrated observation
noise), so the prediction is ex ante in the same sense as coast_law.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ota_sync import SDRSimulationConfig
from ota_sync.core import REAL_DTYPE
from ota_sync.sdr import _FlickerFrequencyNoise, _measurement_covariance, make_sync_preamble

SPEED_OF_LIGHT = 299792458.0
NULL_DIRECTION = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float64) / math.sqrt(2.0)


def gramian_rank_and_null(
    dt: float, steps: int = 12, with_anchor: bool = False
) -> tuple[int, torch.Tensor]:
    """Observability Gramian of (F, H) over `steps` substeps.

    H rows: one-way phase [1,0,1] every step plus frequency [0,1,0] at
    step 0 (the frame slot); with_anchor adds the anchor rows
    [1,0,0],[0,1,0],[0,0,1] at step 0. Returns (rank, unit null vector
    of the Gramian - the zero vector if full rank)."""

    F = torch.tensor(
        [[1.0, dt, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    rows = []
    power = torch.eye(3, dtype=torch.float64)
    for step in range(steps):
        h_phase = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float64)
        rows.append(h_phase @ power)
        if step == 0:
            rows.append(
                torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64) @ power
            )
            if with_anchor:
                rows.append(torch.eye(3, dtype=torch.float64) @ power)
        power = power @ F
    observability = torch.cat(rows, dim=0)
    gramian = observability.T @ observability
    eigenvalues, eigenvectors = torch.linalg.eigh(gramian)
    rank = int(torch.sum(eigenvalues > 1e-9 * eigenvalues.max()).item())
    null = (
        eigenvectors[:, 0]
        if rank < 3
        else torch.zeros(3, dtype=torch.float64)
    )
    if null[0] < 0:
        null = -null
    return rank, null


def jakes_channel_innovation(
    speed_mps: float,
    sync_interval: float,
    carrier_hz: float,
    diffuse_phase_var: float = 0.153**2,
) -> float:
    """Per-interval channel-phase innovation variance q_psi(v),
    2*sigma_c^2*(1 - J0(2 pi f_D T)); sigma_c^2 = 1/(2 K_rice) for the
    TDL-D composite (~(153 mrad)^2, matching the measured resampling
    diffuse power)."""

    if speed_mps <= 0.0:
        return 0.0
    doppler = speed_mps * carrier_hz / SPEED_OF_LIGHT
    argument = 2.0 * math.pi * doppler * sync_interval
    bessel = torch.special.bessel_j0(
        torch.tensor(argument, dtype=torch.float64)
    ).item()
    return 2.0 * diffuse_phase_var * (1.0 - bessel)


@dataclass(frozen=True)
class CyclePrediction:
    """Steady-state uncertainty over one anchor cycle."""

    believed_theta_std: float  # filter's own claim, rad
    true_theta_std: float  # mismatched-truth residual prediction, rad
    true_split_std: float  # hidden-coordinate (theta-psi)/sqrt2 std, rad
    believed_split_std: float


def _link_noise_matrices(
    settings: SDRSimulationConfig,
    n_obs: int,
    oneway_phase_var: float,
    oneway_freq_var: float,
    device: torch.device,
) -> dict:
    """The believed process/measurement pieces exactly as
    run_piggyback_star builds them (its lines 458-470), plus the anchor
    noise from the ZC preamble."""

    dt = settings.sync_interval / n_obs
    dt_samples = int(round(dt * settings.sample_rate))
    q_theta_sub = (
        2.0 * settings.phase_process_std_rad**2 / n_obs
        + settings.phase_noise_std_rad**2 * dt_samples
    )
    flicker = _FlickerFrequencyNoise(
        settings.flicker_frequency_std_hz,
        dt,
        settings.num_iterations * settings.sync_interval,
        device,
        None,
    )
    q_omega_sub = (
        2.0 * (2.0 * math.pi * settings.frequency_process_std_hz) ** 2 / n_obs
        + flicker.innovation_variance
    )
    q_psi_hat_sub = 0.01**2 / n_obs

    preamble = make_sync_preamble(settings, device)
    zc_noise = _measurement_covariance(settings, preamble, device)
    return {
        "dt": dt,
        "q_theta_sub": q_theta_sub,
        "q_omega_sub": float(q_omega_sub),
        "q_psi_hat_sub": q_psi_hat_sub,
        "r1_phase": oneway_phase_var,
        "r1_freq": oneway_freq_var,
        "r2_phase": 0.5 * float(zc_noise[0, 0].item()),
        "r2_freq": 0.5 * float(zc_noise[2, 2].item()),
    }


def split_uncertainty_cycle(
    settings: SDRSimulationConfig,
    n_obs: int,
    anchor_every_intervals: int,
    q_psi_true_interval: float,
    oneway_phase_var: float,
    oneway_freq_var: float,
    max_cycles: int = 400,
    tolerance: float = 1e-14,
) -> CyclePrediction:
    """Mismatched-Kalman steady state over one anchor cycle.

    Believed model: run_piggyback_star's (Q_hat with the fixed 0.01
    channel prior). True model: channel innovation q_psi_true added
    once per interval. Anchors beyond the run length (K*T > horizon)
    are handled by the caller passing the effective K."""

    device = torch.device("cpu")
    pieces = _link_noise_matrices(
        settings, n_obs, oneway_phase_var, oneway_freq_var, device
    )
    dt = pieces["dt"]
    F = torch.tensor(
        [[1.0, dt, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    q_hat = torch.diag(
        torch.tensor(
            [
                pieces["q_theta_sub"],
                pieces["q_omega_sub"],
                pieces["q_psi_hat_sub"],
            ],
            dtype=torch.float64,
        )
    )
    # True: same oscillator noise, channel innovation only at interval
    # boundaries (the sim redraws taps per interval).
    q_true_base = torch.diag(
        torch.tensor(
            [pieces["q_theta_sub"], pieces["q_omega_sub"], 0.0],
            dtype=torch.float64,
        )
    )
    q_true_interval = q_true_base + torch.diag(
        torch.tensor([0.0, 0.0, q_psi_true_interval], dtype=torch.float64)
    )

    h_oneway3 = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float64
    )
    r_oneway3 = torch.diag(
        torch.tensor(
            [pieces["r1_phase"], pieces["r1_freq"]], dtype=torch.float64
        )
    )
    h_oneway2 = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float64)
    r_oneway2 = torch.tensor([[pieces["r1_phase"]]], dtype=torch.float64)
    h_anchor = torch.eye(3, dtype=torch.float64)
    r_anchor = torch.diag(
        torch.tensor(
            [pieces["r2_phase"], pieces["r2_freq"], pieces["r2_phase"]],
            dtype=torch.float64,
        )
    )

    substeps_per_cycle = anchor_every_intervals * n_obs
    identity = torch.eye(3, dtype=torch.float64)
    believed = torch.diag(
        torch.tensor(
            [math.pi**2, (2.0 * math.pi * 50e3) ** 2, math.pi**2],
            dtype=torch.float64,
        )
    )
    true_cov = believed.clone()

    previous = believed.clone()
    theta_prior_believed: list[float] = []
    theta_prior_true: list[float] = []
    split_true: list[float] = []
    split_believed: list[float] = []
    v = NULL_DIRECTION
    for _ in range(max_cycles):
        theta_prior_believed.clear()
        theta_prior_true.clear()
        split_true.clear()
        split_believed.clear()
        for substep in range(substeps_per_cycle):
            at_frame = substep % n_obs == 0
            at_interval = at_frame  # taps step once per interval
            believed = F @ believed @ F.T + q_hat
            true_cov = F @ true_cov @ F.T + (
                q_true_interval if at_interval else q_true_base
            )
            theta_prior_believed.append(float(believed[0, 0]))
            theta_prior_true.append(float(true_cov[0, 0]))
            if substep == 0:
                h, r = h_anchor, r_anchor
            elif at_frame:
                h, r = h_oneway3, r_oneway3
            else:
                h, r = h_oneway2, r_oneway2
            innovation = h @ believed @ h.T + r
            gain = torch.linalg.solve(innovation, h @ believed).T
            residual_map = identity - gain @ h
            believed = (
                residual_map @ believed @ residual_map.T
                + gain @ r @ gain.T
            )
            true_cov = (
                residual_map @ true_cov @ residual_map.T
                + gain @ r @ gain.T
            )
            split_true.append(float(v @ true_cov @ v))
            split_believed.append(float(v @ believed @ v))
        if torch.max(torch.abs(believed - previous)).item() < tolerance:
            break
        previous = believed.clone()

    return CyclePrediction(
        believed_theta_std=math.sqrt(
            max(sum(theta_prior_believed) / len(theta_prior_believed), 0.0)
        ),
        true_theta_std=math.sqrt(
            max(sum(theta_prior_true) / len(theta_prior_true), 0.0)
        ),
        true_split_std=math.sqrt(max(sum(split_true) / len(split_true), 0.0)),
        believed_split_std=math.sqrt(
            max(sum(split_believed) / len(split_believed), 0.0)
        ),
    )


def los_ramp_bias_cycle(
    settings: SDRSimulationConfig,
    n_obs: int,
    anchor_every_intervals: int,
    speed_mps: float,
    oneway_phase_var: float,
    oneway_freq_var: float,
    max_cycles: int = 400,
) -> float:
    """Steady mean theta error from the DETERMINISTIC line-of-sight
    Doppler ramp (the documented 4th-state defect, now predicted).

    Under motion the specular component's phase advances ~linearly:
    mu = 2 pi f_D T per interval enters psi_true as a deterministic
    input the filter's model lacks (its psi is a zero-mean walk). The
    mean estimation error then obeys the driven closed-loop recursion

        e <- (I - L H) (F e + b),   b = [0, 0, mu] at interval steps,

    with L the believed Kalman gains (the same gains as the covariance
    cycle). The fixed-point mean |e_theta| over a cycle is the
    predicted locked bias. mu uses the full Doppler (upper bound: the
    specular angle factor cos(aoa) <= 1 is not modeled)."""

    if speed_mps <= 0.0:
        return 0.0
    device = torch.device("cpu")
    pieces = _link_noise_matrices(
        settings, n_obs, oneway_phase_var, oneway_freq_var, device
    )
    dt = pieces["dt"]
    F = torch.tensor(
        [[1.0, dt, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    q_hat = torch.diag(
        torch.tensor(
            [
                pieces["q_theta_sub"],
                pieces["q_omega_sub"],
                pieces["q_psi_hat_sub"],
            ],
            dtype=torch.float64,
        )
    )
    h_oneway3 = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float64
    )
    r_oneway3 = torch.diag(
        torch.tensor(
            [pieces["r1_phase"], pieces["r1_freq"]], dtype=torch.float64
        )
    )
    h_oneway2 = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float64)
    r_oneway2 = torch.tensor([[pieces["r1_phase"]]], dtype=torch.float64)
    h_anchor = torch.eye(3, dtype=torch.float64)
    r_anchor = torch.diag(
        torch.tensor(
            [pieces["r2_phase"], pieces["r2_freq"], pieces["r2_phase"]],
            dtype=torch.float64,
        )
    )
    doppler = speed_mps * settings.carrier_frequency_hz / SPEED_OF_LIGHT
    ramp = 2.0 * math.pi * doppler * settings.sync_interval

    identity = torch.eye(3, dtype=torch.float64)
    believed = torch.diag(
        torch.tensor(
            [math.pi**2, (2.0 * math.pi * 50e3) ** 2, math.pi**2],
            dtype=torch.float64,
        )
    )
    error = torch.zeros(3, dtype=torch.float64)
    substeps_per_cycle = anchor_every_intervals * n_obs
    theta_bias: list[float] = []
    previous_error = error.clone()
    for _ in range(max_cycles):
        theta_bias.clear()
        for substep in range(substeps_per_cycle):
            at_frame = substep % n_obs == 0
            believed = F @ believed @ F.T + q_hat
            error = F @ error
            if at_frame:
                error = error + torch.tensor(
                    [0.0, 0.0, ramp], dtype=torch.float64
                )
            if substep == 0:
                h, r = h_anchor, r_anchor
            elif at_frame:
                h, r = h_oneway3, r_oneway3
            else:
                h, r = h_oneway2, r_oneway2
            innovation = h @ believed @ h.T + r
            gain = torch.linalg.solve(innovation, h @ believed).T
            residual_map = identity - gain @ h
            believed = (
                residual_map @ believed @ residual_map.T
                + gain @ r @ gain.T
            )
            error = residual_map @ error
            theta_bias.append(abs(float(error[0])))
        if torch.max(torch.abs(error - previous_error)).item() < 1e-12:
            break
        previous_error = error.clone()
    return sum(theta_bias) / len(theta_bias)


def identifiability_condition(
    settings: SDRSimulationConfig,
    n_obs: int,
    anchor_every_intervals: int,
    speed_mps: float,
    budget_rad: float = 0.314,
) -> tuple[bool, float]:
    """The closed-form inequality (docstring part ii): the hidden-split
    growth over one anchor cycle must stay under ~ (2 b)^2 /2 per the
    theta share. Returns (identifiable?, predicted split std)."""

    device = torch.device("cpu")
    pieces = _link_noise_matrices(settings, n_obs, 1e-4, 1.0, device)
    q_psi = jakes_channel_innovation(
        speed_mps, settings.sync_interval, settings.carrier_frequency_hz
    )
    growth = anchor_every_intervals * (
        n_obs * pieces["q_theta_sub"] + q_psi
    ) / 2.0
    anchored = pieces["r2_phase"] / 2.0
    split_std = math.sqrt(anchored + growth)
    return split_std / math.sqrt(2.0) <= budget_rad, split_std
