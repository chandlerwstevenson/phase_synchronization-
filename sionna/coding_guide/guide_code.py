"""Code-along companion for the phase synchronization coding guide.

This is the complete program the guide builds block by block: two
stations with drifting oscillators synchronize by two-way pilot
exchanges over a realistic Sionna channel (3GPP tapped-delay-line
model "D" + thermal noise), with a real acquisition chain (packet
timing, coarse and fine frequency, matched-filter phase) and an
extended Kalman filter driving closed-loop corrections.

It is the simplified twin of ota_sync/coherent.py.

Run it:  python3 guide_code.py     (needs torch + sionna >= 2.0)
"""

import math

import numpy as np
import torch
from sionna.phy.channel import AWGN, ApplyTimeChannel
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.channel.utils import (
    cir_to_time_channel,
    time_lag_discrete_time_channel,
)

rng = np.random.default_rng(0)


# ---- Block 1: phase arithmetic ------------------------------------
def wrap(x):
    """Map any angle into (-pi, pi]."""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


# ---- Block 2: the oscillator --------------------------------------
DT = 0.05                          # seconds between sync exchanges
PHASE_WALK_STD = 0.002             # radians of phase wander per interval
FREQ_WALK_STD = 2 * np.pi * 0.1    # rad/s of frequency wander per interval


class Oscillator:
    """A clock that drifts: phase random walk + frequency random walk."""

    def __init__(self, phase, freq_hz):
        self.theta = phase                  # phase (radians)
        self.omega = 2 * np.pi * freq_hz    # frequency offset (rad/s)

    def step(self):
        self.theta = wrap(self.theta + self.omega * DT
                          + rng.normal(0.0, PHASE_WALK_STD))
        self.omega += rng.normal(0.0, FREQ_WALK_STD)

    def correct(self, dtheta, domega):
        self.theta = wrap(self.theta + dtheta)
        self.omega += domega


# ---- Block 3: the pilot waveform and its receiver -----------------
FS = 1e6                # sample rate (1 MHz -> 1 sample per microsecond)
TS = 1.0 / FS
SHORT_LEN = 16          # short training sequence (repeated for coarse sync)
SHORT_REPS = 16
# The long field sets measurement quality, and its length is a LOOP
# design choice, not a detail: the frequency-estimate error times the
# 50 ms interval must stay well inside the pi/2 branch boundary, or
# the loop falls into branch-flip chaos. 2047x2 (the repo's choice)
# gives ~0.1 Hz frequency error -> ~0.04 rad of drift per interval.
LONG_LEN = 2047
LONG_REPS = 2


def zadoff_chu(length, root):
    """Constant-amplitude chirp with perfect autocorrelation (as in LTE/5G)."""
    n = np.arange(length)
    if length % 2:
        phase = -np.pi * root * n * (n + 1.0) / length
    else:
        phase = -np.pi * root * n**2 / length
    return np.exp(1j * phase)


SHORT = zadoff_chu(SHORT_LEN, 1)
LONG = zadoff_chu(LONG_LEN, 25)
PILOT = np.concatenate([np.tile(SHORT, SHORT_REPS), np.tile(LONG, LONG_REPS)])
PILOT_LEN = PILOT.size
SHORT_FIELD = SHORT_LEN * SHORT_REPS


def estimate(rx, start=None):
    """Timing, frequency, and phase from one received pilot capture.

    Three stages, a miniature of the repo's SDRSynchronizer:
      1. coarse: the short field repeats every 16 samples, so
         rx[n]*conj(rx[n-16]) has angle 2*pi*f*16*Ts -- frequency,
         unambiguous up to +-fs/32 = +-31 kHz; the correlation peak
         gives coarse timing.
      2. fine frequency: the two long fields are identical and 512
         samples apart; the angle between them refines f.
      3. phase: matched-filter both long fields after removing the
         frequency ramp; the correlation angle is the carrier phase.

    start: reuse a known timing instead of re-detecting. The two
    directions of a reciprocal exchange share the same propagation
    delay, so the reverse capture reuses the forward capture's
    timing -- otherwise noise can land the two detections one sample
    apart and the channel's fractional-delay phase no longer cancels
    in the half-difference.
    """
    lag = SHORT_LEN
    width = SHORT_FIELD - lag
    prod = np.conj(rx[:-lag]) * rx[lag:]
    csum = np.cumsum(np.concatenate([[0.0 + 0j], prod]))
    corr = csum[width:] - csum[:-width]
    if start is None:
        power = np.cumsum(np.concatenate([[0.0], np.abs(rx) ** 2]))
        energy = power[width:] - power[:-width]
        metric = np.abs(corr[: rx.size - PILOT_LEN + 1]) / np.maximum(
            energy[: rx.size - PILOT_LEN + 1], 1e-12
        )
        start = int(np.argmax(metric))
    freq = np.angle(corr[start]) / (lag * TS)          # rad/s, coarse

    seg = rx[start : start + PILOT_LEN]
    n = np.arange(PILOT_LEN)
    seg = seg * np.exp(-1j * freq * n * TS)            # remove coarse ramp

    first = seg[SHORT_FIELD : SHORT_FIELD + LONG_LEN]
    second = seg[SHORT_FIELD + LONG_LEN : SHORT_FIELD + 2 * LONG_LEN]
    fine = np.angle(np.sum(np.conj(first) * second)) / (LONG_LEN * TS)
    freq += fine

    seg = rx[start : start + PILOT_LEN] * np.exp(-1j * freq * n * TS)
    long_part = seg[SHORT_FIELD:]
    phase = np.angle(np.sum(np.conj(np.tile(LONG, LONG_REPS)) * long_part))
    return phase, freq, start


# ---- Block 4: the Sionna channel ----------------------------------
SNR_DB = 20.0
L_MIN, L_MAX = time_lag_discrete_time_channel(FS, 3e-6)
L_TOT = L_MAX - L_MIN + 1


class RadioChannel:
    """One propagation direction: 3GPP TDL multipath + thermal noise.

    reciprocal_of: the reverse link of a two-way pair shares the
    forward link's taps -- radio propagation is reciprocal.
    """

    def __init__(self, seed, reciprocal_of=None):
        if reciprocal_of is not None:
            self.taps = reciprocal_of.taps
        else:
            torch.manual_seed(seed)
            tdl = TDL(model="D", delay_spread=100e-9,
                      carrier_frequency=915e6, min_speed=0.0,
                      max_speed=0.0, precision="double", device="cpu")
            coeff, delays = tdl(batch_size=1, num_time_steps=1,
                                sampling_frequency=1.0 / DT)
            self.taps = cir_to_time_channel(
                FS, coeff, delays, L_MIN, L_MAX, normalize=True
            )
        self._apply = ApplyTimeChannel(
            num_time_samples=PILOT_LEN, l_tot=L_TOT,
            precision="double", device="cpu",
        )
        self._awgn = AWGN(precision="double", device="cpu")

    def transmit(self, tx_osc, rx_osc):
        """Send the pilot from tx_osc's radio to rx_osc's radio."""
        t = np.arange(PILOT_LEN) * TS
        waveform = PILOT * np.exp(1j * (tx_osc.theta + tx_osc.omega * t))

        x = torch.tensor(waveform, dtype=torch.complex128)
        x = x.reshape(1, 1, 1, PILOT_LEN)
        h = self.taps[..., 0, :].unsqueeze(-2).expand(
            1, 1, 1, 1, 1, PILOT_LEN + L_TOT - 1, L_TOT
        )
        y = self._apply(x, h).reshape(-1).numpy()

        t_rx = np.arange(y.size) * TS
        baseband = y * np.exp(-1j * (rx_osc.theta + rx_osc.omega * t_rx))

        power = np.mean(np.abs(baseband) ** 2)
        noise_power = power / (10.0 ** (SNR_DB / 10.0))
        noisy = self._awgn(
            torch.tensor(baseband, dtype=torch.complex128),
            torch.tensor(noise_power),
        )
        return noisy.numpy()


# ---- Block 5: the two-way exchange --------------------------------
def two_way_exchange(master, slave, link_fwd, link_rev):
    """One reciprocal exchange. Returns (half_phase, half_freq).

    The forward capture measures  theta_m - theta_s + angle(channel);
    the reverse capture measures  theta_s - theta_m + angle(channel).
    Half the difference cancels the channel phase exactly -- but the
    divide-by-two leaves the result known only modulo pi.
    """
    fwd_phase, fwd_freq, timing = estimate(link_fwd.transmit(master, slave))
    rev_phase, rev_freq, _ = estimate(
        link_rev.transmit(slave, master), start=timing
    )
    half_phase = wrap(wrap(fwd_phase - rev_phase) / 2.0)
    half_freq = (fwd_freq - rev_freq) / 2.0
    return half_phase, half_freq


def pick_branch(half_phase, reference):
    """True offset is half_phase OR half_phase + pi; pick the
    candidate closer to the reference (the filter's prediction)."""
    alt = wrap(half_phase + np.pi)
    if abs(wrap(half_phase - reference)) <= abs(wrap(alt - reference)):
        return half_phase
    return alt


# ---- Block 6: the extended Kalman filter --------------------------
F = np.array([[1.0, DT],
              [0.0, 1.0]])
# Both clocks wander and we track their difference: variances doubled.
Q = np.diag([2 * PHASE_WALK_STD**2, 2 * FREQ_WALK_STD**2])
# Measurement noise from the pilot's signal-to-noise ratio (the same
# reasoning as _measurement_covariance in ota_sync/sdr.py), halved by
# the two-direction average.
snr = 10.0 ** (SNR_DB / 10.0)
PHASE_VAR = 1.0 / (2.0 * snr * LONG_LEN * LONG_REPS)
SEP = LONG_LEN * TS
FREQ_VAR = 1.0 / (snr * LONG_LEN * SEP**2)
R = np.diag([0.5 * PHASE_VAR, 0.5 * PHASE_VAR, 0.5 * FREQ_VAR])


class PhaseFrequencyEKF:
    """Tracks x = [relative phase, relative frequency].

    The measurement enters as [cos(phase), sin(phase), freq] so the
    filter never subtracts two wrapped angles -- that nonlinearity is
    what makes it an EXTENDED Kalman filter.
    """

    def __init__(self):
        self.x = np.zeros(2)
        self.P = np.diag([np.pi**2, (2 * np.pi * 5000.0) ** 2])

    def predict(self):
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def h(self, x):
        return np.array([np.cos(x[0]), np.sin(x[0]), x[1]])

    def H(self, x):
        return np.array([[-np.sin(x[0]), 0.0],
                         [np.cos(x[0]), 0.0],
                         [0.0, 1.0]])

    def update(self, phase_meas, freq_meas):
        z = np.array([np.cos(phase_meas), np.sin(phase_meas), freq_meas])
        x0, P0 = self.x.copy(), self.P.copy()
        x = x0.copy()
        # Iterated update: re-linearize around each new estimate; a
        # single shot under-corrects for large offsets.
        for _ in range(6):
            H = self.H(x)
            S = H @ P0 @ H.T + R
            K = P0 @ H.T @ np.linalg.inv(S)
            x_new = x0 + K @ (z - self.h(x) - H @ (x0 - x))
            if np.max(np.abs(x_new - x)) < 1e-10:
                x = x_new
                break
            x = x_new
        self.x = x
        # Joseph form keeps P symmetric and positive.
        I_KH = np.eye(2) - K @ self.H(x)
        self.P = I_KH @ P0 @ I_KH.T + K @ R @ K.T


# ---- Block 7: the closed loop -------------------------------------
def run(num_steps=120, seed=0, verbose=True):
    global rng
    rng = np.random.default_rng(seed)

    master = Oscillator(rng.uniform(-np.pi, np.pi), freq_hz=0.0)
    slave = Oscillator(rng.uniform(-np.pi, np.pi), freq_hz=1500.0)
    link_fwd = RadioChannel(seed=seed)
    link_rev = RadioChannel(seed=seed, reciprocal_of=link_fwd)
    ekf = PhaseFrequencyEKF()

    acquired = False
    pi_calibrated = False
    residuals, gains = [], []

    for k in range(num_steps):
        master.step()
        slave.step()

        half_phase, half_freq = two_way_exchange(
            master, slave, link_fwd, link_rev
        )

        ekf.predict()
        if not acquired:
            # First contact: no prediction yet. Guess the candidate
            # nearer zero; the one-time power check below repairs a
            # wrong guess.
            ekf.x = np.array([pick_branch(half_phase, 0.0), half_freq])
            acquired = True
        else:
            phase_meas = pick_branch(half_phase, wrap(ekf.x[0]))
            # Innovation gate: the coarse frequency stage very rarely
            # errs past the fine stage's +-244 Hz ambiguity, and the
            # wrapped estimate then arrives hundreds of Hz off. A
            # measurement that disagrees that violently with the
            # prediction is a broken capture, not information: skip
            # the update and coast on the model (the real code gets
            # the same protection from its detection gating).
            freq_innovation = abs(half_freq - ekf.x[1])
            phase_innovation = abs(wrap(phase_meas - ekf.x[0]))
            if freq_innovation < 2 * np.pi * 50.0 and phase_innovation < 1.0:
                ekf.update(phase_meas, half_freq)

        # Command the slave to cancel the estimated offset, then
        # subtract the command from the filter state: the filter
        # tracks what REMAINS.
        correction = ekf.x.copy()
        slave.correct(correction[0], correction[1])
        ekf.x = ekf.x - correction

        # One-time branch calibration: one coarse combined-power
        # check after the loop settles. Destructive combining means
        # the acquisition guess was the wrong candidate: flip by pi.
        # The half-difference cannot see a pi shift, so the filter
        # needs no reset.
        if not pi_calibrated and k >= 10:
            if np.cos(master.theta - slave.theta) < 0.0:
                slave.correct(np.pi, 0.0)
            pi_calibrated = True

        residual = wrap(master.theta - slave.theta)
        residuals.append(residual)
        # Two stations toward one target:
        # gain = |1 + e^{j residual}|^2 / 4 = cos^2(residual/2).
        gains.append(np.cos(residual / 2.0) ** 2)

    residuals = np.array(residuals)
    gains = np.array(gains)
    steady = slice(20, None)
    if verbose:
        print(f"steady residual rms : "
              f"{np.sqrt(np.mean(residuals[steady] ** 2)):.4f} rad")
        print(f"steady coherent gain: {np.mean(gains[steady]) * 100:.1f} %")
    return residuals, gains


if __name__ == "__main__":
    for seed in range(3):
        print(f"--- seed {seed} ---")
        run(seed=seed)
