# simulation.py

import numpy as np
from basestation import BaseStation
from plots import *

# --------------------------------------------------------
# Parameters
# --------------------------------------------------------

np.random.seed(0)

Tm = 0.05
Ts = 1e-5
pilot_length = 500
num_iterations = 200
snr_db = 20

Q = np.diag([1e-6, 1e-8])

snr = 10 ** (snr_db / 10)
sample_phase_variance = 1 / (2 * snr)
sample_index = np.arange(pilot_length) - (pilot_length - 1) / 2
pilot_time = sample_index * Ts

phase_measurement_variance = sample_phase_variance / pilot_length
frequency_measurement_variance = (
    sample_phase_variance / np.dot(pilot_time, pilot_time)
)

R = np.diag([
    phase_measurement_variance,
    phase_measurement_variance,
    frequency_measurement_variance
])

# --------------------------------------------------------
# Create nodes
# --------------------------------------------------------

master = BaseStation(
    is_master=True,
    Tm=Tm,
    Ts=Ts,
    pilot_length=pilot_length,
    Q=Q,
    phi0=0.0,
    omega0=0.0
)

slave = BaseStation(
    is_master=False,
    Tm=Tm,
    Ts=Ts,
    pilot_length=pilot_length,
    Q=Q,
    R=R,
    phi0=1.2,
    omega0=4.0,
    ekf_Q=2 * Q
)

# --------------------------------------------------------
# Channel
# --------------------------------------------------------

def channel(signal, snr_db):

    power = np.mean(np.abs(signal) ** 2)

    snr = 10 ** (snr_db / 10)

    noise_power = power / snr

    sigma = np.sqrt(noise_power / 2)

    noise = sigma * (
        np.random.randn(len(signal))
        + 1j * np.random.randn(len(signal))
    )

    return signal + noise

# --------------------------------------------------------
# Storage
# --------------------------------------------------------

true_phase = []
estimated_phase = []

true_frequency = []
estimated_frequency = []

phase_error = []
frequency_error = []

P_history = []

# --------------------------------------------------------
# Main loop
# --------------------------------------------------------

for _ in range(num_iterations):

    master.step()
    slave.step()

    # Relative phase/frequency offset
    phi_rel = master.phase - slave.phase
    omega_rel = master.frequency - slave.frequency

    t = pilot_time

    tx = np.exp(1j * (phi_rel + omega_rel * t))

    rx = channel(tx, snr_db)

    z = slave.receive_pilot(rx)

    slave.synchronize(z)

    # Log BEFORE correction
    true_phase.append(np.angle(np.exp(1j * phi_rel)))
    estimated_phase.append(
        np.angle(np.exp(1j * slave.ekf.phase))
    )

    true_frequency.append(omega_rel)
    estimated_frequency.append(slave.ekf.frequency)

    phase_error.append(
        np.angle(np.exp(1j * (phi_rel - slave.ekf.phase)))
    )
    frequency_error.append(omega_rel - slave.ekf.frequency)

    P_history.append(slave.ekf.covariance())

    slave.apply_correction()

# --------------------------------------------------------
# Plot
# --------------------------------------------------------

plot_phase(true_phase, estimated_phase)
plot_frequency(true_frequency, estimated_frequency)
plot_phase_error(phase_error)
plot_frequency_error(frequency_error)
plot_mse(phase_error, frequency_error)
plot_covariance(P_history)

show()
