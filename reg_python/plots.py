# plots.py

import matplotlib.pyplot as plt
import numpy as np


def plot_phase(true_phase, estimated_phase):

    plt.figure(figsize=(8,4))

    plt.plot(true_phase, linewidth=2, label="True Phase")
    plt.plot(
        estimated_phase,
        "--",
        linewidth=2,
        label="Estimated Phase"
    )

    plt.xlabel("Iteration")
    plt.ylabel("Phase (rad)")
    plt.title("Phase Tracking")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()


def plot_frequency(true_frequency, estimated_frequency):

    plt.figure(figsize=(8,4))

    plt.plot(true_frequency, linewidth=2, label="True Frequency")
    plt.plot(
        estimated_frequency,
        "--",
        linewidth=2,
        label="Estimated Frequency"
    )

    plt.xlabel("Iteration")
    plt.ylabel("Frequency Offset (rad/s)")
    plt.title("Frequency Tracking")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()


def plot_phase_error(phase_error):

    plt.figure(figsize=(8,4))

    plt.plot(phase_error)

    plt.xlabel("Iteration")
    plt.ylabel("Phase Error (rad)")
    plt.title("Phase Estimation Error")
    plt.grid(True)

    plt.tight_layout()


def plot_frequency_error(frequency_error):

    plt.figure(figsize=(8,4))

    plt.plot(frequency_error)

    plt.xlabel("Iteration")
    plt.ylabel("Frequency Error (rad/s)")
    plt.title("Frequency Estimation Error")
    plt.grid(True)

    plt.tight_layout()


def plot_mse(phase_error, frequency_error):

    phase_mse = np.square(phase_error)
    frequency_mse = np.square(frequency_error)

    plt.figure(figsize=(8,4))

    plt.plot(phase_mse, label="Phase MSE")
    plt.plot(frequency_mse, label="Frequency MSE")

    plt.xlabel("Iteration")
    plt.ylabel("Squared Error")
    plt.title("Estimation Error")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()


def plot_covariance(P_history):

    P_history = np.asarray(P_history)

    plt.figure(figsize=(8,4))

    plt.plot(P_history[:,0,0], label="P11 (Phase)")
    plt.plot(P_history[:,1,1], label="P22 (Frequency)")

    plt.xlabel("Iteration")
    plt.ylabel("Variance")
    plt.title("EKF Covariance")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()


def show():

    plt.show()