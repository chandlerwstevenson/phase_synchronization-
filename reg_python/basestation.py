# basestation.py

import numpy as np

from oscillator import Oscillator
from receiver import Receiver
from ekf import EKF


class BaseStation:
    """
    Base station (master or slave).

    Master:
        - Has an oscillator
        - Transmits pilot bursts

    Slave:
        - Has an oscillator
        - Receives pilot bursts
        - Runs EKF
        - Corrects its oscillator
    """

    def __init__(
        self,
        is_master,
        Tm,
        Ts,
        pilot_length,
        Q=None,
        R=None,
        phi0=0.0,
        omega0=0.0,
        ekf_Q=None,
    ):

        self.is_master = is_master

        self.Tm = Tm
        self.Ts = Ts
        self.pilot_length = pilot_length

        ####################################################
        # Hardware
        ####################################################

        self.oscillator = Oscillator(
            phi0=phi0,
            omega0=omega0,
            Tm=Tm,
            Q=Q
        )

        ####################################################
        # Only slave nodes have a receiver + EKF
        ####################################################

        if not is_master:

            self.receiver = Receiver(Ts)

            self.ekf = EKF(
                Tm=Tm,
                Q=Q if ekf_Q is None else ekf_Q,
                R=R
            )

        else:

            self.receiver = None
            self.ekf = None

    ####################################################
    # Time evolution
    ####################################################

    def step(self):

        self.oscillator.step()

    ####################################################
    # Pilot transmission
    ####################################################

    def transmit_pilot(self):
        """
        Generate a complex pilot burst.
        """

        N = self.pilot_length

        sample_index = np.arange(N) - (N - 1) / 2
        t = sample_index * self.Ts

        phi = self.oscillator.x[0]
        omega = self.oscillator.x[1]

        pilot = np.exp(
            1j * (phi + omega * t)
        )

        return pilot

    ####################################################
    # Pilot reception
    ####################################################

    def receive_pilot(self, received_signal):

        if self.is_master:
            raise RuntimeError(
                "Master node does not receive pilots."
            )

        z = self.receiver.measure(received_signal)

        return z

    ####################################################
    # EKF
    ####################################################

    def synchronize(self, measurement):

        if self.is_master:
            return

        self.ekf.predict()

        self.ekf.update(measurement)

    ####################################################
    # Oscillator correction
    ####################################################

    def apply_correction(self):

        if self.is_master:
            return

        correction = self.ekf.state()

        self.oscillator.set_state(
            self.oscillator.state() + correction
        )

        self.ekf.x = self.ekf.x - correction

    ####################################################
    # Convenience
    ####################################################

    @property
    def phase(self):
        return self.oscillator.x[0]

    @property
    def frequency(self):
        return self.oscillator.x[1]
