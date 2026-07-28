# oscillator.py

import numpy as np


class Oscillator:
    """
    Oscillator state model from Eq. (1) of the Globecom 2012 paper.

    State:
        x = [phi, omega]^T

    where
        phi   = oscillator phase offset (rad)
        omega = oscillator frequency offset (rad/s)

    State evolution:
        x(k+1) = F x(k) + w(k)

    where
        F = [[1, Tm],
             [0, 1]]

        w ~ N(0, Q)
    """

    def __init__(self,
                 phi0=0.0,
                 omega0=0.0,
                 Tm=0.05,
                 Q=None):

        self.Tm = Tm

        # State vector
        self.x = np.array(
            [phi0, omega0],
            dtype=float
        )

        # State transition matrix
        self.F = np.array([
            [1.0, Tm],
            [0.0, 1.0]
        ])

        # Process noise covariance
        if Q is None:
            self.Q = np.diag([
                1e-6,
                1e-8
            ])
        else:
            self.Q = np.array(Q, dtype=float)

    def step(self):
        """
        Advance oscillator one synchronization interval.
        """

        w = np.random.multivariate_normal(
            mean=np.zeros(2),
            cov=self.Q
        )

        self.x = self.F @ self.x + w

        return self.x.copy()

    @property
    def phase(self):
        return self.x[0]

    @property
    def frequency(self):
        return self.x[1]

    def state(self):
        return self.x.copy()

    def set_state(self, x):
        self.x = np.array(x, dtype=float)