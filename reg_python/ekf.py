# ekf.py

import numpy as np


class EKF:
    """
    Extended Kalman Filter for estimating

        x = [phase, frequency]^T

    from the nonlinear measurements

        z = [cos(phi), sin(phi), omega]^T
    """

    def __init__(self,
                 Tm,
                 Q=None,
                 R=None,
                 x0=None,
                 P0=None):

        self.Tm = Tm

        # State transition matrix (Eq. 1)
        self.F = np.array([
            [1.0, Tm],
            [0.0, 1.0]
        ])

        # Initial state
        if x0 is None:
            self.x = np.zeros(2)
        else:
            self.x = np.array(x0, dtype=float)

        # Initial covariance
        if P0 is None:
            self.P = np.eye(2)
        else:
            self.P = np.array(P0, dtype=float)

        # Process covariance
        if Q is None:
            self.Q = np.eye(2) * 1e-6
        else:
            self.Q = Q

        # Measurement covariance
        if R is None:
            self.R = np.eye(3) * 1e-3
        else:
            self.R = R


    ############################################################
    # Measurement function h(x)   (Eq. 2)
    ############################################################

    def h(self, x):

        phi = x[0]
        omega = x[1]

        return np.array([
            np.cos(phi),
            np.sin(phi),
            omega
        ])


    ############################################################
    # Jacobian H(x)
    ############################################################

    def jacobian(self, x):

        phi = x[0]

        return np.array([
            [-np.sin(phi), 0.0],
            [ np.cos(phi), 0.0],
            [          0., 1.0]
        ])


    ############################################################
    # Prediction step (Eq. 8-9)
    ############################################################

    def predict(self):

        self.x = self.F @ self.x

        self.P = (
            self.F
            @ self.P
            @ self.F.T
            + self.Q
        )


    ############################################################
    # Update step (Eq. 3-7)
    ############################################################

    def update(self, z):

        # Jacobian
        H = self.jacobian(self.x)

        # Predicted measurement
        z_pred = self.h(self.x)

        # Innovation (Eq. 3)
        y = z - z_pred

        # Innovation covariance (Eq. 4)
        S = (
            H
            @ self.P
            @ H.T
            + self.R
        )

        # Kalman gain (Eq. 5)
        K = (
            self.P
            @ H.T
            @ np.linalg.inv(S)
        )

        # State update (Eq. 6)
        self.x = self.x + K @ y

        # Covariance update (Eq. 7)
        I = np.eye(2)

        self.P = (
            I - K @ H
        ) @ self.P


    ############################################################
    # Convenience functions
    ############################################################

    @property
    def phase(self):
        return self.x[0]

    @property
    def frequency(self):
        return self.x[1]

    def state(self):
        return self.x.copy()

    def covariance(self):
        return self.P.copy()