import numpy as np  

class Receiver:  

    def __init__(self, Ts):  
        # Ts : Sampling period 
        self.Ts = Ts  

    def _fit_phase(self, r):
        if len(r) < 2:
            raise ValueError("At least two pilot samples are required.")

        phase = np.unwrap(np.angle(r))
        sample_index = np.arange(len(r)) - (len(r) - 1) / 2
        time = sample_index * self.Ts

        phase_hat = np.mean(phase)
        omega_hat = np.dot(time, phase - phase_hat) / np.dot(time, time)

        return phase_hat, omega_hat

    def estimate_frequency(self, r): 
        _, omega_hat = self._fit_phase(r)
        return omega_hat  
    
    def estimate_phase(self, r): 
        phase_hat, _ = self._fit_phase(r)

        cos_hat = np.cos(phase_hat)
        sin_hat = np.sin(phase_hat)

        return cos_hat, sin_hat 
    
    def measure(self, r): 
        phase_hat, omega_hat = self._fit_phase(r)

        return np.array([
            np.cos(phase_hat),
            np.sin(phase_hat),
            omega_hat
        ])
