"""Core blocks for one-way over-the-air oscillator synchronization.

The master broadcasts a known pilot. At the slave, downconversion by the local
oscillator exposes the relative phase and angular-frequency offsets. Sionna's
AWGN block models the complex baseband channel noise. A pilot phase fit and an
extended Kalman filter estimate the correction applied to the slave oscillator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
from sionna.phy import config as sionna_config
from sionna.phy.channel import AWGN


REAL_DTYPE = torch.float64
COMPLEX_DTYPE = torch.complex128


def wrap_phase(value: torch.Tensor) -> torch.Tensor:
    """Wrap angles in radians to [-pi, pi)."""

    return torch.remainder(value + math.pi, 2.0 * math.pi) - math.pi


def _unwrap_phase(value: torch.Tensor) -> torch.Tensor:
    """Unwrap a one-dimensional phase sequence without a NumPy round trip."""

    if value.ndim != 1:
        raise ValueError("phase input must be one-dimensional")
    if value.numel() < 2:
        return value.clone()

    delta = torch.diff(value)
    wrapped_delta = wrap_phase(delta)
    # Preserve a positive pi step instead of mapping it to negative pi.
    wrapped_delta = torch.where(
        (wrapped_delta == -math.pi) & (delta > 0),
        torch.full_like(wrapped_delta, math.pi),
        wrapped_delta,
    )
    correction = wrapped_delta - delta
    return torch.cat((value[:1], value[1:] + torch.cumsum(correction, dim=0)))


def _as_matrix(
    value: torch.Tensor | list[list[float]], device: torch.device
) -> torch.Tensor:
    return torch.as_tensor(value, dtype=REAL_DTYPE, device=device)


def _covariance_root(covariance: torch.Tensor) -> torch.Tensor:
    """Return a square root for a positive-semidefinite covariance matrix."""

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    if torch.any(eigenvalues < -1e-12):
        raise ValueError("covariance must be positive semidefinite")
    return eigenvectors @ torch.diag(torch.sqrt(torch.clamp(eigenvalues, min=0.0)))


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters for an OTA synchronization run."""

    sync_interval: float = 0.05
    sample_period: float = 1e-5
    pilot_length: int = 500
    num_iterations: int = 200
    snr_db: float = 20.0
    phase_process_variance: float = 1e-6
    frequency_process_variance: float = 1e-8
    master_initial_phase: float = 0.0
    master_initial_frequency: float = 0.0
    slave_initial_phase: float = 1.2
    slave_initial_frequency: float = 4.0
    ekf_process_scale: float = 2.0
    seed: int = 0
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.sync_interval <= 0.0:
            raise ValueError("sync_interval must be positive")
        if self.sample_period <= 0.0:
            raise ValueError("sample_period must be positive")
        if self.pilot_length < 2:
            raise ValueError("pilot_length must be at least two")
        if self.num_iterations < 1:
            raise ValueError("num_iterations must be positive")
        if self.phase_process_variance < 0.0:
            raise ValueError("phase_process_variance cannot be negative")
        if self.frequency_process_variance < 0.0:
            raise ValueError("frequency_process_variance cannot be negative")
        if self.ekf_process_scale < 0.0:
            raise ValueError("ekf_process_scale cannot be negative")


@dataclass(frozen=True)
class SimulationResult:
    """Recorded relative offsets before and after every correction."""

    true_phase: torch.Tensor
    estimated_phase: torch.Tensor
    true_frequency: torch.Tensor
    estimated_frequency: torch.Tensor
    phase_error: torch.Tensor
    frequency_error: torch.Tensor
    post_correction_phase: torch.Tensor
    post_correction_frequency: torch.Tensor
    covariance: torch.Tensor
    device: torch.device

    @property
    def phase_rmse(self) -> float:
        return torch.sqrt(torch.mean(self.phase_error.square())).item()

    @property
    def frequency_rmse(self) -> float:
        return torch.sqrt(torch.mean(self.frequency_error.square())).item()

    @property
    def final_phase_error(self) -> float:
        return self.post_correction_phase[-1].item()

    @property
    def final_frequency_error(self) -> float:
        return self.post_correction_frequency[-1].item()


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto`` to CUDA when available and CPU otherwise."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def centered_sample_times(
    pilot_length: int, sample_period: float, device: torch.device
) -> torch.Tensor:
    sample_index = torch.arange(pilot_length, dtype=REAL_DTYPE, device=device)
    sample_index -= (pilot_length - 1) / 2.0
    return sample_index * sample_period


def measurement_covariance(
    pilot_length: int,
    sample_period: float,
    snr_db: float,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Approximate covariance of [cos(phi), sin(phi), omega]."""

    device = torch.device(device)
    snr = 10.0 ** (snr_db / 10.0)
    sample_phase_variance = 1.0 / (2.0 * snr)
    times = centered_sample_times(pilot_length, sample_period, device)
    phase_variance = sample_phase_variance / pilot_length
    frequency_variance = sample_phase_variance / torch.dot(times, times).item()
    return torch.diag(
        torch.tensor(
            [phase_variance, phase_variance, frequency_variance],
            dtype=REAL_DTYPE,
            device=device,
        )
    )


class Oscillator:
    """Random-walk phase and angular-frequency oscillator model."""

    def __init__(
        self,
        initial_phase: float,
        initial_frequency: float,
        sync_interval: float,
        process_covariance: torch.Tensor,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.device = device
        self.state = torch.tensor(
            [initial_phase, initial_frequency], dtype=REAL_DTYPE, device=device
        )
        self.transition = torch.tensor(
            [[1.0, sync_interval], [0.0, 1.0]],
            dtype=REAL_DTYPE,
            device=device,
        )
        self.process_covariance = _as_matrix(process_covariance, device)
        if self.process_covariance.shape != (2, 2):
            raise ValueError("process_covariance must have shape [2, 2]")
        self._noise_root = _covariance_root(self.process_covariance)
        self._generator = generator

    @property
    def phase(self) -> torch.Tensor:
        return self.state[0]

    @property
    def frequency(self) -> torch.Tensor:
        return self.state[1]

    def step(self) -> torch.Tensor:
        white_noise = torch.randn(
            2,
            dtype=REAL_DTYPE,
            device=self.device,
            generator=self._generator,
        )
        process_noise = self._noise_root @ white_noise
        self.state = self.transition @ self.state + process_noise
        self.state[0] = wrap_phase(self.state[0])
        return self.state.clone()

    def apply_correction(self, correction: torch.Tensor) -> None:
        self.state = self.state + correction.to(dtype=REAL_DTYPE, device=self.device)
        self.state[0] = wrap_phase(self.state[0])


class OTAPilotLink:
    """Master-to-slave pilot link composed with Sionna's complex AWGN block."""

    def __init__(
        self,
        sample_times: torch.Tensor,
        snr_db: float,
        device: torch.device,
    ) -> None:
        self.sample_times = sample_times
        self.snr_db = snr_db
        self.device = device
        self.pilot = torch.ones(
            sample_times.numel(), dtype=COMPLEX_DTYPE, device=device
        )
        self._awgn = AWGN(precision="double", device=str(device))

    def transmit(self, master: Oscillator) -> torch.Tensor:
        """Generate the master's complex analytic pilot waveform."""

        carrier_phase = master.phase + master.frequency * self.sample_times
        return self.pilot * torch.exp(1j * carrier_phase)

    def receive(self, tx: torch.Tensor, slave: Oscillator) -> torch.Tensor:
        """Downconvert at the slave and pass the waveform through Sionna AWGN."""

        local_phase = slave.phase + slave.frequency * self.sample_times
        baseband = tx * torch.exp(-1j * local_phase)
        signal_power = torch.mean(torch.abs(baseband).square())
        noise_power = signal_power / (10.0 ** (self.snr_db / 10.0))
        return self._awgn(baseband, noise_power)

    def __call__(self, master: Oscillator, slave: Oscillator) -> torch.Tensor:
        return self.receive(self.transmit(master), slave)


class PilotReceiver:
    """Least-squares phase and frequency estimator for a centered pilot burst."""

    def __init__(self, sample_times: torch.Tensor) -> None:
        if sample_times.ndim != 1 or sample_times.numel() < 2:
            raise ValueError("at least two one-dimensional sample times are required")
        self.sample_times = sample_times
        self._time_energy = torch.dot(sample_times, sample_times)
        if self._time_energy <= 0.0:
            raise ValueError("sample times must span more than one instant")

    def estimate(self, received_pilot: torch.Tensor) -> torch.Tensor:
        if received_pilot.shape != self.sample_times.shape:
            raise ValueError("received pilot shape does not match sample times")
        phase = _unwrap_phase(torch.angle(received_pilot))
        phase_hat = torch.mean(phase)
        frequency_hat = (
            torch.dot(self.sample_times, phase - phase_hat) / self._time_energy
        )
        return torch.stack((torch.cos(phase_hat), torch.sin(phase_hat), frequency_hat))


class PhaseFrequencyEKF:
    """Iterated EKF for relative phase and angular-frequency state."""

    def __init__(
        self,
        sync_interval: float,
        process_covariance: torch.Tensor,
        measurement_covariance_matrix: torch.Tensor,
        device: torch.device,
        initial_covariance: Optional[torch.Tensor] = None,
    ) -> None:
        self.device = device
        self.state = torch.zeros(2, dtype=REAL_DTYPE, device=device)
        if initial_covariance is None:
            self.covariance = torch.eye(2, dtype=REAL_DTYPE, device=device)
        else:
            self.covariance = _as_matrix(initial_covariance, device)
            if self.covariance.shape != (2, 2):
                raise ValueError("initial_covariance must have shape [2, 2]")
        self.transition = torch.tensor(
            [[1.0, sync_interval], [0.0, 1.0]],
            dtype=REAL_DTYPE,
            device=device,
        )
        self.process_covariance = _as_matrix(process_covariance, device)
        self.measurement_covariance = _as_matrix(measurement_covariance_matrix, device)
        if self.process_covariance.shape != (2, 2):
            raise ValueError("process_covariance must have shape [2, 2]")
        if self.measurement_covariance.shape != (3, 3):
            raise ValueError("measurement covariance must have shape [3, 3]")

    def predict(self) -> None:
        self.state = self.transition @ self.state
        self.covariance = (
            self.transition @ self.covariance @ self.transition.T
            + self.process_covariance
        )

    def _measurement(self) -> torch.Tensor:
        return torch.stack(
            (torch.cos(self.state[0]), torch.sin(self.state[0]), self.state[1])
        )

    def _jacobian(self) -> torch.Tensor:
        phi = self.state[0]
        zero = torch.zeros((), dtype=REAL_DTYPE, device=self.device)
        one = torch.ones((), dtype=REAL_DTYPE, device=self.device)
        return torch.stack(
            (
                torch.stack((-torch.sin(phi), zero)),
                torch.stack((torch.cos(phi), zero)),
                torch.stack((zero, one)),
            )
        )

    def update(self, measurement: torch.Tensor) -> None:
        measurement = measurement.to(dtype=REAL_DTYPE, device=self.device)
        if measurement.shape != (3,):
            raise ValueError("measurement must have shape [3]")

        prior_state = self.state.clone()
        prior_covariance = self.covariance.clone()
        posterior_state = prior_state
        # Re-linearization avoids the large-offset under-correction of a
        # one-shot EKF update (e.g., sin(-1.4) is not approximately -1.4).
        for _ in range(6):
            self.state = posterior_state
            jacobian = self._jacobian()
            innovation_covariance = (
                jacobian @ prior_covariance @ jacobian.T + self.measurement_covariance
            )
            kalman_gain = torch.linalg.solve(
                innovation_covariance,
                jacobian @ prior_covariance,
            ).T
            iterated_innovation = (
                measurement
                - self._measurement()
                - jacobian @ (prior_state - posterior_state)
            )
            next_state = prior_state + kalman_gain @ iterated_innovation
            if torch.max(torch.abs(next_state - posterior_state)) < 1e-10:
                posterior_state = next_state
                break
            posterior_state = next_state
        self.state = posterior_state

        identity = torch.eye(2, dtype=REAL_DTYPE, device=self.device)
        residual_map = identity - kalman_gain @ jacobian
        # Joseph form keeps the covariance symmetric and positive semidefinite.
        self.covariance = (
            residual_map @ prior_covariance @ residual_map.T
            + kalman_gain @ self.measurement_covariance @ kalman_gain.T
        )

    def reset_after_correction(self, correction: torch.Tensor) -> None:
        self.state = self.state - correction


def run_simulation(settings: SimulationConfig = SimulationConfig()) -> SimulationResult:
    """Run the sequential master/slave synchronization experiment."""

    device = resolve_device(settings.device)
    torch.manual_seed(settings.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(settings.seed)
    sionna_config.seed = settings.seed

    generator = torch.Generator(device=device)
    generator.manual_seed(settings.seed)

    process_covariance = torch.diag(
        torch.tensor(
            [
                settings.phase_process_variance,
                settings.frequency_process_variance,
            ],
            dtype=REAL_DTYPE,
            device=device,
        )
    )
    times = centered_sample_times(settings.pilot_length, settings.sample_period, device)
    measurement_noise = measurement_covariance(
        settings.pilot_length,
        settings.sample_period,
        settings.snr_db,
        device,
    )

    master = Oscillator(
        settings.master_initial_phase,
        settings.master_initial_frequency,
        settings.sync_interval,
        process_covariance,
        device,
        generator,
    )
    slave = Oscillator(
        settings.slave_initial_phase,
        settings.slave_initial_frequency,
        settings.sync_interval,
        process_covariance,
        device,
        generator,
    )
    link = OTAPilotLink(times, settings.snr_db, device)
    receiver = PilotReceiver(times)
    ekf = PhaseFrequencyEKF(
        settings.sync_interval,
        settings.ekf_process_scale * process_covariance,
        measurement_noise,
        device,
    )

    history: dict[str, list[torch.Tensor]] = {
        "true_phase": [],
        "estimated_phase": [],
        "true_frequency": [],
        "estimated_frequency": [],
        "phase_error": [],
        "frequency_error": [],
        "post_correction_phase": [],
        "post_correction_frequency": [],
        "covariance": [],
    }

    for _ in range(settings.num_iterations):
        master.step()
        slave.step()

        relative_state = master.state - slave.state
        received_pilot = link(master, slave)
        measurement = receiver.estimate(received_pilot)
        ekf.predict()
        ekf.update(measurement)

        true_phase = wrap_phase(relative_state[0])
        estimated_phase = wrap_phase(ekf.state[0])
        history["true_phase"].append(true_phase.clone())
        history["estimated_phase"].append(estimated_phase.clone())
        history["true_frequency"].append(relative_state[1].clone())
        history["estimated_frequency"].append(ekf.state[1].clone())
        history["phase_error"].append(wrap_phase(true_phase - estimated_phase))
        history["frequency_error"].append((relative_state[1] - ekf.state[1]).clone())
        history["covariance"].append(ekf.covariance.clone())

        correction = ekf.state.clone()
        slave.apply_correction(correction)
        ekf.reset_after_correction(correction)
        residual_state = master.state - slave.state
        history["post_correction_phase"].append(wrap_phase(residual_state[0]).clone())
        history["post_correction_frequency"].append(residual_state[1].clone())

    return SimulationResult(
        **{
            name: torch.stack(values).detach().cpu() for name, values in history.items()
        },
        device=device,
    )
