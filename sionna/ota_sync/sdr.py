"""SDR-like OTA synchronization built from Sionna PHY channel blocks.

This model operates on sampled complex IQ rather than applying noise directly to
an oscillator tone. It uses a repeated short training field for packet detection
and coarse CFO, a cyclic-prefix-protected long training field for fine CFO and
phase, and a 3GPP TR 38.901 tapped-delay-line channel from Sionna.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from sionna.phy import config as sionna_config
from sionna.phy.channel import AWGN, ApplyTimeChannel
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.channel.utils import (
    cir_to_time_channel,
    time_lag_discrete_time_channel,
)

from .core import (
    COMPLEX_DTYPE,
    REAL_DTYPE,
    Oscillator,
    PhaseFrequencyEKF,
    resolve_device,
    wrap_phase,
)


@dataclass(frozen=True)
class SDRSimulationConfig:
    """Configuration of the sampled-IQ SDR synchronization experiment."""

    sync_interval: float = 0.05
    sample_rate: float = 1e6
    num_iterations: int = 100
    snr_db: float = 20.0
    carrier_frequency_hz: float = 915e6

    tdl_model: str = "D"
    delay_spread_s: float = 100e-9
    maximum_channel_delay_s: float = 3e-6
    channel_speed_mps: float = 0.0

    short_sequence_length: int = 16
    short_repetitions: int = 16
    long_sequence_length: int = 2047
    long_cp_length: int = 128
    long_repetitions: int = 2
    capture_guard_samples: int = 64
    timing_jitter_samples: int = 32

    master_initial_phase: float = 0.0
    master_initial_frequency_hz: float = 0.0
    slave_initial_phase: float = 1.2
    slave_initial_frequency_hz: float = 1500.0
    phase_process_std_rad: float = 0.002
    frequency_process_std_hz: float = 0.1

    sample_clock_offset_ppm: float = 10.0
    phase_noise_std_rad: float = 2e-4
    iq_gain_imbalance_db: float = 0.2
    iq_phase_imbalance_deg: float = 1.0
    dc_offset: complex = 0.005 + 0.003j
    adc_bits: int = 12
    dac_bits: int = 12
    tx_amplitude: float = 0.5
    pa_clip_level: float = 0.9
    agc_target_rms: float = 0.25
    detection_threshold: float = 0.25

    phase_correction_bits: int = 16
    frequency_correction_resolution_hz: float = 0.01
    seed: int = 0
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.sync_interval <= 0.0 or self.sample_rate <= 0.0:
            raise ValueError("sync_interval and sample_rate must be positive")
        if self.num_iterations < 1:
            raise ValueError("num_iterations must be positive")
        if self.carrier_frequency_hz <= 0.0:
            raise ValueError("carrier_frequency_hz must be positive")
        if self.tdl_model not in {
            "A",
            "B",
            "C",
            "D",
            "E",
            "A30",
            "B100",
            "C300",
        }:
            raise ValueError("invalid Sionna TDL model")
        if self.delay_spread_s <= 0.0 or self.maximum_channel_delay_s <= 0.0:
            raise ValueError("channel delays must be positive")
        if self.channel_speed_mps < 0.0:
            raise ValueError("channel_speed_mps cannot be negative")
        if self.short_sequence_length < 2 or self.short_repetitions < 3:
            raise ValueError("the short training field needs at least three repeats")
        if self.long_sequence_length < 3 or self.long_repetitions < 2:
            raise ValueError("the long training field needs at least two repeats")
        if not 0 <= self.long_cp_length < self.long_sequence_length:
            raise ValueError("long_cp_length must be shorter than the long sequence")
        if self.capture_guard_samples < 1 or self.timing_jitter_samples < 0:
            raise ValueError("capture guard must be positive and jitter nonnegative")
        if self.phase_process_std_rad < 0.0 or self.frequency_process_std_hz < 0.0:
            raise ValueError("oscillator process deviations cannot be negative")
        if self.phase_noise_std_rad < 0.0:
            raise ValueError("phase_noise_std_rad cannot be negative")
        if self.adc_bits < 2 or self.dac_bits < 2:
            raise ValueError("ADC and DAC resolution must be at least two bits")
        if not 0.0 < self.tx_amplitude <= 1.0:
            raise ValueError("tx_amplitude must be in (0, 1]")
        if self.pa_clip_level <= 0.0 or self.agc_target_rms <= 0.0:
            raise ValueError("PA clip level and AGC target must be positive")
        if not 0.0 <= self.detection_threshold <= 1.0:
            raise ValueError("detection_threshold must be in [0, 1]")
        if self.phase_correction_bits < 1:
            raise ValueError("phase_correction_bits must be positive")
        if self.frequency_correction_resolution_hz <= 0.0:
            raise ValueError("frequency correction resolution must be positive")

    @property
    def sample_period(self) -> float:
        return 1.0 / self.sample_rate


@dataclass(frozen=True)
class SyncPreamble:
    waveform: torch.Tensor
    short_sequence: torch.Tensor
    long_sequence: torch.Tensor
    short_length: int
    long_block_length: int

    @property
    def length(self) -> int:
        return self.waveform.numel()


@dataclass(frozen=True)
class IQCapture:
    samples: torch.Tensor
    inserted_start: int
    expected_arrival: int
    agc_gain: float
    clip_rate: float


@dataclass(frozen=True)
class SDRMeasurement:
    detected: bool
    phase: torch.Tensor
    frequency: torch.Tensor
    timing_index: int
    detection_metric: float


@dataclass(frozen=True)
class SDRSimulationResult:
    """Metrics from an SDR-style synchronization run."""

    true_phase: torch.Tensor
    measured_ota_phase: torch.Tensor
    estimated_ota_phase: torch.Tensor
    channel_phase: torch.Tensor
    true_frequency: torch.Tensor
    measured_frequency: torch.Tensor
    estimated_frequency: torch.Tensor
    ota_phase_error: torch.Tensor
    frequency_error: torch.Tensor
    post_correction_ota_phase: torch.Tensor
    post_correction_oscillator_phase: torch.Tensor
    post_correction_frequency: torch.Tensor
    timing_error_samples: torch.Tensor
    detection_metric: torch.Tensor
    detected: torch.Tensor
    agc_gain: torch.Tensor
    adc_clip_rate: torch.Tensor
    covariance: torch.Tensor
    device: torch.device

    @property
    def detection_rate(self) -> float:
        return torch.mean(self.detected.to(torch.float64)).item()

    @property
    def ota_phase_rmse(self) -> float:
        valid = self.detected
        if not torch.any(valid):
            return float("nan")
        return torch.sqrt(torch.mean(self.ota_phase_error[valid].square())).item()

    @property
    def frequency_rmse(self) -> float:
        valid = self.detected
        if not torch.any(valid):
            return float("nan")
        return torch.sqrt(torch.mean(self.frequency_error[valid].square())).item()

    @property
    def final_ota_phase_error(self) -> float:
        return self.post_correction_ota_phase[-1].item()

    @property
    def final_oscillator_phase_error(self) -> float:
        return self.post_correction_oscillator_phase[-1].item()

    @property
    def final_frequency_error_hz(self) -> float:
        return self.post_correction_frequency[-1].item() / (2.0 * math.pi)


def _zadoff_chu(length: int, root: int, device: torch.device) -> torch.Tensor:
    if math.gcd(length, root) != 1:
        raise ValueError("Zadoff-Chu root and length must be coprime")
    n = torch.arange(length, dtype=REAL_DTYPE, device=device)
    if length % 2:
        phase = -math.pi * root * n * (n + 1.0) / length
    else:
        phase = -math.pi * root * n.square() / length
    return torch.exp(1j * phase).to(COMPLEX_DTYPE)


def make_sync_preamble(
    settings: SDRSimulationConfig, device: torch.device
) -> SyncPreamble:
    """Construct short and long SDR synchronization training fields."""

    short = _zadoff_chu(settings.short_sequence_length, 1, device)
    long_root = 25
    while math.gcd(settings.long_sequence_length, long_root) != 1:
        long_root += 1
    long = _zadoff_chu(settings.long_sequence_length, long_root, device)
    short_field = short.repeat(settings.short_repetitions)
    cyclic_prefix = (
        long[-settings.long_cp_length :] if settings.long_cp_length else long[:0]
    )
    long_block = torch.cat((cyclic_prefix, long))
    long_field = long_block.repeat(settings.long_repetitions)
    waveform = torch.cat((short_field, long_field))
    return SyncPreamble(
        waveform=waveform,
        short_sequence=short,
        long_sequence=long,
        short_length=short_field.numel(),
        long_block_length=long_block.numel(),
    )


def _quantize_iq(samples: torch.Tensor, bits: int) -> tuple[torch.Tensor, float]:
    limit = float(2 ** (bits - 1) - 1)
    real = samples.real
    imag = samples.imag
    clipped = (real.abs() > 1.0) | (imag.abs() > 1.0)
    real = torch.round(torch.clamp(real, -1.0, 1.0) * limit) / limit
    imag = torch.round(torch.clamp(imag, -1.0, 1.0) * limit) / limit
    return torch.complex(real, imag), torch.mean(clipped.to(REAL_DTYPE)).item()


def _soft_limit(samples: torch.Tensor, limit: float) -> torch.Tensor:
    magnitude = torch.abs(samples)
    scale = torch.clamp(limit / torch.clamp(magnitude, min=1e-15), max=1.0)
    return samples * scale


def _resample_clock_offset(samples: torch.Tensor, ppm: float) -> torch.Tensor:
    """Linearly resample IQ at a receiver clock offset from nominal."""

    if ppm == 0.0:
        return samples
    ratio = 1.0 + ppm * 1e-6
    output_index = torch.arange(
        samples.numel(), dtype=REAL_DTYPE, device=samples.device
    )
    source = output_index / ratio
    lower = torch.floor(source).to(torch.int64)
    upper = lower + 1
    valid = upper < samples.numel()
    lower = torch.clamp(lower, 0, samples.numel() - 1)
    upper = torch.clamp(upper, 0, samples.numel() - 1)
    fraction = source - lower.to(REAL_DTYPE)
    output = samples[lower] * (1.0 - fraction) + samples[upper] * fraction
    return torch.where(valid, output, torch.zeros_like(output))


class SDRRadioLink:
    """Sampled-IQ transmit/channel/receive chain using Sionna TDL and AWGN."""

    def __init__(
        self,
        settings: SDRSimulationConfig,
        preamble: SyncPreamble,
        device: torch.device,
        generator: torch.Generator,
    ) -> None:
        self.settings = settings
        self.preamble = preamble
        self.device = device
        self.generator = generator
        self.input_length = (
            preamble.length
            + 2 * settings.capture_guard_samples
            + settings.timing_jitter_samples
        )
        self.l_min, self.l_max = time_lag_discrete_time_channel(
            settings.sample_rate, settings.maximum_channel_delay_s
        )
        self.l_tot = self.l_max - self.l_min + 1

        self.tdl = TDL(
            model=settings.tdl_model,
            delay_spread=settings.delay_spread_s,
            carrier_frequency=settings.carrier_frequency_hz,
            min_speed=settings.channel_speed_mps,
            max_speed=settings.channel_speed_mps,
            precision="double",
            device=str(device),
        )
        coefficients, delays = self.tdl(
            batch_size=1,
            num_time_steps=settings.num_iterations,
            sampling_frequency=1.0 / settings.sync_interval,
        )
        self.channel_taps = cir_to_time_channel(
            settings.sample_rate,
            coefficients,
            delays,
            self.l_min,
            self.l_max,
            normalize=True,
        )
        self._apply_channel = ApplyTimeChannel(
            num_time_samples=self.input_length,
            l_tot=self.l_tot,
            precision="double",
            device=str(device),
        )
        self._awgn = AWGN(precision="double", device=str(device))

    def _random_start(self) -> int:
        jitter = torch.randint(
            self.settings.timing_jitter_samples + 1,
            (),
            device=self.device,
            generator=self.generator,
        ).item()
        return self.settings.capture_guard_samples + jitter

    def _channel_for_frame(self, iteration: int) -> torch.Tensor:
        output_length = self.input_length + self.l_tot - 1
        taps = self.channel_taps[..., iteration, :].unsqueeze(-2)
        return taps.expand(*taps.shape[:-2], output_length, self.l_tot)

    def capture(
        self, master: Oscillator, slave: Oscillator, iteration: int
    ) -> IQCapture:
        settings = self.settings
        start = self._random_start()
        center = start + (self.preamble.length - 1) / 2.0
        sample_index = torch.arange(
            self.input_length, dtype=REAL_DTYPE, device=self.device
        )
        time = (sample_index - center) * settings.sample_period

        frame = torch.zeros(self.input_length, dtype=COMPLEX_DTYPE, device=self.device)
        frame[start : start + self.preamble.length] = (
            settings.tx_amplitude * self.preamble.waveform
        )
        master_carrier = torch.exp(1j * (master.phase + master.frequency * time))
        tx = _soft_limit(frame * master_carrier, settings.pa_clip_level)
        tx, _ = _quantize_iq(tx, settings.dac_bits)

        channel_input = tx.reshape(1, 1, 1, -1)
        clean = self._apply_channel(
            channel_input, self._channel_for_frame(iteration)
        ).reshape(-1)
        active_start = max(0, start - self.l_min)
        active_stop = min(clean.numel(), active_start + self.preamble.length)
        signal_power = torch.mean(torch.abs(clean[active_start:active_stop]).square())
        noise_power = signal_power / (10.0 ** (settings.snr_db / 10.0))
        received = self._awgn(clean, noise_power)

        received = _resample_clock_offset(received, settings.sample_clock_offset_ppm)
        output_index = torch.arange(
            received.numel(), dtype=REAL_DTYPE, device=self.device
        )
        physical_index = (
            output_index / (1.0 + settings.sample_clock_offset_ppm * 1e-6) + self.l_min
        )
        receive_time = (physical_index - center) * settings.sample_period
        slave_carrier = torch.exp(-1j * (slave.phase + slave.frequency * receive_time))
        received = received * slave_carrier

        if settings.phase_noise_std_rad > 0.0:
            phase_noise = torch.cumsum(
                torch.randn(
                    received.numel(),
                    dtype=REAL_DTYPE,
                    device=self.device,
                    generator=self.generator,
                )
                * settings.phase_noise_std_rad,
                dim=0,
            )
            received = received * torch.exp(1j * phase_noise)

        gain_ratio = 10.0 ** (settings.iq_gain_imbalance_db / 40.0)
        phase_skew = math.radians(settings.iq_phase_imbalance_deg)
        in_phase = received.real * gain_ratio
        quadrature = received.imag / gain_ratio
        quadrature = quadrature * math.cos(phase_skew) + in_phase * math.sin(phase_skew)
        received = torch.complex(in_phase, quadrature)

        rms = torch.sqrt(torch.mean(torch.abs(received).square()))
        agc_gain = torch.clamp(settings.agc_target_rms / rms, 0.01, 1000.0)
        received = received * agc_gain + settings.dc_offset
        received, clip_rate = _quantize_iq(received, settings.adc_bits)

        return IQCapture(
            samples=received,
            inserted_start=start,
            expected_arrival=start - self.l_min,
            agc_gain=agc_gain.item(),
            clip_rate=clip_rate,
        )


def _sliding_sum(values: torch.Tensor, width: int) -> torch.Tensor:
    cumulative = torch.cat(
        (
            torch.zeros(1, dtype=values.dtype, device=values.device),
            torch.cumsum(values, dim=0),
        )
    )
    return cumulative[width:] - cumulative[:-width]


class SDRSynchronizer:
    """Packet detector and two-stage carrier synchronizer."""

    def __init__(
        self,
        settings: SDRSimulationConfig,
        preamble: SyncPreamble,
    ) -> None:
        self.settings = settings
        self.preamble = preamble

    def _coarse_timing_and_frequency(
        self, samples: torch.Tensor
    ) -> tuple[int, torch.Tensor]:
        lag = self.settings.short_sequence_length
        width = self.preamble.short_length - lag
        product = torch.conj(samples[:-lag]) * samples[lag:]
        first_power = torch.abs(samples[:-lag]).square()
        second_power = torch.abs(samples[lag:]).square()
        correlation = _sliding_sum(product, width)
        normalization = torch.sqrt(
            _sliding_sum(first_power, width) * _sliding_sum(second_power, width)
        ).clamp_min(1e-15)
        metric = torch.abs(correlation) / normalization
        coarse_start = int(torch.argmax(metric).item())
        frequency = torch.angle(correlation[coarse_start]) / (
            lag * self.settings.sample_period
        )
        return coarse_start, frequency

    def estimate(self, capture: IQCapture) -> SDRMeasurement:
        samples = capture.samples - torch.mean(capture.samples)
        coarse_start, coarse_frequency = self._coarse_timing_and_frequency(samples)

        sample_index = torch.arange(
            samples.numel(), dtype=REAL_DTYPE, device=samples.device
        )
        coarse_corrected = samples * torch.exp(
            -1j
            * coarse_frequency
            * (sample_index - coarse_start)
            * self.settings.sample_period
        )

        if coarse_corrected.numel() < self.preamble.length:
            return SDRMeasurement(
                False,
                torch.zeros((), dtype=REAL_DTYPE, device=samples.device),
                coarse_frequency,
                coarse_start,
                0.0,
            )
        windows = coarse_corrected.unfold(0, self.preamble.length, 1)
        correlations = torch.sum(windows * torch.conj(self.preamble.waveform), dim=-1)
        window_energy = torch.sum(torch.abs(windows).square(), dim=-1)
        reference_energy = torch.sum(torch.abs(self.preamble.waveform).square())
        normalized = torch.abs(correlations) / torch.sqrt(
            (window_energy * reference_energy).clamp_min(1e-15)
        )
        timing = int(torch.argmax(normalized).item())
        detection_metric = normalized[timing].item()

        short = samples[timing : timing + self.preamble.short_length]
        lag = self.settings.short_sequence_length
        short_correlation = torch.sum(torch.conj(short[:-lag]) * short[lag:])
        coarse_frequency = torch.angle(short_correlation) / (
            lag * self.settings.sample_period
        )

        segment = samples[timing : timing + self.preamble.length]
        relative_index = torch.arange(
            segment.numel(), dtype=REAL_DTYPE, device=samples.device
        )
        coarse_segment = segment * torch.exp(
            -1j * coarse_frequency * relative_index * self.settings.sample_period
        )

        first_start = self.preamble.short_length + self.settings.long_cp_length
        last_start = (
            self.preamble.short_length
            + (self.settings.long_repetitions - 1) * self.preamble.long_block_length
            + self.settings.long_cp_length
        )
        first_long = coarse_segment[
            first_start : first_start + self.settings.long_sequence_length
        ]
        last_long = coarse_segment[
            last_start : last_start + self.settings.long_sequence_length
        ]
        separation = (
            self.settings.long_repetitions - 1
        ) * self.preamble.long_block_length
        fine_correlation = torch.sum(torch.conj(first_long) * last_long)
        fine_frequency = torch.angle(fine_correlation) / (
            separation * self.settings.sample_period
        )
        frequency = coarse_frequency + fine_frequency

        centered_index = relative_index - (self.preamble.length - 1) / 2.0
        corrected = segment * torch.exp(
            -1j * frequency * centered_index * self.settings.sample_period
        )
        phase_correlation = torch.zeros((), dtype=COMPLEX_DTYPE, device=samples.device)
        for repeat in range(self.settings.long_repetitions):
            sequence_start = (
                self.preamble.short_length
                + repeat * self.preamble.long_block_length
                + self.settings.long_cp_length
            )
            sequence = corrected[
                sequence_start : sequence_start + self.settings.long_sequence_length
            ]
            phase_correlation += torch.sum(
                torch.conj(self.preamble.long_sequence) * sequence
            )
        phase = torch.angle(phase_correlation)
        detected = detection_metric >= self.settings.detection_threshold
        return SDRMeasurement(
            detected=detected,
            phase=phase,
            frequency=frequency,
            timing_index=timing,
            detection_metric=detection_metric,
        )


def _measurement_covariance(
    settings: SDRSimulationConfig, preamble: SyncPreamble, device: torch.device
) -> torch.Tensor:
    snr = 10.0 ** (settings.snr_db / 10.0)
    phase_variance = 1.0 / (
        2.0 * snr * settings.long_sequence_length * settings.long_repetitions
    )
    separation_time = (
        (settings.long_repetitions - 1)
        * preamble.long_block_length
        * settings.sample_period
    )
    frequency_variance = 1.0 / (
        snr * settings.long_sequence_length * separation_time**2
    )
    return torch.diag(
        torch.tensor(
            [phase_variance, phase_variance, frequency_variance],
            dtype=REAL_DTYPE,
            device=device,
        )
    )


def _quantize_correction(
    correction: torch.Tensor, settings: SDRSimulationConfig
) -> torch.Tensor:
    phase_step = 2.0 * math.pi / (2**settings.phase_correction_bits)
    frequency_step = 2.0 * math.pi * settings.frequency_correction_resolution_hz
    return torch.stack(
        (
            torch.round(correction[0] / phase_step) * phase_step,
            torch.round(correction[1] / frequency_step) * frequency_step,
        )
    )


def run_sdr_simulation(
    settings: SDRSimulationConfig = SDRSimulationConfig(),
) -> SDRSimulationResult:
    """Run an SDR-style one-way OTA phase/frequency synchronization loop."""

    device = resolve_device(settings.device)
    torch.manual_seed(settings.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(settings.seed)
    sionna_config.seed = settings.seed
    generator = torch.Generator(device=device)
    generator.manual_seed(settings.seed + 1)

    frequency_process_std = 2.0 * math.pi * settings.frequency_process_std_hz
    oscillator_covariance = torch.diag(
        torch.tensor(
            [settings.phase_process_std_rad**2, frequency_process_std**2],
            dtype=REAL_DTYPE,
            device=device,
        )
    )
    master = Oscillator(
        settings.master_initial_phase,
        2.0 * math.pi * settings.master_initial_frequency_hz,
        settings.sync_interval,
        oscillator_covariance,
        device,
        generator,
    )
    slave = Oscillator(
        settings.slave_initial_phase,
        2.0 * math.pi * settings.slave_initial_frequency_hz,
        settings.sync_interval,
        oscillator_covariance,
        device,
        generator,
    )
    preamble = make_sync_preamble(settings, device)
    link = SDRRadioLink(settings, preamble, device, generator)
    synchronizer = SDRSynchronizer(settings, preamble)
    measurement_noise = _measurement_covariance(settings, preamble, device)
    ekf = PhaseFrequencyEKF(
        settings.sync_interval,
        2.0 * oscillator_covariance,
        measurement_noise,
        device,
        initial_covariance=torch.diag(
            torch.tensor(
                [math.pi**2, (2.0 * math.pi * 50e3) ** 2],
                dtype=REAL_DTYPE,
                device=device,
            )
        ),
    )
    acquired = False

    history: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "true_phase",
            "measured_ota_phase",
            "estimated_ota_phase",
            "channel_phase",
            "true_frequency",
            "measured_frequency",
            "estimated_frequency",
            "ota_phase_error",
            "frequency_error",
            "post_correction_ota_phase",
            "post_correction_oscillator_phase",
            "post_correction_frequency",
            "timing_error_samples",
            "detection_metric",
            "detected",
            "agc_gain",
            "adc_clip_rate",
            "covariance",
        )
    }

    for iteration in range(settings.num_iterations):
        master.step()
        slave.step()
        relative_state = master.state - slave.state
        capture = link.capture(master, slave, iteration)
        measurement = synchronizer.estimate(capture)
        ekf.predict()

        if measurement.detected:
            if not acquired:
                # Packet acquisition resolves CFO before the wrapped-phase EKF
                # begins tracking. This prevents a large initial CFO from being
                # pulled toward an adjacent phase cycle by state correlation.
                ekf.state = torch.stack((measurement.phase, measurement.frequency))
                ekf.covariance = torch.diag(
                    torch.stack((measurement_noise[0, 0], measurement_noise[2, 2]))
                )
                acquired = True
            else:
                observation = torch.stack(
                    (
                        torch.cos(measurement.phase),
                        torch.sin(measurement.phase),
                        measurement.frequency,
                    )
                )
                ekf.update(observation)
            correction = _quantize_correction(ekf.state.clone(), settings)
        else:
            correction = torch.zeros(2, dtype=REAL_DTYPE, device=device)

        true_phase = wrap_phase(relative_state[0])
        estimated_phase = wrap_phase(ekf.state[0])
        channel_phase = wrap_phase(measurement.phase - true_phase)
        history["true_phase"].append(true_phase.clone())
        history["measured_ota_phase"].append(measurement.phase.clone())
        history["estimated_ota_phase"].append(estimated_phase.clone())
        history["channel_phase"].append(channel_phase.clone())
        history["true_frequency"].append(relative_state[1].clone())
        history["measured_frequency"].append(measurement.frequency.clone())
        history["estimated_frequency"].append(ekf.state[1].clone())
        history["ota_phase_error"].append(
            wrap_phase(measurement.phase - estimated_phase)
        )
        history["frequency_error"].append((relative_state[1] - ekf.state[1]).clone())
        history["timing_error_samples"].append(
            torch.tensor(
                measurement.timing_index - capture.expected_arrival,
                dtype=REAL_DTYPE,
                device=device,
            )
        )
        history["detection_metric"].append(
            torch.tensor(measurement.detection_metric, dtype=REAL_DTYPE, device=device)
        )
        history["detected"].append(
            torch.tensor(measurement.detected, dtype=torch.bool, device=device)
        )
        history["agc_gain"].append(
            torch.tensor(capture.agc_gain, dtype=REAL_DTYPE, device=device)
        )
        history["adc_clip_rate"].append(
            torch.tensor(capture.clip_rate, dtype=REAL_DTYPE, device=device)
        )
        history["covariance"].append(ekf.covariance.clone())

        slave.apply_correction(correction)
        ekf.reset_after_correction(correction)
        residual_state = master.state - slave.state
        history["post_correction_ota_phase"].append(
            wrap_phase(measurement.phase - correction[0]).clone()
        )
        history["post_correction_oscillator_phase"].append(
            wrap_phase(residual_state[0]).clone()
        )
        history["post_correction_frequency"].append(residual_state[1].clone())

    return SDRSimulationResult(
        **{
            name: torch.stack(values).detach().cpu() for name, values in history.items()
        },
        device=device,
    )
