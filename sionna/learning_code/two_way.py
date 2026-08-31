from __future__ import annotations

from dataclasses import dataclass 
import math 
from typing import Optional 

import torch  
from sionna.phy import config as sionna_config 
from sionna.phy.channel import AWGN, ApplyTimeChannel 
# 3gpp tap delay line 
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.channel.utils import (
    cir_to_time_channel, 
    time_lag_discrete_time_channel, 
)

REAL_DTYPE = torch.float64 
COMPLEX_DTYPE = torch.complex128 

def wrap_phase(value: torch.Tensor) -> torch.tensor: 
    "Wraps angles in radians to [-pi, pi]"
    return torch.remainder(value + math.pi , 2.0 * math.pi) - math.pi 

def _as_matrix(
        value: torch.Tensor| list[list[float]], device: torch.device
) -> torch.Tensor:
    return torch.as_tensor(value, dtype=REAL_DTYPE, device=device) 

def _covariance_root(covariance: torch.Tensor) -> torch.Tensor:  
    '''Returns a sqrt for a PSD Covariance Matrix'''
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance) 

    if torch.any(eigenvalues < -1e-12): 
        raise ValueError("covariance must be PSD")
    return eigenvectors @ torch.diag(torch.sqrt(torch.clamp(eigenvalues, min=0.0)))

def resolve_device(requested: str) -> torch.device: 
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested) 
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Cuda was requested but is not available") 
    return device 

class Oscillator:  
    "Random-walk phase and angular-frequency oscillator model" 

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
            device = device 
        )
        self.process_covariance = _as_matrix(process_covariance, device) 
        if self.process_covariance !=(2,2): 
            raise ValueError("Process covariance must be 2x2")
        
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
                generator=self._generator 
            )

            process_noise = self._noise_root@white_noise 
            self.state = self.transition @ self.state + process_noise
            self.state[0] = wrap_phase(self.state[0]) 
            return self.state.close() 
        
        def apply_correction(self, correction: torch.Tensor) -> None:  
            self.state = self.state + correction.to(dtype=REAL_DTYPE, device=self.device) 
            self.state[0] = wrap_phase(self.state[0])  

@dataclass(frozen=True) 
class SDRSimulationConfig:  
    sync_interval: float = 0.05 
    sample_rate: float = 1e6  
    num_iterations: int = 100 
    carrier_frequency_hz: float = 915e6  

    tdl_model: str = "D" 
    delay_spread_s: float = 100e-9  
    maximum_channel_delay_s: float = 3e-6 
    channel_spead_mps: float = 0.0  

    short_sequence_length: int = 16 
    short_repetitions: int = 16 
    long_sequence_length: int = 2047 
    long_cp_length: int = 128 
    long_reptitions: int = 2
    capture_guard_samples: int = 64  
    timing_jitter_samples: int = 32  

    master_initial_phase: float = 0.0 
    master_initial_frequency_hz: float = 0.0

    slave_initial_phase: float = 0.0 
    slave_initial_frequency_hz: float = 0.0  

    phase_process_std_rad: float = 0.002 
    frequency_process_std_hz: float = 0.1 

    sample_clock_offset_ppm: float | None = None 
    phase_noise_std_rad: float = 2e-4 
    phase_noise_white_pm_std_rad: float = 0.005 
    flicker_frequency_std_hz: float = 0.05 
    shadowing_std_db: float = 2.0 
    shadowing_correlation_s: float = 10.0  
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
    correction_latency_intervals: int = 1  

    twoway_chain_assymetry_deg: float = 0.0  

    tdd_turnaround_s: float = 1e-3 
    seed: int = 0  
    device: str = "auto " 

def __post_init__(self) -> None: 
    if self.sync_interval <= 0.0 or self.sample_rate <= 0.0: 
        raise ValueError("sync and sample must be positive")
    if self.num_iterations < 1:  
        raise ValueError("num_iterations must be positive") 
    if self.carrier_frequency_hz <= 0.0: 
        raise ValueError("carrier_frequency_hz must be positive")
    if self.tdl_model not in{ 
        "A", 
        "B", 
        "C", 
        "D", 
        "E" 
        "A30", 
        "B100", 
        "C300"
    }: 
        raise ValueError("invalid Sionna TDL Model") # need to read up more on this 
    
    if self.delay_spread_s <= 0.0 or self.maximum_channel_delay_s <= 0.0: 
        raise ValueError("channel delay must be postive") \
        
    if self.channel_spread_mps < 0.0: 
        raise ValueError("channel spread > 0.0")
    if self.short_sequence_length < 2 or self.short_repetitions < 3: 
        raise  ValueError("STF needs at least 3 repeats") 
    if self.long_sequence_length < 3 or self.long_reptitions < 2: 
        raise ValueError("the LTF needs at least 2 repeats") 
    
    if not 0 <= self.long_cp_length < self.long_sequence_length: 
        raise ValueError("long_cp_length must be shorter than the long sequence") 
    if self.capture_guard_samples < 1 or self.timing_jitter_samples < 0: 
        raise ValueError("capture guard must be positive and jitter, nonegative") 
    
    if self.phase_process_std_rad < 0.0 or self.frequency_process_std_hz < 0.0:
        raise ValueError("oscillator process deviations cannot be negative")
    if self.phase_noise_std_rad < 0.0:
        raise ValueError("phase_noise_std_rad cannot be negative")
    if self.phase_noise_white_pm_std_rad < 0.0:
        raise ValueError("phase_noise_white_pm_std_rad cannot be negative")
    if self.flicker_frequency_std_hz < 0.0:
        raise ValueError("flicker_frequency_std_hz cannot be negative")
    if self.shadowing_std_db < 0.0:
        raise ValueError("shadowing_std_db cannot be negative")
    if self.shadowing_correlation_s <= 0.0:
        raise ValueError("shadowing_correlation_s must be positive")
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
    if self.correction_latency_intervals < 0:
        raise ValueError("correction_latency_intervals cannot be negative")
    if self.tdd_turnaround_s < 0.0:
        raise ValueError("tdd_turnaround_s cannot be negative") 
    
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
    oracle_samples: torch.Tensor
    lo_walk_end: torch.Tensor
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

# Constant-amplitude chirps with a single-spike autocorrelation — the same family LTE and 5G use.
# The preamble is 16 × 16 short repeats (for coarse timing and frequency)
# then 2047 × 2 long sequences with a cyclic prefix (for fine frequency and phase).
# Why 2047: the frequency-estimate error times the 50 ms interval is phase drift
# the loop must ride out, and it must stay far inside the ±π/2 branch boundary
# you will meet later. 

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
    long_field = long_block.repeat(settings.long_reptitions)
    waveform = torch.cat((short_field, long_field))

    return SyncPreamble(
        waveform=waveform, 
        short_sequence=short, 
        long_sequence=long, 
        short_length=short_field.numel(), # counts num of elements 
        long_block_length = long_block.numel() 
    ) 

def _quantize_iq(samples: torch.Tensor, bits:int) -> tuple[torch.Tensor, float]: 
    limit = float(2 ** (bits - 1) -1) 
    real = samples.real 
    imag = samples.imag 
    clipped = (real.abs() > 1.0) | (imag.abs() > 1.0) 
    real = torch.round(torch.clamp(real, -1.0, 1.0)*limit)/limit 
    imag = torch.round(torch.clamp(imag, -1.0, 1.0)*limit)/limit 
    return torch.complex(real, imag), torch.mean(clipped.to(REAL_DTYPE)).item()

def _soft_limit(samples: torch.Tensor, limit:float) -> torch.Tensor: 
    magnitude = torch.abs(samples) 
    scale = torch.clamp(limit / torch.clamp(magnitude, min=1e-15), max=1.0)
    return samples * scale 

def _resample_clock_offset( 
        samples: torch.Tensor, ppm: float, delay_samples = 0.0
) -> torch.Tensor: 
    if ppm == 0.0 and delay_samples == 0.0: 
        return samples 
    # sampling offest ratio (1+ppm%) 
    ratio = 1.0 + ppm * 1e-6 
    output_indx = torch.arange(
        samples.numel(), dtype=REAL_DTYPE, device=samples.device 
    )
    source = output_indx/ ratio - delay_samples 
    lower = torch.floor(source).to(torch.int64) 
    upper = lower + 1 
    valid = (lower >= 0) & (upper < samples.numel())  

    lower = torch.clamp(lower, 0, samples.numel() - 1)
    upper = torch.clamp(upper, 0, samples.numel() - 1)
    fraction = source - lower.to(REAL_DTYPE)  
    output = samples[lower] * (1.0 - fraction) + samples[upper] * fraction 
 
    return torch.where(valid, output, torch.zeros_like(output))   

class _FlickerFrequencyNoise:  

    def __init__(
            self, 
            std_hz: float, 
            interval_s: float, 
            horizon_s: float, 
            device: torch.device, 
            generator: torch.Generator, 
            components:int = 4, 
    ) -> None:  
        self.device = device
        self.generator = generator  
        self.enabled = std_hz > 0.0 
    
    def step(self) -> torch.Tensor: 
        if not self.enabled:  
            self.innovation_variance = 0.0 
            return 
        innovation = (
            torch.randn(
                self.sigma.numel(),
                dtype=REAL_DTYPE,
                device=self.device,
                generator=self.generator,
            )
            * self.sigma 
            * torch.sqrt(1.0 - self.rho.square())

        )
        self.state = self.rho * self.state + innovation 
        return torch.sum(self.state)

class SDRRadioLink:
    """Sampled-IQ transmit/channel/receive chain using Sionna TDL and AWGN."""

    def __init__(
        self,
        settings: SDRSimulationConfig,
        preamble: SyncPreamble,
        device: torch.device,
        generator: torch.Generator,
        mirror_of: "SDRRadioLink | None" = None,
        captures_per_interval: int = 1,
    ) -> None:
        self.settings = settings
        self.preamble = preamble
        self.device = device
        self.generator = generator
        self.captures_per_interval = captures_per_interval
        # A mirrored link models the reverse direction of a reciprocal
        # channel: it shares the taps and shadowing of the forward link (which
        # must capture first each interval) while keeping its own receiver
        # state (timing carry, noise floor).
        self._mirror = mirror_of
        self._last_shadow_amplitude = 1.0
        self.input_length = (
            preamble.length
            + 2 * settings.capture_guard_samples
            + settings.timing_jitter_samples
        )
        self.l_min, self.l_max = time_lag_discrete_time_channel(
            settings.sample_rate, settings.maximum_channel_delay_s
        )
        self.l_tot = self.l_max - self.l_min + 1
        self.interval_samples = int(
            round(settings.sync_interval * settings.sample_rate)
        )
        # Receiver clock error accumulated since the capture window was last
        # re-centered, in samples.
        self._timing_carry = 0.0
        # Thermal noise floor, fixed after the first frame at the nominal
        # (unshadowed) channel gain so fading changes the actual SNR.
        self._noise_power: torch.Tensor | None = None
        self._shadow_rho = math.exp(
            -settings.sync_interval / settings.shadowing_correlation_s
        )
        self._shadow_db = settings.shadowing_std_db * torch.randn(
            (), dtype=REAL_DTYPE, device=device, generator=generator
        )

        if mirror_of is not None:
            self.channel_taps = mirror_of.channel_taps
        else:
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
            generator=self.generator
        ).item() 
        return self.settings.capture_guard_samples + jitter 

    # ---------------------- 
    # Need to understand this 
    # section better 
    # ---------------------- 
    def _channel_for_frame(self, iteration: int) -> torch.Tensor: 
        output_length = self.input_length + self.l_tot - 1 
        taps = self.channel_taps[..., iteration, :].unsqueeze(-2)  
        return taps.expand(*taps.shape[:-2])  
    
    def _step_shadowing(self) -> float:
        """Advance the temporally correlated log-normal shadowing process."""

        if self._mirror is not None:
            # Reciprocity: reuse the forward link's shadowing for this frame.
            return self._mirror._last_shadow_amplitude
        if self.settings.shadowing_std_db <= 0.0:
            self._last_shadow_amplitude = 1.0
            return 1.0
        innovation = torch.randn(
            (), dtype=REAL_DTYPE, device=self.device, generator=self.generator
        )
        self._shadow_db = (
            self._shadow_rho * self._shadow_db
            + math.sqrt(1.0 - self._shadow_rho**2)
            * self.settings.shadowing_std_db
            * innovation
        )
        self._last_shadow_amplitude = float(10.0 ** (self._shadow_db.item() / 20.0))
        return self._last_shadow_amplitude 
    # ======================================================================


    def capture(
            self, 
            master: Oscillator, 
            slave: Oscillator, 
            iteration: int, 
            sfo_ppm: float
    ) -> IQCapture:  
        settings = self.settings 

        self._timing_carry += (
            sfo_ppm * 1e-6 * self.interval_samples / self.captures_per_interval
        )  

        drift_step = int(round(self._timing_carry))  
        self._timing_carry -= drift_step 
        fractional_delay = self._timing_carry  
        start = self._random_start() + drift_step 

        start= min(max(start, 0), self.input_length - self.preamble.length) 
        center = start + (self.preamble.length - 1 )/ 2.0 

        sample_index = torch.arange(
            self.input_length, dtype=REAL_DTYPE, device=self.device 
        ) 

        time = (sample_index - center) * settings.sample_period  

        frame = torch.zeros(self.input_length, dtype=COMPLEX_DTYPE, device=self.device)
        frame[start : start + self.preamble.length] = (
            settings.tx_amplitude * self.preamble.waveform
        ) 

        master_carrier = torch.exp(1j * (master.phase + master.frequency * time)) 
        
        tx = _soft_limit(frame*master_carrier, settings.pa_clip_level)
        tx,_ = _quantize_iq(tx, settings.dac_bits)  

        channel_input = tx.reshape(1, 1, 1, -1) 
        clean = self._apply_channel(
            channel_input, self._channel_for_frame(iteration)
        ).reshape(-1) 
        shadow_amplitude = self._step_shadowing() 
        clean = clean * shadow_amplitude 

        # RX SECTION:  

        active_start = max(0, start - self.l_min) 
        active_stop = min(clean.numel(), active_start + self.preamble.length) 

        if self._noise_power is None:  
            nominal_power = torch.mean(
                torch.abs(clean[active_start:active_stop] / shadow_amplitude).square()
            ) 
            self._noise_power = nominal_power / (10.0 ** (settings.snr_db / 10.0))
        
        received = self._awgn(clean, self._noise_power)
        received = _resample_clock_offset(received, sfo_ppm, fractional_delay) 

        # Implement deterministic oracle:  
        oracle = _resample_clock_offset(received, sfo_ppm, fractional_delay)  
        output_index = torch.arange( 
            received.numel(), dtype=REAL_DTYPE, device=self.device 
        ) 

        physical_index = output_index / (1.0 + sfo_ppm *1e-6) + self.l_min 
        receive_time = (physical_index - center) * settings.sample_period  
        slave_carrier = torch.exp(-1j * (slave.phase + slave.frequency * receive_time)) 

        received = received * slave_carrier 
        oracle = oracle * slave_carrier 

        lo_walk_end = torch.zeros((), dtype=REAL_DTYPE, device=self.device)
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
            # End value of the intra-frame walk; the caller carries it into
            # the oscillator state so the LO noise is one continuous process.
            lo_walk_end = phase_noise[-1]

        if settings.phase_noise_white_pm_std_rad > 0.0:
            jitter = (
                torch.randn(
                    received.numel(),
                    dtype=REAL_DTYPE,
                    device=self.device,
                    generator=self.generator,
                )
                * settings.phase_noise_white_pm_std_rad
            )
            received = received * torch.exp(1j * jitter)

        gain_ratio = 10.0 ** (settings.iq_gain_imbalance_db / 40.0) 
        phase_skew = math.radians(settings.iq_phase_imbalance_deg)
        in_phase = received.real * gain_ratio 
        quadrature = received.imag * gain_ratio

        quadrature = quadrature * math.cos(phase_skew) + in_phase * math.sin(phase_skew)
        received = torch.complex(in_phase, quadrature)

        rms = torch.sqrt(torch.mean(torch.abs(received).square))
        agc_gain = torch.clamp(settings.agc_target_rms / rms, 0.01, 1000.0) 
        received = received * agc_gain + settings.dc_offset 

        received, clip_rate = _quantize_iq(received, settings.adc_bits) 

        return IQCapture(
            samples = received, 
            oracle_samples=oracle, 
            lo_walk_end=lo_walk_end, 
            inserted_start=start, 
            expected_arrival=start - self.l_min, 
            agc_gain= agc_gain.item(),  
            clip_rate=clip_rate
        ) 
    
def _sliding_sum(values: torch.Tensor, width: int) -> torch.Tensor:  
    cumulative = torch.cat(
        (
            torch.zeros(1, dtype=values.dtype, device=values.device), 
            torch.cumsum(values, dim=0)
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
    
    def estimate(self, samples: torch.Tensor) -> SDRMeasurement: 
        samples = samples - torch.mean(samples)
         
        # Define Coarse Start + Frequency 

        coarse_start, coarse_frequency = self._coarse_timing_and_frequency(samples) 

        sample_index = torch.arange(
            samples.numel(), dtype=REAL_DTYPE, device = samples.device  
        )

        coarse_corrected = samples * torch.exp(
            -1j 
            * coarse_frequency 
            *(sample_index - coarse_start)
            * self.settings.sample_period 
        ) 

        if coarse_corrected.numel() < self.preamble.length: 
            return SDRMeasurement(
                False, 
                torch.zeros((), dtype=REAL_DTYPE, device=samples.device), 
                coarse_frequency, 
                coarse_start, 
                0.0
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
    # The receiver LO phase noise accumulates as a random walk across the
    # frame and usually dominates the AWGN Cramer-Rao bound. The phase
    # estimate averages the walk over the long training fields, and the fine
    # CFO estimate differences the walk between the two fields.
    long_field_offset = preamble.short_length + settings.long_cp_length
    long_field_span = preamble.length - long_field_offset
    phase_noise_variance = settings.phase_noise_std_rad**2
    phase_variance += phase_noise_variance * (
        long_field_offset + long_field_span / 3.0
    )
    separation_samples = (
        settings.long_repetitions - 1
    ) * preamble.long_block_length
    frequency_variance += (
        phase_noise_variance * separation_samples / separation_time**2
    )
    # White PM adds an independent phase jitter per sample; it averages down
    # over the long training fields but still belongs in the noise budget.
    white_pm_variance = settings.phase_noise_white_pm_std_rad**2
    total_long_samples = settings.long_sequence_length * settings.long_repetitions
    phase_variance += white_pm_variance / total_long_samples
    frequency_variance += 2.0 * white_pm_variance / (
        settings.long_sequence_length * separation_time**2
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


@dataclass(frozen=True)
class TwoWaySimulationResult:
    """Metrics from a reciprocal two-way synchronization run.

    Unlike the one-way loop, the oscillator offset is identifiable, so all
    phase metrics are referenced to the true relative oscillator state.
    """

    true_phase: torch.Tensor
    measured_phase: torch.Tensor
    estimated_phase: torch.Tensor
    true_frequency: torch.Tensor
    physical_relative_frequency: torch.Tensor
    estimated_frequency: torch.Tensor
    phase_error: torch.Tensor
    frequency_error: torch.Tensor
    post_correction_phase: torch.Tensor
    post_correction_frequency: torch.Tensor
    coherent_gain: torch.Tensor
    detected: torch.Tensor
    correction_active: torch.Tensor
    calibrated: torch.Tensor
    airtime_fraction: float
    device: torch.device

    @property
    def detection_rate(self) -> float:
        return torch.mean(self.detected.to(torch.float64)).item()

    @property
    def phase_rmse(self) -> float:
        valid = self.detected
        if not torch.any(valid):
            return float("nan")
        return torch.sqrt(torch.mean(self.phase_error[valid].square())).item()

    @property
    def steady_state_phase_rms(self) -> float:
        valid = self.detected & self.correction_active & self.calibrated
        if not torch.any(valid):
            return float("nan")
        return torch.sqrt(
            torch.mean(self.post_correction_phase[valid].square())
        ).item()

    @property
    def mean_coherent_gain(self) -> float:
        valid = self.detected & self.correction_active & self.calibrated
        if not torch.any(valid):
            return float("nan")
        return torch.mean(self.coherent_gain[valid]).item()

    @property
    def final_phase_error(self) -> float:
        return self.post_correction_phase[-1].item()

    @property
    def final_frequency_error_hz(self) -> float:
        return self.post_correction_frequency[-1].item() / (2.0 * math.pi)
    

def _pick_half_phase(
    combined_half: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    """Resolve the pi ambiguity of a half-difference phase.

    (a - b)/2 is only defined modulo pi. The candidate closer to the
    reference (the EKF prediction after acquisition) is chosen; the global
    pi ambiguity at acquisition is a real property of two-way carrier
    alignment and is resolved in hardware by a one-time combining check.
    """

    alternate = wrap_phase(combined_half + math.pi)
    keep = torch.abs(wrap_phase(combined_half - reference)) <= torch.abs(
        wrap_phase(alternate - reference)
    )
    return torch.where(keep, combined_half, alternate)  


def run_two_way_simulation(
    settings: SDRSimulationConfig = SDRSimulationConfig(),
) -> TwoWaySimulationResult:
    """Run reciprocal two-way OTA synchronization for open-loop coherence."""

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
    link_forward = SDRRadioLink(settings, preamble, device, generator)
    link_reverse = SDRRadioLink(
        settings, preamble, device, generator, mirror_of=link_forward
    )
    synchronizer = SDRSynchronizer(settings, preamble)
    # Averaging two independent directional measurements halves the noise.
    measurement_noise = 0.5 * _measurement_covariance(settings, preamble, device)

    interval_samples = int(round(settings.sync_interval * settings.sample_rate))
    white_fm_phase_variance = settings.phase_noise_std_rad**2 * interval_samples
    flicker = _FlickerFrequencyNoise(
        settings.flicker_frequency_std_hz,
        settings.sync_interval,
        settings.num_iterations * settings.sync_interval,
        device,
        generator,
    )
    ekf = PhaseFrequencyEKF(
        settings.sync_interval,
        2.0 * oscillator_covariance
        + torch.diag(
            torch.tensor(
                [white_fm_phase_variance, flicker.innovation_variance],
                dtype=REAL_DTYPE,
                device=device,
            )
        ),
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

    chain_bias = math.radians(settings.twoway_chain_asymmetry_deg)
    capture_samples = link_forward.input_length + link_forward.l_tot - 1
    remainder_samples = max(0, interval_samples - 2 * capture_samples)
    airtime_fraction = (
        2.0 * capture_samples / (settings.sync_interval * settings.sample_rate)
    )
    pending_corrections: dict[int, torch.Tensor] = {}
    carried_lo_walk = torch.zeros((), dtype=REAL_DTYPE, device=device)
    flicker_previous = torch.zeros((), dtype=REAL_DTYPE, device=device)
    slave_frequency_corrections = torch.zeros((), dtype=REAL_DTYPE, device=device)
    correction_has_loaded = False
    acquired = False
    settled_corrections = 0
    pi_calibrated = False

    history: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "true_phase",
            "measured_phase",
            "estimated_phase",
            "true_frequency",
            "physical_relative_frequency",
            "estimated_frequency",
            "phase_error",
            "frequency_error",
            "post_correction_phase",
            "post_correction_frequency",
            "coherent_gain",
            "detected",
            "correction_active",
            "calibrated",
        )
    }

    for iteration in range(settings.num_iterations):
        master.step()
        slave.step()
        master.state[0] = wrap_phase(master.state[0] + carried_lo_walk)
        flicker_now = flicker.step()
        master.state[1] = master.state[1] + (flicker_now - flicker_previous)
        flicker_previous = flicker_now

        due_correction = pending_corrections.pop(iteration, None)
        if due_correction is not None:
            slave.apply_correction(due_correction)
            slave_frequency_corrections = (
                slave_frequency_corrections + due_correction[1]
            )
            correction_has_loaded = True

        # One-time pi-ambiguity calibration: a deployment checks once after
        # lock whether the pair combines destructively (a single coarse
        # power measurement) and flips the NCO by pi if so.
        if correction_has_loaded and not pi_calibrated:
            settled_corrections += 1
            if settled_corrections >= 3:
                if torch.cos(master.state[0] - slave.state[0]) < 0.0:
                    # The half-difference measurement is invariant under pi
                    # shifts, so the filter needs no reset: its near-zero
                    # state re-attaches to the true branch after the flip.
                    slave.apply_correction(
                        torch.tensor(
                            [math.pi, 0.0], dtype=REAL_DTYPE, device=device
                        )
                    )
                pi_calibrated = True

        # Physical (correction-free) slave oscillator: what the hardware
        # would do with no synchronization running at all.
        physical_slave_frequency = slave.state[1] - slave_frequency_corrections
        if settings.sample_clock_offset_ppm is not None:
            sfo_forward = settings.sample_clock_offset_ppm
        else:
            sfo_forward = float(
                (physical_slave_frequency - master.state[1]).item()
                / (2.0 * math.pi * settings.carrier_frequency_hz)
                * 1e6
            )
        history["physical_relative_frequency"].append(
            (master.state[1] - physical_slave_frequency).clone()
        )

        relative_state = master.state - slave.state
        # Forward frame (master transmits); its intra-frame LO walk becomes
        # part of the true state before the reverse frame is exchanged, so
        # the two directions see one continuous noise process.
        capture_forward = link_forward.capture(master, slave, iteration, sfo_forward)
        master.state[0] = wrap_phase(master.state[0] + capture_forward.lo_walk_end)
        # Half-duplex turnaround: the radios need real time to switch from
        # receive to transmit, and both oscillators keep running across the
        # gap - deterministic advance at their current frequencies plus the
        # white-FM walk accumulated during it. (The channel is held static
        # across the gap; reciprocity aging under mobility remains a known
        # simplification.)
        turnaround = settings.tdd_turnaround_s
        if turnaround > 0.0:
            walk_std = settings.phase_noise_std_rad * math.sqrt(
                settings.sample_rate * turnaround
            )
            for oscillator in (master, slave):
                wander = torch.zeros((), dtype=REAL_DTYPE, device=device)
                if walk_std > 0.0:
                    wander = (
                        torch.randn(
                            (), dtype=REAL_DTYPE, device=device, generator=generator
                        )
                        * walk_std
                    )
                oscillator.state[0] = wrap_phase(
                    oscillator.state[0] + oscillator.state[1] * turnaround + wander
                )
        capture_reverse = link_reverse.capture(slave, master, iteration, -sfo_forward)
        slave.state[0] = wrap_phase(slave.state[0] + capture_reverse.lo_walk_end)

        if settings.phase_noise_std_rad > 0.0 and remainder_samples > 0:
            carried_lo_walk = torch.randn(
                (), dtype=REAL_DTYPE, device=device, generator=generator
            ) * (settings.phase_noise_std_rad * math.sqrt(remainder_samples))
        else:
            carried_lo_walk = torch.zeros((), dtype=REAL_DTYPE, device=device)
        forward = synchronizer.estimate(capture_forward.samples)
        reverse = synchronizer.estimate(capture_reverse.samples)
        detected = forward.detected and reverse.detected
        # Reciprocity: the channel phase is common to both directions and
        # cancels in the half-difference, leaving the oscillator offset.
        combined_frequency = (forward.frequency - reverse.frequency) / 2.0
        # The reverse frame was measured one turnaround later, so the raw
        # half-difference contains an extra omega*turnaround/2 of drift. A
        # real receiver knows its own turnaround and removes it with its
        # measured CFO; the irreducible residue is the CFO-estimate error
        # times the gap (plus the random walk across it).
        combined_half = wrap_phase(
            wrap_phase(forward.phase - reverse.phase) / 2.0
            + chain_bias
            - combined_frequency * settings.tdd_turnaround_s / 2.0
        )

        ekf.predict()
        if due_correction is not None:
            ekf.reset_after_correction(due_correction)

        if detected:
            if not acquired:
                phase_measurement = _pick_half_phase(
                    combined_half, torch.zeros_like(combined_half)
                )
                ekf.state = torch.stack((phase_measurement, combined_frequency))
                ekf.covariance = torch.diag(
                    torch.stack((measurement_noise[0, 0], measurement_noise[2, 2]))
                )
                acquired = True
            else:
                phase_measurement = _pick_half_phase(
                    combined_half, wrap_phase(ekf.state[0])
                )
                ekf.update(
                    torch.stack(
                        (
                            torch.cos(phase_measurement),
                            torch.sin(phase_measurement),
                            combined_frequency,
                        )
                    )
                )
            predicted = ekf.state.clone()
            for _ in range(settings.correction_latency_intervals):
                predicted = ekf.transition @ predicted
            correction = _quantize_correction(predicted, settings)
        else:
            phase_measurement = combined_half
            correction = None

        true_phase = wrap_phase(relative_state[0])
        estimated_phase = wrap_phase(ekf.state[0])
        history["true_phase"].append(true_phase.clone())
        history["measured_phase"].append(phase_measurement.clone())
        history["estimated_phase"].append(estimated_phase.clone())
        history["true_frequency"].append(relative_state[1].clone())
        history["estimated_frequency"].append(ekf.state[1].clone())
        history["phase_error"].append(wrap_phase(true_phase - estimated_phase))
        history["frequency_error"].append(
            (relative_state[1] - ekf.state[1]).clone()
        )
        history["detected"].append(
            torch.tensor(detected, dtype=torch.bool, device=device)
        )

        if correction is not None:
            if settings.correction_latency_intervals == 0:
                slave.apply_correction(correction)
                ekf.reset_after_correction(correction)
                slave_frequency_corrections = (
                    slave_frequency_corrections + correction[1]
                )
                correction_has_loaded = True
            else:
                pending_corrections[
                    iteration + settings.correction_latency_intervals
                ] = correction

        history["correction_active"].append(
            torch.tensor(correction_has_loaded, dtype=torch.bool, device=device)
        )
        history["calibrated"].append(
            torch.tensor(pi_calibrated, dtype=torch.bool, device=device)
        )
        residual_state = master.state - slave.state 
        residual_phase = wrap_phase(residual_state[0])
        history["post_correction_phase"].append(residual_phase.clone())
        history["post_correction_frequency"].append(residual_state[1].clone())
        # Two-element open-loop combining toward a passive target:
        # |1 + exp(j*residual)|^2 / 4 = cos^2(residual / 2).
        history["coherent_gain"].append(torch.cos(residual_phase / 2.0).square())

    return TwoWaySimulationResult(
        **{
            name: torch.stack(values).detach().cpu() for name, values in history.items()
        },
        airtime_fraction=airtime_fraction,
        device=device,
    )
if __name__ == "__main__":
    result = run_two_way_simulation(SDRSimulationConfig())
    print(f"detection rate     : {result.detection_rate:.2f}")
    print(f"steady phase rms   : {result.steady_state_phase_rms:.4f} rad")
    print(f"mean coherent gain : {result.mean_coherent_gain * 100:.1f} %")
    print(f"airtime fraction   : {result.airtime_fraction * 100:.2f} %")
    print(f"final freq error   : {result.final_frequency_error_hz:.3f} Hz")
