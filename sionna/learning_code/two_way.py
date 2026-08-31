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
    ratio = 1.0 + ppm * 1e-6 
    output_indx = torch.arange(
        samples.numel(), dtype=REAL_DTYPE, device=samples.device 
    )