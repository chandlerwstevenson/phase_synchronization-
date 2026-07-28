# SDR-like OTA synchronization with NVIDIA Sionna

This directory ports the one-way master/slave synchronization loop in
`../reg_python` to sampled complex IQ. The default simulation is built around
Sionna PHY's 3GPP TR 38.901 tapped-delay-line channel, time-domain channel
application, and complex AWGN blocks. The original ideal tone-plus-AWGN model is
still available for comparison.

## What the SDR model includes

- A framed preamble with a repeated short training field and CP-protected long
  Zadoff-Chu training fields.
- Packet detection, sample timing acquisition, wide-range coarse CFO, fine CFO,
  and phase estimation from received IQ only.
- Independent master/slave oscillator phase and frequency random walks.
- Sionna `TDL` channel snapshots with fractional-delay multipath and optional
  Doppler, converted to discrete taps with `cir_to_time_channel` and applied by
  `ApplyTimeChannel`.
- Sionna complex `AWGN` at a specified received SNR.
- Sample-clock offset and IQ resampling, intra-frame phase noise, timing jitter,
  PA limiting, DAC/ADC quantization, AGC, IQ gain/phase imbalance, clipping, and
  receiver DC offset.
- An iterated EKF and quantized phase/frequency corrections, similar to an SDR
  NCO control loop.

The default profile is a stationary 915 MHz TDL-D line-of-sight channel, 1 MS/s,
1500 Hz initial CFO, 10 ppm sample-clock error, 12-bit converters, and 20 dB SNR.

## Run

Python 3.11 or newer and PyTorch 2.9.1 or newer are required by Sionna 2.

```bash
python -m pip install -r requirements.txt
python simulation.py
```

Plot phase synchronization, request the full radio diagnostics, or change radio
conditions:

```bash
python simulation.py --plot
python simulation.py --plot-all
python simulation.py --snr-db 8 --cfo-hz 10000 --tdl-model C
python simulation.py --speed-mps 1.0 --sfo-ppm 20 --adc-bits 10
python simulation.py --device cuda --iterations 500
```

Disable the custom RF impairments while retaining the Sionna multipath channel:

```bash
python simulation.py --no-rf-impairments
```

Run the original idealized model with:

```bash
python simulation.py --model ideal --pilot-length 500
```

Tests:

```bash
python -m pip install '.[test]'
python -m pytest
```

## Signal flow

```text
master clock/NCO
  -> STF + LTF IQ frame
  -> PA + DAC
  -> Sionna 3GPP TDL multipath + Sionna AWGN
  -> slave LO + sample-clock offset + phase noise
  -> IQ imbalance + AGC + ADC
  -> packet detection and timing
  -> coarse CFO -> fine CFO -> phase
  -> iterated EKF -> quantized slave NCO correction
```

## Important physical interpretation

A one-way receiver observes `oscillator phase + channel phase`; it cannot identify
those two terms separately without another reference, channel calibration, or a
two-way protocol. The program therefore reports both:

- **OTA phase residual:** what the SDR can measure and drive toward zero.
- **Raw oscillator phase residual:** the simulated clock-only error, shown to
  expose the channel-phase bias rather than hiding it with oracle information.

For a stationary channel, frequency synchronization remains identifiable and the
channel-phase bias is stable. Under mobility, channel Doppler and phase evolution
become part of the measured OTA loop, as they would on hardware.

The coarse CFO range is approximately
`+/- sample_rate / (2 * short_sequence_length)` Hz. TDL channel snapshots are
sampled once per synchronization interval, so select mobility and interval values
that avoid aliasing the channel Doppler.
