# Abstract — Environment-Limited Synchronization

**Environment-Limited Synchronization: Distributed Arrays That
Phase-Lock on Ordinary Traffic**

The rate at which a distributed transmit array must perform dedicated
synchronization exchanges has always been priced by oscillator
quality: worse clocks, more frequent synchronization, and a
per-station airtime cost that saturates the shared channel as arrays
grow. We show that this design rule uses the wrong variable. When
each station's tracking filter consumes the phase of ordinary data
and sensing transmissions — signals the array radiates regardless —
oscillator drift is tracked continuously at zero marginal airtime,
and the required rate of dedicated two-way exchanges is set instead
by the propagation environment's coherence time. The architecture
rests on an impossibility result: a one-way observation mixes
oscillator phase with propagation phase, and we prove no sequence of
one-way observations can separate them — the observability Gramian's
null space lies exactly along that decomposition. Sparse two-way
exchanges restore the missing degree of freedom, and we derive the
admissible spacing in closed form, π·f_D·T·K < b, containing the
Doppler frequency and no oscillator parameter; measured collapse
boundaries match this zero-fitted-constant prediction at both tested
cadences. Waveform-level simulations with full oscillator and RF
impairments then yield two results that contradict conventional
intuition. First, making dedicated exchanges rarer *improves*
accuracy — from 129 mrad of residual phase error at 67%
synchronization airtime to 86 mrad at 0.42%, because each exchange
itself injects multipath measurement noise — and second, additional
free observations never degrade tracking in any tested environment,
refuting the natural conjecture that channel dynamics eventually
masquerade as oscillator drift at high observation rates. The
consequence at scale: residual phase error stays flat from 2 to 64
stations (55–108 mrad at ≥99.7% coherent gain) on 0.5–30%
synchronization airtime, while every dedicated-signaling scheme
tested is incoherent or frame-saturated at 64 stations.
Synchronization cadence, in this architecture, is a property of the
environment, not of the hardware.

*(All numbers from the experiments in `../phase_sync_idea/` — see its
README for per-experiment provenance. Simulation only; no hardware.)*
