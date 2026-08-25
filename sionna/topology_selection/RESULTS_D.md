# Direction D — dynamic channel-aware topology re-selection

Script: `dirD_dynamic.py` (cache `dirD_cache.json`, 39 runs). Figures:
`figures/dirD_gain_trajectory.png`, `figures/dirD_gain_per_seed.png`.
Testbed: N=8 nodes, 28 candidate two-way edges, active set = a 7-edge
spanning tree serviced every interval — equal sync airtime for every
strategy in every run. Full physical layer (waveform captures, TDL
channels, oscillator noise with flicker, timing jitter, quantized
corrections), randomized per-node initial frequency offsets, seeds
0–2. Blockage episode: the two highest-betweenness edges of the
initial max-SNR tree lose 20 dB of SNR (both directions) for a window
of intervals.

## Control-law note (measured decision, recorded per discipline)

The symmetric half/half consensus law used by the sandbox's
`openloop_topology_study.py` harness does **not** converge at N=8 in
150 intervals — steady gain 0.12–0.39 on both a max-SNR tree and a
star, in that harness and in this adaptation (that harness's own
campaigns are marked not-yet-run, so it carries no validated N=8
result — flagged for its owner). This matches the repo's Phase-1
"consensus tax" finding, whose measured-stable alternative — the
directed elected-root tree law, each child correcting fully toward
its parent — converges here to 0.89–0.95 steady gain with 1–3 branch
flips and zero realigns. All Direction-D results use the directed
law; the study's question is which links get airtime, not which
control law wins.

## Main scenario (episode intervals 100–200, mean over seeds 0–2)

| strategy | pre | episode | post | intervals below 0.90 | switches |
|---|---|---|---|---|---|
| static | 0.962 | 0.683 | 0.912 | 85.0 | 0 |
| channel-aware | 0.962 | **0.938** | 0.960 | 21.7 | 1.0 |
| gain-aware | 0.962 | 0.845 | 0.975 | 35.3 | 1.3 |
| oracle | 0.962 | 0.950 | 0.943 | 21.3 | 4.0 |

- **The measured-SNR (channel-aware) policy captures 95% of the
  oracle's episode-gain advantage** ((0.938−0.683)/(0.950−0.683)),
  with a single switch fired 4–5 intervals after blockage onset, using
  only the detection metric the sync exchanges already compute — no
  oracle information.
- The gain-aware policy captures 61%. The gap is trigger *timing*,
  not decision quality: its trigger waits for observed harm
  (branch-flip chatter, below) while the channel trigger reads the
  degraded link quality directly — a leading indicator beats a
  lagging one. Prediction P2 said the two would be near-identical
  here; measured: channel-aware is better. Miss recorded.
- Only one of the two blocked edges was swapped by the adaptive
  policies: the other's survey SNR was high enough that −20 dB left
  it detectable with metric above threshold — partial degradation
  tolerated rather than churned, which is the desired behavior.
- Switching costs charged throughout: incoming edges re-acquire
  (settling exchanges issue no corrections), subtrees recomputed,
  branch check gated 8 intervals, all inside the same 7-edge airtime.

## Mechanism finding: the dominant harm channel is branch-check churn

Static runs through the episode log **17–21 branch π-flips and 6–10
filter realigns** versus 1–3 flips and 0 realigns in no-disturbance
runs. The degraded edges' noisy corrections repeatedly push their pair
phases across the ±π/2 branch line; each 1-bit check response inverts
a whole subtree, cratering coherent gain, and repeated flips trigger
filter re-acquisition on a still-noisy link. This discrete cascade —
not smooth measurement-noise inflation — is why a first version of
the gain-aware policy (believed variance = measurement + coasting
variance) predicted near-zero harm and never switched; it was fixed
by charging chattering edges the ambiguity-scale variance their own
observable check bits imply ((π/2)² per recent flip, no fitted
constants). Both the failure and the fix are part of this record.

## Control: no-disturbance runs (3 strategies × 3 seeds)

**Zero false switches in all nine runs** (prediction P3 confirmed);
adaptive trajectories bit-match static when nothing is wrong. Baseline
honesty note: the depth-5 max-SNR tree is itself imperfect —
no-disturbance runs spend 19–84 of 300 intervals below 0.90 (seed-
dependent dips to ~0.6–0.8), the price of a deep tree under the
directed law.

## Episode-length boundary (gain-aware minus static, mean gain over episode + 40 intervals)

| episode length (intervals) | ΔG mean | per-seed |
|---|---|---|
| 12 | +0.012 | +0.000 / −0.112 / +0.147 |
| 25 | +0.007 | +0.000 / −0.068 / +0.090 |
| 50 | +0.087 | −0.036 / +0.090 / +0.207 |
| 100 | +0.137 | +0.111 / +0.150 / +0.150 |

Re-selection reliably pays for episodes of roughly **50+ intervals
(≈2.5 s at the 50 ms interval)**; at 12–25 intervals it is a wash with
seed-dependent sign — one seed lost 0.112 by switching for a
12-interval blockage (the switching transient outlived the episode).
Prediction P4 said 10–15 intervals; the measured boundary is larger
because it was measured with the *lagging* (flip-evidence) trigger —
with the channel trigger's 4–5-interval reaction the boundary should
sit lower; not measured here, noted as the natural follow-up. One
more honest observation from the sweep: in one short-episode seed the
*static* run failed to re-lock promptly after the blockage lifted
(post-window gain 0.435 vs the switcher's 0.680) — flip-churn
aftermath can outlast the disturbance itself.

## Prediction scorecard

- P1 (static loses through episode, adaptive recovers in trigger+settle,
  oracle fastest): **confirmed** (0.683 vs 0.938/0.845 vs 0.950).
- P2 (channel ≈ gain here): **wrong** — channel-aware wins by 9 points;
  leading vs lagging trigger. Recorded as a miss.
- P3 (zero false triggers): **confirmed** (0/9 runs).
- P4 (boundary at 10–15 intervals): **partially wrong** — measured
  ~50 intervals with the lagging trigger; see caveat above.

## Scope notes

Per-edge blockage is implemented as noise-floor scaling on both
directions of the already-built radio links (the estimation chain sees
only SNR, so this is equivalent to signal attenuation at this layer);
per-edge *motion* was not exercised (the blockage scenario sufficed
for the re-selection question; the injected-channel hooks remain the
route if needed). Strategies consume different RNG streams once they
diverge — unavoidable for adaptive policies; mitigated by 3 seeds
with spread reported.
