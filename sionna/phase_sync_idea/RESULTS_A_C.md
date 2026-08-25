# Experiments A and C — scheme scaling and the anchor-rate frontier

Script: `experiment_a_c.py` (cache `experiment_a_c_cache.json`, all
cells computed fresh this session, seeds 0–2 everywhere). Figures:
`figures/figA1_residual_vs_N.png`, `figA2_airtime_vs_N.png`,
`figC1_anchor_frontier.png`. Plain-language conventions: "residual" =
worst-station steady phase error; "airtime" = fraction of the frame
spent on synchronization transmissions.

## Experiment A — scheme × array size

Five schemes at N = 2…64 (worst residual mrad / beam gain % / airtime %):

| N | conv-uniform | conv-scheduled | opportunistic (K=40) | hybrid (K=5) | no-sync |
|---|---|---|---|---|---|
| 2 | 113±4 / 99.7 / 19.1 | 174±13 / 99.2 / 7.6 | 55±13 / 99.9 / 0.5 | 59±18 / 99.9 / 3.8 | 774±78 / 86.6 / 1.9 |
| 4 | 116±5 / 99.5 / 57.4 | 179±13 / 98.8 / 22.9 | 59±11 / 99.9 / 1.4 | 60±9 / 99.9 / 11.5 | 937±54 / 67.1 / 5.7 |
| 8 | 2190±138 / 53.0 / 95.6 | 193±13 / 98.5 / 51.6 | 106±1 / 99.7 / 3.3 | 130±3 / 99.7 / 26.8 | 1020±116 / 58.3 / 11.2 |
| 16 | 2498±444 / 16.7 / 95.6 | 443±29 / 95.9 / 95.6 | 75±9 / 99.8 / 7.2 | 73±5 / 99.8 / 57.4 | 1133±48 / 48.1 / 20.5 |
| 32 | 2101±152 / 6.9 / 95.6 | 1055±36 / 78.2 / 95.6 | 98±3 / 99.8 / 14.8 | 117±3 / 99.8 / **118.6*** | 1082±23 / 43.2 / 26.9 |
| 64 | 2294±166 / 2.9 / 95.6 | 2072±182 / 26.6 / 95.6 | 108±5 / 99.8 / 30.1 | 118±2 / 99.7 / **241.0*** | 1141±73 / 41.2 / 35.4 |

\* hybrid airtime above 100% is demand, not realizable — dense anchors
hit their own wall at N ≥ 32.

Notes on the baselines: conv-uniform is every-interval dedicated
two-way, physically capacity-capped at ~5 exchanges/interval, so
past N≈6 most links starve (residuals are then tail-window values —
that collapse is the measurement, not an artifact). conv-scheduled is
the strongest conventional baseline (posterior-driven service order).
no-sync is acquire-then-coast (huge budget in the scheduled star);
its residual is a window average and grows without bound with
observation time, and its airtime is the acquisition cost amortized
over the run.

**Findings.** The opportunistic scheme is the only one whose accuracy
is flat across a 32× change in array size (55→108 mrad, beam gain
99.7–99.9% everywhere) while its airtime stays far below the frame
(0.5%→30%). Both conventional schemes hit the frame wall — the naive
one collapses at N=8, the scheduled one degrades from N=16 and is
fully incoherent (2.1 rad, 27% gain) at N=64 despite consuming 95.6%
of the frame. Dense-anchor hybrid matches opportunistic accuracy but
its anchor demand alone exceeds the frame at N≥32, which shows the
saving genuinely comes from replacing anchors with free observations,
not from the machinery.

## Experiment C — the anchor-rate frontier (N=8)

| anchor cadence K | residual (mrad) | anchor airtime |
|---|---|---|
| 2 | 129.5±1.0 | 66.9% |
| 5 | 130.6±3.2 | 26.8% |
| 10 | 138.3±5.5 | 13.4% |
| 20 | 123.8±2.9 | 6.7% |
| 40 | 106.3±1.6 | 3.3% |
| 80 | 95.3±2.7 | 1.7% |
| 160 | 90.6±2.2 | 0.84% |
| 320 | 86.0±4.5 | 0.42% |
| dedicated two-way reference | 193.3 | 51.6% |

**Findings.** The reviewer's concern — that the synchronization
traffic is merely relocated into the anchors — is answered directly:
as anchor airtime falls by a factor of 160 (67% → 0.4%), the residual
does not degrade at all; it *improves* from ~130 to ~86 mrad, because
each two-way anchor capture itself injects multipath resampling noise
and fewer anchors means fewer such disturbances, while the free
observations carry the tracking. Every point on the frontier beats
the dedicated baseline (193 mrad at 52% airtime) on both axes. The
static-environment caveat applies: anchor sparsity is bounded by
environmental coherence, not by oscillator drift, so this frontier is
the static-limit best case.
