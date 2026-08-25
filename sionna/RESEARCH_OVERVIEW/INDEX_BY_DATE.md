# Chronological index — what was worked on, when

Most recent at the bottom. Each row: what happened, where it lives
(folder numbers refer to the dated links in this directory), and the
one file to read for it.

| date (2026) | what happened | where | read this |
|---|---|---|---|
| Aug 14 | Posterior-gated membership built and measured; gain claim wrong, detection claim right (77→98%) | 01 | `EXPERIMENT_SUMMARY.md` §1–2 |
| Aug 14 | Scheduling/contention studies, 1-bit membership (= oracle, exact identity), hybrid two-tier combiner | 01 | `EXPERIMENT_SUMMARY.md` §3, 9 |
| Aug 14 | Doppler coast term killed (seed noise); coast-time law exact (99.5%); blind phase-diagram rounds (1/10 → 22/32) | 01 | `EXPERIMENT_SUMMARY.md` §4–7 |
| Aug 14 | Clutter/in-band sync idea (user's): ~16× airtime cut at better residual | 01 | `EXPERIMENT_SUMMARY.md` §8 |
| Aug 15 | Everything re-verified fresh + realism upgrades (ray tracing, real OFDM waveform, environments) | 01 | `EXPERIMENT_SUMMARY.md` §10 |
| Aug 15 | Multi-metric comparison (detection vs throughput vs net rate); array-size scaling of everything | 01 | `EXPERIMENT_SUMMARY.md` §11–12 |
| Aug 15 | All 39 figures generated (later redone plain per feedback); figure catalog | 01 | `FIGURES.md` |
| Aug 16 | SNR law + pilot-length lever measured (airtime 19→3% free; pilots ~6× too long) | 01 | `SNR_LAW.md` |
| Aug 17 | Professor-grade weekly update + slide outline | 01 | `WEEKLY_UPDATE.md`, `WEEKLY_SLIDES.md` |
| Aug 18 | Reviewer-driven rigor round: observability/gauge proof, π-ambiguity + 1-bit proof, interior-optimum KILLED (transient), OFDM gap closure | 02 | `RESULTS_DOMINANCE.md` context; `pi_ambiguity_analysis.py` docstring |
| Aug 19 | Reviewer's five experiments on the in-band architecture (coherence regimes, ablations, capture model, scaling, frontier) | 02 | `README.md` there (verdict table) |
| Aug 24 | Literature-audit tool built (ledgered queries, citation walks) after review misses | 03 | `ledger.json`, `audit.py` docstring |
| Aug 24 | Resampling-floor claim KILLED by own controls; corrected to fractional-resampling term (10–25 mrad); all docs corrected in place | 02 | `RESULTS_DOMINANCE.md`, `resampling_law.py` |
| Aug 24 | Scheduling-reversal proved both-halves (only the in-band architecture reverses); six-term floor budget verified additive | 02 | `RESULTS_reversal.md`, `RESULTS_floors.md` |
| Aug 24 | Graph/gauge theory proved (3 theorems) then gated out by literature (Freris–Graham–Kumar etc.) — kept as cited foundations | 02 + 03 | `RESULTS_graph_theory.md` |
| Aug 24 | Topology testbed campaigns: resistance law confirmed 5/6; **quantized winding states discovered** (cycle sums = 2π·w exactly) | 02 | `RESULTS_topology.md` |
| Aug 24 | Topology-selection push (advisor's directions A–D): objective-mismatch theorem, participation window, dynamic re-selection (95% of oracle) | 05 | `RESULTS_B.md`, `RESULTS_C.md`, `RESULTS_D.md` |
| Aug 24 | **THE CURRENT LEAD RESULT — the three-protocol fork**: three mechanisms, three origins; topology ranking inverts across protocols; non-separability thesis earned | 05 | `RESULTS_A2.md`, `ABSTRACTS.md` (last entry) |
| Aug 24 | One-page summaries written (whole project; topology-protocol result alone) | here | `ONE_PAGE_SUMMARY.md`, `ONE_PAGE_TOPOLOGY_PROTOCOL.md` |

**Where you left off:** the three-protocol fork verdict (folder 05,
`RESULTS_A2.md`) and its one-pager. The identified next step, unanimous
across reviews: a 2–4 software-defined-radio hardware demonstration,
with the topology-ranking inversion as the target observable.
