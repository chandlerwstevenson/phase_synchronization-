# topology_selection — coherence-aware synchronization topology selection

Research question (from advisor feedback, 2026-08-24): given a
distributed array and a limited synchronization-airtime budget, select
the synchronization links, participating transmitters, and schedule
that maximize coherent beamforming gain — with link weights set by the
actual wireless environment.

## Conventions (keep the directory clean)

- No copied packages. Physics machinery is imported from
  `../phase_sync_idea/` (the self-contained sync sandbox) via
  `sys.path.insert(0, "../phase_sync_idea")` at the top of each script.
- One file prefix per direction; one results file per direction:
  - `dirA_*` — static airtime-constrained edge selection + baselines → `RESULTS_A.md`
  - `dirB_*` — coherent-gain objective vs phase-MSE objective → `RESULTS_B.md`
  - `dirC_*` — joint node participation + edge selection → `RESULTS_C.md`
  - `dirD_*` — dynamic channel-aware re-selection → `RESULTS_D.md`
- Figures: plain default matplotlib, no in-axes annotations, into
  `figures/` here.
- Literature: every search through `../lit_review/audit.py` (ledgered);
  claims `topo-A` … `topo-D`.
- Discipline (four artifact retractions taught this): predictions
  stated before measurements; surprising results get a discriminating
  control before a mechanism story; ≥3 seeds; randomized per-station
  frequency offsets (never the 1500·s/(N−1) grid).

## The central object

Phase-error covariance of a selected two-way edge set E gives pairwise
error variances via effective resistance (theory verified in
`../phase_sync_idea/openloop_graph_theory.py`; the formulation itself
is prior art — Barooah-Hespanha, Howard et al. — cite, don't claim).
Expected coherent gain with amplitudes a_i:

    E[G(E)] ≈ Σ_ik a_i a_k · exp(−r₂ · R_ik(E) / 2) / (Σ a_i)²

where R_ik(E) is the effective resistance between i and k over the
selected edges. The claimed-novel layer is ONLY: this beamforming
objective (not phase MSE) driving the selection, under an airtime
budget, with SNR-dependent edge weights — and its node-selection and
dynamic extensions.

## Baselines every direction compares against

complete graph / star / ring / minimum-spanning-tree by airtime cost /
max-SNR tree / spectral (max algebraic connectivity) — versus
gain-per-airtime selection.

## Known nearest prior art (do not re-claim; positioning targets)

Larsson "Massive Synchrony" arXiv:2401.11730 (topology vs calibration
error, lines/rings/2-D); Ngo & Larsson arXiv:2509.03722 (calibration
topology density vs spectral efficiency — denser can be worse);
Ouassal/Yan/Nanzer arXiv:1911.10076 (consensus frequency alignment on
graphs); Brown & Poor 2008 (round-trip sync overhead tradeoff);
Howard et al. (Fisher information = weighted graph Laplacian);
Ghosh-Boyd (effective-resistance minimization as convex problem);
Joshi-Boyd (sensor selection); Freris-Graham-Kumar (identifiability).
