# RESEARCH OVERVIEW — start here

Everything from the phase-synchronization research campaign, reachable
from this one folder. The five numbered entries are links to the real
folders (nothing was moved, so all code still runs in place), named
`01…05` in chronological order — higher number = more recent work.

- **What did I last work on?** → `INDEX_BY_DATE.md` (day-by-day
  timeline; the bottom row is where you left off).
- **One-page technical summaries** → `ONE_PAGE_SUMMARY.md` (whole
  project) and `ONE_PAGE_TOPOLOGY_PROTOCOL.md` (the current lead
  result).

---

## If you read only three things

1. **`05_AUG24_CURRENT__topology_selection_results/ABSTRACTS.md`** — five abstracts; the last
   one ("A2 — the earned abstract") is the current strongest result:
   *synchronization topology and update protocol are not separable*,
   with three mechanisms of three different origins (physical /
   numerical / topological), established by a pre-registered
   three-protocol experiment.
2. **`05_AUG24_CURRENT__topology_selection_results/RESULTS_A2.md`** — the decisive experiment
   behind that abstract: 118 cells, predictions printed before runs,
   the verdict table, and the topology-ranking inversion (star worst
   under simultaneous updates → best under directed).
3. **`01_AUG14-17_first_campaign__scheduling_membership_inband_sync__reports_and_figures/WEEKLY_UPDATE.md`** — the professor-grade
   summary of the earlier campaign (scheduling, membership, in-band
   sync, SNR analysis), with corrections where later experiments
   overturned earlier attributions.

## The story, in order (what happened and where it lives)

**Act 1 — the broad campaign** (`01_AUG14-17_first_campaign__scheduling_membership_inband_sync__reports_and_figures/`):
scheduling, membership, in-band ("piggyback") synchronization,
metrics, scaling. Read `EXPERIMENT_SUMMARY.md` (per-experiment
claims → protocol → result, including the correction blocks),
`THEORY.md` (the math), `SNR_LAW.md` (the answer to the SNR
question), `FIGURES.md` (index of the 39 figures in that folder).
Note: several early "discoveries" here were later retracted after
controls — the correction notes in each file say exactly what died
and why. That trail is kept deliberately.

**Act 2 — open-loop sync fundamentals** (`02_AUG18-24_deep_experiments__theory_floors_testbeds__phase_sync_idea/`):
the self-contained sandbox where the deeper experiments ran. Key
reading, in order:
- `RESULTS_graph_theory.md` — three proved theorems on measurement
  graphs (identifiability, branch ambiguity, resistance law). The
  literature audit later showed the first and third are pre-owned by
  adjacent fields (network clock sync, estimation on graphs) — kept
  as internal foundations with proper citations.
- `RESULTS_topology.md` — the N-node testbed campaigns; highlight:
  settled wrong-lock states carry exactly quantized winding numbers.
- `RESULTS_floors.md` — the six-term physical floor budget of one
  two-way link, all terms derived with zero fitted constants,
  additivity verified.
- `RESULTS_DOMINANCE.md` + `resampling_law.py` — the controls that
  killed the "multipath resampling floor" claim and what replaced it.
- `RESULTS_reversal.md` — proof + measurement that "sync less often,
  get better accuracy" exists only in the in-band architecture, never
  in a conventional loop.
- `RESULTS_B_D.md`, `RESULTS_E.md`, `RESULTS_A_C.md` — the reviewer's
  requested experiments on the in-band architecture (coherence-time
  regimes, ablations, capture-length model, scaling, exchange-rate
  frontier).

**Act 3 — topology selection** (`05_AUG24_CURRENT__topology_selection_results/`): the current
push. `README.md` there states the research question and conventions.
Results per direction: `RESULTS_A.md` (the original stability
inversion), `RESULTS_A2.md` (the three-protocol fork — the decisive
one), `RESULTS_B.md` (minimum phase error is the wrong objective —
exact, enumerated), `RESULTS_C.md` (which radios deserve sync — the
participation window), `RESULTS_D.md` (dynamic re-selection under
blockage — 95% of oracle with no oracle). `ABSTRACTS.md` collects the
five abstracts. Figures in `figures/`.

**Paper materials** (`04_AUG24_paper_materials__abstracts_and_slides/`): the abstract iterations for
the in-band synchronization paper, including one retracted version
kept with its retraction banner (`ABSTRACT_resampling_lead.md` —
worth reading as a case study in the process working), and a 14-slide
LaTeX deck (`slides.pdf`) for the earlier framing — note it predates
the topology results.

**Literature audit** (`03_AUG24_literature_audit_tool_and_ledger/`): the search tool
(`audit.py`) and `ledger.json` — every query ever run, every hit,
every verdict, including the zero-hit queries that serve as coverage
evidence for novelty claims.

## Current standing (2026-08-24)

- **Strongest result:** the A2 non-separability finding (three
  mechanisms, three origins; topology ranking inverts across
  protocols; only directed protocols scale to N=16).
- **Strongest supporting results:** B's objective-mismatch theorem
  (exact), C's participation window, D's adaptive policy, the
  quantized-winding measurements, the floor budget.
- **Known competition:** the Larsson group owns topology-vs-error
  analysis and has announced a sync-scheduling follow-up — speed
  matters. Full positioning in the ledger and in
  `05_AUG24_CURRENT__topology_selection_results/README.md`.
- **Everything is simulation.** The consistently identified next step
  across every review: a small hardware demonstration (2–4
  software-defined radios).

## Honesty conventions used throughout

Predictions registered before measurements; every surprising result
required a discriminating control before being reported (four early
findings were retracted by exactly this process, and the retractions
are preserved in place); corrected documents carry dated correction
blocks rather than silent edits; all numbers in abstracts trace to a
results file.
