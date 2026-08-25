# What is in this folder

**If you're here to read the research: go to `RESEARCH_OVERVIEW/` and
open its `README.md`.** Everything is reachable from there through
five date-numbered links, with a day-by-day timeline in
`INDEX_BY_DATE.md` and one-page technical summaries.

This folder itself is the working directory and is deliberately not
tidied — dozens of study scripts import each other by location, and
the 150+ regression tests depend on files staying put. Map of what's
here:

| group | examples | what it is |
|---|---|---|
| Pre-existing project (before Aug 2026 campaign) | `simulation.py`, `ota_sync/`, `hybrid_calibration/`, `detection/`, `teaching_*.pdf`, `project_presentation.*` | the original simulator and docs |
| Campaign study scripts (Aug 14–18) | `gating_*.py`, `*_scaling_study.py`, `coast_law.py`, `phase_diagram_round2.py`, `clutter_sync_*.py`, `metrics*.py`, `snr_law_check.py`, `pilot_lever_check.py`, `observability_*.py`, `pi_ambiguity_*.py`, `fig_*.py` | runnable experiments; results summarized in `RESEARCH_OVERVIEW` link 01 |
| Their data | `*_cache.json`, `*_runs.json`, `*.log` | resumable run caches — safe to ignore when reading |
| Tests | `tests/` | regression suite for everything above (run: `.venv/../python -m pytest tests/`) |
| `figures/studies/` | 39 figures + the campaign reports | link 01 in the overview |
| `phase_sync_idea/` | self-contained deep-experiment sandbox (Aug 18–24) | link 02 |
| `lit_review/` | literature-audit tool + query ledger | link 03 |
| `publishable_work_phase_sync/` | abstracts + slide deck | link 04 |
| `topology_selection/` | the current push and lead result (Aug 24) | link 05 |
| `RESEARCH_OVERVIEW/` | **start here** | the front door |
