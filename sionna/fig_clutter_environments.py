"""Environment-dependence figure (fresh runs, plain default
matplotlib).

Produces figures/studies/clutter_environment_dumbbell.png: all 13
environments from environment_dependence_study.py, piggyback (anchors
every 40) vs the paid two-way baseline, one row per environment.
Seeds mirror the study: 0-2 for statistical channels, 0-1 for
ray-traced scenes, 0 for the loss-charged variant. Cells cache to
fig_cache_clutter_environments.json; delete the cache for a fully
fresh pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from environment_dependence_study import (
    cir_to_frozen_taps,
    rt_station_pair_cir,
    run_cell,
)
from ota_sync import SDRSimulationConfig

CACHE = Path(__file__).resolve().parent / "fig_cache_clutter_environments.json"
FIGDIR = Path(__file__).resolve().parent / "figures" / "studies"
ITERATIONS = 60
CADENCES = (40,)


def save(figure, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / f"{name}.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(path)


def _load() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return {}


def _store(cache: dict) -> None:
    CACHE.write_text(json.dumps(cache, indent=1))


def _cell(cache: dict, key: str, compute) -> dict:
    if key not in cache:
        out = compute()
        cache[key] = {
            "twoway": 1e3 * out["twoway"][0],
            "piggy": 1e3 * out["K40"][0],
            "detect": out["K40"][3],
        }
        _store(cache)
        print(f"  computed {key}: {cache[key]}")
    return cache[key]


def collect(cache: dict) -> list[tuple[str, list[dict]]]:
    rows: list[tuple[str, list[dict]]] = []

    for letter, family in (
        ("D", "LOS Rician (headline)"),
        ("E", "LOS Rician, stronger"),
        ("A", "NLOS Rayleigh"),
        ("B", "NLOS Rayleigh"),
        ("C", "NLOS Rayleigh"),
    ):
        cells = [
            _cell(
                cache, f"tdl{letter}_s{seed}",
                lambda letter=letter, seed=seed: run_cell(
                    SDRSimulationConfig(
                        num_iterations=ITERATIONS, seed=seed,
                        device="cpu", tdl_model=letter,
                    ),
                    CADENCES,
                ),
            )
            for seed in (0, 1, 2)
        ]
        rows.append((f"TDL-{letter} ({family})", cells))

    for spread_ns in (30, 300, 1000):
        cells = [
            _cell(
                cache, f"spread{spread_ns}_s{seed}",
                lambda spread_ns=spread_ns, seed=seed: run_cell(
                    SDRSimulationConfig(
                        num_iterations=ITERATIONS, seed=seed,
                        device="cpu", delay_spread_s=spread_ns * 1e-9,
                    ),
                    CADENCES,
                ),
            )
            for seed in (0, 1, 2)
        ]
        rows.append((f"TDL-D, {spread_ns} ns delay spread", cells))
    # The 100 ns row IS the TDL-D headline cell.
    rows.append(
        ("TDL-D, 100 ns delay spread",
         [cache[f"tdlD_s{seed}"] for seed in (0, 1, 2)])
    )

    for kind, label in (
        ("tworay", "ray-traced two-ray ground"),
        ("urban-los", "ray-traced urban, line of sight"),
        ("urban-nlos", "ray-traced urban, no direct path"),
    ):
        gains, delays, diag = rt_station_pair_cir(kind)
        base = SDRSimulationConfig(
            num_iterations=ITERATIONS, seed=0, device="cpu"
        )
        taps, _ = cir_to_frozen_taps(gains, delays, base)
        cells = [
            _cell(
                cache, f"rt_{kind}_s{seed}",
                lambda seed=seed, taps=taps: run_cell(
                    SDRSimulationConfig(
                        num_iterations=ITERATIONS, seed=seed,
                        device="cpu",
                    ),
                    CADENCES, taps,
                ),
            )
            for seed in (0, 1)
        ]
        rows.append((label, cells))

        if kind == "urban-nlos":
            tworay_excess = rt_station_pair_cir("tworay")[2][
                "excess_loss_db"
            ]
            penalty = max(0.0, diag["excess_loss_db"] - tworay_excess)
            snr_db = SDRSimulationConfig().snr_db - penalty
            snr_cell = _cell(
                cache, "rt_urban-nlos_snr_s0",
                lambda taps=taps, snr_db=snr_db: run_cell(
                    SDRSimulationConfig(
                        num_iterations=ITERATIONS, seed=0,
                        device="cpu", snr_db=snr_db,
                    ),
                    CADENCES, taps,
                ),
            )
            rows.append(
                ("ray-traced urban no-direct-path, loss charged",
                 [snr_cell])
            )
    return rows


def dumbbell(rows: list[tuple[str, list[dict]]]) -> None:
    def agg(cells: list[dict], field: str) -> float:
        return torch.tensor(
            [c[field] for c in cells], dtype=torch.float64
        ).mean().item()

    entries = [
        (label, agg(cells, "piggy"), agg(cells, "twoway"))
        for label, cells in rows
    ]
    entries.sort(key=lambda e: e[2])

    figure, axis = plt.subplots(figsize=(7.6, 6.2))
    for row, (label, piggy, twoway) in enumerate(entries):
        axis.plot(
            [piggy, twoway], [row, row], color="lightgray",
            linewidth=1.4, zorder=1,
        )
        axis.scatter(
            [piggy], [row], color="C0", s=46, zorder=2,
            label="piggyback, anchors every 40 (0.5% airtime)"
            if row == 0 else None,
        )
        axis.scatter(
            [twoway], [row], color="C1", s=46, zorder=2,
            label="two-way baseline (19.1% airtime)"
            if row == 0 else None,
        )
    axis.set_yticks(range(len(entries)))
    axis.set_yticklabels([e[0] for e in entries], fontsize=8.5)
    axis.set_xlabel("steady clock error (mrad RMS)")
    axis.set_xlim(left=0)
    axis.set_title(
        "Steady clock error by environment (N=2, 60 intervals)"
    )
    axis.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2,
        fontsize=8.5,
    )
    save(figure, "clutter_environment_dumbbell")


def main() -> None:
    cache = _load()
    rows = collect(cache)
    dumbbell(rows)


if __name__ == "__main__":
    main()
