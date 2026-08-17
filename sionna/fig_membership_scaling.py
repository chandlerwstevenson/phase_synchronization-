"""Figures for the membership array-size scaling study.

Reads membership_scaling_cache.json, which must have been regenerated
by a fresh run of membership_scaling_study.py immediately before
plotting (this script refuses to plot if any cell is missing).

Fig 1: detection vs array size at matched power, per method.
Fig 2: detection range vs array size, with the ideal N^(3/4) reference.
Fig 3: net throughput vs array size.
Plain default matplotlib; no in-axes annotations.
"""

from __future__ import annotations

import numpy as np

from fig_membership_common import METHOD_COLORS, METHOD_LABELS, save_fig
from membership_scaling_study import (
    CHEAP_SEEDS,
    _load_cache,
    _mean,
    _mean_pd,
)
from metrics_membership_study import METHODS
import matplotlib.pyplot as plt

STATIONS = (6, 10, 14, 20)


def _check_complete(cache: dict) -> None:
    missing = []
    for n in STATIONS:
        for seed in CHEAP_SEEDS:
            for method in METHODS:
                key = f"{n}/{seed}/{method}/gain"
                if key not in cache:
                    missing.append(key)
    if missing:
        raise SystemExit(
            f"cache incomplete ({len(missing)} cells, e.g. {missing[0]}); "
            "run membership_scaling_study.py first"
        )
    print(f"cache complete for N={STATIONS}, seeds {CHEAP_SEEDS}")


def _line_plot(metric, ylabel, title, name, ideal=None):
    figure, axis = plt.subplots(figsize=(6.2, 4.0))
    for method in METHODS:
        values = [metric(n, method) for n in STATIONS]
        style = "--" if method == "1-bit-10%err" else "-"
        if method == "hybrid":
            style = ":"
        axis.plot(
            STATIONS, values, linestyle=style, marker="o",
            color=METHOD_COLORS[method], label=METHOD_LABELS[method],
        )
    if ideal is not None:
        axis.plot(
            STATIONS, [ideal(n) for n in STATIONS],
            linestyle="--", color="gray", linewidth=1.4,
            label="ideal perfect-sync scaling",
        )
    axis.set_xticks(list(STATIONS))
    axis.set_xlabel("Array size (number of stations)")
    axis.set_ylabel(ylabel)
    axis.legend(fontsize=8)
    axis.set_title(title)
    print("saved", save_fig(figure, name))


def main() -> None:
    cache = _load_cache()
    _check_complete(cache)

    _line_plot(
        lambda n, m: 100.0 * _mean_pd(cache, n, m, "matched"),
        "Probability of detection (%)",
        "Probability of detection vs array size (matched power)",
        "scaling_detection_vs_N",
    )

    anchor = _mean(cache, 6, "all-in", "range", CHEAP_SEEDS)
    _line_plot(
        lambda n, m: _mean(cache, n, m, "range", CHEAP_SEEDS),
        "Detection range (m)",
        "Detection range vs array size",
        "scaling_range_vs_N",
        ideal=lambda n: anchor * (n / 6.0) ** 0.75,
    )

    _line_plot(
        lambda n, m: _mean(cache, n, m, "net", CHEAP_SEEDS),
        "Net throughput (bits/s/Hz after sync overhead)",
        "Net throughput vs array size",
        "scaling_net_throughput_vs_N",
    )

    for n in STATIONS:
        row = {
            m: f"{100.0 * _mean_pd(cache, n, m, 'matched'):.1f}"
            for m in METHODS
        }
        print(f"N={n} matched-power detection:", row)


if __name__ == "__main__":
    main()
