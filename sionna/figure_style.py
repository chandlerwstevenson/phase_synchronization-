"""Shared figure style for every study figure in this repo.

One validated categorical palette (eight slots, fixed order — validated
with the dataviz six-checks script: lightness band, chroma floor, CVD
separation, normal-vision floor all PASS on the light surface), one set
of matplotlib defaults, one save helper. Every study figure imports
from here so the whole set reads as a single system.

Rules encoded here (do not work around them in study code):
  - categorical hues assigned in fixed slot order, never cycled/generated
  - one y-axis per figure — never twin axes; use small multiples instead
  - sequential data uses ONE hue, light to dark (`sequential()`)
  - recessive grid and spines; the data carries the ink
  - legend present whenever there are >= 2 series
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical order: blue, orange, aqua, yellow, magenta,
# green, violet, red. Worst adjacent CVD dE 9.1, normal-vision 19.6.
SERIES = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
SURFACE = "#fcfcfb"
INK = "#222222"
MUTED_INK = "#666666"
GRID = "#e3e3df"

FIGURES_DIR = Path(__file__).resolve().parent / "figures" / "studies"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED_INK,
            "ytick.color": MUTED_INK,
            "text.color": INK,
            "lines.linewidth": 1.8,
            "lines.markersize": 6.0,
            "legend.frameon": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "figure.titlesize": 12,
        }
    )


def sequential(n: int, hue: str = "#2a78d6") -> list[str]:
    """One-hue light-to-dark ramp for magnitude encodings."""

    import matplotlib.colors as mcolors

    base = mcolors.to_rgb(hue)
    white = (0.97, 0.97, 0.96)
    steps = []
    for index in range(n):
        t = 0.25 + 0.75 * index / max(n - 1, 1)
        steps.append(
            mcolors.to_hex(tuple(w + (b - w) * t for w, b in zip(white, base)))
        )
    return steps


def save(figure, name: str) -> Path:
    """Save under figures/studies/ and return the path."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path
