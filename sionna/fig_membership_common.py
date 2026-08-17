"""Fixed method-to-color mapping for the membership figures.

Plain matplotlib: colors are the default cycle (C0..C7), assigned per
method so the same method has the same color in every figure.
"""

from __future__ import annotations

METHOD_COLORS = {
    "all-in": "C0",
    "post-gate": "C1",
    "gate": "C1",
    "gate-discard": "C1",
    "1-bit": "C2",
    "1-bit-10%err": "C2",  # dashed line distinguishes the variant
    "hybrid": "C3",
    "hybrid-post": "C3",
    "oracle": "C4",
    "hybrid-oracle": "C4",
    "greedy": "C5",
    "noncoh-all": "C6",
}

METHOD_LABELS = {
    "all-in": "all-in",
    "post-gate": "posterior gate",
    "gate": "posterior gate",
    "gate-discard": "gate (discard)",
    "1-bit": "1-bit alignment",
    "1-bit-10%err": "1-bit, 10% bit errors",
    "hybrid": "hybrid two-tier",
    "hybrid-post": "hybrid (posterior)",
    "oracle": "oracle",
    "hybrid-oracle": "hybrid (oracle)",
    "greedy": "greedy oracle",
    "noncoh-all": "noncoherent all",
}

FIGURES_DIR = None


def save_fig(figure, name: str):
    """Save under figures/studies/ with plain settings."""

    import os

    directory = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "figures", "studies"
    )
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}.png")
    figure.savefig(path, dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(figure)
    return path
