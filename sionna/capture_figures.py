"""Headless figure capture: run any of this repo's plotting commands
and save every figure that would have opened as a window.

Usage:
    .venv/bin/python capture_figures.py <prefix> <script.py> [args...]

Every plt.show() call saves all currently open figures to
figures/deck/<prefix>_NN.png (140 dpi) and closes them, so interactive
scripts run to completion unattended. The combined annotated figure is
always the FIRST png of a show-call group (the repo opens it before
the clean per-panel windows).
"""

import os
import runpy
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "figures", "deck")
os.makedirs(OUTDIR, exist_ok=True)

prefix = sys.argv[1]
counter = [0]


def _saving_show(*args, **kwargs):
    for number in plt.get_fignums():
        figure = plt.figure(number)
        counter[0] += 1
        path = os.path.join(OUTDIR, f"{prefix}_{counter[0]:02d}.png")
        figure.savefig(path, dpi=140, bbox_inches="tight",
                       facecolor=figure.get_facecolor())
        print(f"[capture] wrote {path}", flush=True)
    plt.close("all")


plt.show = _saving_show

sys.argv = sys.argv[2:]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
runpy.run_path(sys.argv[0], run_name="__main__")
_saving_show()  # catch figures created but never shown
