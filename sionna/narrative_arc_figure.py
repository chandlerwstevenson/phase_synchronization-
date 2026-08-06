"""One-figure summary of the method story, in five numbered beats:

  (1) one-way fails (channel ambiguity)   -> panel A
  (2) two-way fixes it                    -> panel A
  (3) micro corrects more often           -> panel A
  (4) hybrid wins the accuracy/airtime trade -> panel B
  (5) airtime becomes the wall as N grows -> panel C

All numbers are the measured seed-0 results already documented in
figures/deck/FIGURES.md (single link, 50 ms cadence, TDL-D, all
impairments on); nothing here is re-simulated.

Usage: python narrative_arc_figure.py   -> figures/deck/arc_01.png
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Method colors follow the deck (two-way blue, micro orange, hybrid green);
# hybrid's green is stepped to a teal-green so the orange/green pair stays
# separable under red-green color blindness. One-way is gray on purpose:
# it is disqualified, not a contender.
C_ONEWAY = "#7f7f7f"
C_TWOWAY = "#1f77b4"
C_MICRO = "#ff7f0e"
C_HYBRID = "#1baf7a"
C_DFPC = "#8c564b"
C_KFDFPC = "#e377c2"
C_THRESH = "tab:red"

THRESH_MRAD = 314.0  # 18 deg: keeps >= 90% coherent gain

# method -> (true residual mrad, coherent gain %, airtime % per link).
# DFPC / KF-DFPC numbers are the reciprocity-steelman runs (channel-free
# measurement exchange granted, as their publication assumes).
RESULTS = {
    "one-way": (2906.0, 2.2, 9.6),
    "two-way": (83.5, 99.83, 19.1),
    "micro": (28.1, 99.98, 26.0),
    "hybrid": (33.8, 99.97, 14.9),
    "DFPC": (153.0, 99.42, 19.1),
    "KF-DFPC": (82.8, 99.83, 19.1),
}

# largest N whose pilots fit one channel at the 50 ms cadence
# (scalability_sweep.py, star topology: N-1 links share the channel).
# One-way's number is arithmetic only (100/9.6 + 1): cheap airtime,
# but the array it "fits" never actually combines (2.2% gain).
MAX_N_50MS = {"one-way": 11, "two-way": 6, "micro": 4, "hybrid": 7}

COLORS = {
    "one-way": C_ONEWAY,
    "two-way": C_TWOWAY,
    "micro": C_MICRO,
    "hybrid": C_HYBRID,
    "DFPC": C_DFPC,
    "KF-DFPC": C_KFDFPC,
}
MARKERS = {
    "one-way": "x",
    "two-way": "o",
    "micro": "s",
    "hybrid": "D",
    "DFPC": "^",
    "KF-DFPC": "v",
}


def main():
    fig = plt.figure(figsize=(17.5, 5.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.95, 1.15], wspace=0.26)
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1])
    axC = fig.add_subplot(gs[2])

    # ---- Panel A: true oscillator error per method (beats 1-3) ----
    methods = list(RESULTS)
    xs = np.arange(len(methods))
    residuals = [RESULTS[m][0] for m in methods]
    axA.bar(
        xs,
        residuals,
        width=0.62,
        color=[COLORS[m] for m in methods],
        zorder=3,
    )
    axA.axhline(
        THRESH_MRAD, color=C_THRESH, linestyle="--", linewidth=1.2, zorder=2
    )
    axA.text(
        len(methods) - 0.45,
        THRESH_MRAD * 0.72,
        "314 mrad",
        color=C_THRESH,
        fontsize=8.5,
        ha="right",
    )
    for x, m in zip(xs, methods):
        r, gain, _ = RESULTS[m]
        axA.text(
            x,
            r * 1.15,
            f"{r:.0f}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    axA.set_yscale("log")
    axA.set_ylim(10, 9000)
    axA.set_xticks(xs)
    axA.set_xticklabels(methods)
    axA.set_ylabel("true oscillator residual (mrad, log)")
    axA.set_title("True residual", fontsize=11)
    axA.grid(True, which="both", axis="y", alpha=0.25)

    # ---- Panel B: accuracy vs airtime, one link (beat 4) ----
    # KF-DFPC lands on top of two-way (83 vs 84 mrad at the same 19.1%
    # airtime), so it is drawn smaller with a white ring and the two
    # share one label.
    LABELS_B = {
        "two-way": ("two-way / KF-DFPC", 0.7, 1.2),
        "micro": ("micro", 0.7, 1.3),
        "hybrid": ("hybrid", -4.3, 1.35),
        "DFPC": ("DFPC", 0.7, 1.2),
        "KF-DFPC": None,
    }
    for m in ["two-way", "micro", "hybrid", "DFPC", "KF-DFPC"]:
        r, _, air = RESULTS[m]
        small = m == "KF-DFPC"
        axB.plot(
            air,
            r,
            MARKERS[m],
            color=COLORS[m],
            markersize=6 if small else 10,
            markeredgecolor="white" if small else COLORS[m],
            markeredgewidth=1.2 if small else 0,
            zorder=5 if small else 4,
        )
        if LABELS_B[m] is None:
            continue
        text, dx, dy = LABELS_B[m]
        axB.annotate(
            text,
            xy=(air, r),
            xytext=(air + dx, r * dy),
            fontsize=8.5,
        )
    r1, _, air1 = RESULTS["one-way"]
    axB.plot(air1, r1, "x", color=C_ONEWAY, markersize=11, markeredgewidth=2.5)
    axB.annotate(
        "one-way",
        xy=(air1, r1),
        xytext=(air1 + 1.0, r1 * 0.72),
        fontsize=8.5,
        color="#555555",
    )
    # hybrid is the Pareto pick: ring it
    rh, _, airh = RESULTS["hybrid"]
    axB.plot(
        airh,
        rh,
        "o",
        markersize=17,
        markerfacecolor="none",
        markeredgecolor=C_HYBRID,
        markeredgewidth=1.6,
        zorder=3,
    )
    axB.axhline(THRESH_MRAD, color=C_THRESH, linestyle="--", linewidth=1.2)
    axB.set_yscale("log")
    axB.set_ylim(10, 9000)
    axB.set_xlim(6, 32)
    axB.set_xlabel("pilot airtime per link (% of channel)")
    axB.set_ylabel("true oscillator residual (mrad, log)")
    axB.set_title("Residual vs airtime", fontsize=11)
    axB.grid(True, which="both", alpha=0.25)

    # ---- Panel C: airtime vs array size (beat 5) ----
    n = np.arange(2, 13)
    for m in ["micro", "two-way", "hybrid", "one-way"]:
        air_per_link = RESULTS[m][2]
        style = "--" if m == "one-way" else "-"
        if m == "one-way":
            label = "one-way (9.6%/link — invalid)"
        elif m == "two-way":
            label = "two-way = DFPC = KF-DFPC (19.1%/link)"
        else:
            label = f"{m} ({air_per_link:.1f}%/link)"
        axC.plot(
            n,
            air_per_link * (n - 1),
            MARKERS[m] + style,
            color=COLORS[m],
            markersize=5,
            linewidth=2,
            label=label,
        )
        if m == "one-way":
            continue  # no wall marker: the array it fits never combines
        n_max = MAX_N_50MS[m]
        axC.plot(
            n_max,
            air_per_link * (n_max - 1),
            MARKERS[m],
            color=COLORS[m],
            markersize=11,
            markerfacecolor="white",
            markeredgewidth=2,
            zorder=4,
        )
    axC.axhline(100, color=C_THRESH, linestyle="--", linewidth=1.2)
    axC.text(2.1, 103, "100% of channel", color=C_THRESH, fontsize=8.5)
    axC.legend(loc="upper left", bbox_to_anchor=(0.02, 0.88), fontsize=8.5)
    axC.set_xlabel("number of stations N (star, one shared channel)")
    axC.set_ylabel("total pilot airtime (% of channel)")
    axC.set_title("Airtime vs array size", fontsize=11)
    axC.set_xlim(2, 12.4)
    axC.set_ylim(0, 210)
    axC.set_xticks(np.arange(2, 13, 2))
    axC.grid(True, alpha=0.25)

    fig.subplots_adjust(left=0.055, right=0.985, top=0.91, bottom=0.11)
    out = "figures/deck/arc_01.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}")

    # standalone bar chart: pilot airtime per link, per method
    fig2, ax = plt.subplots(figsize=(8, 5.2))
    methods = list(RESULTS)
    xs = np.arange(len(methods))
    airtimes = [RESULTS[m][2] for m in methods]
    ax.bar(xs, airtimes, width=0.62, color=[COLORS[m] for m in methods], zorder=3)
    for x, a in zip(xs, airtimes):
        ax.text(x, a + 0.5, f"{a:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(methods)
    ax.set_ylabel("pilot airtime per link (% of channel)")
    ax.set_ylim(0, 30)
    ax.set_title("Pilot airtime", fontsize=11)
    ax.grid(True, axis="y", alpha=0.25)
    fig2.tight_layout()
    out2 = "figures/deck/arc_02.png"
    fig2.savefig(out2, dpi=110)
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
