"""When does synchronizing LESS OFTEN give BETTER phase accuracy?

The reviewer's demand: establish the scheduling reversal as a derived
REGIME condition, not a blanket inversion of practice. This module
derives the condition, predicts which configurations reverse, and
measures the boundary.

-----------------------------------------------------------------------
DERIVATION
-----------------------------------------------------------------------
Setting (a): dedicated-only synchronization. A two-station link
exchanges a dedicated two-way sync frame every M intervals (interval
T). Each exchange yields a half-difference phase measurement with
variance

    r = r_th + r_extra,

where r_th is the thermal/estimation part the deployed filter models
and r_extra is the per-exchange EXCESS the filter does not model.
ATTRIBUTION UPDATE (dominance study, RESULTS_DOMINANCE.md): r_extra
is OSCILLATOR-DERIVED (intra-capture walk plus flicker-FM treated as
white), class-proportional (~33/134/673 mrad for ocxo/tcxo/sdr,
sibling-inferred), and independent of timing jitter, channel, and
SNR — it is NOT multipath resampling (frozen-oscillator control:
jitter x multipath < 0.05 mrad). The derivation below never depended
on the source: the reversal condition needs only that each exchange
carries r_extra. The oscillator pair additionally drifts with
phase-walk variance q per interval (plus a frequency random walk
q_w).

Proposition 1 (scalar phase model: NO reversal, ever).
With state theta only, the sampled-chain steady posterior obeys the
scalar Riccati equation with process Mq and measurement r:

    P+ = [ -Mq + sqrt((Mq)^2 + 4 M q r) ] / 2,

and the time-averaged held error is  sig2(M) = P+ + q (M-1)/2.
Both terms increase with M, for a correctly specified filter AND for
a misspecified one (gain from r_th, error propagated with r: the
mismatched fixed point E+ = [(1-k)^2 Mq + k^2 r] / (1-(1-k)^2) is
also increasing in M; the mismatch inflates the level by roughly
(1 + r/r_th)/2 but never flips the slope). White measurement noise,
uncorrelated across exchanges, cannot make more measurements worse in
a phase-only loop — it only sets a floor. Any measured reversal must
therefore come from OUTSIDE the scalar model.

Proposition 2 (the reversal channel: frequency estimation x coast).
The real loop is two-state [theta, omega] with actuation latency L.
Between exchanges the loop coasts on its frequency estimate, so the
held error j intervals after a correction is

    var_j = e_th + 2 jT c_tw + (jT)^2 e_w + j q,

with (e_th, c_tw, e_w) the terms of the ACTUAL estimation-error
covariance at correction time — computed from the mismatched Riccati
recursion (gain from the believed covariances built on r_th; error
propagated with the actual r). Sparser exchanges lengthen the
frequency baseline, so e_w falls roughly like r / (M T)^2 while the
drift term grows like q M / 2. Averaging var_j over the coast and
differentiating in M gives the REVERSAL CONDITION: synchronizing less
often improves held accuracy at spacing M iff

    d/dM [ e_th(M) + T^2 (M-1)(2M-1)/6 * e_w(M) ] < - q / 2 .

Because e_w(M) ~ beta r / (M T)^2, the left side is ~ -c r / M^2 for
small M, so the reversal regime is (up to the numeric Riccati
factors)

    r_th + r_extra  >~  q M^3 / c     (c an O(1) loop constant).

With the CORRECTED attribution this inequality can essentially never
hold in a dedicated-only loop: r_extra is oscillator-derived and
scales with the same class constant as q, so a class that inflates
r_extra inflates q with it — the ratio (r_th + r_extra)/q is
class-bounded and the exact mismatched-Riccati evaluation finds no
reversal cell (confirmed by measurement: all 30 dedicated cells are
flat-or-worse with sparser exchanges, and jitter has no effect).
The reversal is therefore EXCLUSIVELY a property of setting (b): it
requires an alternative observation channel that is cheaper per unit
information than the dedicated exchange, so that the exchange's
r_extra is net injection rather than net information.

Setting (b): with free observations carrying the tracking, dedicated
exchanges only re-pin the gauge, and their noise is pure injection.
The measured frontier (N=8: ~130 mrad at K<=10 saturating, declining
to 86 mrad at K=320) is modeled in reduced form as

    sig2(K) = floor2 + A / K + b K,

where A/K is the per-interval share of exchange-injected variance
(calibrated from the declining branch), and b = g S / 3 is fixed
INDEPENDENTLY from the ablation-measured realized gauge drift
(g = 1.7e-7 rad^2 per substep, S = 5 substeps: the static-environment
value, three orders below the covariance bound). The minimum sits at
K* = sqrt(A/b); with the static-environment g the predicted rise
beyond K* is sub-mrad out to K = 640 — i.e. the measurable
prediction is FLATNESS, and a visible turn-up requires environmental
motion (where g inflates to the channel-innovation rate and K*
collapses to the coherence-time boundary of experiment B).

Inputs declared, not fitted: q, q_w, r_th from the coast-law link
reconstruction (datasheet + link budget); r_extra per oscillator
class from the dominance study's uniform-service inference
(~33/134/673 mrad for ocxo/tcxo/sdr), jitter-independent. The
original version of this module assumed r_extra scaled with timing
jitter (the multipath-resampling hypothesis); the measured sweep
refuted that (j2 vs j32 columns statistically identical in all 30
cells) and the class-based values replaced it.

Usage:
    python scheduling_reversal.py --part theory
    python scheduling_reversal.py --part dedicated   # experiment 2
    python scheduling_reversal.py --part frontier    # experiment 3
    python scheduling_reversal.py --part figs
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from coast_law import link_matrices
from ota_sync import SDRSimulationConfig
from dataclasses import replace

HERE = Path(__file__).resolve().parent
CACHE = HERE / "scheduling_reversal_cache.json"
AC_CACHE = HERE / "experiment_a_c_cache.json"
FIGDIR = HERE / "figures"

CLASSES = ["ocxo", "tcxo", "sdr"]
JITTERS = [2, 32]
M_LIST = [1, 2, 4, 8, 16]
SEEDS = [0, 1, 2]
SNR_DB = 20.0
# Per-exchange excess measurement noise (half-difference, rad), by
# oscillator class — sibling dominance study's uniform-service
# inference; oscillator-derived, jitter/channel/SNR-independent.
R_EXTRA_BY_CLASS = {"ocxo": 0.033, "tcxo": 0.134, "sdr": 0.673}
# Ablation-measured realized gauge drift, static environment.
GAUGE_DRIFT_PER_SUBSTEP = 1.7e-7  # rad^2
SUBSTEPS = 5


def _load_cache() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return {}


def _save_cache(cache: dict) -> None:
    CACHE.write_text(json.dumps(cache, indent=1))


# ---------------------------------------------------------------------
# Theory: mismatched two-state Riccati prediction of the held error
# ---------------------------------------------------------------------

def link_constants(profile: str, jitter: int) -> dict:
    settings = SDRSimulationConfig(
        device="cpu", timing_jitter_samples=jitter
    )
    matrices = link_matrices(settings, profile, SNR_DB, 60 * settings.sync_interval)
    q = float(matrices.process[0, 0])
    q_w = float(matrices.process[1, 1])
    q_c = float(matrices.process[0, 1])
    r_th = float(matrices.measurement[0, 0])
    r_w = float(matrices.measurement[1, 1])
    s_extra = R_EXTRA_BY_CLASS[profile] ** 2
    return {
        "q": q, "q_w": q_w, "q_c": q_c, "r_th": r_th, "r_w": r_w,
        "s_mp": s_extra, "T": settings.sync_interval,
    }


def held_error_prediction(profile: str, jitter: int, M: int) -> float:
    """Mismatched-Riccati held rms (rad) for exchange spacing M.

    Gain built from the believed covariances (thermal-only measurement
    model, exactly what the deployed filter assumes); actual error
    covariance propagated with the resampling-inflated measurement.
    Held average includes coast drift and the frequency-coast term.
    """

    c = link_constants(profile, jitter)
    T, q, q_w, q_c = c["T"], c["q"], c["q_w"], c["q_c"]
    F = torch.tensor([[1.0, T], [0.0, 1.0]], dtype=torch.float64)
    Q1 = torch.tensor([[q, q_c], [q_c, q_w]], dtype=torch.float64)
    H = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    R_believed = torch.diag(torch.tensor([c["r_th"], c["r_w"]],
                                         dtype=torch.float64))
    R_actual = torch.diag(torch.tensor(
        [c["r_th"] + c["s_mp"], c["r_w"]], dtype=torch.float64))

    FM = torch.linalg.matrix_power(F, M)
    QM = torch.zeros(2, 2, dtype=torch.float64)
    acc = torch.eye(2, dtype=torch.float64)
    for _ in range(M):
        QM = F @ QM @ F.T + Q1
    P = torch.eye(2, dtype=torch.float64)  # believed
    E = torch.eye(2, dtype=torch.float64)  # actual
    identity = torch.eye(2, dtype=torch.float64)
    for _ in range(4000):
        P_prior = FM @ P @ FM.T + QM
        S = H @ P_prior @ H.T + R_believed
        K_gain = P_prior @ H.T @ torch.linalg.inv(S)
        closed = identity - K_gain @ H
        P_new = closed @ P_prior @ closed.T + K_gain @ R_believed @ K_gain.T
        E_prior = FM @ E @ FM.T + QM
        E_new = closed @ E_prior @ closed.T + K_gain @ R_actual @ K_gain.T
        if torch.max(torch.abs(P_new - P)) < 1e-16 and \
           torch.max(torch.abs(E_new - E)) < 1e-16:
            P, E = P_new, E_new
            break
        P, E = P_new, E_new
    # Held error averaged over the coast: at j intervals past the
    # correction the error is a_j = [1, jT] e + accumulated drift.
    total = 0.0
    for j in range(M):
        a = torch.tensor([[1.0, j * T]], dtype=torch.float64)
        var = float(a @ E @ a.T) + j * q
        total += var
    return math.sqrt(total / M)


def reversal_cells() -> list[tuple[str, int]]:
    """Configurations the theory predicts as reversing (held error at
    some M>1 below the M=1 value)."""

    out = []
    for profile in CLASSES:
        for jitter in JITTERS:
            base = held_error_prediction(profile, jitter, 1)
            best = min(held_error_prediction(profile, jitter, m)
                       for m in M_LIST)
            if best < base * 0.98:
                out.append((profile, jitter))
    return out


# ---------------------------------------------------------------------
# Experiment 2: dedicated-only measured sweep
# ---------------------------------------------------------------------

def run_dedicated_cell(profile: str, jitter: int, M: int, seed: int) -> float:
    from pi_ambiguity_study import run_branch_star

    intervals = max(80, 24 * M)
    settings = SDRSimulationConfig(
        num_iterations=intervals, seed=seed, device="cpu",
        timing_jitter_samples=jitter,
    )
    result = run_branch_star(
        settings,
        num_stations=2,
        service_every=M,
        oscillator_profiles=[profile, profile],
    )
    settle = max(12, 8 * M)
    tail = result.residuals[0, settle:]
    return 1e3 * math.sqrt(float(torch.mean(tail ** 2)))


def part_dedicated() -> None:
    cache = _load_cache()
    for profile in CLASSES:
        for jitter in JITTERS:
            for M in M_LIST:
                for seed in SEEDS:
                    key = f"D|{profile}|j{jitter}|M{M}|s{seed}"
                    if key in cache:
                        continue
                    cache[key] = run_dedicated_cell(profile, jitter, M, seed)
                    _save_cache(cache)
                    print(key, round(cache[key], 1), flush=True)


# ---------------------------------------------------------------------
# Experiment 3: extended frontier
# ---------------------------------------------------------------------

def part_frontier() -> None:
    from piggyback_largen_study import run_piggyback_variant

    cache = _load_cache()
    for K in [640]:
        for seed in SEEDS:
            key = f"F|8|K{K}|s{seed}"
            if key in cache:
                continue
            intervals = 4 * K + 8
            settings = SDRSimulationConfig(
                num_iterations=intervals, seed=seed, device="cpu"
            )
            res = run_piggyback_variant(
                settings, num_stations=8,
                anchor_every_intervals=K, inflate_process=True,
            )
            cache[key] = res.star.worst_rms_mrad
            _save_cache(cache)
            print(key, round(cache[key], 1), flush=True)


def frontier_model() -> dict:
    """Reduced-form sig2(K) = floor2 + A/K + bK; b fixed from the
    ablation gauge drift; (floor2, A) calibrated on the declining
    branch K in {40..320} of the recorded experiment-C frontier."""

    ac = json.loads(AC_CACHE.read_text())
    measured = {}
    for K in [2, 5, 10, 20, 40, 80, 160, 320]:
        vals = [ac[f"C|opportunistic|8|{s}|K{K}"]["worst_mrad"]
                for s in SEEDS
                if f"C|opportunistic|8|{s}|K{K}" in ac]
        if vals:
            measured[K] = sum(vals) / len(vals)
    b = GAUGE_DRIFT_PER_SUBSTEP * SUBSTEPS / 3.0 * 1e6  # mrad^2/interval
    ks = [K for K in [40, 80, 160, 320] if K in measured]
    # least squares on sig2 = floor2 + A/K (b K subtracted first)
    ys = [measured[K] ** 2 - b * K for K in ks]
    xs = [1.0 / K for K in ks]
    n = len(ks)
    xbar, ybar = sum(xs) / n, sum(ys) / n
    A = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / \
        sum((x - xbar) ** 2 for x in xs)
    floor2 = ybar - A * xbar
    k_star = math.sqrt(A / b) if A > 0 else float("inf")
    return {"measured": measured, "A": A, "floor2": floor2, "b": b,
            "k_star": k_star}


# ---------------------------------------------------------------------
# Reporting and figures
# ---------------------------------------------------------------------

def dedicated_table(cache: dict) -> dict:
    table = {}
    for profile in CLASSES:
        for jitter in JITTERS:
            row = []
            for M in M_LIST:
                vals = [cache.get(f"D|{profile}|j{jitter}|M{M}|s{s}")
                        for s in SEEDS]
                vals = [v for v in vals if v is not None]
                row.append(sum(vals) / len(vals) if vals else None)
            table[(profile, jitter)] = row
    return table


def part_theory() -> None:
    print("mismatched-Riccati held-rms predictions (mrad):")
    print(f"{'class':>6} {'jitter':>7} " +
          "".join(f"M={m:<6}" for m in M_LIST))
    for profile in CLASSES:
        for jitter in JITTERS:
            row = [1e3 * held_error_prediction(profile, jitter, m)
                   for m in M_LIST]
            print(f"{profile:>6} {jitter:>7} " +
                  "".join(f"{v:<8.1f}" for v in row))
    print("\npredicted reversal cells (best M>1 beats M=1):",
          reversal_cells())
    model = frontier_model()
    print(f"\nfrontier reduced model: A={model['A']:.0f} mrad^2*intervals, "
          f"floor={math.sqrt(max(model['floor2'],0)):.1f} mrad, "
          f"b={model['b']:.3f} mrad^2/interval, K*={model['k_star']:.0f}")
    for K in [320, 640, 1280]:
        pred = math.sqrt(model["floor2"] + model["A"] / K + model["b"] * K)
        print(f"  predicted sigma(K={K}) = {pred:.1f} mrad")


def part_figs() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cache = _load_cache()
    table = dedicated_table(cache)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, jitter in zip(axes, JITTERS):
        for profile in CLASSES:
            measured = table[(profile, jitter)]
            ax.plot(M_LIST, measured, "o-", label=f"{profile} measured")
            theory = [1e3 * held_error_prediction(profile, jitter, m)
                      for m in M_LIST]
            ax.plot(M_LIST, theory, "--", label=f"{profile} theory")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("exchange spacing M (intervals)")
        ax.set_ylabel("held residual (mrad)")
        ax.set_title(f"timing jitter +/-{jitter} samples")
        ax.legend(fontsize=7)
    fig.tight_layout()
    FIGDIR.mkdir(exist_ok=True)
    fig.savefig(FIGDIR / "figR1_dedicated_vs_M.png", dpi=200)
    plt.close(fig)

    model = frontier_model()
    ks = sorted(model["measured"])
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(ks, [model["measured"][K] for K in ks], "o",
            label="measured (seeds 0-2)")
    ext = [cache.get(f"F|8|K640|s{s}") for s in SEEDS]
    ext = [v for v in ext if v is not None]
    if ext:
        ax.plot([640], [sum(ext) / len(ext)], "s",
                label="measured K=640 (this study)")
    grid = [k for k in range(10, 1300, 5)]
    ax.plot(grid,
            [math.sqrt(max(model["floor2"], 0) + model["A"] / k
                       + model["b"] * k) for k in grid],
            "--", label="reduced model (b from ablation drift)")
    ax.axvline(model["k_star"], linestyle=":",
               label=f"predicted minimum K*")
    ax.set_xscale("log")
    ax.set_xlabel("dedicated-exchange spacing K (intervals)")
    ax.set_ylabel("worst-station residual (mrad)")
    ax.set_title("Frontier: residual vs dedicated-exchange spacing (N=8)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "figR2_frontier_extended.png", dpi=200)
    plt.close(fig)
    print("figures written")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", required=True,
                        choices=["theory", "dedicated", "frontier", "figs"])
    args = parser.parse_args()
    if args.part == "theory":
        part_theory()
    elif args.part == "dedicated":
        part_dedicated()
    elif args.part == "frontier":
        part_frontier()
    else:
        part_figs()


if __name__ == "__main__":
    main()
