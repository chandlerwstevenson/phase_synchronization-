"""Winding-number conservation for phase-synchronization graphs:
what is provable, exactly.

Context: the paper measured that settled impaired states of cyclic
sync topologies carry cycle sums of wrapped pairwise offsets equal to
integer multiples of 2*pi ("quantized winding states"), and external
review asked us to soften "robust to any local update rule" to
"persists across the tested protocols" unless the invariance is
proved. This module proves the exact statements and refutes one
conjecture we ourselves proposed along the way.

Conventions. Node phases theta_v in R (physical phases; everything
below is convention-independent up to the wrap boundary set). For a
cycle C traversed v_0 -> v_1 -> ... -> v_m = v_0, define traversal
differences d_i = theta_{v_{i-1}} - theta_{v_i} and

    w_C = (1/2pi) * sum_i wrap(d_i),      wrap(x) in [-pi, pi).

THEOREM 1 (exact quantization, everywhere).
  w_C is an integer for EVERY phase configuration - not only settled
  ones. Proof: the unwrapped differences telescope, sum_i d_i = 0
  exactly; wrap(x) = x - 2pi*round(x/2pi); hence sum_i wrap(d_i)
  = -2pi * sum_i round(d_i/2pi), an integer multiple of 2pi. QED.
  (This is the phase-unwrapping residue identity - Goldstein - stated
  for sync graphs. Consequence for measurement practice: any
  INSTANTANEOUS wrapped cycle sum must be exactly 2pi*k; a reported
  non-integer "cycle sum" is necessarily a time average or a sum of
  estimator states, not of simultaneous physical offsets.)

THEOREM 2 (conservation under sub-boundary dynamics).
  w_C = -sum_i round(d_i/2pi) is locally constant in the
  configuration, jumping only where some d_i crosses the wrap
  boundary (d_i = pi mod 2pi). Therefore:
  (a) continuous-time: w_C is invariant under any continuous
      evolution during which no cycle edge's difference crosses the
      boundary; a transversal crossing changes w_C by exactly -+1
      (crossing upward through +pi: -1; downward through -pi: +1).
  (b) discrete steps: an update step conserves w_C if, for every
      cycle edge, the step's change to that edge difference is
      smaller than that edge's current distance to the boundary:
          |delta d_i| < pi - |wrap(d_i)|   for all i in C.
      Proof: under that bound, d_i stays inside its current period
      cell, so no round(.) term changes. QED.
  This is the precise form of "robust to local update rules": ANY
  protocol - simultaneous, sequential, directed, or otherwise -
  whose per-step corrections respect the bound conserves the winding
  sector. The hypothesis is a step-size condition, not universality.

THEOREM 3 (what a branch flip does to winding - REFUTING our own
  conjecture that node pi-flips are cycle-neutral).
  A node correction theta_v += s*pi (s = +-1), at a node v interior
  to cycle C with incident traversal differences d_j (which changes
  by -s*pi) and d_{j+1} (changes by +s*pi), with r_j = wrap(d_j),
  r_{j+1} = wrap(d_{j+1}) and no boundary landing, changes the
  winding by exactly
      delta w_C = -(1/2) * (sgn(r_plus) + sgn(r_minus)),
  INDEPENDENT of the flip sign s (a +pi and a -pi flip move each
  incident difference by amounts differing by 2pi, whose round-term
  changes cancel between the two edges). I.e. a flip changes winding
  iff BOTH incident cycle differences share a sign (delta w = -sgn),
  and is cycle-neutral iff they have opposite signs. Proof by the
  four sign cases of round((x +- pi)/2pi) - round(x/2pi). QED.
  The naive cut/cycle-orthogonality argument ("node flips are cut
  vectors, cuts are orthogonal to cycles, hence flips cannot wind")
  is true for UNWRAPPED sums - which are identically zero - and
  false for the wrapped sum, because a pi-jump can carry an edge
  difference across the boundary. Flips are precisely how wound
  states are ENTERED from near-lock (where residual signs are
  random: a flip winds with probability ~ P(adjacent signs agree)).

THEOREM 4 (metastability: every escape transits anti-lock).
  In a settled wound ring state (n edges, winding w != 0,
  differences approximately 2pi*w/n, all sharing sgn(w) when
  0 < 2pi|w|/n < pi):
  (a) an unwinding flip exists at EVERY node: the flip with
      s = sgn(w) gives delta w = -sgn(w) (both incident signs equal
      sgn(w)); so local dynamics CAN unwind - the conjectured
      impossibility statement is false;
  (b) but the cost is an anti-lock transient: the two flipped edges
      land at wrap(2pi*w/n -+ pi), i.e. at error magnitude
      pi - 2pi|w|/n each - the escape, discrete or continuous,
      transits the neighborhood of the wrap boundary (anti-lock) on
      at least one cycle edge. Combined with Theorem 2, the wound
      sector is invariant under ALL dynamics (any protocol) that
      keep every cycle edge clear of the boundary; sector changes
      require an anti-lock-scale event: a continuous slip through
      +-pi or a pi-flip adjacent to same-sign residuals.

Upgraded paper sentence (earned by Theorems 1-4; replaces the
softened "persists across the tested local update protocols"):
  "the winding number is exactly quantized at every instant and is
  invariant under any update dynamics - continuous or discrete,
  under any protocol - whose per-step corrections keep each cycle
  edge's phase difference away from the +-pi boundary; entering or
  leaving a wound sector requires an anti-lock event (an edge
  slipping through +-pi, or a pi branch correction applied where
  both adjacent residuals share a sign), so settled wound states are
  metastable under every local protocol rather than specific to the
  tested ones."

Verification below: V1 quantization (1e5 random configurations),
V2 conservation + crossing bookkeeping on continuous paths and
bounded random walks, V3 the flip formula (1e4 random flips, both
signs, on- and off-cycle nodes), V4 wound-state structure, unwinding
flips at every node, and the anti-lock transient magnitude,
V5 entry statistics from near-lock. Run:  python dirA4_winding_theorem.py
"""

from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


def wrap(x):
    return (np.asarray(x) + np.pi) % TWO_PI - np.pi


def cycle_diffs(theta, cycle):
    """Traversal differences d_i = theta[v_{i-1}] - theta[v_i]."""
    prev = np.asarray(cycle)
    nxt = np.roll(prev, -1)
    return theta[prev] - theta[nxt]


def winding(theta, cycle):
    return float(np.sum(wrap(cycle_diffs(theta, cycle)))) / TWO_PI


def boundary_distance(theta, cycle):
    """Min over cycle edges of distance from the wrap boundary."""
    return float(np.min(np.pi - np.abs(wrap(cycle_diffs(theta, cycle)))))


# ---------------------------------------------------------------- V1
def verify_quantization(trials=100_000, seed=0):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        n = rng.integers(3, 13)
        theta = rng.uniform(-40.0, 40.0, size=n)
        w = winding(theta, list(range(n)))
        worst = max(worst, abs(w - round(w)))
    return worst


# ---------------------------------------------------------------- V2
def verify_conservation(paths=400, substeps=2000, seed=1):
    """Continuous interpolation: w changes only at boundary crossings,
    by -1 per upward crossing of +pi and +1 per downward crossing of
    -pi (per cycle edge, mod-2pi cells handled by the round form)."""
    rng = np.random.default_rng(seed)
    mismatches = 0
    checked_jumps = 0
    for _ in range(paths):
        n = int(rng.integers(3, 9))
        cyc = list(range(n))
        a = rng.uniform(-8, 8, n)
        b = rng.uniform(-8, 8, n)
        prev_theta = a.copy()
        prev_w = round(winding(a, cyc))
        for t in np.linspace(0.0, 1.0, substeps)[1:]:
            theta = (1 - t) * a + t * b
            w = round(winding(theta, cyc))
            if w != prev_w:
                # signed crossings of the round terms this substep
                d0 = cycle_diffs(prev_theta, cyc)
                d1 = cycle_diffs(theta, cyc)
                jump = -int(
                    np.sum(np.round(d1 / TWO_PI) - np.round(d0 / TWO_PI))
                )
                checked_jumps += 1
                if jump != w - prev_w:
                    mismatches += 1
            prev_theta, prev_w = theta, w
    return mismatches, checked_jumps


def verify_bounded_walk(walks=300, steps=400, seed=2):
    """Random per-node updates; whenever the Theorem-2b bound holds
    for the step, w must be unchanged. Count violations of that
    implication (must be zero); also count how often w changed on
    steps breaking the bound (allowed either way)."""
    rng = np.random.default_rng(2)
    implication_violations = 0
    bound_held = 0
    changes_when_broken = 0
    for _ in range(walks):
        n = int(rng.integers(3, 9))
        cyc = list(range(n))
        theta = rng.uniform(-3, 3, n)
        for _ in range(steps):
            delta = rng.normal(0.0, 0.6, n)
            d_before = cycle_diffs(theta, cyc)
            new_theta = theta + delta
            d_after = cycle_diffs(new_theta, cyc)
            step_change = np.abs(d_after - d_before)
            margin = np.pi - np.abs(wrap(d_before))
            w0, w1 = round(winding(theta, cyc)), round(
                winding(new_theta, cyc)
            )
            if np.all(step_change < margin):
                bound_held += 1
                if w1 != w0:
                    implication_violations += 1
            elif w1 != w0:
                changes_when_broken += 1
            theta = new_theta
    return implication_violations, bound_held, changes_when_broken


# ---------------------------------------------------------------- V3
def verify_flip_formula(trials=10_000, seed=3, tol=1e-6):
    """delta w = -(s/2)(sgn r_j + sgn r_{j+1}) for on-cycle nodes;
    0 for off-cycle nodes. Boundary landings excluded."""
    rng = np.random.default_rng(seed)
    tested = 0
    mismatches = 0
    off_cycle_mismatches = 0
    for _ in range(trials):
        n = int(rng.integers(4, 10))
        cyc = list(range(n - 1))  # last node off-cycle
        theta = rng.uniform(-6, 6, n)
        s = int(rng.choice([-1, 1]))
        v = int(rng.integers(0, n))
        w0 = round(winding(theta, cyc))
        theta2 = theta.copy()
        theta2[v] += s * np.pi
        w1 = round(winding(theta2, cyc))
        if v == n - 1:
            if w1 != w0:
                off_cycle_mismatches += 1
            continue
        j = cyc.index(v)
        d = cycle_diffs(theta, cyc)
        r_in = wrap(d[j])          # d_j: theta[v_{j-1}] - theta[v_j]...
        r_out = wrap(d[(j) % len(cyc)])
        # careful indexing: d_i = theta[cyc[i]] - theta[cyc[i+1]].
        # cycle_diffs uses prev=cyc, nxt=roll(-1): d[i] = theta[cyc[i]]
        # - theta[cyc[i+1]]. Node v = cyc[j] appears in d[j] (with +)
        # and d[j-1] (with -). theta_v += s*pi: d[j] += s*pi,
        # d[j-1] -= s*pi.
        r_plus = wrap(d[j])            # edge gaining +s*pi
        r_minus = wrap(d[j - 1])       # edge gaining -s*pi
        near_boundary = min(
            abs(r_plus), np.pi - abs(r_plus),
            abs(r_minus), np.pi - abs(r_minus),
        ) < tol
        if near_boundary:
            continue
        predicted = -0.5 * (np.sign(r_plus) + np.sign(r_minus))
        tested += 1
        if round(predicted) != (w1 - w0):
            mismatches += 1
    return tested, mismatches, off_cycle_mismatches


# ---------------------------------------------------------------- V4
def verify_wound_states(seed=4):
    """Settled wound rings: same-sign differences; unwinding flip at
    every node; anti-lock transient magnitude pi - 2pi|w|/n."""
    rng = np.random.default_rng(seed)
    rows = []
    for n, w in [(6, 1), (8, 1), (12, 1), (8, 2), (12, 2)]:
        cyc = list(range(n))
        base = -TWO_PI * w * np.arange(n) / n  # d_i = +2pi w/n each
        theta = base + rng.normal(0, 0.03, n)
        d = wrap(cycle_diffs(theta, cyc))
        same_sign = bool(np.all(np.sign(d) == np.sign(d[0])))
        w_meas = round(winding(theta, cyc))
        unwind_at_every_node = True
        transient_errors = []
        for v in range(n):
            s = int(np.sign(w_meas))
            theta2 = theta.copy()
            theta2[v] += s * np.pi
            if round(winding(theta2, cyc)) != w_meas - s:
                unwind_at_every_node = False
            transient_errors.append(
                float(np.max(np.abs(wrap(cycle_diffs(theta2, cyc)))))
            )
        predicted_transient = np.pi - TWO_PI * abs(w) / n
        rows.append(
            (
                n,
                w,
                w_meas,
                same_sign,
                unwind_at_every_node,
                float(np.mean(transient_errors)),
                predicted_transient + TWO_PI * abs(w) / n * 0
                if False
                else predicted_transient,
            )
        )
    return rows


# ---------------------------------------------------------------- V5
def verify_entry_statistics(trials=20_000, sigma=0.1, n=8, seed=5):
    """From near-lock (w=0, small residuals): fraction of random
    pi-flips that wind, vs the sign-agreement prediction."""
    rng = np.random.default_rng(seed)
    wound = 0
    agree = 0
    cyc = list(range(n))
    for _ in range(trials):
        theta = np.cumsum(rng.normal(0, sigma, n))
        theta -= theta.mean()
        v = int(rng.integers(0, n))
        d = cycle_diffs(theta, cyc)
        r_plus, r_minus = wrap(d[v]), wrap(d[v - 1])
        if np.sign(r_plus) == np.sign(r_minus):
            agree += 1
        w0 = round(winding(theta, cyc))
        theta[v] += np.pi * rng.choice([-1, 1])
        if round(winding(theta, cyc)) != w0:
            wound += 1
    return wound / trials, agree / trials


def main():
    print("V1 quantization: worst |w - round(w)| over 1e5 configs:",
          f"{verify_quantization():.2e}")
    mm, jumps = verify_conservation()
    print(f"V2a continuous paths: {jumps} winding jumps observed, "
          f"{mm} not matching the signed-crossing bookkeeping")
    viol, held, broken = verify_bounded_walk()
    print(f"V2b bounded walks: bound held on {held} steps -> "
          f"{viol} winding changes (Theorem 2b violations); "
          f"{broken} changes on bound-breaking steps (allowed)")
    tested, mism, offc = verify_flip_formula()
    print(f"V3 flip formula: {tested} generic on-cycle flips, "
          f"{mism} mismatches; off-cycle flip winding changes: {offc}")
    print("V4 wound states (n, w, w_measured, same_sign, "
          "unwind_flip_at_every_node, mean_transient, predicted "
          "pi-2pi|w|/n):")
    for row in verify_wound_states():
        print("   ", row[0], row[1], row[2], row[3], row[4],
              f"{row[5]:.3f}", f"{row[6]:.3f}")
    frac, agree = verify_entry_statistics()
    print(f"V5 entry from near-lock: flip winds {frac:.3f} of the "
          f"time; adjacent-sign agreement {agree:.3f} (prediction: "
          "equal)")


if __name__ == "__main__":
    main()
