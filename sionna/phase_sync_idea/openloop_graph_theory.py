"""Graph/gauge theory of open-loop distributed carrier-phase sync.

Generalizes the 2-node identifiability result (observability_analysis.py:
one-way data sees only theta+psi and omega; the null direction is the
theta/psi split) to N nodes and arbitrary measurement graphs.

SETUP
-----
N nodes with clock phases theta_i (and frequencies omega_i). Two edge
types in the measurement graph:

  ONE-WAY edge e = (i -> j):  observes  z_e = theta_i - theta_j + psi_e
      with psi_e an unknown per-edge propagation phase (constant while
      the environment is static).
  TWO-WAY edge f = {i, j}:    observes  y_f = theta_i - theta_j
      (reciprocity cancels psi), but only modulo pi (the half-difference
      doubling ambiguity, pi_ambiguity_analysis.py).

Unknowns x = (theta in R^N, psi in R^{m1}). Stacking observations, the
(linearized, mod-free) observation matrix is the block form

        H = [ B1^T   I_{m1} ]        B1 = signed incidence of one-way
            [ B2^T   0      ]        B2 = signed incidence of two-way

======================================================================
THEOREM 1 (gauge dimension / identifiability).
    nullity(H) = c2,
the number of connected components of the TWO-WAY subgraph counted over
all N vertices (isolated vertices are components). The phases are
identifiable up to one global phase iff the two-way edges alone form a
connected spanning subgraph. One-way edges contribute EXACTLY NOTHING
to phase identifiability - regardless of how many there are and
regardless of their cycle structure.

Proof. H(dtheta, dpsi) = 0 iff (a) B2^T dtheta = 0, i.e. dtheta is
constant on each two-way component, and (b) dpsi = -B1^T dtheta, which
is satisfiable for ANY dtheta because each one-way edge has its own
free psi. So the null space is {(dtheta, -B1^T dtheta): dtheta constant
on two-way components}, of dimension c2. One-way cycles do give closure
constraints (the theta's telescope out of a cycle sum, so the cycle sum
of z equals the cycle sum of psi) - but those constraints bind psi
combinations only; along any null direction the induced dpsi is a
gradient field (-B1^T dtheta), whose cycle sums vanish identically, so
cycles never shrink the null space. QED.

Consistency with the 2-node result: N = 2, one one-way edge: c2 = 2
(two isolated vertices in the two-way subgraph), nullity 2 = the global
phase plus exactly one nontrivial direction, which is (1, 0 | -1) -
the (theta, psi) split direction retained from the 2-node analysis.

THEOREM 1b (frequency rides every edge).
With observations at two or more epochs, differencing a one-way edge's
observations cancels the constant psi_e:
    z_e(t2) - z_e(t1) = (omega_i - omega_j)(t2 - t1) + noise,
so EVERY edge (either type) observes the frequency difference cleanly.
Hence: omega is identifiable up to a global offset iff the UNION graph
G1 u G2 is connected, while theta needs the TWO-WAY subgraph alone to
be connected (Theorem 1). Phase and frequency live on two different
graphs. (Design reading: cheap one-way traffic synchronizes frequency
everywhere; static phase is pinned only by the two-way skeleton.)

======================================================================
THEOREM 2 (branch ambiguity is node flips; cycles detect, never resolve).
Each two-way edge determines its phase difference only mod pi. For a
two-way component with n nodes (any cycle structure):

 (a) The set of globally consistent branch assignments (one bit s_e per
     edge: difference = d_e or d_e + pi) is exactly a coset of the CUT
     SPACE of the component over F2 - equivalently, every residual
     ambiguity is a set of NODE pi-flips: theta_i -> theta_i + pi for
     i in A, any A subset of the nodes. Cardinality: 2^(n-1)
     (A and its complement give the same configuration up to the
     continuous global phase), INDEPENDENT of the cycle structure.
     A tree with n-1 edges has 2^(n-1) free assignments; adding cycle
     edges adds parity constraints that eliminate exactly the
     non-cut assignments - the count never drops below 2^(n-1) and
     never exceeds it. Cycles do NOT help resolve branches.

 (b) What cycles DO buy is error DETECTION: a spurious flip of a single
     edge's branch (a measurement error, not a node flip) violates the
     parity of every cycle through that edge, so it is detectable iff
     the edge is NOT a bridge. Valid configurations form the cut space
     = the dual of the graph's cycle code; single-edge errors on
     bridges are undetectable (they coincide with a legitimate node
     flip of one side).

 (c) Minimum resolution cost: n-1 one-bit EDGE checks (a check
     measures sign(cos) of one edge's true difference, resolving that
     edge's branch), and the check edges must form a connected spanning
     subgraph of the component - so any spanning tree is optimal, and
     no set of fewer than n-1 checks suffices (the ambiguity has
     2^(n-1) elements). Checks on non-tree edges are redundant: their
     branches are implied by cut-consistency.

Proof sketches. (a) Subtracting the true assignment, an assignment is
consistent iff its F2-difference has zero parity around every cycle,
i.e. lies in the cycle space's orthogonal complement = the cut space,
whose dimension is n - 1; cut vectors are exactly boundaries of node
sets, delta(A). (b) A single-edge vector is a cut iff the edge is a
bridge. (c) The residual ambiguity after checking edge set S is the
set of cuts vanishing on S; delta(A) vanishes on S iff A is a union of
components of (V, S); that is trivial iff (V, S) is connected. QED.

======================================================================
THEOREM 3 (two-resistance law).
Fix the gauge (two-way subgraph connected; ground one node). With
per-observation noise variance r2 on two-way edges and r1 on one-way
edges:

 (a) STATIC PHASE: the BLUE/Fisher variance of any theta_i - theta_j
     is   var = r2 * R_eff^{G2}(i, j),
     the effective resistance between i and j in the electrical network
     built on the TWO-WAY EDGES ONLY (each two-way edge = conductance
     1/r2). One-way edges carry EXACTLY ZERO Fisher information about
     theta - a single edge, parallel duplicated edges (each with its
     own psi), one-way cycles, and even unlimited repeated observations
     of the same edge (which share one psi) all contribute nothing:
     profiling the free psi_e makes the likelihood flat in theta.

 (b) The dichotomy is really about psi-SHARING, not protocol: a
     reciprocal PAIR (i->j) and (j->i) that share one psi is exactly a
     two-way edge - the half-difference cancels psi - and contributes
     conductance with effective variance r1/2 per pair. "Two-way" =
     "psi shared between two opposite-signed observations".

 (c) FREQUENCY: from epoch-differenced observations, every edge of
     either type conducts, so
     var(omega_i - omega_j) = r_dot * R_eff^{G1 u G2}(i, j).
     Phase accuracy and frequency accuracy are governed by two
     DIFFERENT resistance networks on the same nodes: one-way edges
     conduct frequency but insulate phase.

Proof. (a) Full Fisher over (theta, psi) is block [[Bt R^-1 Bt^T, C],
[C^T, D]]; the Schur complement removes each one-way row's information
because its psi appears in that row alone (D block diagonal entry
1/r1 exactly cancels the row's contribution). Equivalent profile
argument: for any theta, psi_e := z_e - (theta_i - theta_j) attains the
maximum, so the profile likelihood is theta-free. The remaining
two-way-only Fisher is (1/r2) * grounded Laplacian of G2, whose inverse
gives effective resistances (Kirchhoff; the estimation form is the
known BLUE-equals-resistance correspondence). QED.

Known-vs-new (for the literature audit): the cut/cycle duality in
Theorem 2 is textbook algebraic graph theory, and mod-pi node-flip
ambiguity is structurally Z2 group synchronization (Bandeira et al.) -
flag; BLUE = effective resistance for relative measurements is
Barooah-Hespanha - flag. What we believe is new structure here: the
MIXED-type identifiability formula (Theorem 1: one-way edges exactly
null for phase, gauge dimension = two-way component count), the
phase/frequency two-graph separation (1b, 3c), the psi-sharing
reframing (3b), and the sync-protocol readings of 2b/2c (cycles as
branch-error syndrome; spanning-tree check placement).

Everything below verifies these claims numerically; run this file.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

RNG = np.random.default_rng(7)
TOL = 1e-8


# ---------------------------------------------------------------------
# graph helpers
# ---------------------------------------------------------------------

def incidence(num_nodes: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """Signed incidence matrix, one column per edge (+1 at head i,
    -1 at tail j for edge (i, j))."""

    matrix = np.zeros((num_nodes, len(edges)))
    for column, (i, j) in enumerate(edges):
        matrix[i, column] = 1.0
        matrix[j, column] = -1.0
    return matrix


def component_count(num_nodes: int, edges: list[tuple[int, int]]) -> int:
    parent = list(range(num_nodes))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in edges:
        parent[find(i)] = find(j)
    return len({find(v) for v in range(num_nodes)})


def observation_matrix(
    num_nodes: int,
    oneway: list[tuple[int, int]],
    twoway: list[tuple[int, int]],
) -> np.ndarray:
    b1 = incidence(num_nodes, oneway)
    b2 = incidence(num_nodes, twoway)
    m1 = len(oneway)
    top = np.hstack([b1.T, np.eye(m1)])
    bottom = np.hstack([b2.T, np.zeros((len(twoway), m1))])
    return np.vstack([top, bottom]) if len(twoway) else top


def nullity(matrix: np.ndarray) -> int:
    if matrix.size == 0:
        return matrix.shape[1]
    s = np.linalg.svd(matrix, compute_uv=False)
    return matrix.shape[1] - int(np.sum(s > 1e-9 * max(1.0, s[0])))


def random_edges(num_nodes, count, directed_ok=True):
    edges = []
    for _ in range(count):
        i, j = RNG.choice(num_nodes, size=2, replace=False)
        edges.append((int(i), int(j)))
    return edges


# ---------------------------------------------------------------------
# Theorem 1 verification
# ---------------------------------------------------------------------

def verify_identifiability(trials: int = 150) -> dict:
    mismatches = 0
    cases = 0
    for _ in range(trials):
        n = int(RNG.choice([5, 10, 20]))
        m1 = int(RNG.integers(0, 2 * n))
        m2 = int(RNG.integers(0, 2 * n))
        oneway = random_edges(n, m1)
        twoway = random_edges(n, m2)
        h = observation_matrix(n, oneway, twoway)
        predicted = component_count(n, twoway)
        measured = nullity(h) if h.size else n + m1
        cases += 1
        if measured != predicted:
            mismatches += 1
    # targeted structures
    targeted = []
    n = 8
    ring = [(k, (k + 1) % n) for k in range(n)]          # one-way ring
    targeted.append(("one-way ring only", nullity(
        observation_matrix(n, ring, [])), n))            # expect c2 = n
    tree = [(k, k + 1) for k in range(n - 1)]            # two-way path
    targeted.append(("two-way path only", nullity(
        observation_matrix(n, [], tree)), 1))
    mixed = nullity(observation_matrix(n, ring, tree))
    targeted.append(("ring + two-way path", mixed, 1))
    half = [(k, k + 1) for k in range(3)]                # 2 components
    targeted.append(("two-way covers half", nullity(
        observation_matrix(n, ring, half)), n - len(half)))
    return {"cases": cases, "mismatches": mismatches, "targeted": targeted}


def verify_frequency_union(trials: int = 100) -> dict:
    """Frequency observations: every edge differences to
    omega_i - omega_j, no psi. Nullity should be the component count of
    the UNION graph."""

    mismatches = 0
    for _ in range(trials):
        n = int(RNG.choice([5, 10, 20]))
        oneway = random_edges(n, int(RNG.integers(0, 2 * n)))
        twoway = random_edges(n, int(RNG.integers(0, 2 * n)))
        union = oneway + twoway
        if not union:
            continue
        h = incidence(n, union).T
        if nullity(h) != component_count(n, union):
            mismatches += 1
    return {"trials": trials, "mismatches": mismatches}


# ---------------------------------------------------------------------
# Theorem 2 verification
# ---------------------------------------------------------------------

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def count_valid_branch_assignments(
    num_nodes: int, twoway: list[tuple[int, int]], theta: np.ndarray
) -> tuple[int, int]:
    """Enumerate all 2^m branch assignments; count those consistent
    with SOME global phase assignment (checked by spanning-tree
    propagation + closure of the remaining edges). Also verify each
    valid assignment differs from the truth by a cut vector."""

    m = len(twoway)
    d = np.array([
        (theta[i] - theta[j]) % np.pi for (i, j) in twoway
    ])
    # spanning tree of the (assumed connected) two-way graph
    tree_columns, seen = [], {0}
    frontier = True
    while frontier:
        frontier = False
        for column, (i, j) in enumerate(twoway):
            if column in tree_columns:
                continue
            if (i in seen) != (j in seen):
                tree_columns.append(column)
                seen |= {i, j}
                frontier = True
    assert len(seen) == num_nodes, "two-way graph must be connected"

    valid = 0
    cut_violations = 0
    truth = None
    for bits in itertools.product([0, 1], repeat=m):
        candidate = d + np.pi * np.array(bits)
        # propagate phases along the tree from node 0
        phase = np.full(num_nodes, np.nan)
        phase[0] = 0.0
        for _ in range(num_nodes):
            for column in tree_columns:
                i, j = twoway[column]
                if not math.isnan(phase[i]) and math.isnan(phase[j]):
                    phase[j] = phase[i] - candidate[column]
                elif not math.isnan(phase[j]) and math.isnan(phase[i]):
                    phase[i] = phase[j] + candidate[column]
        ok = True
        for column, (i, j) in enumerate(twoway):
            gap = wrap(phase[i] - phase[j] - candidate[column])
            if abs(gap) > 1e-6:
                ok = False
                break
        if ok:
            valid += 1
            # is (bits XOR truth-bits) a cut vector? recover node flips
            flips = np.abs(wrap(phase - theta + (theta[0] - phase[0])))
            flip_set = flips > np.pi / 2
            expected_bits = np.array([
                int(flip_set[i] != flip_set[j]) for (i, j) in twoway
            ])
            true_bits = np.array([
                int(abs(wrap((theta[i] - theta[j]) - d[column])) > 1e-6)
                for column, (i, j) in enumerate(twoway)
            ])
            if not np.array_equal(
                (np.array(bits) ^ true_bits), expected_bits
            ):
                cut_violations += 1
            if np.array_equal(np.array(bits), true_bits):
                truth = bits
    assert truth is not None, "true assignment must be counted"
    return valid, cut_violations


def verify_branch_counts() -> list[tuple[str, int, int, int]]:
    """(name, n, measured count, predicted 2^(n-1)) over structures
    with very different cycle content."""

    rows = []
    for name, n, edges in [
        ("path (tree)", 5, [(0, 1), (1, 2), (2, 3), (3, 4)]),
        ("ring (1 cycle)", 5,
         [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]),
        ("theta graph (2 cycles)", 5,
         [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 4)]),
        ("K4 (3 cycles)", 4,
         [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
        ("K5 minus edge", 5,
         [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3),
          (1, 4), (2, 3), (2, 4)]),
    ]:
        theta = RNG.uniform(-np.pi, np.pi, size=n)
        count, cut_violation = count_valid_branch_assignments(
            n, edges, theta
        )
        assert cut_violation == 0
        rows.append((name, n, count, 2 ** (n - 1)))
    return rows


def verify_detectability() -> list[tuple[str, bool, bool]]:
    """Single-edge branch errors: detectable iff the edge is not a
    bridge. Structure: a ring with a pendant node - ring edges are
    cycle edges, the pendant edge is a bridge."""

    n = 5
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (3, 4)]  # last is bridge
    theta = RNG.uniform(-np.pi, np.pi, size=n)
    d = np.array([(theta[i] - theta[j]) % np.pi for (i, j) in edges])
    true_bits = np.array([
        int(abs(wrap((theta[i] - theta[j]) - d[c])) > 1e-6)
        for c, (i, j) in enumerate(edges)
    ])
    results = []
    for column in range(len(edges)):
        bits = true_bits.copy()
        bits[column] ^= 1
        candidate = d + np.pi * bits
        # consistency = zero parity around the single independent cycle
        cycle = [0, 1, 2, 3]  # columns of the ring
        signs = [1, 1, 1, 1]  # orientation consistent by construction
        parity_violated = False
        if column in cycle:
            total = sum(
                s * candidate[c] for s, c in zip(signs, cycle)
            )
            parity_violated = abs(wrap(total)) > 1e-6
        is_bridge = column == 4
        results.append(
            (f"edge {edges[column]}", is_bridge, parity_violated)
        )
    return results


# ---------------------------------------------------------------------
# Theorem 3 verification
# ---------------------------------------------------------------------

def effective_resistance(num_nodes, edges, conductances) -> np.ndarray:
    laplacian = np.zeros((num_nodes, num_nodes))
    for (i, j), g in zip(edges, conductances):
        laplacian[i, i] += g
        laplacian[j, j] += g
        laplacian[i, j] -= g
        laplacian[j, i] -= g
    pinv = np.linalg.pinv(laplacian)
    resistance = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            resistance[i, j] = (
                pinv[i, i] + pinv[j, j] - 2 * pinv[i, j]
            )
    return resistance


def blue_theta_covariance(
    num_nodes, oneway, twoway, r1, r2,
    shared_psi_pairs: list[tuple[int, int]] | None = None,
    repeats_per_oneway: int = 1,
):
    """Fisher information over (theta_2..theta_N grounded, psi...),
    return the marginal covariance of theta differences. Supports
    repeated observations of a one-way edge (sharing one psi) and
    reciprocal shared-psi pairs (rows +/- b with one psi)."""

    rows = []
    weights = []
    m1 = len(oneway)
    num_shared = len(shared_psi_pairs or [])
    psi_total = m1 + num_shared
    width = (num_nodes - 1) + psi_total

    def theta_row(i, j):
        row = np.zeros(width)
        if i > 0:
            row[i - 1] += 1.0
        if j > 0:
            row[j - 1] -= 1.0
        return row

    for column, (i, j) in enumerate(oneway):
        for _ in range(repeats_per_oneway):
            row = theta_row(i, j)
            row[(num_nodes - 1) + column] = 1.0
            rows.append(row)
            weights.append(1.0 / r1)
    for pair_index, (i, j) in enumerate(shared_psi_pairs or []):
        column = (num_nodes - 1) + m1 + pair_index
        forward = theta_row(i, j)
        forward[column] = 1.0
        reverse = -theta_row(i, j)
        reverse[column] = 1.0
        rows.extend([forward, reverse])
        weights.extend([1.0 / r1, 1.0 / r1])
    for (i, j) in twoway:
        rows.append(theta_row(i, j))
        weights.append(1.0 / r2)

    h = np.array(rows)
    fisher = h.T @ np.diag(weights) @ h
    covariance = np.linalg.pinv(fisher)
    return covariance[: num_nodes - 1, : num_nodes - 1]


def pair_variance(theta_covariance, i, j):
    full = np.zeros(
        (theta_covariance.shape[0] + 1, theta_covariance.shape[0] + 1)
    )
    full[1:, 1:] = theta_covariance
    return full[i, i] + full[j, j] - 2 * full[i, j]


def verify_resistance() -> list[tuple[str, float, float]]:
    rows = []
    r2 = 0.04
    for name, n, twoway in [
        ("chain N=8, ends", 8, [(k, k + 1) for k in range(7)]),
        ("ring N=8, opposite", 8,
         [(k, (k + 1) % 8) for k in range(8)]),
        ("star N=8, two leaves", 8, [(0, k) for k in range(1, 8)]),
        ("random N=10", 10, None),
    ]:
        if twoway is None:
            twoway = [(k, k + 1) for k in range(9)] + random_edges(10, 6)
        i, j = (0, n - 1) if "opposite" not in name else (0, 4)
        if "leaves" in name:
            i, j = 1, 7
        resistance = effective_resistance(
            n, twoway, [1.0 / r2] * len(twoway)
        )
        predicted = resistance[i, j]  # conductance 1/r2 => R in units of r2
        covariance = blue_theta_covariance(n, [], twoway, 0.01, r2)
        measured = pair_variance(covariance, i, j)
        rows.append((name, measured, predicted))
    return rows


def verify_oneway_zero_information() -> list[tuple[str, float]]:
    """Adding one-way structure of every kind must leave the theta
    covariance unchanged to machine precision."""

    n, r1, r2 = 8, 0.0001, 0.04  # one-way even 400x LESS noisy
    twoway = [(k, k + 1) for k in range(7)]
    base = blue_theta_covariance(n, [], twoway, r1, r2)
    results = []
    single = blue_theta_covariance(n, [(0, 7)], twoway, r1, r2)
    results.append(("single long-range one-way",
                    float(np.max(np.abs(single - base)))))
    parallel = blue_theta_covariance(
        n, [(0, 7)] * 6, twoway, r1, r2
    )
    results.append(("6 parallel one-way (distinct psi)",
                    float(np.max(np.abs(parallel - base)))))
    ring = blue_theta_covariance(
        n, [(k, (k + 1) % n) for k in range(n)], twoway, r1, r2
    )
    results.append(("one-way ring (cycle closure)",
                    float(np.max(np.abs(ring - base)))))
    repeated = blue_theta_covariance(
        n, [(0, 7)], twoway, r1, r2, repeats_per_oneway=50
    )
    results.append(("50 repeats sharing one psi",
                    float(np.max(np.abs(repeated - base)))))
    return results


def verify_shared_psi_pair() -> tuple[float, float]:
    """A reciprocal pair sharing one psi must act as a two-way edge of
    variance r1/2: compare against an explicit two-way edge."""

    n, r1, r2 = 6, 0.02, 0.04
    twoway = [(k, k + 1) for k in range(4)]  # leaves node 5 dangling
    with_pair = blue_theta_covariance(
        n, [], twoway, r1, r2, shared_psi_pairs=[(4, 5)]
    )
    equivalent = blue_theta_covariance(
        n, [], twoway + [(4, 5)], r1, r1 / 2.0
    )
    # compare the (4,5) pair variance under both constructions
    return (
        pair_variance(with_pair, 4, 5),
        pair_variance(equivalent, 4, 5),
    )


def verify_frequency_resistance() -> tuple[float, float]:
    """Frequency Fisher uses ALL edges (epoch-differenced, no psi):
    variance = r_dot * R_eff on the union graph."""

    n, r_dot = 8, 0.09
    twoway = [(k, k + 1) for k in range(7)]
    oneway = [(0, 7), (2, 6)]
    union = twoway + oneway
    resistance = effective_resistance(
        n, union, [1.0 / r_dot] * len(union)
    )
    predicted = resistance[0, 7]
    h = incidence(n, union).T[:, 1:]  # ground node 0
    fisher = h.T @ h / r_dot
    covariance = np.linalg.pinv(fisher)
    measured = pair_variance(covariance, 0, 7)
    return measured, predicted


# ---------------------------------------------------------------------

def main() -> None:
    print("THEOREM 1 - identifiability: nullity(H) = two-way components")
    result = verify_identifiability()
    print(f"  random graphs: {result['cases']} cases, "
          f"{result['mismatches']} mismatches")
    for name, measured, predicted in result["targeted"]:
        print(f"  {name:<24} nullity {measured} (predicted {predicted})")

    result = verify_frequency_union()
    print("THEOREM 1b - frequency graph = union graph: "
          f"{result['trials']} cases, {result['mismatches']} mismatches")

    print("\nTHEOREM 2 - branch ambiguity count = 2^(n-1), any cycles")
    for name, n, measured, predicted in verify_branch_counts():
        print(f"  {name:<24} n={n}  measured {measured:>4}  "
              f"predicted {predicted:>4}")
    print("  detectability (single-edge branch error):")
    for name, is_bridge, detected in verify_detectability():
        kind = "bridge" if is_bridge else "cycle edge"
        print(f"    {name:<14} {kind:<11} parity violated: {detected}")

    print("\nTHEOREM 3 - two-resistance law")
    print("  static phase = r2 * R_eff on two-way edges only:")
    for name, measured, predicted in verify_resistance():
        print(f"    {name:<24} BLUE {measured:.6f}  "
              f"resistance {predicted:.6f}")
    print("  one-way edges carry zero phase information "
          "(max |cov change|):")
    for name, delta in verify_oneway_zero_information():
        print(f"    {name:<34} {delta:.2e}")
    pair, equivalent = verify_shared_psi_pair()
    print(f"  shared-psi reciprocal pair vs two-way(r1/2): "
          f"{pair:.6f} vs {equivalent:.6f}")
    measured, predicted = verify_frequency_resistance()
    print(f"  frequency on union graph: BLUE {measured:.6f}  "
          f"resistance {predicted:.6f}")


if __name__ == "__main__":
    main()
