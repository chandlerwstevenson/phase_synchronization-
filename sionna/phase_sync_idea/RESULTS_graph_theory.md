# Graph theory of open-loop distributed phase sync — theorems and verification

Scope: open-loop distributed carrier-phase synchronization only. This
generalizes the project's 2-node identifiability result (the one-way
observation cannot separate clock phase from path phase) to N nodes and
arbitrary measurement networks. Full derivations are in the docstring of
`openloop_graph_theory.py`; running that file reproduces every number
below. All claims here are proved (elementary linear algebra / graph
theory arguments in the docstring) and then verified numerically; none
depend on the physical simulator, so none can be configuration
artifacts.

Model: each node has a clock phase; a **one-way link** (overhearing a
transmission) measures the clock difference *plus* an unknown
propagation phase specific to that link; a **dedicated two-way
exchange** measures the clock difference alone, but only up to a
half-cycle (the known π ambiguity).

---

## Theorem 1 — who determines the phases: only the two-way skeleton

**Statement.** The number of undetermined degrees of freedom equals the
number of connected components of the *two-way subgraph* (isolated
nodes count). The array's phases are determined, up to one global
phase, exactly when the two-way exchanges alone connect every node.
One-way links contribute *exactly nothing* to phase identifiability —
in any number, in any arrangement, including cycles.

**Why cycles don't help.** A loop of one-way links does give a closure
constraint — the clock terms cancel around the loop — but that
constraint binds only the sum of the unknown propagation phases. Along
any unidentifiable direction, the induced propagation-phase shifts form
a gradient field, whose loop sums vanish automatically. So loops
constrain the nuisance parameters, never the clock/path split itself.

**Verification.** 150 random mixed graphs (5–20 nodes, random one-way
and two-way edge counts): predicted vs computed nullity matched in all
150. Targeted structures:

| structure (8 nodes) | computed nullity | predicted |
|---|---|---|
| one-way ring only | 8 | 8 (nothing identifiable) |
| two-way path only | 1 | 1 (fully identifiable) |
| one-way ring + two-way path | 1 | 1 |
| two-way covers 4 of 8 nodes | 5 | 5 |

**Theorem 1b (frequency rides every edge).** Differencing a one-way
link's measurements over time cancels its constant propagation phase,
so *every* link — either type — cleanly measures the frequency
difference. Hence frequency is determined by the *union* graph while
phase needs the two-way subgraph: **phase and frequency live on two
different networks.** Design reading: cheap overheard traffic
synchronizes frequency everywhere; static phase is pinned only by the
dedicated-exchange skeleton. Verified: 100 random graphs, 0 mismatches.

## Theorem 2 — the half-cycle ambiguity is node flips; cycles detect but never resolve

**Statement.** For a connected two-way network of n nodes, the residual
ambiguity after all measurements is exactly the set of "flip a subset
of nodes by half a cycle" operations: 2^(n−1) configurations,
*independent of how many loops the network has*. Loops eliminate
inconsistent per-edge branch guesses, but the surviving set (the
graph's cut space) always has exactly 2^(n−1) elements — the same as a
tree. So redundant links never resolve the ambiguity.

**What loops do buy — error detection.** A *single edge's* branch being
wrong (a measurement error rather than a coherent node flip) violates
loop parity and is detectable — exactly when that edge lies on a loop.
On a bridge it is undetectable, because it coincides with a legitimate
node flip of everything on one side. Valid configurations form the cut
space, the dual of the graph's cycle code: the sync network *is* a
binary code, loops are its parity checks.

**Resolution cost.** Each one-bit branch check (the sign-of-cosine test
from the 2-node work) resolves one edge. Minimum to resolve an n-node
component: n−1 bits, and the checked edges must form a connected
spanning subgraph — any spanning tree is optimal; checks on loop edges
are redundant (implied by cut-consistency).

**Verification.** Exhaustive enumeration over all branch assignments:

| structure | nodes | valid configs found | predicted 2^(n−1) |
|---|---|---|---|
| path (tree) | 5 | 16 | 16 |
| ring (1 loop) | 5 | 16 | 16 |
| theta graph (2 loops) | 5 | 16 | 16 |
| complete graph K4 | 4 | 8 | 8 |
| K5 minus an edge | 5 | 16 | 16 |

Every valid configuration verified to be a node-flip (cut) of the
truth, zero exceptions. Detectability: flipping each edge of a
ring-plus-pendant graph — all four ring edges violated parity, the
pendant bridge did not (exactly as predicted).

## Theorem 3 — the two-resistance law

**Statement.** Treat the sync network as an electrical circuit. Then:

- **Static phase accuracy** between two nodes = (two-way measurement
  variance) × (effective resistance between them in the circuit built
  from *two-way edges only*). One-way links conduct nothing: a single
  link, six parallel links, a full one-way ring, or fifty repeated
  measurements of one link (sharing one propagation phase) all leave
  the phase accuracy unchanged to machine precision — even when the
  one-way links are 400× less noisy.
- **What "two-way" really means:** a reciprocal pair of one-way links
  sharing one propagation phase behaves exactly as a two-way edge with
  half the single-link variance. The conducting/insulating dichotomy is
  about *sharing the propagation phase between opposite-signed
  measurements*, not about protocol.
- **Frequency accuracy** = (rate-measurement variance) × effective
  resistance in the circuit built from *all* edges. One-way links
  conduct frequency but insulate phase.

**Verification** (best-linear-unbiased/Fisher computation vs resistance
formula):

| network | estimator variance | resistance prediction |
|---|---|---|
| 8-chain, end to end | 0.280000 | 0.280000 |
| 8-ring, opposite nodes | 0.080000 | 0.080000 |
| 8-star, leaf to leaf | 0.080000 | 0.080000 |
| random 10-node graph | 0.067826 | 0.067826 |

Zero-contribution checks: max covariance change from adding one-way
structure of any kind: 5×10⁻¹² or below. Shared-phase reciprocal pair
vs explicit two-way edge at half variance: 0.010000 vs 0.010000.
Frequency on the union graph: 0.071250 vs 0.071250.

## Consequences worth stating (all follow from the theorems)

1. **A chain of dedicated exchanges degrades linearly with hops**
   (resistance of a path ~ length): open-loop daisy-chained arrays pay
   variance proportional to network diameter; rings quarter it; a hub
   caps leaf-to-leaf variance at two hops regardless of size.
2. **Overheard traffic can never substitute for the two-way skeleton in
   a static snapshot** — its role is frequency (and, dynamically,
   drift tracking), which is exactly the division of labor measured in
   the piggyback experiments, now explained structurally.
3. **Branch-check budgeting is a spanning-tree problem**: n−1 one-bit
   checks per component, placed on any spanning tree; loop edges are
   free parity checks that detect (and can localize) branch errors.

## Known vs new — flags for the literature audit

Suspected known (must be checked and cited): the cut/cycle-space
duality (textbook algebraic graph theory); best-linear-unbiased
variance = effective resistance for relative measurements
(Barooah–Hespanha line); the node-flip ambiguity structure is formally
ℤ₂ group synchronization (Bandeira et al.) — our Theorem 2 is that
structure arising from the half-difference measurement, plus the
detection/resolution and check-placement readings.

Believed new structure: the mixed-type identifiability formula
(Theorem 1: one-way edges exactly null for phase; gauge dimension =
two-way component count — the N-node form of this project's retained
2-node result); the phase/frequency two-network separation (1b, 3c);
the propagation-phase-sharing reframing of "two-way" (3b); and the
sync-protocol consequences (spanning-tree check placement, loops as
branch-error syndromes, resistance scaling of open-loop topologies).

## Verification provenance

Single runnable harness: `openloop_graph_theory.py` (numpy only, no
physical simulation, seed fixed). Totals: 250 random-graph
identifiability/frequency cases with 0 mismatches; 5 exhaustive branch
enumerations, all exactly 2^(n−1), 0 cut-membership violations; all
resistance identities to 6 decimals; all zero-information checks at
≤5×10⁻¹².
