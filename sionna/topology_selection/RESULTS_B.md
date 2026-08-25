# Direction B — the coherent-gain objective is not the phase-MSE objective

Script: `dirB_objective_gap.py` (predictions P1–P4 written in the
docstring before any run; data in `dirB_results.json`; figure
`figures/dirB_mismatch_cost.png`). All selection searches are
exhaustive enumerations of edge subsets under a budget, so results are
optima, not heuristics. Seeds 0–2 throughout.

## The exact gain formula (validated, P1)

For a selected two-way edge set E with per-measurement variance r₂,
Gaussian phase errors give exactly

    E[G(E)] = Σ_ik a_i a_k · exp(−r₂ · R_ik(E)/2) / (Σ a_i)²

with R_ik the effective resistance over E, and pairs split across
components contributing zero. Validation: against 10⁶ Monte-Carlo
Gaussian draws from the exact BLUE covariance the formula agrees to
0.013% (inside Monte-Carlo error); the resistance-based pair variances
match the independent BLUE covariance computation to 5·10⁻¹⁵. The
per-pair Gaussian relation is textbook robust beamforming — what is
used here is its consequence for *topology selection*.

## Finding 1 — divergence is real, and amplitude weighting is the operative mechanism (P2, P3)

Exhaustive search, N=6, budgets {5,6,7} edges, residuals
r₂ ∈ {0.05, 0.5} rad², 3 seeds (36 configurations; divergence =
the MSE-optimal topology actually loses gain, not merely a tie —
symmetric ties such as the star family sharing one Kirchhoff index are
excluded by construction):

- Equal amplitudes: **0 of 18** configurations diverge.
- Heterogeneous amplitudes (a = exp(0.8·z)): **16 of 18** diverge.
- Cost of using the MSE-optimal topology: up to **8.6 points of
  coherent gain** (seed 2, budget 5, r₂ = 0.5: gain-optimal 83.4%,
  MSE-optimal 74.8%).

Sweep of amplitude heterogeneity h (budget 5, r₂ = 0.2): mismatch cost
rises monotonically from exactly 0 at h = 0 to mean 2.7 / max 4.8
points at h = 1.6. First-order mechanism, as derived: linearizing the
exponential, maximizing gain is minimizing the *amplitude-weighted*
Kirchhoff index Σ a_i a_k R_ik — different from the unweighted index
whenever amplitudes differ.

## Finding 2 — the saturation mechanism exists but is impractical (P2)

With equal amplitudes the objectives can still diverge through
curvature (exp(−x) is convex: at fixed total resistance, gain prefers
an uneven resistance profile — writing off an already-bad pair is free
for gain, costly for MSE). Measured: zero divergence through
r₂ = 1.0 rad²; first divergence at r₂ = 2.0 rad² (3.1 points). That
corresponds to ~81° rms pair error — far beyond any useful sync
operating point at this size. Honest conclusion: in practice the
objective mismatch is an *amplitude-heterogeneity* phenomenon, not a
saturation phenomenon; the regime condition "objectives coincide when
r₂·R ≪ 1 and amplitudes are equal" has both branches confirmed.

## Finding 3 — the gain objective abandons nodes (P4)

In 3 of the 16 divergent configurations the gain-optimal edge set is
**disconnected**: it drops a weak-amplitude node entirely and spends
the freed budget tightening the strong core. Under the MSE objective
this is forbidden (infinite variance); under the gain objective it is
merely a zero contribution from the abandoned pairs — often worth it.
Node selection is therefore not an add-on to topology selection; it
*emerges* from the correct objective. (Feeds direction C.)

## Finding 4 — the sharpest teachable instance (N=7)

Amplitudes (1.12, 0.89, 1.78, 1.10, 0.62, 1.39, **3.23**), budget 6
(tree), r₂ = 0.4. Every star has the same Kirchhoff index, so the MSE
objective *cannot distinguish which node should be the hub* (the
enumeration returns an arbitrary star, on node 3). The gain objective
picks the star centered on the strongest-amplitude node — worth
**+3.6 points** of coherent gain over the MSE-optimal choice. One
sentence: MSE treats the hub as a graph-theoretic choice; gain knows
the hub should be the node whose phase matters most.

## Known vs claimed

Known (cited, not claimed): BLUE variance = effective resistance
(Barooah–Hespanha; Karp et al.); Fisher information as weighted graph
Laplacian (Howard et al.); E[e^{jε}] = e^{−σ²/2} robust-beamforming
relation. Claimed as this direction's content: the exact
topology-level gain formula built from pair resistances; the
demonstrated argmax divergence between gain and MSE topology selection
with its two-mechanism explanation (first-order amplitude weighting,
second-order saturation) and regime condition; the emergence of node
abandonment from the gain objective; the hub-selection instance. The
literature sibling audits these.

## Limits

Exhaustive enumeration at N ≤ 7 (optimality guaranteed, scale not
addressed — direction A owns algorithms); uniform edge costs and
uniform r₂ per measurement (SNR-weighted edges are direction A's
axis); static Gaussian errors from the BLUE covariance, not waveform
runs; amplitudes treated as known.
