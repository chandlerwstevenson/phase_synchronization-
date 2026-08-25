# Abstracts — one per direction (2026-08-24)

Each abstract is scoped to what its run established; caveats are in
the text, not omitted. Numbers trace to RESULTS_A–D.md in this folder.

---

## A — Stability, Not Information, Governs Synchronization Topology at the Waveform Level

Selecting which links of a distributed array carry synchronization
measurements is usually framed as an estimation problem: choose the
graph that minimizes phase-error covariance under an airtime budget.
We show, in waveform-level simulation of an eight-node actuated array
with full oscillator and RF impairments, that estimation quality does
not decide the outcome. At equal airtime, measured coherent gain is
governed by three dynamical constraints, each isolated by controls: a
cadence ceiling (links serviced less often than every ~2 update
intervals destabilize — the driver is frequency walk between
services, not measurement noise, established by pilot-length
controls), a degree ceiling (a seven-link hub collapses to 24% gain
amid branch-flip storms while a degree-3 tree holds 80% at identical
cost), and cycle exclusion (one added chord halves a chain's gain,
consistent with the predicted winding states of the measurement
graph's cycle space). Consequently, a selector that greedily
maximizes predicted gain loses to simple minimum-variance spanning
trees in every tested budget and geometry — it optimizes an objective
that does not bind. All three constraints are properties of the
symmetric simultaneous-update control law; whether they persist under
sequential and directed control is the decisive open experiment, and
either outcome constrains topology design in a way current selection
frameworks do not represent.

## B — Minimum Phase Error Is the Wrong Objective for Synchronization Topology

Synchronization-topology design conventionally minimizes
phase-estimation error. A beamforming array's actual objective —
expected coherent gain — weights each node's phase error by its beam
amplitude, and we show these two objectives select different
topologies, with quantified cost. Using an exact expression for
expected gain over the error covariance of a selected measurement
graph (validated against Monte-Carlo draws to 0.013%), and exhaustive
enumeration so every optimum is exact, we find the divergence is
specifically an amplitude-heterogeneity phenomenon: with equal
amplitudes the objectives essentially coincide at realistic error
levels, while with heterogeneous amplitudes they diverge in 16 of 18
configurations, and adopting the error-minimizing topology forfeits
up to 8.6 points of coherent gain at identical airtime. The mechanism
is closed-form: to first order, expected gain is an amplitude-weighted
Kirchhoff index, whereas the conventional objective is unweighted. A
sharp instance makes the distinction unambiguous — under a
spanning-tree budget all star topologies share one total phase error,
so the conventional objective provably cannot choose the hub, while
the gain objective places it at the strongest node for +3.6 points.
In some configurations the gain-optimal measurement graph is
deliberately disconnected, abandoning a weak node to tighten the
strong core — a choice the estimation objective can never make.

## C — Which Radios Deserve Synchronization: Joint Participation and Topology Selection Under an Airtime Budget

When synchronization airtime is scarce, a distributed array faces a
coupled choice: which radios participate in the coherent beam, and
which measurement links keep them synchronized — a node left out
contributes no amplitude but costs no airtime. Solving this joint
problem exactly at small scale (verified against full enumeration)
and by matched heuristic at larger scale, with phase-and-frequency
error dynamics propagated through the schedule each budget affords,
we find that partial participation wins within a budget *window*
rather than universally: below ~2% airtime nothing coheres and
benching only loses power; near 5%, every tested instance across
geometries and amplitude models benches three of eight radios and
beats the best full-array topology by up to 78% relative gain; by 10%
the full array is optimal again. The optimal roster grows nestedly
with budget, and its choices follow amplitude-versus-cost tradeoffs
no strongest-k or nearest-k heuristic reproduces — retaining an
expensive high-amplitude outlier while benching a cheap weak one.
Waveform-level validation agrees to 0.3% in the single-hop regime;
multi-hop validation is currently bounded by the simultaneous-update
control law's known convergence tax, and completing it under
sequential control is the identified next step.

## D — When to Rewire: Dynamic Synchronization-Link Selection Under Blockage

A distributed array's synchronization topology is chosen for a
propagation environment that does not stay fixed. We evaluate dynamic
re-selection at the waveform level: an eight-node array whose two
most load-bearing measurement links lose 20 dB for a hundred-interval
episode, comparing static topology, channel-aware re-selection
(triggered by the link-quality metric the synchronization exchanges
already compute), predicted-gain-triggered re-selection, and an
oracle — all at equal airtime. The practical channel-aware policy
captures 95% of the oracle's benefit, restoring 0.938 of coherent
gain through the episode versus 0.683 static, while producing zero
false switches across all undisturbed control runs. Two boundaries
emerge with design consequences: re-selection pays only for
disturbances outlasting ~50 update intervals — on shorter episodes
the switching transient can outlive the blockage and lose gain — and
partial degradation that leaves a link detectable is best tolerated
rather than rewired. Mechanistically, the dominant harm under partial
blockage is not gradual noise inflation but discrete churn: storms of
branch-ambiguity flips and forced re-acquisitions, which the
re-selection removes. The costlier predicted-gain trigger
underperforms the simple link-quality trigger (61% of oracle) because
evidence of harm arrives later than evidence of link change.

---

## A2 — the earned abstract (post three-law fork, 2026-08-24)

**Synchronization Topology and Update Protocol Are Not Separable:
Three Mechanisms Governing the Stability of Distributed Coherent
Arrays**

The measurement topology of an over-the-air synchronized array — who
exchanges synchronization signals with whom — is conventionally
chosen to minimize phase-estimation error, and prior analyses relate
topology to estimation covariance or consensus convergence rate. We
show, in waveform-level simulation of actuated eight- and
sixteen-node arrays with full oscillator and RF impairments, that the
topology's effect on actual coherent beamforming gain is governed by
three mechanisms with three different origins, established by
pre-registered predictions and controlled comparison across three
update protocols (simultaneous, sequential, and directed-tree). A
*cadence* mechanism is physical: links serviced less often than
roughly every two update intervals destabilize through oscillator
frequency walk, under every protocol, persisting at zero frequency
offset — though its catastrophic expression is protocol-dependent,
since bidirectional protocols amplify branch-ambiguity errors into
flip cascades that a directed protocol confines to one subtree. A
*degree* mechanism is numerical: a seven-link hub collapses to 24%
gain under simultaneous updates — the divergence its linearized
(Jacobi) iteration predicts — and is cured by direction or damping
(92%), but *not* by sequencing alone, because turn-taking dilutes
each of the hub's links to every-seventh-slot service and re-enters
the cadence mechanism. A *cycle* mechanism is topological: adding one
chord to a chain halves its gain under both bidirectional protocols,
and every settled impaired state carries an exactly quantized winding
number — pairwise offsets summing to integer multiples of 2π around
each cycle — and we prove this quantization holds at every instant
(a residue identity of the wrapped cycle sum), is conserved under any
protocol whose per-step corrections keep cycle edges away from ±π,
and renders wound states metastable under *every* local update rule:
escape requires transit through an anti-lock event of magnitude
π − 2π|w|/n, a barrier we verify against the measured transients.
Linearized spectral radii rank the protocols correctly but are
necessary rather than sufficient; the binding nonlinearity is the
modulo-π branch layer of the two-way measurement. The design
consequence is an inversion: the star topology is the *worst* graph
under simultaneous updates and the *best* (99.4% gain) under directed
updates, and at sixteen nodes every bidirectional protocol collapses
on every topology while the directed protocol holds 89–99% —
synchronization topology and update protocol cannot be chosen
independently, and selection frameworks that optimize the graph
against an estimation objective alone, without the protocol in the
loop, optimize the wrong problem.

*(Numbers: RESULTS_A2.md, 118 cells, symmetric arm bit-identical to
RESULTS_A; winding quantization independently measured in
../phase_sync_idea/RESULTS_topology.md campaign 4 and predicted by
openloop_graph_theory.py. Simulation only, N=8/16, one geometry,
static channel.)*
