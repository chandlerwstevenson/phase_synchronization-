"""Render the actual measurement topologies used in the fork and
Pareto campaigns - the graphs themselves, drawn from the SAME builder
functions and node positions the experiments ran on (imported from
dirA2_threelaw / dirA_selection), so these are the experimental
graphs, not illustrations.

Figure 4: the five N=8 topologies. Nodes at their true deployment
coordinates (meters); each line = one two-way synchronization link
(the pairwise exchange whose half-difference measures that pair's
clock offset). Node labels = station indices.

Figure 5: what the update protocols do on the same graph (star):
simultaneous (all links active together), sequential (edge-colored
turns), directed (arrows child->parent toward the elected root, node
0, drawn filled).

Plain default matplotlib; only node-identification labels in axes.
"""

import sys

sys.path.insert(0, ".")
sys.path.insert(0, "../phase_sync_idea")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dirA_selection import make_positions
from dirA2_threelaw import get_sigma2, topology

positions = make_positions("uniform")
sigma2 = get_sigma2(positions)
n = positions.shape[0]

NAMES = [
    ("star", "star (hub degree 7)"),
    ("deg3tree", "degree-3 tree"),
    ("mst", "min-variance tree"),
    ("chain", "chain"),
    ("ring", "ring (chain + 1 chord)"),
]


def draw_graph(ax, edges, title, arrows_to_parent=None, colors=None):
    for k, (i, j) in enumerate(edges):
        color = colors[k] if colors else "C0"
        ax.plot(
            [positions[i, 0], positions[j, 0]],
            [positions[i, 1], positions[j, 1]],
            "-",
            color=color,
            linewidth=1.5,
            zorder=1,
        )
    if arrows_to_parent:
        for child, parent in arrows_to_parent:
            dx = positions[parent] - positions[child]
            start = positions[child] + 0.25 * dx
            ax.annotate(
                "",
                xy=positions[child] + 0.75 * dx,
                xytext=start,
                arrowprops=dict(arrowstyle="->", color="C2", lw=2),
                zorder=2,
            )
    root_mask = np.zeros(n, dtype=bool)
    root_mask[0] = arrows_to_parent is not None
    ax.scatter(
        positions[~root_mask, 0], positions[~root_mask, 1],
        s=120, facecolor="white", edgecolor="black", zorder=3,
    )
    if root_mask.any():
        ax.scatter(
            positions[0, 0], positions[0, 1], s=160,
            facecolor="black", edgecolor="black", zorder=3,
        )
    for i in range(n):
        ax.annotate(
            str(i), positions[i], ha="center", va="center",
            fontsize=8, zorder=4,
            color="white" if root_mask[i] else "black",
        )
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


# ---- Figure 4: the five topologies -------------------------------
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for ax, (name, label) in zip(axes.flat, NAMES):
    draw_graph(ax, topology(name, positions, sigma2), label)
axes.flat[-1].axis("off")
fig.tight_layout()
fig.savefig("paper_figures/fig4_topologies.png", dpi=200)
plt.close(fig)

# ---- Figure 5: protocols on the star ------------------------------
star = topology("star", positions, sigma2)
hub = max(range(n), key=lambda v: sum(v in e for e in star))
# directed: BFS from node 0 (the elected root, per dirA2 docstring)
adjacency = {i: [] for i in range(n)}
for i, j in star:
    adjacency[i].append(j)
    adjacency[j].append(i)
parent = {0: None}
frontier = [0]
while frontier:
    nxt = []
    for v in frontier:
        for w in adjacency[v]:
            if w not in parent:
                parent[w] = v
                nxt.append(w)
    frontier = nxt
child_parent = [(c, p) for c, p in parent.items() if p is not None]

# sequential: proper edge coloring of the star = each edge its own turn
turn_colors = [f"C{k % 10}" for k in range(len(star))]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
draw_graph(axes[0], star, "simultaneous: all links correct at once")
draw_graph(
    axes[1], star,
    "sequential: links fire in colored turns",
    colors=turn_colors,
)
draw_graph(
    axes[2], star,
    "directed: corrections toward elected root (filled node)",
    arrows_to_parent=child_parent,
)
fig.tight_layout()
fig.savefig("paper_figures/fig5_protocols.png", dpi=200)
plt.close(fig)

print("hub node:", hub)
print("wrote paper_figures/fig4_topologies.png, fig5_protocols.png")
