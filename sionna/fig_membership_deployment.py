"""Deployment map for the membership studies.

Top-down view of the N=10, seed-0 deployment used throughout the
membership family: station positions (reference marked distinctly),
the two 1.2 km edge detection targets, and the near/edge user
locations used by the throughput metrics. Plain default matplotlib.
"""

from __future__ import annotations

import numpy as np

from fig_membership_common import save_fig
from ota_sync.network import place_stations
import matplotlib.pyplot as plt

N = 10
SEED = 0
RADIUS_M = 500.0


def main() -> None:
    positions = place_stations(N, RADIUS_M, SEED)
    centroid = positions.mean(axis=0)
    edge_targets = np.array(
        [centroid + [1200.0, 150.0], centroid + [-1200.0, 150.0]]
    )
    near_user = centroid + np.array([400.0, 0.0])
    edge_user = edge_targets[0]

    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    axis.scatter(
        positions[1:, 0], positions[1:, 1], marker="o", s=60,
        color="C0", label="stations",
    )
    axis.scatter(
        positions[0, 0], positions[0, 1], marker="^", s=110,
        color="C1", label="reference station",
    )
    axis.scatter(
        edge_targets[:, 0], edge_targets[:, 1], marker="x", s=90,
        color="C3", label="edge detection targets (1.2 km)",
    )
    axis.scatter(
        [near_user[0]], [near_user[1]], marker="s", s=70,
        color="C2", label="near user (throughput metric)",
    )
    axis.scatter(
        [edge_user[0]], [edge_user[1]], marker="D", s=55,
        color="C4", label="edge user (co-located with target A)",
    )
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_aspect("equal")
    axis.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8,
    )
    axis.set_title(f"Deployment map (N={N}, seed {SEED})")
    print("saved", save_fig(figure, "membership_deployment_map"))
    print("positions:\n", np.round(positions, 1))
    print("centroid:", np.round(centroid, 1))


if __name__ == "__main__":
    main()
