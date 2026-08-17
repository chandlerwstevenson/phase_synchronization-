"""Scene-layout figures for the ray-traced environments (plain
default matplotlib) plus the N=6 star deployment map.

Produces, in figures/studies/:
  scene_tworay_layout.png
  scene_urban_los_layout.png
  scene_urban_nlos_layout.png
  clutter_deployment_map.png

Each scene figure is a top-down view: building footprints, the two
stations, and the actual ray-traced propagation path polylines (from
the path solver's interaction vertices). Scene geometry replicates
environment_dependence_study.rt_station_pair_cir exactly, and each
figure run VERIFIES that the replica's path delays match the study's
(raises if not), so the drawing shows the same channel the
simulations used.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from environment_dependence_study import rt_station_pair_cir
from ota_sync.network import place_stations

FIGDIR = Path(__file__).resolve().parent / "figures" / "studies"
DISTANCE_M = 500.0
HEIGHT_M = 15.0


def save(figure, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / f"{name}.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(path)


def scene_footprints(kind: str) -> list[tuple[float, float, float, float]]:
    """(x, y, width, depth) rectangles, replicating the study's
    geometry parameters in the study's own construction order."""

    rects: list[tuple[float, float, float, float]] = []
    if kind in {"urban-los", "urban-nlos"}:
        rects.append((0.0, 60.0, 1.2 * DISTANCE_M, 2.0))
        rects.append((0.0, -60.0, 1.2 * DISTANCE_M, 2.0))
        rng = np.random.default_rng(7)
        for _ in range(2, 8):
            x = rng.uniform(-0.5 * DISTANCE_M, 0.5 * DISTANCE_M)
            sign = float(rng.choice([-1.0, 1.0]))
            y = sign * rng.uniform(90.0, 160.0)
            width = rng.uniform(20.0, 45.0)
            depth = rng.uniform(20.0, 45.0)
            rng.uniform(12.0, 35.0)  # height (unused in top view)
            rects.append((x, y, width, depth))
    if kind == "urban-nlos":
        rects.append((0.0, 0.0, 40.0, 80.0))
    return rects


def trace_paths(kind: str):
    """Re-trace the scene with the study's exact parameters and return
    (polylines, delays). Polyline = tx -> interaction vertices -> rx."""

    import sionna.rt as rt

    from detection.rt_echo import _building_ply, _ground_ply

    scene = rt.load_scene()
    scene.frequency = 915e6
    scene.tx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, pattern="iso", polarization="V"
    )
    scene.rx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, pattern="iso", polarization="V"
    )
    temp_files: list[str] = []
    try:
        ground_material = rt.RadioMaterial(
            "ground-material",
            thickness=10.0,
            relative_permittivity=15.0,
            conductivity=0.035,
        )
        path = tempfile.NamedTemporaryFile(
            "w", suffix=".ply", delete=False
        )
        path.write(_ground_ply(1.5 * max(DISTANCE_M, 1000.0)))
        path.close()
        temp_files.append(path.name)
        scene.edit(
            add=rt.SceneObject(
                fname=path.name,
                name="ground-plane",
                radio_material=ground_material,
            )
        )

        heights = {"urban-los": {}, "urban-nlos": {}}
        rects = scene_footprints(kind)
        building_heights: list[float] = []
        if kind in {"urban-los", "urban-nlos"}:
            building_heights = [25.0, 25.0]
            rng = np.random.default_rng(7)
            for _ in range(2, 8):
                rng.uniform(-0.5 * DISTANCE_M, 0.5 * DISTANCE_M)
                rng.choice([-1.0, 1.0])
                rng.uniform(90.0, 160.0)
                rng.uniform(20.0, 45.0)
                rng.uniform(20.0, 45.0)
                building_heights.append(rng.uniform(12.0, 35.0))
        if kind == "urban-nlos":
            building_heights.append(30.0)
        for index, ((x, y, width, depth), height) in enumerate(
            zip(rects, building_heights)
        ):
            concrete = rt.RadioMaterial(
                f"building-material-{index}",
                thickness=0.3,
                relative_permittivity=5.24,
                conductivity=0.123,
            )
            handle = tempfile.NamedTemporaryFile(
                "w", suffix=".ply", delete=False
            )
            handle.write(_building_ply(x, y, width, depth, height))
            handle.close()
            temp_files.append(handle.name)
            scene.edit(
                add=rt.SceneObject(
                    fname=handle.name,
                    name=f"building-{index}",
                    radio_material=concrete,
                )
            )

        half = DISTANCE_M / 2.0
        scene.add(
            rt.Transmitter("station-a", position=[-half, 0.0, HEIGHT_M])
        )
        scene.add(
            rt.Receiver("station-b", position=[half, 0.0, HEIGHT_M])
        )
        paths = rt.PathSolver()(
            scene,
            max_depth=3,
            los=True,
            specular_reflection=True,
            diffuse_reflection=False,
            refraction=False,
        )
        a, tau = paths.cir(normalize_delays=False, out_type="numpy")
        vertices = np.asarray(paths.vertices)  # (depth, 1, 1, paths, 3)
        interactions = np.asarray(paths.interactions)  # (depth, 1, 1, paths)
    finally:
        for name in temp_files:
            os.unlink(name)

    gains = a[0, 0, 0, 0, :, 0]
    delays = tau.reshape(-1, tau.shape[-1])[0].astype(np.float64)
    alive = np.abs(gains) > 0.0
    tx = np.array([-half, 0.0, HEIGHT_M])
    rx = np.array([half, 0.0, HEIGHT_M])
    polylines = []
    for p in range(len(gains)):
        if not alive[p]:
            continue
        points = [tx]
        for d in range(vertices.shape[0]):
            if interactions[d, 0, 0, p] != 0:
                points.append(vertices[d, 0, 0, p].astype(np.float64))
        points.append(rx)
        polylines.append(np.stack(points))
    return polylines, np.sort(delays[alive])


def verify_against_study(kind: str, delays: np.ndarray) -> None:
    _, study_delays, _ = rt_station_pair_cir(kind)
    study_sorted = np.sort(np.asarray(study_delays, dtype=np.float64))
    if len(study_sorted) != len(delays) or not np.allclose(
        study_sorted, delays, atol=1e-9
    ):
        raise RuntimeError(
            f"scene replica for '{kind}' does not match the study's "
            f"ray trace: {len(delays)} vs {len(study_sorted)} paths"
        )
    print(f"  {kind}: replica matches study ({len(delays)} paths)")


def scene_figure(
    kind: str, filename: str, title: str, view: str = "top"
) -> None:
    polylines, delays = trace_paths(kind)
    verify_against_study(kind, delays)
    rects = scene_footprints(kind)
    half = DISTANCE_M / 2.0

    figure, axis = plt.subplots(figsize=(7.6, 4.6))
    if view == "side":
        # x-z view: shows the ground bounce the top view would hide.
        axis.axhline(0.0, color="0.4", linewidth=1.2, label="ground")
        for index, line in enumerate(polylines):
            axis.plot(
                line[:, 0], line[:, 2], color="C0", linewidth=1.2,
                alpha=0.9,
                label="propagation path" if index == 0 else None,
            )
        axis.scatter(
            [-half, half], [HEIGHT_M, HEIGHT_M], color="C3",
            marker="^", s=90, zorder=3, label="station",
        )
        axis.text(-half, HEIGHT_M + 3.0, "A", ha="center",
                  va="bottom", fontsize=10)
        axis.text(half, HEIGHT_M + 3.0, "B", ha="center",
                  va="bottom", fontsize=10)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("z (m)")
        axis.set_xlim(-330, 330)
        axis.set_ylim(-5, 40)
    else:
        for index, (x, y, width, depth) in enumerate(rects):
            axis.add_patch(
                Rectangle(
                    (x - width / 2.0, y - depth / 2.0), width, depth,
                    facecolor="0.75", edgecolor="0.4",
                    label="building footprint" if index == 0 else None,
                )
            )
        for index, line in enumerate(polylines):
            axis.plot(
                line[:, 0], line[:, 1], color="C0", linewidth=1.0,
                alpha=0.8,
                label="propagation path" if index == 0 else None,
            )
        axis.scatter(
            [-half, half], [0.0, 0.0], color="C3", marker="^", s=90,
            zorder=3, label="station",
        )
        axis.text(-half, -14.0, "A", ha="center", va="top", fontsize=10)
        axis.text(half, -14.0, "B", ha="center", va="top", fontsize=10)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_xlim(-330, 330)
        axis.set_ylim(-200, 200)
        axis.set_aspect("equal")
    axis.set_title(title)
    axis.legend(loc="upper right", fontsize=8)
    save(figure, filename)


def deployment_map() -> None:
    positions = place_stations(6, 500.0, 0)
    figure, axis = plt.subplots(figsize=(5.6, 5.2))
    axis.scatter(
        positions[1:, 0], positions[1:, 1], color="C0", s=70,
        label="station",
    )
    axis.scatter(
        [positions[0, 0]], [positions[0, 1]], color="C3", marker="^",
        s=110, label="reference station",
    )
    for index, (x, y) in enumerate(positions):
        axis.text(x, y - 22.0, str(index), ha="center", va="top",
                  fontsize=9)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_aspect("equal")
    axis.set_title(
        "Station deployment, N=6 star (seed 0, 500 m radius)"
    )
    axis.legend(loc="upper right", fontsize=8)
    save(figure, "clutter_deployment_map")


def main() -> None:
    scene_figure(
        "tworay", "scene_tworay_layout",
        "Ray-traced scene, side view: two-ray ground (stations A-B, "
        "500 m apart, 15 m high)",
        view="side",
    )
    scene_figure(
        "urban-los", "scene_urban_los_layout",
        "Ray-traced scene, top view: urban with line of sight",
    )
    scene_figure(
        "urban-nlos", "scene_urban_nlos_layout",
        "Ray-traced scene, top view: urban with the direct path blocked",
    )
    deployment_map()


if __name__ == "__main__":
    main()
