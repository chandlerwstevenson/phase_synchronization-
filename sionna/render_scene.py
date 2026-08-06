"""Render the EXACT 3-D scene the waveform detection study ray-traces.

Same deployment (place_stations, seed 0, 500 m disc), same ground
material/solver settings, same heights (BS masts 15 m, drone 60 m),
same drone path (2.4x radius crossing, 150 m lateral offset).

Visualization-only additions (labeled, not part of the physics):
  - mast poles under each station antenna (the sim uses point antennas)
  - a drone body + orange waypoint cubes along the flight path, ENLARGED
    to be visible at km scale (the real target is ~0.3 m and enters the
    simulation as a point probe + analytic RCS - RT cannot sample
    bounces off an object that small)
One probe receiver is placed at the mid-path drone position and the
solver is run with the study's exact settings so the traced rays
(line-of-sight + ground bounce) can be drawn.
"""

import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, "/Users/chandlerstevenson/Downloads/Princeton_Research/ota_sync/phase_synchronization-/sionna")

from ota_sync.network import place_stations
import sionna.rt as rt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

N = 6
RADIUS = 500.0
SEED = 0
BS_H = 15.0
DRONE_H = 60.0
PATH_OFFSET = 150.0
WAYPOINTS = 12

positions = place_stations(N, RADIUS, SEED)
centroid = positions.mean(axis=0)
span = 2.4 * RADIUS
path_x = np.linspace(-span, span, WAYPOINTS)
waypoints = np.stack(
    (centroid[0] + path_x, np.full(WAYPOINTS, centroid[1] + PATH_OFFSET)),
    axis=1,
)


def ground_ply(half):
    return (
        "ply\nformat ascii 1.0\nelement vertex 4\n"
        "property float x\nproperty float y\nproperty float z\n"
        "element face 2\nproperty list uchar int vertex_indices\n"
        "end_header\n"
        f"-{half} -{half} 0\n{half} -{half} 0\n"
        f"{half} {half} 0\n-{half} {half} 0\n"
        "3 0 1 2\n3 0 2 3\n"
    )


def box_ply(cx, cy, z0, w, d, h):
    x0, x1 = cx - w / 2, cx + w / 2
    y0, y1 = cy - d / 2, cy + d / 2
    v = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z0 + h), (x1, y0, z0 + h), (x1, y1, z0 + h),
        (x0, y1, z0 + h),
    ]
    f = [
        (0, 2, 1), (0, 3, 2),
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
        (4, 5, 6), (4, 6, 7),
    ]
    lines = [
        "ply", "format ascii 1.0", f"element vertex {len(v)}",
        "property float x", "property float y", "property float z",
        f"element face {len(f)}",
        "property list uchar int vertex_indices", "end_header",
    ]
    lines += [f"{a} {b} {c}" for a, b, c in v]
    lines += [f"3 {a} {b} {c}" for a, b, c in f]
    return "\n".join(lines) + "\n"


def add_mesh(scene, ply_text, name, material):
    handle = tempfile.NamedTemporaryFile("w", suffix=".ply", delete=False)
    handle.write(ply_text)
    handle.close()
    try:
        obj = rt.SceneObject(fname=handle.name, name=name,
                             radio_material=material)
        scene.edit(add=obj)
    finally:
        os.unlink(handle.name)


scene = rt.load_scene()
scene.frequency = 915e6
scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso",
                                polarization="V")
scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso",
                                polarization="V")

# ground: the study's exact radio material, colored like dry grass
ground_material = rt.RadioMaterial(
    "ground-material", thickness=10.0,
    relative_permittivity=15.0, conductivity=0.035,
    color=(0.55, 0.65, 0.45),
)
extent = 1.5 * max(np.abs(positions).max(), np.abs(waypoints).max(), 1000.0)
add_mesh(scene, ground_ply(extent), "ground-plane", ground_material)

# radio endpoints exactly as the study places them; one probe receiver
# at the mid-path drone position. Paths are solved BEFORE the
# visualization meshes are added, so the traced rays are the study's
# true physics (a drone-sized box around the probe would block them).
di = WAYPOINTS // 2
dx, dy = waypoints[di]
for i, (x, y) in enumerate(positions):
    scene.add(rt.Transmitter(f"bs-{i}", position=[float(x), float(y), BS_H]))
scene.add(rt.Receiver("drone-probe", position=[float(dx), float(dy),
                                               float(DRONE_H)]))

solver = rt.PathSolver()
paths = solver(
    scene,
    max_depth=2,
    los=True,
    specular_reflection=True,
    diffuse_reflection=False,
    refraction=False,
)

# visualization-only meshes
steel = rt.RadioMaterial("mast-material", thickness=0.05,
                         relative_permittivity=1.0, conductivity=1e7,
                         color=(0.20, 0.25, 0.55))
for i, (x, y) in enumerate(positions):
    add_mesh(scene, box_ply(x, y, 0.0, 4.0, 4.0, BS_H), f"mast-{i}", steel)
    add_mesh(scene, box_ply(x, y, BS_H, 10.0, 10.0, 4.0),
             f"head-{i}", steel)

orange = rt.RadioMaterial("wp-material", thickness=0.003,
                          relative_permittivity=3.0, conductivity=1e-4,
                          color=(0.95, 0.55, 0.10))
for i, (x, y) in enumerate(waypoints):
    if i == di:
        continue
    add_mesh(scene, box_ply(x, y, DRONE_H - 5.0, 10.0, 10.0, 10.0),
             f"wp-{i}", orange)

red = rt.RadioMaterial("drone-material", thickness=0.003,
                       relative_permittivity=3.0, conductivity=1e-4,
                       color=(0.85, 0.10, 0.10))
add_mesh(scene, box_ply(dx, dy, DRONE_H - 2.0, 24.0, 6.0, 4.0),
         "drone-body", red)
add_mesh(scene, box_ply(dx, dy, DRONE_H - 2.0, 6.0, 24.0, 4.0),
         "drone-cross", red)

views = {
    "scene_overview": dict(
        position=[centroid[0] - 200.0, centroid[1] - 1800.0, 1000.0],
        look_at=[centroid[0], centroid[1], 30.0],
    ),
    "scene_rays": dict(
        position=[centroid[0] - 700.0, centroid[1] - 1300.0, 500.0],
        look_at=[dx, dy, DRONE_H / 2.0],
        paths=paths,
    ),
    "scene_side_profile": dict(
        position=[centroid[0] - 1500.0, centroid[1] + 150.0, 80.0],
        look_at=[centroid[0], centroid[1] + 150.0, 50.0],
    ),
    "scene_drone_closeup": dict(
        position=[dx - 150.0, dy - 150.0, 110.0],
        look_at=[dx, dy, DRONE_H],
    ),
    "scene_station_closeup": dict(
        position=[positions[0][0] - 80.0, positions[0][1] - 80.0, 35.0],
        look_at=[positions[0][0], positions[0][1], 12.0],
    ),
}

for name, kw in views.items():
    cam = rt.Camera(
        position=[float(v) for v in kw["position"]],
        look_at=[float(v) for v in kw["look_at"]],
    )
    path = os.path.join(OUT, f"{name}.png")
    scene.render_to_file(
        camera=cam, filename=path, resolution=(1400, 900),
        num_samples=128, show_devices=False,
        paths=kw.get("paths"),
    )
    print("wrote", path)

n_paths = paths.a[0].shape[-2] if hasattr(paths, "a") else "?"
print("traced paths to mid-path drone probe:", n_paths,
      "(expect 12: 6 line-of-sight + 6 ground bounce)")
print("station positions (m):")
for i, (x, y) in enumerate(positions):
    print(f"  bs-{i}: ({x:8.1f}, {y:8.1f}, {BS_H})")
print(f"drone path: y = {centroid[1] + PATH_OFFSET:.1f} m, "
      f"x from {waypoints[0][0]:.0f} to {waypoints[-1][0]:.0f} m, "
      f"alt {DRONE_H} m")
