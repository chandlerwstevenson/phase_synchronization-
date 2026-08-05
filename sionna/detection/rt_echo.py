"""Ray-traced propagation legs for the waveform detection test.

How Sionna RT is used (and why this way): shooting-ray solvers cannot
reliably find specular/diffuse bounces off a drone-sized (~0.3 m)
object hundreds of meters away — the hit probability per ray is
~1e-8, so paths are simply never sampled (verified: 0 paths found).
The standard ISAC coupling is used instead:

  1. RT computes the station -> target-position PROPAGATION legs
     exactly, by placing a probe receiver at the drone position (point
     receivers are traced analytically, no hit-probability problem).
     This captures what RT is genuinely good at: line-of-sight
     geometry in 3D (station masts, drone altitude), the ground-bounce
     multipath that dominates low-altitude UHF links (two-ray lobing),
     and obstruction if scene geometry is added.
  2. The target's radar cross-section is applied analytically at the
     probe point (Swerling-1 draw, as before): the echo for the pair
     (transmit k, receive j) is

         e_kj = h_k * h_j * sqrt(4*pi*sigma) / lambda

     which reproduces the bistatic radar equation exactly when h are
     free-space legs (validated against FSPL amplitude and phase to
     float32 precision).
  3. Reciprocity supplies the return leg (same h).

The returned legs are pre-multiplied by the antenna amplitude gain and
de-rotated by the LoS steering hypothesis (the array steers with KNOWN
geometry), so free space yields real positive legs and the ground
bounce shows up as the complex ripple the steering cannot remove -
exactly the physical effect ray tracing adds to the study.
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np

from .viability import SPEED_OF_LIGHT


def _ground_ply(half_size_m: float) -> str:
    return (
        "ply\nformat ascii 1.0\nelement vertex 4\n"
        "property float x\nproperty float y\nproperty float z\n"
        "element face 2\nproperty list uchar int vertex_indices\n"
        "end_header\n"
        f"-{half_size_m} -{half_size_m} 0\n{half_size_m} -{half_size_m} 0\n"
        f"{half_size_m} {half_size_m} 0\n-{half_size_m} {half_size_m} 0\n"
        "3 0 1 2\n3 0 2 3\n"
    )


def _building_ply(x: float, y: float, width: float, depth: float,
                  height: float) -> str:
    """Axis-aligned box (walls + roof) as an ASCII PLY string."""

    x0, x1 = x - width / 2.0, x + width / 2.0
    y0, y1 = y - depth / 2.0, y + depth / 2.0
    vertices = [
        (x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0),
        (x0, y0, height), (x1, y0, height), (x1, y1, height),
        (x0, y1, height),
    ]
    faces = [
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
        (4, 5, 6), (4, 6, 7),
    ]
    lines = [
        "ply", "format ascii 1.0", f"element vertex {len(vertices)}",
        "property float x", "property float y", "property float z",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices", "end_header",
    ]
    lines += [f"{vx} {vy} {vz}" for vx, vy, vz in vertices]
    lines += [f"3 {a} {b} {c}" for a, b, c in faces]
    return "\n".join(lines) + "\n"


def rt_steered_legs(
    positions: np.ndarray,
    waypoints: np.ndarray,
    station_height_m: float = 15.0,
    target_height_m: float = 60.0,
    carrier_frequency_hz: float = 915e6,
    antenna_gain_dbi: float = 6.0,
    with_ground: bool = True,
    num_buildings: int = 0,
    building_seed: int = 0,
) -> np.ndarray:
    """(num_waypoints, num_stations) steered complex leg gains via RT.

    ``positions`` and ``waypoints`` are 2-D ground coordinates; heights
    lift them into the 3-D scene. One PathSolver run computes all
    station -> waypoint legs at once.
    """

    import sionna.rt as rt

    scene = rt.load_scene()
    scene.frequency = carrier_frequency_hz
    scene.tx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, pattern="iso", polarization="V"
    )
    scene.rx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, pattern="iso", polarization="V"
    )
    if with_ground:
        # Typical medium-dry ground around 1 GHz (ITU-R P.527 class
        # values); the built-in ITU table does not cover 915 MHz.
        ground_material = rt.RadioMaterial(
            "ground-material",
            thickness=10.0,
            relative_permittivity=15.0,
            conductivity=0.035,
        )
        extent = 1.5 * max(
            np.abs(positions).max(), np.abs(waypoints).max(), 1000.0
        )
        handle = tempfile.NamedTemporaryFile("w", suffix=".ply", delete=False)
        handle.write(_ground_ply(extent))
        handle.close()
        try:
            ground = rt.SceneObject(
                fname=handle.name,
                name="ground-plane",
                radio_material=ground_material,
            )
            scene.edit(add=ground)
        finally:
            os.unlink(handle.name)

    if num_buildings > 0:
        # Concrete-class walls (ITU-R P.2040 values near 1 GHz).
        rng = np.random.default_rng(building_seed)
        extent = 0.8 * max(np.abs(positions).max(), 500.0)
        for index in range(num_buildings):
            bx, by = rng.uniform(-extent, extent, size=2)
            width, depth = rng.uniform(15.0, 40.0, size=2)
            height = rng.uniform(10.0, 35.0)
            concrete = rt.RadioMaterial(
                f"building-material-{index}",
                thickness=0.3,
                relative_permittivity=5.24,
                conductivity=0.123,
            )
            handle = tempfile.NamedTemporaryFile(
                "w", suffix=".ply", delete=False
            )
            handle.write(_building_ply(bx, by, width, depth, height))
            handle.close()
            try:
                building = rt.SceneObject(
                    fname=handle.name,
                    name=f"building-{index}",
                    radio_material=concrete,
                )
                scene.edit(add=building)
            finally:
                os.unlink(handle.name)

    stations_3d = np.column_stack(
        (positions, np.full(positions.shape[0], station_height_m))
    )
    targets_3d = np.column_stack(
        (waypoints, np.full(waypoints.shape[0], target_height_m))
    )
    for index, station in enumerate(stations_3d):
        scene.add(rt.Transmitter(f"bs-{index}", position=station.tolist()))
    for index, target in enumerate(targets_3d):
        scene.add(rt.Receiver(f"probe-{index}", position=target.tolist()))

    solver = rt.PathSolver()
    paths = solver(
        scene,
        max_depth=3 if num_buildings > 0 else (2 if with_ground else 1),
        los=True,
        specular_reflection=with_ground,
        diffuse_reflection=False,
        refraction=False,
    )
    a, _ = paths.cir(normalize_delays=False, out_type="numpy")
    # a: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_steps]
    legs = a[:, 0, :, 0, :, 0].sum(axis=-1)  # (num_waypoints, num_stations)

    # Steering hypothesis: de-rotate by the known 3-D LoS geometry and
    # apply the antenna amplitude gain on this leg.
    wavelength = SPEED_OF_LIGHT / carrier_frequency_hz
    distances = np.linalg.norm(
        targets_3d[:, None, :] - stations_3d[None, :, :], axis=-1
    )
    steering = np.exp(1j * 2.0 * math.pi * distances / wavelength)
    amplitude_gain = math.sqrt(10.0 ** (antenna_gain_dbi / 10.0))
    return amplitude_gain * legs * steering
