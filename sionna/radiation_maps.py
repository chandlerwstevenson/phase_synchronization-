"""Radiation graphs for the distributed detection array.

All field values come from the study's own RT leg solver
(detection.rt_echo.rt_steered_legs: ground-bounce two-ray, same
materials/heights), evaluated on grids at drone altitude (60 m).

Three figures:
  1. per-station "radiation map": |leg_k|^2 in dB at the 60 m plane -
     each single isotropic element's effective pattern INCLUDING the
     ground bounce (two-ray lobing rings). The element itself is
     isotropic; this is what it actually illuminates.
  2. the coherent array pattern: all 6 stations conjugate-phased to
     focus at the mid-path drone cell; |sum_k leg_k * e^{j phi_k}|^2
     relative to the value AT the focus (wide map). Pixels near the
     masts exceed 0 dB - spherical spreading beats the focusing gain
     up close; that is physical, not an artifact.
  3. zoom on the focal spot (+-2 m, 2.5 cm pixels - the mainlobe is
     ~25 cm wide) + a 1-D cut along the flight path.

rt_steered_legs returns g * raw_k(x) * exp(+j 2 pi d_k(x)/lambda); the
per-point de-rotation is stripped and replaced by the fixed focus-cell
steering to get the true spatial pattern. Solved legs are cached as
.npz next to the figures, so plot tweaks do not re-trace.
"""

import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/chandlerstevenson/Downloads/Princeton_Research/ota_sync/phase_synchronization-/sionna")

from ota_sync.network import place_stations
from detection.rt_echo import rt_steered_legs

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

N = 6
RADIUS = 500.0
SEED = 0
BS_H = 15.0
DRONE_H = 60.0
PATH_OFFSET = 150.0
FREQ = 915e6
LAMBDA = 299792458.0 / FREQ

positions = place_stations(N, RADIUS, SEED)
centroid = positions.mean(axis=0)
# mid-path waypoint of the study's 12-point path
path_x = np.linspace(-2.4 * RADIUS, 2.4 * RADIUS, 12)
focus = np.array([centroid[0] + path_x[6], centroid[1] + PATH_OFFSET])

stations_3d = np.column_stack((positions, np.full(N, BS_H)))


def distances_3d(points_2d):
    pts = np.column_stack(
        (points_2d, np.full(points_2d.shape[0], DRONE_H))
    )
    return np.linalg.norm(pts[:, None, :] - stations_3d[None, :, :], axis=-1)


def raw_legs(points_2d, cache_name=None, chunk=2500):
    """g * raw_k at each point: solver output with per-point steering
    stripped (leaves antenna gain in, as the study applies it)."""
    if cache_name:
        cache_path = os.path.join(OUT, f"legs_{cache_name}.npz")
        if os.path.exists(cache_path):
            data = np.load(cache_path)
            if (data["points"].shape == points_2d.shape
                    and np.allclose(data["points"], points_2d)):
                print(f"  {cache_name}: loaded from cache")
                return data["legs"]
    out = np.empty((points_2d.shape[0], N), dtype=complex)
    for start in range(0, points_2d.shape[0], chunk):
        block = points_2d[start:start + chunk]
        steered = rt_steered_legs(
            positions, block,
            station_height_m=BS_H, target_height_m=DRONE_H,
            carrier_frequency_hz=FREQ, antenna_gain_dbi=6.0,
            with_ground=True,
        )
        d = distances_3d(block)
        out[start:start + chunk] = steered * np.exp(
            -1j * 2.0 * math.pi * d / LAMBDA
        )
        print(f"  solved {min(start + chunk, points_2d.shape[0])}"
              f"/{points_2d.shape[0]} grid points", flush=True)
    if cache_name:
        np.savez_compressed(cache_path, points=points_2d, legs=out)
    return out


def grid(x0, x1, y0, y1, nx, ny):
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    gx, gy = np.meshgrid(xs, ys)
    return xs, ys, np.column_stack((gx.ravel(), gy.ravel()))


INK = "#333333"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#cccccc", "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "font.size": 9,
})

# reference: the coherent field exactly at the focus
focus_legs = raw_legs(focus[None, :])
d_focus = distances_3d(focus[None, :])[0]                     # (N,)
phases = np.exp(1j * 2.0 * math.pi * d_focus / LAMBDA)
focal_power = np.abs((focus_legs * phases[None, :]).sum()) ** 2
focal_db = 10.0 * np.log10(focal_power)
print(f"focal coherent power: {focal_db:.1f} dB (reference = 0 dB)")

# ---------------- figure 1: per-station maps ----------------
print("figure 1: per-station two-ray illumination maps (60 m plane)...")
nx = ny = 81
xs, ys, pts = grid(centroid[0] - 1200, centroid[0] + 1200,
                   centroid[1] - 1200, centroid[1] + 1200, nx, ny)
legs = raw_legs(pts, cache_name="wide")
power_db = 10.0 * np.log10(np.abs(legs) ** 2 + 1e-30)

fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.2), sharex=True,
                         sharey=True, constrained_layout=True)
vmax = np.percentile(power_db, 99.5)
vmin = vmax - 35.0
for k, ax in enumerate(axes.ravel()):
    img = ax.imshow(
        power_db[:, k].reshape(ny, nx), origin="lower",
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        vmin=vmin, vmax=vmax, cmap="magma", rasterized=True,
    )
    ax.plot(positions[k, 0], positions[k, 1], "^", ms=7, mfc="#7fd4ff",
            mec="black", mew=0.6)
    ax.axhline(centroid[1] + PATH_OFFSET, color="white", lw=0.8,
               ls=(0, (4, 3)), alpha=0.8)
    ax.set_title(f"bs-{k}  ({positions[k, 0]:.0f}, {positions[k, 1]:.0f}) m",
                 fontsize=9)
    ax.set_aspect("equal")
for ax in axes[1]:
    ax.set_xlabel("x (m)")
for ax in axes[:, 0]:
    ax.set_ylabel("y (m)")
cb = fig.colorbar(img, ax=axes, shrink=0.85, pad=0.02)
cb.set_label("one-way gain |leg|$^2$ (dB, incl. 6 dBi antenna)")
fig.suptitle(
    "Per-station illumination at drone altitude (60 m) — single isotropic "
    "element + ground bounce (RT two-ray)\n"
    "dashed line = drone flight path; rings are ground-bounce lobing, "
    "NOT an antenna array pattern",
    fontsize=11,
)
path1 = os.path.join(OUT, "radiation_per_station.png")
fig.savefig(path1, dpi=150)
plt.close(fig)
print("wrote", path1)

# ---------------- figure 2: coherent array pattern, wide ----------------
print("figure 2: coherent focusing pattern (wide)...")
field = (legs * phases[None, :]).sum(axis=1)
pattern_db = 10.0 * np.log10(np.abs(field) ** 2 + 1e-30) - focal_db

fig, ax = plt.subplots(figsize=(9.5, 8.2), constrained_layout=True)
img = ax.imshow(
    pattern_db.reshape(ny, nx), origin="lower",
    extent=[xs[0], xs[-1], ys[0], ys[-1]],
    vmin=-35.0, vmax=float(pattern_db.max()), cmap="magma",
    rasterized=True,
)
ax.plot(positions[:, 0], positions[:, 1], "^", ms=6, mfc="#7fd4ff",
        mec="white", mew=0.6, ls="none", zorder=5)
ax.axhline(centroid[1] + PATH_OFFSET, color="#e08214", lw=1.0,
           ls=(0, (4, 3)), zorder=4)
ax.plot(*focus, "x", ms=8, mew=2, color="#d7301f", zorder=6)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_aspect("equal")
ax.set_title(
    "Coherent array pattern at 60 m altitude — 6 stations conjugate-phased "
    "at the drone cell (red x)\n0 dB = power at the focus; near-mast pixels "
    "exceed it (spreading), the sub-meter focal spot is unresolved at "
    "30 m pixels — see zoom",
    fontsize=10,
)
cb = fig.colorbar(img, ax=ax, shrink=0.9, pad=0.02)
cb.set_label("combined power relative to focus (dB)")
path2 = os.path.join(OUT, "array_pattern_wide.png")
fig.savefig(path2, dpi=150)
plt.close(fig)
print("wrote", path2)

# ---------------- figure 3: focal-spot zoom + path cut ----------------
print("figure 3: focal zoom + cut...")
half = 2.0
nz = 161
zxs, zys, zpts = grid(focus[0] - half, focus[0] + half,
                      focus[1] - half, focus[1] + half, nz, nz)
zlegs = raw_legs(zpts, cache_name="zoom")
zfield = (zlegs * phases[None, :]).sum(axis=1)
zdb = 10.0 * np.log10(np.abs(zfield) ** 2 + 1e-30) - focal_db

cut_half = 10.0
ncut = 2001
cut_pts = np.column_stack((
    np.linspace(focus[0] - cut_half, focus[0] + cut_half, ncut),
    np.full(ncut, centroid[1] + PATH_OFFSET),
))
cut_legs = raw_legs(cut_pts, cache_name="cut")
cut_field = (cut_legs * phases[None, :]).sum(axis=1)
cut_db = 10.0 * np.log10(np.abs(cut_field) ** 2 + 1e-30) - focal_db

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(12.5, 5.4), constrained_layout=True,
    gridspec_kw={"width_ratios": [1.0, 1.35]},
)
img = ax1.imshow(
    zdb.reshape(nz, nz), origin="lower",
    extent=[zxs[0] - focus[0], zxs[-1] - focus[0],
            zys[0] - focus[1], zys[-1] - focus[1]],
    vmin=-30.0, vmax=0.0, cmap="magma", rasterized=True,
)
for spine_val in (-0.35, 0.35):
    ax1.plot([spine_val, spine_val * 2.2], [0, 0], color="#7fd4ff", lw=1.2)
    ax1.plot([0, 0], [spine_val, spine_val * 2.2], color="#7fd4ff", lw=1.2)
ax1.set_xlabel("x offset from focus (m)")
ax1.set_ylabel("y offset from focus (m)")
ax1.set_aspect("equal")
ax1.set_title("Focal spot (4 m window, 2.5 cm pixels)", fontsize=10)
cb = fig.colorbar(img, ax=ax1, shrink=0.9, pad=0.03)
cb.set_label("power relative to focus (dB)")

ax2.plot(cut_pts[:, 0] - focus[0], cut_db, color="#4a5fc1", lw=1.0)
ax2.axhline(10.0 * math.log10(1.0 / N), color="#999999", lw=1.0,
            ls=(0, (4, 3)))
ax2.text(9.6, 10.0 * math.log10(1.0 / N) + 0.6,
         "1/N incoherent speckle level (−7.8 dB)", ha="right",
         fontsize=8, color="#666666")
ax2.set_xlabel("x offset along flight path (m)")
ax2.set_ylabel("power relative to focus (dB)")
ax2.set_ylim(-32, 2)
ax2.grid(color="#e8e8e8", lw=0.6)
ax2.set_title("Cut along the flight path through the focus", fontsize=10)
fig.suptitle(
    "Distributed-array focal region — 6 elements over ~1 km aperture, "
    "915 MHz, RT ground-bounce legs, perfect sync",
    fontsize=11,
)
path3 = os.path.join(OUT, "array_pattern_zoom.png")
fig.savefig(path3, dpi=150)
plt.close(fig)
print("wrote", path3)

xs_cut = cut_pts[:, 0] - focus[0]
main = np.where(cut_db >= -3.0)[0]
if main.size:
    print("approx -3 dB mainlobe width along path: "
          f"{xs_cut[main].max() - xs_cut[main].min():.2f} m")
else:
    print("no cut sample within 3 dB of the focal value - check focusing")
print("median level in zoom window: "
      f"{np.median(zdb):.1f} dB (expect near -7.8 dB = 1/N speckle)")
