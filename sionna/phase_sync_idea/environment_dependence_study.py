"""Is clutter-referenced sync environment-dependent?

clutter_sync_study.py's headline (42 mrad at 0.48% sync airtime vs 87
mrad at 19.1% two-way) was measured on ONE environment: frozen TDL-D
(LOS Rician, 100 ns delay spread). This study answers whether the
result generalizes, two ways:

PART 1 - statistical families: the same N=2 comparison re-run across
Sionna TDL letters (D, E: LOS Rician; A, B, C: NLOS Rayleigh) and
across delay spreads at TDL-D (30/100/300/1000 ns), with the measured
per-observation one-way noise alongside, so the boundary can be stated
as a channel parameter rather than a vibe.

PART 2 - ray-traced geometry: the TDL draw replaced by a station-to-
station channel impulse response ray-traced in three deterministic
scenes (built from PLY geometry exactly as detection/rt_echo.py does):

  tworay      ground plane only - LOS + ground bounce, the canonical
              open-field deployment
  urban-los   ground + 12 concrete buildings AROUND the link (LOS
              corridor kept clear) - LOS plus rich static clutter
  urban-nlos  same city but a 30 m building centered ON the link -
              no direct path; sync must ride reflections alone

The RT taps are injected from the outside: an SDRRadioLink subclass
overwrites the frozen channel taps in place after construction, swapped
into hybrid_calibration.hybrid's and ota_sync.coherent's namespaces for
the duration of a run (same pattern as gating_study's RecordingEKF; no
existing file is modified). Delays are gated to the first arrival (a
real receiver time-gates); taps are energy-normalized per the repo's
normalize=True convention, so environments differ by STRUCTURE while
snr_db stays the budget knob - the measured excess path loss of each
scene is reported separately, and the NLOS scene is additionally run
with snr_db knocked down by that excess.

Usage:
    .venv/bin/python environment_dependence_study.py           # full
    .venv/bin/python environment_dependence_study.py --quick   # 1 seed
    .venv/bin/python environment_dependence_study.py --skip-rt
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import tempfile
from dataclasses import replace

import numpy as np
import torch

import clutter_sync_ofdm
import hybrid_calibration.hybrid as hybrid_module
import ota_sync.coherent as coherent_module
from clutter_sync_ofdm import calibrate_oneway_noise
from clutter_sync_study import piggyback_airtime_fraction, run_clutter_referenced
from ota_sync import SDRSimulationConfig, run_two_way_simulation
from ota_sync.sdr import SDRRadioLink
from sionna.phy.channel import cir_to_time_channel, time_lag_discrete_time_channel

SPEED_OF_LIGHT = 299792458.0


# ---------------------------------------------------------------------
# RT channel -> frozen time taps, and outside-in injection
# ---------------------------------------------------------------------

def cir_to_frozen_taps(
    gains: np.ndarray,
    delays_s: np.ndarray,
    settings: SDRSimulationConfig,
) -> tuple[torch.Tensor, int]:
    """Bandlimited time taps for one frozen station-to-station CIR,
    matching SDRRadioLink's own conversion (same helper, same l_min/
    l_max window, energy-normalized). Delays are re-referenced to the
    first arrival - the receiver time-gates to it. Returns (taps of
    shape [l_tot], number of paths dropped for exceeding the window).
    """

    order = np.argsort(delays_s)
    gains = np.asarray(gains, dtype=np.complex128)[order]
    delays = np.asarray(delays_s, dtype=np.float64)[order]
    excess = delays - delays[0]
    keep = excess <= settings.maximum_channel_delay_s
    dropped = int((~keep).sum())
    gains, excess = gains[keep], excess[keep]

    l_min, l_max = time_lag_discrete_time_channel(
        settings.sample_rate, settings.maximum_channel_delay_s
    )
    # Same tensor ranks the TDL produces: a [batch, rx, rx_ant, tx,
    # tx_ant, paths, steps], tau [batch, rx, tx, paths]; one time step
    # suffices - the environment is frozen.
    a = torch.tensor(gains, dtype=torch.complex128).reshape(
        1, 1, 1, 1, 1, -1, 1
    )
    tau = torch.tensor(excess, dtype=torch.float64).reshape(1, 1, 1, -1)
    taps = cir_to_time_channel(
        settings.sample_rate, a, tau, l_min, l_max, normalize=True
    )
    flat = taps.reshape(-1).to(torch.complex128)
    flat = flat / torch.sqrt(torch.sum(torch.abs(flat) ** 2))
    return flat, dropped


class _InjectedRadioLink(SDRRadioLink):
    """SDRRadioLink whose frozen taps are replaced, in place, by an
    externally supplied CIR. Mirrors share the forward link's tensor,
    so overwriting in place propagates to every direction and to the
    micro-pilot links, exactly like a real shared physical channel."""

    injected_taps: torch.Tensor | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if kwargs.get("mirror_of") is None and self.injected_taps is not None:
            flat = self.injected_taps.to(self.channel_taps.dtype)
            if flat.shape[-1] != self.channel_taps.shape[-1]:
                raise ValueError(
                    "injected taps do not match the link's l_tot window"
                )
            self.channel_taps.copy_(flat.expand_as(self.channel_taps))


@contextlib.contextmanager
def injected_channel(taps: torch.Tensor | None):
    """Swap the injection subclass into every module that constructs
    links (hybrid one-way/anchor machinery, two-way baseline, the
    one-way noise probe). No file is modified; classes are restored."""

    if taps is None:
        yield
        return
    _InjectedRadioLink.injected_taps = taps
    originals = (
        hybrid_module.SDRRadioLink,
        coherent_module.SDRRadioLink,
        clutter_sync_ofdm.SDRRadioLink,
    )
    hybrid_module.SDRRadioLink = _InjectedRadioLink
    coherent_module.SDRRadioLink = _InjectedRadioLink
    clutter_sync_ofdm.SDRRadioLink = _InjectedRadioLink
    try:
        yield
    finally:
        (
            hybrid_module.SDRRadioLink,
            coherent_module.SDRRadioLink,
            clutter_sync_ofdm.SDRRadioLink,
        ) = originals
        _InjectedRadioLink.injected_taps = None


# ---------------------------------------------------------------------
# Ray-traced station-to-station scenes
# ---------------------------------------------------------------------

def _write_ply(content: str) -> str:
    handle = tempfile.NamedTemporaryFile("w", suffix=".ply", delete=False)
    handle.write(content)
    handle.close()
    return handle.name


def rt_station_pair_cir(
    scene_kind: str,
    distance_m: float = 500.0,
    station_height_m: float = 15.0,
    carrier_frequency_hz: float = 915e6,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Ray-trace the station->station CIR in one deterministic scene.

    Returns (complex path gains, absolute delays s, diagnostics). The
    scene geometry reuses detection/rt_echo.py's PLY builders and
    material classes so the two RT users of this repo stay consistent.
    """

    import sionna.rt as rt

    from detection.rt_echo import _building_ply, _ground_ply

    scene = rt.load_scene()
    scene.frequency = carrier_frequency_hz
    scene.tx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, pattern="iso", polarization="V"
    )
    scene.rx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, pattern="iso", polarization="V"
    )

    temp_files: list[str] = []
    try:
        if scene_kind != "freespace":
            ground_material = rt.RadioMaterial(
                "ground-material",
                thickness=10.0,
                relative_permittivity=15.0,
                conductivity=0.035,
            )
            path = _write_ply(_ground_ply(1.5 * max(distance_m, 1000.0)))
            temp_files.append(path)
            scene.edit(
                add=rt.SceneObject(
                    fname=path,
                    name="ground-plane",
                    radio_material=ground_material,
                )
            )

        def add_building(index, x, y, width, depth, height):
            concrete = rt.RadioMaterial(
                f"building-material-{index}",
                thickness=0.3,
                relative_permittivity=5.24,
                conductivity=0.123,
            )
            path = _write_ply(_building_ply(x, y, width, depth, height))
            temp_files.append(path)
            scene.edit(
                add=rt.SceneObject(
                    fname=path,
                    name=f"building-{index}",
                    radio_material=concrete,
                )
            )

        if scene_kind in {"urban-los", "urban-nlos"}:
            # Street canyon: two long concrete walls flanking the link
            # guarantee specular station->wall->station paths (random
            # building fields turned out to produce none - reflection
            # geometry has to be designed, not sampled). A few outer
            # blocks add texture at reflection depth 2-3.
            add_building(0, 0.0, 60.0, 1.2 * distance_m, 2.0, 25.0)
            add_building(1, 0.0, -60.0, 1.2 * distance_m, 2.0, 25.0)
            rng = np.random.default_rng(7)
            for index in range(2, 8):
                add_building(
                    index,
                    rng.uniform(-0.5 * distance_m, 0.5 * distance_m),
                    float(rng.choice([-1.0, 1.0])) * rng.uniform(90.0, 160.0),
                    rng.uniform(20.0, 45.0),
                    rng.uniform(20.0, 45.0),
                    rng.uniform(12.0, 35.0),
                )
        if scene_kind == "urban-nlos":
            # An 80 m-deep slab centered on the link blocks the direct
            # path AND the ground bounce (both cross x=0 inside |y|<40),
            # while the canyon-wall reflections at y = +-60 clear it:
            # the stations can only hear each other via the walls.
            add_building(99, 0.0, 0.0, 40.0, 80.0, 30.0)

        half = distance_m / 2.0
        scene.add(
            rt.Transmitter("station-a", position=[-half, 0.0, station_height_m])
        )
        scene.add(
            rt.Receiver("station-b", position=[half, 0.0, station_height_m])
        )
        solver = rt.PathSolver()
        paths = solver(
            scene,
            max_depth=1 if scene_kind == "freespace" else 3,
            los=True,
            specular_reflection=scene_kind != "freespace",
            diffuse_reflection=False,
            refraction=False,
        )
        a, tau = paths.cir(normalize_delays=False, out_type="numpy")
    finally:
        for path in temp_files:
            os.unlink(path)

    if a.shape[-2] == 0:
        raise RuntimeError(f"scene '{scene_kind}': no propagation paths")
    gains = a[0, 0, 0, 0, :, 0]
    # tau is [num_rx, num_tx, num_paths] for synthetic arrays in this
    # sionna-rt version, [num_rx, rx_ant, num_tx, tx_ant, num_paths]
    # otherwise - flatten the leading singleton axes either way.
    delays = tau.reshape(-1, tau.shape[-1])[0].astype(np.float64)
    alive = np.abs(gains) > 0.0
    gains, delays = gains[alive], delays[alive]
    if gains.size == 0:
        raise RuntimeError(f"scene '{scene_kind}': no propagation paths")

    power = np.abs(gains) ** 2
    total = power.sum()
    los_delay = distance_m / SPEED_OF_LIGHT
    # 10 ns window = 3 m path tolerance: tight enough that the canyon
    # wall bounce (+46 ns) is not mistaken for a direct path.
    has_los = bool(
        np.any(
            (np.abs(delays - los_delay) < 10e-9)
            & (power > 0.01 * power.max())
        )
    )
    strongest = power.max()
    k_factor_db = 10.0 * math.log10(
        strongest / max(total - strongest, 1e-30)
    )
    mean_delay = float((power * delays).sum() / total)
    rms_spread_ns = 1e9 * math.sqrt(
        float((power * (delays - mean_delay) ** 2).sum() / total)
    )
    free_space_power = (
        SPEED_OF_LIGHT / carrier_frequency_hz / (4.0 * math.pi * distance_m)
    ) ** 2
    excess_loss_db = 10.0 * math.log10(free_space_power / total)
    diagnostics = {
        "num_paths": int(gains.size),
        "has_los": has_los,
        "k_factor_db": k_factor_db,
        "rms_spread_ns": rms_spread_ns,
        "excess_loss_db": excess_loss_db,
    }
    return gains, delays, diagnostics


# ---------------------------------------------------------------------
# One comparison cell
# ---------------------------------------------------------------------

def run_cell(
    settings: SDRSimulationConfig,
    cadences: tuple[int, ...],
    taps: torch.Tensor | None = None,
) -> dict:
    """Two-way baseline plus clutter-referenced at each anchor cadence,
    on one environment (TDL settings, or injected RT taps)."""

    out: dict = {}
    with injected_channel(taps):
        twoway = run_two_way_simulation(settings)
        out["twoway"] = (
            twoway.steady_state_phase_rms,
            twoway.mean_coherent_gain,
            twoway.airtime_fraction,
        )
        for cadence in cadences:
            result, _, piggyback = run_clutter_referenced(settings, cadence)
            out[f"K{cadence}"] = (
                result.steady_state_phase_rms,
                result.mean_coherent_gain,
                piggyback,
                result.detection_rate,
            )
    return out


def _agg(cells: list[tuple], index: int, scale: float = 1.0):
    values = torch.tensor(
        [scale * c[index] for c in cells], dtype=torch.float64
    )
    return values.mean().item(), (
        values.std().item() if len(cells) > 1 else 0.0
    )


def oneway_noise_mrad(settings: SDRSimulationConfig) -> tuple[float, float]:
    """Measured per-observation one-way (phase std mrad, detect rate)
    for this environment's ZC pilot; the module cache is cleared first
    because its key does not include the channel parameters."""

    clutter_sync_ofdm._CALIBRATION_CACHE.clear()
    phase_var, _, detect = calibrate_oneway_noise(
        settings, "zc", torch.device("cpu")
    )
    clutter_sync_ofdm._CALIBRATION_CACHE.clear()
    return 1e3 * math.sqrt(max(phase_var, 0.0)), detect


# ---------------------------------------------------------------------
# The study
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="environment dependence of clutter-referenced sync"
    )
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--cadences", type=str, default="5,40")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-rt", action="store_true")
    args = parser.parse_args()

    seeds = [0] if args.quick else [int(s) for s in args.seeds.split(",")]
    cadences = tuple(int(k) for k in args.cadences.split(","))
    verdict_rows: list[tuple[str, str, str, str]] = []

    def report(label: str, cells: list[dict], noise: tuple | None = None):
        two_rms, two_std = _agg([c["twoway"] for c in cells], 0, 1e3)
        two_air, _ = _agg([c["twoway"] for c in cells], 2, 100.0)
        line = (
            f"  {label:<28} twoway {two_rms:6.1f}±{two_std:4.1f} mrad "
            f"@{two_air:5.1f}%"
        )
        best = None
        for cadence in cadences:
            rows = [c[f"K{cadence}"] for c in cells]
            rms, rms_std = _agg(rows, 0, 1e3)
            air, _ = _agg(rows, 2, 100.0)
            detect, _ = _agg(rows, 3, 100.0)
            line += (
                f" | K{cadence} {rms:6.1f}±{rms_std:4.1f} @{air:5.2f}% "
                f"det {detect:5.1f}%"
            )
            if cadence == max(cadences):
                best = (rms, air, detect)
        if noise is not None:
            line += f" | obs {noise[0]:6.1f} mrad det {100 * noise[1]:5.1f}%"
        print(line)
        works = (
            best is not None
            and best[0] <= two_rms + two_std + 10.0
            and best[2] > 90.0
        )
        verdict_rows.append(
            (
                label,
                f"{best[0]:.0f} mrad @ {best[1]:.2f}%",
                f"{two_rms:.0f} mrad @ {two_air:.1f}%",
                "y" if works else "n",
            )
        )

    # ---- PART 1a: TDL statistical families -------------------------
    print(
        f"PART 1a - TDL letters (N=2, {args.iterations} intervals, "
        f"seeds {seeds}, static, K in {list(cadences)})"
    )
    for letter, family in (
        ("D", "LOS Rician (headline)"),
        ("E", "LOS Rician, stronger"),
        ("A", "NLOS Rayleigh"),
        ("B", "NLOS Rayleigh"),
        ("C", "NLOS Rayleigh"),
    ):
        cells = []
        for seed in seeds:
            settings = SDRSimulationConfig(
                num_iterations=args.iterations, seed=seed, device="cpu",
                tdl_model=letter,
            )
            cells.append(run_cell(settings, cadences))
        noise = oneway_noise_mrad(
            SDRSimulationConfig(
                num_iterations=args.iterations, seed=0, device="cpu",
                tdl_model=letter,
            )
        )
        report(f"TDL-{letter} ({family})", cells, noise)

    # ---- PART 1b: delay-spread sweep at TDL-D ----------------------
    print(
        "\nPART 1b - delay spread at TDL-D (resampling-noise scaling; "
        "100 ns is the headline)"
    )
    for spread_ns in (30, 100, 300, 1000):
        cells = []
        for seed in seeds:
            settings = SDRSimulationConfig(
                num_iterations=args.iterations, seed=seed, device="cpu",
                delay_spread_s=spread_ns * 1e-9,
            )
            cells.append(run_cell(settings, cadences))
        noise = oneway_noise_mrad(
            SDRSimulationConfig(
                num_iterations=args.iterations, seed=0, device="cpu",
                delay_spread_s=spread_ns * 1e-9,
            )
        )
        report(f"TDL-D {spread_ns:>4} ns", cells, noise)

    # ---- PART 2: ray-traced scenes ---------------------------------
    if not args.skip_rt:
        print(
            "\nPART 2 - ray-traced station-to-station scenes (frozen "
            "geometry, taps energy-normalized; excess loss reported and "
            "the NLOS scene re-run with it charged to SNR)"
        )
        rt_seeds = seeds[: max(1, min(2, len(seeds)))]
        for kind in ("tworay", "urban-los", "urban-nlos"):
            gains, delays, diag = rt_station_pair_cir(kind)
            base = SDRSimulationConfig(
                num_iterations=args.iterations, seed=0, device="cpu"
            )
            taps, dropped = cir_to_frozen_taps(gains, delays, base)
            print(
                f"  [{kind}] paths {diag['num_paths']} "
                f"LOS {'yes' if diag['has_los'] else 'NO'} "
                f"K-factor {diag['k_factor_db']:6.1f} dB "
                f"spread {diag['rms_spread_ns']:7.1f} ns "
                f"excess loss {diag['excess_loss_db']:5.1f} dB"
                + (f" ({dropped} paths outside window)" if dropped else "")
            )
            cells = []
            for seed in rt_seeds:
                settings = SDRSimulationConfig(
                    num_iterations=args.iterations, seed=seed, device="cpu"
                )
                cells.append(run_cell(settings, cadences, taps))
            with injected_channel(taps):
                noise = oneway_noise_mrad(base)
            report(f"RT {kind}", cells, noise)

            if kind == "urban-nlos":
                # Charge the measured excess loss (beyond the two-ray
                # scene's) to the link budget.
                tworay_excess = rt_station_pair_cir("tworay")[2][
                    "excess_loss_db"
                ]
                penalty = max(0.0, diag["excess_loss_db"] - tworay_excess)
                snr_db = SDRSimulationConfig().snr_db - penalty
                settings = SDRSimulationConfig(
                    num_iterations=args.iterations, seed=0, device="cpu",
                    snr_db=snr_db,
                )
                cells = [run_cell(settings, cadences, taps)]
                with injected_channel(taps):
                    noise = oneway_noise_mrad(settings)
                report(
                    f"RT urban-nlos @snr {snr_db:.0f} dB", cells, noise
                )

    # ---- PART 3: verdict table -------------------------------------
    print("\nPART 3 - verdict (works = piggyback at largest K matches or")
    print("beats the two-way residual within noise, detection > 90%)")
    print(f"  {'environment':<28} {'piggyback':<22} {'two-way':<20} works")
    for row in verdict_rows:
        print(f"  {row[0]:<28} {row[1]:<22} {row[2]:<20} {row[3]}")


if __name__ == "__main__":
    main()
