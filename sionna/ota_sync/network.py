"""N-station synchronization over a randomly deployed area.

Topology: a star. Station 0 is the reference; every other station runs
the chosen pairwise synchronization scheme against it. The reference's
radio is shared, so the pairwise exchanges are time-multiplexed (TDMA)
on one channel: total pilot airtime grows linearly with the number of
stations, which is the fundamental scaling cost this module exposes.

Geometry and link budget: stations are placed uniformly at random in a
disc (rejection-sampled to a minimum separation), and each link's SNR
follows the log-distance path-loss model

    SNR_k = SNR_ref - 10 * n * log10(d_k / d_ref)

where ``SNR_ref`` is the configured nominal SNR at the reference
distance ``d_ref`` and ``n`` is the path-loss exponent (~2 free space,
2.7-3.9 urban macro per 3GPP TR 38.901). Distant stations therefore see
noisier pilots and, past some radius, start missing detections - the
second scaling limit.

Coherence metric: with the reference's phase as the datum (theta_0 = 0)
and theta_k the true crystal residual of station k, the N-station array
coherent gain is

    G(t) = | 1 + sum_k exp(j * theta_k(t)) |^2 / N^2

i.e. the fraction of the ideal N^2 combined power the array would
deliver if all stations transmitted together.

Honest simplifications:
  - The pairwise loops are simulated independently, so the reference
    oscillator's noise is drawn independently per link. Physically that
    noise is common-mode across stations and partially cancels in the
    station-to-station spread, so the array gain reported here is
    slightly conservative.
  - TDMA separation is assumed perfect (no cross-link interference),
    and every link reuses the same channel profile with an independent
    realization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import torch

from .sdr import SDRSimulationConfig

#: Per-link SNR ceiling; stations that land very close to the reference
#: would otherwise get physically silly link budgets.
MAX_LINK_SNR_DB = 50.0


def place_stations(
    num_stations: int,
    radius_m: float,
    seed: int,
    min_separation_m: float = 10.0,
) -> np.ndarray:
    """Uniform random positions in a disc with a minimum separation."""

    if num_stations < 2:
        raise ValueError("a network needs at least 2 stations")
    generator = np.random.default_rng(seed)
    positions: list[np.ndarray] = []
    attempts = 0
    while len(positions) < num_stations:
        attempts += 1
        if attempts > 10000 * num_stations:
            raise ValueError(
                "cannot place stations with the requested minimum separation"
            )
        radius = radius_m * math.sqrt(generator.uniform())
        angle = generator.uniform(0.0, 2.0 * math.pi)
        candidate = np.array(
            [radius * math.cos(angle), radius * math.sin(angle)]
        )
        if all(
            np.linalg.norm(candidate - existing) >= min_separation_m
            for existing in positions
        ):
            positions.append(candidate)
    return np.stack(positions)


@dataclass(frozen=True)
class NetworkLink:
    """One reference-to-station synchronization link."""

    station: int
    distance_m: float
    snr_db: float
    residual: torch.Tensor
    steady_mask: torch.Tensor
    detection_rate: float
    airtime_fraction: float

    @property
    def steady_rms(self) -> float:
        if not torch.any(self.steady_mask):
            return float("nan")
        return torch.sqrt(
            torch.mean(self.residual[self.steady_mask].square())
        ).item()


@dataclass(frozen=True)
class NetworkSyncResult:
    """Metrics for an N-station star synchronized to station 0."""

    positions: np.ndarray
    links: list[NetworkLink]
    array_gain: torch.Tensor
    array_steady_mask: torch.Tensor

    @property
    def num_stations(self) -> int:
        return len(self.links) + 1

    @property
    def mean_array_gain(self) -> float:
        if not torch.any(self.array_steady_mask):
            return float("nan")
        return torch.mean(self.array_gain[self.array_steady_mask]).item()

    @property
    def total_airtime_fraction(self) -> float:
        """Pilot airtime summed over the TDMA-multiplexed links."""

        return float(sum(link.airtime_fraction for link in self.links))

    @property
    def worst_station_rms(self) -> float:
        values = [link.steady_rms for link in self.links]
        clean = [value for value in values if value == value]
        return max(clean) if clean else float("nan")

    @property
    def min_detection_rate(self) -> float:
        return min(link.detection_rate for link in self.links)


def run_network_simulation(
    settings: SDRSimulationConfig,
    num_stations: int,
    link_runner,
    extract,
    radius_m: float = 500.0,
    path_loss_exponent: float = 2.7,
    reference_distance_m: float = 100.0,
    min_separation_m: float = 10.0,
) -> NetworkSyncResult:
    """Synchronize ``num_stations`` randomly placed stations to station 0.

    ``link_runner(settings) -> result`` runs the chosen pairwise scheme;
    ``extract(result) -> (residual, steady_mask, detection_rate,
    airtime_fraction)`` pulls the standard metrics out of its result.
    """

    positions = place_stations(
        num_stations, radius_m, settings.seed, min_separation_m
    )
    links: list[NetworkLink] = []
    for station in range(1, num_stations):
        distance = float(
            np.linalg.norm(positions[station] - positions[0])
        )
        distance = max(distance, 1.0)
        snr_db = settings.snr_db - 10.0 * path_loss_exponent * math.log10(
            distance / reference_distance_m
        )
        snr_db = min(snr_db, MAX_LINK_SNR_DB)
        link_settings = replace(
            settings,
            snr_db=snr_db,
            # Independent channel/noise realization per link; the same
            # base seed keeps the whole network reproducible.
            seed=settings.seed + 977 * station,
        )
        result = link_runner(link_settings)
        residual, steady_mask, detection_rate, airtime = extract(result)
        links.append(
            NetworkLink(
                station=station,
                distance_m=distance,
                snr_db=snr_db,
                residual=residual.detach().cpu(),
                steady_mask=steady_mask.detach().cpu(),
                detection_rate=detection_rate,
                airtime_fraction=airtime,
            )
        )

    residuals = torch.stack([link.residual for link in links])
    phasors = torch.exp(1j * residuals.to(torch.complex128))
    # theta_0 = 0: the reference is its own datum.
    total = 1.0 + torch.sum(phasors, dim=0)
    array_gain = torch.abs(total).square() / float(num_stations) ** 2
    steady = links[0].steady_mask.clone()
    for link in links[1:]:
        steady = steady & link.steady_mask
    return NetworkSyncResult(
        positions=positions,
        links=links,
        array_gain=array_gain.real.to(torch.float64),
        array_steady_mask=steady,
    )
