"""Coherent detection viability of a passive target (drone) using the
synchronized distributed array. Pure addition: consumes the sync
simulators' results through their public APIs, modifies nothing."""

from .viability import (
    DetectionParams,
    MethodViability,
    coherent_snr_factor,
    detection_range_m,
    evaluate_method,
    probability_of_detection,
    required_snr,
    single_node_snr,
)

__all__ = [
    "DetectionParams",
    "MethodViability",
    "coherent_snr_factor",
    "detection_range_m",
    "evaluate_method",
    "probability_of_detection",
    "required_snr",
    "single_node_snr",
]
