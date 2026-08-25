"""The pi ambiguity of two-way phase synchronization: derivation,
1-bit sufficiency, and the periodic-check requirement.

This file is the analysis the implementation relies on; the companion
measurement study is ``pi_ambiguity_study.py``.

1. Where the ambiguity comes from
---------------------------------

Stations A and B exchange captures over a reciprocal channel. With
oscillator offset theta = phi_A - phi_B and common channel phase c,
the two one-way phase measurements are (up to noise)

    phi_fwd = wrap( theta + c ),        phi_rev = wrap( -theta + c ),

and the reciprocity trick forms the half-difference

    z = (1/2) * wrap( phi_fwd - phi_rev ).

Without the wrap, phi_fwd - phi_rev = 2*theta and z = theta exactly.
With the wrap, only wrap(2*theta) is observable, and the doubling map
is 2-to-1 on the circle:

    wrap(2*(theta + pi)) = wrap(2*theta + 2*pi) = wrap(2*theta).

So theta and theta + pi produce the *identical* observable. Halving
wrap(2*theta) returns a value in (-pi/2, pi/2]; the true offset is
either that value z or z + pi, and no amount of averaging, SNR, or
filtering can distinguish the two from half-difference data alone:
the measurement determines theta only modulo pi. The ambiguity set
has exactly two elements, {z, wrap(z + pi)}.

2. Why the wrong branch is an attractor, not a glitch
------------------------------------------------------

The tracking loop resolves the branch by picking the candidate
closest to its own prediction (``_pick_half_phase``). At acquisition
the prediction is 0, so the loop acquires the branch nearest zero:
if the true offset lies in (pi/2, pi], the loop locks to theta - pi.
After the correction is applied, the residual sits at +-pi -- and
every subsequent measurement is *consistent* with the estimate
(innovation ~ 0), because theta = pi maps to the same observable as
theta = 0. The anti-phase point is a fixed point of the closed loop
with zero innovation: internally the loop looks locked while the two
transmitters cancel. This is the mechanism behind the measured
anti-phase capture of consensus sync over a real channel (3009 mrad
residual, 0.68% coherent gain, seed 0) and behind branch loss during
coasting: if the true phase drifts more than pi/2 from the
prediction between services, the next service picks the wrong branch
and the error self-sustains.

3. Why one bit is exactly sufficient
-------------------------------------

The residual ambiguity is a binary choice: theta* in {z, z + pi}.
The two candidates differ by exactly pi, so

    cos(theta - z)  and  cos(theta - (z + pi)) = -cos(theta - z)

differ in sign. Any physical comparison that reveals
b = sign(cos(theta_err)) -- e.g. a beacon transmitted by both
stations and compared for constructive vs destructive combining --
decides the branch: b >= 0 keeps the current branch, b < 0 flips it
by adding pi. One bit resolves a two-element ambiguity set; by
counting, no less information suffices and no more is needed. The
implementation's check (flip when cos(ref - slave) < -0.2, with the
-0.2 hysteresis so noise near +-pi/2 does not chatter) is this bit.

4. Why the check must be periodic, and how often
-------------------------------------------------

A one-shot check at acquisition is insufficient: the branch can be
lost *later*, whenever the coasting/prediction error exceeds pi/2 in
magnitude. If the phase prediction error at service time is
~ N(0, sigma^2) (sigma = the loop's prediction-error scale, set by
oscillator class and coast length), the per-service probability of a
branch crossing is

    p_cross(sigma) = P(|e| > pi/2) = erfc( (pi/2) / (sigma*sqrt(2)) ).

For a check every C service intervals, a crossing goes uncorrected
for on average d = (C + 1)/2 intervals, so the steady fraction of
time spent anti-phase follows the renewal estimate

    f_anti(C) ~= p_cross * d / (1 + p_cross * d),   d = (C + 1)/2.

Inverting gives the check-period requirement for a target anti-phase
dwell fraction f_t:

    C_max ~= 2 * f_t / ( p_cross * (1 - f_t) ) - 1.

The oscillator-class dependence is severe because p_cross is a tail
probability: at sigma ~ 0.13 rad (TCXO-class per-interval prediction
error) p_cross ~ 1e-33 and any check period works after acquisition;
at sigma ~ 0.64 rad (SDR-class) p_cross ~ 1.5e-2 and the check must
run every few intervals. A check bit with error rate eps also
*causes* flips when correctly aligned, adding ~ eps * d dwell, so
the bit channel must satisfy eps << f_t / d.
"""

from __future__ import annotations

import math


def p_cross(sigma_rad: float) -> float:
    """Per-service probability that the prediction error crosses the
    pi/2 branch boundary: P(|N(0, sigma^2)| > pi/2)."""

    if sigma_rad <= 0.0:
        return 0.0
    return math.erfc((math.pi / 2.0) / (sigma_rad * math.sqrt(2.0)))


def anti_phase_fraction(sigma_rad: float, check_every: int | None,
                        bit_error: float = 0.0) -> float:
    """Renewal estimate of the steady anti-phase dwell fraction for a
    check every ``check_every`` service intervals (None = no check,
    which is absorbing: fraction -> ~1/2 of post-crossing time; we
    return 1.0 as 'unbounded dwell' sentinel)."""

    p = p_cross(sigma_rad)
    if check_every is None:
        return 1.0 if p > 0.0 else 0.0
    d = (check_every + 1) / 2.0
    crossings = p * d + bit_error * d
    return crossings / (1.0 + crossings)


def max_check_period(sigma_rad: float, target_fraction: float) -> float:
    """Largest check period keeping the anti-phase dwell fraction at or
    below ``target_fraction`` (bit errors ignored)."""

    p = p_cross(sigma_rad)
    if p <= 0.0:
        return float("inf")
    return 2.0 * target_fraction / (p * (1.0 - target_fraction)) - 1.0


if __name__ == "__main__":
    print("per-service branch-crossing probability p_cross(sigma):")
    for sigma in (0.13, 0.3, 0.45, 0.64, 0.9):
        print(f"  sigma {sigma:4.2f} rad  p_cross {p_cross(sigma):.3e}  "
              f"C_max(f_t=5%) {max_check_period(sigma, 0.05):.1f}")
