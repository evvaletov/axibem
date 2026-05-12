"""Axisymmetric free-space electrostatic kernel.

The rotationally-integrated Green's function for a ring of unit linear charge
density at `(r_s, z_s)` evaluated at the field point `(r, z)` is expressible
in terms of the complete elliptic integral of the first kind.
"""

import numpy as np
from scipy.special import ellipk

EPS0 = 8.8541878128e-12  # F/m


def ring_potential_kernel(r, z, r_s, z_s):
    """Potential at (r, z) from a ring of unit linear charge density at (r_s, z_s).

    Returns the rotationally-integrated free-space Green's function value with
    a factor of `1 / eps0` already included, so the contribution to the
    potential from a ring of total charge `q_ring = 2 * pi * r_s * sigma` is
    `sigma * ring_potential_kernel(...)`.

    Inputs may be scalars or broadcast-compatible numpy arrays.
    """
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    r_s = np.asarray(r_s, dtype=float)
    z_s = np.asarray(z_s, dtype=float)

    dz = z - z_s
    rad_sum_sq = (r + r_s) ** 2 + dz ** 2
    m = 4.0 * r * r_s / np.maximum(rad_sum_sq, 1e-300)
    m = np.clip(m, 0.0, 1.0 - 1e-15)

    K = ellipk(m)
    denom = np.sqrt(rad_sum_sq)
    return (1.0 / (np.pi * EPS0)) * r_s * K / denom
