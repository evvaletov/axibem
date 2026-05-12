"""Charged thin disk: self-capacitance via axisymmetric collocation BEM."""

import numpy as np
from scipy.linalg import solve

from .kernel import EPS0, ring_potential_kernel


def _disk_panels(radius, n_panels):
    """Equal-width annular panels on a thin disk of given radius.

    Each panel is a thin ring located at z = 0 with inner radius `r_inner`,
    outer radius `r_outer`, and a collocation midpoint at `r_mid`.
    """
    edges = np.linspace(0.0, radius, n_panels + 1)
    r_inner = edges[:-1]
    r_outer = edges[1:]
    r_mid = 0.5 * (r_inner + r_outer)
    # Panel arc length in the meridional plane (a flat disk has dl = dr).
    dl = r_outer - r_inner
    return r_mid, dl


def _influence_matrix(r_mid, dl, n_quad=8):
    """Assemble the dense influence matrix A_ij = potential at panel i from unit
    surface charge density on panel j.

    Each panel's contribution is integrated along its arc length using
    Gauss-Legendre quadrature; this handles the K(m) logarithmic singularity
    on the diagonal adequately for moderate panel counts.
    """
    n = r_mid.size
    A = np.zeros((n, n))

    # Gauss-Legendre nodes/weights on [-1, 1]
    gx, gw = np.polynomial.legendre.leggauss(n_quad)

    for j in range(n):
        # Source panel j spans [r_mid[j] - dl[j]/2, r_mid[j] + dl[j]/2] at z=0
        a = r_mid[j] - 0.5 * dl[j]
        b = r_mid[j] + 0.5 * dl[j]
        # Map Gauss-Legendre nodes from [-1, 1] to [a, b]
        rs = 0.5 * (b - a) * gx + 0.5 * (a + b)
        ws = 0.5 * (b - a) * gw
        zs = np.zeros_like(rs)
        # Sum contributions at each observation panel midpoint
        for i in range(n):
            G = ring_potential_kernel(r_mid[i], 0.0, rs, zs)
            A[i, j] = np.sum(ws * G)
    return A


def disk_capacitance(radius=1.0, n_panels=200, n_quad=8, voltage=1.0):
    """Self-capacitance of an isolated thin disk at uniform potential `voltage`.

    Returns the computed capacitance in farads. Kelvin's exact result for the
    isolated thin disk is `C = 8 * eps0 * R`, providing a clean reference.

    Parameters
    ----------
    radius : float
        Disk radius in metres.
    n_panels : int
        Number of equal-width annular panels.
    n_quad : int
        Gauss-Legendre quadrature order per panel.
    voltage : float
        Disk potential in volts.
    """
    r_mid, dl = _disk_panels(radius, n_panels)
    A = _influence_matrix(r_mid, dl, n_quad=n_quad)

    rhs = voltage * np.ones_like(r_mid)
    sigma = solve(A, rhs)  # surface charge density on each panel

    # Total charge: sigma is the (total, both-faces-combined) surface charge
    # density on the mathematical thin disk, so Q = ∫ σ dA with dA = 2π r dr.
    ring_area = 2.0 * np.pi * r_mid * dl
    Q_total = np.sum(sigma * ring_area)

    return Q_total / voltage


def kelvin_disk_capacitance(radius):
    """Kelvin's closed-form self-capacitance of an isolated thin disk."""
    return 8.0 * EPS0 * radius
