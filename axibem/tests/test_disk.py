"""Sanity test: axibem disk capacitance vs Kelvin closed form."""

from axibem.disk import disk_capacitance, kelvin_disk_capacitance


def test_disk_unit_radius():
    C_bem = disk_capacitance(radius=1.0, n_panels=200, n_quad=8)
    C_ref = kelvin_disk_capacitance(1.0)
    rel_err = abs(C_bem - C_ref) / C_ref
    # Constant-density piecewise-constant BEM is a low-order method; we want
    # the answer within a few percent of Kelvin's exact result.
    assert rel_err < 0.05, f"BEM C = {C_bem:.4e}, ref = {C_ref:.4e}, rel_err = {rel_err:.3%}"


def test_disk_scales_with_radius():
    R = 2.5
    C_bem = disk_capacitance(radius=R, n_panels=200, n_quad=8)
    C_ref = kelvin_disk_capacitance(R)
    assert abs(C_bem - C_ref) / C_ref < 0.05
