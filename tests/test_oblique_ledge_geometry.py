"""The oblique-ledge course must render the edge angle it was asked for.

Until 2026-08-02 ``build_terrain_xml`` anchored the cliff slab by its BURIED
edge, so the exposed edge drifted sideways by 2*pl*sin(phi).  The slab's end
corner then reached the driving line at exactly tan(phi) = pw/(2*pl) = 1: at
45 deg the course rendered a wedge APEX (both wheels leaving together, i.e. a
square ledge) and past 45 deg the exposed boundary was the slab's side edge,
so 60 deg rendered as a mirrored 30 deg.  Nothing crashed -- the numbers were
just measured on the wrong terrain.
"""
import numpy as np
import pytest

from scripts.ramp_step_tests import build_terrain_xml, ground_z_xy


def _edge_y(x, g, sign):
    """First eff_y (metres along the drive) at which the ground drops away."""
    ys = np.arange(g["y1"], g["y1"] + 3.2, 0.002)
    on = [y for y in ys if ground_z_xy(x, sign * y, g) >= g["h"] - 1e-9]
    assert on, f"no platform found at x={x}"
    return max(on)


@pytest.mark.parametrize("angle", [15.0, 30.0, 45.0, 60.0])
@pytest.mark.parametrize("sign", [-1, 1])
def test_rendered_edge_angle_matches_request(angle, sign):
    xml, g = build_terrain_xml(0.30, sign=sign, edge_angle_deg=angle)
    xml.unlink()

    xs = np.linspace(-0.20, 0.20, 9)          # the track the robot occupies
    ys = np.array([_edge_y(x, g, sign) for x in xs])

    # A true oblique edge is a straight line in (x, y), monotone in x, with
    # slope tan(phi).  A wedge apex is symmetric -- it fails monotonicity.
    d = np.diff(ys)
    assert np.all(d > 0) or np.all(d < 0), (
        f"edge is not monotone in x at {angle} deg -- apex/corner, not an edge:"
        f" {np.round(ys, 3)}")

    fitted = np.degrees(np.arctan(abs(np.polyfit(xs, ys, 1)[0])))
    assert fitted == pytest.approx(angle, abs=1.5), (
        f"requested {angle} deg, rendered {fitted:.1f} deg")

    resid = np.max(np.abs(ys - np.polyval(np.polyfit(xs, ys, 1), xs)))
    assert resid < 0.02, f"edge not straight across the track (resid {resid:.3f} m)"


def test_both_wheels_do_not_cross_together_at_45deg():
    """The regression's signature: at 45 deg the old course dropped the whole
    track at one y, so the stagger the oblique case exists to test was zero."""
    xml, g = build_terrain_xml(0.30, sign=-1, edge_angle_deg=45.0)
    xml.unlink()
    stagger = abs(_edge_y(-0.139, g, -1) - _edge_y(0.139, g, -1))  # half-track
    assert stagger == pytest.approx(0.278 * np.tan(np.radians(45.0)), abs=0.03)
