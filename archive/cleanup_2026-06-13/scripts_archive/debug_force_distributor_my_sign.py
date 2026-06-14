"""Verify force distributor My sign convention.

Tests whether the force distributor produces the correct sign of My
for a given force asymmetry with X-axis lateral wheel separation.
"""

import jax.numpy as jnp
import numpy as np

from wheeled_biped.controllers.simple_force_distributor import SimpleForceDistributor
from wheeled_biped.controllers.contact_jacobian import ContactJacobian


def test_my_sign_convention():
    """Test My sign convention with actual wheel positions."""
    print("=" * 80)
    print("FORCE DISTRIBUTOR My SIGN CONVENTION TEST")
    print("=" * 80)

    distributor = SimpleForceDistributor(
        tau_hip_roll_max=15.0,
        max_force_asymmetry=40.0,
        min_wheel_force=10.0,
    )

    # Actual wheel positions from geometry audit (X-axis lateral separation)
    # Left wheel at X = +0.173, Right wheel at X = -0.173
    wheel_pos_left = jnp.array([+0.173, 0.0, 0.0])
    wheel_pos_right = jnp.array([-0.173, 0.0, 0.0])

    print(f"\n[Wheel Positions]")
    print(f"  Left:  X={wheel_pos_left[0]:+.3f}, Y={wheel_pos_left[1]:+.3f}, Z={wheel_pos_left[2]:+.3f}")
    print(f"  Right: X={wheel_pos_right[0]:+.3f}, Y={wheel_pos_right[1]:+.3f}, Z={wheel_pos_right[2]:+.3f}")

    # Test case 1: Request positive My (roll right)
    print(f"\n[Test 1: Request My = +10 Nm (roll right)]")
    wrench_pos = jnp.array([0.0, 0.0, 80.0, 0.0, +10.0, 0.0])

    f_left, f_right, tau_hip_roll, diagnostics = distributor.distribute_wrench_contact_aware(
        desired_wrench=wrench_pos,
        left_contact=True,
        right_contact=True,
        wheel_pos_left=wheel_pos_left,
        wheel_pos_right=wheel_pos_right,
        hip_roll_authority_scale=1.0,
        distribution_mode="absolute",
    )

    # Compute achieved My from forces
    x_l = float(wheel_pos_left[0])
    x_r = float(wheel_pos_right[0])
    fz_l = float(f_left[2])
    fz_r = float(f_right[2])

    # Standard moment formula: M = r × F
    # For roll moment My about Y-axis with forces in Z-direction:
    # My = x_l * fz_l + x_r * fz_r (standard cross product)
    achieved_my_standard = x_l * fz_l + x_r * fz_r

    # Alternative formula (what we implemented):
    # My = -x_l * fz_l - x_r * fz_r
    achieved_my_implemented = -x_l * fz_l - x_r * fz_r

    print(f"  Requested My: {+10.0:+.2f} Nm")
    print(f"  Left force:   {fz_l:+.2f} N")
    print(f"  Right force:  {fz_r:+.2f} N")
    print(f"  Force diff:   {fz_l - fz_r:+.2f} N")
    print(f"  Achieved My (standard formula):     {achieved_my_standard:+.2f} Nm")
    print(f"  Achieved My (implemented formula):  {achieved_my_implemented:+.2f} Nm")
    print(f"  Error (standard):     {abs(achieved_my_standard - 10.0):.2f} Nm")
    print(f"  Error (implemented):  {abs(achieved_my_implemented - 10.0):.2f} Nm")

    # Test case 2: Request negative My (roll left)
    print(f"\n[Test 2: Request My = -10 Nm (roll left)]")
    wrench_neg = jnp.array([0.0, 0.0, 80.0, 0.0, -10.0, 0.0])

    f_left, f_right, tau_hip_roll, diagnostics = distributor.distribute_wrench_contact_aware(
        desired_wrench=wrench_neg,
        left_contact=True,
        right_contact=True,
        wheel_pos_left=wheel_pos_left,
        wheel_pos_right=wheel_pos_right,
        hip_roll_authority_scale=1.0,
        distribution_mode="absolute",
    )

    fz_l = float(f_left[2])
    fz_r = float(f_right[2])

    achieved_my_standard = x_l * fz_l + x_r * fz_r
    achieved_my_implemented = -x_l * fz_l - x_r * fz_r

    print(f"  Requested My: {-10.0:+.2f} Nm")
    print(f"  Left force:   {fz_l:+.2f} N")
    print(f"  Right force:  {fz_r:+.2f} N")
    print(f"  Force diff:   {fz_l - fz_r:+.2f} N")
    print(f"  Achieved My (standard formula):     {achieved_my_standard:+.2f} Nm")
    print(f"  Achieved My (implemented formula):  {achieved_my_implemented:+.2f} Nm")
    print(f"  Error (standard):     {abs(achieved_my_standard + 10.0):.2f} Nm")
    print(f"  Error (implemented):  {abs(achieved_my_implemented + 10.0):.2f} Nm")

    print(f"\n[Conclusion]")
    if abs(achieved_my_standard - 10.0) < 0.1 and abs(achieved_my_standard + 10.0) < 0.1:
        print(f"  ✓ Standard formula (My = x_l*fz_l + x_r*fz_r) is CORRECT")
        print(f"  ✗ Implemented formula (My = -x_l*fz_l - x_r*fz_r) is WRONG")
        print(f"  FIX: Remove negative signs from moment formula")
    elif abs(achieved_my_implemented - 10.0) < 0.1 and abs(achieved_my_implemented + 10.0) < 0.1:
        print(f"  ✗ Standard formula (My = x_l*fz_l + x_r*fz_r) is WRONG")
        print(f"  ✓ Implemented formula (My = -x_l*fz_l - x_r*fz_r) is CORRECT")
    else:
        print(f"  ✗ Neither formula produces correct My")
        print(f"  Need to investigate moment sign convention")


if __name__ == "__main__":
    test_my_sign_convention()
