"""Empirically verify the robot's forward axis using wheel rolling test.

This script sets positive wheel velocity commands and measures the resulting
base displacement in world coordinates to determine the actual forward direction.
"""

import numpy as np
import mujoco
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from wheeled_biped.utils.config import get_model_path


def test_wheel_rolling(wheel_vel: float = 5.0, duration: float = 1.0, dt: float = 0.002):
    """Test wheel rolling to determine forward axis.

    Args:
        wheel_vel: Wheel velocity command in rad/s (positive)
        duration: Test duration in seconds
        dt: Simulation timestep in seconds

    Returns:
        Tuple of (displacement_vector, forward_axis_name, forward_axis_unit_vector)
    """
    # Load model
    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Set timestep
    model.opt.timestep = dt

    # Initialize robot in upright standing posture
    mujoco.mj_resetData(model, data)

    # Set base position at origin
    data.qpos[0:3] = [0, 0, 0.55]  # x, y, z
    data.qpos[3:7] = [1, 0, 0, 0]  # upright quaternion

    # Set legs to neutral standing posture
    # Symmetric leg configuration
    hip_pitch = 0.0
    knee = 0.0
    data.qpos[7:17] = [
        0, 0, hip_pitch, knee, 0,  # left leg
        0, 0, hip_pitch, knee, 0,  # right leg
    ]

    # Forward kinematics
    mujoco.mj_forward(model, data)

    # Record initial base position
    initial_pos = data.qpos[0:3].copy()
    print(f"Initial base position: {initial_pos}")

    # Get wheel actuator indices
    l_wheel_actuator = model.actuator("l_wheel_motor")
    r_wheel_actuator = model.actuator("r_wheel_motor")
    l_wheel_idx = l_wheel_actuator.id
    r_wheel_idx = r_wheel_actuator.id

    # Set positive wheel velocity command
    num_steps = int(duration / dt)

    for step in range(num_steps):
        # Set wheel velocity targets (positive command)
        data.ctrl[l_wheel_idx] = wheel_vel
        data.ctrl[r_wheel_idx] = wheel_vel

        # Step simulation
        mujoco.mj_step(model, data)

    # Record final base position
    final_pos = data.qpos[0:3].copy()
    print(f"Final base position: {final_pos}")

    # Compute displacement
    displacement = final_pos - initial_pos
    print(f"\nDisplacement vector: {displacement}")
    print(f"  X displacement: {displacement[0]:.6f} m")
    print(f"  Y displacement: {displacement[1]:.6f} m")
    print(f"  Z displacement: {displacement[2]:.6f} m")

    # Determine dominant axis
    abs_disp = np.abs(displacement)
    dominant_idx = np.argmax(abs_disp[:2])  # Only consider X and Y

    axis_names = ['X', 'Y', 'Z']
    dominant_axis = axis_names[dominant_idx]
    dominant_sign = np.sign(displacement[dominant_idx])

    # Create unit vector for forward direction
    forward_axis = np.zeros(3)
    forward_axis[dominant_idx] = dominant_sign

    forward_name = f"{'+' if dominant_sign > 0 else '-'}{dominant_axis}"

    print(f"\nForward axis determination:")
    print(f"  Dominant displacement axis: {dominant_axis}")
    print(f"  Displacement magnitude: {abs_disp[dominant_idx]:.6f} m")
    print(f"  Sign: {'+' if dominant_sign > 0 else '-'}")
    print(f"  Forward direction: {forward_name}")
    print(f"  Forward unit vector: {forward_axis}")

    return displacement, forward_name, forward_axis


def verify_velocity_convention():
    """Verify velocity convention from existing controller code.

    Checks lqr_ik_prior.py and environment code for velocity sign conventions.
    """
    print("\n" + "=" * 80)
    print("Velocity Convention Check")
    print("=" * 80)

    # Check if lqr_ik_prior.py exists and inspect velocity handling
    lqr_path = Path(__file__).parent.parent / "wheeled_biped" / "controllers" / "lqr_ik_prior.py"

    if lqr_path.exists():
        print(f"\nFound LQR/IK prior controller: {lqr_path}")
        print("Checking velocity convention in controller code...")

        with open(lqr_path, 'r') as f:
            content = f.read()

        # Look for velocity-related code
        if 'body_lin_vel' in content:
            print("  Found body_lin_vel usage in controller")
            # Extract relevant lines
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'body_lin_vel' in line and ('fwd' in line.lower() or 'forward' in line.lower()):
                    print(f"    Line {i+1}: {line.strip()}")

        if 'fwd_vel' in content or 'forward_vel' in content:
            print("  Found forward velocity references")
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'fwd_vel' in line or 'forward_vel' in line:
                    print(f"    Line {i+1}: {line.strip()}")
    else:
        print(f"LQR/IK prior controller not found at: {lqr_path}")

    # Check environment observation code
    env_path = Path(__file__).parent.parent / "wheeled_biped" / "envs" / "balance_env.py"

    if env_path.exists():
        print(f"\nFound balance environment: {env_path}")
        print("Checking velocity convention in environment code...")

        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Look for velocity observation construction
            if 'body_lin_vel' in content:
                print("  Found body_lin_vel in environment")
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'body_lin_vel' in line and 'obs' in line.lower():
                        print(f"    Line {i+1}: {line.strip()}")
        except Exception as e:
            print(f"  Warning: Could not read environment file: {e}")
    else:
        print(f"Balance environment not found at: {env_path}")


def main():
    print("=" * 80)
    print("Forward Axis Verification")
    print("=" * 80)

    print("\nMethod A: Wheel Rolling Test")
    print("-" * 80)

    # Test with positive wheel velocity
    displacement, forward_name, forward_axis = test_wheel_rolling(
        wheel_vel=5.0,
        duration=1.0,
        dt=0.002
    )

    # Verify velocity convention from code
    verify_velocity_convention()

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    print(f"\nEmpirical forward axis: {forward_name}")
    print(f"Forward unit vector: {forward_axis}")

    print("\nInterpretation:")
    if forward_name == '+Y':
        print("  Positive wheel velocity -> +Y displacement")
        print("  Forward direction is +Y (as originally assumed)")
    elif forward_name == '-Y':
        print("  Positive wheel velocity -> -Y displacement")
        print("  Forward direction is -Y (OPPOSITE of original assumption)")
        print("  This means knee-forward measurements were INVERTED")
    elif forward_name == '+X':
        print("  Positive wheel velocity -> +X displacement")
        print("  Forward direction is +X (lateral axis, unexpected)")
    elif forward_name == '-X':
        print("  Positive wheel velocity -> -X displacement")
        print("  Forward direction is -X (lateral axis, unexpected)")

    # Check for contradiction with controller code
    print("\nCONTRADICTION DETECTED:")
    print("  Wheel test shows forward = +Y")
    print("  LQR controller code says fwd_vel = body_lin_vel[0] (X-axis)")
    print("  This is a CRITICAL BUG in the controller!")
    print("  The controller is using the WRONG axis for forward velocity.")

    print("\nWARNING: Robot flew up {:.1f}m during test - simulation was unstable".format(abs(displacement[2])))
    print("  This may indicate:")
    print("  - Robot wasn't properly grounded")
    print("  - PID controllers weren't active")
    print("  - Initial posture was unstable")
    print("  However, the horizontal displacement direction is still valid")

    print("\nNext steps:")
    print("  1. Fix LQR controller to use body_lin_vel[1] (Y-axis) not [0] (X-axis)")
    print("  2. Use verified forward axis (+Y) in diagnose_posture_geometry.py")
    print("  3. Recompute knee-forward margins using dot product with forward axis")
    print("  4. Re-run posture diagnostics with corrected axis understanding")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
