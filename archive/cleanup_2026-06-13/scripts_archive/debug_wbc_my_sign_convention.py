"""Verify WBC My sign convention.

Tests whether the WBC correctly maps roll error to My correction sign.
For a wheeled biped with X-axis lateral wheel separation:
- Negative roll (robot rolling left) should request positive My (roll right correction)
- Positive roll (robot rolling right) should request negative My (roll left correction)
"""

import jax.numpy as jnp
import mujoco
import numpy as np
from pathlib import Path

from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)
from wheeled_biped.controllers.centroidal_wrench_computer import CentroidalWrenchComputer
from wheeled_biped.controllers.orientation_utils import compute_orientation_from_gravity


def load_model():
    """Load MuJoCo model."""
    xml_path = Path("assets/robot/wheeled_biped_real.xml")
    if not xml_path.exists():
        raise FileNotFoundError(f"Model file not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    return model, data


def test_wbc_my_sign_convention():
    """Test WBC My sign convention."""
    print("=" * 80)
    print("WBC My SIGN CONVENTION TEST")
    print("=" * 80)

    model, data = load_model()

    # Reset to keyframe and calibrate
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    # Create state estimator
    state_estimator = CentroidalStateEstimator(
        CentroidalStateEstimatorConfig(
            robot_mass=8.1,
            torso_inertia=jnp.array([0.1, 0.1, 0.05]),
        ),
        mj_model=model,
    )

    # Create wrench computer
    wrench_computer = CentroidalWrenchComputer(
        k_pitch=100.0,
        k_pitch_rate=20.0,
        k_roll=100.0,
        k_roll_rate=20.0,
        k_height=500.0,
        k_height_rate=100.0,
        k_capture_point=50.0,
        correction_only_mode=True,
    )

    # Set equilibrium reference
    centroidal_state_eq, _ = state_estimator.estimate(jnp.zeros(42), data, None)
    base_body_id = 1
    R_eq = np.array(data.xmat[base_body_id]).reshape(3, 3)
    gravity_world = np.array([0.0, 0.0, -9.81])
    gravity_body_eq = R_eq.T @ gravity_world
    pitch_x_eq, roll_y_eq = compute_orientation_from_gravity(jnp.array(gravity_body_eq))

    wrench_computer.set_equilibrium_reference(
        com_pos=centroidal_state_eq.com_pos,
        com_z=float(centroidal_state_eq.com_pos[2]),
        pitch_x=float(pitch_x_eq),
        roll_y=float(roll_y_eq),
        capture_point=centroidal_state_eq.capture_point,
        joint_pos=jnp.array(data.qpos[7:17]),
    )

    print(f"\n[Equilibrium Reference]")
    print(f"  Pitch: {float(pitch_x_eq)*57.3:.2f} deg")
    print(f"  Roll:  {float(roll_y_eq)*57.3:.2f} deg")

    # Test case 1: Negative roll error (robot rolling left)
    print(f"\n[Test 1: Negative roll error (robot rolling left)]")

    # Simulate robot with negative roll
    state_neg_roll = centroidal_state_eq.replace(
        roll_y=float(roll_y_eq) - 0.1,  # -0.1 rad = -5.7 deg roll left
        roll_rate_y=0.0,
    )

    wrench_neg_roll = wrench_computer.compute_desired_wrench_from_state(state_neg_roll)

    roll_error = float(state_neg_roll.roll_y - roll_y_eq)
    my_correction = float(wrench_neg_roll[4])

    print(f"  Roll error: {roll_error*57.3:+.2f} deg (negative = rolling left)")
    print(f"  My correction: {my_correction:+.2f} Nm")
    print(f"  Expected: positive My (roll right correction)")

    if my_correction > 0:
        print(f"  ✓ CORRECT: Negative roll error produces positive My")
    else:
        print(f"  ✗ WRONG: Negative roll error produces negative My (amplifies error!)")

    # Test case 2: Positive roll error (robot rolling right)
    print(f"\n[Test 2: Positive roll error (robot rolling right)]")

    state_pos_roll = centroidal_state_eq.replace(
        roll_y=float(roll_y_eq) + 0.1,  # +0.1 rad = +5.7 deg roll right
        roll_rate_y=0.0,
    )

    wrench_pos_roll = wrench_computer.compute_desired_wrench_from_state(state_pos_roll)

    roll_error = float(state_pos_roll.roll_y - roll_y_eq)
    my_correction = float(wrench_pos_roll[4])

    print(f"  Roll error: {roll_error*57.3:+.2f} deg (positive = rolling right)")
    print(f"  My correction: {my_correction:+.2f} Nm")
    print(f"  Expected: negative My (roll left correction)")

    if my_correction < 0:
        print(f"  ✓ CORRECT: Positive roll error produces negative My")
    else:
        print(f"  ✗ WRONG: Positive roll error produces positive My (amplifies error!)")

    print(f"\n[Conclusion]")
    print(f"  If both tests show WRONG, the WBC My sign convention is inverted")
    print(f"  FIX: Invert My sign in centroidal_wrench_computer.py")


if __name__ == "__main__":
    test_wbc_my_sign_convention()
