"""End-to-end roll correction sign test.

Tests the complete chain: roll error → WBC My → force distributor → wheel forces → achieved My.
This will definitively determine if the sign convention is correct or inverted.
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
from wheeled_biped.controllers.simple_force_distributor import SimpleForceDistributor
from wheeled_biped.controllers.contact_jacobian import ContactJacobian
from wheeled_biped.controllers.orientation_utils import compute_orientation_from_gravity


def load_model():
    """Load MuJoCo model."""
    xml_path = Path("assets/robot/wheeled_biped_real.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    return model, data


def test_end_to_end_roll_correction():
    """Test complete roll correction chain."""
    print("=" * 80)
    print("END-TO-END ROLL CORRECTION SIGN TEST")
    print("=" * 80)

    model, data = load_model()
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    # Create components
    state_estimator = CentroidalStateEstimator(
        CentroidalStateEstimatorConfig(
            robot_mass=8.1,
            torso_inertia=jnp.array([0.1, 0.1, 0.05]),
        ),
        mj_model=model,
    )

    wrench_computer = CentroidalWrenchComputer(
        robot_mass=8.1,
        k_roll=100.0,
        k_roll_rate=20.0,
        k_pitch=100.0,
        k_pitch_rate=20.0,
        k_height=500.0,
    )

    distributor = SimpleForceDistributor(
        tau_hip_roll_max=15.0,
        max_force_asymmetry=40.0,
        min_wheel_force=10.0,
    )

    contact_jacobian = ContactJacobian(model)

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

    # Get wheel positions
    l_contact_world, r_contact_world = contact_jacobian.get_wheel_contact_points(data)
    com_pos_np = np.array(centroidal_state_eq.com_pos)
    wheel_pos_left = jnp.array(np.array(l_contact_world) - com_pos_np)
    wheel_pos_right = jnp.array(np.array(r_contact_world) - com_pos_np)

    print(f"\n[Wheel Positions Relative to CoM]")
    print(f"  Left:  X={wheel_pos_left[0]:+.3f}, Y={wheel_pos_left[1]:+.3f}, Z={wheel_pos_left[2]:+.3f}")
    print(f"  Right: X={wheel_pos_right[0]:+.3f}, Y={wheel_pos_right[1]:+.3f}, Z={wheel_pos_right[2]:+.3f}")
    print(f"  Separation: X={wheel_pos_left[0] - wheel_pos_right[0]:.3f} m (lateral)")

    # Test case: Negative roll error (robot rolling left)
    print(f"\n[Test: Negative Roll Error (Robot Rolling Left)]")
    print(f"  Scenario: Robot has rolled -10° to the left")
    print(f"  Expected: Corrections should push RIGHT wheel harder to restore balance")
    print(f"  Expected: Achieved My should OPPOSE the roll error")

    # Create state with negative roll
    state_neg_roll = centroidal_state_eq.replace(
        body_roll_y=float(roll_y_eq) - 0.175,  # -10 deg
        body_roll_rate_y=0.0,
    )

    # Step 1: WBC computes correction wrench
    desired_force, desired_moment = wrench_computer.compute_desired_wrench_from_state(
        state_neg_roll, height_cmd=0.55, roll_integral=0.0
    )
    wrench = jnp.concatenate([desired_force, desired_moment])
    my_wbc = float(wrench[4])

    print(f"\n[Step 1: WBC Correction]")
    print(f"  Roll error: {-10.0:.1f} deg (negative = rolling left)")
    print(f"  WBC My correction: {my_wbc:+.2f} Nm")

    # Step 2: Force distributor converts to wheel forces
    f_left, f_right, tau_hip_roll, diagnostics = distributor.distribute_wrench_contact_aware(
        desired_wrench=wrench,
        left_contact=True,
        right_contact=True,
        wheel_pos_left=wheel_pos_left,
        wheel_pos_right=wheel_pos_right,
        hip_roll_authority_scale=1.0,
        distribution_mode="absolute",
    )

    fz_left = float(f_left[2])
    fz_right = float(f_right[2])

    print(f"\n[Step 2: Force Distribution]")
    print(f"  Left wheel Fz:  {fz_left:+.2f} N")
    print(f"  Right wheel Fz: {fz_right:+.2f} N")
    print(f"  Force asymmetry: {fz_left - fz_right:+.2f} N")

    # Step 3: Compute achieved moment from wheel forces
    # My = -x_l * fz_l - x_r * fz_r (cross product formula)
    x_l = float(wheel_pos_left[0])
    x_r = float(wheel_pos_right[0])
    achieved_my = -x_l * fz_left - x_r * fz_right

    print(f"\n[Step 3: Achieved Moment]")
    print(f"  Achieved My: {achieved_my:+.2f} Nm")
    print(f"  Formula: My = -x_l*fz_l - x_r*fz_r")
    print(f"  Formula: My = -({x_l:+.3f})*({fz_left:+.1f}) - ({x_r:+.3f})*({fz_right:+.1f})")

    # Step 4: Verify correction opposes error
    roll_error = -10.0  # degrees

    print(f"\n[Step 4: Verification]")
    print(f"  Roll error: {roll_error:+.1f} deg")
    print(f"  Achieved My: {achieved_my:+.2f} Nm")

    # For negative roll (rolling left), we need positive My to roll right
    # For positive roll (rolling right), we need negative My to roll left
    # So: achieved_my should have OPPOSITE sign of roll_error

    if roll_error < 0 and achieved_my > 0:
        print(f"  [OK] CORRECT: Negative roll produces positive My (opposes error)")
        verdict = "CORRECT"
    elif roll_error < 0 and achieved_my < 0:
        print(f"  [!!] WRONG: Negative roll produces negative My (amplifies error!)")
        verdict = "INVERTED"
    else:
        print(f"  [??] UNCLEAR: Achieved My is near zero")
        verdict = "UNCLEAR"

    print(f"\n[Conclusion]")
    print(f"  Sign convention: {verdict}")
    if verdict == "INVERTED":
        print(f"  FIX: Invert WBC roll correction sign")
        print(f"  Change: m_roll_y = +k_roll * roll_error")
        print(f"  To:     m_roll_y = -k_roll * roll_error")
    elif verdict == "CORRECT":
        print(f"  Current sign convention is correct")
        print(f"  Issue must be elsewhere (gains too weak, saturation, etc.)")

    return verdict


if __name__ == "__main__":
    test_end_to_end_roll_correction()
