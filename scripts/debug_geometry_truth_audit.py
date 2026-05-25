"""Geometry Truth Audit.

Establishes ground truth for:
1. Keyframe/root orientation (is robot upright, side-lying, or upside-down?)
2. Wheel body/geom positions (are left/right contact points distinct?)
3. Actual wheel morphology (left/right vs front/back separation?)
4. Which moments can Fz asymmetry physically generate?

Critical issues from axis audit:
- Equilibrium roll = -105.85° (robot appears upside down)
- Both wheels have identical y-coordinates, destroying moment arms
"""

import jax.numpy as jnp
import mujoco
import numpy as np
from pathlib import Path

from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)
from wheeled_biped.controllers.contact_jacobian import ContactJacobian
from wheeled_biped.controllers.orientation_utils import (
    compute_orientation_from_quaternion,
    compute_robot_frame_orientation_from_quaternion,
)


def load_model():
    """Load MuJoCo model."""
    xml_path = Path("assets/robot/wheeled_biped_real.xml")
    if not xml_path.exists():
        raise FileNotFoundError(f"Model file not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    return model, data


def calibrate_root_z_contact(model, data, target_dist=-5e-4, max_iters=50):
    """Calibrate root z-position to achieve target contact distance."""
    for _ in range(max_iters):
        mujoco.mj_forward(model, data)

        if data.ncon == 0:
            data.qpos[2] -= 0.01
            continue

        min_dist = min(data.contact[i].dist for i in range(data.ncon))

        if abs(min_dist - target_dist) < 1e-5:
            break

        delta_z = -(min_dist - target_dist)
        data.qpos[2] += delta_z


def task1_keyframe_root_orientation_audit(model, data, state_estimator):
    """Task 1: Keyframe/root orientation audit."""
    print("=" * 80)
    print("TASK 1: Keyframe/Root Orientation Audit")
    print("=" * 80)

    # Reset to keyframe 0 (same as simulate_hierarchical_controller.py)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    print("\n[Step 1] After keyframe reset:")
    print(f"  qpos[0:3] (root pos): [{data.qpos[0]:.6f}, {data.qpos[1]:.6f}, {data.qpos[2]:.6f}]")
    print(f"  qpos[3:7] (root quat): [{data.qpos[3]:.6f}, {data.qpos[4]:.6f}, {data.qpos[5]:.6f}, {data.qpos[6]:.6f}]")

    # Forward kinematics
    mujoco.mj_forward(model, data)
    print("\n[Step 2] After mj_forward:")

    # Root-z contact calibration
    calibrate_root_z_contact(model, data)
    print(f"\n[Step 3] After root-z contact calibration:")
    print(f"  qpos[2] (root z): {data.qpos[2]:.6f}")

    # Zero velocities and accelerations
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    print("\n[Step 4] After qvel=0 and mj_forward:")

    # Extract orientation information
    quat = data.qpos[3:7]
    print(f"\n[Orientation Analysis]")
    print(f"  Root quaternion: [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]")

    # Get torso body
    torso_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    torso_xmat = data.xmat[torso_body_id].reshape(3, 3)
    print(f"\n  Torso body xmat:")
    print(f"    [{torso_xmat[0,0]:+.6f}, {torso_xmat[0,1]:+.6f}, {torso_xmat[0,2]:+.6f}]")
    print(f"    [{torso_xmat[1,0]:+.6f}, {torso_xmat[1,1]:+.6f}, {torso_xmat[1,2]:+.6f}]")
    print(f"    [{torso_xmat[2,0]:+.6f}, {torso_xmat[2,1]:+.6f}, {torso_xmat[2,2]:+.6f}]")

    # Torso up vector (third column of rotation matrix)
    torso_up = torso_xmat[:, 2]
    world_up = np.array([0.0, 0.0, 1.0])
    up_alignment = np.dot(torso_up, world_up)
    print(f"\n  Torso up vector: [{torso_up[0]:+.6f}, {torso_up[1]:+.6f}, {torso_up[2]:+.6f}]")
    print(f"  World up vector: [{world_up[0]:+.6f}, {world_up[1]:+.6f}, {world_up[2]:+.6f}]")
    print(f"  Alignment (dot product): {up_alignment:+.6f}")

    # Euler angles
    roll_euler, pitch_euler, yaw_euler = compute_orientation_from_quaternion(quat)
    print(f"\n  Euler angles (world frame):")
    print(f"    Roll:  {roll_euler * 180 / np.pi:+8.3f} deg")
    print(f"    Pitch: {pitch_euler * 180 / np.pi:+8.3f} deg")
    print(f"    Yaw:   {yaw_euler * 180 / np.pi:+8.3f} deg")

    # Robot frame
    body_pitch_x, body_roll_y, body_yaw_z = compute_robot_frame_orientation_from_quaternion(quat)
    print(f"\n  Robot frame (from orientation_utils):")
    print(f"    body_pitch_x: {body_pitch_x * 180 / np.pi:+8.3f} deg")
    print(f"    body_roll_y:  {body_roll_y * 180 / np.pi:+8.3f} deg")
    print(f"    body_yaw_z:   {body_yaw_z * 180 / np.pi:+8.3f} deg")

    # State estimator
    state, _ = state_estimator.estimate(jnp.zeros(42), data, None)
    print(f"\n  State estimator:")
    print(f"    state.pitch_x: {state.pitch_x * 180 / np.pi:+8.3f} deg")
    print(f"    state.roll_y:  {state.roll_y * 180 / np.pi:+8.3f} deg")

    # Classification
    print(f"\n[Classification]")
    if up_alignment > 0.9:
        orientation = "UPRIGHT"
    elif up_alignment < -0.9:
        orientation = "UPSIDE-DOWN"
    elif abs(up_alignment) < 0.1:
        orientation = "SIDE-LYING"
    else:
        orientation = "TILTED"

    print(f"  Robot orientation: {orientation}")
    print(f"  Torso up alignment with world +Z: {up_alignment:+.6f}")

    if orientation != "UPRIGHT":
        print(f"\n  WARNING: Robot is {orientation}, not UPRIGHT!")
        print(f"  This explains why roll = {roll_euler * 180 / np.pi:.1f} deg in equilibrium.")
        print(f"  FIX REQUIRED: Keyframe/root orientation must be corrected.")

    return orientation, up_alignment


def task2_wheel_body_geom_truth_audit(model, data):
    """Task 2: Wheel/body/geom truth audit."""
    print("\n" + "=" * 80)
    print("TASK 2: Wheel/Body/Geom Truth Audit")
    print("=" * 80)

    # Get wheel body IDs
    l_wheel_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel")
    r_wheel_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel")

    # Get wheel geom IDs
    l_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")

    print(f"\n[Left Wheel]")
    print(f"  Body ID: {l_wheel_body_id}, Name: l_wheel")
    print(f"  Geom ID: {l_wheel_geom_id}, Name: l_wheel_collision")
    print(f"  Geom body ID: {model.geom_bodyid[l_wheel_geom_id]}")
    print(f"  Body xpos: [{data.xpos[l_wheel_body_id][0]:.6f}, {data.xpos[l_wheel_body_id][1]:.6f}, {data.xpos[l_wheel_body_id][2]:.6f}]")
    print(f"  Geom xpos: [{data.geom_xpos[l_wheel_geom_id][0]:.6f}, {data.geom_xpos[l_wheel_geom_id][1]:.6f}, {data.geom_xpos[l_wheel_geom_id][2]:.6f}]")

    print(f"\n[Right Wheel]")
    print(f"  Body ID: {r_wheel_body_id}, Name: r_wheel")
    print(f"  Geom ID: {r_wheel_geom_id}, Name: r_wheel_collision")
    print(f"  Geom body ID: {model.geom_bodyid[r_wheel_geom_id]}")
    print(f"  Body xpos: [{data.xpos[r_wheel_body_id][0]:.6f}, {data.xpos[r_wheel_body_id][1]:.6f}, {data.xpos[r_wheel_body_id][2]:.6f}]")
    print(f"  Geom xpos: [{data.geom_xpos[r_wheel_geom_id][0]:.6f}, {data.geom_xpos[r_wheel_geom_id][1]:.6f}, {data.geom_xpos[r_wheel_geom_id][2]:.6f}]")

    # Wheel radius
    wheel_radius = model.geom_size[l_wheel_geom_id][0]
    print(f"\n[Wheel Geometry]")
    print(f"  Wheel radius: {wheel_radius:.6f} m")

    # Fallback contact points
    l_fallback = data.geom_xpos[l_wheel_geom_id] - wheel_radius * np.array([0, 0, 1])
    r_fallback = data.geom_xpos[r_wheel_geom_id] - wheel_radius * np.array([0, 0, 1])
    print(f"\n[Fallback Contact Points]")
    print(f"  Left:  [{l_fallback[0]:.6f}, {l_fallback[1]:.6f}, {l_fallback[2]:.6f}]")
    print(f"  Right: [{r_fallback[0]:.6f}, {r_fallback[1]:.6f}, {r_fallback[2]:.6f}]")

    # Actual contacts
    print(f"\n[Actual Contacts]")
    print(f"  Number of contacts: {data.ncon}")

    l_contacts = []
    r_contacts = []

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)

        geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1) or f"geom_{geom1}"
        geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2) or f"geom_{geom2}"

        print(f"\n  Contact {i}:")
        print(f"    Pair: {geom1_name} <-> {geom2_name}")
        print(f"    Position: [{contact.pos[0]:.6f}, {contact.pos[1]:.6f}, {contact.pos[2]:.6f}]")
        print(f"    Distance: {contact.dist:.6f}")

        # Compute force
        force_contact = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force_contact)
        frame = np.array(contact.frame).reshape(3, 3)
        force_world = frame.T @ force_contact[:3]
        print(f"    Force (world): [{force_world[0]:.3f}, {force_world[1]:.3f}, {force_world[2]:.3f}] N")

        if geom1 == l_wheel_geom_id or geom2 == l_wheel_geom_id:
            l_contacts.append(contact.pos.copy())
        if geom1 == r_wheel_geom_id or geom2 == r_wheel_geom_id:
            r_contacts.append(contact.pos.copy())

    print(f"\n[Wheel-Floor Contacts]")
    print(f"  Left wheel contacts: {len(l_contacts)}")
    print(f"  Right wheel contacts: {len(r_contacts)}")

    # Get CoM
    com_pos = data.subtree_com[1]
    print(f"\n[CoM Position]")
    print(f"  CoM: [{com_pos[0]:.6f}, {com_pos[1]:.6f}, {com_pos[2]:.6f}]")

    # Compute wheel positions relative to CoM
    l_wheel_pos = data.xpos[l_wheel_body_id]
    r_wheel_pos = data.xpos[r_wheel_body_id]

    l_rel = l_wheel_pos - com_pos
    r_rel = r_wheel_pos - com_pos

    print(f"\n[Wheel Positions Relative to CoM]")
    print(f"  Left:  [{l_rel[0]:+.6f}, {l_rel[1]:+.6f}, {l_rel[2]:+.6f}]")
    print(f"  Right: [{r_rel[0]:+.6f}, {r_rel[1]:+.6f}, {r_rel[2]:+.6f}]")

    # Difference vector
    diff = l_rel - r_rel
    print(f"\n[Difference Vector (Left - Right)]")
    print(f"  dx: {diff[0]:+.6f} m")
    print(f"  dy: {diff[1]:+.6f} m")
    print(f"  dz: {diff[2]:+.6f} m")

    return l_rel, r_rel, diff


def task3_determine_wheel_morphology(diff):
    """Task 3: Determine actual wheel morphology and controllable moments."""
    print("\n" + "=" * 80)
    print("TASK 3: Determine Wheel Morphology and Controllable Moments")
    print("=" * 80)

    dx = abs(diff[0])
    dy = abs(diff[1])
    dz = abs(diff[2])

    print(f"\n[Wheel Separation]")
    print(f"  |dx| (lateral):   {dx:.6f} m")
    print(f"  |dy| (sagittal):  {dy:.6f} m")
    print(f"  |dz| (vertical):  {dz:.6f} m")

    print(f"\n[XML Convention]")
    print(f"  X = lateral (left/right)")
    print(f"  Y = sagittal (front/back)")
    print(f"  Z = vertical (up/down)")

    print(f"\n[Classification]")

    if dx > 0.1 and dy < 0.05:
        morphology = "LEFT/RIGHT SEPARATED"
        moment_channel = "My (roll moment)"
        formula = "My = y_l * fz_l + y_r * fz_r"
        print(f"  Morphology: {morphology}")
        print(f"  Wheels are separated along X-axis (lateral)")
        print(f"  Vertical Fz asymmetry controls: {moment_channel}")
        print(f"  Formula: {formula}")
    elif dy > 0.1 and dx < 0.05:
        morphology = "FRONT/BACK SEPARATED"
        moment_channel = "Mx (pitch moment)"
        formula = "Mx = y_l * fz_l + y_r * fz_r"
        print(f"  Morphology: {morphology}")
        print(f"  Wheels are separated along Y-axis (sagittal)")
        print(f"  Vertical Fz asymmetry controls: {moment_channel}")
        print(f"  Formula: {formula}")
        print(f"\n  WARNING: Roll cannot be controlled by wheel Fz asymmetry!")
        print(f"  Roll must be handled by hip-roll/internal actuation only.")
    elif dx < 0.01 and dy < 0.01:
        morphology = "IDENTICAL POSITIONS (BROKEN)"
        moment_channel = "NONE"
        formula = "N/A"
        print(f"  Morphology: {morphology}")
        print(f"  ERROR: Wheels have identical positions!")
        print(f"  Moment arm is zero, cannot generate any moment.")
        print(f"\n  CRITICAL BUG: Wheel position extraction or XML geometry is broken.")
    else:
        morphology = "DIAGONAL/COMPLEX"
        moment_channel = "MIXED"
        formula = "Complex"
        print(f"  Morphology: {morphology}")
        print(f"  Wheels have significant separation in multiple axes")
        print(f"  Moment generation is complex, requires careful analysis")

    return morphology, moment_channel


def task4_check_contact_point_extraction(model, data):
    """Task 4: Check whether contact point extraction is wrong."""
    print("\n" + "=" * 80)
    print("TASK 4: Contact Point Extraction Verification")
    print("=" * 80)

    contact_jacobian = ContactJacobian(model)

    # Get wheel geom IDs
    l_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")

    print(f"\n[Geom ID Verification]")
    print(f"  l_wheel_geom_id: {l_wheel_geom_id}")
    print(f"  r_wheel_geom_id: {r_wheel_geom_id}")
    print(f"  IDs are distinct: {l_wheel_geom_id != r_wheel_geom_id}")

    # Get wheel body IDs
    l_wheel_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel")
    r_wheel_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel")

    print(f"\n[Geom-Body Assignment]")
    print(f"  l_wheel_geom belongs to body: {model.geom_bodyid[l_wheel_geom_id]} (expected: {l_wheel_body_id})")
    print(f"  r_wheel_geom belongs to body: {model.geom_bodyid[r_wheel_geom_id]} (expected: {r_wheel_body_id})")
    print(f"  Left assignment correct: {model.geom_bodyid[l_wheel_geom_id] == l_wheel_body_id}")
    print(f"  Right assignment correct: {model.geom_bodyid[r_wheel_geom_id] == r_wheel_body_id}")

    # Get contact points using ContactJacobian
    l_contact_world, r_contact_world = contact_jacobian.get_wheel_contact_points(data)

    print(f"\n[ContactJacobian.get_wheel_contact_points()]")
    print(f"  Left contact:  [{l_contact_world[0]:.6f}, {l_contact_world[1]:.6f}, {l_contact_world[2]:.6f}]")
    print(f"  Right contact: [{r_contact_world[0]:.6f}, {r_contact_world[1]:.6f}, {r_contact_world[2]:.6f}]")

    diff = l_contact_world - r_contact_world
    print(f"\n[Contact Point Difference]")
    print(f"  dx: {diff[0]:+.6f} m")
    print(f"  dy: {diff[1]:+.6f} m")
    print(f"  dz: {diff[2]:+.6f} m")

    if abs(diff[0]) < 0.01 and abs(diff[1]) < 0.01:
        print(f"\n  ERROR: Contact points are nearly identical!")
        print(f"  This destroys moment arms and explains the axis audit failure.")
        print(f"  FIX REQUIRED: ContactJacobian.get_wheel_contact_points() is broken.")


def task5_produce_decision(orientation, morphology, l_rel, r_rel):
    """Task 5: Produce a decision."""
    print("\n" + "=" * 80)
    print("TASK 5: Decision and Required Fixes")
    print("=" * 80)

    print(f"\n[Summary]")
    print(f"  1. Keyframe orientation: {orientation}")
    print(f"  2. Wheel morphology: {morphology}")
    print(f"  3. Left wheel rel to CoM:  [{l_rel[0]:+.6f}, {l_rel[1]:+.6f}, {l_rel[2]:+.6f}]")
    print(f"  4. Right wheel rel to CoM: [{r_rel[0]:+.6f}, {r_rel[1]:+.6f}, {r_rel[2]:+.6f}]")

    print(f"\n[Decision Tree]")

    if orientation != "UPRIGHT":
        print(f"\n  DECISION: Fix keyframe/root orientation FIRST")
        print(f"  - Robot is {orientation}, not UPRIGHT")
        print(f"  - All controller work is invalid until orientation is fixed")
        print(f"  - Action: Correct keyframe quaternion or root pose")
        print(f"  - Then: Rerun Stage 1/2 equilibrium reference capture")
        print(f"  - Do NOT modify force distributor yet")
        return

    if morphology == "IDENTICAL POSITIONS (BROKEN)":
        print(f"\n  DECISION: Fix wheel position extraction")
        print(f"  - Wheels have identical positions, destroying moment arms")
        print(f"  - Action: Fix ContactJacobian.get_wheel_contact_points()")
        print(f"  - Or: Fix XML geom/body assignment")
        print(f"  - Add: Regression test that left/right contact points are distinct")
        return

    if morphology == "FRONT/BACK SEPARATED":
        print(f"\n  DECISION: Update moment contract for front/back wheels")
        print(f"  - Wheels are separated along Y-axis (sagittal)")
        print(f"  - Vertical Fz asymmetry controls Mx (pitch), NOT My (roll)")
        print(f"  - Roll must be handled by hip-roll/internal actuation only")
        print(f"  - Action: Stop using Fz asymmetry for roll correction")
        print(f"  - Update: Force distributor moment contract")
        return

    if morphology == "LEFT/RIGHT SEPARATED":
        print(f"\n  DECISION: Fix delta mode formula to use correct axis")
        print(f"  - Wheels are separated along X-axis (lateral)")
        print(f"  - Vertical Fz asymmetry controls My (roll)")
        print(f"  - Current delta mode may use wrong coordinate")
        print(f"  - Action: Verify delta mode uses correct moment arm formula:")
        print(f"    My = y_l * fz_l + y_r * fz_r (for lateral separation)")
        print(f"  - Add: Tests comparing achieved wrench from build_wrench_matrix")
        return

    print(f"\n  DECISION: Complex morphology requires detailed analysis")
    print(f"  - Wheels have diagonal/complex separation")
    print(f"  - Manual analysis required to determine controllable moments")


def run_geometry_truth_audit():
    """Run complete geometry truth audit."""
    print("=" * 80)
    print("GEOMETRY TRUTH AUDIT")
    print("=" * 80)

    model, data = load_model()

    # Create state estimator
    state_estimator = CentroidalStateEstimator(
        CentroidalStateEstimatorConfig(
            robot_mass=8.1,
            torso_inertia=jnp.array([0.1, 0.1, 0.05]),
        ),
        mj_model=model,
    )

    # Task 1: Keyframe/root orientation
    orientation, up_alignment = task1_keyframe_root_orientation_audit(model, data, state_estimator)

    # Task 2: Wheel/body/geom truth
    l_rel, r_rel, diff = task2_wheel_body_geom_truth_audit(model, data)

    # Task 3: Determine wheel morphology
    morphology, moment_channel = task3_determine_wheel_morphology(diff)

    # Task 4: Check contact point extraction
    task4_check_contact_point_extraction(model, data)

    # Task 5: Produce decision
    task5_produce_decision(orientation, morphology, l_rel, r_rel)

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_geometry_truth_audit()
