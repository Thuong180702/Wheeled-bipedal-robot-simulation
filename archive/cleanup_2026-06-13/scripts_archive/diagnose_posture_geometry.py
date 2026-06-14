"""Diagnose current IK posture geometry.

Analyzes whether the current geometric IK produces knee-forward or knee-backward
postures, and measures CoM alignment with wheel contact points.
"""

import numpy as np
import mujoco
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from wheeled_biped.utils.config import get_model_path
from wheeled_biped.controllers.lqr_ik_prior import create_lqr_ik_prior


@dataclass
class PostureGeometry:
    """Geometry measurements for a posture."""
    height_cmd: float
    hip_pitch: float
    knee: float
    torso_pitch_deg: float
    wheel_contact_y: float
    l_hip_y: float
    l_knee_y: float
    l_wheel_y: float
    r_hip_y: float
    r_knee_y: float
    r_wheel_y: float
    com_y: float
    knee_forward_margin_l: float
    knee_forward_margin_r: float
    com_error_y: float


def get_body_com(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Compute whole-body center of mass position.

    Args:
        model: MuJoCo model
        data: MuJoCo data with current state

    Returns:
        CoM position [x, y, z] in world frame
    """
    # MuJoCo computes subtree CoM for each body
    # We need to compute the full system CoM
    total_mass = 0.0
    com_weighted = np.zeros(3)

    for i in range(model.nbody):
        body_mass = model.body_mass[i]
        if body_mass > 0:
            # xipos is the CoM position of body i in world frame
            body_com = data.xipos[i]
            com_weighted += body_mass * body_com
            total_mass += body_mass

    if total_mass > 0:
        return com_weighted / total_mass
    else:
        return np.zeros(3)


def diagnose_posture(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller,
    height_cmd: float,
) -> PostureGeometry:
    """Diagnose posture geometry for a given height.

    Args:
        model: MuJoCo model
        data: MuJoCo data (will be modified)
        controller: LQR/IK controller
        height_cmd: Target height in meters

    Returns:
        PostureGeometry with measurements
    """
    # Get IK solution
    hip_pitch_des, knee_des = controller.height_ik(height_cmd)

    # Reset and set posture
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0, 0, height_cmd]
    data.qpos[3:7] = [1, 0, 0, 0]  # upright quaternion
    data.qpos[7:17] = [
        0, 0, hip_pitch_des, knee_des, 0,  # left leg
        0, 0, hip_pitch_des, knee_des, 0,  # right leg
    ]

    # Forward kinematics
    mujoco.mj_forward(model, data)

    # Get body IDs for kinematic chain
    # We'll use body positions to approximate joint locations
    # The body frame origin is at the joint that connects it to its parent
    l_hip_yaw_body_id = model.body("l_hip_yaw_link").id
    l_thigh_body_id = model.body("l_thigh").id
    l_knee_link_body_id = model.body("l_knee_link").id
    l_wheel_body_id = model.body("l_wheel_link").id
    r_hip_yaw_body_id = model.body("r_hip_yaw_link").id
    r_thigh_body_id = model.body("r_thigh").id
    r_knee_link_body_id = model.body("r_knee_link").id
    r_wheel_body_id = model.body("r_wheel_link").id

    # Body frame origins (these are at the joints connecting to parent)
    # l_thigh body frame origin is at the hip_pitch joint
    # l_knee_link body frame origin is at the knee joint
    l_hip_yaw_xmat = data.xmat[l_hip_yaw_body_id].reshape(3, 3)
    l_thigh_xmat = data.xmat[l_thigh_body_id].reshape(3, 3)
    l_knee_link_xmat = data.xmat[l_knee_link_body_id].reshape(3, 3)

    # Body positions give us the body frame origin in world frame
    # For MuJoCo, xpos is the body CoM, not the frame origin
    # We need to use the body frame transformation
    l_hip_yaw_pos = data.xpos[l_hip_yaw_body_id]
    l_thigh_pos = data.xpos[l_thigh_body_id]
    l_knee_link_pos = data.xpos[l_knee_link_body_id]
    l_wheel_pos = data.xpos[l_wheel_body_id]

    r_hip_yaw_pos = data.xpos[r_hip_yaw_body_id]
    r_thigh_pos = data.xpos[r_thigh_body_id]
    r_knee_link_pos = data.xpos[r_knee_link_body_id]
    r_wheel_pos = data.xpos[r_wheel_body_id]

    # For knee-forward measurement, use body CoM positions as proxies
    # Hip position: hip_yaw_link body (top of leg)
    # Knee position: knee_link body (shin)
    l_hip_pos = l_hip_yaw_pos
    l_knee_pos = l_knee_link_pos
    r_hip_pos = r_hip_yaw_pos
    r_knee_pos = r_knee_link_pos

    # Compute whole-body CoM
    com_pos = get_body_com(model, data)

    # Wheel contact x (average of left and right wheels)
    # In MuJoCo default frame: X=lateral, Y=sagittal, Z=vertical
    wheel_contact_y = (l_wheel_pos[1] + r_wheel_pos[1]) / 2.0

    # Knee-forward margin (positive = knee is forward of hip in sagittal plane)
    # Measure in Y-axis (forward-backward), not X-axis (left-right)
    knee_forward_margin_l = l_knee_pos[1] - l_hip_pos[1]
    knee_forward_margin_r = r_knee_pos[1] - r_hip_pos[1]

    # CoM error (positive = CoM is forward of wheel contact in sagittal plane)
    com_error_y = com_pos[1] - wheel_contact_y

    # Torso pitch from quaternion
    quat = data.qpos[3:7]
    # Convert quaternion to pitch angle
    # For small angles: pitch ≈ -2 * (qw * qy + qx * qz)
    # More accurate: use rotation matrix
    rot_mat = np.zeros(9)
    mujoco.mju_quat2Mat(rot_mat, quat)
    rot_mat = rot_mat.reshape(3, 3)
    # Pitch is rotation around y-axis
    # pitch = atan2(-R[2,0], sqrt(R[0,0]^2 + R[1,0]^2))
    torso_pitch_rad = np.arctan2(-rot_mat[2, 0], np.sqrt(rot_mat[0, 0]**2 + rot_mat[1, 0]**2))
    torso_pitch_deg = np.degrees(torso_pitch_rad)

    return PostureGeometry(
        height_cmd=height_cmd,
        hip_pitch=hip_pitch_des,
        knee=knee_des,
        torso_pitch_deg=torso_pitch_deg,
        wheel_contact_y=wheel_contact_y,
        l_hip_y=l_hip_pos[1],
        l_knee_y=l_knee_pos[1],
        l_wheel_y=l_wheel_pos[1],
        r_hip_y=r_hip_pos[1],
        r_knee_y=r_knee_pos[1],
        r_wheel_y=r_wheel_pos[1],
        com_y=com_pos[1],
        knee_forward_margin_l=knee_forward_margin_l,
        knee_forward_margin_r=knee_forward_margin_r,
        com_error_y=com_error_y,
    )


def render_side_view(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    height_cmd: float,
    output_path: Path,
) -> None:
    """Render side view of posture.

    Args:
        model: MuJoCo model
        data: MuJoCo data with current state
        height_cmd: Height command for filename
        output_path: Output directory
    """
    try:
        # Create renderer
        renderer = mujoco.Renderer(model, height=480, width=640)

        # Set camera to side view
        renderer.update_scene(data, camera="side_view")

        # Render
        pixels = renderer.render()

        # Save image
        output_path.mkdir(parents=True, exist_ok=True)
        filename = output_path / f"h_{height_cmd:.2f}.png"

        plt.figure(figsize=(8, 6))
        plt.imshow(pixels)
        plt.axis('off')
        plt.title(f"Height {height_cmd:.2f}m - Side View")
        plt.tight_layout()
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"  Saved side view: {filename}")

    except Exception as e:
        print(f"  Warning: Could not render side view: {e}")


def main():
    print("=" * 80)
    print("Posture Geometry Diagnostic")
    print("=" * 80)

    # Load model
    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Create controller
    config_path = Path(__file__).parent.parent / "configs" / "controllers" / "gain_scheduled_lqr.yaml"
    controller = create_lqr_ik_prior(config_path, model)

    # Test heights
    height_grid = [0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40]

    # Output directory
    output_dir = Path(__file__).parent.parent / "outputs" / "diagnostics" / "posture_geometry"

    print("\nAnalyzing posture geometry across height range:")
    print("-" * 80)

    results = []
    for height_cmd in height_grid:
        print(f"\nHeight {height_cmd:.2f}m:")

        geom = diagnose_posture(model, data, controller, height_cmd)
        results.append(geom)

        print(f"  Joint angles:")
        print(f"    hip_pitch: {geom.hip_pitch:.4f} rad ({np.degrees(geom.hip_pitch):.2f}°)")
        print(f"    knee:      {geom.knee:.4f} rad ({np.degrees(geom.knee):.2f}°)")
        print(f"  Torso pitch: {geom.torso_pitch_deg:.2f}°")
        print(f"  Y-positions (sagittal):")
        print(f"    wheel_contact: {geom.wheel_contact_y:.4f} m")
        print(f"    l_hip:         {geom.l_hip_y:.4f} m")
        print(f"    l_knee:        {geom.l_knee_y:.4f} m")
        print(f"    r_hip:         {geom.r_hip_y:.4f} m")
        print(f"    r_knee:        {geom.r_knee_y:.4f} m")
        print(f"    whole_body_CoM: {geom.com_y:.4f} m")
        print(f"  Knee-forward margin:")
        print(f"    left:  {geom.knee_forward_margin_l:.4f} m")
        print(f"    right: {geom.knee_forward_margin_r:.4f} m")
        print(f"  CoM error: {geom.com_error_y:.4f} m (positive = CoM forward of wheels)")

        # Render side view
        render_side_view(model, data, height_cmd, output_dir)

    # Summary analysis
    print("\n" + "=" * 80)
    print("Summary Analysis")
    print("=" * 80)

    # Check knee-forward direction
    knee_forward_count = sum(1 for r in results if r.knee_forward_margin_l > 0)
    knee_backward_count = len(results) - knee_forward_count

    print(f"\nKnee direction:")
    print(f"  Forward (human-like):  {knee_forward_count}/{len(results)} heights")
    print(f"  Backward (non-human):  {knee_backward_count}/{len(results)} heights")

    if knee_forward_count == len(results):
        print("  [OK] All postures have knee-forward geometry")
    elif knee_backward_count == len(results):
        print("  [FAIL] All postures have knee-backward geometry (non-human-like)")
    else:
        print("  [WARNING] Mixed knee directions across height range")

    # CoM alignment
    avg_com_error = np.mean([abs(r.com_error_y) for r in results])
    max_com_error = max([abs(r.com_error_y) for r in results])

    print(f"\nCoM alignment:")
    print(f"  Average |CoM error|: {avg_com_error:.4f} m")
    print(f"  Maximum |CoM error|: {max_com_error:.4f} m")

    if max_com_error < 0.01:
        print("  [OK] CoM well-aligned with wheel contact")
    elif max_com_error < 0.05:
        print("  [WARNING] CoM moderately misaligned")
    else:
        print("  [FAIL] CoM significantly misaligned (explains balance failure)")

    # Detailed table
    print("\n" + "=" * 80)
    print("Detailed Measurements")
    print("=" * 80)
    print(f"{'Height':>7} | {'hip_pitch':>10} | {'knee':>10} | {'knee_fwd_L':>11} | "
          f"{'knee_fwd_R':>11} | {'CoM_err':>9}")
    print("-" * 80)

    for r in results:
        print(f"{r.height_cmd:7.2f} | {r.hip_pitch:10.4f} | {r.knee:10.4f} | "
              f"{r.knee_forward_margin_l:11.4f} | {r.knee_forward_margin_r:11.4f} | "
              f"{r.com_error_y:9.4f}")

    print("\n" + "=" * 80)
    print("Conclusion")
    print("=" * 80)

    if knee_backward_count > 0:
        print("\nThe current geometric IK produces knee-backward postures for some heights.")
        print("This is not human-like and may contribute to balance instability.")
        print("Recommendation: Implement knee-forward constraint in balanced posture generator.")

    if max_com_error > 0.02:
        print(f"\nThe CoM is misaligned with wheel contact by up to {max_com_error:.4f}m.")
        print("This explains why the robot immediately pitches when initialized at IK posture.")
        print("Recommendation: Implement CoM-aware posture optimization.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
