"""Direct test of height IK mapping at h=0.65m."""

import numpy as np
import mujoco
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from wheeled_biped.utils.config import get_model_path
from wheeled_biped.controllers.lqr_ik_prior import create_lqr_ik_prior


def main():
    # Load model
    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # Create controller
    config_path = Path(__file__).parent.parent / "configs" / "controllers" / "gain_scheduled_lqr.yaml"
    controller = create_lqr_ik_prior(config_path, model)

    # Test height IK at 0.65m
    height_cmd = 0.65
    hip_pitch_des, knee_des = controller.height_ik(height_cmd)

    print(f"Height IK test for h={height_cmd}m:")
    print(f"  hip_pitch_des = {hip_pitch_des:.4f} rad")
    print(f"  knee_des = {knee_des:.4f} rad")

    # Normalize these values
    hip_pitch_limits = controller.joint_limits["hip_pitch"]
    knee_limits = controller.joint_limits["knee"]

    hip_pitch_mid = (hip_pitch_limits[0] + hip_pitch_limits[1]) / 2.0
    hip_pitch_half_range = (hip_pitch_limits[1] - hip_pitch_limits[0]) / 2.0
    hip_pitch_norm = (hip_pitch_des - hip_pitch_mid) / hip_pitch_half_range

    knee_mid = (knee_limits[0] + knee_limits[1]) / 2.0
    knee_half_range = (knee_limits[1] - knee_limits[0]) / 2.0
    knee_norm = (knee_des - knee_mid) / knee_half_range

    print(f"\nNormalized values:")
    print(f"  hip_pitch_norm = {hip_pitch_norm:.4f}")
    print(f"  knee_norm = {knee_norm:.4f}")

    print(f"\nHeight IK mapping info:")
    print(f"  Height range: [{controller.height_ik.height_range[0]:.3f}, {controller.height_ik.height_range[1]:.3f}] m")
    print(f"  Hip pitch poly: {controller.height_ik.hip_pitch_poly}")
    print(f"  Knee poly: {controller.height_ik.knee_poly}")


if __name__ == "__main__":
    main()
