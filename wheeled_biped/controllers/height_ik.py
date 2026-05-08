"""Height inverse kinematics for wheeled biped.

Provides contact-aware height IK mapping from torso height to joint angles.
Uses FK scan + polynomial fit approach.
"""

from pathlib import Path
from typing import Tuple

import mujoco
import numpy as np


class HeightIK:
    """Height IK mapper using FK scan and polynomial fit."""

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        scan_points: int = 25,
        polynomial_degree: int = 2,
        symmetric_fold: bool = True,
    ):
        """Initialize height IK mapper.

        Args:
            mj_model: MuJoCo model for FK scan.
            scan_points: Number of samples for FK scan.
            polynomial_degree: Polynomial degree for fitting.
            symmetric_fold: If True, use knee = 2 * hip_pitch constraint.
        """
        self.model = mj_model
        self.scan_points = scan_points
        self.polynomial_degree = polynomial_degree
        self.symmetric_fold = symmetric_fold

        # Build mapping
        self.height_range, self.hip_pitch_poly, self.knee_poly = self._build_mapping()

    def _build_mapping(self) -> Tuple[Tuple[float, float], np.ndarray, np.ndarray]:
        """Build height IK mapping from FK scan.

        Returns:
            (height_range, hip_pitch_poly, knee_poly)
        """
        # Try to load empirical targets first
        empirical_path = Path("outputs/phase_b9_task6_empirical_ik_standing_only/empirical_ik_targets.json")

        if empirical_path.exists():
            import json
            with open(empirical_path, 'r') as f:
                empirical_data = json.load(f)

            targets = [t for t in empirical_data['targets'] if t['achievable']]

            if targets:
                targets.sort(key=lambda t: t['target_height'])

                heights = np.array([t['target_height'] for t in targets])
                hip_pitches = np.array([t['hip_pitch'] for t in targets])
                knees = np.array([t['knee'] for t in targets])

                hip_pitch_poly = np.polyfit(heights, hip_pitches, self.polynomial_degree)
                knee_poly = np.polyfit(heights, knees, self.polynomial_degree)

                return (float(heights.min()), float(heights.max())), hip_pitch_poly, knee_poly

        # Fallback: FK scan
        hip_pitch_min = 0.0
        hip_pitch_max = 1.5
        knee_min = 0.0
        knee_max = 2.5

        hip_pitch_samples = np.linspace(hip_pitch_min, hip_pitch_max, self.scan_points)
        knee_samples = np.linspace(knee_min, knee_max, self.scan_points)

        if self.symmetric_fold:
            knee_samples = 2.0 * hip_pitch_samples
            knee_samples = np.clip(knee_samples, knee_min, knee_max)

        heights = []
        data = mujoco.MjData(self.model)

        L_HIP_PITCH_QPOS = 7 + 2
        L_KNEE_QPOS = 7 + 3
        R_HIP_PITCH_QPOS = 7 + 7
        R_KNEE_QPOS = 7 + 8

        for hip_pitch, knee in zip(hip_pitch_samples, knee_samples):
            mujoco.mj_resetData(self.model, data)

            data.qpos[L_HIP_PITCH_QPOS] = hip_pitch
            data.qpos[L_KNEE_QPOS] = knee
            data.qpos[R_HIP_PITCH_QPOS] = hip_pitch
            data.qpos[R_KNEE_QPOS] = knee
            data.qpos[2] = 1.0

            mujoco.mj_forward(self.model, data)

            l_wheel_body_id = self.model.body("l_wheel_link").id
            r_wheel_body_id = self.model.body("r_wheel_link").id
            l_wheel_z = data.xpos[l_wheel_body_id, 2]
            r_wheel_z = data.xpos[r_wheel_body_id, 2]
            lowest_wheel_z = min(l_wheel_z, r_wheel_z)

            root_z_correction = -lowest_wheel_z
            data.qpos[2] += root_z_correction

            mujoco.mj_forward(self.model, data)

            torso_height = data.qpos[2]
            heights.append(torso_height)

        heights = np.array(heights)

        hip_pitch_poly = np.polyfit(heights, hip_pitch_samples, self.polynomial_degree)
        knee_poly = np.polyfit(heights, knee_samples, self.polynomial_degree)

        return (float(heights.min()), float(heights.max())), hip_pitch_poly, knee_poly

    def compute_ik_targets(self, height_cmd: float) -> dict[str, float]:
        """Compute joint targets for desired height.

        Args:
            height_cmd: Desired torso height [m].

        Returns:
            Dict with 'hip_pitch' and 'knee' in radians.
        """
        h_clipped = np.clip(height_cmd, self.height_range[0], self.height_range[1])
        hip_pitch = float(np.polyval(self.hip_pitch_poly, h_clipped))
        knee = float(np.polyval(self.knee_poly, h_clipped))

        return {
            "hip_pitch": hip_pitch,
            "knee": knee,
        }
