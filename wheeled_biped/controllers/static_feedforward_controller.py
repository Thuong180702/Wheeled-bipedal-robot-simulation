"""Stage 2B: Static Feedforward Controller for gravity compensation.

Applies empirical feedforward torques to selected joint groups to compensate
for gravity at h=0.404m equilibrium. Validated configuration from Phase C:
- Sign: +empirical
- Scale: 0.5
- Joint group: knee [3, 8]
- Ramp: instant
- Effective feedforward: -7.75, -7.90 Nm
"""

import numpy as np
from typing import Literal, Optional


# Joint indices
KNEE_INDICES = [3, 8]
HIP_PITCH_INDICES = [2, 7]
HIP_PITCH_KNEE_INDICES = [2, 3, 7, 8]

JOINT_GROUP_MAP = {
    "knee": KNEE_INDICES,
    "hip_pitch": HIP_PITCH_INDICES,
    "hip_pitch_knee": HIP_PITCH_KNEE_INDICES,
}


class StaticFeedforwardController:
    """Applies empirical feedforward torques for gravity compensation.

    Validated Phase C configuration:
    - sign: positive (+empirical)
    - scale: 0.5
    - joint_group: knee
    - ramp: instant
    """

    def __init__(
        self,
        empirical_feedforward: np.ndarray,
        scale: float = 0.5,
        joint_group: Literal["knee", "hip_pitch", "hip_pitch_knee"] = "knee",
        ramp_mode: Literal["instant", "short", "medium"] = "instant",
        sign: Literal["positive", "negative"] = "positive",
    ):
        """Initialize StaticFeedforwardController.

        Args:
            empirical_feedforward: 10-element empirical feedforward torque vector (Nm)
            scale: Feedforward scale factor (default: 0.5, validated)
            joint_group: Joint group to apply feedforward (default: "knee", validated)
            ramp_mode: Ramp mode for feedforward application (default: "instant", validated)
            sign: Sign of feedforward (default: "positive", validated)
        """
        if empirical_feedforward.shape != (10,):
            raise ValueError(f"empirical_feedforward must be shape (10,), got {empirical_feedforward.shape}")

        if joint_group not in JOINT_GROUP_MAP:
            raise ValueError(f"joint_group must be one of {list(JOINT_GROUP_MAP.keys())}, got {joint_group}")

        if ramp_mode not in ["instant", "short", "medium"]:
            raise ValueError(f"ramp_mode must be one of ['instant', 'short', 'medium'], got {ramp_mode}")

        if sign not in ["positive", "negative"]:
            raise ValueError(f"sign must be one of ['positive', 'negative'], got {sign}")

        self.empirical_feedforward = empirical_feedforward.copy()
        self.scale = scale
        self.joint_group = joint_group
        self.joint_indices = JOINT_GROUP_MAP[joint_group]
        self.ramp_mode = ramp_mode
        self.sign = sign

        # Compute base feedforward (scaled and signed)
        sign_multiplier = 1.0 if sign == "positive" else -1.0
        self.base_feedforward = sign_multiplier * scale * empirical_feedforward

        # Ramp parameters
        if ramp_mode == "instant":
            self.ramp_steps = 0
        elif ramp_mode == "short":
            self.ramp_steps = 5
        elif ramp_mode == "medium":
            self.ramp_steps = 10

        self.current_step = 0

    def compute_feedforward(self, step: Optional[int] = None) -> np.ndarray:
        """Compute feedforward torque for current step.

        Args:
            step: Optional explicit step number (if None, uses internal counter)

        Returns:
            10-element feedforward torque vector (Nm)
        """
        if step is not None:
            current_step = step
        else:
            current_step = self.current_step
            self.current_step += 1

        # Compute ramp factor
        if self.ramp_mode == "instant" or current_step >= self.ramp_steps:
            ramp_factor = 1.0
        else:
            ramp_factor = current_step / self.ramp_steps

        # Apply feedforward only to selected joint group
        tau_feedforward = np.zeros(10)
        tau_feedforward[self.joint_indices] = ramp_factor * self.base_feedforward[self.joint_indices]

        return tau_feedforward

    def get_telemetry(self) -> dict:
        """Get telemetry data for logging.

        Returns:
            Dictionary with telemetry fields
        """
        tau_ff = self.compute_feedforward(step=self.current_step - 1)

        return {
            "tau_feedforward_per_joint": tau_ff.tolist(),
            "tau_feedforward_norm": float(np.linalg.norm(tau_ff)),
            "feedforward_ramp": self.ramp_mode,
            "feedforward_joint_group": self.joint_group,
            "feedforward_scale": self.scale,
            "feedforward_sign": self.sign,
            "feedforward_step": self.current_step - 1,
            "feedforward_ramp_factor": 1.0 if self.current_step - 1 >= self.ramp_steps else (self.current_step - 1) / max(1, self.ramp_steps),
        }

    def reset(self):
        """Reset internal step counter."""
        self.current_step = 0


def load_empirical_feedforward_from_telemetry(telemetry_path: str) -> np.ndarray:
    """Load empirical feedforward from gain sweep telemetry CSV.

    Args:
        telemetry_path: Path to telemetry CSV file

    Returns:
        10-element empirical feedforward torque vector (Nm)
    """
    import csv

    with open(telemetry_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) < 20:
        raise ValueError(f"Insufficient telemetry data: only {len(rows)} rows")

    # Extract tau_posture_per_joint from steps 5-20
    stable_start = min(5, len(rows) - 1)
    stable_end = min(20, len(rows))

    if "tau_posture_per_joint" not in rows[0]:
        raise ValueError("Column 'tau_posture_per_joint' not found in telemetry CSV")

    tau_samples = []
    for i in range(stable_start, stable_end):
        if rows[i]["tau_posture_per_joint"]:
            tau = [float(x) for x in rows[i]["tau_posture_per_joint"].split(",")]
            tau_samples.append(tau)

    if not tau_samples:
        raise ValueError("No valid tau_posture_per_joint data found in telemetry")

    tau_array = np.array(tau_samples)
    tau_median = np.median(tau_array, axis=0)

    return tau_median
