# wheeled_biped/validation/telemetry_schema_checker.py
"""Telemetry schema validation for balance-core controller."""

from typing import List
import pandas as pd


class MissingFieldError(Exception):
    """Raised when required telemetry fields are missing."""
    pass


class TelemetrySchemaChecker:
    """Validates that all required telemetry fields exist."""

    REQUIRED_METADATA_FIELDS = [
        "control_mode",
        "step",
        "time",
    ]

    REQUIRED_STATE_FIELDS = [
        "pitch_x_rad",
        "roll_y_rad",
        "yaw_z_rad",
        "pitch_rate_rad_s",
        "roll_rate_rad_s",
        "yaw_rate_rad_s",
        "com_x_m",
        "com_y_m",
        "com_z_m",
    ]

    REQUIRED_POSTURE_FIELDS = [
        "joint_positions",
        "joint_velocities",
    ]

    REQUIRED_CONTACT_FIELDS = [
        "contact_supervisor_state",
        "contact_duration_s",
    ]

    REQUIRED_TORQUE_FIELDS = [
        "tau_shape_posture_per_joint",
        "tau_support_feedforward_per_joint",
        "tau_sagittal_wheel_balance_per_joint",
        "tau_lateral_roll_balance_per_joint",
        "tau_total_raw_per_joint",
        "tau_total_clipped_per_joint",
        "tau_final_per_joint",
        "active_torque_owner_per_joint",
        "ownership_violation_count",
    ]

    REQUIRED_ACTUATOR_FIELDS = [
        "actuator_ctrl_per_joint",
    ]

    REQUIRED_SAFETY_FIELDS = [
        "torque_saturation_mask_per_joint",
        "torque_rate_saturation_mask_per_joint",
    ]

    REQUIRED_HIDDEN_TORQUE_FIELDS = [
        "hidden_torque_norm",
    ]

    def validate(self, df: pd.DataFrame) -> None:
        """Validate telemetry schema.

        Args:
            df: Telemetry dataframe

        Raises:
            MissingFieldError: If any required field is missing
        """
        missing_fields = []

        all_required = (
            self.REQUIRED_METADATA_FIELDS
            + self.REQUIRED_STATE_FIELDS
            + self.REQUIRED_POSTURE_FIELDS
            + self.REQUIRED_CONTACT_FIELDS
            + self.REQUIRED_TORQUE_FIELDS
            + self.REQUIRED_ACTUATOR_FIELDS
            + self.REQUIRED_SAFETY_FIELDS
            + self.REQUIRED_HIDDEN_TORQUE_FIELDS
        )

        for field in all_required:
            if field not in df.columns:
                missing_fields.append(field)

        if missing_fields:
            raise MissingFieldError(
                f"Missing required telemetry fields: {', '.join(missing_fields)}"
            )
