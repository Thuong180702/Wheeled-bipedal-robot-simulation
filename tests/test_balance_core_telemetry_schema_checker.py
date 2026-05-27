# tests/test_balance_core_telemetry_schema_checker.py
import pandas as pd
import pytest
from wheeled_biped.validation.telemetry_schema_checker import (
    TelemetrySchemaChecker,
    MissingFieldError,
)


def test_missing_metadata_fields_raises_error():
    """Missing controller_mode should raise MissingFieldError."""
    df = pd.DataFrame({
        "step": [0, 1, 2],
        "time": [0.0, 0.002, 0.004],
        # Missing controller_mode
    })

    checker = TelemetrySchemaChecker()
    with pytest.raises(MissingFieldError, match="controller_mode"):
        checker.validate(df)


def test_complete_schema_passes():
    """Complete telemetry schema should pass validation."""
    df = pd.DataFrame({
        # Metadata
        "controller_mode": ["balance-core"] * 3,
        "step": [0, 1, 2],
        "time": [0.0, 0.002, 0.004],
        # State
        "pitch_x_rad": [0.0, 0.01, 0.02],
        "roll_y_rad": [0.0, 0.0, 0.0],
        "yaw_z_rad": [0.0, 0.0, 0.0],
        "pitch_rate_rad_s": [0.0, 0.5, 1.0],
        "roll_rate_rad_s": [0.0, 0.0, 0.0],
        "yaw_rate_rad_s": [0.0, 0.0, 0.0],
        "com_x_m": [0.0, 0.0, 0.0],
        "com_y_m": [0.0, 0.0, 0.0],
        "com_z_m": [0.45, 0.45, 0.45],
        # Posture
        "joint_positions": ["[0.0]*10"] * 3,
        "joint_velocities": ["[0.0]*10"] * 3,
        # Contact
        "contact_supervisor_state": ["DOUBLE_CONTACT"] * 3,
        "contact_duration_s": [0.0, 0.002, 0.004],
        # Torque
        "tau_shape_posture_per_joint": ["[0.0]*10"] * 3,
        "tau_support_feedforward_per_joint": ["[0.0]*10"] * 3,
        "tau_sagittal_wheel_balance_per_joint": ["[0.0]*10"] * 3,
        "tau_lateral_roll_balance_per_joint": ["[0.0]*10"] * 3,
        "tau_total_raw_per_joint": ["[0.0]*10"] * 3,
        "tau_total_clipped_per_joint": ["[0.0]*10"] * 3,
        "tau_final_per_joint": ["[0.0]*10"] * 3,
        "active_torque_owner_per_joint": ["['shape_posture']*10"] * 3,
        "ownership_violation_count": [0, 0, 0],
        # Actuator
        "actuator_ctrl_per_joint": ["[0.0]*10"] * 3,
        # Safety
        "torque_saturation_mask_per_joint": ["[False]*10"] * 3,
        "torque_rate_saturation_mask_per_joint": ["[False]*10"] * 3,
        # Hidden
        "hidden_torque_norm": [0.0, 0.0, 0.0],
    })

    checker = TelemetrySchemaChecker()
    checker.validate(df)  # Should not raise
