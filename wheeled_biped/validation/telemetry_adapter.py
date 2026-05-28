# wheeled_biped/validation/telemetry_adapter.py
"""Telemetry adapter for balance-core validation compatibility.

Adds canonical field names and validation-specific fields to simulation telemetry
without modifying controller behavior.
"""

import numpy as np


def add_validation_telemetry_fields(
    telemetry: dict,
    control_dt: float,
    csv_path: str,
    survival_steps_override: int | None = None,
) -> None:
    """Add canonical validation fields to telemetry dict in-place.

    Args:
        telemetry: Telemetry dict with simulation data
        control_dt: Control timestep in seconds
        csv_path: Path where telemetry CSV will be saved
        survival_steps_override: Authoritative simulated-step count when telemetry rows
            are decimated and no longer match the number of written rows
    """
    n_steps = len(telemetry["time"])
    survival_steps = survival_steps_override if survival_steps_override is not None else n_steps

    # 1. Metadata fields
    if "source_step_index" in telemetry:
        telemetry["step"] = telemetry["source_step_index"].copy()
    else:
        telemetry["step"] = list(range(n_steps))
    telemetry["sim_time_s"] = telemetry["time"].copy()
    telemetry["control_dt_s"] = [control_dt] * n_steps
    telemetry["controller_mode"] = telemetry["control_mode"].copy()
    telemetry["survival_steps"] = [survival_steps] * n_steps
    telemetry["telemetry_file_path"] = [str(csv_path)] * n_steps

    # 2. Posture aliases
    telemetry["joint_positions"] = telemetry["joint_pos"].copy()
    telemetry["joint_velocities"] = telemetry["joint_vel"].copy()

    # Compute joint error fields from existing data
    joint_error_per_joint = []
    support_joint_error_norm = []
    knee_error_left_rad = []
    knee_error_right_rad = []
    hip_pitch_error_left_rad = []
    hip_pitch_error_right_rad = []

    for i in range(n_steps):
        # Parse joint_pos_error CSV format
        error_str = telemetry["joint_pos_error"][i]
        errors = [float(x) for x in error_str.split(",")]

        joint_error_per_joint.append(",".join(f"{x:.6f}" for x in errors))

        # Support joints: hip_pitch (2,7) and knee (3,8)
        support_errors = [errors[2], errors[3], errors[7], errors[8]]
        support_joint_error_norm.append(float(np.linalg.norm(support_errors)))

        knee_error_left_rad.append(errors[3])
        knee_error_right_rad.append(errors[8])
        hip_pitch_error_left_rad.append(errors[2])
        hip_pitch_error_right_rad.append(errors[7])

    telemetry["joint_error_per_joint"] = joint_error_per_joint
    telemetry["support_joint_error_norm"] = support_joint_error_norm
    telemetry["knee_error_left_rad"] = knee_error_left_rad
    telemetry["knee_error_right_rad"] = knee_error_right_rad
    telemetry["hip_pitch_error_left_rad"] = hip_pitch_error_left_rad
    telemetry["hip_pitch_error_right_rad"] = hip_pitch_error_right_rad

    # 3. Actuator control field (actual MuJoCo actuator commands)
    # In balance-core mode, this is tau_final_per_joint
    # In legacy mode, this is tau_smooth_per_joint
    if "tau_final_per_joint" in telemetry:
        telemetry["actuator_ctrl_per_joint"] = telemetry["tau_final_per_joint"].copy()
    else:
        telemetry["actuator_ctrl_per_joint"] = telemetry["tau_smooth_per_joint"].copy()

    # 4. Hidden/legacy torque validation fields (should be zero in balance-core mode)
    # These track legacy controllers that should not be active in balance-core
    tau_legacy_wheel_balance_norm = []
    tau_legacy_hip_roll_centering_norm = []
    tau_posture_regularizer_norm = []
    tau_leg_position_norm = []
    hidden_torque_norm = []

    for i in range(n_steps):
        # Parse legacy torque vectors
        wheel_balance_str = telemetry["tau_wheel_balance_per_joint"][i]
        wheel_balance = [float(x) for x in wheel_balance_str.split(",")]
        tau_legacy_wheel_balance_norm.append(float(np.linalg.norm(wheel_balance)))

        hip_roll_str = telemetry["tau_hip_roll_centering_per_joint"][i]
        hip_roll = [float(x) for x in hip_roll_str.split(",")]
        tau_legacy_hip_roll_centering_norm.append(float(np.linalg.norm(hip_roll)))

        posture_str = telemetry["tau_posture_per_joint"][i]
        posture = [float(x) for x in posture_str.split(",")]
        tau_posture_regularizer_norm.append(float(np.linalg.norm(posture)))

        leg_pos_str = telemetry["tau_leg_position_per_joint"][i]
        leg_pos = [float(x) for x in leg_pos_str.split(",")]
        tau_leg_position_norm.append(float(np.linalg.norm(leg_pos)))

        # Hidden torque = sum of all legacy sources
        hidden = float(np.linalg.norm(wheel_balance)) + float(np.linalg.norm(hip_roll)) + \
                 float(np.linalg.norm(posture)) + float(np.linalg.norm(leg_pos))
        hidden_torque_norm.append(hidden)

    telemetry["tau_legacy_wheel_balance_norm"] = tau_legacy_wheel_balance_norm
    telemetry["tau_legacy_hip_roll_centering_norm"] = tau_legacy_hip_roll_centering_norm
    telemetry["tau_posture_regularizer_norm"] = tau_posture_regularizer_norm
    telemetry["tau_leg_position_norm"] = tau_leg_position_norm
    telemetry["hidden_torque_norm"] = hidden_torque_norm


def normalize_balance_core_owner_names(telemetry: dict) -> None:
    """Normalize active_torque_owner_per_joint to canonical names.

    Removes 'tau_' prefixes and ensures canonical owner names:
    - lateral_roll_balance
    - shape_posture
    - support_feedforward
    - shape_posture+support_feedforward
    - sagittal_wheel_balance
    - none

    Args:
        telemetry: Telemetry dict with active_torque_owner_per_joint field
    """
    if "active_torque_owner_per_joint" not in telemetry:
        return

    normalized = []
    for owner_str in telemetry["active_torque_owner_per_joint"]:
        # Parse CSV format: "owner1,owner2,owner3,..."
        owners = [s.strip() for s in owner_str.split(",")]

        # Normalize each owner name
        normalized_owners = []
        for owner in owners:
            # Remove 'tau_' prefix if present
            if owner.startswith("tau_"):
                owner = owner[4:]

            # Map to canonical names
            if owner == "shape_posture":
                normalized_owners.append("shape_posture")
            elif owner == "support_feedforward":
                normalized_owners.append("support_feedforward")
            elif owner == "sagittal_wheel_balance":
                normalized_owners.append("sagittal_wheel_balance")
            elif owner == "lateral_roll_balance":
                normalized_owners.append("lateral_roll_balance")
            elif owner == "none":
                normalized_owners.append("none")
            elif "+" in owner:
                # Handle composite owners like "shape_posture+support_feedforward"
                parts = [p.strip() for p in owner.split("+")]
                normalized_parts = []
                for part in parts:
                    if part.startswith("tau_"):
                        part = part[4:]
                    normalized_parts.append(part)
                normalized_owners.append("+".join(normalized_parts))
            else:
                # Unknown owner - keep as-is but remove tau_ prefix
                normalized_owners.append(owner)

        normalized.append(",".join(normalized_owners))

    telemetry["active_torque_owner_per_joint"] = normalized
