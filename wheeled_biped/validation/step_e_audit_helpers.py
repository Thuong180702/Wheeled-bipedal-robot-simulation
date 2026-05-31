from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class StepEAuditClassification:
    primary_classification: str
    support_position_velocity_mean: float
    support_position_error_mean: float
    tau_balance_before_position_mean: float
    tau_position_clipped_mean: float
    net_balance_mean: float
    physical_motor_limit: bool
    sign_error: bool
    continuous_position_drift: bool
    WBC_active: bool
    E0_logic_active: bool


def classify_steady_state_balance_torque_bias(
    df: pd.DataFrame,
    *,
    velocity_epsilon: float = 1e-3,
    net_torque_epsilon: float = 1e-3,
    support_error_epsilon: float = 1e-3,
    hidden_torque_epsilon: float = 1e-9,
) -> StepEAuditClassification:
    required = {
        "support_position_velocity_m_s",
        "support_position_error_m",
        "tau_balance_before_position",
        "tau_position_clipped",
        "hidden_torque_norm",
        "ownership_violation_count",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    support_position_velocity_mean = float(df["support_position_velocity_m_s"].mean())
    support_position_error_mean = float(df["support_position_error_m"].mean())
    tau_balance_before_position_mean = float(df["tau_balance_before_position"].mean())
    tau_position_clipped_mean = float(df["tau_position_clipped"].mean())
    net_balance_mean = tau_balance_before_position_mean + tau_position_clipped_mean

    physical_motor_limit = False
    if "wheel_torque_saturation_left" in df.columns and "wheel_torque_saturation_right" in df.columns:
        physical_motor_limit = bool(df["wheel_torque_saturation_left"].any() or df["wheel_torque_saturation_right"].any())

    sign_error = not (
        tau_balance_before_position_mean > 0.0 and tau_position_clipped_mean < 0.0
    )
    continuous_position_drift = abs(support_position_velocity_mean) > velocity_epsilon
    WBC_active = bool((df["hidden_torque_norm"] > hidden_torque_epsilon).any())
    E0_logic_active = bool((df["ownership_violation_count"] > 0).any())

    if (
        abs(support_position_velocity_mean) <= velocity_epsilon
        and support_position_error_mean > support_error_epsilon
        and tau_balance_before_position_mean > 0.0
        and tau_position_clipped_mean < 0.0
        and abs(net_balance_mean) <= net_torque_epsilon
        and not physical_motor_limit
        and not sign_error
        and not continuous_position_drift
        and not WBC_active
        and not E0_logic_active
    ):
        primary = "steady_state_balance_torque_bias"
    else:
        primary = "not_steady_state_balance_torque_bias"

    return StepEAuditClassification(
        primary_classification=primary,
        support_position_velocity_mean=support_position_velocity_mean,
        support_position_error_mean=support_position_error_mean,
        tau_balance_before_position_mean=tau_balance_before_position_mean,
        tau_position_clipped_mean=tau_position_clipped_mean,
        net_balance_mean=net_balance_mean,
        physical_motor_limit=physical_motor_limit,
        sign_error=sign_error,
        continuous_position_drift=continuous_position_drift,
        WBC_active=WBC_active,
        E0_logic_active=E0_logic_active,
    )
