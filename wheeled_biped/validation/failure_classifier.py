# wheeled_biped/validation/failure_classifier.py
"""Temporal root-cause failure classification for balance-core."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np


class FailureMode(Enum):
    """Balance-core failure modes."""
    # Priority 0: Architecture Regression
    HIDDEN_LEGACY_TORQUE = "F0.1"
    OWNERSHIP_VIOLATION = "F0.2"
    NON_FINITE_TORQUE = "F0.3"
    WBC_ACTIVE = "F0.4"
    FAKE_CONTACT_FORCE = "F0.5"
    INVALID_TORQUE_OWNER = "F0.6"

    # Priority 1: Support and Contact
    KNEE_SUPPORT_COLLAPSE = "F1.1"
    HEIGHT_COLLAPSE = "F1.2"
    CONTACT_LOSS = "F1.3"

    # Priority 2: Primary Balance Axes
    PITCH_DIVERGENCE = "F2.1"
    ROLL_DIVERGENCE = "F2.2"

    # Priority 3: Dynamic Quality
    WHEEL_VELOCITY_RUNAWAY = "F3.1"
    EXCESSIVE_WHEEL_ACCELERATION = "F3.2"
    OSCILLATION = "F3.3"
    POSITION_DRIFT = "F3.4"


@dataclass
class ThresholdCrossing:
    """Records when a threshold was crossed."""
    failure_mode: FailureMode
    step: int
    time_s: float
    value: float
    threshold: float


@dataclass
class ClassificationResult:
    """Result of failure classification."""
    primary_failure_mode: FailureMode
    first_threshold_crossing_step: int
    first_threshold_crossing_time_s: float
    secondary_threshold_crossings: List[ThresholdCrossing]
    responsible_component: str
    evidence_fields: Dict[str, Any]
    fix_allowed_in_balance_core: bool
    recommended_fix_scope: str


class FailureClassifier:
    """Classifies failure modes using temporal root-cause analysis."""

    # Thresholds from spec
    PITCH_X_MAX = 0.30  # rad
    ROLL_Y_MAX = 0.20  # rad
    COM_Z_DROP_MAX = 0.05  # m
    KNEE_ERROR_MAX = 0.15  # rad
    WHEEL_VEL_MAX = 50.0  # rad/s
    WHEEL_ACC_MAX = 100.0  # rad/s²
    POSITION_DRIFT_MAX = 0.5  # m

    def classify(self, df: pd.DataFrame) -> ClassificationResult:
        """Classify failure mode from telemetry.

        Args:
            df: Telemetry dataframe

        Returns:
            ClassificationResult with primary failure and evidence
        """
        # Find all threshold crossings in temporal order
        crossings = self._find_all_crossings(df)

        if not crossings:
            raise ValueError("No threshold crossings found - simulation may have succeeded")

        # Sort by step to get temporal order
        crossings.sort(key=lambda c: c.step)

        # First crossing is the primary failure
        primary = crossings[0]
        secondary = crossings[1:] if len(crossings) > 1 else []

        # Map failure mode to responsible component
        component = self._map_to_component(primary.failure_mode)

        # Determine if fix is allowed in balance-core
        fix_allowed = self._is_fix_allowed_in_balance_core(primary.failure_mode)

        # Generate recommended fix scope
        fix_scope = self._get_fix_scope(primary.failure_mode)

        # Collect evidence fields
        evidence = self._collect_evidence(df, primary)

        return ClassificationResult(
            primary_failure_mode=primary.failure_mode,
            first_threshold_crossing_step=primary.step,
            first_threshold_crossing_time_s=primary.time_s,
            secondary_threshold_crossings=secondary,
            responsible_component=component,
            evidence_fields=evidence,
            fix_allowed_in_balance_core=fix_allowed,
            recommended_fix_scope=fix_scope,
        )

    def _find_all_crossings(self, df: pd.DataFrame) -> List[ThresholdCrossing]:
        """Find all threshold crossings in temporal order."""
        crossings = []

        # Check pitch divergence
        pitch_violations = df[df["pitch_x_rad"].abs() > self.PITCH_X_MAX]
        if len(pitch_violations) > 0:
            first = pitch_violations.iloc[0]
            crossings.append(ThresholdCrossing(
                failure_mode=FailureMode.PITCH_DIVERGENCE,
                step=int(first["step"]),
                time_s=float(first["time"]),
                value=float(first["pitch_x_rad"]),
                threshold=self.PITCH_X_MAX,
            ))

        # Check roll divergence
        roll_violations = df[df["roll_y_rad"].abs() > self.ROLL_Y_MAX]
        if len(roll_violations) > 0:
            first = roll_violations.iloc[0]
            crossings.append(ThresholdCrossing(
                failure_mode=FailureMode.ROLL_DIVERGENCE,
                step=int(first["step"]),
                time_s=float(first["time"]),
                value=float(first["roll_y_rad"]),
                threshold=self.ROLL_Y_MAX,
            ))

        # Check height collapse (CoM drop from initial)
        if len(df) > 0:
            initial_com_z = df.iloc[0]["com_z_m"]
            com_z_drop = initial_com_z - df["com_z_m"]
            height_violations = df[com_z_drop > self.COM_Z_DROP_MAX]
            if len(height_violations) > 0:
                first = height_violations.iloc[0]
                crossings.append(ThresholdCrossing(
                    failure_mode=FailureMode.HEIGHT_COLLAPSE,
                    step=int(first["step"]),
                    time_s=float(first["time"]),
                    value=float(first["com_z_m"]),
                    threshold=initial_com_z - self.COM_Z_DROP_MAX,
                ))

        # Check contact loss
        contact_loss = df[df["contact_supervisor_state"] == "NO_CONTACT"]
        if len(contact_loss) > 0:
            first = contact_loss.iloc[0]
            crossings.append(ThresholdCrossing(
                failure_mode=FailureMode.CONTACT_LOSS,
                step=int(first["step"]),
                time_s=float(first["time"]),
                value=0.0,
                threshold=0.0,
            ))

        return crossings

    def _map_to_component(self, failure_mode: FailureMode) -> str:
        """Map failure mode to responsible balance-core component."""
        mapping = {
            FailureMode.PITCH_DIVERGENCE: "SagittalWheelBalanceController",
            FailureMode.ROLL_DIVERGENCE: "LateralRollBalanceController",
            FailureMode.HEIGHT_COLLAPSE: "ShapePostureController or SupportFeedforwardController",
            FailureMode.KNEE_SUPPORT_COLLAPSE: "ShapePostureController or SupportFeedforwardController",
            FailureMode.CONTACT_LOSS: "ContactSupervisor (if primary) or earlier failure",
            FailureMode.WHEEL_VELOCITY_RUNAWAY: "SagittalWheelBalanceController",
            FailureMode.EXCESSIVE_WHEEL_ACCELERATION: "SagittalWheelBalanceController or SafetyLimiter",
            FailureMode.OSCILLATION: "Controller for oscillating axis",
            FailureMode.POSITION_DRIFT: "Future outer-loop controller (defer)",
        }
        return mapping.get(failure_mode, "Unknown")

    def _is_fix_allowed_in_balance_core(self, failure_mode: FailureMode) -> bool:
        """Determine if fix is allowed within balance-core architecture."""
        # Priority 0: Must fix architecture
        if failure_mode.value.startswith("F0"):
            return True

        # Priority 1-2: Fix within balance-core
        if failure_mode in [
            FailureMode.PITCH_DIVERGENCE,
            FailureMode.ROLL_DIVERGENCE,
            FailureMode.HEIGHT_COLLAPSE,
            FailureMode.KNEE_SUPPORT_COLLAPSE,
        ]:
            return True

        # Priority 3: Some allowed, some deferred
        if failure_mode == FailureMode.POSITION_DRIFT:
            return False  # Defer to outer-loop

        return True

    def _get_fix_scope(self, failure_mode: FailureMode) -> str:
        """Get recommended fix scope for failure mode."""
        if failure_mode == FailureMode.PITCH_DIVERGENCE:
            return "SagittalWheelBalanceController: verify inputs, sign, saturation, then adjust gains"
        elif failure_mode == FailureMode.ROLL_DIVERGENCE:
            return "LateralRollBalanceController: verify inputs, sign, saturation, then adjust gains"
        elif failure_mode == FailureMode.HEIGHT_COLLAPSE:
            return "ShapePostureController or SupportFeedforwardController: verify support torque"
        elif failure_mode == FailureMode.POSITION_DRIFT:
            return "Defer to future outer-loop position controller"
        else:
            return "Component-specific diagnostic required"

    def _collect_evidence(self, df: pd.DataFrame, primary: ThresholdCrossing) -> Dict[str, Any]:
        """Collect evidence fields for the primary failure."""
        evidence = {
            "primary_failure_value": primary.value,
            "primary_failure_threshold": primary.threshold,
        }

        # Add relevant time-series statistics if fields exist
        if primary.failure_mode == FailureMode.PITCH_DIVERGENCE:
            evidence["pitch_max_rad"] = float(df["pitch_x_rad"].abs().max())
            if "pitch_rate_rad_s" in df.columns:
                evidence["pitch_rate_max_rad_s"] = float(df["pitch_rate_rad_s"].abs().max())
        elif primary.failure_mode == FailureMode.ROLL_DIVERGENCE:
            evidence["roll_max_rad"] = float(df["roll_y_rad"].abs().max())
            if "roll_rate_rad_s" in df.columns:
                evidence["roll_rate_max_rad_s"] = float(df["roll_rate_rad_s"].abs().max())

        return evidence
