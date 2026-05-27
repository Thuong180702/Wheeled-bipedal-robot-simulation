"""Temporal root-cause failure classifier for balance-core controller.

This module implements temporal analysis to identify primary failure modes
from telemetry data. The first threshold crossing in time is classified as
the primary failure; subsequent crossings are secondary.

Failure taxonomy:
- F0.x: Telemetry/data quality issues
- F1.x: Contact state violations
- F2.x: Orientation divergence (pitch/roll)
- F3.x: Height/position failures
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import pandas as pd


class FailureMode(Enum):
    """Failure mode taxonomy for balance-core controller."""

    # F0.x: Telemetry/data quality issues
    TELEMETRY_MISSING_FIELDS = "F0.1"
    TELEMETRY_NAN_VALUES = "F0.2"
    TELEMETRY_INFINITE_VALUES = "F0.3"
    TELEMETRY_TIMESTAMP_REGRESSION = "F0.4"
    TELEMETRY_DUPLICATE_STEPS = "F0.5"
    TELEMETRY_STEP_GAP = "F0.6"

    # F1.x: Contact state violations
    CONTACT_STATE_INVALID = "F1.1"
    CONTACT_LOSS_UNEXPECTED = "F1.2"
    CONTACT_OSCILLATION = "F1.3"

    # F2.x: Orientation divergence
    PITCH_DIVERGENCE = "F2.1"
    ROLL_DIVERGENCE = "F2.2"

    # F3.x: Height/position failures
    HEIGHT_COLLAPSE = "F3.1"
    HEIGHT_OVERSHOOT = "F3.2"
    XY_DRIFT_EXCESSIVE = "F3.3"
    VELOCITY_RUNAWAY = "F3.4"


@dataclass
class ThresholdCrossing:
    """Record of a threshold violation."""

    failure_mode: FailureMode
    step: int
    time: float
    value: float
    threshold: float


@dataclass
class ClassificationResult:
    """Result of failure classification."""

    primary_failure_mode: FailureMode
    first_threshold_crossing_step: int
    first_threshold_crossing_time: float
    responsible_component: str
    fix_allowed_in_balance_core: bool
    recommended_fix_scope: str
    secondary_threshold_crossings: List[ThresholdCrossing] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)


class FailureClassifier:
    """Temporal root-cause classifier for balance-core failures.

    Uses temporal analysis: the first threshold crossing in time is the
    primary failure mode. All subsequent crossings are secondary effects.
    """

    # Threshold constants
    PITCH_X_MAX = 0.30  # rad (~17 deg)
    ROLL_Y_MAX = 0.20  # rad (~11 deg)
    COM_Z_DROP_MAX = 0.05  # m (5 cm drop from initial)
    COM_Z_OVERSHOOT_MAX = 0.10  # m (10 cm above initial)
    XY_DRIFT_MAX = 0.50  # m (50 cm drift)
    VELOCITY_RUNAWAY_THRESHOLD = 5.0  # m/s

    def classify(self, df: pd.DataFrame) -> ClassificationResult:
        """Classify failure mode from telemetry dataframe.

        Args:
            df: Telemetry dataframe with required fields

        Returns:
            ClassificationResult with primary and secondary failures
        """
        # Find all threshold crossings in temporal order
        crossings = self._find_all_crossings(df)

        if not crossings:
            # No failures detected - this shouldn't happen if called on failed episode
            # Return a default classification
            return ClassificationResult(
                primary_failure_mode=FailureMode.TELEMETRY_MISSING_FIELDS,
                first_threshold_crossing_step=0,
                first_threshold_crossing_time=0.0,
                responsible_component="Unknown",
                fix_allowed_in_balance_core=False,
                recommended_fix_scope="Unknown",
                secondary_threshold_crossings=[],
                evidence={},
            )

        # First crossing is primary
        primary = crossings[0]
        secondary = crossings[1:]

        # Map to responsible component
        component = self._map_to_component(primary.failure_mode)

        # Determine if fix is allowed in balance-core
        fix_allowed = self._is_fix_allowed_in_balance_core(primary.failure_mode)

        # Get recommended fix scope
        fix_scope = self._get_fix_scope(primary.failure_mode)

        # Collect evidence
        evidence = self._collect_evidence(df, primary)

        return ClassificationResult(
            primary_failure_mode=primary.failure_mode,
            first_threshold_crossing_step=primary.step,
            first_threshold_crossing_time=primary.time,
            responsible_component=component,
            fix_allowed_in_balance_core=fix_allowed,
            recommended_fix_scope=fix_scope,
            secondary_threshold_crossings=secondary,
            evidence=evidence,
        )

    def _find_all_crossings(self, df: pd.DataFrame) -> List[ThresholdCrossing]:
        """Find all threshold crossings in temporal order.

        Args:
            df: Telemetry dataframe

        Returns:
            List of ThresholdCrossing sorted by step
        """
        crossings = []

        # Check pitch divergence (F2.1)
        if "pitch_x_rad" in df.columns:
            pitch_violations = df[df["pitch_x_rad"].abs() > self.PITCH_X_MAX]
            if not pitch_violations.empty:
                first_violation = pitch_violations.iloc[0]
                crossings.append(
                    ThresholdCrossing(
                        failure_mode=FailureMode.PITCH_DIVERGENCE,
                        step=int(first_violation["step"]),
                        time=float(first_violation["time"]),
                        value=float(first_violation["pitch_x_rad"]),
                        threshold=self.PITCH_X_MAX,
                    )
                )

        # Check roll divergence (F2.2)
        if "roll_y_rad" in df.columns:
            roll_violations = df[df["roll_y_rad"].abs() > self.ROLL_Y_MAX]
            if not roll_violations.empty:
                first_violation = roll_violations.iloc[0]
                crossings.append(
                    ThresholdCrossing(
                        failure_mode=FailureMode.ROLL_DIVERGENCE,
                        step=int(first_violation["step"]),
                        time=float(first_violation["time"]),
                        value=float(first_violation["roll_y_rad"]),
                        threshold=self.ROLL_Y_MAX,
                    )
                )

        # Check height collapse (F3.1)
        if "com_z_m" in df.columns and len(df) > 0:
            initial_height = df["com_z_m"].iloc[0]
            height_drop = initial_height - df["com_z_m"]
            height_violations = df[height_drop > self.COM_Z_DROP_MAX]
            if not height_violations.empty:
                first_violation = height_violations.iloc[0]
                crossings.append(
                    ThresholdCrossing(
                        failure_mode=FailureMode.HEIGHT_COLLAPSE,
                        step=int(first_violation["step"]),
                        time=float(first_violation["time"]),
                        value=float(first_violation["com_z_m"]),
                        threshold=initial_height - self.COM_Z_DROP_MAX,
                    )
                )

        # Check height overshoot (F3.2)
        if "com_z_m" in df.columns and len(df) > 0:
            initial_height = df["com_z_m"].iloc[0]
            height_overshoot = df["com_z_m"] - initial_height
            overshoot_violations = df[height_overshoot > self.COM_Z_OVERSHOOT_MAX]
            if not overshoot_violations.empty:
                first_violation = overshoot_violations.iloc[0]
                crossings.append(
                    ThresholdCrossing(
                        failure_mode=FailureMode.HEIGHT_OVERSHOOT,
                        step=int(first_violation["step"]),
                        time=float(first_violation["time"]),
                        value=float(first_violation["com_z_m"]),
                        threshold=initial_height + self.COM_Z_OVERSHOOT_MAX,
                    )
                )

        # Check XY drift (F3.3)
        if "com_x_m" in df.columns and "com_y_m" in df.columns and len(df) > 0:
            initial_x = df["com_x_m"].iloc[0]
            initial_y = df["com_y_m"].iloc[0]
            drift = ((df["com_x_m"] - initial_x) ** 2 + (df["com_y_m"] - initial_y) ** 2) ** 0.5
            drift_violations = df[drift > self.XY_DRIFT_MAX]
            if not drift_violations.empty:
                first_violation = drift_violations.iloc[0]
                drift_value = float(
                    ((first_violation["com_x_m"] - initial_x) ** 2 + (first_violation["com_y_m"] - initial_y) ** 2)
                    ** 0.5
                )
                crossings.append(
                    ThresholdCrossing(
                        failure_mode=FailureMode.XY_DRIFT_EXCESSIVE,
                        step=int(first_violation["step"]),
                        time=float(first_violation["time"]),
                        value=drift_value,
                        threshold=self.XY_DRIFT_MAX,
                    )
                )

        # Check velocity runaway (F3.4)
        if "com_vx_m_s" in df.columns and "com_vy_m_s" in df.columns:
            velocity_mag = (df["com_vx_m_s"] ** 2 + df["com_vy_m_s"] ** 2) ** 0.5
            velocity_violations = df[velocity_mag > self.VELOCITY_RUNAWAY_THRESHOLD]
            if not velocity_violations.empty:
                first_violation = velocity_violations.iloc[0]
                vel_value = float(
                    (first_violation["com_vx_m_s"] ** 2 + first_violation["com_vy_m_s"] ** 2) ** 0.5
                )
                crossings.append(
                    ThresholdCrossing(
                        failure_mode=FailureMode.VELOCITY_RUNAWAY,
                        step=int(first_violation["step"]),
                        time=float(first_violation["time"]),
                        value=vel_value,
                        threshold=self.VELOCITY_RUNAWAY_THRESHOLD,
                    )
                )

        # Sort by step (temporal order)
        crossings.sort(key=lambda x: x.step)

        return crossings

    def _map_to_component(self, failure_mode: FailureMode) -> str:
        """Map failure mode to responsible component.

        Args:
            failure_mode: The failure mode

        Returns:
            Name of responsible component
        """
        component_map = {
            FailureMode.PITCH_DIVERGENCE: "SagittalWheelBalanceController",
            FailureMode.ROLL_DIVERGENCE: "LateralHipRollController",
            FailureMode.HEIGHT_COLLAPSE: "HeightController",
            FailureMode.HEIGHT_OVERSHOOT: "HeightController",
            FailureMode.XY_DRIFT_EXCESSIVE: "SagittalWheelBalanceController",
            FailureMode.VELOCITY_RUNAWAY: "SagittalWheelBalanceController",
            FailureMode.CONTACT_STATE_INVALID: "ContactSupervisor",
            FailureMode.CONTACT_LOSS_UNEXPECTED: "ContactSupervisor",
            FailureMode.CONTACT_OSCILLATION: "ContactSupervisor",
        }
        return component_map.get(failure_mode, "Unknown")

    def _is_fix_allowed_in_balance_core(self, failure_mode: FailureMode) -> bool:
        """Determine if fix is allowed within balance-core scope.

        Args:
            failure_mode: The failure mode

        Returns:
            True if fix is allowed in balance-core
        """
        # Balance-core can fix orientation and height issues
        allowed_modes = {
            FailureMode.PITCH_DIVERGENCE,
            FailureMode.ROLL_DIVERGENCE,
            FailureMode.HEIGHT_COLLAPSE,
            FailureMode.HEIGHT_OVERSHOOT,
        }
        return failure_mode in allowed_modes

    def _get_fix_scope(self, failure_mode: FailureMode) -> str:
        """Get recommended fix scope for failure mode.

        Args:
            failure_mode: The failure mode

        Returns:
            Recommended fix scope description
        """
        scope_map = {
            FailureMode.PITCH_DIVERGENCE: "Tune SagittalWheelBalanceController gains",
            FailureMode.ROLL_DIVERGENCE: "Tune LateralHipRollController gains",
            FailureMode.HEIGHT_COLLAPSE: "Tune HeightController gains or increase leg stiffness",
            FailureMode.HEIGHT_OVERSHOOT: "Tune HeightController gains or reduce leg stiffness",
            FailureMode.XY_DRIFT_EXCESSIVE: "Requires higher-level locomotion controller (out of scope)",
            FailureMode.VELOCITY_RUNAWAY: "Requires higher-level locomotion controller (out of scope)",
            FailureMode.CONTACT_STATE_INVALID: "Fix ContactSupervisor logic (out of scope)",
            FailureMode.CONTACT_LOSS_UNEXPECTED: "Fix ContactSupervisor logic (out of scope)",
            FailureMode.CONTACT_OSCILLATION: "Fix ContactSupervisor logic (out of scope)",
        }
        return scope_map.get(failure_mode, "Unknown scope")

    def _collect_evidence(self, df: pd.DataFrame, primary: ThresholdCrossing) -> dict:
        """Collect evidence fields for primary failure.

        Args:
            df: Telemetry dataframe
            primary: Primary threshold crossing

        Returns:
            Dictionary of evidence fields
        """
        evidence = {
            "primary_failure_mode": primary.failure_mode.value,
            "first_crossing_step": primary.step,
            "first_crossing_time": primary.time,
            "first_crossing_value": primary.value,
            "threshold": primary.threshold,
        }

        # Add relevant state at failure time
        failure_row = df[df["step"] == primary.step]
        if not failure_row.empty:
            row = failure_row.iloc[0]
            if "pitch_x_rad" in df.columns:
                evidence["pitch_at_failure_rad"] = float(row["pitch_x_rad"])
            if "roll_y_rad" in df.columns:
                evidence["roll_at_failure_rad"] = float(row["roll_y_rad"])
            if "com_z_m" in df.columns:
                evidence["height_at_failure_m"] = float(row["com_z_m"])
            if "contact_supervisor_state" in df.columns:
                evidence["contact_state_at_failure"] = str(row["contact_supervisor_state"])

        return evidence
