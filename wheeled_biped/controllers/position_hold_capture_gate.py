"""Smart position hold gating for sagittal balance controller.

Physics-based gating that selectively reduces tau_position when it opposes
the required pitch capture direction, while preserving position hold benefits
during steady-state and recovery phases.

Design principle:
- During forward pitch lean, wheels may need to accelerate forward to catch CoM
- If tau_position < 0 (tries to return backward), it opposes capture
- Gate tau_position only during this conflict
- Restore smoothly after pitch reversal or capture recovery

This is NOT the failed T1-T4 diagnostic modes. This is a new physics-based approach.
"""

import jax.numpy as jnp
from jax import Array
from typing import NamedTuple


class CaptureGateState(NamedTuple):
    """State for position hold capture gate."""
    gate_factor: float  # 0.0 = fully gated, 1.0 = fully active
    conflict_detected: bool
    pitch_reversal_detected: bool
    capture_recovery_detected: bool
    required_capture_direction: float  # +1.0 forward, -1.0 backward, 0.0 none


class CaptureGateDiagnostics(NamedTuple):
    """Diagnostics for position hold capture gate."""
    required_capture_direction: float
    tau_position_direction: float
    position_opposes_capture: bool
    gate_factor: float
    gate_active: bool
    gate_reason: str
    pitch_reversal_detected: bool
    capture_recovery_detected: bool
    tau_position_raw: float
    tau_position_gated: float
    capture_point_relative_to_support_m: float
    com_support_error_y_m: float


class PositionHoldCaptureGate:
    """Smart position hold gating based on pitch capture direction.

    Gates tau_position only when it opposes the required capture direction,
    preserving position hold benefits during steady-state and recovery.

    Args:
        pitch_threshold_rad: Pitch magnitude threshold to activate capture mode (default 0.05 rad ~2.9 deg)
        gate_factor_conflict: Gate factor during conflict (default 0.0 = fully gate)
        gate_factor_normal: Gate factor during normal operation (default 1.0 = no gating)
        smooth_ramp_steps: Number of steps for smooth gate factor transitions (default 10)
        enable_capture_point: Use capture point for direction (default True)
        gravity_m_s2: Gravity constant for capture point calculation (default 9.81)
    """

    def __init__(
        self,
        pitch_threshold_rad: float = 0.05,
        gate_factor_conflict: float = 0.0,
        gate_factor_normal: float = 1.0,
        smooth_ramp_steps: int = 10,
        enable_capture_point: bool = True,
        gravity_m_s2: float = 9.81,
        warmup_steps: int = 100,
    ):
        self.pitch_threshold_rad = pitch_threshold_rad
        self.gate_factor_conflict = gate_factor_conflict
        self.gate_factor_normal = gate_factor_normal
        self.smooth_ramp_steps = smooth_ramp_steps
        self.enable_capture_point = enable_capture_point
        self.gravity_m_s2 = gravity_m_s2
        self.warmup_steps = warmup_steps

        # State
        self._gate_factor = gate_factor_normal
        self._conflict_count = 0
        self._recovery_count = 0
        self._step_count = 0

    def compute_required_capture_direction(
        self,
        pitch_x_rad: float,
        pitch_rate_x_rad_s: float,
        com_y_m: float,
        com_vy_m_s: float,
        support_center_y_m: float,
        com_z_m: float,
    ) -> tuple[float, float, float]:
        """Compute required capture direction from physical state.

        Args:
            pitch_x_rad: Body pitch angle (rad). Positive = forward lean.
            pitch_rate_x_rad_s: Body pitch rate (rad/s).
            com_y_m: CoM position in sagittal direction (m).
            com_vy_m_s: CoM velocity in sagittal direction (m/s).
            support_center_y_m: Support center position in sagittal direction (m).
            com_z_m: CoM height (m).

        Returns:
            required_capture_direction: +1.0 forward, -1.0 backward, 0.0 none
            capture_point_relative_to_support_m: Capture point position relative to support
            com_support_error_y_m: CoM position error relative to support
        """
        # CoM-support error
        com_support_error_y_m = com_y_m - support_center_y_m

        if self.enable_capture_point and com_z_m > 0.1:
            # Compute capture point using inverted pendulum model
            # cp = com + com_vel / omega, where omega = sqrt(g / h)
            omega = jnp.sqrt(self.gravity_m_s2 / com_z_m)
            capture_point_y_m = com_y_m + com_vy_m_s / omega
            capture_point_relative_to_support_m = capture_point_y_m - support_center_y_m

            # Required capture direction based on capture point
            # Use larger threshold (10cm) to avoid false positives in steady state
            if capture_point_relative_to_support_m > 0.10:  # 10cm threshold
                required_capture_direction = 1.0  # forward
            elif capture_point_relative_to_support_m < -0.10:
                required_capture_direction = -1.0  # backward
            else:
                required_capture_direction = 0.0  # no capture needed
        else:
            # Fallback: use pitch error and pitch rate
            capture_point_relative_to_support_m = 0.0

            if abs(pitch_x_rad) > self.pitch_threshold_rad:
                # Use pitch sign as capture direction
                # Positive pitch (forward lean) requires forward wheel acceleration
                required_capture_direction = jnp.sign(pitch_x_rad)
            else:
                required_capture_direction = 0.0

        return (
            float(required_capture_direction),
            float(capture_point_relative_to_support_m),
            float(com_support_error_y_m),
        )

    def detect_conflict(
        self,
        tau_position_raw: float,
        required_capture_direction: float,
    ) -> bool:
        """Detect if tau_position opposes required capture direction.

        Args:
            tau_position_raw: Raw position hold torque (Nm).
            required_capture_direction: Required capture direction (+1/-1/0).

        Returns:
            True if conflict detected (tau_position opposes capture).
        """
        if required_capture_direction == 0.0:
            return False

        tau_position_direction = jnp.sign(tau_position_raw)

        # Conflict: tau_position direction is opposite to required capture direction
        conflict = tau_position_direction == -required_capture_direction

        return bool(conflict)

    def detect_recovery(
        self,
        pitch_x_rad: float,
        pitch_rate_x_rad_s: float,
        capture_point_relative_to_support_m: float,
    ) -> tuple[bool, bool]:
        """Detect pitch reversal or capture recovery for gate restoration.

        Args:
            pitch_x_rad: Body pitch angle (rad).
            pitch_rate_x_rad_s: Body pitch rate (rad/s).
            capture_point_relative_to_support_m: Capture point relative to support (m).

        Returns:
            pitch_reversal_detected: True if pitch is reversing toward upright
            capture_recovery_detected: True if capture point is near support
        """
        # Pitch reversal: pitch magnitude small and pitch rate indicates return
        pitch_reversal_detected = (
            abs(pitch_x_rad) < self.pitch_threshold_rad * 0.5 and
            abs(pitch_rate_x_rad_s) < 0.1
        )

        # Capture recovery: capture point is close to support
        # Only check if capture point is enabled, otherwise always False
        if self.enable_capture_point:
            capture_recovery_detected = abs(capture_point_relative_to_support_m) < 0.10  # 10cm
        else:
            capture_recovery_detected = False

        return bool(pitch_reversal_detected), bool(capture_recovery_detected)

    def update_gate_factor(
        self,
        conflict_detected: bool,
        pitch_reversal_detected: bool,
        capture_recovery_detected: bool,
    ) -> float:
        """Update gate factor with smooth transitions.

        Args:
            conflict_detected: True if tau_position opposes capture.
            pitch_reversal_detected: True if pitch is reversing.
            capture_recovery_detected: True if capture point is near support.

        Returns:
            Updated gate factor (0.0 = fully gated, 1.0 = fully active).
        """
        # Determine target gate factor
        if conflict_detected and not (pitch_reversal_detected or capture_recovery_detected):
            target_gate_factor = self.gate_factor_conflict
            self._conflict_count += 1
            self._recovery_count = 0
        else:
            target_gate_factor = self.gate_factor_normal
            self._recovery_count += 1
            self._conflict_count = 0

        # Smooth ramp toward target
        if self.smooth_ramp_steps > 0:
            ramp_rate = 1.0 / self.smooth_ramp_steps
            if self._gate_factor < target_gate_factor:
                self._gate_factor = min(self._gate_factor + ramp_rate, target_gate_factor)
            elif self._gate_factor > target_gate_factor:
                self._gate_factor = max(self._gate_factor - ramp_rate, target_gate_factor)
        else:
            self._gate_factor = target_gate_factor

        return self._gate_factor

    def apply_gate(
        self,
        tau_position_raw: float,
        pitch_x_rad: float,
        pitch_rate_x_rad_s: float,
        com_y_m: float,
        com_vy_m_s: float,
        support_center_y_m: float,
        com_z_m: float,
    ) -> tuple[float, CaptureGateDiagnostics]:
        """Apply smart position hold gating.

        Args:
            tau_position_raw: Raw position hold torque (Nm).
            pitch_x_rad: Body pitch angle (rad).
            pitch_rate_x_rad_s: Body pitch rate (rad/s).
            com_y_m: CoM position in sagittal direction (m).
            com_vy_m_s: CoM velocity in sagittal direction (m/s).
            support_center_y_m: Support center position in sagittal direction (m).
            com_z_m: CoM height (m).

        Returns:
            tau_position_gated: Gated position hold torque (Nm).
            diagnostics: Capture gate diagnostics.
        """
        # Increment step counter
        self._step_count += 1

        # During warmup, bypass gating entirely
        if self._step_count <= self.warmup_steps:
            diagnostics = CaptureGateDiagnostics(
                required_capture_direction=0.0,
                tau_position_direction=0.0,
                position_opposes_capture=False,
                gate_factor=1.0,
                gate_active=False,
                gate_reason="warmup",
                pitch_reversal_detected=False,
                capture_recovery_detected=False,
                tau_position_raw=tau_position_raw,
                tau_position_gated=tau_position_raw,
                capture_point_relative_to_support_m=0.0,
                com_support_error_y_m=0.0,
            )
            return tau_position_raw, diagnostics

        # Compute required capture direction
        (
            required_capture_direction,
            capture_point_relative_to_support_m,
            com_support_error_y_m,
        ) = self.compute_required_capture_direction(
            pitch_x_rad,
            pitch_rate_x_rad_s,
            com_y_m,
            com_vy_m_s,
            support_center_y_m,
            com_z_m,
        )

        # Detect conflict
        conflict_detected = self.detect_conflict(tau_position_raw, required_capture_direction)

        # Detect recovery
        pitch_reversal_detected, capture_recovery_detected = self.detect_recovery(
            pitch_x_rad,
            pitch_rate_x_rad_s,
            capture_point_relative_to_support_m,
        )

        # Update gate factor
        gate_factor = self.update_gate_factor(
            conflict_detected,
            pitch_reversal_detected,
            capture_recovery_detected,
        )

        # Apply gate
        tau_position_gated = gate_factor * tau_position_raw

        # Determine gate reason
        if conflict_detected and gate_factor < 0.5:
            gate_reason = "conflict_active"
        elif pitch_reversal_detected:
            gate_reason = "pitch_reversal"
        elif capture_recovery_detected:
            gate_reason = "capture_recovery"
        elif gate_factor < 1.0:
            gate_reason = "ramping_up"
        else:
            gate_reason = "normal"

        # Build diagnostics
        tau_position_direction = float(jnp.sign(tau_position_raw)) if tau_position_raw != 0.0 else 0.0

        diagnostics = CaptureGateDiagnostics(
            required_capture_direction=required_capture_direction,
            tau_position_direction=tau_position_direction,
            position_opposes_capture=conflict_detected,
            gate_factor=gate_factor,
            gate_active=(gate_factor < 1.0),
            gate_reason=gate_reason,
            pitch_reversal_detected=pitch_reversal_detected,
            capture_recovery_detected=capture_recovery_detected,
            tau_position_raw=tau_position_raw,
            tau_position_gated=tau_position_gated,
            capture_point_relative_to_support_m=capture_point_relative_to_support_m,
            com_support_error_y_m=com_support_error_y_m,
        )

        return tau_position_gated, diagnostics

    def reset(self):
        """Reset gate state."""
        self._gate_factor = self.gate_factor_normal
        self._conflict_count = 0
        self._recovery_count = 0
        self._step_count = 0
