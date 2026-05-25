"""Simple force distributor without QP optimization.

Replaces the slow QP-based force distribution with direct force mapping
that runs in microseconds instead of seconds.
"""

import jax.numpy as jnp
from jax import Array


class SimpleForceDistributor:
    """Direct force distribution without QP optimization."""

    def __init__(
        self,
        tau_hip_roll_max: float = 30.0,
        max_force_asymmetry: float = 40.0,
        min_wheel_force: float = 10.0,
    ):
        """Initialize force distributor with safety limits.

        Args:
            tau_hip_roll_max: Maximum hip roll torque per side (Nm) - increased to 30.0 for stronger roll correction
            max_force_asymmetry: Maximum allowed vertical force difference (N) - increased to 40.0 for more authority
            min_wheel_force: Minimum safe wheel force to prevent liftoff (N) - reduced to 10.0 to allow larger asymmetry
        """
        self.tau_hip_roll_max = tau_hip_roll_max
        self.max_force_asymmetry = max_force_asymmetry
        self.min_wheel_force = min_wheel_force

    def _roll_moment_to_hip_roll_torque(self, mx: Array) -> Array:
        tau_hip_roll = jnp.array([-mx / 2.0, mx / 2.0])
        return jnp.clip(tau_hip_roll, -self.tau_hip_roll_max, self.tau_hip_roll_max)

    def distribute_wrench_contact_aware(
        self,
        desired_wrench: Array,
        left_contact: bool,
        right_contact: bool,
        wheel_pos_left: Array | None = None,
        wheel_pos_right: Array | None = None,
        hip_roll_authority_scale: float = 1.0,
        recovery_mode: bool = False,
        distribution_mode: str = "absolute",
        max_delta_fz: float = 30.0,
    ) -> tuple[Array, Array, Array, dict]:
        """Distribute wrench to contact forces.

        Args:
            desired_wrench: (6,) [Fx, Fy, Fz, Mx, My, Mz]
            left_contact: Left wheel in contact
            right_contact: Right wheel in contact
            wheel_pos_left: Left wheel position relative to CoM
            wheel_pos_right: Right wheel position relative to CoM
            hip_roll_authority_scale: Hip roll authority scaling
            recovery_mode: If True, apply min_recovery_force behavior for single-contact recovery.
            distribution_mode: "absolute" (default) treats wrench as absolute contact force,
                             "delta" treats wrench as correction/delta force
            max_delta_fz: Maximum delta force per wheel in delta mode (N)

        Returns:
            Tuple of (f_left, f_right, tau_hip_roll, diagnostics)
        """
        Fx, Fy, Fz, Mx, My, Mz = desired_wrench

        _ = Mz

        # For wheeled biped with left/right wheels separated along X-axis (lateral)
        # Cross product: M = r × F, so My = -x * Fz for vertical forces
        # My = -x_l * fz_l - x_r * fz_r (roll moment from lateral force asymmetry)
        # Mx = y_l * fz_l + y_r * fz_r (pitch moment from sagittal force asymmetry)
        x_l = float(wheel_pos_left[0]) if wheel_pos_left is not None else 0.0
        x_r = float(wheel_pos_right[0]) if wheel_pos_right is not None else 0.0
        x_denom = x_l - x_r

        y_l = float(wheel_pos_left[1]) if wheel_pos_left is not None else 0.0
        y_r = float(wheel_pos_right[1]) if wheel_pos_right is not None else 0.0
        y_denom = y_l - y_r

        hip_roll_authority_scale = float(jnp.clip(hip_roll_authority_scale, 0.0, 1.0))
        tau_hip_roll = self._roll_moment_to_hip_roll_torque(My) * hip_roll_authority_scale

        # DELTA DISTRIBUTION MODE: For correction-only WBC
        if distribution_mode == "delta":
            # Treat wrench as delta/correction force, not absolute contact force
            # Allow negative delta forces (reducing baseline contact load)

            wrench_norm = float(jnp.linalg.norm(desired_wrench))
            is_near_zero = wrench_norm < 1.0

            if is_near_zero:
                return (
                    jnp.zeros(3),
                    jnp.zeros(3),
                    jnp.zeros(2),
                    {"feasible": True, "reason": "zero_correction_delta"},
                )

            active_count = int(left_contact) + int(right_contact)

            if active_count == 2:
                # Double contact: solve for delta forces that achieve correction wrench
                # delta_fz_l + delta_fz_r = correction_Fz
                # -x_l * delta_fz_l - x_r * delta_fz_r = correction_My
                if abs(float(x_denom)) < 1e-6:
                    delta_fz_left = Fz / 2.0
                    delta_fz_right = Fz / 2.0
                else:
                    delta_fz_left = (-My - x_r * Fz) / x_denom
                    delta_fz_right = Fz - delta_fz_left

                # Clip delta forces by max_delta_fz (not liftoff threshold)
                delta_fz_left = jnp.clip(delta_fz_left, -max_delta_fz, max_delta_fz)
                delta_fz_right = jnp.clip(delta_fz_right, -max_delta_fz, max_delta_fz)

                f_left = jnp.array([Fx / 2.0, Fy / 2.0, delta_fz_left])
                f_right = jnp.array([Fx / 2.0, Fy / 2.0, delta_fz_right])

                return f_left, f_right, tau_hip_roll, {"feasible": True, "reason": "delta_double_contact"}

            elif active_count == 1:
                # Single contact: apply correction to contact wheel only
                if left_contact:
                    f_left = jnp.array([Fx, Fy, Fz])
                    f_right = jnp.zeros(3)
                else:
                    f_left = jnp.zeros(3)
                    f_right = jnp.array([Fx, Fy, Fz])

                return f_left, f_right, tau_hip_roll, {"feasible": True, "reason": "delta_single_contact"}

            else:
                # No contact: zero forces, preserve hip-roll torques
                return (
                    jnp.zeros(3),
                    jnp.zeros(3),
                    tau_hip_roll,
                    {"feasible": True, "reason": "delta_no_contact"},
                )

        # ABSOLUTE DISTRIBUTION MODE: Original behavior for non-correction WBC
        # CRITICAL: In normal mode (recovery_mode=False), zero wrench must produce zero force
        # Check if wrench is near zero (correction-only mode at equilibrium)
        wrench_norm = float(jnp.linalg.norm(desired_wrench))
        is_near_zero = wrench_norm < 1.0  # 1 N threshold

        if is_near_zero and not recovery_mode:
            # Zero correction wrench in normal mode → zero force
            return (
                jnp.zeros(3),
                jnp.zeros(3),
                jnp.zeros(2),
                {"feasible": True, "reason": "zero_correction"},
            )

        def split_fz_from_my(total_fz: Array, my_cmd: Array) -> tuple[Array, Array]:
            if abs(float(x_denom)) < 1e-6:
                fz_left_raw = total_fz / 2.0
                fz_right_raw = total_fz / 2.0
            else:
                fz_left_raw = (-my_cmd - x_r * total_fz) / x_denom
                fz_right_raw = total_fz - fz_left_raw

            fz_diff_raw = fz_left_raw - fz_right_raw
            fz_avg = total_fz / 2.0
            liftoff_threshold = 2.0 * (fz_avg - self.min_wheel_force - 5.0)
            max_safe_diff = jnp.maximum(0.0, jnp.minimum(self.max_force_asymmetry, liftoff_threshold))
            fz_diff = jnp.clip(fz_diff_raw, -max_safe_diff, max_safe_diff)
            fz_left = total_fz / 2.0 + fz_diff / 2.0
            fz_right = total_fz / 2.0 - fz_diff / 2.0
            return fz_left, fz_right

        if not left_contact and not right_contact:
            fz_left, fz_right = split_fz_from_my(Fz, My)
            f_left = jnp.array([Fx / 2.0, Fy / 2.0, fz_left])
            f_right = jnp.array([Fx / 2.0, Fy / 2.0, fz_right])
            return (
                f_left,
                f_right,
                tau_hip_roll,
                {"feasible": True, "reason": "flight_phase_anticipatory"},
            )

        active_count = int(left_contact) + int(right_contact)

        if active_count == 2:
            fz_left, fz_right = split_fz_from_my(Fz, My)
            f_left = jnp.array([Fx / 2.0, Fy / 2.0, fz_left])
            f_right = jnp.array([Fx / 2.0, Fy / 2.0, fz_right])
        else:
            # Single contact case
            if recovery_mode:
                # Recovery mode: apply aggressive roll correction and min_recovery_force
                recovery_roll_gain = 5.0
                recovery_mx = Mx * recovery_roll_gain
                min_recovery_force = 50.0
                tau_hip_roll = self._roll_moment_to_hip_roll_torque(recovery_mx) * hip_roll_authority_scale

                if left_contact:
                    f_left = jnp.array([Fx, Fy, Fz])
                    f_right = jnp.array([0.0, 0.0, min_recovery_force])
                else:
                    f_left = jnp.array([0.0, 0.0, min_recovery_force])
                    f_right = jnp.array([Fx, Fy, Fz])
            else:
                # Normal mode: solve for wrench, non-contact wheel gets ZERO force
                if left_contact:
                    f_left = jnp.array([Fx, Fy, Fz])
                    f_right = jnp.zeros(3)  # CRITICAL: No fake force on non-contact wheel
                else:
                    f_left = jnp.zeros(3)  # CRITICAL: No fake force on non-contact wheel
                    f_right = jnp.array([Fx, Fy, Fz])

        return f_left, f_right, tau_hip_roll, {"feasible": True, "reason": "ok"}

    def distribute_wrench(
        self,
        *args,
    ) -> tuple[Array, Array, Array]:
        """Distribute desired 6D wrench to wheel forces and hip roll torques.

        Uses simple heuristics instead of QP optimization:
        - Split vertical force equally between wheels
        - Split sagittal force equally between wheels
        - Use vertical force asymmetry across ±x for My when needed
        - Map Mx directly to hip roll torques

        Args:
            desired_wrench: Desired 6D wrench [Fx, Fy, Fz, Mx, My, Mz]

        Returns:
            Tuple of (f_left, f_right, tau_hip_roll) where:
                - f_left: Left wheel force (3,) [fx, fy, fz]
                - f_right: Right wheel force (3,) [fx, fy, fz]
                - tau_hip_roll: Hip roll torques (2,) [left, right]
        """
        desired_wrench = args[0] if len(args) == 1 else args[1]
        Fx, Fy, Fz, Mx, My, Mz = desired_wrench

        # Split vertical force equally
        fz_per_wheel = Fz / 2.0

        # Split sagittal force equally
        fy_per_wheel = Fy / 2.0

        # Use lateral force differential for roll moment
        # Mx = wheel_spacing * (f_right_x - f_left_x) / 2
        # Assuming wheel_spacing ~0.23m (from robot geometry)
        wheel_spacing = 0.23
        fx_diff = Mx / (wheel_spacing / 2.0)
        fx_left = Fx / 2.0 - fx_diff / 2.0
        fx_right = Fx / 2.0 + fx_diff / 2.0

        # Construct wheel forces
        f_left = jnp.array([fx_left, fy_per_wheel, fz_per_wheel])
        f_right = jnp.array([fx_right, fy_per_wheel, fz_per_wheel])

        # Map roll moment to hip roll torques
        tau_hip_roll = self._roll_moment_to_hip_roll_torque(Mx)

        return f_left, f_right, tau_hip_roll
