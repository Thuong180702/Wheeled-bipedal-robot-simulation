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
        hip_roll_authority_scale: float = 1.0,
    ) -> tuple[Array, Array, Array, dict]:
        """Distribute desired wrench only through active wheel contacts.

        When no contact is detected, still distribute forces as if both wheels
        are in contact. This allows the controller to prepare for landing and
        prevents collapse during brief flight phases.
        """
        Fx, Fy, Fz, Mx, My, Mz = desired_wrench

        hip_roll_authority_scale = float(jnp.clip(hip_roll_authority_scale, 0.0, 1.0))

        if not left_contact and not right_contact:
            wheel_x_offset = 0.135
            tau_hip_roll = self._roll_moment_to_hip_roll_torque(Mx) * hip_roll_authority_scale
            hip_roll_moment_provided = tau_hip_roll[1] - tau_hip_roll[0]
            remaining_mx = Mx - hip_roll_moment_provided
            if abs(remaining_mx) > 0.5:
                fz_diff_desired = remaining_mx / wheel_x_offset
                fz_avg = Fz / 2.0
                liftoff_threshold = 2.0 * (fz_avg - self.min_wheel_force - 5.0)
                max_safe_diff = min(self.max_force_asymmetry, liftoff_threshold)
                fz_diff = jnp.clip(fz_diff_desired, -max_safe_diff, max_safe_diff)
            else:
                fz_diff = 0.0
            f_left = jnp.array([Fx / 2.0, Fy / 2.0, Fz / 2.0 + fz_diff / 2.0])
            f_right = jnp.array([Fx / 2.0, Fy / 2.0, Fz / 2.0 - fz_diff / 2.0])
            return (
                f_left,
                f_right,
                tau_hip_roll,
                {"feasible": True, "reason": "flight_phase_anticipatory"},
            )

        active_count = int(left_contact) + int(right_contact)

        print(f"[FORCE_DIST] left_contact={left_contact}, right_contact={right_contact}, active_count={active_count}")
        print(f"[FORCE_DIST] desired_wrench: Fx={Fx:.2f}, Fy={Fy:.2f}, Fz={Fz:.2f}, Mx={Mx:.2f}, My={My:.2f}, Mz={Mz:.2f}")

        # CRITICAL FIX: Create asymmetric vertical forces to generate roll moment
        # Wheels are positioned at x = ±0.135m (left/right sides), not y-axis
        # Roll moment from cross product: Mx = r_left_x * f_left_z + r_right_x * f_right_z
        # With r_left_x = +0.135 and r_right_x = -0.135:
        # Mx = 0.135 * f_left_z - 0.135 * f_right_z = 0.135 * (f_left_z - f_right_z)
        # Therefore: f_left_z - f_right_z = Mx / 0.135
        wheel_x_offset = 0.135  # meters, actual x-position of wheels from CoM

        if active_count == 2:
            print(f"[FORCE_DIST] CONTACT-AWARE BRANCH: Both wheels in contact")

            # STRATEGY: Prioritize hip roll torques over vertical force asymmetry
            # Hip roll torques can provide strong roll control without risking liftoff
            # Only use vertical force asymmetry when hip torques saturate

            # Positive Mx uses opposite hip-roll signs in this model.
            tau_hip_roll = self._roll_moment_to_hip_roll_torque(Mx) * hip_roll_authority_scale

            # Calculate remaining roll moment that hip torques cannot provide
            # (due to saturation or insufficient authority)
            hip_roll_moment_provided = tau_hip_roll[1] - tau_hip_roll[0]
            remaining_mx = Mx - hip_roll_moment_provided

            # RECOVERY MECHANISM 3: Predictive contact maintenance
            # Only use vertical force asymmetry if hip torques are saturated
            # AND the remaining moment is significant
            # TUNING: Lowered threshold from 1.0 to 0.5 Nm to engage earlier as preventive measure
            if abs(remaining_mx) > 0.5:  # Only if >0.5 Nm remains after hip saturation
                # Use differential vertical forces as SECONDARY/backup for remaining moment
                fz_diff_desired = remaining_mx / wheel_x_offset  # f_left_z - f_right_z

                # SAFETY LIMIT: Constrain force asymmetry to prevent liftoff
                fz_avg = Fz / 2.0
                liftoff_threshold = 2.0 * (fz_avg - self.min_wheel_force - 5.0)  # 5N safety margin
                max_safe_diff = min(
                    self.max_force_asymmetry,
                    liftoff_threshold,  # Never exceed liftoff threshold
                )
                fz_diff = jnp.clip(fz_diff_desired, -max_safe_diff, max_safe_diff)
            else:
                # Hip torques are sufficient - use symmetric vertical forces
                fz_diff = 0.0

            fz_left = Fz / 2.0 + fz_diff / 2.0
            fz_right = Fz / 2.0 - fz_diff / 2.0

            f_left = jnp.array([Fx / 2.0, Fy / 2.0, fz_left])
            f_right = jnp.array([Fx / 2.0, Fy / 2.0, fz_right])
        else:

            # RECOVERY MECHANISM 1: Active contact recovery via hip roll moment
            # Generate strong hip roll torque to tilt robot back toward lifted wheel
            recovery_roll_gain = 5.0  # Amplify roll correction during single contact (increased from 3.5)
            recovery_mx = Mx * recovery_roll_gain

            # RECOVERY MECHANISM 2: Asymmetric leg stiffness
            # Maintain minimum vertical force on lifted leg to keep it extended
            min_recovery_force = 50.0  # N - enough to keep leg extended and generate recovery torque (increased from 35.0)

            if left_contact:
                # Left wheel in contact, right wheel lifted
                f_left = jnp.array([Fx, Fy, Fz])
                f_right = jnp.array([0.0, 0.0, min_recovery_force])  # Keep right leg extended
                tau_hip_roll = self._roll_moment_to_hip_roll_torque(recovery_mx) * hip_roll_authority_scale
            else:  # right_contact
                # Right wheel in contact, left wheel lifted
                f_left = jnp.array([0.0, 0.0, min_recovery_force])  # Keep left leg extended
                f_right = jnp.array([Fx, Fy, Fz])
                tau_hip_roll = self._roll_moment_to_hip_roll_torque(recovery_mx) * hip_roll_authority_scale

        return f_left, f_right, tau_hip_roll, {"feasible": True, "reason": "ok"}

    def distribute_wrench(
        self,
        *args,
    ) -> tuple[Array, Array, Array]:
        """Distribute desired 6D wrench to wheel forces and hip roll torques.

        Uses simple heuristics instead of QP optimization:
        - Split vertical force equally between wheels
        - Split sagittal force equally between wheels
        - Use lateral force for roll moment via differential wheel forces
        - Map roll moment directly to hip roll torques

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
