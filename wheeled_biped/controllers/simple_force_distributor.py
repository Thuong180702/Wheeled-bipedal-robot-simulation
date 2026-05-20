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

    def distribute_wrench_contact_aware(
        self,
        desired_wrench: Array,
        left_contact: bool,
        right_contact: bool,
    ) -> tuple[Array, Array, Array, dict]:
        """Distribute desired wrench only through active wheel contacts.

        When no contact is detected, still distribute forces as if both wheels
        are in contact. This allows the controller to prepare for landing and
        prevents collapse during brief flight phases.
        """
        Fx, Fy, Fz, Mx, My, Mz = desired_wrench

        if not left_contact and not right_contact:
            # FIXED: Don't return zero torque during flight phase
            # Distribute forces as if both wheels will contact soon
            # This prevents collapse when robot briefly loses contact
            f_left = jnp.array([Fx / 2.0, Fy / 2.0, Fz / 2.0])
            f_right = jnp.array([Fx / 2.0, Fy / 2.0, Fz / 2.0])
            tau_hip_roll = jnp.clip(
                jnp.array([Mx / 2.0, Mx / 2.0]),
                -self.tau_hip_roll_max,
                self.tau_hip_roll_max,
            )
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

            # CRITICAL FIX: Hip roll torques must have SAME SIGN to generate roll moment
            # For a wheeled biped, both hip roll actuators work together:
            #   - To correct positive roll (tilting right): both hips apply positive torque
            #   - To correct negative roll (tilting left): both hips apply negative torque
            # This creates a net roll moment about the CoM

            # CRITICAL: Calculate remaining moment BEFORE clipping
            # If we calculate after clipping, the clipped torques always sum to match demand,
            # so remaining_mx is always ~0 even when saturated
            tau_hip_roll_desired = jnp.array([Mx / 2.0, Mx / 2.0])

            # DEBUG: Print clipping parameters
            print(f"[CLIP DEBUG] tau_hip_roll_max={self.tau_hip_roll_max}")
            print(f"[CLIP DEBUG] tau_hip_roll_desired={tau_hip_roll_desired}")
            print(f"[CLIP DEBUG] clip_bounds=[-{self.tau_hip_roll_max}, +{self.tau_hip_roll_max}]")

            tau_hip_roll = jnp.clip(
                tau_hip_roll_desired,
                -self.tau_hip_roll_max,
                self.tau_hip_roll_max,
            )

            print(f"[CLIP DEBUG] tau_hip_roll_after_clip={tau_hip_roll}")

            # Calculate remaining roll moment that hip torques cannot provide
            # (due to saturation or insufficient authority)
            # Note: Both hip torques have same sign, so they add up to provide total moment
            hip_roll_moment_provided = tau_hip_roll[0] + tau_hip_roll[1]
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
                liftoff_threshold = fz_avg - self.min_wheel_force - 5.0  # 5N safety margin
                max_safe_diff = min(
                    self.max_force_asymmetry,
                    liftoff_threshold,  # Never exceed liftoff threshold
                )
                fz_diff = jnp.clip(fz_diff_desired, -max_safe_diff, max_safe_diff)

                if abs(fz_diff_desired) > max_safe_diff:
                    print(f"[FORCE_DIST] LIFTOFF PREVENTION: reducing asymmetry from {fz_diff_desired:.2f} to {fz_diff:.2f}")
            else:
                # Hip torques are sufficient - use symmetric vertical forces
                fz_diff = 0.0

            fz_left = Fz / 2.0 + fz_diff / 2.0
            fz_right = Fz / 2.0 - fz_diff / 2.0

            print(f"[FORCE_DIST] Mx={Mx:.2f}, hip_moment={hip_roll_moment_provided:.2f}, remaining={remaining_mx:.2f}")
            print(f"[FORCE_DIST] fz_diff={fz_diff:.2f}, fz_left={fz_left:.2f}, fz_right={fz_right:.2f}")

            f_left = jnp.array([Fx / 2.0, Fy / 2.0, fz_left])
            f_right = jnp.array([Fx / 2.0, Fy / 2.0, fz_right])

            print(f"[FORCE_DIST] f_left={f_left}, f_right={f_right}")
            print(f"[FORCE_DIST] tau_hip_roll={tau_hip_roll}")
        else:
            print(f"[FORCE_DIST] SINGLE CONTACT RECOVERY: active_count={active_count}")

            # RECOVERY MECHANISM 1: Active contact recovery via hip roll moment
            # Generate strong hip roll torque to tilt robot back toward lifted wheel
            recovery_roll_gain = 3.5  # Amplify roll correction during single contact (increased from 2.0)
            recovery_mx = Mx * recovery_roll_gain

            # RECOVERY MECHANISM 2: Asymmetric leg stiffness
            # Maintain minimum vertical force on lifted leg to keep it extended
            min_recovery_force = 35.0  # N - enough to keep leg extended and generate recovery torque (increased from 25.0)

            if left_contact:
                # Left wheel in contact, right wheel lifted
                f_left = jnp.array([Fx, Fy, Fz])
                f_right = jnp.array([0.0, 0.0, min_recovery_force])  # Keep right leg extended
                tau_hip_roll = jnp.clip(
                    jnp.array([recovery_mx / 2.0, recovery_mx / 2.0]),  # Both hips work to recover
                    -self.tau_hip_roll_max,
                    self.tau_hip_roll_max,
                )
                print(f"[FORCE_DIST] LEFT CONTACT ONLY: f_right_z={min_recovery_force:.2f}N (recovery), recovery_mx={recovery_mx:.2f}")
            else:  # right_contact
                # Right wheel in contact, left wheel lifted
                f_left = jnp.array([0.0, 0.0, min_recovery_force])  # Keep left leg extended
                f_right = jnp.array([Fx, Fy, Fz])
                tau_hip_roll = jnp.clip(
                    jnp.array([recovery_mx / 2.0, recovery_mx / 2.0]),  # Both hips work to recover
                    -self.tau_hip_roll_max,
                    self.tau_hip_roll_max,
                )
                print(f"[FORCE_DIST] RIGHT CONTACT ONLY: f_left_z={min_recovery_force:.2f}N (recovery), recovery_mx={recovery_mx:.2f}")

        print(f"[FORCE_DIST] FINAL OUTPUT: f_left_z={f_left[2]:.2f}, f_right_z={f_right[2]:.2f}, asymmetry={abs(f_left[2] - f_right[2]):.2f}")
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
        # Each hip roll contributes directly to roll moment
        tau_hip_roll_per_side = Mx / 2.0
        tau_hip_roll = jnp.clip(
            jnp.array([tau_hip_roll_per_side, tau_hip_roll_per_side]),
            -self.tau_hip_roll_max,
            self.tau_hip_roll_max,
        )

        return f_left, f_right, tau_hip_roll
