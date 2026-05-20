"""Integrated whole-body controller with proper force-to-torque mapping.

Combines centroidal wrench computation, force distribution, and Jacobian mapping
to produce joint torques that achieve desired control objectives.
"""

import jax.numpy as jnp
import mujoco
import numpy as np
from jax import Array

WBC_MASKED_LEG_JOINT_INDICES = jnp.array([2, 3, 7, 8])

from wheeled_biped.controllers.centroidal_state_estimator import CentroidalState
from wheeled_biped.controllers.centroidal_wrench_computer import (
    CentroidalWrenchComputer,
)
from wheeled_biped.controllers.contact_jacobian import ContactJacobian
from wheeled_biped.controllers.simple_force_distributor import SimpleForceDistributor


class IntegratedWBC:
    """Integrated whole-body controller with proper Jacobian-based force mapping."""

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        k_roll: float = 200.0,
        k_roll_rate: float = 40.0,
        k_roll_integral: float = 100.0,
        k_pitch: float = 5000.0,
        k_pitch_rate: float = 100.0,
        k_com_lateral: float = 15.0,
        k_com_lateral_damping: float = 3.0,
        k_com_sagittal: float = 10.0,
        k_com_sagittal_damping: float = 2.0,
        k_cp_lateral: float = 25.0,
        k_cp_sagittal: float = 20.0,
        k_height: float = 5.0,
        robot_mass: float = 15.0,
        gravity: float = 9.81,
        max_roll_moment: float | None = None,
        wbc_authority_budget: float = 0.6,
        max_actuator_torque: float = 30.0,
        force_feedback_gain: float = 1.5,
        force_feedback_warmup_steps: int = 5,
        tau_hip_roll_max: float = 30.0,
        max_force_asymmetry: float = 40.0,
        min_wheel_force: float = 10.0,
        roll_integral_limit: float = 0.52,
        dt: float = 0.01,
    ):
        """Initialize integrated WBC.

        Args:
            mj_model: MuJoCo model with robot definition
            k_roll: Roll stabilization gain (proportional, increased to 20.0)
            k_roll_rate: Roll rate damping gain (derivative, increased to 4.0)
            k_roll_integral: Roll integral gain (eliminates steady-state error)
            k_pitch: Pitch stabilization gain
            k_pitch_rate: Pitch rate damping gain
            k_com_lateral: CoM lateral position gain
            k_com_lateral_damping: CoM lateral velocity damping
            k_com_sagittal: CoM sagittal position gain
            k_com_sagittal_damping: CoM sagittal velocity damping
            k_cp_lateral: Capture point lateral gain
            k_cp_sagittal: Capture point sagittal gain
            k_height: Height tracking gain
            robot_mass: Robot mass in kg
            gravity: Gravity constant
            max_roll_moment: Optional roll moment clamp in Nm
            wbc_authority_budget: Authority budget as fraction (0.0-1.0)
            max_actuator_torque: Maximum actuator torque in Nm
            force_feedback_gain: Gain for closed-loop force correction (1.0 = no correction)
            force_feedback_warmup_steps: Number of steps to skip force feedback (avoid mj_forward artifacts)
            tau_hip_roll_max: Maximum hip roll torque for force distributor (increased to 30.0 Nm)
            max_force_asymmetry: Maximum allowed vertical force difference (increased to 40.0 N)
            min_wheel_force: Minimum safe wheel force to prevent liftoff (reduced to 10.0 N)
            roll_integral_limit: Anti-windup limit for roll integral (radians, ~30 degrees)
            dt: Control timestep in seconds
        """
        self.mj_model = mj_model
        self.wbc_authority_budget = wbc_authority_budget
        self.max_actuator_torque = max_actuator_torque
        self.force_feedback_gain = force_feedback_gain
        self.force_feedback_warmup_steps = force_feedback_warmup_steps
        self.step_count = 0
        self.dt = dt
        self.roll_integral_limit = roll_integral_limit

        # PID state: roll integral accumulator with anti-windup
        self.roll_integral = 0.0

        # Initialize components
        self.wrench_computer = CentroidalWrenchComputer(
            k_roll=k_roll,
            k_roll_rate=k_roll_rate,
            k_roll_integral=k_roll_integral,
            k_pitch=k_pitch,
            k_pitch_rate=k_pitch_rate,
            k_com_lateral=k_com_lateral,
            k_com_lateral_damping=k_com_lateral_damping,
            k_com_sagittal=k_com_sagittal,
            k_com_sagittal_damping=k_com_sagittal_damping,
            k_cp_lateral=k_cp_lateral,
            k_cp_sagittal=k_cp_sagittal,
            k_height=k_height,
            robot_mass=robot_mass,
            gravity=gravity,
            max_roll_moment=max_roll_moment,
        )
        self.force_distributor = SimpleForceDistributor(
            tau_hip_roll_max=tau_hip_roll_max,
            max_force_asymmetry=max_force_asymmetry,
            min_wheel_force=min_wheel_force,
        )
        self.contact_jacobian = ContactJacobian(mj_model)

        # Find wheel body IDs for position computation
        # Note: ContactJacobian also looks up these IDs for Jacobian computation.
        # This duplication is intentional - each component uses the IDs for different
        # purposes and maintains its own state independently.
        self.l_wheel_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link"
        )
        self.r_wheel_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link"
        )

        # Validate body IDs were found
        if self.l_wheel_id == -1 or self.r_wheel_id == -1:
            raise ValueError(
                "Wheel body IDs not found in model. Expected 'l_wheel_link' and 'r_wheel_link'."
            )

    def _build_direct_hip_roll_torque(self, tau_hip_roll: Array) -> Array:
        tau = jnp.zeros(10, dtype=tau_hip_roll.dtype)
        tau = tau.at[0].set(tau_hip_roll[0])
        tau = tau.at[5].set(tau_hip_roll[1])
        return tau

    def compute_wbc_torque(
        self,
        mj_data: mujoco.MjData,
        obs: Array,
        state: CentroidalState,
        height_cmd: float,
    ) -> Array:
        """Compute WBC joint torques via unified QP force distribution."""
        tau_wbc, _ = self.compute_wbc_torque_with_diagnostics(
            mj_data, obs, state, height_cmd
        )
        return tau_wbc

    def compute_wbc_torque_with_diagnostics(
        self,
        mj_data: mujoco.MjData,
        obs: Array,
        state: CentroidalState,
        height_cmd: float,
    ) -> tuple[Array, dict]:
        """Compute WBC joint torques and diagnostics via unified force distribution.

        Integration flow:
        1. Update roll integral with anti-windup
        2. CentroidalWrenchComputer: obs + state + roll_integral → desired_wrench (6D)
        3. Compute wheel positions relative to CoM from MuJoCo data
        4. UnifiedForceDistributor: QP solve → (f_left, f_right, tau_hip_roll)
        5. ContactJacobian: forces + hip torques → joint torques (10D)
        6. Clip to authority budget

        Args:
            mj_data: MuJoCo data with current robot state
            obs: Observation array
            state: CentroidalState with CoM, capture point, etc.
            height_cmd: Desired height command

        Returns:
            Tuple of (tau_wbc, diagnostics) where:
                - tau_wbc: Joint torques (10,) that achieve desired control objectives
                - diagnostics: Dict with QP solver metrics
        """
        import time

        _ = obs

        # Update roll integral with anti-windup
        # Only accumulate when roll is small (< 15 degrees) to prevent windup during falls
        roll_rad = float(state.roll)
        if abs(roll_rad) < self.roll_integral_limit:
            self.roll_integral += roll_rad * self.dt
            # Clamp integral to prevent excessive accumulation
            self.roll_integral = float(jnp.clip(self.roll_integral, -self.roll_integral_limit, self.roll_integral_limit))
        else:
            # Reset integral when roll is too large (robot falling)
            self.roll_integral = 0.0

        print(f"[PID STATE] roll={roll_rad*180/3.14159:.2f}°, roll_integral={self.roll_integral:.4f}, integral_contribution={self.wrench_computer.k_roll_integral * self.roll_integral:.2f} Nm")

        desired_force, desired_moment = self.wrench_computer.compute_desired_wrench_from_state(
            state, height_cmd, self.roll_integral
        )
        desired_wrench = jnp.concatenate([desired_force, desired_moment])

        # CRITICAL FIX: Extract sagittal force (Fy) for diagnostics
        # The contact Jacobian will convert this into wheel torques automatically
        Fy_total = desired_wrench[1]  # Sagittal force (forward/backward)

        # Compute expected wheel torques for diagnostics (not used in control)
        wheel_radius = 0.06  # meters (from robot XML: wheel geom size="0.06 0.025")
        left_contact = bool(state.left_wheel_contact)
        right_contact = bool(state.right_wheel_contact)
        active_wheels = int(left_contact) + int(right_contact)

        if active_wheels == 0:
            tau_wheel_expected = Fy_total * wheel_radius / 2.0
        elif active_wheels == 1:
            tau_wheel_expected = Fy_total * wheel_radius
        else:
            tau_wheel_expected = Fy_total * wheel_radius / 2.0

        print(f"[WHEEL TORQUE DIAGNOSTIC] Fy_total={Fy_total:.4f} N, active_wheels={active_wheels}, expected_wheel_torque={tau_wheel_expected:.4f} Nm (Jacobian will produce this)")

        # CRITICAL FIX: Do NOT add wheel torques separately
        # The contact Jacobian ALREADY converts sagittal force Fy into wheel torques
        # Adding them separately causes cancellation: Jacobian produces -0.0456 Nm,
        # then we add +0.0456 Nm, resulting in zero net wheel torque
        #
        # The force distributor needs Fy to create pitch control through:
        # 1. Contact Jacobian mapping Fy → wheel torques (this is sufficient)
        # 2. Differential vertical forces → leg joint torques for pitch stabilization
        #
        # Previously we thought we needed to add wheel torques directly because they
        # were "100-1000x too small", but that was due to the wheel radius mismatch
        # (0.05 vs 0.06 m) and this double-counting cancellation bug.

        solve_start = time.perf_counter()
        f_left, f_right, tau_hip_roll, distribution_diagnostics = (
            self.force_distributor.distribute_wrench_contact_aware(
                desired_wrench,  # Use full wrench including Fy
                left_contact=bool(state.left_wheel_contact),
                right_contact=bool(state.right_wheel_contact),
            )
        )
        solve_time_ms = (time.perf_counter() - solve_start) * 1000.0

        # Keep contact-force sign separate from direct hip-roll torque sign.
        tau_contact = -self.contact_jacobian.map_contact_forces_to_torques(
            mj_data, f_left, f_right, tau_hip_roll=None
        )
        tau_hip = self._build_direct_hip_roll_torque(tau_hip_roll)
        tau_wbc_raw = tau_contact - tau_hip

        # REMOVED: Direct wheel torque addition (was causing cancellation bug)
        # The contact Jacobian already converts sagittal force Fy into wheel torques
        # Adding them separately caused: Jacobian produces -0.0456 Nm, we add +0.0456 Nm → zero net torque
        #
        # Previous code (BUGGY):
        # tau_wbc_raw = tau_wbc_raw.at[4].add(tau_wheel_left)
        # tau_wbc_raw = tau_wbc_raw.at[9].add(tau_wheel_right)

        # FIXED: Remove masking to allow WBC to command all joints including knee
        # The posture controller provides no feedforward gravity compensation,
        # so WBC must handle all joints to prevent collapse
        tau_wbc_masked = tau_wbc_raw  # Use full WBC torques without masking

        actual_fz_total = float(state.total_contact_force_z)
        desired_fz_total = float(f_left[2] + f_right[2])
        contact_force_valid = bool(getattr(state, "contact_force_valid", False))

        # Force feedback control: scale torque based on force error
        # CRITICAL: Skip force feedback during warmup to avoid reacting to mj_forward artifacts
        # At t=0, mj_forward produces large penetration forces (143N) that are not real
        # After mj_step, actual forces drop to ~18N, but controller already reduced torque
        # This delays response during critical first timesteps when robot is unstable
        if self.step_count < self.force_feedback_warmup_steps:
            force_scale = 1.0
            print(f"[FORCE FEEDBACK] Warmup step {self.step_count}/{self.force_feedback_warmup_steps}, scale=1.0 (no correction)")
        elif not contact_force_valid:
            force_scale = 1.0
            print("[FORCE FEEDBACK] Contact force invalid before first mj_step, scale=1.0 (no correction)")
        elif desired_fz_total > 1e-3:  # Avoid division by zero
            force_error_ratio = (actual_fz_total - desired_fz_total) / desired_fz_total
            # force_scale = 1.0 means no correction
            # force_scale < 1.0 means reduce torque (actual > desired)
            # force_scale > 1.0 means increase torque (actual < desired)
            force_scale = 1.0 - self.force_feedback_gain * force_error_ratio
            force_scale = float(jnp.clip(force_scale, 0.1, 2.0))  # Limit scale range
            print(f"[FORCE FEEDBACK] Active: actual={actual_fz_total:.1f}N, desired={desired_fz_total:.1f}N, scale={force_scale:.3f}")
        else:
            force_scale = 1.0

        self.step_count += 1
        tau_before_clip = tau_wbc_masked * force_scale
        print(f"[WBC PIPELINE] Before clipping - wheel torques: L={tau_before_clip[4]:.4f} Nm, R={tau_before_clip[9]:.4f} Nm, max_all={jnp.max(jnp.abs(tau_before_clip)):.4f} Nm")
        tau_wbc = self.clip_to_authority_budget(tau_before_clip)
        print(f"[WBC PIPELINE] After authority clipping: {tau_wbc}")
        print(f"[WBC PIPELINE] After clipping - wheel torques: L={tau_wbc[4]:.4f} Nm, R={tau_wbc[9]:.4f} Nm")
        print(f"[WBC PIPELINE] Max final torque: {jnp.max(jnp.abs(tau_wbc)):.4f} Nm")

        wheel_pos_left, wheel_pos_right = self._compute_wheel_positions_relative_to_com(
            mj_data, state.com_pos
        )
        solution = jnp.concatenate([f_left, f_right, tau_hip_roll])
        A_wrench = self.contact_jacobian.build_wrench_matrix(
            mj_data, wheel_pos_left, wheel_pos_right
        )
        achieved_wrench = A_wrench @ solution
        wrench_error = desired_wrench - achieved_wrench
        wrench_error_norm = float(jnp.linalg.norm(wrench_error))

        diagnostics = {
            "solve_time_ms": solve_time_ms,
            "wrench_error_norm": wrench_error_norm,
            "f_left_z": float(f_left[2]),
            "f_right_z": float(f_right[2]),
            "desired_wrench_Fx": float(desired_wrench[0]),
            "desired_wrench_Fy": float(desired_wrench[1]),
            "desired_wrench_Fz": float(desired_wrench[2]),
            "desired_wrench_Mx": float(desired_wrench[3]),
            "desired_wrench_My": float(desired_wrench[4]),
            "desired_wrench_Mz": float(desired_wrench[5]),
            "f_left": f_left,
            "f_right": f_right,
            "tau_hip_roll": tau_hip_roll,
            "actual_fz_total": actual_fz_total,
            "desired_fz_total": desired_fz_total,
            "force_scale": float(force_scale),
            "contact_force_valid": contact_force_valid,
            "left_contact_active": bool(state.left_wheel_contact),
            "right_contact_active": bool(state.right_wheel_contact),
            "left_contact_force_world": state.left_contact_force_world,
            "right_contact_force_world": state.right_contact_force_world,
            "total_contact_force_z": actual_fz_total,
            "force_distribution_feasible": bool(distribution_diagnostics["feasible"]),
            "force_distribution_reason": distribution_diagnostics["reason"],
            "distributed_left_fx": float(f_left[0]),
            "distributed_left_fy": float(f_left[1]),
            "distributed_left_fz": float(f_left[2]),
            "distributed_right_fx": float(f_right[0]),
            "distributed_right_fy": float(f_right[1]),
            "distributed_right_fz": float(f_right[2]),
            "roll_integral": self.roll_integral,
            "roll_integral_contribution_Nm": self.wrench_computer.k_roll_integral * self.roll_integral,
        }

        return tau_wbc, diagnostics

    def _compute_wheel_positions_relative_to_com(
        self,
        mj_data: mujoco.MjData,
        com_pos: Array,
    ) -> tuple[Array, Array]:
        """Compute wheel positions relative to CoM from MuJoCo data.

        Args:
            mj_data: MuJoCo data with current robot state
            com_pos: Center of mass position (3,) in world frame

        Returns:
            Tuple of (wheel_pos_left, wheel_pos_right) where each is (3,) [x, y, z]
            relative to CoM in world frame
        """
        # Get wheel body positions from MuJoCo (world frame)
        l_wheel_pos_world = np.array(mj_data.xpos[self.l_wheel_id])
        r_wheel_pos_world = np.array(mj_data.xpos[self.r_wheel_id])

        # Convert to JAX arrays and compute relative positions
        com_pos_np = np.array(com_pos)
        wheel_pos_left = jnp.array(l_wheel_pos_world - com_pos_np)
        wheel_pos_right = jnp.array(r_wheel_pos_world - com_pos_np)

        return wheel_pos_left, wheel_pos_right

    def _measure_total_vertical_contact_force(self, mj_data: mujoco.MjData) -> float:
        """Measure total vertical contact force from MuJoCo contacts.

        Args:
            mj_data: MuJoCo data with current contact state

        Returns:
            Total vertical contact force in N
        """
        total_fz = 0.0
        for i in range(mj_data.ncon):
            if i < len(mj_data.efc_force):
                total_fz += mj_data.efc_force[i]
        return total_fz

    def clip_to_authority_budget(self, tau: Array) -> Array:
        """Clip torque to WBC authority budget.

        Args:
            tau: Desired torque array (10,)

        Returns:
            Clipped torque array (10,) within authority budget
        """
        budget_limit = self.wbc_authority_budget * self.max_actuator_torque

        max_tau = jnp.max(jnp.abs(tau))
        scale_factor = jnp.where(max_tau <= budget_limit, 1.0, budget_limit / max_tau)
        tau_clipped = tau * scale_factor

        return tau_clipped

    def reset_integral(self):
        """Reset integral state (call when robot resets or falls)."""
        self.roll_integral = 0.0
        self.step_count = 0
