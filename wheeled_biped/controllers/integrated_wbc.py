"""Integrated whole-body controller with proper force-to-torque mapping.

Combines centroidal wrench computation, force distribution, and Jacobian mapping
to produce joint torques that achieve desired control objectives.
"""

import jax.numpy as jnp
import mujoco
import numpy as np
from jax import Array

from wheeled_biped.controllers.centroidal_state_estimator import CentroidalState
from wheeled_biped.controllers.centroidal_wrench_computer import CentroidalWrenchComputer
from wheeled_biped.controllers.contact_jacobian import ContactJacobian
from wheeled_biped.controllers.unified_force_distributor import UnifiedForceDistributor


class IntegratedWBC:
    """Integrated whole-body controller with proper Jacobian-based force mapping."""

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        k_roll: float = 20.0,
        k_roll_rate: float = 4.0,
        k_com_lateral: float = 15.0,
        k_com_lateral_damping: float = 3.0,
        k_com_sagittal: float = 10.0,
        k_com_sagittal_damping: float = 2.0,
        k_cp_lateral: float = 25.0,
        k_cp_sagittal: float = 20.0,
        k_height: float = 5.0,
        robot_mass: float = 15.0,
        gravity: float = 9.81,
        wbc_authority_budget: float = 0.6,
        max_actuator_torque: float = 30.0,
    ):
        """Initialize integrated WBC.

        Args:
            mj_model: MuJoCo model with robot definition
            k_roll: Roll stabilization gain
            k_roll_rate: Roll rate damping gain
            k_com_lateral: CoM lateral position gain
            k_com_lateral_damping: CoM lateral velocity damping
            k_com_sagittal: CoM sagittal position gain
            k_com_sagittal_damping: CoM sagittal velocity damping
            k_cp_lateral: Capture point lateral gain
            k_cp_sagittal: Capture point sagittal gain
            k_height: Height tracking gain
            robot_mass: Robot mass in kg
            gravity: Gravity constant
            wbc_authority_budget: Authority budget as fraction (0.0-1.0)
            max_actuator_torque: Maximum actuator torque in Nm
        """
        self.mj_model = mj_model
        self.wbc_authority_budget = wbc_authority_budget
        self.max_actuator_torque = max_actuator_torque

        # Initialize components
        self.wrench_computer = CentroidalWrenchComputer(
            k_roll=k_roll,
            k_roll_rate=k_roll_rate,
            k_com_lateral=k_com_lateral,
            k_com_lateral_damping=k_com_lateral_damping,
            k_com_sagittal=k_com_sagittal,
            k_com_sagittal_damping=k_com_sagittal_damping,
            k_cp_lateral=k_cp_lateral,
            k_cp_sagittal=k_cp_sagittal,
            k_height=k_height,
            robot_mass=robot_mass,
            gravity=gravity,
        )
        self.force_distributor = UnifiedForceDistributor(mj_model)
        self.contact_jacobian = ContactJacobian(mj_model)

        # Find wheel body IDs for position computation
        # Note: ContactJacobian also looks up these IDs for Jacobian computation.
        # This duplication is intentional - each component uses the IDs for different
        # purposes and maintains its own state independently.
        self.l_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
        self.r_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")

        # Validate body IDs were found
        if self.l_wheel_id == -1 or self.r_wheel_id == -1:
            raise ValueError("Wheel body IDs not found in model. Expected 'l_wheel_link' and 'r_wheel_link'.")

    def compute_wbc_torque(
        self,
        mj_data: mujoco.MjData,
        obs: Array,
        state: CentroidalState,
        height_cmd: float,
    ) -> tuple[Array, dict]:
        """Compute WBC joint torques via unified QP force distribution.

        Integration flow:
        1. CentroidalWrenchComputer: obs + state → desired_wrench (6D)
        2. Compute wheel positions relative to CoM from MuJoCo data
        3. UnifiedForceDistributor: QP solve → (f_left, f_right, tau_hip_roll)
        4. ContactJacobian: forces + hip torques → joint torques (10D)
        5. Clip to authority budget

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

        # Step 1: Compute desired centroidal wrench (6D vector)
        desired_wrench = self.wrench_computer.compute_desired_wrench_vector(
            obs, state, height_cmd
        )

        # Step 2: Compute wheel positions relative to CoM
        wheel_pos_left, wheel_pos_right = self._compute_wheel_positions_relative_to_com(
            mj_data, state.com_pos
        )

        # Step 3: Unified QP force distribution (with timing)
        solve_start = time.perf_counter()
        f_left, f_right, tau_hip_roll = self.force_distributor.distribute_wrench(
            mj_data, desired_wrench, wheel_pos_left, wheel_pos_right
        )
        solve_time_ms = (time.perf_counter() - solve_start) * 1000.0

        # Step 4: Map contact forces and hip roll torques to joint torques via Jacobians
        tau_wbc = self.contact_jacobian.map_contact_forces_to_torques(
            mj_data, f_left, f_right, tau_hip_roll
        )

        # Step 5: Clip to authority budget
        tau_wbc = self.clip_to_authority_budget(tau_wbc)

        # Compute diagnostics
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
