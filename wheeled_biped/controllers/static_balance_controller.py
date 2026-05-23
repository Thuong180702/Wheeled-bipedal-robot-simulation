"""Static balance controller wrapper.

Cancels WBC static equilibrium bias by computing static reference torques
once at initialization, then removing equilibrium bias at runtime.
"""

import jax.numpy as jnp
import mujoco
import numpy as np
from jax import Array
from numpy.typing import NDArray

# Import existing calibration helper from simulate_hierarchical_controller
# Do not duplicate - reuse the tested implementation
from scripts.simulate_hierarchical_controller import calibrate_root_z_for_wheel_floor_contact
from wheeled_biped.controllers.centroidal_state_estimator import CentroidalState


class StaticBalanceController:
    """Wrapper that cancels WBC static equilibrium bias."""

    # Observation dimension constant
    OBS_DIM = 42

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        wbc_pipeline,
        calibration_config: dict | None = None,
    ):
        """Initialize with calibrated equilibrium references.

        Args:
            mj_model: MuJoCo model
            mj_data: MuJoCo data (will be copied, not mutated)
            wbc_pipeline: Existing WBC pipeline to wrap
            calibration_config: Config for calibrated initialization
        """
        self.mj_model = mj_model
        self.wbc_pipeline = wbc_pipeline
        self.calibration_config = calibration_config or {}

        # Will be computed in initialization
        self.tau_static_ref = None
        self.tau_wbc_equilibrium = None
        self.equilibrium_state = None
        self.qfrc_inverse_ref = None
        self.qfrc_bias_ref = None
        self.qfrc_constraint_ref = None
        self.geom_ids = None

        # Compute references using copied data
        self._compute_equilibrium_references(mj_data)

    def _compute_equilibrium_references(self, mj_data: mujoco.MjData) -> None:
        """Compute static reference torques at calibrated equilibrium.

        This method:
        1. Copies MuJoCo data to avoid mutation
        2. Runs calibrated initialization sequence (5 steps)
        3. Captures equilibrium state with proper orientation
        4. Computes tau_static_ref using inverse dynamics
        5. Computes tau_wbc_equilibrium using WBC at zero-error state
        6. Logs initialization diagnostics
        """
        from wheeled_biped.controllers.orientation_utils import (
            compute_robot_frame_orientation_from_quaternion,
        )

        print("\n" + "="*80)
        print("STATIC BALANCE CONTROLLER INITIALIZATION")
        print("="*80)

        # Step 1: Copy MuJoCo data to avoid mutation
        data_copy = mujoco.MjData(self.mj_model)
        data_copy.qpos[:] = mj_data.qpos
        data_copy.qvel[:] = mj_data.qvel
        data_copy.ctrl[:] = mj_data.ctrl

        print("\n[STEP 1] MuJoCo data copied")
        print(f"  Initial qpos[2] (root_z): {data_copy.qpos[2]:.6f} m")

        # Step 2: Run calibrated initialization sequence (5 steps)
        max_iters = self.calibration_config.get("max_iters", 5)
        print(f"\n[STEP 2] Running calibrated initialization ({max_iters} steps)...")
        self.geom_ids = calibrate_root_z_for_wheel_floor_contact(
            self.mj_model,
            data_copy,
            max_iters=max_iters,
        )
        print(f"  Calibrated qpos[2] (root_z): {data_copy.qpos[2]:.6f} m")

        # Step 3: Capture equilibrium state with proper orientation
        print("\n[STEP 3] Capturing equilibrium state...")

        # Extract quaternion and compute orientation
        quat = data_copy.qpos[3:7]  # [w, x, y, z]
        pitch_x, roll_y, yaw_z = compute_robot_frame_orientation_from_quaternion(quat)

        print(f"  Quaternion: [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]")
        print(f"  Orientation: pitch_x={pitch_x*180/np.pi:.4f}°, roll_y={roll_y*180/np.pi:.4f}°, yaw_z={yaw_z*180/np.pi:.4f}°")

        # Store equilibrium state
        com_z = self._compute_com_z(data_copy)
        self.equilibrium_state = {
            'qpos': data_copy.qpos.copy(),
            'qvel': data_copy.qvel.copy(),
            'pitch_x': pitch_x,
            'roll_y': roll_y,
            'yaw_z': yaw_z,
            'com_z': com_z,
            'geom_ids': self.geom_ids,
        }

        print(f"  CoM height: {self.equilibrium_state['com_z']:.6f} m")
        print(f"  Joint positions (support): {data_copy.qpos[[9, 10, 14, 15]]}")  # [l_hip_pitch, l_knee, r_hip_pitch, r_knee]

        # Step 4: Compute tau_static_ref using inverse dynamics
        print("\n[STEP 4] Computing static reference torques via inverse dynamics...")

        # Set zero acceleration for static equilibrium
        data_copy.qacc[:] = 0.0

        # Compute inverse dynamics: tau = M*qacc + C(q,qvel) + g(q)
        # With qacc=0: tau = C(q,qvel) + g(q)
        # With qvel=0: tau = g(q) (pure gravity compensation)
        mujoco.mj_inverse(self.mj_model, data_copy)

        # Extract joint torques (indices 6:16 in qfrc_inverse correspond to 10 actuated joints)
        self.tau_static_ref = data_copy.qfrc_inverse[6:16].copy()

        # Validate that computed reference is finite
        if not np.all(np.isfinite(self.tau_static_ref)):
            raise ValueError(f"tau_static_ref contains NaN or Inf: {self.tau_static_ref}")

        # Store additional inverse dynamics components for diagnostics
        self.qfrc_inverse_ref = data_copy.qfrc_inverse.copy()
        self.qfrc_bias_ref = data_copy.qfrc_bias.copy()

        # Store constraint forces if they exist
        if hasattr(data_copy, 'qfrc_constraint') and data_copy.qfrc_constraint is not None:
            self.qfrc_constraint_ref = data_copy.qfrc_constraint.copy()
        else:
            self.qfrc_constraint_ref = None

        print(f"  tau_static_ref (support joints [2,3,7,8]): {self.tau_static_ref[[2,3,7,8]]}")
        print(f"  tau_static_ref (all joints): {self.tau_static_ref}")
        print(f"  Max |tau_static_ref|: {np.max(np.abs(self.tau_static_ref)):.4f} Nm")

        # Step 5: Compute tau_wbc_equilibrium using WBC at zero-error state
        print("\n[STEP 5] Computing WBC equilibrium bias at zero-error state...")

        # Build zero-error observation
        obs_equilibrium = self._build_zero_error_observation(data_copy)

        print(f"  Zero-error observation built (dim={len(obs_equilibrium)})")
        print(f"    Gravity body frame: {obs_equilibrium[0:3]}")
        print(f"    Joint positions: {obs_equilibrium[6:16]}")
        print(f"    Joint velocities: {obs_equilibrium[16:26]}")
        print(f"    Height command: {obs_equilibrium[36]:.6f} m")
        print(f"    Current height: {obs_equilibrium[37]:.6f} m")

        # Compute WBC torque at equilibrium
        self.tau_wbc_equilibrium = self._compute_wbc_at_equilibrium(data_copy, obs_equilibrium)

        # Validate that computed reference is finite
        if not np.all(np.isfinite(self.tau_wbc_equilibrium)):
            raise ValueError(f"tau_wbc_equilibrium contains NaN or Inf: {self.tau_wbc_equilibrium}")

        print(f"  tau_wbc_equilibrium (support joints [2,3,7,8]): {self.tau_wbc_equilibrium[[2,3,7,8]]}")
        print(f"  tau_wbc_equilibrium (all joints): {self.tau_wbc_equilibrium}")
        print(f"  Max |tau_wbc_equilibrium|: {np.max(np.abs(self.tau_wbc_equilibrium)):.4f} Nm")

        # Step 6: Log initialization diagnostics
        self._log_initialization()

        print("\n" + "="*80)
        print("INITIALIZATION COMPLETE")
        print("="*80 + "\n")

    def _compute_com_z(self, mj_data: mujoco.MjData) -> float:
        """Compute center of mass height.

        Args:
            mj_data: MuJoCo data

        Returns:
            CoM height in meters
        """
        return float(mj_data.subtree_com[0, 2])

    def _log_initialization(self) -> None:
        """Log initialization diagnostics with support bias analysis."""
        print("\n[STEP 6] Initialization diagnostics:")
        print(f"  Static equilibrium bias (support joints): {self.tau_wbc_equilibrium[[2,3,7,8]] - self.tau_static_ref[[2,3,7,8]]}")
        print(f"  Static equilibrium bias (all joints): {self.tau_wbc_equilibrium - self.tau_static_ref}")
        print(f"  Max |bias|: {np.max(np.abs(self.tau_wbc_equilibrium - self.tau_static_ref)):.4f} Nm")

    def _build_zero_error_observation(self, mj_data: mujoco.MjData) -> Array:
        """Build zero-error observation for WBC equilibrium computation.

        Observation structure (42-dim):
        - [0:3]: gravity in body frame (should be [0, 0, -9.81] at equilibrium)
        - [3:6]: body linear velocity (zeros at equilibrium)
        - [6:16]: joint positions (10 actuated joints)
        - [16:26]: joint velocities (zeros at equilibrium)
        - [26:36]: previous action (zeros at equilibrium)
        - [36]: height command (current CoM height)
        - [37]: current torso height (current CoM height)
        - [38:42]: reserved/unused (zeros)

        Args:
            mj_data: MuJoCo data at equilibrium state

        Returns:
            Zero-error observation array (42,)
        """
        obs = jnp.zeros(self.OBS_DIM)

        # Gravity in body frame at equilibrium (upright orientation)
        # At perfect equilibrium, body frame aligns with world frame
        gravity_body = jnp.array([0.0, 0.0, self.mj_model.opt.gravity[2]])
        obs = obs.at[0:3].set(gravity_body)

        # Body linear velocity (zero at equilibrium)
        obs = obs.at[3:6].set(jnp.zeros(3))

        # Joint positions from qpos[7:17] (10 actuated joints)
        joint_pos = jnp.array(mj_data.qpos[7:17])
        obs = obs.at[6:16].set(joint_pos)

        # Joint velocities (zero at equilibrium)
        obs = obs.at[16:26].set(jnp.zeros(10))

        # Previous action (zero at equilibrium)
        obs = obs.at[26:36].set(jnp.zeros(10))

        # Height command and current height (both equal at equilibrium)
        com_height = float(mj_data.subtree_com[0, 2])
        obs = obs.at[36].set(com_height)  # height_cmd
        obs = obs.at[37].set(com_height)  # current height

        # Reserved/unused (already zeros)

        return obs

    def _compute_wbc_at_equilibrium(self, mj_data: mujoco.MjData, obs: Array) -> NDArray:
        """Compute WBC torque at equilibrium state.

        Args:
            mj_data: MuJoCo data at equilibrium
            obs: Zero-error observation

        Returns:
            WBC torque at equilibrium (10,)
        """
        # Create a minimal CentroidalState for equilibrium
        # At equilibrium, all velocities and rates are zero
        com_pos = jnp.array(mj_data.subtree_com[0])
        com_height = float(com_pos[2])

        # Create equilibrium state with zero velocities
        state = CentroidalState(
            com_pos=com_pos,
            com_vel=jnp.zeros(3),
            capture_point=jnp.array([com_pos[0], com_pos[1]]),  # CP = CoM at zero velocity
            divergence=jnp.zeros(2),
            linear_momentum=jnp.zeros(3),
            angular_momentum=jnp.zeros(3),
            left_wheel_contact=True,  # Assume both wheels in contact at equilibrium
            right_wheel_contact=True,
            left_wheel_force=0.0,  # Will be computed by WBC
            right_wheel_force=0.0,
            base_quat=jnp.array(mj_data.qpos[3:7]),
            base_ang_vel=jnp.zeros(3),
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            roll_rate=0.0,
            pitch_rate=0.0,
            yaw_rate=0.0,
            body_pitch_x=0.0,
            body_roll_y=0.0,
            body_yaw_z=0.0,
            body_pitch_rate_x=0.0,
            body_roll_rate_y=0.0,
            body_yaw_rate_z=0.0,
            pitch_x=0.0,
            roll_y=0.0,
            yaw_z=0.0,
            pitch_rate_x=0.0,
            roll_rate_y=0.0,
            yaw_rate_z=0.0,
            left_contact_force_world=jnp.zeros(3),
            right_contact_force_world=jnp.zeros(3),
            total_contact_force_z=0.0,
            contact_force_valid=False,
        )

        # Call WBC pipeline to compute torque at equilibrium
        # Use height command equal to current height (zero height error)
        tau_wbc = self.wbc_pipeline.compute_wbc_torque(
            mj_data,
            obs,
            state,
            height_cmd=com_height,
            hip_roll_authority_scale=1.0,
        )

        # Convert JAX array to numpy for storage
        return np.array(tau_wbc)

    def wrap(
        self,
        tau_wbc_current: NDArray,
        current_state: dict,
    ) -> tuple[NDArray, dict]:
        """Wrap WBC torque to remove equilibrium bias.

        Args:
            tau_wbc_current: Current WBC output (10,)
            current_state: Current robot state for error metrics
                Required keys: com_z, pitch_x, roll_y, joint_pos, com_vel, angular_vel

        Returns:
            tau_wbc_wrapped: Bias-corrected WBC torque (10,)
            telemetry: Dict with all diagnostic values
        """
        # Compute correction torque (remove equilibrium bias)
        tau_wbc_correction = tau_wbc_current - self.tau_wbc_equilibrium

        # Compute wrapped WBC torque
        tau_wbc_wrapped = self.tau_static_ref + tau_wbc_correction

        # Compute equilibrium error metrics
        # Extract joint positions from equilibrium qpos[7:17]
        equilibrium_joint_pos = self.equilibrium_state['qpos'][7:17]
        posture_error_norm = np.linalg.norm(
            current_state['joint_pos'] - equilibrium_joint_pos
        )
        com_height_error = current_state['com_z'] - self.equilibrium_state['com_z']
        pitch_x_error = current_state['pitch_x'] - self.equilibrium_state['pitch_x']
        roll_y_error = current_state['roll_y'] - self.equilibrium_state['roll_y']
        com_velocity_norm = np.linalg.norm(current_state.get('com_vel', np.zeros(3)))
        angular_velocity_norm = np.linalg.norm(current_state.get('angular_vel', np.zeros(3)))

        # Safety diagnostic (not a hard control switch)
        if posture_error_norm > 0.1 or abs(com_height_error) > 0.05:
            print(f"WARNING: Fixed static reference may no longer be physically exact "
                  f"(posture_error={posture_error_norm:.3f}, com_height_error={com_height_error:.3f})")

        # Build telemetry dict
        support_joints = [2, 3, 7, 8]
        telemetry = {
            # Torque components (full 10-dim arrays)
            'tau_static_ref': self.tau_static_ref.copy(),
            'tau_wbc_equilibrium': self.tau_wbc_equilibrium.copy(),
            'tau_wbc_current': tau_wbc_current.copy(),
            'tau_wbc_correction': tau_wbc_correction.copy(),
            'tau_wbc_wrapped': tau_wbc_wrapped.copy(),

            # Support joint bias removed (for diagnostics)
            'support_joint_bias_removed': self.tau_wbc_equilibrium[support_joints].copy(),

            # Equilibrium error metrics
            'posture_error_norm': posture_error_norm,
            'com_height_error': com_height_error,
            'pitch_x_error': pitch_x_error,
            'roll_y_error': roll_y_error,
            'com_velocity_norm': com_velocity_norm,
            'angular_velocity_norm': angular_velocity_norm,
        }

        return tau_wbc_wrapped, telemetry
