"""Force distribution for wheeled biped contact points.

Distributes desired centroidal wrench (force + moment on CoM) to contact forces
at left and right wheel contact points.
"""

import jax.numpy as jnp
import mujoco
from jax import Array


class ForceDistributor:
    """Distributes centroidal wrench to wheel contact forces."""

    def __init__(self, mj_model: mujoco.MjModel):
        """Initialize force distributor.

        Args:
            mj_model: MuJoCo model with robot definition
        """
        self.mj_model = mj_model

        # Find wheel body IDs
        self.l_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
        self.r_wheel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")

    def get_contact_positions(self, mj_data: mujoco.MjData, com_pos: Array) -> tuple[Array, Array]:
        """Get wheel contact positions relative to CoM.

        Args:
            mj_data: MuJoCo data with current robot state
            com_pos: CoM position (3,) in world frame

        Returns:
            Tuple of (r_left, r_right) where each is (3,) vector from CoM to contact point
        """
        # Get wheel body positions (center of wheel link)
        l_wheel_pos = jnp.array(mj_data.xpos[self.l_wheel_id])
        r_wheel_pos = jnp.array(mj_data.xpos[self.r_wheel_id])

        # For wheels, contact point is approximately at the bottom of the wheel
        # Assume wheel radius ~0.05m (adjust based on actual robot)
        wheel_radius = 0.05
        contact_offset = jnp.array([0.0, 0.0, -wheel_radius])

        l_contact_pos = l_wheel_pos + contact_offset
        r_contact_pos = r_wheel_pos + contact_offset

        # Position vectors from CoM to contact points
        r_left = l_contact_pos - com_pos
        r_right = r_contact_pos - com_pos

        return r_left, r_right

    def distribute_wrench(
        self,
        mj_data: mujoco.MjData,
        com_pos: Array,
        desired_force: Array,
        desired_moment: Array,
    ) -> tuple[Array, Array]:
        """Distribute desired wrench to wheel contact forces.

        Uses analytical solution assuming:
        - Equal vertical force distribution (load balancing)
        - Horizontal forces distributed to satisfy moment constraint
        - Moment primarily controlled by differential vertical forces

        Args:
            mj_data: MuJoCo data with current robot state
            com_pos: CoM position (3,) in world frame
            desired_force: Desired total force (3,) [fx, fy, fz]
            desired_moment: Desired moment (3,) [mx, my, mz] about CoM

        Returns:
            Tuple of (f_left, f_right) where each is (3,) contact force in world frame
        """
        # Get contact positions relative to CoM
        r_left, r_right = self.get_contact_positions(mj_data, com_pos)

        # === Force distribution strategy ===

        # 1. Vertical forces: distribute equally + differential for roll moment
        # F_z_total = f_left_z + f_right_z
        # M_x = r_left_y * f_left_z + r_right_y * f_right_z (roll moment from vertical forces)

        f_z_total = desired_force[2]
        m_x_desired = desired_moment[0]  # roll moment

        # Solve for differential vertical force to create roll moment
        # Assuming symmetric stance: r_left_y ≈ -r_right_y = d/2 (lateral separation)
        lateral_separation = jnp.abs(r_left[1] - r_right[1])

        # Avoid division by zero
        lateral_separation = jnp.maximum(lateral_separation, 0.01)

        # Differential force to create roll moment: Δf_z = M_x / (d/2)
        delta_f_z = m_x_desired / (lateral_separation / 2.0)

        # Distribute vertical force with differential
        f_left_z = (f_z_total / 2.0) + (delta_f_z / 2.0)
        f_right_z = (f_z_total / 2.0) - (delta_f_z / 2.0)

        # Clamp to non-negative (wheels can only push, not pull)
        f_left_z = jnp.maximum(f_left_z, 0.0)
        f_right_z = jnp.maximum(f_right_z, 0.0)

        # 2. Horizontal forces: distribute equally
        # For sagittal (x) and lateral (y) forces, split evenly
        f_left_x = desired_force[0] / 2.0
        f_right_x = desired_force[0] / 2.0

        f_left_y = desired_force[1] / 2.0
        f_right_y = desired_force[1] / 2.0

        # Assemble contact forces
        f_left = jnp.array([f_left_x, f_left_y, f_left_z])
        f_right = jnp.array([f_right_x, f_right_y, f_right_z])

        return f_left, f_right
