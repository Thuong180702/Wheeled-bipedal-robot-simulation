import jax.numpy as jnp

from wheeled_biped.controllers.leg_position_controller import LegPositionController


def test_leg_position_controller_uses_per_step_full_target():
    controller = LegPositionController(
        kp_hip_yaw=5.0,
        kd_hip_yaw=1.0,
        kp_hip_pitch=20.0,
        kd_hip_pitch=2.0,
        kp_knee=30.0,
        kd_knee=3.0,
        max_torque=40.0,
    )
    joint_pos = jnp.array([0.2, -0.2, 0.8, 1.6, 1.0, -0.2, 0.2, 1.0, 1.9, -1.0])
    joint_vel = jnp.zeros(10)
    target_joint_pos = jnp.array([0.0, -0.1, 1.0, 1.7, 0.0, 0.0, 0.1, 0.9, 1.8, 0.0])

    tau = controller.compute_leg_torques(joint_pos, joint_vel, target_joint_pos)

    assert jnp.isclose(tau[0], 0.0)
    assert jnp.isclose(tau[1], 0.5)
    assert jnp.isclose(tau[2], 4.0)
    assert jnp.isclose(tau[3], 3.0)
    assert jnp.isclose(tau[4], 0.0)
    assert jnp.isclose(tau[5], 0.0)
    assert jnp.isclose(tau[6], -0.5)
    assert jnp.isclose(tau[7], -2.0)
    assert jnp.isclose(tau[8], -3.0)
    assert jnp.isclose(tau[9], 0.0)


def test_leg_position_controller_damps_velocity_and_clips():
    controller = LegPositionController(
        kp_hip_yaw=5.0,
        kd_hip_yaw=1.0,
        kp_hip_pitch=20.0,
        kd_hip_pitch=2.0,
        kp_knee=30.0,
        kd_knee=3.0,
        max_torque=5.0,
    )
    joint_pos = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    joint_vel = jnp.array([0.0, 2.0, -2.0, -2.0, 0.0, 0.0, -2.0, 2.0, 2.0, 0.0])
    target_joint_pos = jnp.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0])

    tau = controller.compute_leg_torques(joint_pos, joint_vel, target_joint_pos)

    assert jnp.isclose(tau[1], -2.0)
    assert jnp.isclose(tau[2], 5.0)
    assert jnp.isclose(tau[3], 5.0)
    assert jnp.isclose(tau[6], 2.0)
    assert jnp.isclose(tau[7], -5.0)
    assert jnp.isclose(tau[8], -5.0)
