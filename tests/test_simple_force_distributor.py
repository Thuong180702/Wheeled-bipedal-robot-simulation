import jax.numpy as jnp

from wheeled_biped.controllers.simple_force_distributor import SimpleForceDistributor


def test_both_contacts_split_vertical_force():
    distributor = SimpleForceDistributor()
    f_left, f_right, tau_hip_roll, diagnostics = distributor.distribute_wrench_contact_aware(
        jnp.array([0.0, 0.0, 80.0, 0.0, 0.0, 0.0]),
        left_contact=True,
        right_contact=True,
    )
    assert float(f_left[2]) == 40.0
    assert float(f_right[2]) == 40.0
    assert float(jnp.linalg.norm(tau_hip_roll)) == 0.0
    assert diagnostics["feasible"]


def test_left_only_contact_keeps_lifted_leg_extended_for_recovery():
    distributor = SimpleForceDistributor()
    f_left, f_right, _, diagnostics = distributor.distribute_wrench_contact_aware(
        jnp.array([0.0, 0.0, 80.0, 0.0, 0.0, 0.0]),
        left_contact=True,
        right_contact=False,
    )
    assert float(f_left[2]) == 80.0
    assert float(f_right[2]) == 50.0
    assert diagnostics["feasible"]


def test_no_contact_outputs_anticipatory_support():
    distributor = SimpleForceDistributor()
    f_left, f_right, tau_hip_roll, diagnostics = distributor.distribute_wrench_contact_aware(
        jnp.array([1.0, 2.0, 80.0, 3.0, 4.0, 5.0]),
        left_contact=False,
        right_contact=False,
    )
    assert jnp.allclose(f_left, jnp.array([0.5, 1.0, 40.0]))
    assert jnp.allclose(f_right, jnp.array([0.5, 1.0, 40.0]))
    assert float(jnp.linalg.norm(tau_hip_roll)) > 0.0
    assert diagnostics["feasible"]
    assert diagnostics["reason"] == "flight_phase_anticipatory"


def test_no_contact_anticipatory_support_preserves_roll_force_asymmetry():
    distributor = SimpleForceDistributor(
        tau_hip_roll_max=15.0,
        max_force_asymmetry=40.0,
        min_wheel_force=20.0,
    )

    f_left, f_right, tau_hip_roll, diagnostics = distributor.distribute_wrench_contact_aware(
        jnp.array([0.0, 0.0, 80.0, -25.0, 0.0, 0.0]),
        left_contact=False,
        right_contact=False,
        hip_roll_authority_scale=0.25,
    )

    assert jnp.allclose(tau_hip_roll, jnp.array([3.125, -3.125]))
    assert jnp.allclose(jnp.array([f_left[2], f_right[2]]), jnp.array([25.0, 55.0]))
    assert jnp.isclose(f_left[2] + f_right[2], 80.0)
    assert diagnostics["feasible"]
    assert diagnostics["reason"] == "flight_phase_anticipatory"


def test_reduced_hip_roll_authority_reallocates_roll_moment_to_vertical_forces():
    distributor = SimpleForceDistributor(
        tau_hip_roll_max=15.0,
        max_force_asymmetry=40.0,
        min_wheel_force=20.0,
    )

    f_left, f_right, tau_hip_roll, diagnostics = distributor.distribute_wrench_contact_aware(
        jnp.array([0.0, 0.0, 80.0, 25.0, 0.0, 0.0]),
        left_contact=True,
        right_contact=True,
        hip_roll_authority_scale=0.25,
    )

    assert jnp.allclose(tau_hip_roll, jnp.array([-3.125, 3.125]))
    assert jnp.allclose(jnp.array([f_left[2], f_right[2]]), jnp.array([55.0, 25.0]))
    assert jnp.isclose(f_left[2] + f_right[2], 80.0)
    assert diagnostics["feasible"]
