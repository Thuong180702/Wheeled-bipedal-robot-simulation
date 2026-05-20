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


def test_left_only_contact_sends_right_force_to_zero():
    distributor = SimpleForceDistributor()
    f_left, f_right, _, diagnostics = distributor.distribute_wrench_contact_aware(
        jnp.array([0.0, 0.0, 80.0, 0.0, 0.0, 0.0]),
        left_contact=True,
        right_contact=False,
    )
    assert float(f_left[2]) == 80.0
    assert float(f_right[2]) == 0.0
    assert diagnostics["feasible"]


def test_no_contact_outputs_zero_and_infeasible():
    distributor = SimpleForceDistributor()
    f_left, f_right, tau_hip_roll, diagnostics = distributor.distribute_wrench_contact_aware(
        jnp.array([1.0, 2.0, 80.0, 3.0, 4.0, 5.0]),
        left_contact=False,
        right_contact=False,
    )
    assert float(jnp.linalg.norm(f_left)) == 0.0
    assert float(jnp.linalg.norm(f_right)) == 0.0
    assert float(jnp.linalg.norm(tau_hip_roll)) == 0.0
    assert not diagnostics["feasible"]
    assert diagnostics["reason"] == "no_support_contact_lost"
