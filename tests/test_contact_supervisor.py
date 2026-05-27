from wheeled_biped.controllers.balance_core_types import ContactSupervisorState
from wheeled_biped.controllers.contact_supervisor import ContactSupervisor


def test_contact_supervisor_reports_double_contact():
    supervisor = ContactSupervisor()
    output = supervisor.update(
        left_wheel_contact=True,
        right_wheel_contact=True,
        contact_force_valid=True,
        left_normal_force_n=40.0,
        right_normal_force_n=41.0,
    )

    assert output.state == ContactSupervisorState.DOUBLE_CONTACT
    assert output.left_wheel_contact is True
    assert output.right_wheel_contact is True
    assert output.contact_force_valid is True
    assert output.left_normal_force_n == 40.0
    assert output.right_normal_force_n == 41.0


def test_contact_supervisor_reports_left_only_without_fake_right_force():
    supervisor = ContactSupervisor()
    output = supervisor.update(
        left_wheel_contact=True,
        right_wheel_contact=False,
        contact_force_valid=True,
        left_normal_force_n=55.0,
        right_normal_force_n=999.0,
    )

    assert output.state == ContactSupervisorState.LEFT_ONLY
    assert output.left_normal_force_n == 55.0
    assert output.right_normal_force_n == 0.0


def test_contact_supervisor_reports_no_contact_and_zero_forces():
    supervisor = ContactSupervisor(control_dt=0.02)
    output = supervisor.update(
        left_wheel_contact=False,
        right_wheel_contact=False,
        contact_force_valid=False,
        left_normal_force_n=10.0,
        right_normal_force_n=20.0,
    )

    assert output.state == ContactSupervisorState.FLIGHT_OR_NO_CONTACT
    assert output.contact_force_valid is False
    assert output.left_normal_force_n == 0.0
    assert output.right_normal_force_n == 0.0


def test_contact_supervisor_exposes_future_recovery_hook_fields():
    supervisor = ContactSupervisor(control_dt=0.02)
    first = supervisor.update(True, True, True, 40.0, 41.0)
    second = supervisor.update(True, False, True, 40.0, 41.0)

    assert first.previous_state is None
    assert first.transition_event == "initial_double_contact"
    assert second.previous_state == ContactSupervisorState.DOUBLE_CONTACT
    assert second.transition_event == "double_contact_to_left_only"
    assert second.contact_duration_s == 0.02
    assert second.recovery_hook_fields == {
        "entered_single_contact": True,
        "entered_no_contact": False,
        "force_valid_for_recovery": True,
    }
