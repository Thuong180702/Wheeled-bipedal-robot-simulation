from wheeled_biped.controllers.balance_core_types import (
    ContactSupervisorOutput,
    ContactSupervisorState,
)


class ContactSupervisor:
    """Read-only contact classifier for balance-core telemetry and state gating."""

    def __init__(self, control_dt: float = 0.02):
        self.control_dt = control_dt
        self.previous_state = None
        self.contact_duration_s = 0.0

    def update(
        self,
        left_wheel_contact: bool,
        right_wheel_contact: bool,
        contact_force_valid: bool,
        left_normal_force_n: float,
        right_normal_force_n: float,
    ) -> ContactSupervisorOutput:
        if left_wheel_contact and right_wheel_contact:
            state = ContactSupervisorState.DOUBLE_CONTACT
        elif left_wheel_contact:
            state = ContactSupervisorState.LEFT_ONLY
        elif right_wheel_contact:
            state = ContactSupervisorState.RIGHT_ONLY
        else:
            state = ContactSupervisorState.FLIGHT_OR_NO_CONTACT

        previous_state = self.previous_state
        if previous_state == state:
            self.contact_duration_s += self.control_dt
            transition_event = "none"
        else:
            self.contact_duration_s = 0.0 if previous_state is None else self.control_dt
            transition_event = (
                f"initial_{state.value}"
                if previous_state is None
                else f"{previous_state.value}_to_{state.value}"
            )
        self.previous_state = state

        left_force = float(left_normal_force_n) if left_wheel_contact and contact_force_valid else 0.0
        right_force = float(right_normal_force_n) if right_wheel_contact and contact_force_valid else 0.0

        recovery_hook_fields = {
            "entered_single_contact": state in {ContactSupervisorState.LEFT_ONLY, ContactSupervisorState.RIGHT_ONLY}
            and previous_state != state,
            "entered_no_contact": state == ContactSupervisorState.FLIGHT_OR_NO_CONTACT and previous_state != state,
            "force_valid_for_recovery": bool(contact_force_valid),
        }

        return ContactSupervisorOutput(
            state=state,
            previous_state=previous_state,
            left_wheel_contact=bool(left_wheel_contact),
            right_wheel_contact=bool(right_wheel_contact),
            contact_force_valid=bool(contact_force_valid),
            left_normal_force_n=left_force,
            right_normal_force_n=right_force,
            contact_duration_s=float(self.contact_duration_s),
            transition_event=transition_event,
            recovery_hook_fields=recovery_hook_fields,
        )
