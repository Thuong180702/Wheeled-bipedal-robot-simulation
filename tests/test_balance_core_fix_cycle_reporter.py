# tests/test_balance_core_fix_cycle_reporter.py
import pytest
from wheeled_biped.validation.fix_cycle_reporter import FixCycleReporter, FixCycleRecord


def test_generate_fix_cycle_report():
    """Should generate structured fix cycle documentation."""
    record = FixCycleRecord(
        cycle_number=1,
        classified_failure_mode="F2.1",
        responsible_component="SagittalWheelBalanceController",
        evidence_fields={"pitch_max_rad": 0.35},
        allowed_fix_scope="SagittalWheelBalanceController only",
        files_changed=["wheeled_biped/controllers/sagittal_wheel_balance_controller.py"],
        parameters_before={"kp_pitch": 50.0},
        parameters_after={"kp_pitch": 75.0},
        validation_command="python scripts/simulate_hierarchical_controller.py --controller-mode balance-core --steps 100",
        validation_result_before="FAIL at step 30: pitch divergence",
        validation_result_after="PASS: 100 steps completed",
        failure_resolved=True,
        new_failure_appeared=False,
        structural_invariants_after_fix={"all": "PASS"},
    )

    reporter = FixCycleReporter()
    report = reporter.generate_markdown(record)

    assert "# Fix Cycle 1" in report
    assert "F2.1" in report
    assert "SagittalWheelBalanceController" in report
    assert "kp_pitch" in report
