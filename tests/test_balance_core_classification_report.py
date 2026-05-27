# tests/test_balance_core_classification_report.py
import json
import pytest
from wheeled_biped.validation.classification_report import ClassificationReportGenerator
from wheeled_biped.validation.failure_classifier import (
    ClassificationResult,
    FailureMode,
    ThresholdCrossing,
)


def test_generate_json_report():
    """Should generate valid JSON report from classification result."""
    result = ClassificationResult(
        primary_failure_mode=FailureMode.PITCH_DIVERGENCE,
        first_threshold_crossing_step=30,
        first_threshold_crossing_time_s=0.06,
        secondary_threshold_crossings=[],
        responsible_component="SagittalWheelBalanceController",
        evidence_fields={"pitch_max_rad": 0.35},
        fix_allowed_in_balance_core=True,
        recommended_fix_scope="SagittalWheelBalanceController: verify inputs, sign, saturation",
    )

    generator = ClassificationReportGenerator()
    report_json = generator.to_json(result)

    # Should be valid JSON
    parsed = json.loads(report_json)
    assert parsed["primary_failure_mode"] == "F2.1"
    assert parsed["responsible_component"] == "SagittalWheelBalanceController"
    assert parsed["fix_allowed_in_balance_core"] is True


def test_generate_markdown_report():
    """Should generate readable markdown report."""
    result = ClassificationResult(
        primary_failure_mode=FailureMode.PITCH_DIVERGENCE,
        first_threshold_crossing_step=30,
        first_threshold_crossing_time_s=0.06,
        secondary_threshold_crossings=[
            ThresholdCrossing(
                failure_mode=FailureMode.HEIGHT_COLLAPSE,
                step=40,
                time_s=0.08,
                value=0.39,
                threshold=0.40,
            )
        ],
        responsible_component="SagittalWheelBalanceController",
        evidence_fields={"pitch_max_rad": 0.35, "pitch_rate_max_rad_s": 2.5},
        fix_allowed_in_balance_core=True,
        recommended_fix_scope="SagittalWheelBalanceController: verify inputs",
    )

    generator = ClassificationReportGenerator()
    markdown = generator.to_markdown(result)

    assert "# Balance-Core Failure Classification Report" in markdown
    assert "F2.1" in markdown
    assert "SagittalWheelBalanceController" in markdown
    assert "Secondary Threshold Crossings" in markdown
    assert "F1.2" in markdown
