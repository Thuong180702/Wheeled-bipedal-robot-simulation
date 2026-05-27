# tests/test_balance_core_failure_classifier.py
import pandas as pd
import pytest
from wheeled_biped.validation.failure_classifier import (
    FailureClassifier,
    FailureMode,
)


def test_pitch_divergence_classified_as_primary():
    """Pitch exceeding threshold before other failures should be classified as F2.1."""
    df = pd.DataFrame({
        "step": [0, 10, 20, 30, 40],
        "time": [0.0, 0.02, 0.04, 0.06, 0.08],
        "pitch_x_rad": [0.0, 0.1, 0.2, 0.35, 0.4],  # Exceeds 0.30 at step 30
        "roll_y_rad": [0.0, 0.0, 0.0, 0.0, 0.0],
        "com_z_m": [0.45, 0.45, 0.44, 0.43, 0.42],
        "contact_supervisor_state": ["DOUBLE_CONTACT"] * 5,
    })

    classifier = FailureClassifier()
    result = classifier.classify(df)

    assert result.primary_failure_mode == FailureMode.PITCH_DIVERGENCE
    assert result.first_threshold_crossing_step == 30
    assert result.responsible_component == "SagittalWheelBalanceController"


def test_height_collapse_secondary_to_pitch():
    """Height collapse after pitch divergence should be classified as secondary."""
    df = pd.DataFrame({
        "step": [0, 10, 20, 30, 40, 50],
        "time": [0.0, 0.02, 0.04, 0.06, 0.08, 0.10],
        "pitch_x_rad": [0.0, 0.1, 0.2, 0.35, 0.4, 0.45],  # Exceeds at step 30
        "roll_y_rad": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "com_z_m": [0.45, 0.45, 0.44, 0.43, 0.39, 0.35],  # Drops >0.05 at step 40
        "contact_supervisor_state": ["DOUBLE_CONTACT"] * 6,
    })

    classifier = FailureClassifier()
    result = classifier.classify(df)

    assert result.primary_failure_mode == FailureMode.PITCH_DIVERGENCE
    assert result.first_threshold_crossing_step == 30
    assert len(result.secondary_threshold_crossings) == 1
    assert result.secondary_threshold_crossings[0].failure_mode == FailureMode.HEIGHT_COLLAPSE
    assert result.secondary_threshold_crossings[0].step == 40


def test_roll_divergence_classified():
    """Roll exceeding threshold should be classified as F2.2."""
    df = pd.DataFrame({
        "step": [0, 10, 20, 30],
        "time": [0.0, 0.02, 0.04, 0.06],
        "pitch_x_rad": [0.0, 0.0, 0.0, 0.0],
        "roll_y_rad": [0.0, 0.1, 0.25, 0.3],  # Exceeds 0.20 at step 20
        "com_z_m": [0.45, 0.45, 0.45, 0.45],
        "contact_supervisor_state": ["DOUBLE_CONTACT"] * 4,
    })

    classifier = FailureClassifier()
    result = classifier.classify(df)

    assert result.primary_failure_mode == FailureMode.ROLL_DIVERGENCE
    assert result.first_threshold_crossing_step == 20
    assert result.responsible_component == "LateralRollBalanceController"
