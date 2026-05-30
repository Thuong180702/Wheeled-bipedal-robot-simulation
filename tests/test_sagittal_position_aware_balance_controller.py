"""Tests for sagittal position-aware balance controller and dynamics identification."""

import pytest

from scripts.identify_sagittal_balance_dynamics import (
    build_identified_model_payload,
    model_is_usable,
)


def test_build_identified_model_payload_includes_state_space_keys():
    payload = build_identified_model_payload(
        A=[[1, 0], [0, 1]],
        B=[[0], [1]],
        state_names=["pos", "vel"],
        input_name="wheel_torque",
    )

    assert sorted(payload.keys()) == ["A", "B", "input_name", "state_names"]


def test_model_is_usable_requires_all_quality_gates():
    assert model_is_usable(
        one_step_r2=0.85,
        rollout_r2=0.65,
        residual_mean_abs=0.05,
        sign_response_ok=True,
        nominal_fit_ok=True,
        height_variant_fit_ok=True,
    ) is True

    assert model_is_usable(
        one_step_r2=0.79,
        rollout_r2=0.65,
        residual_mean_abs=0.05,
        sign_response_ok=True,
        nominal_fit_ok=True,
        height_variant_fit_ok=True,
    ) is False

    assert model_is_usable(
        one_step_r2=0.85,
        rollout_r2=0.55,
        residual_mean_abs=0.05,
        sign_response_ok=True,
        nominal_fit_ok=True,
        height_variant_fit_ok=True,
    ) is False

    assert model_is_usable(
        one_step_r2=0.85,
        rollout_r2=0.65,
        residual_mean_abs=0.15,
        sign_response_ok=True,
        nominal_fit_ok=True,
        height_variant_fit_ok=True,
    ) is False

    assert model_is_usable(
        one_step_r2=0.85,
        rollout_r2=0.65,
        residual_mean_abs=0.05,
        sign_response_ok=False,
        nominal_fit_ok=True,
        height_variant_fit_ok=True,
    ) is False
