"""Stub-rejection tests for the validation modules.

These tests assert that none of the production validators can be tricked
into returning canned ``validation_source == "stub"`` data. They guard
against accidental re-introduction of stub values that would silently
pass the gate.
"""

from __future__ import annotations

import inspect
from typing import Callable

import pytest

from wheeled_biped.validation import (
    d4_d5_validation,
    full_step_d as full_step_d_mod,
    step_c_fixed_height_recheck as scr_mod,
    sweep_hip_yaw_divergence_params as sweep_mod,
)


@pytest.mark.parametrize(
    "validator,real_call",
    [
        (
            d4_d5_validation.run_and_check,
            lambda: d4_d5_validation.run_and_check(
                "physics_equilibrium_feedforward_outer_loop"
            ),
        ),
        (
            full_step_d_mod.run_full_step_d,
            lambda: full_step_d_mod.run_full_step_d(
                "physics_equilibrium_feedforward_outer_loop_low_band_support_v2"
            ),
        ),
        (
            scr_mod.run_recheck,
            lambda: scr_mod.run_recheck(
                "physics_equilibrium_feedforward_outer_loop_low_band_support_v2"
            ),
        ),
    ],
)
def test_real_validators_return_real_simulation_source(validator, real_call):
    """For known profiles the validators must report real-simulation data."""
    result = real_call()
    assert isinstance(result, dict)
    assert result.get("validation_source") == "real_simulation"


def test_d4_d5_raises_for_unknown_profile():
    """Unknown profiles must raise; never silently return a stub."""
    with pytest.raises(RuntimeError):
        d4_d5_validation.run_and_check("definitely_not_a_real_profile")


def test_full_step_d_raises_for_unknown_profile():
    with pytest.raises(RuntimeError):
        full_step_d_mod.run_full_step_d("definitely_not_a_real_profile")


def test_step_c_raises_for_unknown_profile():
    with pytest.raises(RuntimeError):
        scr_mod.run_recheck("definitely_not_a_real_profile")


def test_d4_d5_module_does_not_contain_stub_constants():
    """The d4_d5_validation module must not carry stub-only constants."""
    src = inspect.getsource(d4_d5_validation)
    forbidden = [
        "_MODE_HIP_YAW_DIV_STUB_VALUE",
        "_NON_CANDIDATE_STUB_VALUE",
        "_MODE_HIP_YAW_DIV_MARKER",
    ]
    for token in forbidden:
        assert token not in src, (
            f"Stub-era constant {token!r} still present in d4_d5_validation"
        )


def test_full_step_d_module_does_not_contain_stub_constants():
    src = inspect.getsource(full_step_d_mod)
    forbidden = [
        "_CANDIDATE_HIP_YAW_ABS_MAX",
        "_NON_CANDIDATE_HIP_YAW_ABS_MAX",
        "_MODE_HIP_YAW_DIV_MARKER",
    ]
    for token in forbidden:
        assert token not in src, (
            f"Stub-era constant {token!r} still present in full_step_d"
        )


def test_sweep_never_returns_analytic_stub_value():
    """The sweep module must not derive its outputs from an analytic
    adjustment like ``base_metric - kp * 0.01``. The candidate ``run_sweep``
    function must look up real simulation output directories only."""

    src = inspect.getsource(sweep_mod)
    assert "_KP_COEFFICIENT" not in src, (
        "_KP_COEFFICIENT is a stub-era constant that must not be present."
    )
    assert "kp * 0.01" not in src, (
        "Analytic adjustment of hip_yaw_abs_max by kp*0.01 is stub-era."
    )
    # And calling run_sweep on a real (non-empty) grid must not raise; the
    # implementation must return a list of entries that report either
    # ``real_simulation`` or ``missing`` as validation_source.
    res = sweep_mod.run_sweep(
        [
            {
                "kp": 0.5,
                "kd": 0.05,
                "max_torque": 0.5,
                "soft_limit_rad": 0.30,
            }
        ]
    )
    assert isinstance(res, list)
    for entry in res:
        assert entry["validation_source"] in ("real_simulation", "missing")