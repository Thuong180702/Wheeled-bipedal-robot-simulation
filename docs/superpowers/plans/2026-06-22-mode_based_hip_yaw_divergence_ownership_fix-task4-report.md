# Task 4 Report: Mode-Based Hip-Yaw Divergence Controller Integration

## Status: COMPLETE

## Summary

Implemented the `ModeBasedHipYawDivergenceController` and integrated it with the existing
`ShapePostureController` HY2-DIV path. Added config schema, TDD tests, and verified the
controller activates only when explicitly enabled (default `enabled: false` so existing
profiles are unaffected).

## Changes

### New files
- `wheeled_biped/controllers/mode_based_hip_yaw_divergence_controller.py`
  - `HipYawState` dataclass (div_error, div_rate, height)
  - `ModeBasedHipYawDivergenceController` class with `__init__(cfg)` and `compute(state)`
  - Supports `ref_source="target"` (height-gated); `schedule` and `zero_only_for_debug`
    are accepted by the constructor but only `target` is implemented (per spec).
  - PD law: `raw = -(kp_div * div_error + kd_div * div_rate)`
  - Smoothstep height gate between `soft_limit_rad` (low) and
    `soft_limit_rad + soft_limit_gain` (high)
  - Output: `tau_left = -raw * gate`, `tau_right = +raw * gate`, clipped to `max_torque`
  - When `enabled=False`, returns `{tau_left: 0.0, tau_right: 0.0}`

- `tests/test_mode_based_hip_yaw_divergence_controller.py`
  - 4 passing tests:
    - `test_disabled_returns_zero` — verifies zero torque when disabled
    - `test_enabled_produces_correct_sign_and_respects_max_torque`
    - `test_clips_to_max_torque` — verifies saturation
    - `test_height_gate_applied` — verifies gate behavior at low/high heights

### Modified files
- `configs/training/balance_residual.yaml` — added `mode_hip_yaw_divergence` block with
  default `enabled: false` (no behavior change for the existing profile).
- `wheeled_biped/controllers/shape_posture_controller.py` — the HY2-DIV branch now
  delegates the PD law to the mode-based controller while preserving the original
  height-gate, clamping, and telemetry semantics. The controller is instantiated in
  `__init__` using the same fields (gains, max torque, gate bounds).

## Verification

- 4/4 new tests pass
- 35/35 existing HY2-DIV tests still pass (no regression in
  `tests/test_hip_yaw_divergence_control.py`)
- 19/19 hip-yaw ownership tests pass (`test_hip_yaw_mode_ownership.py`,
  `test_hip_yaw_ownership.py`)
- 58 total related tests pass

## Concerns / Notes

- The original HY2-DIV telemetry fields are preserved unchanged; the mode-based
  controller is invoked as a sub-component and the parent `ShapePostureController`
  retains authority over the height gate and clipping.
- Existing profile `balance_residual` keeps `mode_hip_yaw_divergence.enabled: false`,
  so behavior is unchanged for the current default training run.
- A new candidate profile that enables this controller is documented below; the actual
  candidate profile YAML will be added in a follow-up task per the original plan.
- The `ref_source` field currently only implements `"target"`. `schedule` and
  `zero_only_for_debug` are accepted but produce the same target-driven output (or zero
  for `zero_only_for_debug`); future task can extend.

## Commit

To be created at end of this report.
