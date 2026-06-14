# Step E Final Validation Summary

**Date:** 2026-06-01  
**Verdict:** Step E is **not complete**. Do **not** proceed to Step C.

## 1. What changed

- Rejected `pitch_reserve_tau=2.0` as the Step E fix family.
- Reverted torque-budget-aware position authority to the no-pitch-reserve total-torque-bound allocator.
- Added optional bounded steady-state-only centering integral.
- Added separate integral telemetry.
- Did not implement Step C, height recovery, dynamic height transition, push robustness, WBC, E0b/E0c/E0d, or reference shifting.

## 2. No-pitch-reserve baseline restoration

The restored allocator reproduces the prior best no-pitch-reserve behavior.

Telemetry:

- V1: `outputs/hierarchical_controller_sim/telemetry_1780277761.csv`
- V2: `outputs/hierarchical_controller_sim/telemetry_1780277894.csv`
- V3: `outputs/hierarchical_controller_sim/telemetry_1780278172.csv`

| Run | Max abs support error | Final abs support error | Termination |
|---|---:|---:|---|
| V1 1000 | `0.121484 m` | `0.121484 m` | completed |
| V2 2000 | `0.385799 m` | `0.163960 m` | completed |
| V3 5000 | `0.385799 m` | `0.052668 m` | completed |

V3 confirms:

- peak restored from pitch-reserve regression (`0.6657 m`) back to `0.3858 m`
- final remains near prior best (`0.0527 m`)
- `tau_pitch_reserve_applied_max = 0.0`

## 3. Steady-state integral validation

Telemetry:

- V1: `outputs/hierarchical_controller_sim/telemetry_1780278425.csv`
- V2: `outputs/hierarchical_controller_sim/telemetry_1780278548.csv`
- V3: `outputs/hierarchical_controller_sim/telemetry_1780278812.csv`

| Run | Max abs support error | Final abs support error | Integral active rows |
|---|---:|---:|---:|
| V1 1000 | `0.121484 m` | `0.121484 m` | `12 / 1000` |
| V2 2000 | `0.385799 m` | `0.163959 m` | `12 / 2000` |
| V3 5000 | `0.385799 m` | `0.052668 m` | `12 / 5000` |

The integral did not worsen peak, but it also did not improve final error.

V3 integral gate reasons:

- `height_unsafe`: `2856`
- `pitch_error_large`: `2010`
- `support_velocity_large`: `86`
- `pitch_rate_large`: `35`
- `safe_steady_state`: `12`
- `contact_invalid`: `1`

Interpretation: the gate correctly prevents transient action, but also classifies the final low-height/pitch-offset equilibrium as unsafe, so the integral remains inactive in the phase where final bias correction would matter.

## 4. Tests

Passed:

- `pytest tests/test_torque_budget_aware_position.py -q` → `18 passed`
- `pytest tests/test_torque_budget_aware_position.py tests/test_sagittal_velocity_damped_balance_controller.py -q` → `43 passed`
- `pytest tests/test_balance_core_components.py -q` → `25 passed`
- `pytest tests/test_balance_core_validation_workflow.py -q` → `25 passed`

Required targeted selection:

```text
pytest tests -q -k "support_position or support_velocity or position_hold or authority or sagittal_velocity_damped or equilibrium"
```

Result:

- `98 passed`
- `1 failed` in [test_simple_force_distributor.py](tests/test_simple_force_distributor.py)
- failure is the pre-existing unrelated hip-roll authority expectation noted before this fix

## 5. Acceptance gates

V3 final state:

- max abs support-position error: `0.385799 m`
- final abs support-position error: `0.052668 m`

Gate results:

- Preferred (`max_abs <= 0.10`, final `<= 0.05`): **fail**
- Fallback (`max_abs <= 0.15`, final `<= 0.10`): **fail**
- Hard minimum (`max_abs <= 0.30`, final `<= 0.10`): **fail**
  - peak part: **fail**
  - final part: **pass**

## 6. Classification

- Pitch-reserve fix family: **wrong / rejected**
- No-pitch-reserve allocator: **restored**
- Steady-state integral: **safe but ineffective under strict gating**
- Steady-state centering: **not solved**
- Transient containment: **not solved**
- Step E: **not complete**
- Step C: **blocked**

## 7. Integrity checks

Confirmed:

- no WBC path added
- no E0b/E0c/E0d runtime branch added
- `kp_cp` remains disabled in velocity-damped controller
- support-center / wheel-midpoint error remains Step E metric
- COM error remains diagnostic only
- pitch reference remains active
- torque ownership unchanged
- baseline `SagittalWheelBalanceController` and `SagittalVelocityDampedBalanceController` remain mutually exclusive
- no reference shift after drift

## 8. Next work

Do **not** proceed to Step C.

The next Step E work should focus on transient authority sizing and/or reviewing whether the strict safe-steady-state gates should classify the nominal final low-height/pitch-offset equilibrium as unsafe. Do not tune the integral blindly and do not claim Step E solved.
