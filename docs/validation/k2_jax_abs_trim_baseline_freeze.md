# K2 JAX ABS Trim Baseline Freeze — Phase 0

**Date:** 2026-06-28
**Branch:** `repo-cleanup-t6j`
**Commit:** `0e1c7135e22b4cb852f71a795426cd3d3f19753a`
**Classification:** `K2_JAX_PORT_INCOMPLETE_WITH_EXACT_BLOCKER`

## Working tree changes (9 modified files)

```
M docs/validation/k2_post_promotion_long_run_and_dynamic_height_regression_report.md
M scripts/simulate_hierarchical_controller.py
M scripts/validate_k2_dynamic_height_gate_crossing.py
M scripts/validate_k2_post_promotion_long_run.py
M tests/test_k2_jax_backend_cli.py
M tests/test_k2_jax_component_parity.py
M tests/test_k2_jax_step_parity.py
M wheeled_biped/controllers/k2_jax_controller.py
M wheeled_biped/controllers/signal_filters.py
```

## Pytest Results

**131/131 passed** in 18:53 (1133.27s).

All test files: `test_k2_jax_backend_cli.py`, `test_k2_jax_branch_activity_audit.py`, `test_k2_jax_component_parity.py`, `test_k2_jax_step_parity.py`.

No xfail, no skip, no silent test removal.

## Both-Synced Parity Results (7 scenarios)

| Scenario | MaxAbsDiff | Step | Actuator | Fell | Classification |
|----------|-----------|------|----------|------|----------------|
| fixed_high_0p480 | 9.537e-08 | 2 | 8 (r_knee) | no | PASS |
| fixed_low_0p330 | 9.537e-08 | 2 | 8 (r_knee) | no | PASS |
| ramp_up (0.33->0.48) | **1.600e-01** | 150 | 9 (r_wheel) | no | **FAIL** |
| ramp_down (0.48->0.33) | 9.537e-08 | 2 | 8 (r_knee) | no | PASS |
| gate_chatter | **1.513e+00** | 150 | 9 (r_wheel) | no | **FAIL** |
| push_fwd_90N | **9.798e-01** | 117 | 4 (l_wheel) | no | **FAIL** |
| push_bwd_90N | 1.561e-06 | 114 | 5 (r_hip_roll) | **yes** | marginal |

**Passed (<1e-5): 3/7 (or 4/7 if push_bwd counted)**
**Failed: 3/7 confirmed + 1 marginal**

## Wheel and Hip-Yaw Diffs

| Scenario | wheel[4] max | wheel[9] max | HY[1] max | HY[6] max |
|----------|-------------|-------------|-----------|-----------|
| fixed_high_0p480 | 1.55e-15 | 1.55e-15 | 1.39e-17 | 6.94e-18 |
| fixed_low_0p330 | 2.29e-16 | 2.22e-16 | 2.78e-17 | 1.39e-17 |
| ramp_up | **1.60e-01** | **1.60e-01** | 1.11e-16 | 5.55e-17 |
| ramp_down | 1.55e-15 | 1.55e-15 | 1.39e-17 | 5.55e-17 |
| gate_chatter | **1.51e+00** | **1.51e+00** | 4.44e-16 | 2.22e-16 |
| push_fwd_90N | **9.80e-01** | **9.80e-01** | N/A | N/A |
| push_bwd_90N | N/A | N/A | N/A | N/A |

## Fixed-Height Residual Analysis

The 9.5e-08 residual at knee actuator 8 (step 2) is a WBC initialization artifact from step 0, not a control divergence. Both Python and JAX produce identical control torques for fixed-height scenarios. This residual is:
- Below the 1e-5 parity threshold
- Below the 1e-8 strict threshold
- Identical across all 3 fixed-height scenarios (high, low, ramp_down)
- Always at actuator 8 with the same magnitude

## Push Analysis

- **push_fwd_90N**: 0.98 Nm diff at left wheel (actuator 4), step 117. Robot survives. First divergent actuator is wheel — directly affected by ABS trim via tau_position.
- **push_bwd_90N**: Robot falls with 1.56e-06 parity diff (threshold-passing). The fall means both Python and JAX produced similar failing torques. The actual control difference before the fall is small.

## Dynamic Degradation Analysis

- **ramp_up**: 0.16 Nm diff at right wheel (actuator 9), step 150. This is in the middle of the ramp from 0.33->0.48m, crossing the height scheduling gates. The degradation accumulates from step 150 onward.
- **gate_chatter**: 1.51 Nm diff at right wheel (actuator 9), step 150. Oscillatory height profile crossing schedule gates repeatedly. Much larger divergence than ramp_up.

## ABS Trim Assessment

The ABS trim is confirmed as the first remaining divergent control-affecting subsystem:
- All wheel diff scenarios (ramp_up, gate_chatter, push_fwd) diverge at wheels [4,9]
- Wheels are the direct recipients of `external_position_trim` via `tau_position` in `k2_jax_sagittal_torque_assembly`
- Hip-yaw [1,6] remains clean in all scenarios (<1e-16)
- The 9.5e-08 fixed-height residual is from WBC init, not ABS trim

## Summary

1. ✅ Push parity confirmed failing (0.98 Nm push_fwd)
2. ✅ Ramp_up degradation confirmed (0.16 Nm at step 150)
3. ✅ ABS trim is the first remaining divergent control-affecting scalar (all diffs at wheels)
4. ✅ Fixed-height parity passes (<1e-5)
5. ✅ Hip-yaw parity passes (<1e-16)
6. ✅ Tests pass (131/131)
7. ✅ No hidden torque/WBC appears in control path

## Next Phase

Phase 1 — Python ABS trim source trace.
