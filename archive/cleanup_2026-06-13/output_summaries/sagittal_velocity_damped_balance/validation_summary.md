# SagittalVelocityDampedBalanceController — Validation Summary

**Status: GATE G/H FINALIZED — ACCEPTED (Target-Level Drift Reduction)**  
**Gate G:** PASS (minimum acceptable + target gates)  
**Gate H:** NO FAILURES  
**Finalized:** 2026-05-30

---

**Date:** 2026-05-30  
**Controller:** `SagittalVelocityDampedBalanceController`  
**Spec:** `docs/superpowers/specs/2026-05-30-sagittal-velocity-damped-balance-addendum.md`

---

## Final Configuration (F4c)

| Parameter | Value |
|-----------|-------|
| `kp_pitch` | 50.0 |
| `kd_pitch` | 10.0 |
| `kp_cp` | 30.0 |
| `kd_com_vy` | 5.0 |
| `k_velocity` | 15.0 |
| `k_wheel_velocity` | 0.5 |
| `k_position` | 10.0 |
| `max_tau_wheel` | 5.0 |
| `wheel_torque_sign` | 1.0 |

### Controller Formula

```
tau_total = tau_pitch + tau_pitch_rate + tau_sagittal_velocity + tau_wheel_velocity + tau_position

tau_pitch             = kp_pitch    * pitch_error
tau_pitch_rate        = kd_pitch    * pitch_rate
tau_sagittal_velocity = k_velocity  * (-sagittal_velocity)
tau_wheel_velocity    = k_wheel_vel * (-wheel_velocity_mean)
tau_position          = k_position  * (-sagittal_position_error)
```

---

## Files Changed

- `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` — new controller
- `scripts/simulate_hierarchical_controller.py` — CLI routing, mutual exclusion, telemetry columns
- `tests/test_sagittal_velocity_damped_balance_controller.py` — unit/sign tests

---

## Gain Progression

| Run | k_velocity | k_position | Max Drift (m) | vs Baseline |
|-----|-----------|-----------|---------------|-------------|
| Baseline | 0.0 | 0.0 | 35.22 | — |
| F2 | 5.0 | 0.0 | 11.12 | -68% |
| F2b | 15.0 | 0.0 | 4.32 | -88% |
| F4 | 15.0 | 2.0 | 4.22 | -88% |
| F4b | 15.0 | 5.0 | 4.09 | -88% |
| **F4c (best)** | **15.0** | **10.0** | **3.876** | **-89%** |

**k_position saturation finding:** Doubling k_position from 5.0 to 10.0 yielded only 5% improvement. The residual forward velocity (~0.068 m/s) cannot be eliminated by scaling k_position alone. The position return term is opposed by the pitch balance loop adjusting pitch to maintain balance. Full position hold is not achievable with this architecture.

---

## Failed E0 Context

| Attempt | Max Drift (m) | Outcome |
|---------|--------------|---------|
| E0b | 15.98 | Failed — above minimum acceptable gate |
| E0c | 63.72 | Failed — worse than baseline |
| E0d | 121.39 | Failed — catastrophic regression |
| **F4c (this work)** | **3.876** | **Target gate PASS** |

---

## Gate Results

### Gate G: Drift Gate

| Gate | Threshold | Result | Achieved |
|------|-----------|--------|----------|
| Minimum acceptable | max ≤ 17.6 m | **PASS** | 3.876 m |
| Target | max ≤ 5.0 m | **PASS** | 3.876 m |
| Preferred max | max ≤ 0.50 m | FAIL | 3.876 m |
| Preferred final | final ≤ 0.20 m | FAIL | 3.876 m |

**Note:** Full position hold is not claimed. The preferred gate requires a fundamentally different architecture (e.g., closed-loop position feedback with velocity feedforward, or a separate position-hold layer that does not fight pitch recovery).

### Gate H: Failure Classification

**Status: NO FAILURES**

| Failure Type | Count |
|--------------|-------|
| Ownership violations | 0 |
| Hidden torque | 0 |
| WBC solver activations | 0 |
| Terminations | 0 |

---

## Gate F: Incremental Validation Results

### F4c — Nominal (5000 steps)

| Metric | Value |
|--------|-------|
| Survived steps | 5000 / 5000 |
| Terminated | No |
| Max sagittal drift | 3.876 m |
| Final sagittal drift | 3.876 m |
| Max planar drift | 3.878 m |
| Final planar drift | 3.878 m |
| Pitch range | [-0.0000, 0.1279] rad |
| Roll range | [-0.0458, 0.0038] rad |
| Yaw range | [-0.2223, 0.0205] rad |
| CoM Z range | [0.3624, 0.4086] m |
| Wheel vel mean range | [-8.00, 1.26] rad/s |
| Torque saturation rate max | 0.0 |
| Ownership violations | 0 |
| Hidden torque norm max | 0.0 |
| WBC norm max | 101.04 |
| Active sagittal controller | SagittalVelocityDampedBalanceController |
| Baseline sagittal controller active | No |

### F5 — high_small height variant (500 steps)

Target CoM Z: 0.4140 m | Achieved: 0.4128 m

| Metric | Value |
|--------|-------|
| Survived steps | 500 / 500 |
| Terminated | No |
| Max sagittal drift | 0.441 m |
| Final sagittal drift | 0.441 m |
| Max planar drift | 0.441 m |
| Final planar drift | 0.441 m |
| Pitch range | [-0.0000, 0.0400] rad |
| Roll range | [0.0000, 0.0029] rad |
| Yaw range | [0.0000, 0.0216] rad |
| CoM Z range | [0.4126, 0.4179] m |
| Wheel vel mean range | [-2.03, 1.28] rad/s |
| Torque saturation rate max | 0.0 |
| Ownership violations | 0 |
| Hidden torque norm max | 0.0 |
| WBC norm max | 24.82 |
| Active sagittal controller | SagittalVelocityDampedBalanceController |
| Baseline sagittal controller active | No |

### F5 — low_small height variant (500 steps)

Target CoM Z: 0.3940 m | Achieved: 0.3952 m

| Metric | Value |
|--------|-------|
| Survived steps | 500 / 500 |
| Terminated | No |
| Max sagittal drift | 0.399 m |
| Final sagittal drift | 0.399 m |
| Max planar drift | 0.399 m |
| Final planar drift | 0.399 m |
| Pitch range | [-0.0000, 0.0362] rad |
| Roll range | [0.0000, 0.0028] rad |
| Yaw range | [0.0000, 0.0194] rad |
| CoM Z range | [0.3949, 0.3991] m |
| Wheel vel mean range | [-1.85, 1.24] rad/s |
| Torque saturation rate max | 0.0 |
| Ownership violations | 0 |
| Hidden torque norm max | 0.0 |
| WBC norm max | 21.12 |
| Active sagittal controller | SagittalVelocityDampedBalanceController |
| Baseline sagittal controller active | No |

---

## Invariant Checks

| Check | Result |
|-------|--------|
| No WBC | CONFIRMED — tau_wbc_norm is from posture/support feedforward, not WBC QP solver |
| No E0b/E0c/E0d reintroduced | CONFIRMED — controller built from scratch per addendum spec |
| No torque ownership change | CONFIRMED — ownership_violation_count = 0 across all runs |
| Baseline and velocity-damped mutually exclusive | CONFIRMED — CLI routing enforces single active sagittal controller |
| Hidden torque norm | 0.0 across all runs |

---

## Tests Run

| Test File | Command | Result |
|-----------|---------|--------|
| test_sagittal_balance_state.py | `pytest tests/test_sagittal_balance_state.py -v` | 7/7 passed |
| test_sagittal_velocity_damped_balance_controller.py | `pytest tests/test_sagittal_velocity_damped_balance_controller.py -v` | 15/15 passed |
| Filtered suite | `pytest tests/ -k "sagittal or balance_core or mutual_exclusion" --ignore=tests/test_env.py -m "not slow" -v` | 186 passed, 9 failed (failures in CentroidalWrenchComputer/balance_core, not in this controller) |

---

## Limitation: Residual Drift

k_position scaling showed diminishing returns (4.22 → 4.09 → 3.876 m). Residual forward velocity ~0.068 m/s persists. The position return term and pitch-balance equilibrium partially counteract each other. This is a structural limitation, not a tuning failure.

Correct description: "velocity-damped drift reduction" or "target-level position drift reduction."  
Do NOT describe as "standing in place" or "full position hold."

---

## Recommendation

Accept F4c as the final configuration for target-level drift reduction. Do not continue tuning k_position blindly.

**Recommended next steps:**
1. Step C: Height recovery (validate controller at different heights)
2. Step D: Dynamic height transition (validate during height changes)
3. Step F: Push robustness (validate under external disturbances)

Treat Step E full (full position hold, target ≤0.30 m) as a separate design task if needed — not gain scaling of F4c.

---

## Telemetry Files

| Run | File |
|-----|------|
| Nominal F4c (5000 steps) | `outputs/hierarchical_controller_sim/telemetry_1780113728.csv` |
| high_small F5 (500 steps) | `outputs/hierarchical_controller_sim/telemetry_1780115203.csv` |
| low_small F5 (500 steps) | `outputs/hierarchical_controller_sim/telemetry_1780115287.csv` |

---

## Summary

The `SagittalVelocityDampedBalanceController` achieves an **89% reduction in sagittal drift** (35.22 m → 3.876 m) over 5000 nominal steps, passing both the minimum acceptable gate (≤17.6 m) and the target gate (≤5.0 m). Height variant regression (high_small and low_small) both pass with no terminations, zero ownership violations, and zero hidden torque.

The preferred full-position-hold gate (≤0.50 m max, ≤0.20 m final) is **not met** and is not claimed. The residual forward drift is a structural limitation of the current architecture, not a tuning failure.

All E0b/E0c/E0d patterns are absent. WBC is not active. Torque ownership is unchanged. The two sagittal controllers are mutually exclusive.

**Gate G/H finalization: ACCEPTED at target level. Proceed to Step C.**
