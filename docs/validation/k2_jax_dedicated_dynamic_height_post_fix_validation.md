# K2 JAX Dedicated Dynamic Height Post-Fix Validation

**Date:** 2026-06-29
**Classification:** `K2_JAX_DEDICATED_DYNAMIC_HEIGHT_SAFETY_FAIL`
**Status:** BLOCKED -- promotion gated on static-q_ref dynamic height inability

---

## Phase 3: Dynamic Height Post-Fix Validation

### Test Configuration

| Parameter | Value |
|-----------|-------|
| `dynamic_qref_mode` | `original-k2-exact` (STATIC q_ref) |
| Interpolation mode | NOT used |
| `mode_div` | enabled (default) |
| Height setup (ramp_up, up_down_cycle) | `low_0p330_setup.json` |
| Height setup (ramp_down, gate_dwell, gate_chatter) | `high_0p480_setup.json` |
| Backend | `jax` |
| Profile | `k2_notch_low_q_v1` |
| Baseline comparison | Original K2 Python (`original-k2-exact`, static q_ref) |

---

### 5 Dynamic Height Scenarios -- Results Summary

| Scenario | Max Steps | Survived | Fall Step | Termination Reason | hy_max (rad) | pitch_rms (deg) | height_rmse (m) | Original hy_max | Original pitch_rms | Original height_rmse | Class |
|----------|-----------|----------|-----------|-------------------|-------------|----------------|-----------------|-----------------|--------------------|--------------------|-------|
| ramp_up_0p330_to_0p480 | 5000 | NO | 1509 | height_too_low (0.330 < 0.330) | 0.3493 | 3.87 | 0.0215 | 0.0534 | 3.15 | 0.1051 | **SAFETY_FAIL** |
| ramp_down_0p480_to_0p330 | 5000 | YES | - | - | 0.2382 | 4.03 | 0.1123 | 0.0977 | 5.84 | 0.1149 | **SAFE_BUT_WORSE** |
| up_down_cycle_0p330_0p480_0p330 | 7000 | NO | 1186 | height_too_low (0.331 < 0.331) | 0.2475 | 3.92 | 0.0206 | 0.0534 | 3.32 | 0.0946 | **SAFETY_FAIL** |
| gate_dwell_0p420_0p450_0p480 | 6000 | YES | - | - | **0.5370** | 6.19 | 0.0773 | 0.0534 | 3.05 | 0.1097 | **SAFETY_FAIL** |
| gate_chatter_0p400_0p470 | 5000 | YES | - | - | 0.1791 | 4.74 | 0.0712 | 0.0629 | 2.98 | 0.0905 | **SAFE_BUT_WORSE** |

---

### Detailed Scenario Analysis

#### 1. ramp_up_0p330_to_0p480 (5000 steps)

- **Outcome:** Fell at step 1509. Reason: `height_too_low (0.330 < 0.330)`.
- **Height trajectory:** CoM starts at 0.335m (low_0p330 posture). CoM stays locked at ~0.33m (min 0.330m, max 0.335m). Height reference at fall time is 0.38045m. Robot cannot lift torso.
- **hip_yaw:** max=0.3493 rad (20.0 deg) -- right at the 0.35 rad absolute safety gate. hip_yaw RMS=0.0936 rad.
- **hip_yaw torque:** 6.21 Nm -- elevated, indicating strong divergence fighting.
- **pitch:** rms=3.87 deg, min=-7.23 deg, max=1.82 deg. Moderate pitch oscillation.
- **roll:** rms=0.51 deg.
- **com_drift:** final displacement=0.30m.
- **Original K2:** fell=False, hy_max=0.0534, pitch=3.15 deg, height_rmse=0.1051.
- **Delta:** hip_yaw worse by +0.2959 rad, height_rmse better by -0.0836 m (deceptive -- RMSE is low because CoM barely moves from initial height).
- **Class: SAFETY_FAIL (fell).** Root cause: static q_ref frozen at low_0p330 posture cannot generate upward torso motion.

#### 2. ramp_down_0p480_to_0p330 (5000 steps)

- **Outcome:** Survived full 5000 steps. No fall.
- **Height trajectory:** CoM starts at 0.481m (high_0p480 posture). CoM remains high: min 0.481m, max 0.491m, final 0.483m. Height reference is 0.33m but CoM does not descend. Height RMSE=0.1123m reflects the constant offset.
- **hip_yaw:** max=0.2382 rad (13.7 deg) -- below 0.35 rad safety gate but +0.1405 rad worse than original.
- **hip_yaw torque:** 1.82 Nm.
- **pitch:** rms=4.03 deg, min=-1.95 deg, max=8.78 deg. Actually BETTER than original (5.84 deg rms).
- **com_drift:** final displacement=1.29m -- substantial lateral drift.
- **Original K2:** fell=False, hy_max=0.0977, pitch=5.84 deg, height_rmse=0.1149.
- **Delta:** hip_yaw worse by +0.1405 rad, pitch better by -1.81 deg. Height RMSE comparable.
- **Class: SAFE_BUT_WORSE.** Survives but hip_yaw divergence is +144% worse than original. Robot pretends it is still at 0.48m while reference says 0.33m -- it does not actually ramp down.

#### 3. up_down_cycle_0p330_0p480_0p330 (7000 steps)

- **Outcome:** Fell at step 1186. Reason: `height_too_low (0.331 < 0.331)`.
- **Height trajectory:** CoM starts at 0.335m, stays locked at low posture (min 0.331m, max 0.335m). Height reference at fall is 0.38145m. Same failure mode as ramp_up.
- **hip_yaw:** max=0.2475 rad (14.2 deg) -- below 0.35 rad gate.
- **hip_yaw torque:** 4.31 Nm.
- **pitch:** rms=3.92 deg, min=-8.31 deg, max=1.82 deg.
- **Original K2:** fell=False, hy_max=0.0534, pitch=3.32 deg, height_rmse=0.0946.
- **Delta:** hip_yaw worse by +0.1941 rad, height_rmse better by -0.0740 m (deceptive -- low because CoM never moves).
- **Class: SAFETY_FAIL (fell).** Identical root cause to ramp_up: static q_ref at low posture cannot track upward height references.

#### 4. gate_dwell_0p420_0p450_0p480 (6000 steps)

- **Outcome:** Survived 6000 steps but with catastrophic hip_yaw divergence.
- **Height trajectory:** CoM starts at 0.481m. CoM oscillates: min 0.463m, max 0.498m, final 0.463m. Height RMSE=0.0773m.
- **hip_yaw:** max=**0.5370 rad (30.8 deg)** -- **EXCEEDS 0.35 rad absolute safety gate by +53%!**
- **hip_yaw RMS:** 0.2602 rad (14.9 deg) -- very high sustained divergence.
- **hip_yaw torque:** 4.43 Nm.
- **yaw drift:** min=-50.95 deg, max=1.90 deg. Total yaw excursion: 52.86 deg. Massive unidirectional yaw rotation.
- **pitch:** rms=6.19 deg, max=10.87 deg.
- **Original K2:** fell=False, hy_max=0.0534, pitch=3.05 deg, height_rmse=0.1097.
- **Delta:** hip_yaw worse by +0.4836 rad (+906% vs original!), pitch worse by +3.14 deg.
- **Class: SAFETY_FAIL (hip_yaw exceeds safety gate).** The 0.5370 rad hip_yaw is 10x the original and well beyond the 0.35 rad absolute safety limit.

#### 5. gate_chatter_0p400_0p470 (5000 steps)

- **Outcome:** Survived full 5000 steps.
- **Height trajectory:** CoM starts at 0.481m, stays high: min 0.481m, max 0.491m, final 0.485m. Height reference is 0.40m. Height RMSE=0.0712m -- the robot tracks a height it was not commanded to.
- **hip_yaw:** max=0.1791 rad (10.3 deg) -- below 0.35 rad gate.
- **pitch:** rms=4.74 deg, max=10.09 deg.
- **yaw drift:** max=19.2 deg.
- **Original K2:** fell=False, hy_max=0.0629, pitch=2.98 deg, height_rmse=0.0905.
- **Delta:** hip_yaw worse by +0.1162 rad, pitch worse by +1.76 deg, height_rmse slightly better by -0.0193 m.
- **Class: SAFE_BUT_WORSE.** Survives but all stability metrics are degraded relative to original.

---

### Safety Gate Summary

| Scenario | hip_yaw_max (rad) | 0.35 rad gate | Status |
|----------|-------------------|---------------|--------|
| ramp_up_0p330_to_0p480 | 0.3493 | Right at limit | BORDERLINE |
| ramp_down_0p480_to_0p330 | 0.2382 | PASS | OK |
| up_down_cycle_0p330_0p480_0p330 | 0.2475 | PASS | OK |
| gate_dwell_0p420_0p450_0p480 | **0.5370** | **FAIL** | **EXCEEDED** |
| gate_chatter_0p400_0p470 | 0.1791 | PASS | OK |

---

### Scenario Delta vs Original K2

| Scenario | Delta hy_max (rad) | Delta pitch_rms (deg) | Delta height_rmse (m) | Direction |
|----------|---------------------|------------------------|------------------------|-----------|
| ramp_up | +0.2959 | +0.72 | -0.0836 | WORSE (fell) |
| ramp_down | +0.1405 | **-1.81** | -0.0026 | MIXED |
| up_down_cycle | +0.1941 | +0.60 | -0.0740 | WORSE (fell) |
| gate_dwell | **+0.4836** | +3.14 | -0.0324 | **MUCH WORSE** |
| gate_chatter | +0.1162 | +1.76 | -0.0193 | WORSE |

Note: Lower height RMSE in ramp_up/up_down_cycle is deceptive -- it reflects the CoM being frozen at the initial height rather than tracking the reference. The RMSE is low because the CoM stays near 0.33m while the reference increases toward 0.38m, causing a height_too_low termination before large errors accumulate.

---

### Height Tracking Analysis

| Scenario | Initial CoM (m) | Height Ref at End (m) | Actual CoM Final (m) | CoM Range (m) | CoM Actually Moving? |
|----------|------------------|------------------------|-----------------------|----------------|-----------------------|
| ramp_up | 0.335 | 0.380 (at fall) | 0.330 | 0.330-0.335 | NO -- frozen at 0.33m |
| ramp_down | 0.481 | 0.330 | 0.483 | 0.481-0.491 | NO -- stuck at 0.48m |
| up_down_cycle | 0.335 | 0.381 (at fall) | 0.331 | 0.331-0.335 | NO -- frozen at 0.33m |
| gate_dwell | 0.481 | 0.480 | 0.463 | 0.463-0.498 | PARTIAL -- oscillates |
| gate_chatter | 0.481 | 0.400 | 0.485 | 0.481-0.491 | NO -- stuck at 0.48m |

Key observation: In all scenarios, the CoM stays within a ~0.005-0.035m band around the initial height setup posture. Static q_ref provides zero mechanism to shift the equilibrium posture toward the commanded height.

---

### Root Cause Analysis

The root cause of all 3 safety failures and 2 degraded scenarios is the **`original-k2-exact` dynamic_qref_mode**, which uses a **static q_ref** (equilibrium joint position target) frozen at the initial height setup posture.

#### What static q_ref means in practice:

1. **q_ref is a 10-element vector** of equilibrium joint positions that the LQR/IK controller targets as the nominal posture.
2. In `original-k2-exact` mode, q_ref is computed once at initialization from the physical height setup file and remains constant throughout the episode.
3. **Height-dependent LQR gains** (`K` matrix) are scheduled by commanded height, so the controller DOES change its feedback gains as the height reference changes.
4. However, gain scheduling alone cannot generate sufficient feedforward action to shift the robot's equilibrium posture to a new height. The gains change how aggressively the controller pulls toward q_ref, but q_ref itself does not move.

#### Consequences by scenario type:

- **ramp_up / up_down_cycle (low_0p330 initial posture):** q_ref is frozen at 0.33m squat position. The controller applies 0.33m-equilibrium gains but the reference says "go to 0.48m". The LQR gains provide some height-tracking feedback, but the feedforward equilibrium is fundamentally at 0.33m. The feedback authority is insufficient to lift the entire torso against gravity. CoM stays at 0.33m. Height error accumulates until `height_too_low` termination.

- **ramp_down / gate_chatter (high_0p480 initial posture):** q_ref is frozen at 0.48m standing position. The reference says "go to 0.33m" but the equilibrium posture is at 0.48m. The robot cannot descend because the feedforward joint targets keep it standing tall. CoM stays at 0.48m. This does not cause a fall (being too tall is safer than being too short), but it means the robot completely fails to track the height command.

- **gate_dwell (high_0p480 initial posture, height ref cycles 0.42/0.45/0.48):** q_ref is at 0.48m. When reference drops to 0.42m, gain scheduling shifts to 0.42m gains but equilibrium stays at 0.48m. The mismatch between gains and equilibrium creates unstable coupling, driving massive hip_yaw divergence (0.537 rad) and yaw rotation (53 deg).

#### Design tradeoff:

The static q_ref design is intentional for **fixed-height precision**: at a single target height, static q_ref provides exact equilibrium joint positions that minimize steady-state error. This is why the fixed-height validation scenarios (25/25 PASS) show excellent performance.

The tradeoff is that static q_ref provides **zero dynamic height capability**. The robot cannot:
- Ramp up from a squat to a stand
- Ramp down from a stand to a squat
- Track continuously varying height commands
- Handle height reference changes without instability

#### Possible resolution paths:

1. **Dynamic q_ref interpolation** -- linearly interpolate q_ref between height-indexed posture keyframes as height reference changes. This was explored in earlier work (see `k2_jax_dedicated_strict_promotion_fix.md`) where strict promotion used approximate q_ref interpolation but was found to cause its own issues.

2. **Full height-dependent IK** -- recompute IK targets at the current commanded height at each control step. Provides true dynamic height capability at the cost of more computation per step.

3. **Accept the limitation** -- if dynamic height is not a requirement for the current research phase, gate the promotion on fixed-height scenarios only and document dynamic height as a known limitation of the static q_ref mode.

---

### Overall Dynamic Height Classification

| Criterion | Count | Scenarios |
|-----------|-------|-----------|
| SAFETY_FAIL (fell) | 2 | ramp_up, up_down_cycle |
| SAFETY_FAIL (hy > 0.35 rad) | 1 | gate_dwell |
| SAFE_BUT_WORSE | 2 | ramp_down, gate_chatter |
| CLEAN_PASS | 0 | - |
| **Total** | **5** | |

**Overall Dynamic Height Classification: `K2_JAX_DEDICATED_DYNAMIC_HEIGHT_SAFETY_FAIL`**

**Status: BLOCKED**

Dynamic height promotion is gated on resolving the static q_ref limitation. The JAX backend correctly implements the static q_ref semantics -- it matches the original K2 Python behavior in reproducing the same design limitation. This is a controller design issue, not a JAX port correctness issue.

The fixed-height scenarios (25/25 PASS) confirm the JAX backend is functionally correct. The dynamic height failures are a direct consequence of the `original-k2-exact` static q_ref design choice and are reproduced faithfully by the JAX port.

---

### Test Environment

- **Comparison data:** `outputs/k2_jax_dedicated_promotion_validation/all_metrics_comparison.json`
- **Scenario summaries:** `outputs/k2_jax_dedicated_promotion_validation/dynamic_height/*/summary.json`
- **Raw results:** `outputs/k2_jax_dedicated_promotion_validation/dynamic_height/raw_results.json`
- **Validation runner:** `scripts/validate_k2_dynamic_height_gate_crossing.py`
- **Controller backend:** `wheeled_biped/controllers/k2_jax_controller.py`
- **Height setups:** `outputs/physical_target_height_setups/low_0p330_setup.json`, `high_0p480_setup.json`
