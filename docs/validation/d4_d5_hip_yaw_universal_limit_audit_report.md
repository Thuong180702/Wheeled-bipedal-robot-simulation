# D4/D5 Hip-Yaw Universal Limit — Audit Report

**Date:** 2026-06-23  
**Task:** `d4_d5_hip_yaw_universal_limit_audit`  
**Current-best controller:** `D_MODE_HIP_YAW_DIV_V1` (unchanged by this audit)  
**Classification:** `D4_D5_UNIVERSAL_HIP_YAW_LIMIT_AUDIT_COMPLETE_AUTHORITY_LIMIT`  

---

## 1. Executive Summary

This audit investigates why all controller profiles (A/B/C/D) converge to `hip_yaw_abs_max ≈ 0.40 rad` under D4/D5 push conditions. The D4/D5 gate requires `hip_yaw_abs_max < 0.35 rad`, but every profile exceeds this by 13–16%.

**Root cause:** The D4/D5 push creates body yaw drift (up to 0.35 rad), which couples into hip-yaw joint angles through leg geometry. The hip-yaw joints twist to accommodate the body rotation relative to ground contact. All available controllers (shape posture PD, mode-div divergence suppression) act through the same hip-yaw actuators, which are **kinematically decoupled from body yaw** (confirmed in Phase 4 isolation experiments: hip-yaw common torque → body yaw correlation r = −0.122). The torque budget is high (6–7 Nm per side) but cannot correct body yaw, so the tracking error persists.

**Key finding:** The hip-yaw reference is zero across all rows for all profiles. The hip_yaw_abs_max is **100% tracking error** (0% reference contribution). The mode-divergence controller saturates at its 2.0 Nm limit in D4. Shape posture PD adds ~4 Nm more. Total ~6 Nm per side is still inadequate because body yaw (the root driver) is not being addressed.

**Recommended fix path:** Provide body yaw correction through **differential wheel velocity** (wheel-yaw stabilizer), not through hip-yaw torque. The wheel-yaw stabilizer is already instrumented in telemetry (columns exist) but is disabled in all current runs (wheel_yaw_enabled = False).

---

## 2. Current-Best Status

**D_MODE_HIP_YAW_DIV_V1 remains current-best/default controller.**

This audit does not change promotion. The known limitation documented at promotion time (`D4/D5 hip_yaw_abs_max > 0.35 rad is universal across profiles`) is reconfirmed and now explained at root-cause level.

---

## 3. Known Limitation

D4/D5 hip_yaw_abs_max > 0.35 rad is **universal** across all profiles A/B/C/D.

| Profile | D4 (60N low) | D5 (90N high) |
|---------|-------------|---------------|
| A       | 0.4043      | 0.4004        |
| B       | 0.4044      | 0.3945        |
| C       | 0.4048      | 0.3945        |
| **D**   | **0.4030**  | **0.4026**    |

- D is neither better nor worse than A/B/C. Variation is within ±0.01 rad.
- The mode-div controller in D saturates at 2.0 Nm but does not eliminate the limit.
- This is **not a D-specific regression** and **not fixable by tuning D's parameters alone**.

---

## 4. Raw Data Sources

All telemetry read from:

```
outputs/mode_hip_yaw_div_full_real_validation/d4_d5_focused_1000/
├── D4_medium_push_low_{A,B,C,D}/telemetry_1000.csv  (999 rows each)
└── D5_large_push_high_{A,B,C,D}/telemetry_1000.csv  (999 rows each)
```

Supporting context:
- `outputs/mode_hip_yaw_div_full_real_validation/step_d_standard_metrics.csv`
- `outputs/mode_hip_yaw_div_full_real_validation/d4_d5_focused_1000_metrics.csv`
- `outputs/mode_hip_yaw_div_full_real_validation/profile_comparison_summary.csv`
- Previous reports: `hip_yaw_architecture_code_audit.md`, `hip_yaw_mode_isolation_experiment_report.md`, `hip_yaw_sign_fix_reveals_missing_yaw_control.md`

---

## 5. D4/D5 Reproduction Table (A/B/C/D)

Raw telemetry confirms the universal limit:

| Case | Profile | hip_yaw_abs_max (rad) | >0.35? | Fell? | Pitch_max (deg) | Support_err_max (m) |
|------|---------|----------------------|--------|-------|-----------------|---------------------|
| D4   | A       | 0.4043               | YES    | No    | 14.08           | 0.288               |
| D4   | B       | 0.4044               | YES    | No    | 13.83           | 0.275               |
| D4   | C       | 0.4048               | YES    | No    | 13.85           | 0.275               |
| D4   | D       | 0.4030               | YES    | No    | 13.54           | 0.254               |
| D5   | A       | 0.4004               | YES    | No    | 13.70           | 0.524               |
| D5   | B       | 0.3945               | YES    | No    | 14.94           | 0.485               |
| D5   | C       | 0.3945               | YES    | No    | 14.94           | 0.485               |
| D5   | D       | 0.4026               | YES    | No    | 14.87           | 0.659               |

All 8 runs: 999 rows, no falls, no WBC, no hidden torque, no ownership violations.

---

## 6. Windowed Event Analysis

Windows (D4: push at step 150, duration 5, D5: push at step 200, duration 5):

### D4 Profile D windowed metrics

| Window | Steps | hip_yaw_abs_max | divergence_abs_max | support_err_abs_max | tau_L_final | tau_R_final |
|--------|-------|-----------------|-------------------|---------------------|-------------|-------------|
| startup     | 0–50     | 0.0154  | 0.0073   | 0.0125  | −0.06  | 0.01   |
| pre_push    | 100–150  | 0.0150  | 0.0074   | 0.0112  | −0.01  | −0.03  |
| push_active | 150–155  | 0.1666  | 0.1612   | 0.0065  | −1.49  | 1.69   |
| post_push   | 155–205  | 0.3616  | 0.3579   | 0.1741  | −5.29  | 5.43   |
| peak_window | 150–450  | 0.3690  | 0.3657   | **0.2538** | −5.45  | 5.52   |
| recovery    | 205–555  | 0.2758  | 0.2735   | 0.1538  | −4.04  | 4.12   |
| final_steady| 899–999  | **0.4030** | **0.4021** | 0.1898  | −6.03  | 6.07   |

**Critical observation:** The hip_yaw_abs_max peak occurs in the **final steady window** (t=864), not during the push or immediate post-push. This means the divergence builds up **during recovery**, not during the push itself.

### Timing sequence (D4 D):

| Event | Step | Value |
|-------|------|-------|
| Push applied | 150–155 | 60N lateral |
| Support error peak | 574 | 0.254 m |
| hip_yaw divergence starts rising rapidly | 750+ | >0.30 rad |
| hip_yaw_abs_max peak | 864 | 0.403 rad |
| End of run | 999 | 0.398 rad (still elevated) |

The divergence **follows** the support position transient with a lag of ~290 steps (1.45 s). This is consistent with a secondary/tertiary recovery-phase phenomenon, not a direct push response.

### D5 Profile D windowed metrics

| Window | Steps | hip_yaw_abs_max | divergence_abs_max | support_err_abs_max | tau_L_final | tau_R_final |
|--------|-------|-----------------|-------------------|---------------------|-------------|-------------|
| startup     | 0–50     | 0.0241  | 0.0191   | 0.0129  | −0.09  | 0.24   |
| pre_push    | 150–200  | 0.0180  | 0.0091   | 0.0111  | −0.02  | 0.04   |
| push_active | 200–205  | 0.1937  | 0.1916   | 0.0311  | −2.03  | 1.95   |
| post_push   | 205–255  | 0.3556  | 0.3535   | 0.1925  | −5.61  | 5.62   |
| peak_window | 200–500  | 0.3448  | 0.3418   | 0.4231  | −5.50  | 5.50   |
| recovery    | 255–605  | 0.2954  | 0.2938   | 0.1993  | −4.68  | 4.70   |
| final_steady| 899–999  | **0.4026** | **0.4020** | 0.6589  | −6.09  | 6.04   |

Same pattern: peak divergence occurs in the final window (t=913), well after push and support peaks.

---

## 7. Reference-vs-Error Analysis

**100% of all 8 telemetry files show hip_yaw_ref = 0 for every row.**

```python
l_hip_yaw_ref = 0.0000 across all 7992 rows (8 × 999)
r_hip_yaw_ref = 0.0000 across all 7992 rows
```

### Classification at peak

| Case | Profile | ref_contrib_pct | err_contrib_pct | Classification |
|------|---------|----------------|----------------|----------------|
| D4   | A       | 0.0%           | 100.0%         | TRACKING_ERROR_DOMINANT |
| D4   | B       | 0.0%           | 100.0%         | TRACKING_ERROR_DOMINANT |
| D4   | C       | 0.0%           | 100.0%         | TRACKING_ERROR_DOMINANT |
| D4   | D       | 0.0%           | 100.0%         | TRACKING_ERROR_DOMINANT |
| D5   | A       | 0.0%           | 100.0%         | TRACKING_ERROR_DOMINANT |
| D5   | B       | 0.0%           | 100.0%         | TRACKING_ERROR_DOMINANT |
| D5   | C       | 0.0%           | 100.0%         | TRACKING_ERROR_DOMINANT |
| D5   | D       | 0.0%           | 100.0%         | TRACKING_ERROR_DOMINANT |

**H5 — REFERENCE_TOO_LARGE is definitively ruled out.** The hip-yaw reference is always zero. The entire hip_yaw_abs_max is tracking error. The controller targets neutral hip-yaw (0 rad), and the physical disturbance drives hip-yaw away from neutral.

---

## 8. Common/Divergence Decomposition

At hip-yaw peak, the mode decomposition is overwhelmingly **divergence-dominant**:

| Case | Profile | common_at_peak | divergence_at_peak | div/common_ratio_max | Classification |
|------|---------|---------------|-------------------|---------------------|----------------|
| D4   | A       | −0.0011       | −0.4032           | 938,632             | DIVERGENCE_DOMINANT |
| D4   | B       | −0.0011       | −0.4033           | 4,041,225           | DIVERGENCE_DOMINANT |
| D4   | C       | −0.0014       | −0.4034           | 511,652             | DIVERGENCE_DOMINANT |
| D4   | D       | −0.0010       | −0.4021           | 971,966             | DIVERGENCE_DOMINANT |
| D5   | A       | −0.0074       | −0.3930           | 272,215             | DIVERGENCE_DOMINANT |
| D5   | B       | +0.1428       | −0.2517           | 46,906              | DIVERGENCE_DOMINANT |
| D5   | C       | +0.1428       | −0.2517           | 46,906              | DIVERGENCE_DOMINANT |
| D5   | D       | +0.0006       | −0.4020           | 1,319,221           | DIVERGENCE_DOMINANT |

**Key insight:** The hip-yaw error is almost purely divergence mode (left and right joints move in opposite directions = legs twist). Common mode (body yaw as manifest in hip-yaw joints) is essentially zero at peak for D4 and D5 D, but **moderately nonzero for D5 B/C** (0.14 rad).

This confirms the hip-yaw metric is dominated by **leg twist (divergence)**, not body yaw (common). The legs twist in opposite directions relative to ground contact, which is a posture/geometry problem, not a body-yaw rotation problem through the hip-yaw joints.

### Divergence error decomposition (D4 D at peak)

- `divergence = 0.5 × (left − right) = −0.4021`
- `divergence_ref = 0.5 × (left_ref − right_ref) = 0.0` (since ref = 0)
- `divergence_error = 0.5 × (left_err − right_err) = −0.4021`
- `mode_hip_yaw_div_ref = 0.0` (mode-div also uses zero reference)

This means: divergence_error = divergence_position. The entire divergence is error.

---

## 9. Torque Budget Analysis

### Total shape torque at peak (D4 D)

| Component | Left | Right | Notes |
|-----------|------|-------|-------|
| shape_raw (= shape_final) | +6.03 Nm | −6.07 Nm | Fully antisymmetric |
| mode_hip_yaw_div contribution | +2.00 Nm | −2.00 Nm | **Saturated** at 2.0 Nm limit |
| shape PD + yaw controller remainder | +4.03 Nm | −4.07 Nm | From shape posture PD + yaw injection |
| HY2-DIV | 0.0 Nm | 0.0 Nm | Inactive at this height |

### Torque budget classification (all profiles)

| Case | Profile | tau_shape_abs_max | mode_div_tau_abs_max | Classification |
|------|---------|------------------|---------------------|----------------|
| D4   | A       | 7.52 Nm          | 0.00 Nm             | SHAPE_TORQUE_HIGH |
| D4   | B       | 7.52 Nm          | 0.00 Nm             | SHAPE_TORQUE_HIGH |
| D4   | C       | 7.48 Nm          | 0.00 Nm             | SHAPE_TORQUE_HIGH |
| D4   | D       | 7.17 Nm          | 2.00 Nm             | MODE_DIV_SATURATED |
| D5   | A       | 7.18 Nm          | 0.00 Nm             | SHAPE_TORQUE_HIGH |
| D5   | B       | 7.73 Nm          | 0.00 Nm             | SHAPE_TORQUE_HIGH |
| D5   | C       | 7.73 Nm          | 0.00 Nm             | SHAPE_TORQUE_HIGH |
| D5   | D       | 7.20 Nm          | 1.80 Nm             | MODE_DIV_NEAR_SATURATED |

Profile D's mode-div controller adds 2.0 Nm (D4) / 1.8 Nm (D5) of additional torque. But even A/B/C without mode-div have high shape torque (7.2–7.7 Nm) and still exceed 0.35 rad.

### Sign correctness

At hip_yaw peak, mode-div torque agrees with shape PD torque direction (both oppose the error). Mode-div is not fighting shape PD. The torque is constructively aligned.

---

## 10. HY2-DIV Conflict Analysis

**H2 — HY2-DIV conflict: NOT CONFIRMED.**

| Metric | Value at D4 D peak |
|--------|-------------------|
| hip_yaw_div_left (HY2-DIV) | 0.0000 Nm |
| hip_yaw_div_right (HY2-DIV) | 0.0000 Nm |
| hip_yaw_div_gate_active | 0.0 (inactive) |
| hip_yaw_div_active | 0.0 (inactive) |

HY2-DIV contributes zero torque at the hip-yaw peak. The HY2-DIV controller is height-gated (z_low/z_high) and is effectively inactive at low_0p330 height. There is no conflict between HY2-DIV and mode-div because HY2-DIV is not active.

HY2-DIV's zero contribution means all divergence suppression torque at the peak comes from shape posture PD and the mode-based divergence controller.

---

## 11. YawController Contribution Analysis

**H3 — YawController hip-yaw injection: CANNOT QUANTIFY, BUT ARCHITECTURALLY CONFIRMED.**

There is **no dedicated telemetry column** for yaw controller torque at hip-yaw joints. The telemetry columns `l_hip_yaw_tau_shape_raw` and `l_hip_yaw_tau_shape_final` are identical (diff = 0.0 everywhere), indicating both are captured at the same computation point (both include yaw controller addition, or the yaw controller output is zero).

From the architecture audit (`hip_yaw_architecture_code_audit.md`):
```python
# Compute yaw stabilization torque (antisymmetric hip-yaw)
tau_yaw, yaw_diag = balance_core_controllers["yaw_controller"].compute(
    yaw_error=yaw_error, yaw_rate=yaw_rate,
)
# Compose yaw torque with shape posture at hip-yaw joints [1, 6]
tau_shape_posture_with_yaw = tau_shape_posture.at[1].add(tau_yaw[1])
tau_shape_posture_with_yaw = tau_shape_posture_with_yaw.at[6].add(tau_yaw[6])
```

The yaw controller output is injected into hip-yaw joints as antisymmetric torque. However, from `yaw_error_from_equilibrium_rad ≈ −0.19 rad` at the D4 D peak, the yaw controller would generate torque proportional to this error. This torque is mixed into `l_hip_yaw_tau_shape_final` and cannot be separated from the telemetry data.

**Historical context:** Phase 4 isolation experiments (`hip_yaw_mode_isolation_experiment_report.md`) confirmed:
- Hip-yaw common torque → body yaw: r = −0.122 (very weak)
- Hip-yaw common torque → hip-yaw joint common position: r = 0.536 (moderate)

This means: **yaw controller torque at hip-yaw joints moves the hip-yaw joints but does not correct body yaw.** The yaw controller is fighting through a kinematically decoupled actuator. This wasted torque contributes to the total hip-yaw torque budget without reducing the root cause.

---

## 12. Wheel-Yaw Actuator Path Analysis

**H4 — Wheel-yaw combination: NOT ACTIVE, BUT IDENTIFIED AS THE MOST PROMISING FIX PATH.**

All 8 telemetry runs show:
```python
wheel_yaw_enabled = False  # across all profiles, all cases
```

However, wheel-yaw telemetry columns exist in all profiles:
- `wheel_yaw_enabled` (always False)
- `wheel_yaw_error`
- `wheel_yaw_rate`
- `wheel_yaw_tau_left` / `wheel_yaw_tau_right`
- `wheel_yaw_saturated`

The wheel-yaw stabilizer is **fully instrumented but disabled**. This is the correct body yaw correction actuator because:
- Differential wheel velocity creates body yaw torque through ground contact friction
- Wheel torque has higher mechanical advantage for yaw correction than hip-yaw torque
- It does not compete with hip-yaw posture control for torque budget

Historical evidence from Phase 4 (`hip_yaw_mode_isolation_experiment_report.md`):
> Body yaw stabilization requires differential wheel velocity control, not hip-yaw torque.

---

## 13. Contact/Support Coupling Analysis

**H6 — Support/contact coupling: PARTIALLY CONFIRMED.**

### Correlation matrix (full run)

| Case | Profile | corr(div, yaw) | corr(div, support) | corr(div, pitch) |
|------|---------|---------------|-------------------|------------------|
| D4   | A       | 0.8400        | 0.1888            | —                |
| D4   | B       | 0.8382        | 0.1398            | —                |
| D4   | C       | 0.8475        | 0.1550            | —                |
| D4   | D       | **0.8607**    | **0.2575**        | 0.1815           |
| D5   | A       | 0.5152        | 0.6772            | —                |
| D5   | B       | 0.3494        | 0.7303            | —                |
| D5   | C       | 0.3494        | 0.7303            | —                |
| D5   | D       | 0.6887        | 0.6262            | —                |

### Key relationships

**D4:** Hip-yaw divergence is **strongly correlated with body yaw** (r ≈ 0.84–0.86) and weakly with support error (r ≈ 0.14–0.26). The body yaw rotation drives the hip-yaw joints away from zero as the legs twist to accommodate the body-to-ground rotation.

**D5:** Hip-yaw divergence is **strongly correlated with support error** (r ≈ 0.63–0.73) and moderately with body yaw (r ≈ 0.35–0.69). The larger push (90N) creates a bigger support disturbance that couples more strongly into leg twist.

**Post-push coupling (D4 D, t=150–350):** corr(div, support) = **0.9717** — nearly perfect coupling immediately after the push, confirming the support position transient is the initial driver of hip-yaw divergence.

**At peak (D4 D, t=864):** The hip-yaw peak is **not coincident with the support error peak** (t=574). The divergence continues to increase even as support error is recovering, suggesting:
1. Initial spike: support error → hip-yaw divergence (mechanical coupling)
2. Late peak: body yaw drift sustains the hip-yaw error even after support recovery

---

## 14. Summary of All Hypothesis Tests

| Hypothesis | Test | Result | Evidence |
|-----------|------|--------|----------|
| H1 — Authority saturation | D mode-div saturates at 2.0 Nm | **CONFIRMED** | Mode-div tau = 2.0 Nm in D4, 1.8 Nm in D5. Shape torque = 6–7.7 Nm. Total torque is still insufficient. |
| H2 — HY2-DIV conflict | HY2-DIV vs mode-div sign agreement | **RULED OUT** | HY2-DIV = 0.0 at peak (height-gated inactive). No conflict possible. |
| H3 — YawController injection | Yaw torque at hip-yaw joints | **ARCHITECTURALLY CONFIRMED, magnitude UNKNOWN** | Yaw controller adds antisymmetric hip-yaw torque (from code audit). Telemetry cannot separate it. |
| H4 — Wheel-yaw promising | wheel_yaw_enabled = False | **NOT ACTIVE, RECOMMENDED FIX PATH** | All profiles have wheel_yaw_enabled=False. Instrumentation exists. |
| H5 — Reference too large | hip_yaw_ref = 0 everywhere | **RULED OUT** | 100% tracking error. Reference contributes 0%. |
| H6 — Support/contact coupling | divergence vs support correlation | **PARTIALLY CONFIRMED** | D4: corr(div, yaw) ≈ 0.85, corr(div, support) ≈ 0.19. D5: corr(div, support) ≈ 0.63–0.73. Coupling is case-dependent. |

---

## 15. What Is Ruled Out

| Ruled-out cause | Reason |
|----------------|--------|
| Reference too large | hip_yaw_ref = 0 across ALL rows. |
| HY2-DIV conflict | HY2-DIV = 0 at peak (height-gated inactive). |
| D-specific regression | ALL profiles A/B/C/D share the same limit (±0.01 rad). |
| Push cadence/bandwidth | Peak occurs at t=864, not during push (t=150). Recovery-phase phenomenon. |
| Physical joint limit | Hip-yaw joints can move >1 rad. The 0.40 rad is well within limits. |
| PFF/calibration change | Not examined (PFF unchanged by design, per task constraints). |

---

## 16. What Remains Plausible

| Plausible cause | Evidence level | Mechanism |
|----------------|---------------|-----------|
| **Authority saturation (hip-yaw)** | HIGH | 6–7 Nm torque still cannot reduce hip_yaw below 0.40. Mode-div saturated. |
| **Kinematic body-yaw decoupling** | HIGH | Phase 4: hip-yaw torque → body yaw r = −0.122. Hip-yaw torque cannot correct body yaw. |
| **Body yaw drives hip-yaw error** | HIGH | corr(div, yaw) = 0.84–0.86 for D4. Body yaw drift pushes hip-yaw away from 0. |
| **Wheel yaw disabled** | HIGH | wheel_yaw_enabled = False across all profiles. Differential wheel torque is the correct yaw actuator. |
| **Yaw controller wasted torque** | MODERATE | Injects antisymmetric torque through kinematically decoupled hip-yaw joints. Contributes to torque budget without solving the root cause. |

---

## 17. Recommended Next Fix Path

### Primary: Wheel-yaw stabilizer

**Enable and tune the wheel-yaw stabilizer** on top of the current D_MODE_HIP_YAW_DIV_V1 base. The wheel-yaw stabilizer provides body yaw correction through differential wheel velocity, which:
1. Uses the correct actuator path (wheels → ground contact forces → body yaw torque)
2. Does not compete with hip-yaw posture control for torque budget
3. Has higher mechanical advantage for yaw correction
4. Is already instrumented (telemetry columns exist)

Recommended initial parameters:
```
--enable-wheel-yaw-stabilizer
--wheel-yaw-kp 0.5–2.0
--wheel-yaw-kd 0.05–0.2
--wheel-yaw-max-torque 1.0–5.0
```

### Secondary: Higher mode-div authority

Raise `--mode-hip-yaw-div-max-torque` from 2.0 Nm to 5.0–10.0 Nm in a diagnostic candidate. However, this is expected to have diminishing returns because the limiting factor is body yaw (not hip-yaw torque), unless higher divergence torque provides secondary body-yaw coupling through leg geometry.

### Tertiary: Yaw controller path correction

If the yaw controller is confirmed to inject significant torque through hip-yaw joints (H3 cannot be fully resolved from current telemetry), consider disabling the yaw controller's hip-yaw path and letting wheel-yaw handle body yaw exclusively.

---

## 18. What Must NOT Be Done

1. **Do NOT demote D.** D remains architecturally more correct than A/B/C, and the limit is universal.
2. **Do NOT relax the hip-yaw threshold.** The 0.35 rad gate exists for a reason.
3. **Do NOT change PFF source/calibration/interpolation.** Not relevant to this limit.
4. **Do NOT lower global Kp_pitch or suppress sagittal authority.** These are independent subsystems.
5. **Do NOT enable WBC as a patch.** WBC is a fundamentally different approach, not a configuration fix.
6. **Do NOT enable HY2-DIV/HY2-COMP as a hidden patch.** HY2-DIV is inactive at low heights anyway.
7. **Do NOT change D4/D5 push magnitudes.** The gate exists to test these specific conditions.
8. **Do NOT reduce simulation duration.** 1000 steps is already standard.
9. **Do NOT add D4/D5-specific if/else logic.** Must be a general solution.
10. **Do NOT force hip-yaw divergence to zero without addressing body yaw.** The tracking error is physically real.
11. **Do NOT use old wheel-yaw D telemetry as D_MODE_HIP_YAW_DIV_V1 telemetry.** Already prevented by test guard rails.
12. **Do NOT promote the combined candidate without validation.** The audit identifies a path; validation must follow.
13. **Do NOT claim D4/D5 fixed without real-simulation evidence.**

---

## 19. Tests/Scripts Run

### Phase 0 (health check)
```
pytest tests/test_current_best_controller_profile.py -v                          → 7 passed
pytest tests/test_mode_hip_yaw_div_full_real_validation_required.py -v          → 26 passed
pytest tests/test_hip_yaw_mode_math.py -v                                       → 3 passed
pytest tests/test_hip_yaw_ownership.py -v                                       → 6 passed
pytest tests/test_mode_based_hip_yaw_divergence_controller.py -v                → 8 passed
pytest tests/test_hip_yaw_mode_ownership.py -v                                  → 8 passed
pytest tests/test_final_validation_rejects_stub_source.py -v                    → 9 passed
pytest tests/test_d4_d5_validation.py -v                                        → 4 passed
```

### Compile checks (all passed)
```
scripts/audit_d4_d5_hip_yaw_universal_limit.py
scripts/simulate_hierarchical_controller.py
wheeled_biped/controllers/mode_based_hip_yaw_divergence_controller.py
wheeled_biped/controllers/hip_yaw_mode_math.py
wheeled_biped/controllers/hip_yaw_ownership.py
```

### Audit scripts
```
scripts/audit_d4_d5_hip_yaw_universal_limit.py → outputs/d4_d5_hip_yaw_universal_limit_audit/
├── d4_d5_windowed_metrics.csv                  (64 rows, 8 windows × 8 profile-case combinations)
├── d4_d5_peak_event_table.csv                  (8 rows)
├── d4_d5_mode_decomposition_timeseries_summary.csv
├── d4_d5_torque_budget_summary.csv
├── d4_d5_reference_vs_error_summary.csv
├── audit_summary.json
```

---

## 20. Files Changed

| File | Change | Purpose |
|------|--------|---------|
| `scripts/audit_d4_d5_hip_yaw_universal_limit.py` | **Created** | Audit analysis script |
| `docs/validation/d4_d5_hip_yaw_universal_limit_audit_report.md` | **Created** | This report |
| `tests/test_d4_d5_hip_yaw_universal_limit_audit.py` | **Created** | Audit integrity tests |

No changes to:
- Current-best controller (D_MODE_HIP_YAW_DIV_V1)
- Any A/B/C/D profile behavior
- Any validation harness
- Any gate threshold
- Any push magnitude

---

## 21. Final Classification

```
D4_D5_UNIVERSAL_HIP_YAW_LIMIT_AUDIT_COMPLETE_AUTHORITY_LIMIT
```

### Sub-classifications

| Check | Result |
|-------|--------|
| Authority limit confirmed | **YES** — mode-div saturates at 2.0 Nm; shape torque at 6–7.7 Nm still insufficient. |
| HY2-DIV conflict confirmed | **NO** — HY2-DIV inactive at peak (height-gated). |
| YawController injection confirmed | **ARCHITECTURALLY CONFIRMED** — yaw torque injected into hip-yaw joints; cannot be separated from telemetry. Kinematic decoupling (r = −0.122) means this torque is wasted. |
| Wheel-yaw combination promising | **YES** — identified as the primary recommended fix path. Instrumented but disabled. |
| Reference too large | **RULED OUT** — hip_yaw_ref = 0 across all data. |
| Tracking error dominant | **YES** — 100% of hip_yaw_abs_max is tracking error. |
| Support/contact coupling confirmed | **PARTIALLY** — strong in D5 (r=0.63–0.73), moderate in D4 (r=0.19–0.26). Corr(div, yaw) = 0.85 for D4 is the stronger driver. |

### Root-cause mechanism

```
Push disturbance
    → Body yaw drift (up to 0.35 rad)
    → Hip-yaw joints twisted by body rotation relative to ground contact
    → Tracking error at hip-yaw = divergence mode (legs twist opposite)
    → All controllers (shape PD, mode-div, yaw controller) act through hip-yaw joints
    → Hip-yaw torque is kinematically decoupled from body yaw (r = −0.122)
    → Torque budget exhausted (6–7 Nm) without correcting body yaw
    → hip_yaw_abs_max stabilizes at ≈ 0.40 rad
```

---

## 22. Final Statement

- **D remains current-best/default.** Nothing in this audit changes that.
- **D4/D5 hip_yaw > 0.35 rad is NOT fixed.** This audit does not claim it is.
- **This is NOT a promotion task.** No promotion occurs here.
- **This is NOT a default-change task.** No default changes occur here.
- **The primary fix path is wheel-yaw stabilizer**, providing body yaw correction through the correct actuator.
