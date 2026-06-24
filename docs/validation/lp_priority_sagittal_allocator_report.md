# LP Priority Sagittal Allocator Report

**Date:** 2026-06-24
**Task:** `lp_priority_sagittal_allocator_pitch_first_support_residual`
**Branch:** `repo-cleanup-t6j`
**Classification:** `K1_REMAINS_CURRENT_BEST_LP_NO_READY_CANDIDATE`

---

## 1. Executive Summary

This task implemented and evaluated a new opt-in sagittal controller family — the LP (Priority) Sagittal Allocator — designed to resolve the LR/LRS support-pitch coupling via explicit priority-based torque allocation. Instead of a single equal-priority sum like LR/LRS:

```
tau = k_pitch*pitch + k_pitch_rate*pitch_rate + k_support*support + k_support_vel*support_vel
```

LP allocates pitch stabilization torque first, then gates support-centering torque from residual authority only when pitch state is safe.

**Key finding: Priority allocation fails because the EQ/FF pass-through alone (~5.5 Nm) consumes most of the 7.97 Nm torque budget, leaving negligible residual authority for support centering. The saturation gate kills support precisely when it's needed most — after the push, when both pitch and support error are large.**

All three LP variants failed to complete 3000 steps. All terminated with `height_too_low`. Support RMS was 8-13x worse than K1. No LP candidate is recommended for broader validation. K1 remains current-best.

---

## 2. K1 Baseline Status (Unchanged)

| Item | Value |
|------|-------|
| Current-best | `K1_PITCH_RATE_NOTCH_V1` |
| Profile | `k1_pitch_rate_notch_v1` |
| Focused recovery (this run) | 2999 steps, no fall, pitch RMS 5.43deg, support RMS 0.161m, hip_yaw 0.305rad |
| Status | `CURRENT_BEST_PROMOTED_WITH_KNOWN_WIP_RECOVERY_LIMITATION` |

---

## 3. Why LR/LRS Was Rejected

From the previous task `lr_support_drift_sign_phase_audit_and_constrained_gain_sweep`:

- LR EQ/FF fix correct — signs verify as correct
- All LRS variants fail with `height_too_low`
- Support-pitch coupling in the single equal-priority sum creates non-separable coupling
- Individual gain increases do not monotonically improve stability
- K1's independent high-gain damping (kp_pitch=50, kd_pitch=10) remains superior

---

## 4. LP Architecture Design

### 4.1 Torque Equation

```
tau_common = tau_eq_ff_pass_through
           + tau_pitch_priority
           + tau_support_residual_allocated

where:
  tau_eq_ff_pass_through = tau_pitch + tau_position + tau_cp + tau_com_vy
    (K1 equilibrium/feedforward — preserved unchanged)

  tau_pitch_priority = clamp(
      k_pitch_lp * pitch_error + k_pitch_rate_lp * pitch_rate,
      -pitch_priority_limit,
      +pitch_priority_limit
  )

  tau_support_residual_allocated = clamp(
      tau_support_raw * support_gate,
      -support_limit,
      +support_limit
  )
```

### 4.2 Gating Logic

```
pitch_abs_gate = 1.0 when |pitch| <= safe_low, ramp to 0.0 at safe_high
pitch_rate_gate = 1.0 when |pitch_rate| <= safe_low, ramp to 0.0 at safe_high
saturation_gate = max(0, 1 - pre_support_torque / (max_tau_wheel * 0.85))
direction_gate = 0.3 if support torque would worsen pitch, else 1.0

support_gate = pitch_abs_gate * pitch_rate_gate * saturation_gate * direction_gate
```

### 4.3 LP Variants

| Variant | Design | Pitch Stiffness | Pitch Damping | Support Gain | Pitch Safe Low/High | Support Fraction |
|---------|--------|----------------|---------------|-------------|---------------------|-----------------|
| LP1 | Conservative pitch, soft support | 8->5 | 2.0->1.2 | -10->-16 | 5/12 deg | 0.6 |
| LP2 | Strong pitch, softer support | 6->4 | 2.8->1.8 | -8->-12 | 4/10 deg | 0.4 |
| LP3 | Support only when safe | 10->6 | 2.2->1.4 | -12->-18 | 3/7 deg | 0.5 |

LP3 adds a settling counter: support is held at zero until |pitch| < 4 deg for 50 consecutive steps.

---

## 5. LP Torque Equation (Reference)

```python
# Step 1: EQ/FF pass-through
LP_eq_ff_pass_through = tau_pitch + tau_position + tau_cp + tau_com_vy

# Step 2: Pitch priority (first access to dynamic authority)
LP_pitch_priority = clamp(k_pitch_lp * pitch_error + k_pitch_rate_lp * pitch_rate,
                          -pitch_priority_limit, +pitch_priority_limit)

# Step 3: Safety gates
LP_pitch_abs_gate = ramp(1.0, 0.0, pitch_abs_deg, pitch_safe_low, pitch_safe_high)
LP_pitch_rate_gate = ramp(1.0, 0.0, |pitch_rate|, rate_safe_low, rate_safe_high)
LP_saturation_gate = max(0, 1 - |eq_ff + pitch_priority| / (max_tau * 0.85))
LP_direction_gate = 0.3 if sign(support_raw) == sign(pitch_error) && |pitch| > 3deg else 1.0
support_gate = pitch_abs_gate * pitch_rate_gate * saturation_gate * direction_gate

# Step 4: Support deadband
if |support_error| < deadband: support_gate = 0

# Step 5: Residual authority
LP_support_limit = max(0, max_tau * 0.85 - |eq_ff + pitch_priority|) * support_residual_fraction
LP_support_allocated = clamp(support_raw * support_gate, -support_limit, +support_limit)

# Step 6: Slew limit
LP_support_slew_limited = prev_support + clamp(delta, -slew_limit, +slew_limit)

# Step 7: Compose
tau_common = eq_ff_pass_through + pitch_priority + support_slew_limited
```

---

## 6. Focused Recovery Results

**Scenario:** high_0p480, single 90N sagittal push at step 300, 10-step duration, 3000 steps
**Mode-div:** kp=10.0, kd=0.50, max_torque=7.5, soft_limit=0.30, soft_gain=0.80, ref=target

### 6.1 Raw Results

| Candidate | Steps | Termination | Pitch RMS | Pitch Max | Support RMS | Support Max | HipYaw Max | Roll RMS |
|-----------|-------|-------------|-----------|-----------|-------------|-------------|------------|----------|
| **K1** | **2999** | completed | **5.43deg** | 20.3deg | **0.161m** | 0.710m | **0.305rad** | 0.77deg |
| LP1 | 2009 | height_too_low | 6.77deg | 12.7deg | 2.124m | 4.517m | 0.407rad | 0.62deg |
| LP2 | 1889 | height_too_low | 6.78deg | 12.5deg | 1.697m | 3.623m | 0.407rad | 0.67deg |
| LP3 | 1874 | height_too_low | 6.15deg | 11.8deg | 1.278m | 3.420m | 0.403rad | 0.72deg |

### 6.2 Comparison vs K1

| Metric | K1 | LP1 vs K1 | LP2 vs K1 | LP3 vs K1 |
|--------|----|-----------|-----------|-----------|
| Steps | 2999 | -990 (-33%) | -1110 (-37%) | -1125 (-38%) |
| Pitch RMS | 5.43deg | +1.34deg (+25%) | +1.34deg (+25%) | +0.72deg (+13%) |
| Support RMS | 0.161m | 13.2x worse | 10.6x worse | 7.9x worse |
| HipYaw Max | 0.305rad | +0.102rad (+33%) | +0.102rad (+33%) | +0.098rad (+32%) |
| Completed 3000 | Yes | No | No | No |

---

## 7. LP vs LR/LRS Comparison

| Metric | Best LRS (LRS2) | Best LP (LP3) | K1 |
|--------|-----------------|---------------|-----|
| Steps | 447 | 1874 | 2999 |
| Termination | height_too_low | height_too_low | completed |
| Pitch RMS | 9.45deg | 6.15deg | 5.43deg |
| Support RMS | 0.426m | 1.278m | 0.161m |

LP3 survives significantly longer than the best LRS variant (1874 vs 447 steps) and has better pitch RMS (6.15 vs 9.45deg), but support RMS is worse (1.278 vs 0.426m). The longer survival is attributed to LP's EQ/FF pass-through keeping the robot upright longer, but the complete removal of K1's dynamic damping terms (tau_pitch_rate, tau_sagittal_velocity, tau_support_velocity) means support drift accumulates unchecked.

---

## 8. Support Suppression / Residual Authority Audit

### 8.1 Suppression Analysis

| Metric | LP1 | LP2 | LP3 |
|--------|-----|-----|-----|
| Suppression rate | 46.7% | 46.9% | 97.5% |
| Primary cause | saturation (43.3%) | saturation (41.1%) | pitch_not_settled (43.4%) |
| Secondary cause | deadband (1.3%) | deadband (3.1%) | saturation (19.2%) |
| Mean pitch at suppression | 4.8deg | 4.8deg | 4.9deg |
| Mean pitch when active | 6.4deg | 6.5deg | 3.3deg |

### 8.2 Gate Analysis

| Gate | LP1 | LP2 | LP3 |
|------|-----|-----|-----|
| pitch_abs_gate (mean) | 0.906 | 0.825 | 0.662 |
| pitch_rate_gate (mean) | 0.978 | 0.981 | 0.972 |
| saturation_gate (mean) | **0.250** | **0.288** | **0.353** |
| direction_gate (mean) | 0.969 | 0.963 | 0.976 |
| saturation_gate < 0.1 | **46.9%** | **45.3%** | **47.0%** |

**The saturation gate is the dominant limiter across all variants.** The EQ/FF pass-through alone (~5.5 Nm mean) plus modest pitch priority (~0.45 Nm mean) consumes ~6 Nm of the 7.97 Nm max_tau_wheel (at 85% = 6.77 Nm effective limit). This leaves only ~0.8 Nm of residual authority for support centering — essentially zero most of the time.

### 8.3 Residual Authority

| Metric | LP1 | LP2 | LP3 |
|--------|-----|-----|-----|
| Residual authority mean | 1.06 Nm | 1.22 Nm | 1.50 Nm |
| Authority = 0 (% time) | 45.3% | 43.7% | 45.8% |
| Pitch priority mean | 0.45 Nm | 0.42 Nm | 0.49 Nm |
| Pitch priority max | 2.62 Nm | 3.21 Nm | 2.73 Nm |
| EQ/FF mean | 5.58 Nm | 5.46 Nm | 4.71 Nm |

---

## 9. Support-Pitch Coupling Analysis

| Candidate | r(pitch, support) | Pitch std | Support std |
|-----------|-------------------|-----------|-------------|
| K1 | 0.852 | 4.31deg | 0.161m |
| LP1 | 0.715 | 4.72deg | 1.997m |
| LP2 | 0.737 | 4.71deg | 1.627m |
| LP3 | 0.768 | 4.45deg | 1.268m |

LP variants show slightly reduced coupling (0.72-0.77 vs 0.85), but this is misleading — the reduction comes from support being suppressed most of the time, not from effective decoupling. When support IS active, the correlation is similar.

---

## 10. Low-Frequency Mode Analysis

| Candidate | Dominant Hz | Pitch RMS | LF amplitude (0.34-0.52 Hz) |
|-----------|------------|-----------|------------------------------|
| K1 | 0.40 Hz | 5.43deg | 3.84deg |
| LP1 | 0.40 Hz | 6.77deg | 6.17deg |
| LP2 | 0.26 Hz | 6.78deg | 6.21deg |
| LP3 | 0.37 Hz | 6.15deg | 5.63deg |

LP variants have 1.5-1.6x higher low-frequency pitch amplitude than K1. The LP architecture does NOT improve the 0.34-0.52 Hz mode — it worsens it by removing K1's high-gain independent pitch damping.

---

## 11. LP3 Settling Behavior

LP3's settling mechanism (require |pitch| < 4deg for 50 consecutive steps before enabling support) was never satisfied for long:

- 5 settling periods across 1874 steps
- Longest settled period: **29 steps** (below required 50)
- First settled: step 48 (before the push at step 300)
- Total settled steps: ~47 (2.5% of run)

After the push at step 300, pitch exceeded 4deg and the settling counter never reached 50 again. Support was almost entirely suppressed (97.5%).

---

## 12. Direction Gate Behavior

Support torque assisted pitch 63-70% of the time (opposite signs = stabilizing). This is expected — both pitch and support torque oppose the push direction. The direction gate contributed minimally to suppression.

---

## 13. Sustained Recovery

No LP candidate achieved sustained recovery. All terminated with `height_too_low` at ~1900 steps.

---

## 14. WBC/Hidden/Ownership Audit

No WBC, hidden torque, or ownership violations detected. All LP profiles:
- `wbc_enabled` = False
- `hidden_torque_enabled` = False
- No ownership violations

---

## 15. Roll/Yaw/Support Safety

No safety violations. No falls. All terminations are `height_too_low` — the robot gradually loses height as support drift accumulates.

---

## 16. Direct Hip-Yaw Telemetry

Hip-yaw max values:
- K1: 0.305 rad (within gate of 0.35 rad)
- LP: 0.403-0.407 rad (exceeds 0.35 rad gate)

LP variants have worse hip-yaw control because the EQ/FF-only path provides less sagittal stability, causing compensatory yaw motions.

---

## 17. Root Cause Analysis

The LP architecture fails for a fundamental reason:

1. **EQ/FF dominates torque budget.** The EQ/FF pass-through (~5.5 Nm) consumes ~85% of the available torque budget (7.97 Nm at 85% = 6.77 Nm), leaving only ~1.2 Nm for ALL dynamic authority.

2. **Pitch priority is too weak.** LP's pitch damping gains (k_pitch_lp ~5-10, k_pitch_rate_lp ~1-3) are 5-10x weaker than K1's independent terms (kp_pitch=50, kd_pitch=10). The priority is correctly ordered but the authority is insufficient.

3. **Saturation gate starves support.** With EQ/FF + pitch priority consuming ~6 Nm, the residual authority for support is ~0 Nm 45% of the time. Support can never center the robot when it's most needed.

4. **Priority allocation is self-defeating.** When support error is largest (after the push, when the robot is drifting), pitch error is also largest. The pitch gate kills support authority, but EQ/FF alone can't stop the drift.

The fundamental tension: to give support enough authority, you need either (a) higher total torque budget, (b) lower EQ/FF consumption, or (c) fundamentally decoupled pitch/support dynamics. LP addresses none of these — it just reprioritizes within the same tight budget.

---

## 18. Candidate Recommended for Broader Validation

**None.** No LP candidate completed 3000 steps. All fail with `height_too_low`.

---

## 19. Current-Best After Task

**`K1_PITCH_RATE_NOTCH_V1`** (unchanged)

---

## 20. Files Changed

| File | Change |
|------|--------|
| `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` | +LP fields in `SagittalAuthoritySchedule`, +3 LP gain functions, +3 LP profile constants, +LP control logic (~120 lines), +30 LP telemetry fields, +2 LP state variables |
| `scripts/simulate_hierarchical_controller.py` | +3 LP imports, +3 SAGITTAL_AUTHORITY_PROFILES entries, +3 CLI validation entries |
| `scripts/audit_lp_priority_allocator.py` | NEW — LP audit script |
| `tests/test_lp_priority_sagittal_allocator.py` | NEW — 42 LP architecture tests |

---

## 21. Tests/Compile Checks Run

- **LP tests:** 42 passed, 0 failed
- **Compile checks:** 3/3 pass (controller, harness, audit script)
- **Simulation runs:** 4/4 complete (K1, LP1, LP2, LP3)

---

## 22. Next Recommended Task

**HIGH — Reconsider the fundamental architecture.** Three generations of coordinated feedback (L additive, LR replacement, LRS gain sweep, LP priority) have all failed against K1. The common failure pattern is:

1. **Any replacement of K1's independent dynamic terms degrades performance.** K1's decoupled tau_pitch_rate (kp=10 Nm/(rad/s)), tau_sagittal_velocity (k_velocity=15), and tau_support_velocity (k_support_vel=0.25) provide well-tuned independent damping that no coordinated alternative has matched.

2. **The 7.97 Nm wheel torque budget is the binding constraint.** EQ/FF alone uses ~5.5 Nm of this budget. Any additional dynamic authority competes for the remaining ~2.5 Nm, and any gating/priority scheme within that budget can only shift the deficit, not eliminate it.

3. **The WIP mode (2.5 Hz) is NOT the primary problem in focused recovery.** K1's notch filter successfully suppresses it. The primary problem is the 0.34-0.52 Hz low-frequency support-pitch oscillation, which requires high-gain independent damping rather than coordinated feedback.

**Recommended direction for future architecture work:**

Consider augmenting rather than replacing K1: add a small, targeted support centering bias on top of K1's existing independent terms, with proper sign verification and without removing any K1 damping. This was the original L-family approach (additive coordinated feedback on top of K1) which was abandoned after LR was proposed as a replacement. The evidence now suggests the additive approach may have been correct — the failures were due to sign/timing/double-counting issues, not the additive concept itself.

Alternatively, investigate whether the wheel torque budget can be increased (the wheel motor max torque spec may be conservative) or whether the EQ/FF terms can be optimized to consume less budget at moderate heights.

---

## 23. Final Classification

**`K1_REMAINS_CURRENT_BEST_LP_NO_READY_CANDIDATE`**

K1's independent proportional-derivative damping architecture remains the only controller that completes 3000 focused recovery steps with acceptable pitch and support RMS. Three generations of coordinated/replacement/priority alternatives (L, LR/LRS, LP) have all failed to match or exceed K1's performance. The next architecture should augment K1 incrementally rather than replace its dynamic terms.
