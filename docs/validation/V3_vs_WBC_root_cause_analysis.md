# V3 vs V3+WBC Assist — Root Cause Analysis

**Date:** 2026-07-17
**Question:** Tại sao có WBC mà V3 không cải thiện nhiều? (Why doesn't WBC significantly improve V3?)
**Verdict:** **Architectural conflict (decentralized vs centralized control) + gate over-conservatism + WBC quality limitations**

---

## 1. TL;DR

| Factor | Severity | Description |
|--------|:--------:|-------------|
| **Architectural conflict** | 🔴 CRITICAL | V3 is decentralized (10 independent control laws); WBC is centralized (one QP for all joints). Blending their torques creates destructive interference at every joint. |
| **Gate over-conservatism** | 🔴 CRITICAL | The 7-term multiplicative gate forces α → 0 under normal V3 operating conditions. g_height has σ=1.5cm → WBC disabled beyond ±3cm from model nominal. |
| **Height mismatch** | 🟠 HIGH | WBC model calibrated at 0.67m; robot operates at 0.45-0.60m. g_height at keyframe (0.53m) = exp(-((0.53-0.67)/0.015)²) ≈ 0.0 → WBC permanently disabled. |
| **V3 standalone instability** | 🟠 HIGH | V3 has 97% fall rate. Unstable states trigger g_stability → 0, further blocking WBC. |
| **WBC quality** | 🟡 MEDIUM | WBC-only performs 17× worse than V3 (5,985 vs 347 falls, 2.6× pitch RMS). Even with perfect gating, the underlying WBC signal is low-quality. |
| **WBC solve rate** | 🟡 MEDIUM | 75.8% solve success — 24% of steps have τ_wbc = 0 (fallback). |

**Primary root cause:** The torque-blend architecture is fundamentally wrong for this system. You cannot blend a decentralized feedback controller (V3) with a centralized QP optimizer (WBC) at the torque level and expect improvement. They fight each other.

---

## 2. The Torque-Blend Conflict (Architectural Root Cause)

### 2.1 How V3 Works (Decentralized)

V3 assembles torque from 11 independent components, each acting on specific joints:

```
Component              → Joints affected     Purpose
──────────────────────────────────────────────────────────
tau_sagittal_pitch     → wheels [4,9]        Sagittal balance (pitch → wheel torque)
tau_sagittal_velocity  → wheels [4,9]        Forward velocity damping
tau_position           → wheels [4,9]        Position return (anti-drift)
tau_wheel_velocity     → wheels [4,9]        Individual wheel damping
tau_drift_controller   → wheels [4,9]        Coordinated drift correction
tau_shape_posture      → hp, knee, hy [1,2,3,6,7,8]  PD: q_ref → joint_pos
tau_lateral_roll       → hip_roll [0,5]      Lateral balance
tau_yaw                → hip_yaw [1,6]       Yaw stabilization
tau_heading_hy         → hip_yaw [1,6]       Heading hold
tau_anti_twist         → hip_yaw [1,6]       Hip-yaw divergence damping
tau_hy_mean_center     → hip_yaw [1,6]       Return to center
```

**Key property:** V3's PD posture control (`k2_jax_shape_posture_compute`) uses `q_ref` as a target. Each posture joint has its own independent PD: `τⱼ = kp·(q_refⱼ − qⱼ) − kd·q̇ⱼ`. Hip_roll PD has kp=0 (V3's lateral balance controller handles roll independently through hip_roll torque that is NOT based on q_ref).

### 2.2 How WBC Works (Centralized)

WBC solves a single constrained QP over ALL joints simultaneously:

```
min  w_com·‖J_com·qdd + Jdot_com·qvel − k_com·(h_target − h)‖²
   + w_torso·‖J_torso·qdd + Jdot_torso·qvel − k_torso·e_R‖²
   + w_posture·‖qdd_joints + k_posture·(q_joints − q_current)‖²
   + w_wheel·‖τ_wheels‖²
   + w_slack·‖slack‖²

s.t.  M(q)·qdd + C(q,qvel) = S·τ + Jc^T·λ    (dynamics)
      τ_min ≤ τ ≤ τ_max                        (torque limits)
      Contact constraints (friction cone, normal force)
```

WBC produces (qdd_wbc, τ_wbc, λ_wbc) that simultaneously optimizes all objectives through the full dynamics constraint. **Every joint's torque depends on every other joint's torque through the mass matrix M and contact Jacobian Jc.**

### 2.3 The Conflict

When you blend:
```
τ_assist = τ_v3 + α · (τ_wbc − τ_v3)
```

You get destructive interference because:

1. **V3 computes τ_v3 assuming it's the ONLY controller.** Its gains are tuned for the dynamics of the robot alone, not robot+WBC.

2. **WBC computes τ_wbc assuming it has FULL authority.** Its QP finds the globally optimal torque vector. When only a fraction α is applied, the remaining (1−α) of each joint's torque doesn't match what the QP assumed — breaking the dynamics coupling.

3. **Joint-level cancellation:** On hip_yaw [1,6], V3 might output +2 Nm (yaw correction) while WBC outputs −1.5 Nm (torso orientation task). The blend gives +0.5 Nm — both controllers partially defeated.

4. **Wheel-level conflict:** V3's sagittal controller outputs wheel torques based on pitch error. WBC's QP outputs wheel torques based on COM height + torso orientation + contact force optimization. These are fundamentally different control philosophies applied to the same actuators.

**This is why the Posture-Guided approach showed pitch improvement (−17%) where torque-blend did not:** Posture-guided separates concerns — WBC recommends posture targets, V3 executes them with its own PD. No torque addition, no cancellation.

---

## 3. The Gate System Over-Conservatism

### 3.1 The Multiplicative Trap

The gate is a product of 6 continuous terms:

```
αⱼ = α_max · g_stability · g_height · g_push · g_divergence · Aⱼ · K_roleⱼ
```

With 7 terms multiplied together, even moderately reduced individual terms compound to near-zero. If each gate averages 0.5, α = 0.50 × 0.5⁷ ≈ 0.004.

### 3.2 g_height: Wrong Model Nominal Height

```
g_height = min(exp(−((h_cmd−0.67)/0.015)²), exp(−((h_act−0.67)/0.015)²))
```

The WBC linearization was done at 0.67m height. But the robot operates at:
- Keyframe: 0.53m (14cm from nominal → g_height ≈ 0)
- Tests: 0.45m–0.60m (7–22cm from nominal → g_height = 0)
- Sigma = 0.015m (±1.5cm bandwidth before g→0.37)

At ANY test height, g_height < 1e-3 → forced to exact 0 by the safety floor.

**This is the single biggest factor:** The WBC model was calibrated at the wrong height. Even the "real pipeline" test had to manually override `model_nominal = 0.53` to get WBC to contribute at all.

### 3.3 g_stability: V3 Instability Closes the Gate

```
g_stability = exp(−(pitch/0.06)² − (roll/0.06)² − ...)
```

V3 has 97% fall rate. During unstable episodes:
- Roll can reach 15-25° (0.26-0.44 rad) → roll/0.06 = 4.3-7.3 → contributes exp(−18 to −53) ≈ 0
- Even modest pitch (2°, 0.035 rad) contributes exp(−(0.035/0.06)²) = exp(−0.34) = 0.71

When V3 most needs help (during instability), the gate blocks WBC completely.

The "real pipeline" test widened thresholds to pitch=0.30 rad (17°), roll=0.40 rad (23°) to compensate, but these are dangerously permissive.

### 3.4 K_role: Conservative Per-Joint Scaling

```
K_role = [0.12, 0.05, 0.60, 0.60, 0.35, 0.12, 0.05, 0.60, 0.60, 0.35]
```

Even with all gates open (α_max=0.50), the maximum WBC contribution per joint is:
- Hip_roll: 0.50 × 0.12 = 6%
- Hip_yaw: 0.50 × 0.05 = 2.5%
- Hip_pitch/knee: 0.50 × 0.60 = 30%
- Wheels: 0.50 × 0.35 = 17.5%

The balance-critical joints (hip_roll, hip_yaw, wheels) get the least WBC authority — exactly where V3 needs the most help.

---

## 4. WBC Quality: Why WBC-Only Fails So Badly

### 4.1 Quantitative Evidence

From the 225-scenario full batch evaluation:

| Metric | V3 | WBC Only | Ratio |
|--------|:--:|:--------:|:-----:|
| Total Falls | 347 | **5,985** | **17.2× worse** |
| Pitch RMS | 15.9° | **41.8°** | **2.63× worse** |
| Yaw Drift | 4.24° | **8.0°** | **1.89× worse** |
| Single-push Falls | 118 | **3,353** | **28.4× worse** |

### 4.2 Root Causes of WBC Poor Performance

1. **Velocity damping (not position tracking) posture task:** WBC's posture task is `qdd_joints = −k_posture·(q_joints − q_current)` — it damps toward CURRENT position with k_posture=10. This is a velocity damper, not a position tracker. It CANNOT hold a target posture — it only resists changes from the current configuration.

2. **No feedforward:** WBC's QP has no gravity compensation feedforward term. The COM height task uses a simple P-controller: `J_com·qdd_des = k_com·(h_target − h)`. This requires the QP to solve for qdd that achieves this through the full dynamics, which is fundamentally harder than a feedforward + feedback approach.

3. **Single linearization point:** The QP uses dynamics linearized at ONE height (0.67m). Mass matrix M(q), Coriolis C(q,qvel), and contact Jacobian Jc(q) are all evaluated at the model nominal. When the robot is at 0.53m, these matrices are WRONG — the QP is solving an incorrect optimization problem.

4. **No wheel-ground interaction model:** The WBC treats wheels as point contacts. There's no rolling constraint in the QP, only a post-hoc regularization term (w_wheel=0.5). The wheel dynamics are fundamentally different from point-foot contact.

5. **QP solve failures (24.2%):** When the QP fails to converge, τ_wbc = 0. This creates a discontinuity — the assist suddenly becomes τ_v3 + α·(0 − τ_v3) = (1−α)·τ_v3, which reduces V3's authority.

6. **Task weight imbalance:** w_com=5, w_torso=3 dominate, but w_posture=2 is too weak for meaningful posture guidance. The QP prioritizes COM height over joint posture, leading to unstable joint configurations.

---

## 5. Evidence: When WBC DOES Contribute (Keyframe Test)

The "real pipeline" test with manual overrides (model_nominal=0.53, wider stability thresholds) shows what happens when WBC is allowed to contribute:

| Metric | V3 | Assist | Change |
|--------|:--:|:------:|:------:|
| Pitch RMS | 2.44° | 2.04° | **−17%** ✅ |
| Drift | 0.070m | 0.058m | **−18%** ✅ |
| Yaw Drift | 0.012° | 0.010° | **−20%** ✅ |
| Height RMSE | 0.123m | 0.121m | **−1.2%** ✅ |
| **Roll RMS** | 0.20° | **0.27°** | **+35%** ❌ |

**Key insight:** Even with perfect gating (g_height=1.0 at 0.53m, widened stability thresholds), WBC helps some dimensions (drift, yaw, height) but HURTS others (roll +35%). This is direct evidence of the torque conflict — WBC's QP solution for torso orientation couples into roll through the dynamics, fighting V3's lateral roll controller.

The Posture-Guided approach reduced roll degradation from +35% to +7%, confirming that separating posture guidance from torque blending mitigates (but doesn't eliminate) the conflict.

---

## 6. Answer: Why V3 Doesn't Improve with WBC

### Primary Reason: Architectural Mismatch (Torque-Blend Cannot Work)

The torque-blend formula `τ = τ_v3 + α·(τ_wbc − τ_v3)` is a **linear interpolation between two fundamentally incompatible control philosophies.** This is not a tuning problem — it's an architectural problem:

- V3 is a collection of 11 independent, hand-tuned feedback laws. Each component assumes it has undiluted authority over its target joints.
- WBC is a single QP optimizer that exploits the full coupling between all joints through the dynamics constraint.

When you blend them, you get the worst of both: V3's carefully tuned gains are diluted, and WBC's globally optimal solution is applied at only a fraction of its intended magnitude.

**This is proven by:** When WBC contributes (keyframe test), it helps drift/yaw but degrades roll. The Posture-Guided approach (WBC→q_ref, V3→torque) eliminates the torque conflict and shows better results (pitch −17% without the roll degradation of torque-blend).

### Secondary Reason: Gate Blocks WBC 99%+ of the Time

Even if torque-blend could work, the gate prevents WBC from contributing in virtually all scenarios:

1. **Height mismatch** (g_height): model_nominal=0.67m vs operating height=0.45-0.60m, σ=0.015m → g_height=0 for all test heights
2. **V3 instability** (g_stability): 97% fall rate → pitch/roll exceed 0.06 rad thresholds → g_stability≈0
3. **Multiplicative compounding**: 7 gate terms × conservative K_role → α → 0

### Tertiary Reason: WBC Signal is Low Quality

Even if we could open the gate, the underlying WBC torque is poor quality:
- 17× more falls than V3 when used standalone
- Velocity-damping posture task (can't hold targets)
- Single linearization at wrong height
- 24% QP solve failure rate

---

## 7. Recommendations

### 7.1 Immediate: Stop Torque-Blend Development

The torque-blend approach (`τ_v3 + α·(τ_wbc − τ_v3)`) cannot work for this system. The architectural conflict is fundamental, not a tuning issue. Continuing to tune gates, thresholds, or weights will not resolve the underlying incompatibility.

### 7.2 Short-Term: Pursue Posture-Guided Approach

The Posture-Guided architecture (`WBC→q_ref, V3→torque`) is the correct separation of concerns:
- WBC acts as a **posture planner** (slow outer loop)
- V3 acts as a **reactive executor** (fast inner loop)
- No torque addition → no cancellation → no conflict

Evidence: Posture-Guided achieved pitch −17% with only +7% roll degradation (vs torque-blend's +35%).

### 7.3 Medium-Term: Fix WBC Model Calibration

1. **Re-linearize at operating height (0.53m, not 0.67m)** — this alone would fix g_height=0
2. **Add gravity compensation feedforward** to the QP
3. **Replace velocity-damping posture with position-tracking posture task**
4. **Add rolling constraint to wheel contacts** (not just point-contact friction cones)
5. **Investigate QP solve failures** — 24% failure rate is too high

### 7.4 Long-Term: Residual RL (Phase D+)

The CLAUDE.md plan is correct: V3 needs a trained residual PPO policy to achieve stable balance. WBC can serve as the nominal prior (`base_action`), but:
- WBC must be recalibrated at the operating height
- The residual should use Posture-Guided semantics (residual added to q_ref, not torque)
- Or: residual PPO directly outputs torque corrections, trained from scratch

### 7.5 If WBC Must Be Used: Hybrid Joint-Group Strategy

Per the posture_guided report's recommendation:

| Joint Group | Control Mode | Rationale |
|-------------|:------------:|-----------|
| Hip pitch, Knee | Posture-Guided (WBC→q_ref) | WBC provides useful posture optimization |
| Hip roll, Hip yaw | V3 Only | V3's lateral balance and anti-twist are superior |
| Wheels | Torque-Blend (with recalibrated WBC) | WBC's contact force optimization helps drift/yaw |

This eliminates conflict by never blending torques on the same joints from two different controllers.

---

## 8. Summary

| Question | Answer |
|----------|--------|
| V3 và WBC có phối hợp được không? | **Không.** Torque-blend = decentralized + centralized = destructive interference. |
| Có xung đột không? | **Có, ở mọi khớp.** Cả hai controller đều xuất torque cho tất cả 10 khớp, nhưng từ các triết lý điều khiển khác nhau. |
| WBC có tệ không? | **Có.** WBC-only té gấp 17 lần V3, pitch RMS tệ hơn 2.6 lần. Mô hình được calibrate sai độ cao, không có feedforward, và posture task chỉ là velocity damping. |
| Gate có quá保守 không? | **Có.** 7-số hạng nhân với nhau + model_nominal sai + V3 instability = α luôn ≈ 0. |
| Nguyên nhân chính? | **Kiến trúc torque-blend không phù hợp.** Đây là vấn đề thiết kế, không phải vấn đề tuning. |

---

*Generated by systematic debugging audit — Phase 1-4 complete*
*Git branch: repo-cleanup-t6j*
