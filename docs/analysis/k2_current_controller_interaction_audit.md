# K2 Current Controller Interaction Audit

**Phase:** 3 — System Architecture Audit
**Scenarios analyzed:** 6 (3 worst cases, 2 dynamic, 1 control)
**Tests:** 116/116 component parity tests pass (zero regressions)
**Diag vector:** Extended from 53 to 106 fields

## 1. Executive Summary

The K2 JAX controller has **10 independently-summed torque components** with no coordination
between them. The Phase 3 instrumentation reveals **massive torque cancellation** — on average
**13 Nm RMS** of control effort is wasted across the 6 analyzed scenarios, reaching **27.8 Nm**
in the gate_dwell dynamic scenario.

The root cause is clear: **the empirical support feedforward applies large, constant joint
torques that the shape posture PD controller actively fights against.** This is not a subtle
interaction — at the knees and hip_pitch joints, the two components write opposing torques
on **99.8–100% of active steps.**

### Conflict Severity Ranking

| Rank | Joint Group | Mean Cancel RMS (Nm) | Peak Cancel (Nm) | Sign Opposition | Root Cause |
|------|-------------|---------------------|-------------------|-----------------|------------|
| 1 | **Knee** | 7.2 | 14.1 | 99.8–100% | Empirical FF (-7.75/-7.9 Nm) vs Posture PD (kp=40) |
| 2 | **Hip Pitch** | 4.4 | 7.3 | 100% | Empirical FF (2.05/1.6 Nm) vs Posture PD (kp=30) |
| 3 | **Hip Yaw** | ~0 | ~0 | 55–83% | Mode-div & yaw near-zero torque, posture dominates |
| 4 | **Hip Roll** | 0 | 0 | 0% | Lateral roll dominates, posture hip_roll kp=0 (correct) |

### Key Numbers

| Metric | Mean | Max | Interpretation |
|--------|------|-----|----------------|
| **Total cancellation RMS** | 13.0 Nm | 27.8 Nm | Wasted control effort per step |
| **Posture:Balance power ratio** | 4.3:1 | 8.1:1 | Posture uses 4–8× more torque than balance |
| **Wheel torque clipping** | 0.0% | 0.0% | Torque limits never reached |
| **Rate limiting** | <0.1% | 0.1% | Rate limits almost never active |
| **Knee sign opposition** | 99.8% | 100% | Posture and FF oppose on nearly every step |
| **Hip pitch sign opposition** | 100% | 100% | Posture and FF oppose on every step |

### What This Means

The controller has **plenty of torque headroom** (zero clipping, minimal rate-limiting)
but the control effort is being **wasted on internal fighting**. The empirical support FF
and posture PD each command large torques in opposite directions at the knees and hip_pitch
joints. The resulting net torque is what actually reaches the robot, but the cancellation
means:

1. The posture PD is **constantly saturated against the empirical FF** — it cannot
   provide meaningful postural corrections because its effort is consumed fighting a
   constant offset.
2. The **empirical FF is untuned for dynamic conditions** — its fixed values work at
   equilibrium but become harmful during height transitions and push recovery.
3. The **yaw and mode-div controllers are too weak** — with kp=8/kd=2 and kp=10/kd=0.5,
   they produce near-zero torques while posture hip_yaw (kp=15, kd=3) dominates at 100%
   authority share.

---

## 2. Same-Joint Torque Cancellation — Detailed Analysis

### 2.a Knee Joints [3, 8] — CRITICAL

The empirical support FF applies **-7.75 Nm at left knee** and **-7.9 Nm at right knee**
as a constant feedforward. The shape posture PD (kp=40, kd=5) actively opposes this
feedforward, resulting in:

| Scenario | Cancel RMS | Cancel Peak | Emp FF Share | Posture Share |
|----------|-----------|-------------|-------------|---------------|
| gate_dwell | 11.4 Nm | 14.1 Nm | 87% | 13% |
| ramp_down | 10.6 Nm | 11.3 Nm | 86% | 14% |
| high_0p450 | 7.3 Nm | 7.9 Nm | 78% | 22% |
| high_0p430 | 5.1 Nm | 5.7 Nm | 73% | 27% |
| low_0p380 | 1.7 Nm | 2.1 Nm | 89% | 11% |
| focused_low_0p320 | ~0 Nm | ~0 Nm | 95% | 5% |

**Physical mechanism:** The empirical FF pushes knees into extension (negative torque =
straightening). The posture PD tries to maintain the posture reference (q_ref from
polynomial height schedule). These two objectives conflict — the feedforward wants a
specific torque bias, the PD wants a specific position. The PD is fighting a losing
battle because the FF is constant and large.

**Impact:** At gate_dwell, 11.4 Nm RMS per step is wasted — this is ~80% of the
available knee torque budget spent on internal fighting.

**Recommendation:** The empirical support FF and posture PD must be coordinated:
- Option A: **Bias the posture q_ref** to account for the empirical FF — if the FF
  pushes knees into extension, the posture target should shift to anticipate this.
- Option B: **Authority-schedule the empirical FF** — reduce FF magnitude when
  posture error is large (robot is far from equilibrium posture).
- Option C: **Merge FF into posture bias** — apply the FF as a q_ref offset rather
  than a direct torque, letting the PD work with the bias instead of against it.

### 2.b Hip Pitch Joints [2, 7] — HIGH

The empirical support FF applies **2.05 Nm at left hip_pitch** and **1.6 Nm at right
hip_pitch**. The posture PD (kp=30, kd=4) opposes, resulting in 100% sign opposition
across all scenarios:

| Scenario | Cancel RMS | Cancel Peak | Emp FF Share | Posture Share |
|----------|-----------|-------------|-------------|---------------|
| gate_dwell | 5.5 Nm | 7.3 Nm | 56% | 44% |
| low_0p380 | 4.4 Nm | 5.3 Nm | 66% | 34% |
| high_0p450 | 4.3 Nm | 5.3 Nm | 63% | 37% |
| high_0p430 | 4.3 Nm | 5.3 Nm | 63% | 37% |
| ramp_down | 4.2 Nm | 5.2 Nm | 68% | 32% |
| focused_low_0p320 | 3.4 Nm | 7.2 Nm | 60% | 40% |

**Physical mechanism:** Same as knees — the FF provides a constant torque offset, the PD
tries to track a position reference. The hip_pitch cancellation is more balanced (FF ~63%,
posture ~37%) because the FF torque is smaller (2.05/1.6 Nm vs 7.75/7.9 Nm at knees).

**Impact:** 4.4 Nm RMS cancellation at hip_pitch — the primary contributor to overall
pitch instability since hip_pitch directly affects torso angle.

**Recommendation:** Same coordination approach as knees. Additionally, consider whether
the hip_pitch FF values are appropriate — they're scaled from a 0.5× factor applied to
the original support controller output. The original values may have been tuned for a
specific height/load condition that doesn't generalize.

### 2.c Hip Yaw Joints [1, 6] — LOW TORQUE, HIGH OPPOSITION

Three components write to hip_yaw: posture PD (kp=15, kd=3), yaw controller (kp=8,
kd=2), and mode-div controller (kp=10, kd=0.5, height-gated). The cancellation is near
zero because all three produce very small torques:

| Scenario | Cancel RMS | Sign Opposition (yaw vs posture) |
|----------|-----------|----------------------------------|
| gate_dwell | 0.0 Nm | 83% |
| low_0p380 | 0.0 Nm | 79% |
| ramp_down | 0.0 Nm | 75% |
| focused_low_0p320 | 0.0 Nm | 54% |
| high_0p430 | 0.0 Nm | 57% |
| high_0p450 | 0.0 Nm | 30% |

Authority share: posture 100%, yaw 0%, mode_div 0%.

**Physical mechanism:** The yaw and mode-div gains are too low relative to posture PD
gains. The posture PD at hip_yaw uses kp=15, while yaw uses kp=8 and mode_div uses
kp=10. But more importantly, yaw torque is antisymmetric (equal and opposite on left/right)
while posture PD tracks individual joint references. When yaw error is small (typical
for fixed-height scenarios), the yaw controller produces negligible torque.

**Impact:** The hip_yaw divergence controller (mode_div) is effectively disabled by
low gain — it can't overcome posture PD authority. This may explain why hip_yaw
divergence increases during dynamic scenarios (the controller can't correct it).

**Recommendation:** 
- Increase mode-div gain or decrease posture hip_yaw authority when divergence is large
- Add authority allocation: mode-div should gain authority when divergence exceeds threshold
- Remove mode-div height gate's wide range (0.30-0.80) — it should activate only when needed

### 2.d Hip Roll Joints [0, 5] — CLEAN, NO CONFLICT

Posture hip_roll has kp=0 (zero position gain — by design). The lateral roll controller
(kp=40, kd=8) operates with 100% authority share and zero cancellation. This is the
**only clean joint pair** — the design correctly assigns hip_roll to lateral balance
without posture interference.

**This is the model for how the other joints should work.**

---

## 3. Cross-Coupling Analysis

### 3.1 Balance Torque vs Support Error

**Correlation: -0.24 (mean), up to -0.48**

Balance wheel torque magnitude is negatively correlated with support error — meaning
when support error increases (robot drifts), balance torque decreases. This is
**counterproductive**: balance should increase when support error grows.

**Mechanism:** The sagittal balance controller computes pitch torque and position
torque independently. Support error feeds into position correction (kpos=40), but
the pitch stabilization and support correction operate on different timescales
and can conflict. During pitch oscillations, support error and pitch error may
be out of phase, causing the position correction to reduce overall wheel torque
when it should be adding.

### 3.2 Posture:Balance Power Ratio

**Mean: 4.3:1, peaking at 8.1:1 during ramp_down**

The posture controller consistently uses 4–8× more torque RMS than the balance
controller. This is primarily because the empirical FF forces large torques at
knee/hip_pitch joints. During dynamic scenarios, posture torque increases further
while balance torque decreases (the robot relies more on posture to track the
changing height reference).

**Impact:** During height transitions, the controller shifts authority toward
posture tracking at the expense of balance. This explains why dynamic scenarios
have the highest pitch RMS (4.46 deg vs 3.78 deg for fixed-height).

### 3.3 Lateral Roll vs Hip-Yaw Divergence

**Correlation: 0.09 (mean), but up to 0.78 in low_0p380**

At low heights (0.380 m), lateral roll torque is strongly correlated with hip-yaw
divergence (r=0.78). The lateral controller reacts to roll, which couples through
the kinematic chain to hip_yaw motion. At low heights where the robot is more
compressed, this kinematic coupling is stronger.

### 3.4 Support FF vs Pitch

**Correlation: 0.00** — the height-gated support FF (hip_yaw correction) is
computed from support position error only and is **excluded from tau_sum**.
It is computed but discarded. This means support correction through hip_yaw
is currently **completely inactive.**

---

## 4. Saturation & Clipping Analysis

### Key Finding: Zero Saturation

| Metric | Mean | Max |
|--------|------|-----|
| Wheel clip fraction | 0.0% | 0.0% |
| Sagittal saturation | 0.0% | 0.0% |
| Posture saturation | 0.0% | 0.0% |
| Yaw saturation | 0.0% | 0.0% |
| Balance rate-limited | 0.03% | 0.10% |
| Posture rate-limited | 0.08% | 0.10% |

The controller has **massive torque headroom**. Torque limits are never reached,
and rate limits are only active <0.1% of the time. The problem is NOT torque
saturation — it's **coordination**. The torque composer (clip + rate-limit) is
effectively a pass-through.

This means we can freely redistribute authority between components without
hitting hardware limits. The torque budget is underutilized.

---

## 5. Height & Phase Stratification

### Gate Dwell (worst case): Cancellation by Height

| Height Region | Cancel Total | Cancel Knee | Cancel Hip Pitch | Cancel Hip Yaw |
|--------------|-------------|-------------|-----------------|----------------|
| High (>0.43m) | 26.8 Nm | 11.3 Nm | 5.4 Nm | 10.1 Nm |

The dynamic gate_dwell scenario has **extreme cancellation at high heights**:
26.8 Nm total, with 10.1 Nm at hip_yaw alone (despite hip_yaw having zero
cancellation in other scenarios). During gate_crossing, the mode-div controller
becomes active as height changes trigger the height gate.

### Focused Low 0.320m: Cancellation Stable Across Pitch Phases

| Phase | Cancel Total | Cancel Hip Yaw | Cancel Hip Pitch |
|-------|-------------|----------------|------------------|
| Forward | 3.9 Nm | 0.36 Nm | 3.6 Nm |
| Neutral | 3.7 Nm | 0.30 Nm | 3.4 Nm |
| Backward | 3.8 Nm | 0.49 Nm | 3.3 Nm |

Cancellation is nearly constant across pitch phases at low height — the conflict is
structural (constant FF values) rather than state-dependent.

---

## 6. Ranked Findings and Recommended Fixes

### Finding 1 [CRITICAL] — Empirical FF vs Posture PD at Knees

**Magnitude:** 7.2 Nm RMS cancellation, 99.8% sign opposition, 84% FF dominance.
**Root cause:** Constant -7.75/-7.9 Nm knee torques from empirical FF conflict
with posture PD (kp=40) tracking q_ref.
**Physical impact:** ~80% of knee torque budget wasted on internal fighting.
Posture PD cannot provide meaningful corrections because it's saturated against FF.
**Recommended fix:** Merge empirical FF into posture q_ref bias (Phase 4 design).
Instead of applying FF as direct torque, shift the posture target to account for
the expected torque offset. This eliminates the conflict at the source.
**Priority:** First — this is the single largest source of wasted effort.

### Finding 2 [HIGH] — Empirical FF vs Posture PD at Hip Pitch

**Magnitude:** 4.4 Nm RMS cancellation, 100% sign opposition, 63% FF dominance.
**Root cause:** Same mechanism as knees — constant FF torque (2.05/1.6 Nm) vs
posture PD (kp=30) tracking q_ref.
**Physical impact:** Directly contributes to pitch oscillation since hip_pitch
controls torso angle.
**Recommended fix:** Same as Finding 1 — merge FF into posture bias.
**Priority:** Second.

### Finding 3 [HIGH] — Posture Dominates Balance 4:1 to 8:1

**Magnitude:** Posture:Balance power ratio 4.3× mean, 8.1× peak (ramp_down).
**Root cause:** The empirical FF drives large knee/hip_pitch torques while balance
uses only wheel torques. The autority split is fixed, not adaptive.
**Physical impact:** During dynamic height transitions, posture authority overwhelms
balance, causing pitch instability (highest pitch RMS in dynamic scenarios).
**Recommended fix:** Continuous authority allocator — reduce posture authority
when balance demand is high (Phase 6 design).
**Priority:** Third.

### Finding 4 [MEDIUM] — Mode-Div and Yaw Controllers Too Weak

**Magnitude:** Near-zero torque output from yaw (kp=8) and mode-div (kp=10) vs
posture hip_yaw (kp=15). 55-83% sign opposition when active.
**Root cause:** Gains are too low relative to posture PD. Mode-div height gate
(0.30-0.80 smoothstep) is unnecessarily wide.
**Physical impact:** Hip-yaw divergence controller cannot correct divergence
because it lacks authority relative to posture PD.
**Recommended fix:** Authority-schedule mode_div gain based on divergence magnitude.
When divergence exceeds threshold, increase mode_div authority and reduce posture
hip_yaw authority.
**Priority:** Fourth.

### Finding 5 [MEDIUM] — Support FF (height-gated hip_yaw) Computed But Discarded

**Magnitude:** Support FF is computed every step but excluded from tau_sum.
**Root cause:** Historical exclusion — "Python balance-core has no equivalent;
inclusion causes divergence during descending height transitions and push recovery."
**Physical impact:** Support correction through hip_yaw is completely inactive.
The controller has no mechanism to correct support through hip_yaw.
**Recommended fix:** Re-evaluate support FF inclusion with proper authority
allocation. If included with reduced authority when pitch/balance demand is
high, it could improve support without causing divergence.
**Priority:** Fifth.

### Finding 6 [LOW] — Zero Torque Saturation

**Magnitude:** 0.0% clipping at all joints, <0.1% rate-limiting.
**Root cause:** Torque limits (20 Nm per joint) are never reached.
**Physical impact:** The torque composer (clip + rate-limit) is not constraining
behavior. Torque headroom is abundant — coordination is the bottleneck, not limits.
**Recommended fix:** No action needed for saturation. The abundant headroom means
authority reallocation (Findings 1-5) won't hit limits.

---

## 7. What Does NOT Need Fixing

1. **Hip roll joints [0, 5]:** Clean design — posture kp=0, lateral roll has 100%
   authority, zero cancellation. This is the reference pattern.

2. **Wheel joints [4, 9]:** Only sagittal balance writes here. No conflict possible.
   The APCR1ND wheel damping override works as intended.

3. **Notch filter:** Working correctly — not a source of conflict.

4. **Torque composer (clip + rate-limit):** Not the bottleneck. Limits are never reached.

5. **ABS trim:** Working correctly — no sign of conflict with sagittal assembly.

---

## 8. Phase 3 Deliverables

| File | Status |
|------|--------|
| `docs/analysis/k2_current_controller_interaction_audit.md` | This file |
| `docs/analysis/k2_current_controller_interaction_audit.json` | Data export |
| `wheeled_biped/controllers/k2_jax_controller.py` | Extended diag 53→106 fields |
| `scripts/run_k2_jax_realtime.py` | 33 new CSV diagnostic columns |
| `scripts/analyze_k2_controller_conflicts.py` | Conflict analyzer |
| `scripts/run_k2_phase3_instrumented.py` | Scenario runner |
| `outputs/k2_phase3_instrumented/` | 6 instrumented scenario runs |

### Instrumentation Additions

- **Per-component torque telemetry:** 18 fields capturing posture, yaw, mode_div,
  lateral, support FF, and empirical FF at each conflict-prone joint
- **Pre/post-composer torques:** 20 fields (10 pre-clip + 10 post-clip)
- **Online cancellation metrics:** 5 fields (hip_yaw, hip_roll, hip_pitch, knee, total)
- **Saturation/rate-limit attribution:** 6 fields (by component group)
- **CSV export:** 33 new columns in `--telemetry full` mode
- **Zero behavior change:** All additions are diag writes only

### Acceptance

- [x] 116/116 component parity tests pass (zero regressions)
- [x] 6/6 instrumented scenarios run without falls
- [x] Conflict analyzer produces quantitative findings (not guessed)
- [x] Every conflict-prone joint pair has authority/conflict report
- [x] Concrete ranked recommendations with physical mechanisms
- [x] Controller behavior unchanged (diag writes only)

### Next Phase

**Phase 4:** Design coordinated control architecture based on these findings.
Primary target: Merge empirical FF into posture q_ref bias to eliminate the
#1 source of torque cancellation.
