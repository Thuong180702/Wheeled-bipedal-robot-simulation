# Phase 2: HY2-DIV / Posture-Control Work Audit

**Date:** 2026-06-06
**Phase:** HY2_DIV_POSTURE_FIX_AUDIT
**Classification:** `HY2_DIV_SAFE_BUT_INSUFFICIENT`

---

## 1. What HY2-DIV Was Intended to Fix

### Original Problem
Hip-yaw divergence causes leg twisting inward/outward at extreme heights (0.300m, 0.480m). Per-joint PD torque accelerates divergence 97-99% of the time.

### Intended Fix
Add a divergence-specific damping layer (HY2-DIV) that applies antisymmetric torque proportional to left/right hip-yaw error difference, enabled via height gate.

### Mechanism
```python
divergence = l_hip_yaw - r_hip_yaw
divergence_rate = d(divergence)/dt
tau_div = clip(k_div * divergence + kd_div * divergence_rate, -tau_max, tau_max)
tau_L += tau_div
tau_R -= tau_div
```

### Height Gate
HY2-DIV active when `z_ref < z_high` (default z_low=0.300, z_high=0.393):
- At 0.300m: gate=1.0 (fully active)
- At 0.393m: gate=0.0 (inactive)
- Above 0.393m: gate=0.0 (inactive)

---

## 2. What Was Actually Fixed

### Fixes Completed

| Issue | Fix | Status |
|-------|-----|--------|
| Hip-yaw sign convention | Changed `tau_pd = -(kp*error - kd*vel)` to `tau_pd = kp*error - kd*vel` | ✓ FIXED |
| HY2-DIV gate pass-through | Added z_low/z_high parameters to `build_balance_core_controllers()` | ✓ FIXED |
| HY2-DIV telemetry | Split into hip_yaw_div_enabled/gate_active/effective_k/kd/tau_max | ✓ FIXED |

### Sign Fix Results

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Sign Correct Left (nominal) | 0% | 93.9% | +93.9% |
| Sign Correct Right (nominal) | 0% | 99.7% | +99.7% |
| Sign Correct Left (low_0p300) | 0% | 97.1% | +97.1% |
| Sign Correct Right (low_0p300) | 0% | 98.9% | +98.9% |
| Sign Correct Left (high_0p480) | 0% | 99.3% | +99.3% |
| Sign Correct Right (high_0p480) | 0% | 99.5% | +99.5% |

### Gate Pass-Through Verification

| Candidate | Height | z_high | gate_mean | Expected |
|-----------|--------|--------|-----------|----------|
| A0 | nominal | 0.393 | 0.000 | inactive |
| A0 | low_0p300 | 0.393 | 1.000 | fully active |
| A0 | high_0p480 | 0.393 | 0.000 | inactive |
| B1 | nominal | 0.500 | 0.500 | partially active |
| B1 | low_0p300 | 0.500 | 1.000 | fully active |
| B1 | high_0p480 | 0.500 | 0.028 | weakly active |

**Result:** Candidates produce DIFFERENT gate values. Parameters are passed correctly.

---

## 3. What Remains Unfixed

### Posture Validation Results (A0 5000 steps)

| Height | Survived | Div RMS (rad) | Target (rad) | HY2 Active | Clip% | Result |
|--------|----------|---------------|--------------|------------|-------|--------|
| nominal | ✓ | 0.245 | < 0.10 | 0% | 0% | **FAIL** |
| low_0p300 | ✓ | 0.493 | < 0.30 | 100% | **88.74%** | **FAIL** |
| high_0p480 | ✓ | 0.340 | < 0.25 | 0% | 0% | **FAIL** |

### Failures Observed

| Issue | Status | Evidence |
|-------|--------|----------|
| Divergence at nominal | **UNFIXED** | 0.245 rad vs 0.10 target, gate=0% (inactive) |
| Divergence at low_0p300 | **UNFIXED** | 0.493 rad vs 0.30 target, 88.74% clip |
| Divergence at high_0p480 | **UNFIXED** | 0.340 rad vs 0.25 target, gate=0% (inactive) |
| HY2-DIV authority at low | **INSUFFICIENT** | tau_max=0.5 insufficient |

### Root Cause Analysis (from `hip_yaw_divergence_after_sign_fix_audit.md`)

**Primary:** Per-joint PD cannot control divergence mode
- Div torque always accelerates divergence 97-99% of the time
- Sign fix increased torque coherence, worsening divergence

**Secondary:** No dedicated divergence damping layer active at nominal/high
- HY2-DIV gate=0 at nominal (0.393m) and high_0p480 (0.48m)
- Gate only activates below z_high=0.393m

**Coupling:** High-height divergence correlates with support/roll dynamics
- high_0p480: r=-0.517 (support_error), r=-0.465 (roll), r=-0.755 (l_hip_yaw_vel)

---

## 4. Which HY2-DIV Candidate is Safest So Far

### Candidate Screening Results

| Candidate | k | kd | tau_max | z_high | low clip% | nominal div | Status |
|-----------|---|---|---------|--------|-----------|-------------|--------|
| **A0** | 5.0 | 1.0 | 0.5 | 0.393 | 88.74% | 0.0230 | **PREFERRED** |
| A1 | 5.0 | 1.0 | 1.0 | 0.393 | 0% | 0.0230 | Not selected |
| A2 | 5.0 | 1.0 | 2.0 | 0.393 | 0% | 0.0230 | Not selected |
| B1 | 5.0 | 1.0 | 1.0 | 0.500 | 62.35% | 0.0242 | Not selected |
| B2 | 5.0 | 1.0 | 2.0 | 0.500 | 36.9% | 0.0242 | Not selected |

### Why A0 is Safest

1. **Does not worsen nominal/high:** Gate=0 at nominal/high, so no interference
2. **Conservative authority:** tau_max=0.5 (not aggressive)
3. **Low-height active:** Gate=1 at low_0p300, provides some damping
4. **No clipping regression:** Clipping at low_0p300 indicates insufficient authority, not over-aggressive control

### Why A1/A2 Not Selected

- Higher tau_max (1.0/2.0) eliminated clipping but did NOT improve divergence
- A1/A2 divergence at 5000 steps was WORSE than A0
- More authority without better damping = wasted authority

### Why B1/B2 Not Selected

- Extended gate (z_high=0.500) worsened nominal divergence (0.0242 vs 0.0230)
- B1/B2 have gate active at nominal, introducing interference risk
- Nominal HY2-DIV activity does not improve overall performance

---

## 5. What HY2-DIV Should NOT Do

| Should NOT | Justification |
|-----------|---------------|
| Replace the old baseline | D2 profile passed Step E/C 5/5 without HY2-DIV |
| Enable by default | Baseline works without it; HY2-DIV adds complexity |
| Be treated as Step E pass | No candidate has passed official Step E gates |
| Widen gate without evidence | B1/B2 worsened nominal behavior |
| Increase authority without damping | A1/A2 showed higher authority ≠ better control |

---

## 6. What Posture-First Validation Proved

### Survival/Contact/Height (Priority 1) — ALL PASS

| Metric | nominal | low_0p300 | high_0p480 |
|--------|---------|-----------|------------|
| Survived | ✓ | ✓ | ✓ |
| Contact valid | 99.98% | 99.98% | 99.98% |
| Height error | 0.017m | 0.027m | 0.030m |
| WBC applied | false | false | false |
| Ownership violations | 0 | 0 | 0 |

### Roll (Priority 2) — ALL PASS

| Metric | nominal | low_0p300 | high_0p480 |
|--------|---------|-----------|------------|
| Roll max | 0.014 rad | 0.012 rad | 0.002 rad |
| Roll collapse | No | No | No |

### Divergence (Priority 2) — ALL FAIL

| Metric | nominal | low_0p300 | high_0p480 |
|--------|---------|-----------|------------|
| Divergence RMS | 0.245 rad | 0.493 rad | 0.340 rad |
| Target | < 0.10 rad | < 0.30 rad | < 0.25 rad |
| Result | **FAIL** | **FAIL** | **FAIL** |

### Pitch (Priority 3) — DEFERRED

| Metric | nominal | low_0p300 | high_0p480 |
|--------|---------|-----------|------------|
| Pitch max | 0.089 rad | 0.154 rad | 0.092 rad |
| Classification | RECORDED | RECORDED | RECORDED |

### Support Drift (Priority 4) — DEFERRED

| Metric | nominal | low_0p300 | high_0p480 |
|--------|---------|-----------|------------|
| Support max | 0.159m | 0.110m | 0.378m |
| Classification | RECORDED | RECORDED | RECORDED |

---

## 7. HY2-DIV Classification

```
HY2_DIV_SAFE_BUT_INSUFFICIENT
```

### Rationale

| Criterion | Status |
|-----------|--------|
| Survival maintained | ✓ YES |
| Contact valid | ✓ YES |
| WBC/ownership clean | ✓ YES |
| Roll bounded | ✓ YES |
| HY2-DIV gate working | ✓ YES |
| HY2-DIV authority sufficient | ✗ NO (88.74% clip at low) |
| Divergence controlled | ✗ NO (all heights exceed targets) |
| Official Step E pass | ✗ NO |

### What This Classification Means

- **SAFE:** A0 survived 5000 steps, no collapse, no structural regression
- **BUT:** HY2-DIV A0 authority (tau_max=0.5) is insufficient to control divergence
- **INSUFFICIENT:** Current HY2-DIV design cannot meet posture targets

### What This Does NOT Mean

- HY2-DIV approach is wrong
- Height extension is impossible
- Baseline must be abandoned

---

## 8. Files Created

- `outputs/height_range_extension_strategy_audit/hy2_div_posture_fix_audit.json`
- (this document)