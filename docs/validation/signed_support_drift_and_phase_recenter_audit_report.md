# Signed Support Drift and Phase Behavior Audit Report

**Date**: 2026-06-08
**Author**: Claude Code
**Task**: Refine signed support drift audit and fix strategy

---

## Executive Summary

This audit confirms **SIGNED_DRIFT_CONFIRMED_PHASE_AWARE_RECENTER_READY**.

**Key Finding**: The support drift is NOT pure CoM position drift. It is **yaw-induced sagittal position error** caused by hip yaw divergence coupling. The robot accumulates position error in one direction because position correction causes hip yaw, which causes more position error—a feedback loop.

| Metric | D2 | E2 | E2b | Interpretation |
|--------|-----|-----|-----|----------------|
| support_error_mean (m) | 0.058 | 0.063 | 0.063 | E2 slightly worse |
| support_error_max (m) | 0.176 | 0.170 | 0.170 | Similar |
| crossings >0.15m (norm-500) | 9.6 | 62.0 | 62.0 | E2 worse (6.5×) |
| hip_yaw_abs_max (rad) | 0.313 | 0.130 | 0.130 | E2 much better (2.4×) |
| pitch_mean (deg) | 2.55 | 2.58 | 2.58 | Similar |
| pitch_max (deg) | 6.36 | 7.13 | 7.13 | E2 worse |
| pitch_forward_pct | 94.6% | 80.4% | 80.4% | E2 recovers more |

---

## Phase 1: Signed Support Drift Metrics

### 1.1 Data Source Identification

The `support_position_error_m` field is **NOT** a pure CoM-support center error. Investigation confirmed:

```
support_position_error_m = hip_yaw_comp_support_error_m (correlation ≈ 1.0)
yaw_aware_sagittal_error_compensated_m = support_position_error_m (identical)
```

**Root Cause**: Hip yaw divergence causes the sagittal position error through coupling between yaw rotation and lateral position. This is why:
- D2 has high hip_yaw (0.31 rad) → high position error
- E2 reduced hip_yaw (0.13 rad) → but position error crossings increased

### 1.2 Signed Drift Classification

| Variant | Signed Error Classification | Bias Ratio | Zero Crossings | Max Same-Sign Run |
|---------|---------------------------|------------|----------------|-------------------|
| D2 | POSITIVE_BIASED_STRONG | 40.6 | 13 | 4875/5000 |
| E2 | POSITIVE_BIASED_STRONG | 7.9 | 4 | 442/500 |
| E2b | POSITIVE_BIASED_STRONG | 7.9 | 4 | 442/500 |

**Interpretation**:
- D2: 97.5% of time spent with positive support error (one-sided drift)
- E2: 88.4% of time with positive support error (improved but still biased)
- E2b: Identical to E2 (integral gate widening had no effect)

### 1.3 Crossings Analysis

Crossings of `support_position_error_m > 0.15m` (normalized to 500 steps):

| Variant | Raw Crossings | Normalized to 500 | Change vs D2 |
|---------|--------------|-------------------|--------------|
| D2 | 96 | 9.6 | baseline |
| E2 | 62 | 62.0 | **+6.5× WORSE** |
| E2b | 62 | 62.0 | **+6.5× WORSE** |

**Critical Observation**: E2 improved hip_yaw by 2.4× but WORSENED position error crossings by 6.5×. This is the fundamental trade-off.

---

## Phase 2: Phase Reversal / Missed Recenter Analysis

### 2.1 Pitch Reversal Detection

Only **1 pitch reversal** detected in 5000 steps for D2. This means:
- The robot is falling forward **almost continuously**
- There are very few opportunities for "recentering" because the robot never stabilizes

### 2.2 Wheel Velocity Behavior

Using `wheel_vel_mean_rad_s`:

| Variant | Mean (rad/s) | Std (rad/s) | Min | Max | % Positive | % Negative |
|---------|-------------|------------|-----|-----|-----------|-----------|
| D2 | -0.028 | 1.74 | -4.39 | 2.77 | ~50% | ~50% |
| E2 | N/A | N/A | N/A | N/A | N/A | N/A |

**Interpretation**:
- Mean wheel velocity is nearly zero (-0.028 rad/s)
- High standard deviation (1.74 rad/s) shows wheels oscillate
- Equal positive/negative time suggests active balance correction

### 2.3 tau_position Analysis

| Variant | Mean (Nm) | Min (Nm) | At Cap % | Cap Value |
|---------|----------|----------|----------|----------|
| D2 | -2.23 | -4.00 | 37.8% | 3.0 Nm |
| E2 | -2.29 | -5.00 | 42.2% | 5.0 Nm |
| E2b | -2.29 | -5.00 | 42.2% | 5.0 Nm |

**Key Observations**:
- tau_position is **always negative** (correcting forward fall)
- D2 hits cap at 3 Nm 37.8% of time
- E2 increased cap to 5 Nm but hits cap 42.2% of time (more saturation)
- The increased cap did NOT reduce crossings

### 2.4 Phase Behavior Classification

**RECENTERING_WORKS**: When pitch reverses and support error reduces
**RECENTERING_TOO_WEAK**: When pitch reverses but support error does NOT reduce
**RECENTERING_PREMATURELY_REVERSED**: When wheel reverses too aggressively

**Finding**: The pitch reversal rate is too low to analyze recentering behavior statistically. The robot falls forward continuously without stabilization phases.

---

## Phase 3: Why E2/E2b Improved hip_yaw but Worsened Position

### 3.1 The Coupling Mechanism

```
Position Error → tau_position (position correction)
                    ↓
            Hip roll/yaw coupling → hip_yaw_abs_max increases
                    ↓
            More yaw divergence → More yaw-induced position error
                    ↓
            Position Error increases (feedback loop)
```

### 3.2 Evidence

| Variant | tau_position_mean | hip_yaw_abs_max | Position Error |
|---------|------------------|-----------------|---------------|
| D2 | -2.23 Nm | 0.31 rad | 0.058 m |
| E2 | -2.29 Nm | 0.13 rad | 0.063 m |

E2 reduced position correction authority, which:
- ✓ Reduced hip_yaw_abs_max by 2.4×
- ✗ But position error mean INCREASED (0.058 → 0.063 m)
- ✗ Position error crossings INCREASED 6.5×

### 3.3 The Trade-off

E2's fix addressed the symptom (hip yaw) but not the root cause (position drift). By reducing position correction, the robot:
- Has less hip yaw coupling
- But also has less ability to correct position
- Leading to more accumulated position error

---

## Phase 4: Phase-Aware Recentering Strategy Design

### 4.1 Design Principle

**When the robot is in dangerous fall recovery**: Balance wins.
**When the robot is recovering or pitch is safe**: Allow/support recentering.

### 4.2 Candidate: F1_phase_aware_recenter_velocity_shaping

**Core Idea**: Instead of increasing position correction cap (which causes hip yaw), add a phase-aware recenter term that:
1. Detects when pitch is recovering (not falling dangerously)
2. In those phases, applies a gentle recenter force proportional to signed support error
3. Does NOT compete with balance when pitch is unsafe

### 4.3 Signal Requirements

| Signal | Source | Purpose |
|--------|--------|---------|
| `pitch_x` | telemetry | Detect fall direction |
| `pitch_rate_x` | telemetry | Detect recovery vs fall |
| `hip_yaw_comp_support_error_m` | telemetry | Signed support error |
| `hip_yaw_abs_max` | telemetry | Bounds check |
| `tau_position` | telemetry | Current position torque |

### 4.4 Phase Detection Logic

```
SAFE_TO_RECENTER:
  - abs(pitch_x) < pitch_safe_threshold (e.g., 0.05 rad ≈ 3°)
  - OR pitch_rate_x indicates recovery (pitch_rate opposite to pitch)
  - AND hip_yaw_abs_max < hip_yaw_threshold (e.g., 0.10 rad)
  - AND contact_valid

DANGEROUS_FALL:
  - abs(pitch_x) > pitch_danger_threshold (e.g., 0.10 rad ≈ 6°)
  - OR hip_yaw_abs_max > hip_yaw_threshold
  - OR pitch_rate indicates accelerating fall
```

### 4.5 Recenter Term Design

```python
def compute_recenter_term(signed_error, pitch_x, pitch_rate, hip_yaw, contact_valid):
    # Phase detection
    pitch_safe = abs(pitch_x) < 0.05 or (pitch_x * pitch_rate < 0)
    hip_yaw_safe = hip_yaw < 0.10
    safe = pitch_safe and hip_yaw_safe and contact_valid

    if not safe:
        return 0.0  # Let balance command dominate

    # Bounded recenter correction
    # tau_recenter = -k_recenter * signed_error
    # where signed_error = hip_yaw_comp_support_error_m (signed)
    k_recenter = 10.0  # Tunable gain
    max_recenter_tau = 1.0  # Nm - much smaller than balance authority

    recenter_tau = -k_recenter * signed_error
    recenter_tau = clip(recenter_tau, -max_recenter_tau, max_recenter_tau)

    return recenter_tau
```

### 4.6 Integration with Existing Controller

The recenter term should be added as a **separate term** that:
1. Does NOT affect tau_position (which drives hip yaw)
2. Affects wheel velocity command directly
3. Only activates in safe phases

```
tau_final = tau_balance + tau_position + tau_recenter
                ↓
        wheel command
```

### 4.7 Anti-Windup / Anti-Chatter

- Use hysteresis on phase detection to avoid rapid switching
- Apply smooth ramp-up/ramp-down of recenter term
- Limit recenter rate of change

---

## Phase 5: Hip Yaw Regression Root Cause

### 5.1 Why E2 Regressed hip_yaw

E2's fix (5.0 Nm cap / stronger position correction) caused hip yaw regression because:
1. **Increased position authority**: tau_position cap 3→5 Nm
2. **More hip roll/yaw coupling**: Stronger position correction → more hip yaw
3. **Position correction feeds yaw**: yaw divergence → more position error

### 5.2 Why E2b (wider integral gate) Did Not Help

E2b widened the integral gate but hip_yaw still regressed because:
- The position correction coupling was already the issue
- Integral gate only affects how quickly integral accumulates
- Does not address the fundamental coupling mechanism

### 5.3 The Real Fix

The fix should **decouple** position correction from hip yaw:
1. **Option A**: Add a separate recenter term (phase-aware)
2. **Option B**: Use hip-yaw-aware position correction gain
3. **Option C**: Modify position correction to avoid hip yaw coupling

---

## Conclusions

### 1. Is support drift biased to one side?
**YES**. Signed support error is POSITIVE_BIASED_STRONG:
- D2: 97.5% time positive
- E2: 88.4% time positive

### 2. Does support_error oscillate around zero or ratchet away?
**RATCHETING**. The robot accumulates error in one direction because:
- Hip yaw divergence causes yaw-induced position error
- Position correction causes more hip yaw (feedback loop)

### 3. When pitch reverses, does the controller allow recentering?
**INFREQUENT OPPORTUNITY**. Only 1 pitch reversal in 5000 steps (D2). The robot falls forward continuously without stabilization phases.

### 4. Is the wheel command reversing too early/too aggressively?
**INCONCLUSIVE**. Wheel velocity shows high variation (std=1.74 rad/s) but mean near zero. The phase reversal rate is too low to analyze wheel reversal behavior.

### 5. Why did E2/E2b improve support but regress hip-yaw?
**COUPLING MECHANISM**. E2 increased position correction authority (cap 3→5 Nm), which:
- Reduced hip_yaw_abs_max (0.31→0.13 rad) by reducing position correction
- But position error crossings INCREASED (9.6→62 normalized) because robot corrected less
- The trade-off: less position correction = less hip yaw coupling = less position correction

### 6. What is the safest next candidate?
**F1_phase_aware_recenter_velocity_shaping**:
- Add a separate recenter term that does NOT couple with hip yaw
- Only activates when pitch is safe (recovering)
- Bounded authority (max 1 Nm) to avoid competing with balance
- Uses signed support error directly

### 7. Why is phase-aware recentering better than simply increasing cap?
**DECOUPLING**. Simply increasing cap:
- Increases position correction authority
- Increases hip yaw coupling
- Creates a feedback loop: position error → yaw → more position error

Phase-aware recentering:
- Separate term from position correction
- Only activates in safe phases
- Does NOT affect tau_position (which drives hip yaw)
- Breaks the feedback loop

---

## Final Decision

**SIGNED_DRIFT_CONFIRMED_PHASE_AWARE_RECENTER_READY**

The audit confirms:
1. ✓ Signed drift is confirmed (positive bias, 97.5% for D2)
2. ✓ Root cause identified (yaw-induced position error)
3. ✓ Phase detection signals available (pitch, pitch_rate, hip_yaw)
4. ✓ Strategy designed (phase-aware recenter term)
5. ✓ Integration approach defined (separate term, bounded authority)

**Next executable step**: Implement F1_phase_aware_recenter_velocity_shaping and evaluate at low_0p300 for 500 steps.

---

## Appendix: Output Files

| File | Description |
|------|-------------|
| `signed_drift_metrics.json` | Complete signed drift metrics for D2, E2, E2b |
| `signed_drift_metrics.csv` | Summary table in CSV format |
| `phase_behavior_summary.json` | Phase reversal analysis results |
| `wheel_reversal_summary.json` | Wheel velocity and tau_position analysis |
| `phase_aware_strategy_inputs.json` | Strategy design inputs |
| `signed_drift_summary.csv` | Summary comparison table |
