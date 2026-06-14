# Cross-Failure Causal Map — Step E Extreme Height Root Cause Audit

**Date:** 2026-06-07
**Baseline:** D2 official
**Cases:** low_0p300 (0.300m), high_0p480 (0.480m)

---

## Event Order Summary

### low_0p300 (0.300m)

| Rank | Event | Step | Time (s) | Value | Gate |
|------|-------|------|----------|-------|------|
| 1 | **support_drift_gate** | 91 | 0.91 | 0.1757 m | 0.15 m |
| 2 | hip_yaw_gate | 328 | 3.28 | 0.3127 rad | 0.10 rad |

**Classification:** `SUPPORT_DRIFT_PRIMARY_HIP_YAW_SECONDARY`

### high_0p480 (0.480m)

| Rank | Event | Step | Time (s) | Value | Gate |
|------|-------|------|----------|-------|------|
| 1 | **wheel_velocity_gate** | 73 | 0.73 | 5.26 rad/s | 5.0 rad/s |
| 2 | support_drift_gate | 108 | 1.08 | 0.1733 m | 0.15 m |
| 3 | hip_yaw_gate | 2426 | 24.26 | 0.2753 rad | 0.10 rad |

**Classification:** `WHEEL_VELOCITY_PRIMARY_SUPPORT_SECONDARY_HIP_YAW_TERTIARY`

---

## Failure Classification

### low_0p300: SUPPORT_DRIFT_PRIMARY

```
support_drift (step 91)
    ↓
hip_yaw divergence (step 328)
    ↓
Both gates FAIL
```

**Key insight:** Support drift appears first and drives hip yaw divergence second.
Hip yaw is secondary, not primary.

### high_0p480: WHEEL_VELOCITY_PRIMARY

```
wheel_velocity spike (step 73)
    ↓
support_drift (step 108)
    ↓
hip_yaw divergence (step 2426)
    ↓
All three gates FAIL
```

**Key insight:** Wheel velocity transient (5.26 rad/s peak) leads support drift.
The wheel velocity spike creates momentum that drives CoM drift.

---

## Root Cause Mechanisms

### 1. Position Authority Saturation (COMMON)

**Evidence:**
- `tau_position_max = 4.0 Nm` (saturated at limit)
- `tau_position_integral_max = 0.0 Nm` (position integral disabled)
- Both heights show identical saturation pattern

**Mechanism:**
```
tau_position = Kp × position_error
         + Ki × integral_error (DISABLED)
         + Kd × velocity_error

Problem: No position integral → cannot correct steady-state drift
Problem: Kp saturates at 4.0 Nm → authority exhausted
```

**Impact:** The sagittal controller cannot hold position → CoM drifts.

### 2. Wheel Velocity Transient (high_0p480 specific)

**Evidence:**
- Peak: 5.26 rad/s at step 83 (transient, only 0.34% of time above gate)
- `sagittal_term_pitch = 3.287 Nm` at peak
- `wheel_vel_leads_support = True`

**Mechanism:**
```
High height → more top-heavy → less stable
     ↓
Pitch disturbance → large sagittal_term_pitch
     ↓
Large wheel velocity command → momentum buildup
     ↓
CoM momentum exceeds wheel authority → drift
```

**Impact:** The wheel velocity spike at 0.73s creates CoM momentum that drives support drift at 1.08s.

### 3. Hip-Yaw Divergence (COMMON)

**Evidence:**
- `hip_yaw_divergence_mode = True` (divergence, not common mode)
- `hy2_div_enabled = False`
- `hy2_div_active = False`
- `hip_yaw_div_left = 0.0 Nm`, `hip_yaw_div_right = 0.0 Nm`

**Mechanism:**
```
Support drift → asymmetric loading
     ↓
Legs experience differential torque
     ↓
No HY2-DIV corrective torque applied
     ↓
Legs rotate outward (diverge)
```

**Impact:** Without corrective hip yaw differential torque, legs diverge.

### 4. WBC False Positive (COMMON — GATE BUG)

**Evidence:**
- `tau_wbc_norm_low_0p300 = 14.49 Nm`
- `tau_wbc_norm_high_0p480 = 20.13 Nm`
- `per_actuator_wbc_authority_enabled = False`

**Mechanism:**
```
Gate definition: wbc_applied = tau_wbc_norm > 0.001

Problem: tau_wbc is QP force distribution output
         → structural physics, not active control
         → always non-zero during contact

Reality: per_actuator_wbc_authority_enabled = False
         → no active WBC torques applied
         → should PASS gate, not FAIL
```

**Impact:** Gate fires incorrectly on structural QP artifact.

---

## Causal Relationships

```
┌─────────────────────────────────────────────────────────────────────┐
│                        low_0p300 (0.300m)                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  NO_POSITION_INTEGRAL ──────────────────┐                           │
│           │                            │                           │
│           ▼                            │                           │
│  POSITION_AUTHORITY_SATURATED ─────────┼──────────────────────────►│
│           │                            │                           │
│           ▼                            │                           │
│  SUPPORT_DRIFT (step 91) ◄─────────────┘                           │
│           │                                                      │
│           ▼                                                      │
│  HY2_DIV_DISABLED ─────────────────────────────────────────────►│
│           │                                                      │
│           ▼                                                      │
│  HIP_YAW_DIVERGENCE (step 328)                                    │
│           │                                                      │
│           ▼                                                      │
│  FAIL: support_position_error, hip_yaw, wbc_applied              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       high_0p480 (0.480m)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  HIGH_HEIGHT ───────────────────────────────────┐                  │
│         │                                       │                  │
│         ▼                                       │                  │
│  LARGE_SAGITTAL_TERM_PITCH ─────────────────────┼──────────────┐   │
│         │                                       │              │   │
│         ▼                                       ▼              │   │
│  WHEEL_VELOCITY_SPIKE (step 73)                │              │   │
│         │                                       │              │   │
│         ▼                                       │              │   │
│  CoM_MOMENTUM_BUILDUP ──────────────────────────┘              │   │
│         │                                                      │   │
│         ▼                                                      ▼   │
│  SUPPORT_DRIFT (step 108)        NO_POSITION_INTEGRAL          │   │
│         │                              │                       │   │
│         ▼                              ▼                       │   │
│  HY2_DIV_DISABLED ─────────────────────────────────────────────┼──►│
│         │                                                      │   │
│         ▼                                                      │   │
│  HIP_YAW_DIVERGENCE (step 2426)                                │   │
│         │                                                      │   │
│         ▼                                                      ▼   │
│  FAIL: wheel_velocity, support_position_error, hip_yaw, wbc_applied│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Shared Root Causes

| Root Cause | low_0p300 | high_0p480 | Notes |
|------------|-----------|------------|-------|
| POSITION_AUTHORITY_SATURATED | ✓ | ✓ | tau_position maxed at 4.0 Nm |
| NO_POSITION_INTEGRAL | ✓ | ✓ | integral_active_ratio = 0 |
| HY2_DIV_DISABLED | ✓ | ✓ | No corrective yaw torque |
| WBC_FALSE_POSITIVE | ✓ | ✓ | Gate bug, not actual WBC |
| OSCILLATORY_DRIFT | ✓ | ✓ | Not monotonic |

---

## Unique to Each Height

| Root Cause | Height | Evidence |
|------------|--------|----------|
| WHEEL_VELOCITY_TRANSIENT | high_0p480 | Peak 5.26 rad/s at step 73 |
| SUPPORT_DRIFT_PRIMARY | low_0p300 | Step 91 before any other gate |
| LARGE_SAGITTAL_TERM_PITCH | high_0p480 | 3.287 Nm at peak |

---

## Gate Status Summary

| Gate | low_0p300 | high_0p480 |
|------|-----------|------------|
| support_position_error < 0.15m | **FAIL** (0.1757m) | **FAIL** (0.1733m) |
| wheel_velocity < 5.0 rad/s | PASS (4.39 rad/s) | **FAIL** (5.26 rad/s) |
| hip_yaw < 0.10 rad | **FAIL** (0.3127 rad) | **FAIL** (0.2753 rad) |
| no_wbc_applied | **FALSE POSITIVE** | **FALSE POSITIVE** |

---

## Critical Findings

### Finding 1: Position Integral is Disabled
Both heights have `integral_active_ratio = 0.0`, meaning the position integral term never activates. This prevents correction of steady-state drift.

### Finding 2: Position Authority Saturates Immediately
`tau_position` hits the 4.0 Nm saturation limit within the first ~110 steps for both heights. The controller exhausts its position authority immediately.

### Finding 3: Wheel Velocity is Transient (high_0p480)
The wheel velocity exceeds 5.0 rad/s for only 0.34% of the simulation (17 steps out of 5000). This is a transient spike, not sustained failure.

### Finding 4: Hip Yaw Divergence Mode
Hip yaw failures are in **divergence mode** (legs rotating opposite directions), not common mode (both legs rotating same direction). This indicates asymmetric loading as the driver.

### Finding 5: WBC Gate is Incorrect
The `wbc_applied` gate fires on `tau_wbc_norm > 0.001`, which is structural QP output, not active WBC. Both heights should PASS this gate since `per_actuator_wbc_authority_enabled = False`.
