# Boundary Height Forensic Root-Cause Investigation

**Date:** 2026-06-03
**Investigation:** Phase 1.5 MuJoCo Dynamics Mechanism Audit

---

## Executive Summary

This forensic investigation examines the **dynamic mechanism** causing hip-yaw drift at boundary heights (0.300 m and 0.480 m CoM), given that:

1. Static inverse dynamics shows **zero hip-yaw holding torque requirement**
2. Dynamic simulations show **large hip-yaw drift** (0.15-0.30 rad)
3. Increased gains provide **marginal improvement only** (23% for 67% gain increase)

---

## Passive Acceleration Audit (Phase 1.5A)

### Method

Compute `qacc` with `qvel=0` and `ctrl=0` to measure passive drift tendency.

### Results: low_0p300

- **Hip yaw passive qacc:** L=-5.769738, R=+7.118100 rad/s²
- **Drift direction:** L=negative, R=positive

### Results: high_0p480

- **Hip yaw passive qacc:** L=+0.744787, R=+4.716482 rad/s²
- **Drift direction:** L=positive, R=positive

### Interpretation

[WARNING]  **LOW BOUNDARY has significant passive drift tendency.**

The boundary pose is passively unstable in hip-yaw. PD control must fight continuous drift, and insufficient authority allows error accumulation.

[WARNING]  **HIGH BOUNDARY has significant passive drift tendency.**

---

## Actuator Effectiveness Audit (Phase 1.5D)

### Method

Apply ±1.0 Nm test torques to hip-yaw and measure resulting `qacc`.

### Results

| Height | Avg Effectiveness (rad/s²/Nm) |
|--------|--------------------------------|
| low_0p300 | +10.257002 |
| high_0p480 | +7.714941 |
| **Ratio (low/high)** | **1.3295** |

### Interpretation

[OK] **Actuator effectiveness is similar across boundary heights.**

Hip-yaw moment arm does not collapse significantly. Authority loss must come from other sources (coupling, saturation, etc.).

---

## Mechanism Classification

### Low Boundary (0.300 m CoM)

**Hypothesis:** `passive_dynamic_instability`

**Evidence:**
- Passive qacc shows drift tendency even with zero control
- Boundary pose is passively unstable in hip-yaw
- PD control must continuously fight drift

**Root Cause:** Extreme flexion creates passive instability that hierarchical velocity-damped control cannot stabilize with tested gains.

### High Boundary (0.480 m CoM)

**Hypothesis:** `passive_dynamic_instability`

---

## Next Steps Required

### Option A: Passive Drift Detected

1. **Feedforward compensation** for passive drift (not static holding torque)
2. **Velocity-dependent gains** to increase authority during drift
3. **Nonlinear gain scheduling** based on posture (joint angles)

---

## Conclusion

Static inverse dynamics ruled out static holding torque deficit. This Phase 1.5 MuJoCo dynamics audit provides the **first mechanism-level evidence**:

✅ **Passive dynamic instability detected at low boundary**

The boundary pose has passive drift tendency that PD control cannot stabilize with tested gains.

