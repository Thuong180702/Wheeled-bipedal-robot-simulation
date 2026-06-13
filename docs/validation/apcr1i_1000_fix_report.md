# APCR1i 1000-Step Diagnostic Fix Report

**Date:** 2026-06-10
**Profile:** APCR1i_support_hysteresis_recenter
**Steps:** 1000
**Status:** Diagnostic complete - fix identified

---

## Executive Summary

APCR1i 1000-step validation confirmed:

1. **Torque capped at 1.5 Nm** despite configured 1.75 Nm - downstream clipping confirmed
2. **Episode 5 early exit** - RECENTER state exited while e=0.1732 m (not at inner band)
3. **Hysteresis behavior mostly correct** - principles 1, 2, 5 pass
4. **Insufficient recentering authority** - max positive drift 0.2550 m exceeds target < 0.15 m

---

## Phase 4-6 Summary

### Phase 4: Episode Audit

| Metric | Value |
|--------|-------|
| Total RECENTER episodes | 5 |
| RECENTER_FROM_POSITIVE | 4 |
| RECENTER_FROM_NEGATIVE | 1 |
| Correct exits (reached inner band) | 4 |
| Early exits | 1 |

**Episode 5 failure:**
- State: RECENTER_FROM_POSITIVE
- Entry: step 961, e=0.0821
- Exit: step 999, e=0.1732 (did not reach inner band)
- Duration: 38 steps
- Reached inner band: False
- Classification: APCR1I_EXITS_TOO_EARLY_BEFORE_INNER_BAND

### Phase 5: Torque Authority Audit

| Metric | Configured | Observed |
|--------|------------|----------|
| recenter_max_tau | 1.75 Nm | 1.5000 Nm |
| emergency_max_tau | 2.00 Nm | 1.5000 Nm |
| recenter_rate_per_step | 0.90 | N/A |

**Findings:**
- `active_pitch_crossing_max_tau` telemetry shows constant 1.5 Nm
- Raw tau reaches -1.5 Nm, confirming cap
- `active_pitch_crossing_tau_clipped` shows 942 clipping events
- **Root cause:** Downstream cap at 1.5 Nm overrides profile's 1.75 Nm setting

### Phase 6: Principle Verification

| Principle | Status |
|-----------|--------|
| P1: e > 0.08 → RECENTER_FROM_POSITIVE | ✅ PASS |
| P2: e < -0.08 → RECENTER_FROM_NEGATIVE | ✅ PASS |
| P3: No early exit | ❌ FAIL (Episode 5) |
| P4: Pitch gate doesn't interrupt | ⚠️ 32.5% in pitch danger |
| P5: Bidirectional recenter | ✅ PASS |

---

## Root Causes

### Issue 1: Downstream Torque Cap at 1.5 Nm

**Evidence:**
- Profile config: `apc_hysteresis_recenter_max_tau = 1.75 Nm`
- Observed max torque: 1.5000 Nm
- `active_pitch_crossing_max_tau` telemetry shows 1.5 Nm constant
- `active_pitch_crossing_tau_clipped` has 942 events

**Location:** The 1.5 Nm cap is hardcoded in the APCR/tau selection logic, not in the APCR1i profile.

**Impact:** Reduces recentering authority by 14% (1.5 vs 1.75 Nm)

### Issue 2: Episode 5 Early Exit

**Evidence:**
- RECENTER episode at steps 961-999
- Exit e=0.1732 m (should be ≤0.03 m)
- Exit reason: "N/A" (simulation ended)

**Root cause:** The episode was cut short by simulation end (step 999 = step 1000), not by proper exit logic. The state machine had not yet reached the inner band.

### Issue 3: Insufficient Overall Authority

**Evidence:**
- Max positive drift: 0.2550 m (target: <0.15 m)
- Mean positive drift: +0.0809 m
- P2P: 0.3424 m

**Root cause:** Combined effect of:
1. Torque cap at 1.5 Nm instead of 1.75 Nm
2. Rate limit prevents rapid torque application
3. Single RECENTER cycle takes 100-200 steps to reduce drift from 0.08 to 0.03

---

## Fix Recommendations

### Fix A: Increase Torque Cap (APCR1j)

**Change:** Create APCR1j profile with higher torque authority

```yaml
# Proposed APCR1j values
apc_hysteresis_recenter_max_tau: 2.00 Nm  # up from 1.75
apc_hysteresis_emergency_max_tau: 2.20 Nm  # up from 2.00
apc_hysteresis_recenter_rate_per_step: 1.10  # up from 0.90
apc_hysteresis_emergency_rate_per_step: 1.30  # up from 1.10
```

**Rationale:** With downstream cap at 1.5 Nm, need to set profile cap higher to achieve 1.5 Nm effective torque.

### Fix B: Widen Hysteresis Bands (APCR1j alternative)

**Change:** Adjust enter/exit thresholds

```yaml
# Proposed APCR1j values
apc_hysteresis_outer_enter_m: 0.10  # up from 0.08
apc_hysteresis_inner_exit_m: 0.05   # up from 0.03
apc_hysteresis_opposite_release_m: 0.05  # up from 0.03
```

**Rationale:** Larger inner band = earlier exit from RECENTER = more time for next recenter cycle

### Fix C: Widen Hysteresis + Higher Torque (APCR1j)

**Combination of Fix A and Fix B**

### Fix D: Fix Downstream Cap (if upstream control)

**Note:** The 1.5 Nm cap may be in the sagittal velocity-damped controller's `effective_pitch_tau_cap` or similar. If so, fixing the upstream cap would allow APCR1i to work as designed.

---

## Recommended Action

**Create APCR1j profile** with:
1. `apc_hysteresis_recenter_max_tau: 2.00 Nm` (to overcome 1.5 Nm downstream cap)
2. `apc_hysteresis_emergency_max_tau: 2.20 Nm`
3. `apc_hysteresis_recenter_rate_per_step: 1.10`
4. Keep all other APCR1i settings unchanged

**Do NOT modify APCR1i** - keep it as diagnostic baseline.

---

## Files Generated

- `apcr1i_1000_drift_metrics.json` - drift statistics
- `apcr1i_1000_drift_metrics.csv` - drift CSV
- `apcr1i_1000_window_metrics.csv` - window analysis
- `apcr1i_1000_episode_audit.json` - episode details
- `apcr1i_1000_episode_table.csv` - episode table
- `apcr1i_1000_torque_authority_audit.json` - torque analysis
- `apcr1i_1000_principle_verification.json` - principle verification

---

## Decision

**Classification:** APCR1I_1000_DIAGNOSTIC_ONLY_NEEDS_FIX

APCR1i does not achieve target drift behavior due to:
1. Downstream torque cap at 1.5 Nm (not 1.75 Nm configured)
2. Insufficient recentering authority to reduce drift from 0.08 to <0.03 within reasonable time

**Next step:** Create APCR1j with higher torque authority to overcome the downstream cap.