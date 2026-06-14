# APCR1i 1000-Step Diagnostic Final Report

**Date:** 2026-06-10
**Profile:** APCR1i_support_hysteresis_recenter
**Steps:** 1000
**Output:** `outputs/hierarchical_controller_sim/telemetry_1781058071.csv`

---

## Decision

**Classification:** `APCR1I_1000_DIAGNOSTIC_ONLY_NEEDS_FIX`

APCR1i does NOT achieve target drift behavior due to downstream torque cap overriding profile settings.

---

## Key Findings

### 1. APCR1i Survived 1000 Steps ✅
- Completed 1000 steps without fall
- Pitch range: -0.075 to +0.136 rad
- Contact: 100% double_contact

### 2. Drift Metrics

| Metric | APCR1i 1000-step | Target | Status |
|--------|-----------------|--------|--------|
| Min e | -0.0873 m | - | - |
| Max e | +0.2550 m | < +0.15 m | ❌ FAIL |
| P2P | 0.3424 m | - | - |
| Max \|e\| | 0.2550 m | < 0.15 m | ❌ FAIL |
| Mean e | +0.0809 m | ~0 | ❌ FAIL |
| Outside ±0.08 | 546 (54.6%) | < 10% | ❌ FAIL |
| Outside ±0.15 | 296 (29.6%) | 0% | ❌ FAIL |

### 3. Root Cause: Downstream Torque Cap at 1.5 Nm

**Evidence:**
- Profile config: `apc_hysteresis_recenter_max_tau = 1.75 Nm`
- Profile config: `apc_hysteresis_emergency_max_tau = 2.00 Nm`
- Observed APCR tau max: **1.5000 Nm** (clipped)
- `active_pitch_crossing_max_tau` telemetry: constant 1.5 Nm

**Root cause location:** Lines 2262-2266 in `sagittal_velocity_damped_balance_controller.py`:
```python
# Clip APC torque - THIS OVERRIDES APCR1i HIGHER MAX TAU
apc_raw_tau = float(jnp.clip(
    apc_raw_tau,
    -self.authority_schedule.apc_max_cross_tau,  # = 1.5 Nm (default)
    self.authority_schedule.apc_max_cross_tau
))
```

The universal clip at 1.5 Nm overrides APCR1i's configured 1.75 Nm max.

### 4. State Machine Behavior

| Metric | Value |
|--------|-------|
| RECENTER_FROM_POSITIVE episodes | 4 |
| RECENTER_FROM_NEGATIVE episodes | 1 |
| HOLD_THROUGH_ZERO episodes | 0 |
| Episodes reaching inner band | 4/5 |
| Early exits | 1 (Episode 5: cut by simulation end) |

**Episode 5** was cut short by simulation end (steps 961-999):
- Entry e=0.0821, Exit e=0.1732
- Did not reach inner band
- APCR applied -1.5 Nm throughout but drift increased

### 5. Principle Verification

| Principle | Status |
|-----------|--------|
| e > +0.08 → RECENTER_FROM_POSITIVE | ✅ PASS (100%) |
| e < -0.08 → RECENTER_FROM_NEGATIVE | ✅ PASS (100%) |
| Correct torque sign | ✅ PASS (100%) |
| RECENTER holds until inner band | ⚠️ 4/5 correct |
| Bidirectional recenter | ✅ PASS (1 NEGATIVE episode occurred) |

**Classification:** `APCR1I_PRINCIPLE_FAILS_EARLY_EXIT` (Episode 5 was early exit but cut by simulation end)

---

## Comparison with APCR1h

APCR1h (500-step reference):
- Max e: 0.1572 m
- P2P: 0.2080 m
- Outside ±0.15: 2.6%

APCR1i (1000-step):
- Max e: 0.2550 m (62% worse)
- P2P: 0.3424 m (65% worse)
- Outside ±0.15: 29.6% (11x worse)

APCR1i is **significantly worse** than APCR1h in 1000 steps.

---

## Fix Recommendation

### Option A: Create APCR1j with Higher `apc_max_cross_tau`

Modify APCR1i profile to override the 1.5 Nm cap:

```python
# In APCR1j profile, add:
apc_max_cross_tau=2.0,  # Override default 1.5 Nm
apc_hysteresis_recenter_max_tau=2.0,  # Higher recenter authority
apc_hysteresis_emergency_max_tau=2.2,  # Higher emergency authority
apc_hysteresis_recenter_rate=1.1,  # Faster rate
```

### Option B: Fix Downstream Clip for APCR1i

Modify lines 2262-2266 to use APCR1i's own max tau when APCR1i is active.

---

## Files Generated

| File | Description |
|------|-------------|
| `apcr1i_1000_drift_metrics.json` | Drift statistics |
| `apcr1i_1000_drift_metrics.csv` | Drift CSV |
| `apcr1i_1000_window_metrics.csv` | Window analysis |
| `apcr1i_1000_episode_audit.json` | Episode details |
| `apcr1i_1000_episode_table.csv` | Episode table |
| `apcr1i_1000_torque_authority_audit.json` | Torque analysis |
| `apcr1i_1000_principle_verification.json` | Principle verification |
| `apcr1i_1000_fix_report.md` | Fix recommendations |

---

## Recommended Next Steps

1. **Create APCR1j** with `apc_max_cross_tau=2.0` to override the 1.5 Nm cap
2. **Run APCR1j 1000-step** validation
3. **Do NOT modify APCR1i** - keep as diagnostic baseline
4. **Do NOT run 2000-step** until APCR1j shows improvement

---

## Current Best Profile

Based on available evidence:
- **D2**: Not tested at extreme height
- **APCR1f**: Limited at low height
- **APCR1h**: Best so far (max_e=0.1572 m)
- **APCR1i**: Worse than APCR1h (max_e=0.2550 m)

**Recommendation:** APCR1h remains the best tested profile. APCR1j should be created and tested.