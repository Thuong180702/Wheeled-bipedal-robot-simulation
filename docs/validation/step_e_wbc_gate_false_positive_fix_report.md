# Step E WBC Gate False Positive Fix Report

**Date:** 2026-06-07
**Decision:** `WBC_GATE_FIX_PASS_REMAINING_CONTROLLER_FAILURES`
**Files Modified:** `scripts/analyze_step_e_extreme_height_d2_official_check.py`
**Tests Added:** `tests/test_step_e_wbc_gate_validator.py`

---

## Executive Summary

The Step E WBC gate had a **false positive** bug where it incorrectly failed the WBC gate because `tau_wbc_norm > 0.001`. The root cause audit correctly identified that `tau_wbc_norm` is structural QP support feedforward output, NOT active WBC control authority.

**Fix:** The WBC gate now uses `per_actuator_wbc_authority_enabled` (actual WBC control flag) instead of `tau_wbc_norm` (structural QP output).

---

## Phase 0: Health Check

All baseline tests passed:

| Test Suite | Result |
|------------|--------|
| `test_balance_core_height_variant_setup.py` | 26/26 PASSED |
| `test_sagittal_velocity_damped_balance_controller.py` | 40/40 PASSED |
| `test_shape_posture_hip_yaw_sign.py` | 9/9 PASSED |

---

## Phase 1: Root Cause

### Bug Location
`scripts/analyze_step_e_extreme_height_d2_official_check.py`, lines 160-162

### Old (Incorrect) Logic
```python
wbc_norm = [parse_float(r.get('tau_wbc_norm', 0)) for r in rows]
wbc_applied = any(v > 0.001 for v in wbc_norm)
```

### Problem
- `tau_wbc_norm` is the L2 norm of WBC torques from `BalanceCoreTorqueComposer`
- At extreme heights (0.300m, 0.480m), `tau_wbc_norm ≈ 13-20 Nm`
- This is the **QP structural solution** for force distribution, NOT active WBC control
- `per_actuator_wbc_authority_enabled = False` at both heights
- Active torque owner is `support_feedforward` only

### Evidence from Telemetry
| Height | `tau_wbc_norm` | `per_actuator_wbc_authority_enabled` | Active Owners |
|--------|----------------|-------------------------------------|---------------|
| 0.300m | 13.5 Nm | False | support_feedforward |
| 0.480m | 20.1 Nm | False | support_feedforward |

---

## Phase 2: Fix Implementation

### New (Correct) Logic
```python
# Method 1: Check per-actuator WBC authority flag (definitive)
per_actuator_wbc_authority_enabled = False
for r in rows:
    val = r.get('per_actuator_wbc_authority_enabled', 'False')
    if str(val).strip().lower() == 'true':
        per_actuator_wbc_authority_enabled = True
        break

# Method 2: Check ownership-based detection as fallback
WBC_ACTUAL_OWNERS = {'wbc', 'wbc_correction', 'full_wbc', 'centroidal_wbc', 'integrated_wbc'}
active_owners = set()
for r in rows:
    owner_str = r.get('active_torque_owner_per_joint', '')
    if owner_str:
        for owner in owner_str.split(','):
            active_owners.add(owner.strip())
has_actual_wbc_owner = bool(active_owners & WBC_ACTUAL_OWNERS)

# WBC is applied only when active authority is enabled OR actual WBC owners are present
wbc_applied = per_actuator_wbc_authority_enabled or has_actual_wbc_owner
```

### Key Changes
1. Added `per_actuator_wbc_authority_enabled` check
2. Added ownership-based fallback detection
3. `tau_wbc_norm` is still reported as `structural_qp_tau_norm` (diagnostic only)
4. Added `active_wbc_owners_detected` diagnostic field

---

## Phase 3: Tests

### New Test File: `tests/test_step_e_wbc_gate_validator.py`

| Test | Description | Result |
|------|-------------|--------|
| `test_wbc_gate_uses_authority_flag_not_tau_norm` | Structural QP output but no active WBC | PASS |
| `test_wbc_gate_fails_when_authority_enabled` | Active WBC authority | PASS |
| `test_wbc_gate_detects_wbc_owners` | WBC owners in ownership | PASS |
| `test_extreme_height_telemetry_wbc_gate` | Real extreme-height telemetry | PASS |

### Regression Tests
| Test Suite | Result |
|------------|--------|
| `test_validate_official_step_e_run.py` | 8/8 PASSED |
| `test_balance_core_validation_workflow.py` | 25/25 PASSED |

---

## Phase 4: Recheck Results

### Before/After WBC Gate Results

| Height | Old Result | New Result | Change |
|--------|------------|------------|--------|
| 0.300m | FAIL (tau_wbc_norm=13.5) | **PASS** | Fixed |
| 0.480m | FAIL (tau_wbc_norm=20.1) | **PASS** | Fixed |

### Remaining True Failures

| Height | Gate | Value | Threshold | Status |
|--------|------|-------|-----------|--------|
| 0.300m | Support position error | 0.176m | < 0.15m | FAIL |
| 0.300m | Hip yaw | 0.313 rad | < 0.10 rad | FAIL |
| 0.480m | Support position error | 0.173m | < 0.15m | FAIL |
| 0.480m | Wheel velocity | 5.26 rad/s | < 5.0 rad/s | FAIL |
| 0.480m | Hip yaw | 0.275 rad | < 0.10 rad | FAIL |

### Official Step E Result (Updated)

| Height | Survived | WBC Gate | Other Gates | Official Step E |
|--------|----------|----------|-------------|----------------|
| 0.300m | ✓ 5000 steps | **✓ PASS** | ✗ support, hip_yaw | **FAIL** |
| 0.480m | ✓ 5000 steps | **✓ PASS** | ✗ support, wheel_vel, hip_yaw | **FAIL** |

---

## Phase 5: Final Report

### Files Changed
- `scripts/analyze_step_e_extreme_height_d2_official_check.py` - Fixed WBC gate logic
- `tests/test_step_e_wbc_gate_validator.py` - New tests for validator logic

### Artifacts Created
- `outputs/step_e_wbc_gate_fix/wbc_gate_logic_inventory.json` - Gate logic inventory
- `outputs/step_e_wbc_gate_fix/wbc_gate_logic_inventory.md` - Gate logic documentation
- `outputs/step_e_wbc_gate_fix/recheck_extreme_heights/` - Recheck results
- `tests/test_step_e_wbc_gate_validator.py` - Validator tests

### Diagnostic Field Preservation
The following diagnostic fields are preserved and now properly labeled:
- `structural_qp_tau_norm` - Formerly `tau_wbc_norm`, now labeled as structural QP output
- `per_actuator_wbc_authority_enabled` - Authority flag
- `active_wbc_owners_detected` - Ownership-based WBC detection

### Decision

**`WBC_GATE_FIX_PASS_REMAINING_CONTROLLER_FAILURES`**

- WBC gate false positive FIXED ✓
- Remaining failures are true controller issues:
  1. Support position error exceeds 0.15m at both heights
  2. Hip yaw exceeds 0.10 rad at both heights
  3. Wheel velocity exceeds 5.0 rad/s at 0.480m

### Next Steps (Not Implemented)

The following controller fixes are needed (out of scope for this task):

1. **Support position error fix** - needs explicit position-holding or drift compensation
2. **Hip yaw divergence fix** - needs explicit yaw authority (HY2-DIV or other)
3. **High-height wheel velocity fix** - needs additional damping at 0.480m

### What This Fix Does NOT Change

- Controller behavior remains unchanged
- No WBC is enabled
- No HY2-DIV is enabled
- No gain tuning
- `tau_wbc_norm` is still logged for diagnostics

---

## Summary JSON

```json
{
  "decision": "WBC_GATE_FIX_PASS_REMAINING_CONTROLLER_FAILURES",
  "bug": "tau_wbc_norm used as proxy for WBC applied - false positive",
  "fix": "per_actuator_wbc_authority_enabled used instead",
  "files_changed": [
    "scripts/analyze_step_e_extreme_height_d2_official_check.py"
  ],
  "tests_added": [
    "tests/test_step_e_wbc_gate_validator.py"
  ],
  "low_0p300_wbc_gate": "PASS (was FAIL)",
  "high_0p480_wbc_gate": "PASS (was FAIL)",
  "remaining_failures": {
    "low_0p300": ["support_position_error", "hip_yaw"],
    "high_0p480": ["support_position_error", "hip_yaw", "wheel_velocity"]
  }
}
```
