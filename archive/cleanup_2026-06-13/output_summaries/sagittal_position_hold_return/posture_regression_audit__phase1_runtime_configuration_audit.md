# Phase 1: Runtime Configuration Audit

**Date:** 2026-05-31  
**Task:** Verify active runtime configuration for visual posture regression investigation

## Executive Summary

**ROOT CAUSE IDENTIFIED: Wrong controller mode (Classification B)**

The visual simulation showing posture/stance regression during steps 1000-2000 was running in **LEGACY/upright mode**, NOT balance-core mode with the velocity-damped sagittal controller.

**Impact:** All visual observations of posture regression are from the legacy controller stack, not from Step E velocity-damped controller fixes. Step E has NOT been tested visually yet.

---

## Telemetry Analysis

### V3 Run (5000 steps) - telemetry_1780203372.csv

**Controller Mode Evidence:**
- `controller_mode` column (304): `0.0` (numeric encoding for "upright")
- `control_mode` column (257): `upright` (string)
- `ablation_mode` column: `LEGACY`
- `capture_gate_enabled`: `False` (0.0) at all steps
- File size: 35.5 MB (5000 steps)
- Timestamp: 2026-05-31 11:56

**Active Controllers:**
- ✗ Balance-core mode: **INACTIVE**
- ✗ Velocity-damped sagittal controller: **INACTIVE**
- ✗ Smart capture gate: **INACTIVE**
- ✓ Legacy upright controller: **ACTIVE**
- ✓ Legacy wheel balance: **ACTIVE**
- ✓ Legacy hip-roll centering: **ACTIVE**

**Configuration Values (from telemetry):**
- `max_position_tau`: 3.0 (column exists but controller inactive)
- `k_position`: Not applicable (legacy mode)
- `k_velocity`: Not applicable (legacy mode)
- `kp_cp`: Not applicable (legacy mode)
- `capture_gate_enabled`: False
- `pitch_rate_correction`: Not applicable (legacy mode)
- `transient_capture_mode`: "none"

### Most Recent Run (1928 steps) - telemetry_1780206278.csv

**Controller Mode Evidence:**
- `controller_mode` column (304): `0.0` (upright)
- `control_mode` column (257): `upright`
- `ablation_mode` column: `LEGACY`
- `capture_gate_enabled`: `False`
- File size: 13.6 MB (1928 steps, terminated early)
- Timestamp: 2026-05-31 12:44

**Same configuration as V3 run - legacy mode active.**

---

## Git Status Audit

### Modified Files

```
modified:   scripts/simulate_hierarchical_controller.py
modified:   tests/test_sagittal_balance_state.py
modified:   tests/test_sagittal_velocity_damped_balance_controller.py
modified:   wheeled_biped/controllers/sagittal_balance_state.py
modified:   wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py
```

### Untracked Files (Step E additions)

```
scripts/analyze_capture_gate_telemetry.py
scripts/analyze_step_e_validation.py
scripts/verify_capture_signs.py
tests/test_pitch_rate_consistency_estimator.py
tests/test_position_hold_capture_gate.py
wheeled_biped/controllers/pitch_rate_consistency_estimator.py
wheeled_biped/controllers/position_hold_capture_gate.py
```

### Unchanged Controller Files

✓ `wheeled_biped/controllers/shape_posture_controller.py` - **NO CHANGES**  
✓ `wheeled_biped/controllers/support_feedforward_controller.py` - **NO CHANGES**  
✓ `wheeled_biped/controllers/lateral_roll_balance_controller.py` - **NO CHANGES**

**Conclusion:** Posture/support controllers are unchanged. No regression introduced in shape/support/lateral code.

---

## Code Changes Audit

### SagittalVelocityDampedBalanceController Changes

**Key modifications:**
1. `kp_cp` default changed: `30.0` → `0.0` (Step E coupling fix)
2. Added `k_position` parameter (default 0.0, configurable)
3. Added `max_position_tau` parameter (default 3.0)
4. Added `enable_capture_gate` parameter (default False)
5. Added `capture_gate_config` parameter (optional dict)
6. Added capture gate integration in `compute()` method
7. Added position-hold term clipping before summing
8. Added capture gate diagnostics to telemetry

**Critical:** These changes only apply when `controller_mode == "balance-core"` AND `sagittal_controller == "velocity-damped"`.

### simulate_hierarchical_controller.py Changes

**Key modifications:**
1. Added CLI arguments for velocity-damped controller configuration:
   - `--vd-k-position` (default 40.0)
   - `--vd-k-velocity` (default 15.0)
   - `--vd-max-position-tau` (default 3.0)
   - `--vd-enable-capture-gate` (default False)
   - `--vd-capture-gate-*` parameters
2. Added capture gate config building in `build_balance_core_controllers()`
3. Updated `SagittalVelocityDampedBalanceController` instantiation with new parameters

**Critical:** Default `controller_mode` is still `"legacy"` (line 740), NOT `"balance-core"`.

---

## Missing Command Argument

### Required Command for Balance-Core Mode

```bash
python scripts/simulate_hierarchical_controller.py \
  --steps 5000 \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-k-position 20.0 \
  --vd-k-velocity 15.0 \
  --vd-max-position-tau 3.0 \
  --vd-enable-capture-gate \
  --vd-capture-gate-use-cp
```

### What Was Actually Run (inferred)

```bash
python scripts/simulate_hierarchical_controller.py --steps 5000
# Missing: --controller-mode balance-core
# Result: Defaulted to legacy mode
```

---

## Diagnostic Mode Audit

### Stale Diagnostic Modes Check

**From telemetry columns:**
- `pitch_rate_corrected_x_rad_s`: Present but unused (pitch_rate_source_used: "measured")
- `transient_detected`: False at all steps
- `transient_by_pitch`: False
- `transient_by_pitch_rate`: False
- `transient_by_height`: False
- `transient_capture_mode`: "none"
- `pitch_rate_boost_factor`: 1.0
- `capture_gate_enabled`: False

**Conclusion:** No stale diagnostic modes active. All diagnostic features correctly disabled by default.

---

## WBC and E0 Audit

### WBC Status

**From telemetry:**
- `tau_wbc_max`: 8.15 Nm (step 0), non-zero in legacy mode
- `tau_wbc_norm`: 11.99 Nm (step 0)
- `tau_wbc_per_joint`: Non-zero values
- `qp_solve_time_ms`: 109.7 ms (step 0)
- `qp_converged`: True

**Status:** WBC is **ACTIVE** in legacy/upright mode (expected behavior for legacy mode).

**Critical:** In balance-core mode, WBC should be OFF. This telemetry confirms legacy mode was active.

### E0b/E0c/E0d Status

**From code inspection:**
- No E0b/E0c/E0d logic found in modified files
- No kp_cp active in balance-core velocity-damped controller (kp_cp=0.0)

**Status:** E0b/E0c/E0d remain removed/disabled as required.

---

## Torque Ownership Audit

### Legacy Mode Ownership (from telemetry)

**Step 0 ownership:**
```
active_torque_owner_per_joint:
  [0] l_hip_roll: none
  [1] l_hip_yaw: none
  [2] l_hip_pitch: support_feedforward
  [3] l_knee: support_feedforward
  [4] l_wheel: none
  [5] r_hip_roll: none
  [6] r_hip_yaw: none
  [7] r_hip_pitch: support_feedforward
  [8] r_knee: support_feedforward
  [9] r_wheel: none
```

**Ownership violations:** 0

**Conclusion:** Torque ownership is correct for legacy mode. No violations detected.

---

## Summary Table

| Configuration Item | Expected (Step E) | Actual (V3 Run) | Status |
|-------------------|-------------------|-----------------|--------|
| `controller_mode` | `balance-core` | `upright` (legacy) | ❌ WRONG |
| `sagittal_controller` | `velocity-damped` | legacy baseline | ❌ WRONG |
| `capture_gate_enabled` | `True` | `False` | ❌ WRONG |
| `k_position` | 20.0 | N/A (legacy) | ❌ WRONG |
| `k_velocity` | 15.0 | N/A (legacy) | ❌ WRONG |
| `max_position_tau` | 3.0 | N/A (legacy) | ❌ WRONG |
| `kp_cp` | 0.0 | N/A (legacy) | ❌ WRONG |
| WBC active | False | True | ❌ WRONG |
| E0b/E0c/E0d active | False | False | ✓ CORRECT |
| Ownership violations | 0 | 0 | ✓ CORRECT |
| Shape/support/lateral code | Unchanged | Unchanged | ✓ CORRECT |

---

## Root Cause Classification

**Classification: B - legacy_mode_or_wrong_command**

**Evidence:**
1. Telemetry shows `controller_mode: upright` (legacy)
2. Telemetry shows `capture_gate_enabled: False`
3. Telemetry shows WBC active (legacy behavior)
4. Missing `--controller-mode balance-core` CLI argument
5. Script defaults to legacy mode when argument omitted

**Not:**
- A (metric/reporting error): Telemetry correctly reports legacy mode
- C (stale diagnostic mode): All diagnostics correctly disabled
- D-I (controller regressions): Step E controller never ran

---

## Recommendation

**IMMEDIATE ACTION REQUIRED:**

1. **Re-run visual simulation with correct command:**
   ```bash
   python scripts/simulate_hierarchical_controller.py \
     --steps 5000 \
     --controller-mode balance-core \
     --sagittal-controller velocity-damped \
     --vd-k-position 20.0 \
     --vd-k-velocity 15.0 \
     --vd-max-position-tau 3.0 \
     --vd-enable-capture-gate \
     --vd-capture-gate-use-cp
   ```

2. **Verify balance-core mode active in telemetry:**
   - Check `controller_mode` column shows "balance-core" or non-zero encoding
   - Check `capture_gate_enabled` column shows True (1.0)
   - Check WBC inactive (tau_wbc_norm near zero)

3. **Only after correct-mode run:**
   - Assess posture/stance behavior
   - Compare against legacy baseline
   - Proceed to Phase 2-9 if needed

**DO NOT:**
- Tune controller gains based on legacy-mode visual observations
- Implement posture fixes for legacy-mode behavior
- Claim Step E failed based on wrong-mode telemetry
- Proceed to Phase 2-9 until correct mode confirmed

---

## Files Generated

- `phase1_runtime_configuration_audit.md` (this file)
- `phase1_runtime_configuration_audit.json` (next)

**Status:** Phase 1 complete. Awaiting correct-mode visual run before Phase 2.
