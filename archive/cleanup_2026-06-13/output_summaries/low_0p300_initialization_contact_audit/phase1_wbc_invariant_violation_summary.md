# Phase 1: WBC Invariant Violation - ROOT CAUSE FOUND

Date: 2026-06-05

## Executive Summary

**CLASSIFICATION: WBC_INVARIANT_VIOLATION**

low_0p300 fails at step 17 because the simulation is running in **LEGACY mode with WBC applied**, not balance-core mode with WBC off.

## Evidence

From telemetry analysis:

```json
{
  "ablation_mode": "LEGACY",
  "tau_wbc_norm_max": 114.94,
  "tau_wbc_max": 57.0,
  "tau_wbc_after_authority_clip": "exists",
  "steps_completed": 17,
  "contact_lost_at_step": 12
}
```

## Root Cause

The evaluation scripts (`scripts/smoke_test_joint_profiles.py`, `scripts/evaluate_joint_low_height_sagittal_yaw_fix.py`) are **missing the `--controller-mode balance-core` flag**.

From [scripts/simulate_hierarchical_controller.py:726](scripts/simulate_hierarchical_controller.py#L726):

```python
def is_balance_core_mode(args) -> bool:
    return args.controller_mode in {"balance-core", "standing-balance"}
```

Without this flag, the simulation defaults to legacy mode.

## Impact

1. **WBC torques (up to 57 Nm per joint) interfere with sagittal controller**
2. Robot receives conflicting control signals from WBC + sagittal controller
3. WBC attempts to stabilize using QP wrench optimization
4. Sagittal controller attempts to stabilize using wheel torques only
5. Conflict causes wheel contact loss by step 12, pitch collapse by step 17

## Why This Violates Requirements

Balance-core controller evaluations **MUST have WBC OFF** because:
- Sagittal controller is designed to work standalone via wheel torques
- WBC adds leg-joint torques that interfere with wheel-based stabilization
- Fair controller comparison requires identical torque ownership
- J0-J3 profile differences cannot be assessed with WBC interference

## Fix Required

Add `--controller-mode balance-core` to all evaluation commands:

**Before:**
```bash
python scripts/simulate_hierarchical_controller.py \
  --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json \
  --steps 500 \
  --vd-sagittal-authority-profile J1
```

**After:**
```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json \
  --steps 500 \
  --vd-sagittal-authority-profile J1
```

## Files to Update

1. `scripts/smoke_test_joint_profiles.py` - add `--controller-mode`, `balance-core` to cmd list
2. `scripts/evaluate_joint_low_height_sagittal_yaw_fix.py` - add `--controller-mode`, `balance-core` to cmd list

## Next Steps

1. Update evaluation scripts to add `--controller-mode balance-core`
2. Rerun Phase 1 audit with corrected command
3. Verify WBC is OFF (tau_wbc_norm should be 0 or near-zero)
4. If WBC is OFF, proceed to Phase 2-7 of initialization audit
5. Rerun smoke tests with corrected commands

## Status

**BLOCKER IDENTIFIED - DO NOT PROCEED WITH PHASE 6 EVALUATION UNTIL FIXED**

The current low_0p300 failure is NOT a valid controller comparison because WBC interference masks true sagittal controller behavior.
