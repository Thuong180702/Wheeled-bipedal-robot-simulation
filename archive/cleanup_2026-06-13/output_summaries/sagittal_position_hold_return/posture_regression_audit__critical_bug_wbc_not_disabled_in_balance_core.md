# CRITICAL BUG REPORT: --controller-mode balance-core Does Not Disable WBC

**Date:** 2026-05-31  
**Severity:** CRITICAL  
**Impact:** All balance-core mode runs are actually running with WBC active

---

## Executive Summary

The `--controller-mode balance-core` CLI argument **does not disable WBC** as intended. The simulation script has a bug where WBC inclusion is controlled by the legacy `stage2b_ablation_mode` flag instead of the `controller_mode` argument.

**Result:** All "balance-core" runs (including V1, V2, V3, and the 2000-step run) have been running with WBC active, making them effectively legacy/hybrid mode, not true balance-core mode.

---

## Evidence

### Telemetry Analysis

**File:** `outputs/hierarchical_controller_sim/telemetry_1780207891.csv` (2000-step run)

**Command used:**
```bash
python scripts/simulate_hierarchical_controller.py \
  --steps 2000 \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-k-position 20.0 \
  --vd-k-velocity 15.0 \
  --vd-max-position-tau 3.0 \
  --vd-enable-capture-gate \
  --vd-capture-gate-use-cp
```

**WBC Status (from telemetry):**
- Step 0: `tau_wbc_scaled_per_joint` = 0.0 (initialization)
- Step 99: `tau_wbc_scaled_per_joint` = -0.046 Nm (**NON-ZERO**)
- Step 1000: `tau_wbc_scaled_per_joint` = 0.037 Nm (**NON-ZERO**)

**Expected:** `tau_wbc_scaled_per_joint` should be 0.0 at ALL steps in balance-core mode.

**Actual:** WBC is active and contributing torque throughout the run.

---

## Root Cause Analysis

### Bug Location

**File:** `scripts/simulate_hierarchical_controller.py`  
**Lines:** 2223-2232

```python
# Stage 2B ablation gating for component isolation
if static_posture_controller is not None and static_feedforward_controller is not None:
    mode = args.stage2b_ablation_mode  # ← BUG: Uses legacy ablation mode
    include_wbc = mode in ["B", "C", "D", "E"] and (not args.disable_wbc_correction)
    include_hip_roll = mode in ["C", "E"] and (not args.disable_hip_roll_centering)
    include_wheel_balance = mode in ["D", "E"] and (not args.disable_wheel_balance)
else:
    mode = "LEGACY"
    include_wbc = True  # ← WBC always enabled in legacy path
    include_hip_roll = True
    include_wheel_balance = True
```

### The Problem

1. **Line 2224**: `mode = args.stage2b_ablation_mode` - Uses legacy ablation mode, NOT `args.controller_mode`
2. **Line 2225**: `include_wbc` is determined by legacy ablation mode ("B", "C", "D", "E"), NOT by `controller_mode`
3. **Line 2230**: If static controllers don't exist, `include_wbc = True` (legacy path)
4. **Missing**: No check for `args.controller_mode == "balance-core"` to force `include_wbc = False`

### Why This Happens

The `controller_mode` argument is only used in two places:
1. `is_balance_core_mode()` function (line 433) - for conditional initialization
2. Validation function (line 453) - to check incompatible flags

But `controller_mode` is **NEVER checked** when deciding whether to include WBC torques in the control loop.

---

## Impact Assessment

### All Previous "Balance-Core" Runs Are Invalid

**Affected runs:**
1. V1 (500 steps) - telemetry_1780202735.csv
2. V2 (1000 steps) - telemetry_1780202891.csv
3. V3 (5000 steps) - telemetry_1780203372.csv
4. 2000-step run - telemetry_1780207891.csv

**Status:** All ran with WBC active despite `--controller-mode balance-core` flag.

**Consequence:** Cannot validate Step E velocity-damped controller in true balance-core mode until bug is fixed.

### Phase 1 Findings Remain Valid

The Phase 1 finding that the **legacy-mode visual run** (telemetry_1780203372.csv showing `controller_mode: upright`) was in wrong mode is still correct. That run was explicitly in legacy/upright mode.

However, the subsequent "balance-core" runs are also compromised by this WBC bug.

---

## Required Fix

### Code Change

**File:** `scripts/simulate_hierarchical_controller.py`  
**Location:** Lines 2223-2232

**Current code:**
```python
if static_posture_controller is not None and static_feedforward_controller is not None:
    mode = args.stage2b_ablation_mode
    include_wbc = mode in ["B", "C", "D", "E"] and (not args.disable_wbc_correction)
    include_hip_roll = mode in ["C", "E"] and (not args.disable_hip_roll_centering)
    include_wheel_balance = mode in ["D", "E"] and (not args.disable_wheel_balance)
else:
    mode = "LEGACY"
    include_wbc = True
    include_hip_roll = True
    include_wheel_balance = True
```

**Proposed fix:**
```python
if static_posture_controller is not None and static_feedforward_controller is not None:
    mode = args.stage2b_ablation_mode
    include_wbc = mode in ["B", "C", "D", "E"] and (not args.disable_wbc_correction)
    include_hip_roll = mode in ["C", "E"] and (not args.disable_hip_roll_centering)
    include_wheel_balance = mode in ["D", "E"] and (not args.disable_wheel_balance)
else:
    mode = "LEGACY"
    include_wbc = True
    include_hip_roll = True
    include_wheel_balance = True

# CRITICAL FIX: Disable WBC in balance-core mode regardless of ablation settings
if args.controller_mode == "balance-core":
    include_wbc = False
    include_hip_roll = False  # Balance-core uses LateralRollBalanceController instead
    include_wheel_balance = False  # Balance-core uses SagittalVelocityDampedBalanceController instead
```

### Verification After Fix

After applying the fix, verify with telemetry:
1. `tau_wbc_scaled_per_joint` = 0.0 at ALL steps
2. `tau_wbc_norm` ≈ 0.0 throughout run
3. `tau_legacy_wheel_balance_norm` = 0.0
4. `tau_legacy_hip_roll_centering_norm` = 0.0
5. `hidden_torque_norm` = 0.0

---

## Alternative Workaround (Not Recommended)

If the fix cannot be applied immediately, use legacy ablation mode "A" to disable WBC:

```bash
python scripts/simulate_hierarchical_controller.py \
  --steps 2000 \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --stage2b-ablation-mode A \
  --disable-wbc-correction \
  --disable-hip-roll-centering \
  --disable-wheel-balance \
  ...
```

**Problem with workaround:** Requires understanding legacy ablation system and may have unintended side effects.

---

## Recommendation

**IMMEDIATE ACTION:**

1. **Apply the code fix** to `scripts/simulate_hierarchical_controller.py` lines 2223-2232
2. **Re-run all balance-core validations** with the fixed script
3. **Verify WBC is actually disabled** by checking `tau_wbc_scaled_per_joint` in telemetry
4. **Only then proceed** with Step E position + posture validation

**DO NOT:**
- Proceed with Step E validation using current buggy script
- Claim balance-core mode works based on current telemetry
- Tune controller gains based on WBC-contaminated runs
- Proceed to Step C until true balance-core mode is validated

---

## Files Generated

- `critical_bug_wbc_not_disabled_in_balance_core.md` (this file)
- `critical_bug_wbc_not_disabled_in_balance_core.json` (next)

**Status:** Bug documented. Awaiting fix before continuing Step E validation.
