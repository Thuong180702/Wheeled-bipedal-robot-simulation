# APCR1n Feature Activation Investigation - Final Report

**Date:** 2026-06-11  
**Task:** Investigate why APCR1n features never activated in successful 5000-step run  
**Classification:** **APCR1N_RUNTIME_CONFIG_FEATURE_CODE_NOT_PRESENT**

---

## Executive Summary

The APCR1n "recenter priority torque boost" features (wheel damping override, position cap boost) **never activated because the feature implementation code did not exist during the successful runs**. The config values were defined in `simulate_hierarchical_controller.py`, but the runtime logic in `sagittal_velocity_damped_balance_controller.py` was added AFTER the runs were executed.

The APCR1n 5000-step success is valid but represents an **APCR1h-lite baseline** (soft-band proportional mode with default D2 parameters), not the intended augmented profile.

---

## Root Cause

### Timeline

1. **Before June 11 09:41:** APCR1n config added to `simulate_hierarchical_controller.py` (lines 1204-1269)
2. **June 11 09:41:** APCR1n 1000-step run - used OLD controller code
3. **June 11 18:03:** APCR1n 2000-step fair comparison - used OLD controller code
4. **June 11 18:19:** APCR1n 5000-step validation - used OLD controller code
5. **After June 11 18:19:** APCR1n feature implementation added to controller (uncommitted)

### Evidence

**1. Telemetry Column Absence**

Expected APCR1n-specific columns from current code:
- `apcr1n_recenter_priority_active`
- `apcr1n_startup_guard_active`
- `apcr1n_wheel_damping_override_active`
- `apcr1n_position_cap_boost_active`
- 12 additional APCR1n diagnostic columns

**Actual columns found:** 0

**2. Runtime Config Values**

| Parameter | Config (lines 1210-1212) | Runtime (telemetry step 100) |
|-----------|-------------------------|------------------------------|
| `continuous_max_position_tau` | True | N/A (not consumed) |
| `max_position_tau_nominal` | 4.0 | 3.0 (default) |
| `velocity_damping_scale` | 1.10 | 1.0 (default) |
| `position_cap_normal_nm` | 4.0 | 3.0 (default) |

**3. Git Status**

```bash
git diff wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py
```

Shows APCR1n feature code (lines 1601-1730, 3177-3192) marked with `+` (uncommitted additions).

**4. Feature Eligibility Analysis**

If the feature code had been present during the APCR1n 1000-step run:

| Condition | Steps | Percentage |
|-----------|-------|------------|
| \|e\| > 0.08 m | 626 / 1000 | 62.6% |
| moving_away | 485 / 1000 | 48.5% |
| **Eligible for drift_priority** | **280 / 1000** | **28.0%** |

The `_apc_drift_priority_active` flag (which gates APCR1n feature activation at line 1639) should have been True for 280 steps, but no drift_priority telemetry columns exist.

---

## What Actually Ran

The APCR1n runs executed an **APCR1h-equivalent soft-band proportional mode**:

- **APC mode:** Proportional soft band (no hysteresis)
- **APC state:** NEUTRAL for all steps (no RECENTER state)
- **Max position tau:** 3.0 Nm (default, not 4.0)
- **Velocity damping scale:** 1.0 (default, not 1.10)
- **Wheel damping override:** Not present
- **Position cap boost:** Not present
- **Recenter priority:** Not present

**Performance achieved (5000 steps, low_0p300):**
- Survived: 5000 / 5000
- Max |e|: 0.1714 m
- P2P: 0.2099 m
- Outside ±0.15 m: 1.1%
- Final e: -0.0129 m
- Wheel velocity max: 4.77 rad/s

**Classification:** APCR1N_LOW_0P300_5000_PASS_WITH_MONITORING (baseline, no features)

---

## Phase 1 Resolution: Config Inconsistency Explained

The documentation inconsistency from prior reports is now resolved:

**Why config values didn't match runtime:**
- Config values exist in `SagittalAuthoritySchedule` dataclass
- Controller runtime logic to READ and USE those values was not present
- Default base class values were used instead

**Conclusion:** Not a config bug, but a **feature implementation timing issue**.

---

## Implications

### 1. Success Validity

✅ **APCR1n 5000-step success is VALID** - it passed Step E low_0p300 criteria  
⚠️ **But it succeeded WITHOUT the new recenter priority features**  
📊 **Represents an APCR1h-lite baseline**, not the augmented profile

### 2. Feature Activation Audit (Phase 2)

❌ **Phase 2 is UNNECESSARY** - features never existed at runtime  
✅ **No activation logic bug** - code wasn't present to activate  
✅ **No gate tuning needed** - gates never blocked anything

### 3. Ablation Study Redesign (Phase 4-8)

The original ablation plan must be revised:

**Old plan (assumed features were wired but inactive):**
- Feature 1 only, Feature 2 only, Feature 3 only, F12, F13, F23, F123

**New plan (features exist in uncommitted code but untested):**
- **Baseline:** Current APCR1n (no features, runs as APCR1h-lite)
- **Candidate:** APCR1n with feature code enabled
- **Ablation:** Only proceed if candidate improves baseline

### 4. Config Parameter Wiring

The following config parameters need runtime verification:
- `continuous_max_position_tau` (line 1210)
- `max_position_tau_nominal` (line 1211)
- `velocity_damping_scale` (line 1212)
- `position_cap_normal_nm` (line 1262)

Currently they are defined but NOT consumed by the controller.

---

## Recommendations

### Immediate Actions

**DO NOT:**
- ❌ Re-run APCR1n 5000 yet
- ❌ Proceed to ablation study
- ❌ Claim APCR1n features are validated

**DO:**
1. ✅ Commit current APCR1n feature code
2. ✅ Add tests for APCR1n feature activation logic
3. ✅ Run smoke test (100 steps) with full telemetry validation
4. ✅ Verify APCR1n telemetry columns exist in output
5. ✅ Verify features CAN activate under eligible conditions
6. ✅ Verify config consumption (max_position_tau, velocity_damping_scale)

### Phase 1b: Feature Code Verification (NEW PHASE)

Before any ablation study:

**Step 1: Smoke Test**
```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile APCR1n_recenter_priority_torque_boost \
  --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json \
  --steps 100 \
  --telemetry-decimation 1 \
  --write-run-summary-sidecar
```

**Step 2: Telemetry Validation**
- Verify all 16 APCR1n telemetry columns exist
- Verify `effective_max_position_tau` = 4.0 (not 3.0)
- Verify `effective_velocity_damping_scale` = 1.10 (not 1.0)
- Verify `apcr1n_startup_guard_active` = True for steps 0-99
- Verify `apcr1n_startup_guard_active` = False for step 100+

**Step 3: Activation Trigger Test**

Create a high-drift scenario to trigger features:
- Initial position offset e = 0.10 m
- Verify `apcr1n_recenter_priority_active` becomes True
- Verify `apcr1n_wheel_damping_override_active` activates
- Verify `apcr1n_position_cap_boost_active` activates when saturated
- Verify `apcr1n_position_cap_current` increases from 4.0 → 5.0

**Step 4: Tests**
```bash
pytest tests/test_sagittal_velocity_damped_balance_controller.py::test_apcr1n_* -v
```

Add new tests:
- `test_apcr1n_features_activate_when_drift_priority_active`
- `test_apcr1n_wheel_damping_override_fights_drift_only`
- `test_apcr1n_position_cap_boost_requires_safety_gates`
- `test_apcr1n_config_parameters_consumed_at_runtime`
- `test_apcr1n_telemetry_columns_exist`

**Step 5: Decision Gate**

Only proceed to ablation IF:
- ✅ All APCR1n telemetry columns exist
- ✅ Config parameters (max_position_tau=4.0, velocity_damping_scale=1.10) are consumed
- ✅ Features CAN activate under eligible conditions
- ✅ Startup guard works correctly
- ✅ Safety gates block correctly
- ✅ All tests pass

If any fails → fix before ablation.

---

## Revised Task Plan

### ~~Phase 0: Health Check~~ ✅ COMPLETE

All 300 tests pass.

### ~~Phase 1: Resolve Config Inconsistency~~ ✅ COMPLETE

**Finding:** Features were never implemented during successful runs.  
**Classification:** APCR1N_RUNTIME_CONFIG_FEATURE_CODE_NOT_PRESENT

### **Phase 1b: Feature Code Verification** ⬅️ **NEW PHASE, START HERE**

Verify current APCR1n feature code runtime behavior.

**Actions:**
1. Run smoke test (100 steps)
2. Validate telemetry columns
3. Verify config consumption
4. Test activation trigger
5. Add feature activation tests
6. Decision gate: proceed to ablation only if all checks pass

**Deliverables:**
- `docs/validation/apcr1n_feature_code_verification_report.md`
- `outputs/.../apcr1n_smoke_100_feature_verification/telemetry.csv`
- New tests in `tests/test_sagittal_velocity_damped_balance_controller.py`

### ~~Phase 2: Feature Activation Audit~~ ❌ SKIPPED

Features were never present. No audit needed.

### Phase 3: Fix Activation Logic ⬅️ **CONDITIONAL**

Only if Phase 1b reveals bugs.

### Phase 4-8: Ablation Study ⬅️ **REDESIGNED**

Simplified two-way comparison:
- **Baseline:** Current APCR1n (no features, known good)
- **Candidate:** APCR1n with feature code enabled

If candidate improves, then run detailed ablation.

### Phase 9-11: Best Candidate Validation, Analysis, Final Report

Unchanged, but only if candidate beats baseline.

---

## Classification: APCR1N_RUNTIME_CONFIG_FEATURE_CODE_NOT_PRESENT

**Definition:** The APCR1n profile config exists and was referenced during runs, but the controller runtime logic to implement the features was not present in the code version used. The runs succeeded using an APCR1h-lite baseline without recenter priority augmentation.

---

## Answer to Original Question

**"Why did APCR1n features not activate?"**

**Answer:** The APCR1n features (recenter priority, wheel damping override, position cap boost) did not activate because **the feature implementation code did not exist during the successful runs**. The config values were defined, but the runtime logic was added later and remains uncommitted. The APCR1n 5000-step success represents a valid baseline, but NOT the augmented recenter-priority profile that was intended.

**Next step:** Phase 1b - verify the current uncommitted feature code works correctly before proceeding to any ablation study.
