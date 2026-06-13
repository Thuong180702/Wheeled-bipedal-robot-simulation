# Hip-Yaw Disturbance Rejection Investigation - Phase 2 Status

**Date:** 2026-06-04  
**Phase:** 2 (Isolation Experiments) - READY TO RUN  
**Status:** Implementation complete, experiments queued

---

## Completed Work

### Phase 0: Health Check ✅
- All files compile
- All tests pass (40/40 sagittal, 52/52 Step C, 30/30 Step E)
- Controller state verified (kp=15.0, kd=3.0 baseline)

### Phase 1: Requirement Document ✅
- Created `docs/validation/hip_yaw_disturbance_rejection_requirement.md`
- Clarified hip-yaw as disturbance rejection requirement
- Documented that "symptom" does not mean "acceptable"

### Phase 2: Isolation Infrastructure ✅
- **Added CLI arguments to simulation script:**
  - `--shape-kp-hip-yaw`: Override hip-yaw proportional gain
  - `--shape-kd-hip-yaw`: Override hip-yaw damping gain
- **Modified `build_balance_core_controllers()`:**
  - Added `shape_kp_hip_yaw` and `shape_kd_hip_yaw` parameters
  - Implemented override logic with fallback to defaults
- **Created isolation experiment script:**
  - `scripts/run_hip_yaw_disturbance_isolation.py`
  - Implements Experiment D (damping sweep) and E (kp/kd matrix)
  - Automated telemetry collection and analysis
  - Generates comprehensive markdown reports

---

## Ready to Execute

### Experiment D: Damping Sweep
**Purpose:** Determine if increased damping can reject support-drift disturbance

**Parameters:**
- kp fixed at 15.0
- kd values: [3 (baseline), 5, 7, 9, 12]
- Variants: low_0p300, high_0p480, nominal
- Total runs: 3 variants × 5 kd values = 15 runs

**Expected outcomes:**
- If damping helps: hip-yaw error decreases monotonically with kd
- If insufficient: error plateaus, no kd passes gate

### Experiment E: kp/kd Matrix
**Purpose:** Find if any authority combination can pass hip-yaw gate

**Parameters:**
- kp values: [15 (baseline), 20, 25]
- kd values: [3 (baseline via Exp D), 5, 7, 9]
- Variant: low_0p300 only (critical case)
- Total runs: 3 kp × 4 kd - 4 (already in Exp D) = 8 new runs

**Expected outcomes:**
- If authority helps: some (kp, kd) combination passes gate
- If insufficient: no combination passes, need feedforward

---

## Files Changed

### Modified:
- `scripts/simulate_hierarchical_controller.py`
  - Added `--shape-kp-hip-yaw` and `--shape-kd-hip-yaw` arguments
  - Modified `build_balance_core_controllers()` signature
  - Updated `ShapePostureController` instantiation with overrides
  - Added parameter passing to function call

### Created:
- `docs/validation/hip_yaw_disturbance_rejection_requirement.md`
- `scripts/run_hip_yaw_disturbance_isolation.py`
- `scripts/quick_hip_yaw_damping_test.py` (planning/doc tool)
- `outputs/hip_yaw_disturbance_rejection_audit/quick_test/isolation_implementation_requirements.md`

---

## Execution Command

```bash
python scripts/run_hip_yaw_disturbance_isolation.py
```

**Estimated runtime:** 
- ~2 seconds per 1000-step run
- 23 total runs (15 + 8)
- ~46 seconds + analysis time
- Total: ~1-2 minutes

---

## Success Criteria

### Hip-Yaw Gate (Primary):
- `hip_yaw_abs_max <= 0.07 rad`
- At least one configuration passes at low_0p300

### Support Position (Must Not Worsen):
- Support drift must not increase >10% vs baseline
- If hip-yaw fix degrades support: REJECT

### Other Constraints:
- Pitch, roll, height remain valid
- No WBC enabled
- Ownership violations = 0

---

## Next Steps After Experiments

### If Experiments Pass:
1. Select best (kp, kd) combination
2. Implement continuous height-based schedule (Phase 4)
3. Evaluate across full test matrix (Phase 5)
4. Add tests (Phase 6)
5. Generate final report (Phase 7)

### If Experiments Fail:
1. Classify mechanism as `hip_yaw_not_fixable_without_support_fix` or `hip_yaw_requires_feedforward`
2. Document why authority increase insufficient
3. Recommend either:
   - Return to sagittal fix first
   - Implement support-error feedforward (HY-FF)
   - Accept coupled solution requirement

---

## Current Baseline (for comparison)

| Variant | hip_yaw_abs_max | support_error | Status |
|---------|----------------|---------------|--------|
| low_0p300 | 0.2137 rad | 0.243 m | ❌ FAIL (3.05× threshold) |
| high_0p480 | 0.0462 rad | 0.234 m | ✅ PASS hip-yaw, ❌ FAIL support |
| nominal | 0.0392 rad | 0.103 m | ✅ PASS both |

**Target:** Reduce low_0p300 hip-yaw from 0.2137 → ≤0.070 rad (67% reduction required)

---

## Files to Watch

- `outputs/hip_yaw_disturbance_rejection_audit/isolation/isolation_experiment_results.json`
- `outputs/hip_yaw_disturbance_rejection_audit/isolation/isolation_experiment_report.md`
- `outputs/hip_yaw_disturbance_rejection_audit/isolation/*_telemetry.csv`

---

**Status:** READY - All infrastructure complete. Execute isolation experiments when ready.
