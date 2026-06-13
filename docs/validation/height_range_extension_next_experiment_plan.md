# Phase 6: Height Range Extension - Next Experimental Plan

**Date:** 2026-06-06
**Phase:** HEIGHT_RANGE_EXTENSION_NEXT_EXPERIMENT
**Status:** READY_FOR_EXECUTION (do not execute yet)

---

## 1. Recommended First Experiment

### Experiment 0: Baseline Ladder Mapping

**Purpose:** Establish where the old controller (D2, no HY2-DIV) first fails on the height ladder.

**Rationale:**
Before trying HY2-DIV or any extensions, we need to know:
1. Where does the baseline controller start failing?
2. Is the failure gradual or sudden?
3. Does the failure mode match expectations?

**Profile Configuration:**
```python
profile_name = "baseline_ladder_mapping"
sagittal_authority_profile = "candidate_D2_wheel_velocity_damping_light"
hy2_div_enabled = False  # No HY2-DIV - pure baseline
```

**Heights to Test:**

| Side | Heights (m) | Setup Source | Priority |
|------|-------------|--------------|----------|
| Low | 0.380, 0.360, 0.340, 0.330, 0.320 | Generate new / use existing | HIGH |
| High | 0.430, 0.450, 0.465 | Generate new / use existing | HIGH |

**Test Sequence:**
1. Run each height for 500 steps (screening)
2. Measure: survival, contact, height_error, divergence_RMS, hip_yaw_abs_max, support_max
3. Stop at first failure point
4. Record pass/fail boundary

**Expected Outcome:**
- Baseline should pass ~0.360-0.380m (mid-gap)
- Baseline should fail at 0.330m or below (very crouched)
- Baseline should fail at 0.450m+ (taller, support drift)

---

## 2. Detailed Experiment: Baseline D2 at 0.360m and 0.450m

### 2.1 Command Template

```bash
# LOW SIDE: 0.360m
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile candidate_D2_wheel_velocity_damping_light \
  --height-variant-setup outputs/physical_target_height_setups/low_0p360_setup.json \
  --steps 500 \
  --telemetry-decimation 1 \
  --output-dir outputs/height_extension_baseline_ladder/low_0p360_500step

# HIGH SIDE: 0.450m
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile candidate_D2_wheel_velocity_damping_light \
  --height-variant-setup outputs/physical_target_height_setups/high_0p450_setup.json \
  --steps 500 \
  --telemetry-decimation 1 \
  --output-dir outputs/height_extension_baseline_ladder/high_0p450_500step
```

### 2.2 Metrics to Record

| Metric | 500-step Gate | Interpretation |
|--------|--------------|----------------|
| `survived_500` | true/false | Basic survival |
| `contact_valid_pct` | >= 99.5% | Contact maintained |
| `height_error_max_m` | < 0.05 m | Height tracking |
| `divergence_RMS` | Record | Posture stability |
| `hip_yaw_abs_max` | < 0.30 rad | Hip-yaw bounded |
| `support_max_m` | < 0.20 m | Support drift |
| `roll_max_rad` | < 0.02 rad | Roll bounded |
| `wbc_applied` | false | Structural invariant |
| `ownership_violations` | = 0 | Structural invariant |

### 2.3 Success Criteria

**Pass:**
- All 500 steps survived
- All structural invariants maintained
- Posture metrics within acceptable range

**Fail:**
- Contact loss
- Height collapse
- Roll instability
- Structural invariant violation

**Ambiguous (needs longer test):**
- All gates pass at 500 steps but trajectory trending toward failure
- → Run 2000-step follow-up

---

## 3. Experiment Sequence After Baseline Mapping

### If Baseline Passes at 0.360m/0.450m

```
Experiment 1: Baseline at 0.330m
  → If PASS: Continue down ladder
  → If FAIL: Stop, try HY2-DIV A0

Experiment 2: Baseline at 0.300m
  → If PASS: Low-side baseline works to 0.300m
  → If FAIL: Try HY2-DIV A0 at 0.300m
```

### If Baseline Fails at 0.360m/0.450m

```
Experiment 1: HY2-DIV A0 at 0.360m
  → If PASS: Low-side extension possible with A0
  → If FAIL: Try A3 or investigate failure

Experiment 1: HY2-DIV A0 at 0.450m
  → If PASS: High-side extension possible with A0
  → If FAIL: Try extended gate (B1) or investigate
```

---

## 4. Stop Conditions

### Immediate Stop (experiment terminates)

| Condition | Action |
|-----------|--------|
| Contact loss | Stop, record FAIL, investigate |
| Height collapse (z < 0.20m) | Stop, record FAIL, investigate |
| Roll instability | Stop, record FAIL, investigate |
| WBC applied = true | Stop, record FAIL, revert controller change |
| Ownership violations > 0 | Stop, record FAIL, revert controller change |

### Continue with Caution

| Condition | Action |
|-----------|--------|
| Divergence growing but bounded | Run longer (2000 steps) |
| Support drift increasing | Monitor, record, defer to support phase |
| Height error increasing | Monitor, record |

---

## 5. What Gets Frozen After Each Phase

### After Experiment 0 (Baseline Mapping)

| Element | Status |
|---------|--------|
| `candidate_D2_wheel_velocity_damping_light` | **FROZEN** - do not modify |
| Default HY2-DIV disabled | **FROZEN** |
| Official Step E/C gates | **FROZEN** |
| Baseline 5-variant results | **FROZEN** |

### After First Extension Pass

| Element | Status |
|---------|--------|
| First height extension profile | **VERSIONED** - named explicitly |
| Extension profile | **OPT-IN ONLY** - not default |
| Any rollback option | **AVAILABLE** - documented |

---

## 6. Rollback Procedures

### Baseline Rollback

```bash
# Verify baseline still works
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile candidate_D2_wheel_velocity_damping_light \
  --height-variant-setup outputs/balance_core_true_height_variants/variant_nominal/variant_setup.json \
  --steps 5000
```

Expected: All 5 variants PASS (as verified in Phase 1).

### HY2-DIV Rollback

```bash
# Disable HY2-DIV and rerun
python scripts/simulate_hierarchical_controller.py \
  ... \
  --no-enable-hip-yaw-divergence-damping  # or remove flag
```

---

## 7. Required Setup Files to Generate

### Generate Missing Setup Files

| Height (m) | Generate | Source |
|------------|---------|--------|
| 0.380 | YES | Interpolate between 0.394 and 0.360 |
| 0.340 | YES | Interpolate between 0.360 and 0.330 |
| 0.320 | YES | Interpolate between 0.330 and 0.300 |
| 0.430 | YES | Interpolate between 0.414 and 0.450 |
| 0.465 | YES | Interpolate between 0.450 and 0.480 |

**Note:** Setup generation is part of Experiment 0 preparation, not the experiment itself.

---

## 8. Files to Create for Experiment

### Before Running

1. `outputs/height_extension_baseline_ladder/` directory
2. Setup files for missing heights (0.380, 0.340, 0.320, 0.430, 0.465)
3. Experiment parameters JSON

### After Running

1. Telemetry CSVs for each height
2. Metrics summary JSON per height
3. Pass/fail summary JSON
4. Failure analysis report (if any)

---

## 9. Decision Points Summary

| Decision | If YES | If NO |
|----------|--------|-------|
| Baseline passes at 0.360m? | Continue down ladder | Try A0 at 0.360m |
| Baseline passes at 0.450m? | Continue up ladder | Try A0 at 0.450m |
| A0 helps at 0.300m? | Extend to 0.300m | Try A3 or stop |
| A0 helps at 0.480m? | Extend to 0.480m | Try B1 or stop |

---

## 10. Files Created

- `outputs/height_range_extension_strategy_audit/height_range_extension_next_experiment_plan.json`
- (this document)

**Note:** This plan is ready for execution. Do NOT execute until user approves.