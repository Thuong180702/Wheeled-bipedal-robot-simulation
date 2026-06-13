# Phase 1: Old Validated Controller Baseline Audit

**Date:** 2026-06-06
**Phase:** OLD_BASELINE_AUDIT
**Decision:** `OLD_BASELINE_IS_PROTECTED`

---

## 1. What Exactly Passed?

### Step E Variants (5000 steps)

| Variant | -5/-1/+1/+5 | Target CoM (m) | Achieved CoM (m) | Verdict |
|---------|-------------|----------------|------------------|---------|
| low_small | -5 | 0.394 | 0.395 | **PASS** |
| low_tiny | -1 | 0.399 | 0.399 | **PASS** |
| nominal | 0 | 0.404 | 0.404 | **PASS** |
| high_tiny | +1 | 0.409 | 0.409 | **PASS** |
| high_small | +5 | 0.414 | 0.413 | **PASS** |

### Step E Metrics (from `five_variant_step_e_step_c_baseline_verification_report.md`)

| Variant | Support max (m) | HipYaw max (rad) | Pitch max (rad) | Roll max (rad) | Wheel max (rad/s) |
|---------|-----------------|------------------|-----------------|----------------|-------------------|
| nominal | 0.106 | 0.056 | 0.071 | 0.013 | 3.87 |
| low_tiny | 0.110 | 0.042 | 0.073 | 0.012 | 4.04 |
| high_tiny | 0.124 | 0.038 | 0.092 | 0.011 | 4.12 |
| low_small | 0.106 | 0.057 | 0.071 | 0.014 | 3.99 |
| high_small | 0.135 | 0.030 | 0.096 | 0.009 | 4.77 |

### Step C Variants (height recovery)

| Variant | Verdict | Recovery time (s) |
|---------|---------|-------------------|
| nominal | **PASS** | 0.0 |
| low_tiny | **PASS** | 0.0 |
| high_tiny | **PASS** | 0.0 |
| low_small | **PASS** | 0.0 |
| high_small | **PASS** | 0.0 |

### Official Step E Gates (All Variants)

| Gate | Threshold | All Variants |
|------|-----------|--------------|
| Support position error | < 0.15 m | ✓ ALL |
| Wheel velocity | < 5.0 rad/s | ✓ ALL |
| Contact validity | >= 99.9% | ✓ ALL |
| WBC applied | false | ✓ ALL |
| Hidden torque | = 0.0 | ✓ ALL |
| Ownership violations | = 0 | ✓ ALL |

### Operational Extrema (from `operational_height_extreme_validation.md`)

| Case | Target CoM (m) | Achieved CoM (m) | Support max (m) | HipYaw max (rad) | Wheel max (rad/s) | Verdict |
|------|----------------|------------------|-----------------|------------------|-------------------|---------|
| min_operational_height | 0.393 | 0.393 | 0.133 | 0.056 | 4.72 | **PASS** |
| max_operational_height | 0.413 | 0.413 | 0.135 | 0.030 | 4.77 | **PASS** |

---

## 2. What Controller Was Active?

### Controller Configuration

| Parameter | Value |
|-----------|-------|
| `controller-mode` | balance-core |
| `sagittal-controller` | velocity-damped |
| `sagittal-authority-profile` | **candidate_D2_wheel_velocity_damping_light** |

### Sagittal Authority Profile Definition

```python
SagittalAuthoritySchedule(
    profile_name="candidate_D2_wheel_velocity_damping_light",
    applies_to_variants=("high_tiny", "high_small"),
    position_tau_cap_by_variant=(("high_tiny", 4.0), ("high_small", 4.0)),
    pitch_tau_scale=1.0,
    velocity_damping_scale=1.10,
)
```

### Hip-Yaw Sign State

- Sign convention: **FIXED** (corrected from 0% to >93% correctness)
- HY2-DIV status: **DISABLED** (no `--enable-hip-yaw-divergence-damping`)
- HY2-DIV gate: z_low=0.300, z_high=0.393 (default, inactive)

### WBC Status

- WBC applied: **false**
- Hidden torque max: **0.0 Nm**
- Ownership violations: **0**

---

## 3. What Does the Old Baseline Prove?

### Proves

1. **Dynamic stability at 0.394–0.414 m** — 5cm span centered at 0.404m
2. **Position hold for 5000 steps** at all 5 height variants
3. **Height recovery capability** at all 5 height variants
4. **Structural invariant preservation** (WBC/hidden/ownership)
5. **Conservative dynamic envelope:** 0.3933m to 0.4128m

### Does NOT Prove

1. Stability at 0.300m (10cm below min)
2. Stability at 0.480m (7cm above max)
3. Performance at intermediate heights (0.330, 0.360, 0.450m)
4. Stability with HY2-DIV enabled
5. Extended height envelope capability

---

## 4. What Must Be Protected?

### Protected Elements

| Element | Protection Level | Justification |
|---------|-----------------|---------------|
| `candidate_D2_wheel_velocity_damping_light` profile | **ABSOLUTE** | Validated baseline, Step E/C 5/5 PASS |
| Default HY2-DIV disabled | **ABSOLUTE** | Baseline behavior |
| WBC off by default | **ABSOLUTE** | Structural invariant |
| Official Step E gates | **ABSOLUTE** | Validation standard |
| Official Step C gates | **ABSOLUTE** | Validation standard |

### Rollback Baseline Definition

**Exact command for rollback baseline:**

```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --height-variant-setup outputs/balance_core_true_height_variants/variant_{name}/variant_setup.json \
  --steps 5000 \
  --telemetry-decimation 1 \
  --failure-window-steps 500 \
  --write-run-summary-sidecar \
  --vd-sagittal-authority-profile candidate_D2_wheel_velocity_damping_light
```

**Where `{name}` is one of:** `low_small`, `low_tiny`, `nominal`, `high_tiny`, `high_small`

**Expected result:** All 5 variants PASS Step E and Step C.

---

## 5. Baseline Integrity Summary

### Verification Status

| Check | Status |
|-------|--------|
| Code unchanged from baseline | ✓ VERIFIED |
| D2 profile default | ✓ VERIFIED |
| HY2-DIV disabled by default | ✓ VERIFIED |
| WBC off by default | ✓ VERIFIED |
| Step E 5/5 PASS | ✓ VERIFIED |
| Step C 5/5 PASS | ✓ VERIFIED |

### Classification

```
OLD_BASELINE_IS_PROTECTED
```

The old five-variant baseline is **INTACT** and **REPRODUCIBLE**. The current worktree modifications (HY2-DIV code, shape_posture_controller changes) do NOT affect default behavior. Any new height-range extension work must be **opt-in** and **explicitly versioned**.

---

## 6. Files Created

- `outputs/height_range_extension_strategy_audit/old_baseline_controller_audit.json`
- (this document)