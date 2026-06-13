# APCR1e Metric Provenance Final Report

## Executive Summary

**Classification: `APCR1E_ANALYSIS_INVALID_RECOMPUTED_NOT_BETTER`**

The APCR1e report contains internally contradictory claims because it mixed two different telemetry columns:

1. **Wrong column** (`support_center_x - support_center_ref_x`): P2P < 0.002 m (near-zero, a mirage)
2. **Correct column** (`active_pitch_crossing_signed_error_m`): P2P = 0.235 m, ranging [-0.064, +0.17] m

The "drift eliminated" claim is **INVALID** because `support_center_ref_x` is updated to track `support_center_x`, making their difference appear near-zero. This is a reference-tracking artifact, not physical drift.

**Corrected Conclusion**: APCR1e P2P = 0.235 m at 2000 steps, which is **WORSE** than D2 (0.161 m) and only marginally better than APCR1c (0.240 m).

---

## Answer to Required Questions

### 1. Which exact column produced the P2P < 0.002 m result?

`support_center_x - support_center_ref_x`

The analyzer computed drift as the difference between the support center position and its reference. Since the reference is updated to track the position (equilibrium capture after variant), this difference is near-zero throughout.

### 2. Is that column physical signed support drift?

**NO.** This is a mirage because:
- `support_center_ref_x` is updated to match `support_center_x` after each height variant
- The difference shows how well the reference is tracking, not physical drift
- Values are near machine precision (±0.001 m), not physical drift magnitudes

### 3. Which exact column produced [-0.064, +0.17]?

`active_pitch_crossing_signed_error_m`

This is the APCR internal signed error, which equals:
- `sagittal_position_error_m`
- `support_position_error_m`
- `hip_yaw_comp_support_error_m`

All four columns are 100% correlated with zero max difference.

### 4. Which metric is correct for support/position drift?

**`active_pitch_crossing_signed_error_m`** (or any of its equivalent columns).

This is the true physical signed drift that APCR is trying to correct.

### 5. Were APCR1e results wrong, mixed, or valid?

**MIXED** - The results contain both:
- Correct elements: adaptive authority behavior (tau_max=1.16 Nm is real)
- Invalid elements: P2P < 0.002 m claim is wrong

### 6. Corrected APCR1e metrics

| Metric | 500-step | 2000-step |
|--------|----------|-----------|
| Min Drift | -0.027 m | -0.064 m |
| Max Drift | +0.162 m | +0.171 m |
| P2P | 0.189 m | 0.235 m |
| Outside ±0.15 | 9.8% | 10.1% |
| Outside ±0.08 | 42.4% | 45.4% |
| Mean | +0.060 m | +0.062 m |
| Abs Mean | 0.069 m | 0.076 m |

### 7. Does APCR1e beat APCR1d/APCR1c after correction?

| Profile | P2P | Outside ±0.15 | Verdict |
|---------|-----|---------------|---------|
| D2 | 0.162 m | 19.2% | **BEST P2P** |
| APCR1c | 0.240 m | 12.6% | Baseline |
| APCR1d | FAIL@18 | N/A | Failed |
| **APCR1e** | **0.235 m** | **10.1%** | Slightly better than APCR1c, **WORSE than D2** |

APCR1e is slightly better than APCR1c on P2P (0.235 vs 0.240) and outside ±0.15 (10.1% vs 12.6%), but **WORSE than D2** on P2P (0.235 vs 0.162).

### 8. Is adaptive authority behavior real?

**YES.** Evidence:
- APCR tau reached 1.16 Nm (exceeds APCR1d's 0.75 Nm cap)
- APCR was active 60.6% of steps
- Mean tau = 0.53 Nm
- The adaptive boosting mechanism appears to function correctly

### 9. Should APCR1e proceed to 5000-step after corrected metrics?

**NO.** `APCR1E_ANALYSIS_INVALID_RECOMPUTED_NOT_BETTER`

Reason: Corrected P2P (0.235 m) is worse than D2 baseline (0.162 m). Running 5000-step would not improve this fundamental issue.

### 10. What analyzer/reporting fixes were made?

1. **Analyzer bug identified**: `compute_signed_drift_metrics()` used `support_center_x - support_center_ref_x`, which is a reference-tracking artifact, not physical drift.

2. **Should use**: `active_pitch_crossing_signed_error_m` (or equivalent columns).

3. **Fix required**: 
   - Change analyzer to use the correct physical drift column
   - Add validation: if `abs(drift) < 0.01` for a low-height run, flag as suspicious
   - Add consistency check: P2P should be plausible relative to mean abs error

---

## Detailed Technical Analysis

### Column Relationship Diagram

```
support_center_x (physical position)
    ↓
    ├── Raw value: min=-0.001, max=+0.001 (small oscillation)
    │
    └── minus
        ↓
    support_center_ref_x (updated to track position)
        ↓
        ├── After equilibrium_capture_after_variant, ref = position
        ├── Difference is near-zero throughout
        └── This is NOT physical drift - it's reference tracking
```

```
active_pitch_crossing_signed_error_m (APCR internal error)
    ↓
    ├── Equals: sagittal_position_error_m = support_position_error_m
    ├── Raw value: min=-0.064, max=+0.171 (real physical drift)
    └── This IS physical drift that APCR is correcting
```

### Why the Mirage Occurs

1. Height variant is applied → equilibrium shifts
2. `equilibrium_capture_after_variant` = True
3. `support_reference_captured_after_variant` = True  
4. `support_center_ref_x` is updated to match current `support_center_x`
5. Subsequent drift is measured against this NEW reference
6. Appears as near-zero drift because reference moved with equilibrium

### Adaptive Authority Behavior (Valid)

Despite incorrect drift metrics, the adaptive authority mechanism appears to work:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| APCR tau max | 1.16 Nm | Exceeds APCR1d's 0.75 Nm cap ✓ |
| APCR tau mean | 0.53 Nm | Higher than APCR1c's ~0.50 Nm |
| APCR active % | 60.6% | Active majority of steps |
| Startup boost | Engaged | Strong correction at beginning |
| Velocity decay disabled | Documented | APCR prevents decay |

---

## Recommendations

1. **Do NOT run 5000-step** for APCR1e - metrics show worse P2P than D2
2. **Reconsider APCR profiles**: D2 (0.162 m P2P) outperforms all APCR profiles on P2P
3. **Fix analyzer** before claiming any future results
4. **Consider reverting to D2** as the best profile for low-height standing

---

## Files Generated

- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1e_metric_provenance_audit/metric_contradiction_resolution.json`
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1e_metric_provenance_audit/apcr1e_corrected_comparison.csv`
- `docs/validation/apcr1e_metric_provenance_final_report.md` (this file)
