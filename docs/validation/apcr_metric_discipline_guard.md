# APCR Metric Discipline Guard

## Purpose

This document defines the required physical drift column for all APCR analyzer scripts and evaluation reports.

## Required Physical Drift Columns (Priority Order)

| Priority | Column Name | Description |
|----------|-------------|-------------|
| 1 | `active_pitch_crossing_signed_error_m` | APCR internal signed error |
| 2 | `sagittal_position_error_m` | Sagittal position error from equilibrium |
| 3 | `support_position_error_m` | Support center position error |
| 4 | `hip_yaw_comp_support_error_m` | Hip yaw compensated support error |

All four columns are equivalent and 100% correlated for drift analysis.

## Forbidden Columns

Do NOT use these columns for drift/P2P analysis:

| Column | Why Forbidden |
|--------|---------------|
| `support_center_x - support_center_ref_x` | Reference-tracking artifact, not physical drift |
| `cp_x` alone | Capture point position, not drift |
| `error_rate` | Derivative, not position |
| `APCR_residual` | Controller output, not drift measurement |
| `torque` field | Control output, not state |
| `reference-tracking difference` | Artificial near-zero artifact |

## Analyzer Requirements

Every APCR analyzer script MUST:

1. **Report the exact column used** in the output:
   ```
   "physical_drift_column_used": "active_pitch_crossing_signed_error_m"
   ```

2. **Fail or warn** if:
   - P2P < 0.01 m while mean_abs > 0.02 m (mirage detection)
   - P2P < 0.01 m while another physical drift column shows P2P > 0.10 m (inconsistency)

3. **Always output** these metrics:
   - min drift (m)
   - max drift (m)
   - P2P = max - min (m)
   - max_abs = max(|min|, |max|) (m)
   - outside ±0.08 (%)
   - outside ±0.15 (%)

## Verification

Run the guard validator:

```python
# Check that analyzer uses correct column
from scripts.validate_apcr_drift_column import validate_analyzer_column

validate_analyzer_column(
    csv_path="outputs/.../telemetry.csv",
    expected_column="active_pitch_crossing_signed_error_m"
)
```

## Mirages to Detect

### Mirage 1: Reference-Tracking Artifact

**Symptom**: P2P < 0.002 m, mean ≈ 0, but robot clearly unstable.

**Root cause**: `support_center_x - support_center_ref_x` where ref is updated to track position.

**Detection**: 
- Check if P2P < 0.01 m AND mean_abs > 0.02 m
- Check if analyzer used `support_center_ref_x` column

**Fix**: Switch to `active_pitch_crossing_signed_error_m`

### Mirage 2: Clipped Range

**Symptom**: P2P looks small but values cluster at boundaries.

**Root cause**: Values capped at ±0.15 or similar limits.

**Detection**: Check for clustering at threshold values.

## Historical Fix

APCR1e audit discovered this issue:
- Old report: P2P < 0.002 m (using wrong column)
- Corrected: P2P = 0.235 m (using correct column)

All future analyses must use the correct physical drift column.