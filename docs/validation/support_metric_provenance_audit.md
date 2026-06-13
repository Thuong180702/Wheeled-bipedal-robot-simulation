# Support Metric Provenance Audit

## Summary

**Classification**: `E1_ANALYZER_USED_WRONG_SUPPORT_METRIC`

The E1 500-step analyzer used the wrong support metric (`abs(cp_x)` instead of `support_position_error_m`), causing a 37x discrepancy with the official Step E report.

## Contradiction Resolved

| Source | Support Metric Used | Value (first 500 rows) |
|--------|---------------------|------------------------|
| Official D2 report | `support_position_error_m` (Euclidean distance) | **0.176 m** |
| E1 analyzer | `abs(cp_x)` (raw capture point x) | **0.0047 m** |

**Resolution**: The official Step E metric is `support_position_error_m` (Euclidean distance of support_center error). The E1 analyzer incorrectly used `abs(cp_x)`.

## Root Cause

**File**: `scripts/analyze_e1_500_before_fix.py`  
**Line**: 33

```python
# WRONG:
support_pos_err = np.abs(df['cp_x'].values) if 'cp_x' in df.columns else np.abs(df['com_error_x'].values)

# CORRECT:
support_pos_err = np.abs(df['support_position_error_m'].values)
```

### Why This Is Wrong

- `cp_x` = capture point x position (raw coordinate, ~0.005 m magnitude)
- `support_position_error_m` = Euclidean distance between support_center and its reference (~0.176 m max)

These are fundamentally different quantities:
- `cp_x` measures where the capture point IS
- `support_position_error_m` measures how far the support center has drifted from its reference

## Evidence

### Telemetry Column Analysis

Both D2 and E1 telemetry files contain `support_position_error_m` column with identical values:

| Step | D2 support_position_error_m | E1 support_position_error_m | D2 support_center_error_mag |
|------|---------------------------|---------------------------|---------------------------|
| 0 | 0.000000 m | 0.000000 m | 0.000000 m |
| 50 | 0.061959 m | 0.061959 m | 0.061960 m |
| 91 | 0.150413 m | 0.150413 m | 0.150417 m |
| 100 | 0.161284 m | 0.161284 m | 0.161288 m |
| 328 | -0.002098 m | -0.002098 m | 0.002321 m |
| 499 | 0.057982 m | 0.057982 m | 0.057997 m |

**Note**: `support_position_error_m` max = **0.175687 m** at step ~91 (crossing 0.15 m threshold)

### E1 Analyzer Confusion

The E1 analyzer confused `cp_x` with support position error:

| Metric | E1 Analyzer | Official Step E |
|--------|-------------|-----------------|
| abs(cp_x) max | 0.004710 m | N/A |
| support_position_error_m max | **0.175687 m** | **0.176 m** |
| First crossing > 0.15 m | None | Step 91 |

## Fix Required

**File**: `scripts/analyze_e1_500_before_fix.py`

**Change line 33**:
```python
# OLD (WRONG):
support_pos_err = np.abs(df['cp_x'].values) if 'cp_x' in df.columns else np.abs(df['com_error_x'].values)

# NEW (CORRECT):
support_pos_err = np.abs(df['support_position_error_m'].values) if 'support_position_error_m' in df.columns else np.abs(df['cp_x'].values)
```

## Corrected E1 500-Step Results (using official metric)

After fixing the metric, E1 vs D2 comparison:

| Metric | E1 Before Fix | D2 Baseline | Official Threshold |
|--------|---------------|-------------|-------------------|
| support_position_error max | **0.175687 m** | **0.175687 m** | < 0.15 m |
| First crossing > 0.15 m | Step 91 | Step 91 | N/A |
| Crossing count > 0.15 | 1 | 1 | == 0 |

**Conclusion**: E1 and D2 have **identical** support position error profiles when using the correct metric. The E1 integral has no measurable effect on the official support metric.

## Phase 3 Decision

**Classification**: `E1_ANALYZER_USED_WRONG_SUPPORT_METRIC`

**Action Required**: Fix the E1 analyzer to use `support_position_error_m` column.
