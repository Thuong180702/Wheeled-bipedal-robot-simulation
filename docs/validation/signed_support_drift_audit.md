# Signed Support Drift Audit

**Date**: 2026-06-08
**Status**: COMPLETE

## Key Findings

### Data Source Identification

The `support_position_error_m` field is **NOT** a pure CoM position error. Investigation confirmed:

```
support_position_error_m = hip_yaw_comp_support_error_m
yaw_aware_sagittal_error_compensated_m = support_position_error_m (identical)
```

**Root Cause**: Hip yaw divergence causes sagittal position error through yaw-position coupling.

### Signed Drift Classification

| Variant | Classification | Bias Ratio | Zero Crossings | % Positive |
|---------|---------------|------------|----------------|------------|
| D2 | POSITIVE_BIASED_STRONG | 40.6 | 13 | 97.5% |
| E2 | POSITIVE_BIASED_STRONG | 7.9 | 4 | 88.4% |
| E2b | POSITIVE_BIASED_STRONG | 7.9 | 4 | 88.4% |

### Metrics Comparison

| Metric | D2 | E2 | E2b | Interpretation |
|--------|-----|-----|-----|----------------|
| support_error_mean (m) | 0.058 | 0.063 | 0.063 | E2 slightly worse |
| support_error_max (m) | 0.176 | 0.170 | 0.170 | Similar |
| crossings >0.15m (norm-500) | 9.6 | 62.0 | 62.0 | E2 6.5× WORSE |
| hip_yaw_abs_max (rad) | 0.313 | 0.130 | 0.130 | E2 2.4× better |
| pitch_forward_pct | 94.6% | 80.4% | 80.4% | E2 recovers more |

## Analysis

### The Coupling Mechanism

```
Position Error → tau_position (position correction)
                    ↓
            Hip roll/yaw coupling → hip_yaw_abs_max increases
                    ↓
            More yaw divergence → More yaw-induced position error
                    ↓
            Position Error increases (feedback loop)
```

### Why E2 Improved hip_yaw but Worsened Position

E2 reduced position correction authority to reduce hip yaw coupling:
- ✓ Reduced hip_yaw_abs_max by 2.4×
- ✗ Position error crossings increased 6.5×
- ✗ Position error mean increased (0.058 → 0.063 m)

The fix addressed the symptom (hip yaw) but not the root cause.

## Conclusion

**SIGNED_DRIFT_CONFIRMED**: Support drift is biased to one side (97.5% positive for D2). The root cause is yaw-induced position error, not pure CoM drift. Phase-aware recentering is recommended.