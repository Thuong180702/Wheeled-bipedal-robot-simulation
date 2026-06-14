# T6F Sign Fix Pitch Suppression Activation Audit

**Date**: 2026-06-12
**Task**: Phase B - Investigate why pitch suppression was 0.0%

## Classification

**PITCH_SUPPRESSION_BUG_CONDITION_TRUE_BUT_NOT_ACTIVE**

## Recommendation

**FIX_PITCH_SUPPRESSION_IMPLEMENTATION**

## Summary

Condition was true 166 times (arch_fix active AND error > 0.10m) but pitch suppression was never activated. This indicates a bug in the pitch suppression implementation - likely wrong variable, placement before arch_fix_active is computed, or overwritten later.

## Condition Analysis

- Steps where `arch_fix_active == True`: 169 (33.9%)
- Steps where `abs(error) > 0.10m`: 194 (38.9%)
- Steps where **BOTH conditions true**: 166 (33.3%)
- Steps where `sign_fix_pitch_suppressed == True`: 0 (0.0%)

## Error Distribution During arch_fix

- **min**: 0.0976 m
- **max**: 0.1916 m
- **mean**: 0.1552 m
- **median**: 0.1614 m
- **p95**: 0.1887 m
- **p99**: 0.1908 m

### Error Histogram During arch_fix

- **0-0.05 m**: 0 steps (0.0%)
- **0.05-0.08 m**: 0 steps (0.0%)
- **0.08-0.10 m**: 3 steps (1.8%)
- **0.10-0.15 m**: 65 steps (38.5%)
- **0.15-0.20 m**: 101 steps (59.8%)
- **0.20-0.30 m**: 0 steps (0.0%)
- **>0.30 m**: 0 steps (0.0%)

## Conclusion

Condition was true 166 times (arch_fix active AND error > 0.10m) but pitch suppression was never activated. This indicates a bug in the pitch suppression implementation - likely wrong variable, placement before arch_fix_active is computed, or overwritten later.

**Next Step**: FIX_PITCH_SUPPRESSION_IMPLEMENTATION