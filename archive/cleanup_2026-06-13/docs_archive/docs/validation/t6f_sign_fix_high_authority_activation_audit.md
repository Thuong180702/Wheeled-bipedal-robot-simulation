# T6F Sign Fix High Authority Activation Audit

**Date**: 2026-06-12
**Task**: Phase C - Investigate why high authority (>4.0 Nm) was rare

## Classification

**HIGH_AUTHORITY_RARE_BECAUSE_500_STEP_WINDOW_TOO_EARLY**

## Summary

Only 8 steps with high authority during 500-step window. This appears to be insufficient sampling - a longer window would capture more.

## High Authority Analysis

- Steps with |final_tau| > 4.0 Nm: 8 (1.6%)
- Max transmitted torque: 6.12 Nm
- Mean transmitted torque: 0.47 Nm
- First high authority step: 113

## Arch Fix Activation

- Steps with arch_fix_active: 169 (33.9%)

## Position Torque Demand

- tau_position max: 7.00 Nm
- tau_position mean: 3.02 Nm
- Steps with |tau_position| > 4.0 Nm: 169 (33.9%)
- Steps with |tau_position| > 6.0 Nm: 102 (20.4%)

## Conclusion

Only 8 steps with high authority during 500-step window. This appears to be insufficient sampling - a longer window would capture more.