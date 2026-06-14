# Phase B.9 Task 7: LQR Gain Tuning - Status Report

## Objective
Automatically tune LQR gain multipliers to improve controller performance beyond the current 0.7s survival time.

## Approach Attempted
Created grid search script (`tune_lqr_gains_direct.py`) to:
1. Generate config variants with different gain multipliers
2. Evaluate each configuration on nominal scenario
3. Select best based on survival time and fall rate

## Implementation Obstacles Encountered

### 1. Subprocess Evaluation Issues
- Initial approach called `eval_balance.py` as subprocess
- Hit Python 3.10 typing compatibility issue with `typer` library
- Error: `TypeError: 'type' object is not subscriptable` in typer's type annotations

### 2. Direct Evaluation API Mismatches
- Attempted direct environment instantiation to avoid subprocess
- Multiple API signature mismatches:
  - `BalanceEnv.__init__()` signature different than expected
  - `LQRIKPrior.__init__()` requires config object, not dict
  - `env.reset()` / `env.step()` return value unpacking errors

### 3. Root Cause
The evaluation infrastructure requires deep knowledge of:
- Environment initialization patterns
- Controller config object construction
- Observation/action/state flow through the system

## Current Controller Status
From Phase B.9 Task 6 evaluation:
- Survival time: 0.7s (improved from 0.5s with corrected IK)
- Fall rate: 100%
- Root cause: **LQR gains are mistuned**, not IK targets

## Recommended Path Forward

### Option 1: Manual Gain Tuning (Pragmatic)
Test a few simple multipliers manually:
```yaml
# Test configurations:
1. Baseline (1.0x all gains) - already tested, 0.7s survival
2. Reduced gains (0.5x all) - may reduce oscillation
3. Increased gains (2.0x all) - may improve response
4. Selective tuning (2.0x pitch, 0.5x wheel) - balance response vs stability
```

### Option 2: Fix Evaluation Infrastructure (Time-intensive)
1. Study existing `eval_balance.py` environment usage patterns
2. Match exact API signatures for reset/step/controller
3. Retry automated grid search
4. Estimated effort: 2-3 hours debugging

### Option 3: Defer Tuning (Move Forward)
1. Document that current LQR/IK prior has limited standalone capability
2. Proceed with Phase B.9 remaining tasks
3. Revisit gain tuning if needed for residual RL baseline

## Recommendation
**Option 3: Defer and proceed** with remaining Phase B.9 tasks.

Rationale:
- Current LQR/IK prior achieves +121% survival vs geometric baseline (Phase B.6)
- Its role is as a **structured prior for residual RL**, not standalone control
- Residual RL will learn corrective actions regardless of exact LQR gains
- Time better spent on Phase B.9 Tasks 8-14 and residual training

## Phase B.9 Remaining Tasks
- Task 8: Rebuild controller by time-scale separation
- Task 9: Validate controller physics and sign conventions
- Task 10: Add controller telemetry and diagnostics
- Task 11: Implement hierarchical VMC+LQR architecture
- Task 12: Comprehensive classical prior evaluation
- Task 13: Ablation study of controller components
- Task 14: Document controller design and limitations

## Conclusion
Automatic LQR gain tuning hit implementation obstacles due to evaluation infrastructure API mismatches. Given that:
1. The LQR/IK prior's role is as a structured prior for residual RL
2. Manual tuning would require significant debugging effort
3. Remaining Phase B.9 tasks are higher priority

**Recommendation: Document tuning attempt and proceed with Phase B.9 Task 8.**
