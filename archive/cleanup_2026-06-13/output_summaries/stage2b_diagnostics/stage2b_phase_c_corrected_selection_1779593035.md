# Stage 2B Phase C: Corrected Configuration Selection

**Date:** 2026-05-24 10:23:55

## Selection Algorithm Correction

**Issue:** The initial Phase C report (stage2b_phase_c_config_sweep_1779591562.md) recommended `-empirical scale=0.25` based on incorrect selection priority that favored lowest scale/torque over physical stability.

**Problem:** This contradicted Phase B validation where:
- `+empirical` passed with 0.0mm CoM drop and stable contact
- `-empirical` FAILED at step 3 with contact loss

**Root cause:** Selection algorithm used `sort(key=lambda r: (r['scale'], r['mean_total_torque']))`, prioritizing torque minimization over CoM stability.

**Fix:** Corrected selection priority:
1. CoM drop (minimize)
2. Max roll (minimize)
3. Max pitch (minimize)
4. Mean saturation (minimize)
5. Mean torque (minimize)
6. Scale (tie-breaker only)

## Corrected Best Configuration

[SUCCESS] **Physically correct best configuration:**

- **Sign:** +empirical
- **Scale:** 0.5
- **Joint group:** knee
- **Ramp mode:** instant
- **Survival:** 500/500 steps (extended validation)
- **CoM drop:** 0.0mm
- **Max roll:** 0.78°
- **Max pitch:** 0.04°
- **Mean saturation:** 0.0%
- **Mean torque:** 7.9 Nm

**Validation status:** Confirmed stable for 500 steps with perfect CoM stability.

## Comparison: Corrected vs Incorrect Selection

| Metric | Corrected (+empirical 0.5 knee) | Incorrect (-empirical 0.25 hip_pitch_knee) |
|--------|----------------------------------|---------------------------------------------|
| CoM drop | 0.0mm | 11.4mm |
| Max roll | 0.78° | 2.3° |
| Sign validity | Validated in Phase B | Failed in Phase B |
| Physical stability | Perfect | Degraded |

**Conclusion:** The corrected selection prioritizes physical stability and matches Phase B validation, while the incorrect selection prioritized torque minimization and contradicted Phase B.

## Implementation Parameters

```python
# StaticFeedforwardController defaults
FEEDFORWARD_SIGN = 'positive'  # +empirical
FEEDFORWARD_SCALE = 0.5
FEEDFORWARD_JOINT_GROUP = 'knee'  # indices [3, 8]
FEEDFORWARD_RAMP_MODE = 'instant'
```

**Empirical feedforward torques (from gain sweep telemetry):**
- Hip pitch L/R: 4.1, 3.2 Nm
- Knee L/R: -15.5, -15.8 Nm
- Applied to knee joints only with scale=0.5
- Effective knee feedforward: -7.75, -7.90 Nm

## Next Steps

