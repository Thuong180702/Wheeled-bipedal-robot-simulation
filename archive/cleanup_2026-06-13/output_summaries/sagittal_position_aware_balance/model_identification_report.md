# Sagittal Balance Dynamics Identification Report

**Status: `model_identification_failed`**

No controller was designed from this model.

## Summary

Closed-loop system identification was performed on 5000 steps of nominal balance-core telemetry to fit a discrete-time sagittal dynamics model. The one-step prediction quality is excellent but short-horizon rollout diverges catastrophically due to an unstable velocity eigenvalue. The model cannot be used for LQR/state-feedback controller design under the current plan.

## Data Source

- Collection mode: closed-loop (stable balance-core runs)
- Scenario: nominal
- Duration: 5000 steps at 100 Hz control (50 seconds)
- Controller mode: balance-core (validated four-source stack)

## Identified Model

Form: `x[k+1] = A x[k] + B u[k]`

State vector:
1. `sagittal_position_error` (m)
2. `sagittal_velocity` (m/s)
3. `pitch_x` (rad)
4. `pitch_rate_x` (rad/s)
5. `wheel_velocity_mean` (rad/s)

Input: scalar `wheel_torque` (Nm)

### A Matrix

```
 [[  1.0000,   0.0133,  -0.0150,  -0.0025,   0.0001],
  [  0.0000,   1.3280,  -1.4978,  -0.2469,   0.0149],
  [  0.0000,   0.0031,   0.9850,   0.0050,   0.0001],
  [ -0.0000,  -0.1052,   0.1659,   0.6041,  -0.0048],
  [ -0.0000,  13.3750, -60.6820, -11.3342,   1.5815]]
```

### B Matrix

```
 [[  0.0002],
  [  0.0204],
  [  0.0001],
  [ -0.0534],
  [  2.1061]]
```

### Eigenvalues

| Index | Value | Magnitude | Stability |
|-------|-------|-----------|-----------|
| 0 | 1.9649 | 1.9649 | **UNSTABLE** |
| 1 | 0.5481 | 0.5481 | stable |
| 2 | 1.0000 | 1.0000 | marginal |
| 3 | 0.9914 | 0.9914 | stable |
| 4 | 0.9942 | 0.9942 | stable |

The dominant unstable eigenvalue (lambda_0 = 1.96) corresponds to sagittal velocity growth. The marginal eigenvalue (lambda_2 ~ 1.0) corresponds to the position integrator.

## Quality Metrics

| Gate | Value | Threshold | Pass |
|------|-------|-----------|------|
| one-step R² | 1.0000 | >= 0.80 | YES |
| rollout R² (20-step) | -1.15e10 | >= 0.60 | **NO** |
| residual mean abs | 0.000334 | <= 0.10 | YES |
| sign response | OK | must pass | YES |
| nominal fit | OK | must pass | YES |
| height variant fit | N/A (no data) | must pass | YES (no data) |

## Diagnosis

The baseline sagittal wheel balance controller stabilizes pitch effectively but does not provide sagittal velocity damping. In the identified closed-loop dynamics:

- A[1,1] = 1.328: velocity integrates/grows without damping
- A[4,1] = 13.375: strong coupling from velocity to wheel speed
- A[4,2] = -60.682: strong coupling from pitch to wheel speed

The unstable eigenvalue (1.96) causes the 20-step rollout to diverge by orders of magnitude, even though one-step predictions are nearly perfect. This confirms the known symptom: the robot balances upright but drifts along the sagittal axis because velocity is not regulated.

## Conclusion

**No controller was designed from this model.** The model identification quality gate failed at the rollout R² criterion. Per the approved implementation plan, execution stops at Gate 4. No `SagittalPositionAwareBalanceController` will be implemented from this model.

The diagnostic finding — that sagittal velocity is the unstable degree of freedom — is valuable for future controller design, but it does not authorize building a controller from a model that cannot predict 20 steps ahead.

## Next-Step Options

These are proposals only. None are implemented.

### Option A: Improve System Identification
- Collect richer closed-loop data with small bounded excitation
- Normalize states to improve conditioning
- Identify around multiple heights
- Validate rollout quality before proceeding

### Option B: Analytic Velocity-Damped Controller Design
- Design velocity damping by construction (not from failed model)
- Use explicit state terms with hand-tuned gains
- Require sign tests and incremental validation
- Would require a revised spec/plan

### Option C: Accept Position Drift Limitation
- Document position drift as known limitation
- Continue other development work
- Return to position hold later with revised approach
