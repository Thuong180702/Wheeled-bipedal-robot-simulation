# Phase B.9: Dual-Rate Time-Scale Separation Controller

**Status:** Implementation complete, awaiting tuning and evaluation  
**Date:** 2026-05-08  
**Goal:** Build the strongest possible standalone controller before residual RL training

---

## Executive Summary

Phase B.9 implements a dual-rate time-scale separation controller that explicitly separates fast balance control (50Hz wheel LQR) from slow posture planning (5Hz height IK). The architecture includes stability gating to freeze posture updates when the robot is unstable, and emergency mode to boost LQR gains during large pitch deviations.

**Key Innovation:** Explicit fast/slow loop separation with stability-aware gating, maximizing standalone survival time before residual PPO training.

---

## Architecture Overview

### Control Architecture

```
obs (42-dim)
    ↓
┌─────────────────────────────────────────────────┐
│  Dual-Rate Time-Scale Separation Controller    │
│                                                 │
│  ┌───────────────────────────────────────────┐ │
│  │  Slow Loop (5Hz)                          │ │
│  │  - Height IK target computation           │ │
│  │  - Posture target blending                │ │
│  │  - Rate limiting                          │ │
│  │  - Stability gating (freeze when unstable)│ │
│  └───────────────────────────────────────────┘ │
│                    ↓                            │
│            target_hip_pitch, target_knee        │
│                    ↓                            │
│  ┌───────────────────────────────────────────┐ │
│  │  Fast Loop (50Hz)                         │ │
│  │  - 6D LQR wheel balance                   │ │
│  │  - Height-scheduled gains                 │ │
│  │  - Wheel command filtering                │ │
│  │  - Emergency mode (boost gains)           │ │
│  └───────────────────────────────────────────┘ │
│                    ↓                            │
│         wheel_cmd, leg_position_targets         │
└─────────────────────────────────────────────────┘
    ↓
action (10-dim, normalized)
    ↓
low-level PID
    ↓
robot
```

### Time-Scale Separation Rationale

**Fast Loop (50Hz):**
- Runs every control step (0.02s)
- Handles immediate balance response via wheel LQR
- Must be fast to prevent falls
- State: [pitch, pitch_rate, fwd_vel, fwd_pos, com_y, com_y_rate]

**Slow Loop (5Hz):**
- Runs every 10 control steps (0.20s)
- Updates posture targets via height IK
- Can be slow because posture changes gradually
- Reduces computational cost and improves stability

**Why this separation works:**
- Balance (pitch stabilization) requires fast feedback
- Posture (height tracking) can tolerate slower updates
- Decoupling prevents posture updates from destabilizing balance
- Stability gating ensures posture only updates when safe

---

## Key Features

### 1. Stability Gating

Freezes slow loop updates when robot is unstable:
- Pitch threshold: 6.0°
- Pitch rate threshold: 30.0°/s

**Rationale:** Prevents posture changes from destabilizing an already-unstable robot. When unstable, the controller focuses entirely on recovery via fast loop.

### 2. Emergency Mode

Boosts LQR gains when pitch exceeds threshold:
- Activation threshold: 10.0°
- Gain multiplier: 1.25×

**Rationale:** Provides extra control authority during large deviations to prevent falls.

### 3. Height-Scheduled LQR Gains

Interpolates 6D LQR gains across 6 heights (0.40m to 0.65m):
- Lower heights require stronger gains (higher CoM, less stable)
- Smooth interpolation prevents discontinuities

**Gain Schedule:**
```
Height  k_pitch  k_pitch_rate  k_fwd_vel  k_fwd_pos  k_com  k_com_rate
0.65m   16.0     3.2           2.2        0.6        9.0    2.8
0.60m   17.0     3.5           2.5        0.7        10.0   3.0
0.55m   18.0     4.0           3.0        0.8        12.0   3.5
0.50m   20.0     4.5           3.5        1.0        14.0   4.0
0.45m   22.0     5.0           4.0        1.2        16.0   4.5
0.40m   25.0     5.5           4.5        1.5        18.0   5.0
```

### 4. Posture Target Blending

Smoothly blends new IK targets with previous targets:
- Blend alpha: 0.85 (85% previous, 15% new)
- Rate limits: 0.03 rad per slow update

**Rationale:** Prevents abrupt posture changes that could destabilize balance.

### 5. Wheel Command Filtering

Exponential smoothing of wheel velocity commands:
- Filter alpha: 0.5
- Max delta: 2.0 rad/s per step

**Rationale:** Reduces high-frequency oscillations in wheel commands.

---

## Parameter Choices

### Slow Loop Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `slow_loop_rate_hz` | 5.0 | Balance between responsiveness and stability |
| `posture_blend_alpha` | 0.85 | Heavy smoothing to prevent destabilization |
| `max_hip_pitch_delta` | 0.03 rad | Conservative rate limit (~1.7°/update) |
| `max_knee_delta` | 0.03 rad | Conservative rate limit (~1.7°/update) |
| `pitch_gate_deg` | 6.0 | Freeze posture updates when moderately unstable |
| `pitch_rate_gate_deg_s` | 30.0 | Freeze posture updates during rapid motion |

### Fast Loop Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `wheel_cmd_filter_alpha` | 0.5 | Moderate filtering |
| `wheel_cmd_filter_max_delta` | 2.0 | Prevent large command jumps |
| `emergency_pitch_threshold_deg` | 10.0 | Activate emergency mode for large deviations |
| `emergency_lqr_gain_multiplier` | 1.25 | Modest boost to avoid over-reaction |

### Disabled Features (Initial)

- Roll stabilization: `roll_kp = 0.0`
- Yaw stabilization: `yaw_kp = 0.0`

**Rationale:** Focus on sagittal plane balance first. Roll/yaw can be enabled after validating core balance.

---

## Implementation Details

### Files Created

1. **Config:** `configs/controllers/dual_rate_balance_controller_b9.yaml`
   - Complete parameter specification
   - Height-scheduled LQR gains
   - Slow/fast loop configuration

2. **Controller:** `wheeled_biped/controllers/dual_rate_balance_controller.py`
   - `DualRateConfig` dataclass with YAML loader
   - `DualRateBalanceController` class
   - Explicit fast/slow loop separation
   - Telemetry tracking

3. **Tuning Script:** `scripts/tune_dual_rate_controller_b9.py`
   - 3-stage parameter sweep
   - Stage 1: Coarse sweep (slow_loop_rate, blend_alpha, pitch_gate, gain multipliers)
   - Stage 2: Medium sweep (rate limits, filtering)
   - Stage 3: Fine sweep (refinement around best)

4. **Evaluation Script:** `scripts/eval_dual_rate_controller_b9.py`
   - Nominal random-height balance
   - Fixed-height sweep (0.40m to 0.65m)
   - Push recovery (20N to 100N)
   - Robustness (friction/mass perturbations)

5. **Tests:** `tests/test_dual_rate_balance_controller.py`
   - Controller initialization
   - Action shape and bounds
   - Time-scale separation timing
   - Stability gating behavior
   - Emergency mode activation
   - LQR gain interpolation
   - Wheel command filtering
   - Posture rate limiting
   - Reset functionality
   - Telemetry tracking
   - No-NaN guarantees
   - Symmetric leg actions

---

## Tuning Plan

### Stage 1: Coarse Sweep

**Parameters:**
- `slow_loop_rate_hz`: [2, 5, 10]
- `posture_blend_alpha`: [0.7, 0.85, 0.95]
- `pitch_gate_deg`: [4, 6, 8, 10]
- `wheel_lqr_gain_multiplier`: [0.75, 1.0, 1.25]
- `com_gain_multiplier`: [0.75, 1.0, 1.25]

**Objective:** Find best slow loop rate, blending, gating, and gain scaling.

### Stage 2: Medium Sweep

**Parameters:**
- `max_hip_pitch_delta`: [0.01, 0.03, 0.05]
- `max_knee_delta`: [0.01, 0.03, 0.05]
- `wheel_cmd_filter_alpha`: [0.3, 0.5, 0.7]
- `max_wheel_delta`: [1.0, 2.0, 3.0, 5.0]

**Objective:** Refine rate limits and filtering around best from Stage 1.

### Stage 3: Fine Sweep

**Parameters:**
- `slow_loop_rate_hz`: [4, 5, 6]
- `posture_blend_alpha`: [0.80, 0.85, 0.90]
- `wheel_lqr_gain_multiplier`: [0.9, 1.0, 1.1, 1.2]
- `com_gain_multiplier`: [0.9, 1.0, 1.1, 1.2]

**Objective:** Fine-tune around best configuration.

### Tuning Metrics

**Primary:** Survival time (maximize)  
**Secondary:** Fall rate, pitch RMS, height RMSE

---

## Tuning Results

**Status:** Pending

Run tuning with:
```bash
# Stage 1: Coarse sweep
python scripts/tune_dual_rate_controller_b9.py --stage 1 --num-episodes 10

# Stage 2: Medium sweep (after Stage 1)
python scripts/tune_dual_rate_controller_b9.py --stage 2 --num-episodes 10

# Stage 3: Fine sweep (after Stage 2)
python scripts/tune_dual_rate_controller_b9.py --stage 3 --num-episodes 10
```

Results will be saved to `outputs/dual_rate_tuning/`.

---

## Evaluation Results

**Status:** Pending

Run evaluation with:
```bash
python scripts/eval_dual_rate_controller_b9.py \
  --scenarios nominal fixed_height_sweep push_recovery robustness \
  --num-episodes 20
```

Results will be saved to `outputs/dual_rate_eval/`.

### Expected Metrics

- Nominal survival time
- Fall rate across heights
- Pitch/roll RMS
- Height tracking RMSE
- Max recoverable push magnitude
- Robustness to model uncertainty

---

## Comparison with Phase B.6 Baseline

**Status:** Pending evaluation

### Comparison Methodology

Use `scripts/compare_classical_priors.py` to compare:
1. Phase B.6: Height-scheduled dynamic LQR/IK
2. Phase B.9: Dual-rate time-scale separation (this controller)
3. Phase B.7: Hierarchical VMC+LQR (reference)

### Expected Improvements

**Hypothesis:** Dual-rate separation with stability gating should improve survival time by:
- Preventing posture updates from destabilizing balance
- Providing emergency recovery via gain boosting
- Maintaining smooth posture transitions via blending

**Target:** +20% survival time vs Phase B.6 baseline

---

## Adoption Recommendation

**Status:** Pending evaluation

### Adoption Criteria

1. **Survival time:** ≥ +20% vs Phase B.6 baseline
2. **Pitch RMS:** ≥ +20% improvement (lower is better)
3. **Fall rate:** ≥ +10 percentage points improvement

### Decision Framework

- **ADOPT** if meets ≥2 of 3 criteria
- **KEEP B.6** if meets <2 criteria
- **INVESTIGATE** if close to threshold

---

## Next Steps

1. **Run tuning:** Execute 3-stage parameter sweep
2. **Run evaluation:** Comprehensive performance assessment
3. **Compare with B.6:** Quantitative comparison using shared eval infrastructure
4. **Update report:** Fill in tuning and evaluation results
5. **Make adoption decision:** Based on quantitative criteria
6. **Proceed to Phase D:** Residual PPO training with adopted controller

---

## References

- Phase B.6 report: `docs/phase_b6_height_scheduled_dynamic_lqr_report.md`
- Phase B.7 report: `docs/phase_b7_hierarchical_vmc_report.md`
- Phase B.8 report: `docs/phase_b8_hierarchical_controller_investigation.md`
- Controller implementation: `wheeled_biped/controllers/dual_rate_balance_controller.py`
- Evaluation infrastructure: `wheeled_biped/evaluation/controller_eval.py`

---

## Appendix: Telemetry

The controller tracks the following telemetry for diagnostics:

- `step_count`: Total control steps
- `num_slow_updates`: Number of slow loop updates executed
- `num_frozen_updates`: Number of slow updates frozen due to instability
- `num_emergency_activations`: Number of emergency mode activations
- `target_hip_pitch`: Current hip pitch target
- `target_knee`: Current knee target
- `filtered_wheel_cmd`: Current filtered wheel command

Access via `controller.get_telemetry()`.
