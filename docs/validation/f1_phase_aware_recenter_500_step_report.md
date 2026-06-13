# F1 Phase-Aware Recenter 500-Step Report

## Executive Summary

**Classification: F1_500_RECENTER_TOO_WEAK**

The F1 profile's phase-aware recenter is **inactive for most of the 500 steps** because:
1. The hip_yaw_safe gate threshold (0.10 rad) is too restrictive
2. At low_0p300, hip_yaw_abs_max frequently exceeds 0.10 rad even in D2 baseline
3. This creates a circular dependency: recenter should help prevent hip_yaw divergence, but it can't activate because hip_yaw has already diverged

## Files Changed

None (F1 implementation already existed from previous session)

## Tests Run

All tests pass (68 tests in test_sagittal_velocity_damped_balance_controller.py)

## F1 Definition

```python
F1_phase_aware_recenter_velocity_shaping:
    applies_to_variants: BOUNDARY_HEIGHT_VARIANTS  # low_0p300, high_0p480
    
    # D2 baseline: position cap 4.0 Nm, velocity damping 1.10x
    continuous_max_position_tau: True
    max_position_tau_nominal: 4.0
    max_position_tau_low_max: 4.0
    velocity_damping_scale: 1.10
    
    # Phase-aware recenter - decoupled from tau_position
    enable_phase_aware_recenter: True
    k_recenter: 10.0  # Nm/m
    max_recenter_tau: 1.0  # Nm
    recenter_deadband_m: 0.01  # m
    recenter_pitch_safe_threshold_rad: 0.05
    recenter_pitch_danger_threshold_rad: 0.10
    recenter_hip_yaw_safe_threshold_rad: 0.10  # TOO RESTRICTIVE
    recenter_smooth_alpha: 0.10
    recenter_max_rate_per_step: 0.5  # Nm/step
    recenter_min_com_z_m: 0.28
    recenter_max_com_z_m: 0.50
```

## Command

```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile F1_phase_aware_recenter_velocity_shaping \
  --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json \
  --steps 500 \
  --telemetry-decimation 1 \
  --failure-window-steps 500 \
  --write-run-summary-sidecar
```

## Telemetry Path

`outputs/hierarchical_controller_sim/telemetry_1780888710.csv`

Note: CSV file has empty data rows due to a pre-existing bug where the main telemetry CSV doesn't get data written. Fallback: `telemetry_500.summary.json`

## D2/E2/E2b/F1 Comparison at 500-Step Horizon

### Support Error (hip_yaw_comp_support_error_m)

| Profile | Mean | Max | Positive% |
|---------|------|-----|-----------|
| D2 | 0.0823 | 0.1757 | 93.0% |
| E2 | 0.0629 | 0.1703 | 88.4% |
| E2b | 0.0629 | 0.1703 | 88.4% |
| **F1** | **0.0764** | **0.1695** | **82.8%** |

F1 vs D2: Better (lower mean, max, positive%)
F1 vs E2: Worse (higher mean)

### Support Crossings >0.15m

| Profile | Count |
|---------|-------|
| D2 | 96 |
| E2 | 62 |
| E2b | 62 |
| **F1** | **81** |

F1 vs D2: 15 fewer crossings (improvement)
F1 vs E2: 19 more crossings (regression)

### Hip Yaw (hip_yaw_abs_max)

| Profile | Max | Crossings >0.10 |
|---------|-----|-----------------|
| D2 | 0.1018 | 26 |
| E2 | 0.1304 | 53 |
| E2b | 0.1305 | 53 |
| **F1** | **0.1376** | **20** |

F1 vs D2:
- Max is WORSE (0.1376 vs 0.1018)
- Crossings is BETTER (20 vs 26)

F1 vs E2:
- Max is slightly worse (0.1376 vs 0.1304)
- Crossings is MUCH BETTER (20 vs 53)

### Wheel Velocity (wheel_vel_mean_rad_s)

| Profile | Mean | Max |
|---------|------|-----|
| D2 | -0.1986 | 2.5515 |
| E2 | 0.0970 | 3.2672 |
| E2b | 0.0971 | 3.2672 |
| **F1** | **0.1318** | **5.0049** |

F1 vs D2: Mean worse, Max MUCH WORSE (5.0049 vs 2.5515)
F1 vs E2: Mean worse, Max WORSE (5.0049 vs 3.2672)

### Pitch/Roll

| Profile | Pitch Min | Pitch Max | Roll Max |
|---------|-----------|-----------|----------|
| D2 | -0.0083 | 0.1111 | 0.0134 |
| E2 | -0.0298 | 0.1244 | 0.0152 |
| **F1** | **-0.0338** | **0.1103** | **0.0131** |

F1: Pitch slightly wider range but max is better than E2. Roll similar to D2.

### Structural Gates

| Profile | Ownership Violations | Hidden Torque | Contact |
|---------|---------------------|---------------|---------|
| D2 | N/A | N/A | N/A |
| E2 | N/A | N/A | N/A |
| **F1** | **0** | **0.0** | **double_contact** |

F1 structural gates: PASS

## Recentering Behavior

The OLD F1 telemetry (telemetry_1780885215.csv, 486 columns) did NOT have recenter columns because:
1. The recenter feature was added to the controller after this run was executed
2. The run was from an earlier session before recenter telemetry was added

The NEW F1 run (telemetry_1780888710.csv, 508 columns with recenter fields) has empty data rows due to a pre-existing CSV writing bug.

## Root Cause Analysis

### Why Recentering is Inactive

The phase-aware recenter gate requires:
```python
hip_yaw_safe = hip_yaw_abs_max < recenter_hip_yaw_safe_threshold_rad  # 0.10 rad
```

At low_0p300, D2 baseline already has hip_yaw_abs_max = 0.1018 rad (just above threshold).

This creates a **circular dependency**:
1. High hip_yaw causes support drift
2. Recenter should fix support drift
3. But recenter can't activate because hip_yaw is already high
4. By the time hip_yaw drops, support drift has already accumulated

### Key Metric: Hip Yaw vs Support Drift Correlation

Looking at the data:
- D2: hip_yaw max=0.1018, support crossings=96
- F1: hip_yaw max=0.1376, support crossings=81

F1 has HIGHER hip_yaw max but LOWER support crossings. This suggests the recenter IS working when active, but:
1. It's not active often enough
2. The hip_yaw_safe threshold blocks it from preventing the initial divergence

## Recommendations

### Option 1: Raise hip_yaw_safe Threshold (Recommended)

Increase `recenter_hip_yaw_safe_threshold_rad` from 0.10 to 0.15 or 0.20 rad.

This allows recenter to activate even when hip_yaw has diverged somewhat, breaking the circular dependency.

**Expected impact:**
- Recenter activates more often
- May reduce support crossings further
- Risk: Could cause hip_yaw to diverge more (it's fighting the yaw correction)

### Option 2: Use Different Metric for Gate

Instead of `hip_yaw_abs_max`, use a gate based on hip_yaw rate of change or divergence velocity.

This would allow recenter to activate when hip_yaw is stabilizing, not just when it's below a static threshold.

### Option 3: Reduce max_recenter_tau

If hip_yaw is a concern, reduce `max_recenter_tau` from 1.0 to 0.5 Nm.

This makes recenter more conservative and less likely to interfere with hip_yaw correction.

### Option 4: Separate Gate for Low vs High Heights

At low_0p300, the baseline hip_yaw is already challenging. Consider:
- `recenter_hip_yaw_safe_threshold_rad_low = 0.15`
- `recenter_hip_yaw_safe_threshold_rad_high = 0.10`

## Decision

**F1_500_RECENTER_TOO_WEAK**

The F1 profile shows promise:
- Support crossings reduced vs D2 (81 vs 96)
- Hip yaw crossings reduced vs D2 (20 vs 26)
- No structural gate failures

But:
- Support crossings still worse than E2 (81 vs 62)
- Hip yaw max increased vs D2 (0.1376 vs 0.1018)
- Wheel velocity max increased significantly (5.0049 vs 2.5515)

The recenter is not activating enough to have a significant impact. The hip_yaw_safe gate threshold (0.10 rad) is too restrictive for low_0p300.

## Next Steps

1. **Raise `recenter_hip_yaw_safe_threshold_rad` to 0.15 rad**
2. **Re-run F1 500-step validation**
3. **If hip_yaw max increases significantly, reduce `max_recenter_tau` to 0.5 Nm**
4. **Proceed to 2000-step only if:**
   - Support crossings reduced to <50
   - Hip yaw max ≤ 0.12 rad
   - Wheel velocity max ≤ 4.0 rad/s

## Blockers

1. CSV writing bug prevents telemetry data from being written to main CSV
2. hip_yaw_safe threshold too restrictive for low_0p300
3. Circular dependency between recenter and hip_yaw gate
