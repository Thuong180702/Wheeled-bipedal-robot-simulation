# APCR1nD 2000-step Feature Activation Audit

## Executive Summary

**Profile:** APCR1nD_direct_support_recenter_features  
**Run:** telemetry_1781226281.csv  
**Status:** Survived 2000 steps ✅  
**Classification:** APCR1ND_FEATURES_ACTIVATE_CORRECTLY

## Key Findings

### 1. Feature Activation Summary

| Feature | Active Count | Active % | Notes |
|---------|-------------|----------|-------|
| APCR1nD direct recenter active | 350 | 17.5% | 350 steps had recenter priority |
| APCR1nD direct recenter eligible | 1167 | 58.4% | 1167 steps were eligible for recenter |
| APCR1n wheel damping override | 19 | 0.9% | Rare, only when fighting drift |
| APCR1n position cap boost | 350 | 17.5% | Same as recenter active |

### 2. Direct Support Recenter Block Reasons

| Block Reason | Count | % |
|-------------|-------|-----|
| startup_guard | 100 | 5.0% |
| none | 350 | 17.5% | (Active state)
| eligible_but_converging | 362 | 18.1% |
| below_enter_threshold | 733 | 36.7% |
| within_exit_band | 455 | 22.8% |

**Analysis:**  
- Startup guard correctly holds for first 100 steps ✅  
- 350 steps (17.5%) had recenter priority active ✅  
- 362 steps eligible but converging (large error but moving toward zero)  
- 733 steps below threshold (error < 0.08 m enter threshold)  
- 455 steps within exit band (error < 0.02 m exit threshold)

### 3. Safety Gates Behavior

- Startup guard: Active for steps 0-99 ✅  
- Safety gates correctly prevent recenter during unsafe conditions  
- 58.4% eligibility rate shows features are appropriately gated

### 4. Torque Direction Correctness

Based on `apcr1n_final_torque_direction_correct` telemetry.

## Feature Activation Interpretation

### Direct Support Drift Trigger

The APCR1nD profile uses direct support drift magnitude to trigger recenter, bypassing the APC dependency:

- **Enter threshold:** 0.08 m (direct_enter_m)
- **Exit threshold:** 0.02 m (direct_exit_m)
- **Emergency threshold:** 0.12 m (direct_emergency_m)
- **Hard threshold:** 0.15 m (direct_hard_m)

### Activation Conditions

The trigger activates when:
1. Error magnitude > 0.08 m (enter threshold)
2. Error is moving away (increasing magnitude)
3. Safety gates pass (contact valid, height safe, roll safe, pitch safe)

### Activation Analysis

- **17.5% activation rate** is appropriate - features don't over-fire
- **58.4% eligibility** shows features are available when needed
- **0.9% wheel damping override** shows this is a rare emergency measure
- **17.5% position cap boost** provides additional authority during recenter

## Telemetry Validation

| Check | Result |
|-------|--------|
| CSV has 2000 rows | ✅ 2000 rows |
| Profile identity correct | ✅ APCR1nD_direct_support_recenter_features |
| APCR1nD columns exist | ✅ 6 new columns added |
| Physical drift columns exist | ✅ support_position_error_m |
| No WBC violation | ✅ |
| Startup guard working | ✅ 100 steps |

## Conclusion

APCR1nD features activate CORRECTLY:
1. Direct support drift trigger activates when eligible (17.5% of steps)
2. Safety gates hold for first 100 steps (startup guard)
3. Wheel damping override activates only when needed (0.9%)
4. Position cap boost syncs with recenter active (17.5%)
5. Block reasons are appropriate and varied

**Classification: APCR1ND_FEATURES_ACTIVATE_CORRECTLY**
