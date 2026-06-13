# Hip-Yaw Divergence Fix Plan

**Date:** 2026-06-05
**Based on:** `hip_yaw_divergence_after_sign_fix_audit.md`
**Decision:** `DIVERGENCE_ROOT_CAUSE_IDENTIFIED_READY_FOR_FIX`

## Root Cause Summary

Per-joint PD control produces torques that **accelerate** divergence 97-99% of the time. The sign fix corrected per-joint torque direction but exposed this fundamental limitation.

## Fix Approach

**Enable HY2-DIV (Hip-Yaw Divergence Damping)** - A dedicated antisymmetric torque layer that applies torque proportional to left/right error difference.

## Why HY2-DIV

The existing code already has HY2-DIV infrastructure:

```python
# In shape_posture_controller.py:
@dataclass(frozen=True)
class HipYawDivergenceProfile:
    k_divergence: float
    k_divergence_rate: float
    tau_max_divergence: float
    z_low: float = 0.300
    z_high: float = 0.393
```

HY2-DIV applies:
```
tau_div_L = -k_div * (error_L - error_R) - k_div_rate * (vel_L - vel_R)
tau_div_R = +k_div * (error_L - error_R) + k_div_rate * (vel_L - vel_R)
```

This is **antisymmetric** torque that **opposes** divergence, unlike per-joint PD which is symmetric and **drives** divergence.

## Implementation

### Step 1: Verify HY2-DIV Code Exists

The code in `shape_posture_controller.py` already has HY2-DIV implementation (lines 218-244). Just need to enable it via config.

### Step 2: Create Evaluation Script

Create `scripts/evaluate_hip_yaw_divergence_fix.py` that:
- Runs simulations with HY2-DIV enabled
- Compares divergence metrics vs post-fix baseline
- Validates at 100, 500, 5000 steps

### Step 3: Run 100-Step Smoke Test

```bash
python scripts/evaluate_hip_yaw_divergence_fix.py --steps 100 --heights nominal low_0p300 high_0p480
```

### Step 4: Run 5000-Step Evaluation

```bash
python scripts/evaluate_hip_yaw_divergence_fix.py --steps 5000 --heights nominal low_0p300 high_0p480
```

## Validation Gates

| Gate | Metric | Threshold | Status |
|------|--------|-----------|--------|
| 1 | Divergence RMS nominal | < 0.10 rad | Target |
| 2 | Divergence RMS low_0p300 | < 0.30 rad | Target |
| 3 | Divergence RMS high_0p480 | < 0.25 rad | Target |
| 4 | Sign Correct L | > 90% | Must maintain |
| 5 | Sign Correct R | > 95% | Must maintain |
| 6 | Survival rate | >= 5000 steps | Must maintain |

## Expected Results

| Height | Pre-Fix div RMS | Post-Sign-Fix div RMS | HY2-DIV Target |
|--------|-----------------|----------------------|-----------------|
| nominal | 0.0447 rad | 0.2446 rad | < 0.10 rad |
| low_0p300 | 0.3575 rad | 0.3690 rad | < 0.30 rad |
| high_0p480 | 0.2825 rad | 0.3399 rad | < 0.25 rad |

## Rollback Rule

If any validation gate fails after HY2-DIV is enabled:
1. Disable HY2-DIV
2. Revert to post-sign-fix baseline
3. Report which gate failed
4. Do NOT tune gains blindly

## Do Not Touch

- WBC paths
- Hip-roll controller
- Sagittal controller
- Support-position controller
- Per-joint PD gains
- Step C
- Step D

## Files to Create/Modify

1. `scripts/evaluate_hip_yaw_divergence_fix.py` - New evaluation script
2. `wheeled_biped/controllers/shape_posture_controller.py` - Enable HY2-DIV via config
3. `scripts/simulate_hierarchical_controller.py` - Add HY2-DIV config option

## Conservative Gain Selection

Start with conservative gains:
```python
HY2_DIV_BASELINE = HipYawDivergenceProfile(
    name="hy2_div_baseline",
    k_divergence=5.0,
    k_divergence_rate=1.0,
    tau_max_divergence=0.5,
)
```

This provides:
- Proportional damping on divergence error
- Derivative damping on divergence velocity
- Limited torque authority (0.5 Nm max)
- Height-gated activation (only at low heights)
