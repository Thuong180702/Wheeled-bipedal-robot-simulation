# Pitch-Safe Joint Sagittal-Yaw Candidate Design

**Date:** 2026-06-05  
**Status:** DESIGN  
**Classification:** PITCH_SAFE_CANDIDATE_FAMILY

## Executive Summary

Designing pitch-safe candidates (J2a-J2d family) to pass all three gates at low_0p300:
- support_position_error ≤ 0.15 m
- hip_yaw_abs_max ≤ 0.07 rad
- pitch_x_max_abs ≤ 0.10 rad

**Strategy:** Reduce position authority (k_position, max_position_tau) from J1-J3 levels while maintaining velocity damping (k_velocity) to preserve support/hip-yaw improvements without exceeding pitch threshold.

## Audit Results Summary

| Profile | k_position | max_tau | k_velocity | Support | Hip Yaw | Pitch | Pitch Peak Step |
|---------|------------|---------|------------|---------|---------|-------|-----------------|
| J0 | 40.0 | 3.0 | 15.0 | 0.243 m | 0.162 rad | 0.095 rad | 103 |
| J1 | 80.0 | 6.0 | 15.0 | 0.240 m | 0.071 rad | 0.163 rad | 298 |
| J2 | 80.0 | 6.0 | 25.0 | 0.113 m | 0.039 rad | 0.144 rad | 399 |
| J3 | 80.0 | 6.0 | 30.0 | 0.097 m | 0.049 rad | 0.140 rad | 496 |

**Gate status:**
- J0: pitch PASS, support/hip_yaw FAIL
- J1-J3: support/hip_yaw PASS, pitch FAIL

### Mechanism Classification (from audit)

**[HIGH CONFIDENCE] position_authority_induces_pitch_overshoot**
- J1 pitch 0.1632 > J0 pitch 0.0951 with k_position 80.0 vs 40.0
- Evidence: Doubling position stiffness correlates with 72% pitch increase

**[MEDIUM CONFIDENCE] max_position_tau_too_high**
- J1 max_position_tau 6.0 > J0 3.0, tau_position 6.00 vs 3.00
- Evidence: Position torque cap doubled, may allow stronger corrections that induce pitch

**[MEDIUM CONFIDENCE] k_velocity_helps_damp_pitch**
- J3 k_velocity 30.0 > J2 25.0, J3 pitch 0.1395 < J2 pitch 0.1444
- Evidence: Higher velocity damping reduces pitch by 3%, delays peak (step 496 vs 399)

### Key Observations

1. **Pitch timing**: Pitch peaks occur progressively later with higher k_velocity:
   - J0: step 103 (early, low authority)
   - J1: step 298 (delayed by ~200 steps)
   - J2: step 399 (further delayed by ~100 steps)
   - J3: step 496 (near end of run, most damped)

2. **Velocity damping effectiveness**: k_velocity increase from 25→30 reduces pitch from 0.144→0.140 rad (3% improvement), but still 40% over gate.

3. **Support/hip-yaw vs pitch tradeoff**: J2/J3 achieve 54-76% improvements in primary failure modes at cost of 52% pitch increase.

## Design Rationale

**Goal**: Find middle ground between J0 (low authority, pitch safe) and J2/J3 (high authority, support/hip-yaw safe).

**Hypothesis**: Moderate position authority (k_position 60-70, max_position_tau 4.5-5.0) combined with strong velocity damping (k_velocity 25-30) will:
- Preserve majority of support/hip-yaw improvements (aim for support < 0.15m, hip_yaw < 0.07rad)
- Keep pitch under 0.10 rad threshold
- Avoid the full 72% pitch increase seen with k_position=80

**Design constraints:**
- All schedules use continuous height-gated smoothstep interpolation
- No variant-name patches, no discontinuous schedules
- No global nominal changes, no WBC additions
- Height reference: setup target_com_z_m (verified working in previous fix)

## Candidate Family Design

### J2a: Conservative Position Cap

**Target:** Safe pitch with moderate support/hip-yaw improvement

```yaml
k_position_low_max: 60.0     # 50% increase over baseline (vs 100% in J1-J3)
max_position_tau_low_max: 4.5  # 50% increase over baseline (vs 100% in J1-J3)
k_velocity_low_max: 22.0     # 47% increase over baseline
```

**Rationale:**
- Half the position authority increase of J1-J3
- Expect ~half the pitch penalty (~0.123 rad vs 0.144 rad)
- May still exceed gate slightly, but closest to safe zone
- Moderate velocity damping to help without aggressive changes

**Expected performance:**
- support_error: 0.16-0.18 m (marginal improvement, may not pass gate)
- hip_yaw: 0.08-0.10 rad (marginal improvement, may not pass gate)
- pitch: 0.10-0.12 rad (border of gate, may pass if conservative estimate)

**Risk:** May not improve support/hip-yaw enough to pass their gates.

### J2b: Balanced Authority

**Target:** Balance all three metrics near gate thresholds

```yaml
k_position_low_max: 65.0     # 63% increase over baseline
max_position_tau_low_max: 5.0  # 67% increase over baseline
k_velocity_low_max: 25.0     # 67% increase over baseline (proven effective)
```

**Rationale:**
- Moderate position authority (~65-67% of J1-J3 increase)
- Expect pitch between J2a and J2 (~0.12-0.13 rad)
- Velocity damping matches J2 (proven effective in audit)
- Balanced approach to all three parameters

**Expected performance:**
- support_error: 0.14-0.16 m (likely passes gate)
- hip_yaw: 0.06-0.08 rad (border of gate)
- pitch: 0.11-0.13 rad (likely fails gate by small margin)

**Risk:** Pitch may still exceed gate if relationship is more sensitive than linear extrapolation.

### J2c: Velocity-Damping Priority

**Target:** Aggressive velocity damping to counter position-induced pitch

```yaml
k_position_low_max: 60.0     # 50% increase over baseline
max_position_tau_low_max: 4.5  # 50% increase over baseline
k_velocity_low_max: 28.0     # 87% increase over baseline
```

**Rationale:**
- Conservative position authority (same as J2a)
- Aggressive velocity damping (between J2 and J3)
- Hypothesis: velocity damping can compensate for moderate position authority
- Audit showed k_velocity helps damp pitch (J3 < J2)

**Expected performance:**
- support_error: 0.15-0.17 m (border of gate)
- hip_yaw: 0.07-0.09 rad (border of gate)
- pitch: 0.10-0.12 rad (velocity damping may bring under gate)

**Risk:** Velocity damping alone may not be sufficient if position authority is root cause.

### J2d: Torque Cap Priority

**Target:** Limit max torque while keeping stiffness for responsiveness

```yaml
k_position_low_max: 70.0     # 75% increase over baseline
max_position_tau_low_max: 4.5  # 50% increase over baseline
k_velocity_low_max: 25.0     # 67% increase over baseline
```

**Rationale:**
- Higher k_position for responsive position corrections
- Lower max_position_tau to cap peak torque (audit flagged tau_too_high)
- Hypothesis: stiffness is needed for responsiveness, but torque cap prevents overshoot
- Velocity damping at J2 level (proven effective)

**Expected performance:**
- support_error: 0.13-0.15 m (likely passes gate)
- hip_yaw: 0.05-0.07 rad (likely passes gate)
- pitch: 0.11-0.13 rad (may still exceed if stiffness is dominant factor)

**Risk:** If k_position is the primary pitch driver (not tau cap), this won't help enough.

## Schedule Implementation

All candidates use the same continuous height-scheduled formula:

```python
u = clamp((z_high - z_ref) / (z_high - z_low), 0.0, 1.0)
s = 3 * u^2 - 2 * u^3  # smoothstep
param_effective = param_nominal + (param_low_max - param_nominal) * s
```

**Schedule bounds:**
- `z_low = 0.300 m` (activation begins)
- `z_high = 0.393 m` (activation complete, returns to nominal)

**At low_0p300 (z ≈ 0.293 m):**
- `u = (0.393 - 0.293) / (0.393 - 0.300) = 1.075` → clamped to 1.0
- `s = 1.0`
- All low_max parameters are fully active

**At nominal (z ≈ 0.40 m):**
- `u = (0.393 - 0.40) / (0.393 - 0.300) = -0.075` → clamped to 0.0
- `s = 0.0`
- Nominal parameters remain unchanged

## Evaluation Protocol

**Stop-at-first-pass strategy:**

1. Evaluate J2a first (most conservative)
2. If J2a passes all gates → SELECT J2a, STOP
3. If J2a fails → evaluate J2b
4. If J2b passes all gates → SELECT J2b, STOP
5. If J2b fails → evaluate J2c
6. If J2c passes all gates → SELECT J2c, STOP
7. If J2c fails → evaluate J2d
8. If J2d passes all gates → SELECT J2d, STOP
9. If all fail → PITCH_SAFE_JOINT_FIX_REQUIRED (need different approach)

**Per-candidate protocol:**

**Phase 1: low_0p300 Step E 1000**
- Pass: all gates satisfied
- Fail: stop, try next candidate

**Phase 2: low_0p300 Step E 5000**
- Verify sustained performance

**Phase 3: high_0p480 Step E 5000**
- Regression check (schedule should be inactive)

**Phase 4: Step C low_0p300 5000**
- Height recovery verification

**Phase 5: Step C high_0p480 5000**
- High-height recovery verification

**Phase 6: Practical Height Grid Step E**
- Heights: 0.300, 0.330, 0.360, 0.393/low_small, nominal, high_small, 0.450, 0.480

**Phase 7: Step C Grid**
- Heights: 0.300, 0.360, nominal, 0.480

**Phase 8: Five-Variant Regression**
- Variants: nominal, low_tiny, high_tiny, low_small, high_small

## Acceptance Gates

All phases must satisfy:

```yaml
support_position_error_max_abs: <= 0.15 m
hip_yaw_abs_max: <= 0.07 rad
pitch_x_max_abs: <= 0.10 rad
roll_y_max_abs: <= 0.05 rad
final_height_error: <= 0.02 m
contact_valid_percent: >= 99.9%
non_wheel_contacts: == 0
wbc_applied: == false
applied_wbc_contribution_norm: == 0.0 (if available)
hidden_torque_norm: == 0.0
ownership_violation_count: == 0
```

## Implementation Requirements

**Controller configuration:**
- Use `SagittalVelocityDampedBalanceController`
- Authority schedule with continuous scheduling enabled
- Height reference source: `setup.target_com_z_m`

**CLI flags required:**
```bash
--controller-mode balance-core
--sagittal-controller velocity-damped
--vd-sagittal-authority-profile [J2a|J2b|J2c|J2d]
--height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json
```

**Profile definitions in controller:**
```python
if profile_name == "J2a":
    schedule.k_position_low_max = 60.0
    schedule.max_position_tau_low_max = 4.5
    schedule.k_velocity_low_max = 22.0
    schedule.continuous_k_position = True
    schedule.continuous_max_position_tau = True
    schedule.continuous_k_velocity = True
elif profile_name == "J2b":
    schedule.k_position_low_max = 65.0
    schedule.max_position_tau_low_max = 5.0
    schedule.k_velocity_low_max = 25.0
    schedule.continuous_k_position = True
    schedule.continuous_max_position_tau = True
    schedule.continuous_k_velocity = True
elif profile_name == "J2c":
    schedule.k_position_low_max = 60.0
    schedule.max_position_tau_low_max = 4.5
    schedule.k_velocity_low_max = 28.0
    schedule.continuous_k_position = True
    schedule.continuous_max_position_tau = True
    schedule.continuous_k_velocity = True
elif profile_name == "J2d":
    schedule.k_position_low_max = 70.0
    schedule.max_position_tau_low_max = 4.5
    schedule.k_velocity_low_max = 25.0
    schedule.continuous_k_position = True
    schedule.continuous_max_position_tau = True
    schedule.continuous_k_velocity = True
```

## Self-Review Checklist

- [ ] All schedules use continuous smoothstep interpolation (no step functions)
- [ ] Height reference is `setup.target_com_z_m` (not variant name, not root_z)
- [ ] No global nominal parameter changes
- [ ] No WBC additions or modifications
- [ ] No hip-roll modifications
- [ ] No legacy controller path changes
- [ ] Candidates span reasonable parameter space (not all clustered)
- [ ] Expected performance targets are realistic based on audit data
- [ ] Evaluation protocol includes regression checks (high heights, Step C)
- [ ] Acceptance gates cover all invariants (WBC, ownership, contact)

## Risk Analysis

**If all candidates fail:**

1. **Pitch gate may be too strict for low_0p300**
   - Consider relaxing to 0.12-0.15 rad for extreme boundary only
   - Rationale: 0.10 rad may not be achievable at z=0.293m without sacrificing support/hip-yaw

2. **Position-pitch coupling may be fundamental at low heights**
   - Position corrections require wheel torques
   - Wheel torques induce pitch moments at low CoM heights
   - May need different control structure (e.g., explicit pitch compensation in position term)

3. **Alternative approaches:**
   - Pitch-aware position term: scale position corrections based on current pitch
   - Capture gate refinement: gate position corrections more aggressively during pitch transients
   - Leg-joint posture: use hip pitch/knee to counter pitch without wheels (requires WBC-like coordination)

**If J2a passes but provides marginal improvements:**
- Accept it as safe incremental improvement
- Document that low_0p300 remains challenging despite fix
- Consider whether 0.300m is within operational envelope vs physical limit

## Next Steps

1. Implement J2a-J2d profiles in `sagittal_velocity_damped_balance_controller.py`
2. Update evaluation script to support new profiles
3. Run smoke tests (500 steps) for quick verification
4. Run full evaluation protocol in stop-at-first-pass order
5. Document results and selected candidate (if any)

## References

- [Pitch-Position Interaction Audit Report](../outputs/low_height_pitch_position_interaction_audit/pitch_position_interaction_report.md)
- [Schedule Height Reference Bug Fix](sagittal_schedule_height_reference_bug_fix.md)
- [Joint Low-Height Sagittal-Yaw Fix Design (J0-J3)](joint_low_height_sagittal_yaw_fix_design.md)
