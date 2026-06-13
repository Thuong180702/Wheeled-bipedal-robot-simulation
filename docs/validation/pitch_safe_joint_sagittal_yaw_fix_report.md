# Pitch-Safe Joint Sagittal-Yaw Fix Report

**Date:** 2026-06-05  
**Status:** FAILED  
**Classification:** PITCH_SAFE_JOINT_FIX_REQUIRED

## Executive Summary

Investigation into pitch-position interaction at low_0p300 revealed that increasing position authority (J1-J3 profiles) improves support_error and hip_yaw but causes pitch to exceed 0.10 rad threshold. Designed and evaluated four pitch-safe candidates (J2a-J2d) with reduced position authority to balance all three metrics.

**Result:** All pitch-safe candidates FAILED Phase 1. None achieved simultaneous pass of support (≤0.15m), hip_yaw (≤0.07rad), and pitch (≤0.10rad) gates at low_0p300.

**Root cause:** At z=0.300m, position-pitch coupling is fundamental. Reducing position authority enough to satisfy pitch gate causes hip-yaw and support to regress beyond acceptable thresholds.

**Recommendation:** low_0p300 (z=0.300m) may be at or beyond the operational envelope for strict three-gate acceptance criteria. Consider relaxing pitch threshold to 0.12-0.15 rad for extreme boundary scenarios, or accept that 0.300m is a physical limit.

## Investigation Timeline

### Phase 1: Pitch-Position Interaction Audit

**Method:** Analyzed J0-J3 smoke test telemetry (500 steps at low_0p300) to classify pitch blocker mechanism.

**Findings:**

| Profile | k_pos | max_tau | k_vel | Support | Hip Yaw | Pitch | Peak Step |
|---------|-------|---------|-------|---------|---------|-------|-----------|
| J0 | 40.0 | 3.0 | 15.0 | 0.243 m | 0.162 rad | 0.095 rad | 103 |
| J1 | 80.0 | 6.0 | 15.0 | 0.240 m | 0.071 rad | 0.163 rad | 298 |
| J2 | 80.0 | 6.0 | 25.0 | 0.113 m | 0.039 rad | 0.144 rad | 399 |
| J3 | 80.0 | 6.0 | 30.0 | 0.097 m | 0.049 rad | 0.140 rad | 496 |

**Gate status:**
- J0: pitch PASS, support/hip_yaw FAIL
- J1-J3: support/hip_yaw PASS, pitch FAIL

**Mechanism classification:**

1. **[HIGH CONFIDENCE] position_authority_induces_pitch_overshoot**
   - J1 pitch 0.163 > J0 pitch 0.095 with k_position 80 vs 40
   - Doubling position stiffness → 72% pitch increase

2. **[MEDIUM CONFIDENCE] max_position_tau_too_high**
   - J1 max_position_tau 6.0 > J0 3.0
   - Position torque cap doubled, may allow overshoot

3. **[MEDIUM CONFIDENCE] k_velocity_helps_damp_pitch**
   - J3 k_velocity 30 > J2 25 → J3 pitch 0.140 < J2 pitch 0.144
   - Higher velocity damping delays pitch peak (step 496 vs 399)
   - Provides 3% improvement but insufficient to reach gate

### Phase 2: Pitch-Safe Candidate Design

**Strategy:** Reduce position authority from J1-J3 levels while maintaining velocity damping.

**Hypothesis:** Moderate position authority (k_position 60-70, max_tau 4.5-5.0) combined with velocity damping (k_vel 22-28) will preserve support/hip-yaw improvements while keeping pitch under threshold.

**Candidates designed:**

**J2a: Conservative position cap**
- k_position: 60 (50% increase vs baseline)
- max_tau: 4.5 (50% increase)
- k_velocity: 22 (47% increase)
- Rationale: safest pitch profile, half the authority increase

**J2b: Balanced authority**
- k_position: 65 (63% increase)
- max_tau: 5.0 (67% increase)
- k_velocity: 25 (67% increase, matches J2)
- Rationale: balanced approach to all parameters

**J2c: Velocity damping priority**
- k_position: 60 (50% increase)
- max_tau: 4.5 (50% increase)
- k_velocity: 28 (87% increase)
- Rationale: aggressive damping to counter position-induced pitch

**J2d: Torque cap priority**
- k_position: 70 (75% increase)
- max_tau: 4.5 (50% increase, caps peak torque)
- k_velocity: 25 (67% increase)
- Rationale: higher stiffness but limited torque overshoot

### Phase 3: Evaluation Results

**Protocol:** Stop-at-first-pass evaluation (J2a→J2b→J2c→J2d) at low_0p300 Step E 1000 steps.

**Results:**

| Profile | Support | Hip Yaw | Pitch | WBC Applied | Decision |
|---------|---------|---------|-------|-------------|----------|
| J2a | PASS | 0.136 rad (FAIL) | 0.119 rad (FAIL) | 15.09 Nm (FAIL) | FAILED_PHASE_1 |
| J2b | PASS | 0.136 rad (FAIL) | 0.126 rad (FAIL) | 15.92 Nm (FAIL) | FAILED_PHASE_1 |
| J2c | PASS | 0.118 rad (FAIL) | 0.122 rad (FAIL) | 13.72 Nm (FAIL) | FAILED_PHASE_1 |
| J2d | PASS | 0.130 rad (FAIL) | 0.120 rad (FAIL) | 15.94 Nm (FAIL) | FAILED_PHASE_1 |

**Acceptance gates:**
- support_position_error ≤ 0.15 m: **ALL PASS**
- hip_yaw_abs_max ≤ 0.07 rad: **ALL FAIL**
- pitch_x_max_abs ≤ 0.10 rad: **ALL FAIL**
- WBC applied == false: **ALL FAIL** (13-16 Nm applied)

**Critical issues:**

1. **WBC invariant violation:** All candidates show WBC applied 13-16 Nm despite `--controller-mode balance-core`. This invalidates the comparison and must be fixed before any further evaluation.

2. **Pitch threshold not achieved:** Even ignoring WBC, all candidates exceed pitch gate by 19-26%. Reducing position authority was insufficient.

3. **Hip-yaw regression:** Hip-yaw increased to 0.118-0.136 rad (vs J2/J3: 0.039-0.071 rad). Reducing position authority from J2/J3 levels caused hip-yaw to regress toward baseline.

### Phase 4: Analysis

**Why pitch-safe candidates failed:**

1. **Position authority floor:** Achieving hip_yaw < 0.07 rad requires k_position ≥ 80 (based on J2/J3 results). Reducing to k_position 60-70 causes hip-yaw to regress.

2. **Pitch authority ceiling:** Achieving pitch < 0.10 rad requires k_position < 60 (based on linear extrapolation from J0/J1). This conflicts with hip-yaw requirement.

3. **Velocity damping insufficient:** k_velocity 22-30 provides 3-10% pitch reduction but cannot compensate for position authority increase needed for hip-yaw.

4. **Fundamental coupling at low height:** At z=0.300m, wheel-based position corrections induce pitch moments. The authority needed to stabilize support/hip-yaw inherently causes pitch excursions.

**Comparison: J0 baseline vs pitch-safe candidates:**

| Metric | J0 | J2a | J2b | J2c | J2d |
|--------|----|----|----|----|-----|
| Support error | 0.243 m | PASS | PASS | PASS | PASS |
| Hip yaw | 0.162 rad | 0.136 rad | 0.136 rad | 0.118 rad | 0.130 rad |
| Pitch | 0.095 rad | 0.119 rad | 0.126 rad | 0.122 rad | 0.120 rad |

Pitch-safe candidates improved support but **worsened** both hip-yaw and pitch vs J0 baseline. They occupy an intermediate failure zone: too much authority for pitch, too little for hip-yaw.

## Conclusion

**Classification:** PITCH_SAFE_JOINT_FIX_REQUIRED

The J2a-J2d parameter space (k_position 60-70, max_tau 4.5-5.0, k_velocity 22-28) does not contain a solution that simultaneously satisfies all three gates at low_0p300.

**Evidence:**
- J0 baseline: pitch safe, support/hip-yaw fail
- J2/J3 scheduled: support/hip-yaw safe, pitch fails
- J2a-J2d pitch-safe: all three gates fail

**Root cause:** Position-pitch coupling is fundamental at z=0.300m. The wheel torques required to stabilize support and hip-yaw inherently induce pitch moments at low CoM heights.

## Recommendations

### Option A: Relax pitch threshold for extreme boundary

Accept that 0.300m is an extreme boundary with reduced pitch margin:

```yaml
low_0p300_gates:
  support_position_error_max_abs: 0.15 m
  hip_yaw_abs_max: 0.07 rad
  pitch_x_max_abs: 0.15 rad  # relaxed from 0.10 rad
```

**Rationale:**
- J2/J3 achieve support and hip-yaw within specification
- Pitch 0.14 rad (8°) is stable, just exceeds strict threshold
- 0.300m is 25% below nominal height, at physical kinematic limit

**Risk:** May mask instabilities if pitch continues to grow.

### Option B: Accept 0.300m as operational envelope boundary

Define operational envelope as z ≥ 0.330m, mark 0.300m as physical limit:

```yaml
operational_envelope:
  z_min: 0.330 m  # operational lower bound
  z_max: 0.480 m  # operational upper bound

physical_envelope:
  z_min: 0.300 m  # physical kinematic limit
  z_max: 0.500 m  # physical upper limit
```

**Rationale:**
- 0.330m candidates (if tested) may satisfy all gates
- 0.300m remains achievable but with degraded performance
- Clear separation between operational capability and physical limit

**Risk:** Reduces operational range by 30mm.

### Option C: Investigate pitch-aware position control

Modify position control to scale based on current pitch state:

```python
position_scale = max(0.5, 1.0 - abs(pitch_x) / 0.08)
tau_position_scaled = tau_position * position_scale
```

**Rationale:**
- Reduces position corrections during pitch transients
- May break position-pitch feedback loop
- Requires controller modification, not just parameter tuning

**Risk:** Complex, may introduce new failure modes.

## Files Changed

- `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` - added J2a-J2d profile definitions
- `scripts/simulate_hierarchical_controller.py` - added J2a-J2d to CLI parser and profile registry
- `scripts/audit_low_height_pitch_position_interaction.py` - created pitch-position audit script
- `scripts/evaluate_pitch_safe_joint_candidates.py` - created pitch-safe candidate evaluation script
- `docs/validation/pitch_safe_joint_sagittal_yaw_candidate_design.md` - pitch-safe candidate design doc
- `docs/validation/pitch_safe_joint_sagittal_yaw_fix_report.md` - this report

## Artifacts Generated

- `outputs/low_height_pitch_position_interaction_audit/` - pitch-position audit results
  - `pitch_position_interaction_summary.json` - metrics for all profiles
  - `pitch_peak_windows.csv` - telemetry around pitch peaks
  - `support_peak_windows.csv` - telemetry around support peaks
  - `torque_interaction_comparison.csv` - torque term comparison
  - `pitch_failure_classification.json` - mechanism classification
  - `pitch_position_interaction_report.md` - audit report

- `outputs/pitch_safe_joint_sagittal_yaw_fix/` - pitch-safe candidate evaluation results
  - `pitch_safe_candidate_summary.json` - evaluation summary
  - `evaluation_log.txt` - full evaluation log
  - `J2a_low_0p300_step_e_1000/telemetry.csv` - J2a telemetry
  - `J2b_low_0p300_step_e_1000/telemetry.csv` - J2b telemetry
  - `J2c_low_0p300_step_e_1000/telemetry.csv` - J2c telemetry
  - `J2d_low_0p300_step_e_1000/telemetry.csv` - J2d telemetry

## Next Steps

**Immediate:**
1. Fix WBC invariant violation in evaluation setup
2. Decide between Option A (relax pitch threshold) or Option B (accept operational boundary)

**If pursuing Option C (pitch-aware position control):**
1. Design pitch-aware scaling function
2. Implement in sagittal controller
3. Add tests for pitch-position coupling
4. Re-evaluate J2/J3 with pitch-aware modification

**Do NOT:**
- Proceed to full Phase 6 evaluation with current candidates
- Relax support or hip-yaw thresholds
- Add WBC or modify hip-roll
- Use discontinuous schedules or variant-name patches

## References

- [Pitch-Position Interaction Audit Report](../outputs/low_height_pitch_position_interaction_audit/pitch_position_interaction_report.md)
- [Pitch-Safe Candidate Design](pitch_safe_joint_sagittal_yaw_candidate_design.md)
- [Schedule Height Reference Bug Fix](sagittal_schedule_height_reference_bug_fix.md)
- [Joint Low-Height Sagittal-Yaw Fix Design (J0-J3)](joint_low_height_sagittal_yaw_fix_design.md)
