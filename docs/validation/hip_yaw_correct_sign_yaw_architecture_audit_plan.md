# Hip-Yaw Correct Sign + Yaw Architecture Audit Plan

**Date:** 2026-06-05

**Status:** DIAGNOSTIC PHASE - Architecture audit in progress

## Executive Summary

The hip-yaw sign fix (wheeled_biped/controllers/shape_posture_controller.py:250) is mathematically correct and passes all 9 unit tests. However, integrating an antisymmetric yaw controller (YawController) on top of the corrected sign fails to stabilize yaw rotation and causes earlier system failure.

**Evidence:**
- Wrong-sign baseline: survives 192 steps, 93° yaw drift
- Correct sign, no yaw controller: survives 192 steps, 93° yaw drift (identical)
- Correct sign + yaw controller (kp=5, kd=1, tau=3): survives 192 steps, 113° yaw drift (WORSE)
- Correct sign + yaw controller (kp=8, kd=2, tau=5): survives 212 steps, 113° yaw drift (WORSE)
- Correct sign + yaw controller (kp=15, kd=3, tau=8): survives 66 steps, 22° yaw drift (FAILS EARLIER)

**Key observation:** Yaw controller executes correctly (verified with debug prints showing ±3-5 Nm antisymmetric torque at step 50), but yaw drift worsens or system destabilizes.

**Hypothesis:** Torque modes are not separated cleanly between shape posture (symmetric) and yaw/divergence control (antisymmetric). The current additive composition may cause:
1. Symmetric shape posture torque dominating/canceling antisymmetric yaw torque
2. Yaw controller fighting shape posture instead of cooperating
3. Mode coupling between divergence (leg geometry) and common-mode (body yaw)
4. Roll-yaw coupling through improper torque decomposition

## Current Architecture

### Shape Posture Controller (Symmetric)
- Per-joint PD control: `tau_pd = -(kp * error + kd * vel)` for each hip-yaw joint
- Left hip-yaw [1]: `tau_L = -(kp * (ref_L - pos_L) - kd * vel_L)`
- Right hip-yaw [6]: `tau_R = -(kp * (ref_R - pos_R) - kd * vel_R)`
- Sign fix applied: entire PD output negated to account for inverted joint axis
- Typical output: -4 to -5 Nm per joint (symmetric)

### Yaw Controller (Antisymmetric)
- Computes: `tau_antisym = kp_yaw * yaw_error - kd_yaw * yaw_rate`
- Applies: `tau_yaw[1] = -tau_antisym`, `tau_yaw[6] = +tau_antisym`
- Typical output: ±3-8 Nm depending on gains
- Intended to generate yaw moments through antisymmetric hip-yaw torque

### Current Composition
```python
tau_shape_posture_with_yaw = tau_shape_posture.at[1].add(tau_yaw[1])
tau_shape_posture_with_yaw = tau_shape_posture_with_yaw.at[6].add(tau_yaw[6])
```

Additive composition: shape posture + yaw controller = final hip-yaw torque.

**Problem:** This composition assumes the modes are orthogonal, but they may not be.

## Mode Decomposition Theory

Hip-yaw joints have two independent modes:

### Common Mode (Body Yaw Rotation)
- `common_error = 0.5 * (e_L + e_R)`
- `common_vel = 0.5 * (v_L + v_R)`
- `common_torque = 0.5 * (tau_L + tau_R)`
- Controls body yaw rotation (robot spinning around vertical axis)
- Should be controlled by antisymmetric torque to generate yaw moment

### Divergence Mode (Leg Geometry / Twist)
- `divergence_error = 0.5 * (e_L - e_R)`
- `divergence_vel = 0.5 * (v_L - v_R)`
- `divergence_torque = 0.5 * (tau_L - tau_R)`
- Controls leg inward/outward twist (symmetric posture)
- Should be controlled by symmetric torque to maintain leg geometry

**Critical insight:** 
- Symmetric torque (`tau_L = tau_R`) affects ONLY divergence mode
- Antisymmetric torque (`tau_L = -tau_R`) affects ONLY common mode (yaw)

**Current architecture violation:**
- Shape posture applies per-joint PD → generates BOTH symmetric AND antisymmetric components
- Yaw controller applies pure antisymmetric → generates ONLY common-mode component
- Additive composition mixes modes without awareness

## Failure Mechanisms (Hypotheses)

### Hypothesis 1: Mode Cancellation
Shape posture generates small antisymmetric component (due to left/right error asymmetry), yaw controller generates large antisymmetric component, but they may oppose each other.

### Hypothesis 2: Symmetric Dominance
Shape posture symmetric component (~4-5 Nm) dominates divergence mode control, making it stiff. This couples with yaw control through roll moments, destabilizing the system.

### Hypothesis 3: Sign Convention Error in Mode Mapping
The mapping from antisymmetric torque to body yaw may have incorrect sign. Pulse tests needed.

### Hypothesis 4: Roll-Yaw Coupling
Divergence mode torque generates roll moments when robot pitches forward. If divergence is uncontrolled, roll destabilizes, which then couples back to yaw.

### Hypothesis 5: Torque Composer Mixing
Balance-core composer may be summing torques incorrectly or clipping in a way that destroys mode orthogonality.

## Audit Plan

### Phase 1: Document Current Failure ✓
This document.

### Phase 2: Torque Decomposition Audit
**Script:** `scripts/audit_hip_yaw_torque_decomposition.py`

**Output:** `outputs/hip_yaw_yaw_architecture_audit/`

**Analyze:**
- Decompose hip-yaw into common/divergence modes at each timestep
- Check if common_torque opposes common_error
- Check if divergence_torque opposes divergence_error
- Check if shape posture and yaw torques cancel
- Classify failure mechanism

**Required artifacts:**
- `hip_yaw_torque_decomposition_summary.json`
- `hip_yaw_torque_decomposition_report.md`
- `hip_yaw_mode_torque_timeseries.csv`
- `hip_yaw_mode_error_timeseries.csv`
- `hip_yaw_roll_yaw_coupling_windows.csv`

### Phase 3: Controller Architecture Code Audit
**Inspect:**
- `wheeled_biped/controllers/shape_posture_controller.py`
- `wheeled_biped/controllers/balance_core_torque_composer.py`
- `scripts/simulate_hierarchical_controller.py`

**Audit:**
1. Where is hip-yaw shape posture torque computed?
2. Is it per-joint PD only?
3. Does it understand common/divergence modes?
4. Where is yaw controller torque added?
5. Is yaw torque added before or after shape posture final clipping?
6. Is yaw torque clipped separately or together?
7. Does the torque composer sum torques or choose ownership?
8. Can a symmetric posture term erase or dominate an antisymmetric yaw term?
9. Are left/right hip-yaw axes mirrored or same-signed?
10. Is body yaw being controlled through the correct mode?

**Output:**
- `hip_yaw_architecture_code_audit.json`
- `hip_yaw_architecture_code_audit_report.md`

### Phase 4: Isolation Experiments
**Script:** `scripts/run_hip_yaw_mode_isolation_experiments.py`

**Output:** `outputs/hip_yaw_yaw_architecture_audit/isolation/`

**Cases:**
- A. Correct sign, shape hip-yaw PD only (baseline)
- B. Correct sign, yaw controller only (disable shape hip-yaw PD)
- C. Correct sign, divergence controller only (explicit divergence mode control)
- D. Correct sign, common-mode controller only (explicit common mode control)
- E. Correct sign, shape posture projected into modes (diagnostic)
- F. Torque pulse tests (common-mode pulse: tau_L = tau_R, divergence-mode pulse: tau_L = -tau_R)

**Measure:**
- Body yaw response
- Roll response
- Hip-yaw divergence response
- Contact/height response

### Phase 5: Design Proper Hip-Yaw/Yaw Architecture
**Document:** `docs/validation/hip_yaw_mode_based_control_design.md`

**Preferred direction:**
Replace additive per-joint hip-yaw + yaw torque with explicit mode-based torque composition:

1. Compute hip-yaw modes:
   - `common_error = 0.5 * (e_L + e_R)`
   - `divergence_error = 0.5 * (e_L - e_R)`

2. Control divergence mode strongly (prevents legs rotating inward/outward)

3. Control common/body-yaw mode separately (stabilizes yaw without fighting divergence posture)

4. Reconstruct joint torques:
   - `tau_L = tau_common + tau_divergence`
   - `tau_R = tau_common - tau_divergence`

5. Apply one final clipping stage after mode recomposition

6. Expose telemetry for both modes

**Candidate designs:**
- M0: current corrected sign baseline
- M1: mode-based divergence control only
- M2: mode-based common + divergence control
- M3: mode-based control with roll-safe damping
- M4: mode-based control with height-gated gains if needed

### Phase 6: Implement Minimal Mode-Based Candidate
Only after Phase 5 design is written.

Implement candidate M1 first, then M2 if M1 doesn't stabilize yaw.

### Phase 7: Validation
Run in stages:
- Stage 1: 100-step smoke (nominal, low_0p300, high_0p480)
- Stage 2: 500-step smoke (only if Stage 1 passes)
- Stage 3: Step E 5000 (only if Stage 2 passes)

### Phase 8: Tests
Add tests for mode decomposition/recomposition correctness.

### Phase 9: Final Report
**Document:** `docs/validation/hip_yaw_mode_based_yaw_control_report.md`

## Success Criteria

**Audit phase success:**
- Root cause classified with high confidence
- Mode decomposition analysis complete
- Isolation experiments identify effective control architecture

**Implementation phase success:**
- Survives 500 steps at all heights
- Yaw drift < 30° (vs 93-113° baseline)
- Hip-yaw divergence < 0.3 rad (vs 0.46 rad wrong-sign baseline)
- No roll collapse
- No contact loss
- Height error <= 0.02 m

## Restrictions

**DO NOT:**
- Revert hip-yaw sign fix (it is mathematically correct)
- Add WBC
- Modify hip-roll unless proven necessary
- Tune gains blindly without architectural fix
- Accept "limited yaw drift" as final answer without exhausting architectural options
- Proceed to Step C or Step D
- Commit

**Allowed:**
- Add diagnostic telemetry
- Audit torque decomposition
- Isolate symmetric vs antisymmetric hip-yaw modes
- Redesign hip-yaw torque composition if justified
- Introduce explicit mode-based control
- Add tests
- Run short smoke tests

## Next Steps

1. Create `scripts/audit_hip_yaw_torque_decomposition.py`
2. Run decomposition audit on existing telemetry
3. Create architecture code audit
4. Run isolation experiments
5. Design mode-based architecture
6. Implement and validate

## Related Files

- Hip-yaw sign fix: [wheeled_biped/controllers/shape_posture_controller.py:250](wheeled_biped/controllers/shape_posture_controller.py#L250)
- Yaw controller: [wheeled_biped/controllers/yaw_controller.py](wheeled_biped/controllers/yaw_controller.py)
- Yaw controller integration: [scripts/simulate_hierarchical_controller.py](scripts/simulate_hierarchical_controller.py)
- Sign fix root cause analysis: [docs/validation/hip_yaw_sign_fix_reveals_missing_yaw_control.md](docs/validation/hip_yaw_sign_fix_reveals_missing_yaw_control.md)
- Sign tests: [tests/test_shape_posture_hip_yaw_sign.py](tests/test_shape_posture_hip_yaw_sign.py)
- Yaw controller tests: [tests/test_yaw_controller.py](tests/test_yaw_controller.py)
