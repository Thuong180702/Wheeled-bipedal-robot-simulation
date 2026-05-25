# Stage 2B Specification: Gravity/Feedforward Compensation

**Date:** 2026-05-24  
**Status:** 🔵 SPECIFICATION - Not yet implemented  
**Goal:** Add gravity/feedforward compensation to achieve stable static standing at h=0.404m equilibrium

---

## Executive Summary

Stage 2 gain sweep conclusively demonstrated that pure PD control is insufficient for h=0.404m equilibrium. Even very_high gains (kp_hip_pitch=120, kp_knee=160) only survived 30/100 steps while approaching actuator limits (56.2 Nm / 57 Nm).

**Root cause:** h=0.404m equilibrium creates ~25-35 Nm gravity torques on support joints. Pure PD control fights gravity reactively after error develops, requiring continuous high torque near saturation.

**Stage 2B solution:** Add feedforward compensation to provide baseline support torques, allowing PD control to focus on small deviations.

---

## Proposed Architecture

### Torque Composition

```
tau_total = tau_static_feedforward + tau_posture_pd + tau_wbc_correction
```

where:
- **`tau_static_feedforward`**: Compensates gravity/internal posture load near calibrated equilibrium
- **`tau_posture_pd`**: Handles small joint deviations (can use lower gains than Stage 2)
- **`tau_wbc_correction`**: Remains correction-only for balance (unchanged from Stage 1)

### Key Principles

1. **Feedforward compensates gravity, PD handles deviations**
   - At equilibrium: feedforward ≈ gravity torque, PD error = 0
   - Small perturbation: feedforward still compensates gravity, PD corrects deviation
   - This is fundamentally different from pure PD fighting gravity continuously

2. **Correction-only WBC semantics preserved**
   - WBC remains equilibrium-relative (Stage 1 semantics)
   - WBC does NOT provide baseline mg support
   - Baseline mg is handled by contact constraints + feedforward joint torques

3. **Feedforward is static, not dynamic**
   - Feedforward torque is constant near equilibrium
   - Does not depend on joint velocity or acceleration
   - Does not require online inverse dynamics
   - Can be precomputed or looked up from calibration

---

## Critical Constraints

### What Stage 2B Must NOT Do

1. **Do NOT map baseline mg through J^T f**
   - This violates correction-only WBC semantics
   - Baseline mg is handled by contact constraints, not WBC
   - WBC provides corrections only

2. **Do NOT use mj_inverse torque blindly as feedforward**
   - mj_inverse includes constraint forces, not just gravity
   - May include contact reaction forces
   - May include numerical artifacts
   - Must validate physically before using

3. **Do NOT increase PD gains further**
   - Gain sweep showed this approach is fundamentally limited
   - very_high gains already near actuator limits
   - Further increase would cause saturation and instability

4. **Do NOT change equilibrium height yet**
   - h=0.404m is the target equilibrium
   - Raising height is a fallback, not the primary solution
   - Stage 2B should solve the problem at h=0.404m

5. **Do NOT add contact recovery or QP yet**
   - Stage 2B focuses on static standing only
   - Contact recovery is Stage 3
   - QP optimization is future work

---

## Diagnostic-First Approach

Stage 2B must follow a diagnostic-first methodology to identify a physically valid feedforward torque reference before implementation.

### Phase A: Gravity/Feedforward Torque Source Audit

**Goal:** Compare candidate feedforward sources at calibrated equilibrium.

**Candidates:**
1. `qfrc_bias[6:16]` - MuJoCo's gravity/Coriolis/centrifugal forces
2. `qfrc_inverse[6:16]` - Inverse dynamics after qvel=0, qacc=0, mj_forward, mj_inverse
3. `qfrc_constraint` contribution if available
4. Empirical holding torque from Stage 2 failure telemetry
5. LegPositionController successful torque at h=0.559m (comparison only)

**For each candidate, report:**
- Support joint torques [2,3,7,8] (hip_pitch, knee for both legs)
- Sign (positive/negative)
- Magnitude vs actuator limits (57 Nm)
- Feasibility under actuator limits
- Physical interpretation (gravity, constraint, numerical artifact)

**Acceptance criteria:**
- Torque magnitude < 50 Nm per joint (leaves margin for PD + WBC)
- Physically interpretable (not numerical artifact)
- Consistent left/right symmetry
- Sign convention matches expected gravity direction

**Rejection criteria:**
- Torque magnitude > 57 Nm (exceeds actuator limits)
- Large left/right asymmetry (> 5 Nm difference)
- Sign inconsistent with gravity direction
- Includes contact reaction forces (violates feedforward semantics)

### Phase B: One-Step Feedforward Validation

**Goal:** Test candidate feedforward in short simulations to validate it reduces height drop.

**Test configurations:**
1. Feedforward only (no PD, no WBC)
2. Feedforward + low PD (kp_hip_pitch=30, kp_knee=40)
3. PD only baseline (for comparison)

**Test duration:** 1, 5, 10, 20 steps

**Metrics to log:**
- `com_z`, `com_vz` (height and vertical velocity)
- `pitch_x`, `roll_y` (orientation)
- `total_contact_force_z` (contact stability)
- `contact_count` (double vs single contact)
- `joint_qpos`, `joint_qvel` (joint state)
- `tau_applied` on support joints [2,3,7,8]
- `saturation_flags` (torque clipping)

**Acceptance criteria:**
- Feedforward reduces height drop compared to PD-only
- Feedforward + low PD performs better than PD-only
- No immediate contact loss (maintains double contact)
- No immediate roll instability (|roll| < 10° for 20 steps)
- Saturation rate < 20%

**Rejection criteria:**
- Feedforward worsens height drop vs PD-only
- Immediate contact loss (single contact within 5 steps)
- Immediate roll instability (|roll| > 20° within 10 steps)
- Continuous saturation (> 50% saturation rate)

### Phase C: Sign and Scaling Sweep

**Goal:** Find optimal sign and scale for feedforward torque.

**Test matrix:**
- Sign: `+candidate`, `-candidate`
- Scale: `0.25`, `0.5`, `0.75`, `1.0`

**Total tests:** 2 signs × 4 scales = 8 configurations

**Test duration:** 50 steps

**Metrics:**
- Survival steps
- Min CoM height
- Max roll/pitch
- Mean contact force
- Saturation rate

**Acceptance criteria:**
- Find sign/scale that survives > 50 steps
- Saturation rate < 20%
- Stable contact forces (> 50 N mean)
- Roll/pitch < 20°

**Selection criteria:**
- Choose sign/scale with longest survival
- If multiple succeed, choose lowest scale (most conservative)
- If none succeed, classify blocker and report

### Phase D: Gradual Feedforward Ramp

**Goal:** Avoid contact impulse by ramping feedforward gradually.

**Implementation:**
```python
tau_static_feedforward = ramp(t) * scale * tau_candidate

where:
ramp(t) = min(t / t_ramp, 1.0)
t_ramp = 20-50 control steps (0.2-0.5 seconds)
```

**Rationale:**
- Sudden feedforward torque at t=0 may cause contact impulse
- Gradual ramp allows contact forces to stabilize
- Robot starts with PD-only, transitions to feedforward+PD

**Test:**
- Compare ramped vs instant feedforward
- Log contact forces during ramp
- Verify no contact loss during ramp

### Phase E: Integration Test

**Goal:** Validate feedforward + PD + WBC for 100-step standing.

**Command:**
```bash
python scripts/simulate_hierarchical_controller.py \
  --enable-stage2-static-posture-hold \
  --enable-stage2b-gravity-feedforward \
  --steps 100
```

**Metrics:**
- `survival_steps` (target: 100/100)
- `termination_reason` (target: none)
- `min_com_z` (target: > 0.38m)
- `mean_contact_fz` (target: > 70 N)
- `max_roll`, `max_pitch` (target: < 15°)
- `saturation_rate` (target: < 20%)
- First 20 steps detailed telemetry

**Acceptance criteria:**
- Survives 100/100 steps
- No termination
- Height stable (min > 0.38m)
- Contact stable (mean > 70 N)
- Orientation stable (roll/pitch < 15°)
- Torque feasible (saturation < 20%)

**If acceptance fails, classify blocker:**
1. Feedforward infeasible due to torque limits
2. Wrong sign / model mismatch
3. Contact solver / contact loss
4. Lateral roll instability
5. Correction-only WBC insufficient for roll/pitch recovery

---

## Expected Feedforward Torque Magnitude

Based on Stage 2 diagnostic findings:

**Gravity torque estimate at h=0.404m:**
- Hip pitch: ~25-30 Nm
- Knee: ~30-35 Nm

**Actuator limits:**
- Max torque: 57 Nm per joint

**Feedforward target:**
- Hip pitch: 20-25 Nm (leaves 32-37 Nm for PD + WBC)
- Knee: 25-30 Nm (leaves 27-32 Nm for PD + WBC)

**Margin for PD + WBC:**
- PD with low gains (kp=30-40): ~5-10 Nm for 0.1 rad error
- WBC correction: ~5-15 Nm for balance
- Total margin: ~10-25 Nm

**Feasibility check:**
- Feedforward + PD + WBC < 57 Nm actuator limit
- 25 Nm + 10 Nm + 15 Nm = 50 Nm < 57 Nm ✓

---

## Implementation Phases

### Phase 1: Diagnostic Script (Phase A-C)

**File:** `scripts/debug_stage2b_feedforward_sources.py`

**Tasks:**
1. Load calibrated equilibrium state
2. Extract candidate feedforward torques
3. Run one-step validation tests
4. Run sign/scaling sweep
5. Generate diagnostic report

**Output:**
- Candidate torque comparison table
- One-step validation results
- Sign/scaling sweep results
- Recommended feedforward source, sign, scale

### Phase 2: Feedforward Controller (Phase D)

**File:** `wheeled_biped/controllers/static_feedforward_controller.py`

**Tasks:**
1. Implement feedforward torque lookup/computation
2. Implement gradual ramp
3. Add diagnostics (feedforward torque, ramp progress)
4. Add unit tests

**Interface:**
```python
class StaticFeedforwardController:
    def __init__(self, feedforward_torques, ramp_steps=20):
        self.feedforward_torques = feedforward_torques  # (10,) array
        self.ramp_steps = ramp_steps
        self.step_count = 0
    
    def compute_feedforward_torque(self) -> tuple[Array, dict]:
        ramp = min(self.step_count / self.ramp_steps, 1.0)
        tau_ff = ramp * self.feedforward_torques
        self.step_count += 1
        
        diagnostics = {
            "feedforward_ramp": ramp,
            "feedforward_norm": float(jnp.linalg.norm(tau_ff)),
            "feedforward_hip_pitch_max": ...,
            "feedforward_knee_max": ...,
        }
        
        return tau_ff, diagnostics
```

### Phase 3: Integration (Phase E)

**File:** `scripts/simulate_hierarchical_controller.py`

**Tasks:**
1. Add `--enable-stage2b-gravity-feedforward` flag
2. Add feedforward torque CLI arguments
3. Initialize StaticFeedforwardController
4. Modify torque composition:
   ```python
   tau_total = tau_static_feedforward + tau_static_posture + tau_wbc_correction
   ```
5. Add feedforward telemetry logging
6. Run 100-step validation

### Phase 4: Tests

**File:** `tests/test_static_feedforward_controller.py`

**Tasks:**
1. Test feedforward torque shape and bounds
2. Test ramp function (0 → 1 over ramp_steps)
3. Test wheel torques remain zero
4. Test left/right symmetry
5. Test diagnostics computed correctly

---

## Success Criteria

Stage 2B is complete when:

1. ✅ Diagnostic script identifies physically valid feedforward source
2. ✅ One-step validation shows feedforward reduces height drop
3. ✅ Sign/scaling sweep finds optimal configuration
4. ✅ StaticFeedforwardController implemented and tested
5. ✅ Integration test survives 100/100 steps
6. ✅ All unit tests pass
7. ✅ Telemetry logs feedforward torques correctly

---

## Failure Modes and Fallbacks

### If no feedforward source passes Phase A

**Blocker:** No physically valid feedforward torque available.

**Fallback options:**
1. Raise equilibrium height to h=0.45-0.50m (lower gravity torques)
2. Use model-based inverse dynamics (requires accurate robot model)
3. Learn feedforward from data (requires training)

### If feedforward passes Phase A but fails Phase B

**Blocker:** Feedforward torque sign or magnitude incorrect.

**Diagnosis:**
- Check sign convention (positive = extension or flexion?)
- Check coordinate frame (body frame vs world frame?)
- Check joint indexing (left vs right, hip vs knee?)

**Fallback:**
- Try negative sign
- Try different scaling
- Try different candidate source

### If feedforward passes Phase B but fails Phase E

**Blocker:** Feedforward + PD insufficient for 100-step standing.

**Possible causes:**
1. Lateral roll instability (feedforward doesn't help roll)
2. Contact loss (feedforward doesn't prevent wheel slip)
3. WBC correction insufficient (need stronger WBC gains)
4. Model mismatch (feedforward based on wrong model)

**Diagnosis:**
- Check roll trajectory (growing vs stable)
- Check contact forces (stable vs intermittent)
- Check WBC correction torques (saturating vs reasonable)
- Check feedforward vs actual gravity torques

**Fallback:**
- Add hip roll feedforward (currently only hip pitch/knee)
- Increase WBC correction gains
- Add contact recovery logic (Stage 3)
- Raise equilibrium height

---

## Telemetry Requirements

Stage 2B must log:

**Feedforward torques:**
- `tau_feedforward_per_joint` (comma-separated, 10 values)
- `tau_feedforward_norm`
- `tau_feedforward_hip_pitch_max`
- `tau_feedforward_knee_max`
- `feedforward_ramp` (0.0 → 1.0)

**Torque composition:**
- `tau_static_feedforward` (10,)
- `tau_static_posture` (10,)
- `tau_wbc_correction` (10,)
- `tau_total` (10,)

**Saturation:**
- `saturation_count_feedforward`
- `saturation_count_posture`
- `saturation_count_wbc`
- `saturation_count_total`

---

## Documentation Requirements

Stage 2B must produce:

1. **Diagnostic report:** `docs/superpowers/reports/2026-05-24-stage2b-feedforward-diagnostics.md`
   - Candidate torque comparison
   - One-step validation results
   - Sign/scaling sweep results
   - Recommended configuration

2. **Implementation report:** `docs/superpowers/reports/2026-05-24-stage2b-implementation-report.md`
   - Implementation summary
   - Integration test results
   - Remaining blockers (if any)
   - Next steps

---

## Timeline Estimate

- Phase A (diagnostic script): 2-3 hours
- Phase B (one-step validation): 1-2 hours
- Phase C (sign/scaling sweep): 1-2 hours
- Phase D (feedforward controller): 2-3 hours
- Phase E (integration test): 1-2 hours
- Tests and documentation: 2-3 hours

**Total:** 9-15 hours

---

## References

- [Stage 2 Diagnostic Report](2026-05-24-stage2-diagnostic-report.md)
- [Stage 2 Gain Sweep Findings](2026-05-24-stage2-gain-sweep-findings.md)
- [WBC Correction-Only Fix Specification](2026-05-23-wbc-correction-only-fix.md)

---

## Approval

This specification must be reviewed and approved before implementation begins.

**Reviewer:** User  
**Status:** 🔵 PENDING REVIEW
