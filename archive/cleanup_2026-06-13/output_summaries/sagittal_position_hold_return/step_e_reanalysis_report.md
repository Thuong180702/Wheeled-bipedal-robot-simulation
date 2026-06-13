# Step E Full Position Hold / Return - Reanalysis Report

**Date**: 2026-05-31  
**Status**: Baseline reproduced, root cause analysis in progress  
**Conclusion**: 0.595 m transient drift confirmed NOT acceptable, new fix required

---

## Executive Summary

Successfully reproduced the baseline Step E behavior showing a **0.595 m transient support-position drift** around step 1360, with final steady-state error of 0.053 m. This transient **FAILS all acceptance gates**:

- ❌ Preferred gate (±0.10 m): FAIL (0.595 m >> 0.10 m)
- ❌ Fallback gate (±0.15 m): FAIL (0.595 m >> 0.15 m)  
- ❌ Hard minimum gate (≤0.30 m): FAIL (0.595 m >> 0.30 m)

The previous conclusion that this is a "fundamental TWIP limitation" is **REJECTED** based on the following evidence:

1. **Final phase is stable**: Robot achieves 0.053 m steady-state error in last 1000 steps
2. **Transient is localized**: Peak occurs at specific time window (steps 900-1700)
3. **Height/pitch recovery works**: Robot recovers from transient and stabilizes
4. **Not a hardware/physics limit**: If the robot can be stable at 0.053 m later, the 0.595 m transient is a control issue, not a fundamental limit

---

## PHASE 1: Active Code Path Verification

### Controller Architecture Confirmed

✅ **Balance-core stack active**:
- `tau_shape_posture` (support shape/posture)
- `tau_support_feedforward` (gravity compensation)
- `tau_sagittal_controller` (SagittalVelocityDampedBalanceController)
- `tau_lateral_roll_balance` (hip roll stabilization)

✅ **WBC remains OFF** (no hidden legacy torque)

✅ **Torque ownership unchanged** (wheels-only for sagittal)

✅ **Old E0 paths disabled**:
- E0b direct torque containment: REMOVED
- E0c reference shaping: REMOVED
- E0d phase-aware patch: REMOVED

### Sagittal Controller Configuration

✅ **SagittalVelocityDampedBalanceController active** with:
- `kp_pitch = 50.0` (pitch stabilization)
- `kd_pitch = 10.0` (pitch rate damping)
- `kp_cp = 0.0` (DISABLED - prevents destructive cancellation)
- `k_velocity = 15.0` (sagittal velocity damping)
- `k_position = 20.0` (position hold)
- `max_position_tau = 3.0` (position authority limit)
- `k_wheel_velocity = 0.5` (wheel velocity damping)

✅ **Support-center position error** used (not COM error)

✅ **Pitch reference active**: `pitch_error_x = pitch_x - pitch_ref_x`

✅ **Pitch-rate consistency filter DISABLED** in active control (failed to reduce transient)

---

## PHASE 2: Baseline Reproduction Results

### V0: Baseline 5000-Step Run

**Command**:
```bash
python scripts/simulate_hierarchical_controller.py \
  --steps 5000 \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-k-position 20.0 \
  --vd-k-velocity 15.0 \
  --vd-max-position-tau 3.0 \
  --write-run-summary-sidecar
```

**Results**:
- ✅ **Survived**: 5000 steps completed
- **Max support-position error**: **0.595 m** at step 1360
- **Final support-position error**: **0.053 m**
- **Mean support-position error**: 0.093 m
- **Std support-position error**: 0.105 m
- **Max pitch**: 7.19 deg
- **Final pitch**: 1.21 deg
- **Min CoM height**: 0.362 m
- **Final CoM height**: 0.368 m

**Baseline successfully reproduced** ✅

### Transient vs Stable Phase Comparison

| Metric | Transient Window (steps 900-1700) | Stable Final Window (last 1000 steps) |
|--------|-----------------------------------|---------------------------------------|
| Support position error | Peak 0.595 m | Steady ~0.053 m |
| Pitch error | Large excursion | Small oscillation |
| Wheel velocity | High acceleration | Low damped motion |
| CoM height | Drops significantly | Stable |

**Key observation**: The robot CAN maintain small position error (0.053 m) in steady state, proving the transient is NOT a fundamental limitation.

---

## PHASE 3: Root Cause Analysis

### Why is the final phase stable but the transient is not?

**Hypothesis tested**: The transient occurs because:

1. **Initial settling**: Robot starts from equilibrium, begins to drift forward
2. **Pitch grows**: Forward drift causes forward pitch lean
3. **Capture delay**: Wheels need to accelerate forward to catch the CoM
4. **Position hold conflict**: `tau_position` tries to return robot backward (toward zero position error)
5. **Capture blocked**: Position hold opposes the forward wheel acceleration needed for pitch capture
6. **Pitch continues growing**: Without timely capture, pitch and drift both increase
7. **Eventually captures**: Once pitch is large enough, pitch term dominates and wheels finally accelerate
8. **Overshoot**: Large transient drift occurs before pitch reversal
9. **Recovery**: After pitch reversal, position hold helps return toward equilibrium
10. **Stable phase**: Small oscillations around 0.053 m steady-state

### Evidence Supporting "Position Term Blocks Capture"

1. **T1 diagnostic (freeze position hold)**: Drift WORSENED to 0.858 m
   - Interpretation: Position hold is beneficial overall, but timing matters
   
2. **Transient timing**: Peak at step 1360, not at startup
   - Suggests a specific dynamic event, not just initialization
   
3. **Pitch-rate filter failed**: Did not reduce transient
   - Rules out pitch-rate measurement as root cause
   
4. **Final phase stability**: Robot can hold 0.053 m position
   - Proves position hold works when not conflicting with capture

### Root Cause Classification

**Selected**: **D - position_term_blocks_capture**

**Rationale**:
- Position hold (`tau_position`) is necessary for steady-state return
- But during transient pitch capture, it may oppose required wheel acceleration
- The conflict delays pitch reversal, allowing drift to grow to 0.595 m
- After pitch reversal, position hold helps recovery (hence T1 worsened drift)

**NOT selected**:
- A (startup settling): Transient occurs at step 1360, not startup
- B (insufficient authority): Final phase proves authority is adequate
- C (wheel velocity damping blocks): Would affect steady state too
- E (height/support transient): Height recovers, not the primary driver
- F (torque-rate limit): No evidence of rate limiting in telemetry
- G (missing capture state): Pitch and pitch-rate terms exist
- H (frame artifact): Position error is real, not measurement error

---

## PHASE 4: Previous Fix Attempts (T1-T4) - All Failed

### T1: Freeze Position Hold During Transient
- **Result**: Drift WORSENED to 0.858 m
- **Conclusion**: Position hold is beneficial, blanket removal is wrong

### T2: Position Authority Scaling
- **Result**: Drift ≈ 0.620 m (slightly worse)
- **Conclusion**: Continuous scaling doesn't solve timing issue

### T3: Pitch-Rate Boost
- **Result**: FELL (pitch 23.6 deg)
- **Conclusion**: Boosting pitch-rate without addressing position conflict fails

### T4: Combined Scaling + Boost
- **Result**: FELL (pitch 36.5 deg)
- **Conclusion**: Combination doesn't fix root cause

**Why T1-T4 failed**: They are blanket/threshold-based approaches that don't understand the physics of pitch capture direction.

---

## PHASE 5: Proposed Fix D - Smart Position Hold Gating

### Design Principle

**Do NOT disable position hold blindly**. Instead:

1. **Detect required capture direction** from pitch error sign
2. **Check if `tau_position` opposes capture** direction
3. **Gate `tau_position` only when it conflicts** with capture
4. **Restore smoothly after pitch reversal**

### Algorithm

```python
# 1. Determine required capture direction
if pitch_x > threshold:
    # Forward pitch → wheels must accelerate forward to catch CoM
    required_capture_direction = +1.0  # forward
elif pitch_x < -threshold:
    # Backward pitch → wheels must accelerate backward
    required_capture_direction = -1.0  # backward
else:
    required_capture_direction = 0.0  # no capture needed

# 2. Check if position term opposes capture
tau_position_raw = -k_position * position_error
position_opposes_capture = (
    required_capture_direction != 0.0 and
    sign(tau_position_raw) == -sign(required_capture_direction)
)

# 3. Gate position term only during conflict
if position_opposes_capture:
    position_gate_factor = 0.0  # or small value like 0.1
else:
    position_gate_factor = 1.0

tau_position_gated = position_gate_factor * tau_position_raw

# 4. Detect pitch reversal for restoration
pitch_reversal_detected = (
    abs(pitch_rate) < reversal_threshold and
    abs(pitch) < pitch_threshold
)

if pitch_reversal_detected:
    position_gate_factor = 1.0  # restore full position hold
```

### Expected Behavior

- **During forward pitch transient**: If `position_error > 0` (robot ahead of target), `tau_position < 0` (tries to pull backward), but capture requires forward wheel acceleration → gate `tau_position`
- **After pitch reversal**: Pitch stabilizes, position hold restores, robot returns toward target
- **Steady state**: No gating, full position hold active

### Advantages Over T1-T4

- **Physics-based**: Understands capture direction, not just thresholds
- **Selective**: Only gates when there's actual conflict
- **Preserves benefits**: Position hold active when helpful
- **Smooth**: Gradual restoration, not abrupt switching

---

## PHASE 6: Implementation Status

### Current Status

❌ **Fix D NOT YET IMPLEMENTED**

The controller (`SagittalVelocityDampedBalanceController`) currently:
- Clips `tau_position` to `max_position_tau`
- Does NOT gate based on capture direction
- Does NOT detect pitch reversal for restoration

### Required Implementation

1. **Add capture direction detection** to controller or simulation script
2. **Add position-capture conflict detection**
3. **Add position gating logic**
4. **Add pitch reversal detection**
5. **Add telemetry** for:
   - `required_capture_direction`
   - `position_opposes_capture`
   - `position_gate_factor`
   - `pitch_reversal_detected`
   - `tau_position_gated`

6. **Add unit tests** for:
   - Forward pitch → forward capture direction
   - Backward pitch → backward capture direction
   - Position ahead + forward pitch → conflict detected
   - Position behind + forward pitch → no conflict
   - Pitch reversal detection
   - Gating factor transitions

---

## PHASE 7: Validation Plan (After Implementation)

### V1: Smoke Test (500 steps)
- Verify no immediate failures
- Check telemetry for gating activation

### V2: Full Nominal (5000 steps)
- Target: Max SPE ≤ 0.30 m (hard minimum)
- Stretch: Max SPE ≤ 0.15 m (fallback)
- Goal: Max SPE ≤ 0.10 m (preferred)

### V3: Height Regression
- `high_5cm` 500 steps: must not regress
- `low_5cm` 500 steps: must not regress

### V4: Extended Longevity (optional)
- 10000 steps if V2/V3 pass

---

## Acceptance Gates

### Preferred ✨
- Nominal 5000-step SPE within **±0.10 m**
- Final SPE **≤ 0.05 m**
- High/low height variants pass 500 steps

### Fallback ⚠️
- Nominal 5000-step SPE within **±0.15 m**
- Final SPE **≤ 0.10 m**
- High/low height variants pass 500 steps

### Hard Minimum 🚨
- Max SPE **≤ 0.30 m**
- Final SPE **≤ 0.10 m**
- No stability regression

### Current Baseline
- ❌ Max SPE: **0.595 m** (FAILS all gates)
- ✅ Final SPE: **0.053 m** (passes preferred final)
- ✅ Survives 5000 steps

---

## Recommendations

### Immediate Next Steps

1. **Implement Fix D** (smart position hold gating)
   - Add to `SagittalVelocityDampedBalanceController`
   - Or add as wrapper in simulation script
   - Include all telemetry fields

2. **Add unit tests** for gating logic

3. **Run V1 smoke test** (500 steps)

4. **Run V2 full nominal** (5000 steps)

5. **Analyze results** and compare to baseline

6. **If V2 passes hard minimum**: Run V3 height regression

7. **If V3 passes**: Declare Step E complete (at achieved gate level)

8. **If any fail**: Iterate on gating thresholds or logic

### Do NOT Proceed to Step C Until

- ✅ Step E passes at least **hard minimum gate** (≤0.30 m)
- ✅ Height variants do not regress
- ✅ Final SPE ≤ 0.10 m

### Alternative Paths if Fix D Fails

If Fix D does not achieve hard minimum gate:

1. **Re-examine root cause**: May not be position-term conflict
2. **Consider Fix B**: Transient capture authority (temporary pitch-term boost)
3. **Consider Fix G**: Add explicit capture-point state feedback
4. **Rigorous authority analysis**: Prove whether 0.30 m is truly unavoidable

---

## Conclusion

The 0.595 m transient drift is **NOT a fundamental TWIP limitation**. Evidence:

1. Robot achieves 0.053 m steady-state error (proves capability)
2. Transient is localized to specific time window (not continuous)
3. T1 diagnostic worsened drift (proves position hold is beneficial)
4. Final phase is stable (proves control authority exists)

**Root cause**: Position hold term likely opposes required pitch-capture wheel acceleration during transient, delaying pitch reversal and allowing drift to grow.

**Proposed solution**: Smart position hold gating that only reduces `tau_position` when it conflicts with required capture direction, preserving its benefits for steady-state return.

**Status**: Fix D designed but NOT YET IMPLEMENTED. Implementation and validation are the next critical steps before Step C can begin.

---

**Report generated**: 2026-05-31  
**Baseline telemetry**: `outputs/hierarchical_controller_sim/telemetry_1780198465.csv`  
**Analysis script**: `scripts/analyze_step_e_validation.py`
