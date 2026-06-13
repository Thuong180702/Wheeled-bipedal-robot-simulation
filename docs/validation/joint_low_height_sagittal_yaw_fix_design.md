# Joint Low-Height Sagittal-Yaw Fix Design

**Date:** 2026-06-05  
**Phase:** 5  
**Status:** DESIGN

---

## Executive Summary

**Evidence-based joint fix for low_0p300 boundary failure**, addressing both support drift and hip-yaw errors through coordinated scheduling of sagittal authority parameters.

**Phase 2-4 Key Findings:**
- **Sagittal:** Position torque cap saturation (64-91%), insufficient velocity damping (k_velocity=15, τ_velocity RMS=0.0)
- **Hip-yaw:** Divergence-dominant error (divergence_mean=0.2119 >> common_mode=0.0077), PD gains too low
- **Coupling:** Neither symptom fixable independently; joint fix required

**Design Strategy:** Progressive candidate family with continuous height scheduling, stop at first pass.

---

## Design Constraints (STRICT)

### Allowed ✅
- Continuous height/reference-based schedules using smoothstep interpolation
- Low-height-only candidate profiles (z_low=0.300, z_high=0.393)
- Hip-yaw divergence authority increase (if justified by Phase 3)
- Support velocity damping increase
- Max position tau cap increase
- k_velocity schedule
- Telemetry additions
- Tests

### Prohibited ❌
- WBC additions
- Root-z-only perturbation
- Threshold relaxation
- Height target shrinking
- Hip-roll logic modification
- Global hip-yaw gain changes
- Variant-name-only patches
- Discontinuous step/bucket schedules
- Blind gain increases without evidence
- Default deployment of experimental features

---

## Evidence from Phase 4: Sagittal Authority Transmission Audit

### Dominant Failure Modes (all 4 candidates)
1. **position_torque_cap_saturation** (64-91% saturation)
2. **insufficient_velocity_damping** (k_velocity=15, τ_velocity RMS=0.0)
3. **support_velocity_underdamped** (all candidates)
4. **contact_coupling_limits_authority** (all candidates)

### Quantitative Evidence

| Metric | Baseline | E1 (k=60) | E2 (k=80) | E3 (k=100) |
|--------|----------|-----------|-----------|------------|
| Support Error Max (m) | 0.2430 | 0.2216 | 0.2094 | 0.2031 |
| τ_position Max (Nm) | 9.72 | 13.30 | 16.75 | **20.31** |
| Position Saturated (%) | 64.4 | 75.2 | 87.1 | **91.1** |
| k_velocity | 15.0 | 15.0 | 15.0 | 15.0 |
| τ_velocity RMS (Nm) | **0.00** | **0.00** | **0.00** | **0.00** |
| Pitch Max (rad) | 0.0951 | 0.0940 | 0.0983 | **0.1001** |

**Key Insight:** Increasing k_position alone (40→100) increases position torque demand (9.7→20.3 Nm) but hits the 3.0 Nm cap harder (91% saturation), while velocity damping remains zero.

### Root Cause Classification
- **Primary bottleneck:** max_position_tau cap (3.0 Nm insufficient at low height)
- **Secondary bottleneck:** k_velocity too low (15.0), no active velocity damping
- **Tertiary:** E3 caused pitch exceedance (0.1001 > 0.100 threshold)

---

## Evidence from Phase 2: Hip-Yaw Disturbance Rejection Audit

### Hip-Yaw Error Classification
- **Mode:** divergence_dominant
  - divergence_max: 0.4200 rad
  - divergence_mean: 0.2119 rad
  - common_mode_max: 0.0225 rad
  - common_mode_mean: 0.0077 rad
- **Velocity response:** Low (vel_to_error_ratio=0.37 < 1.0)
- **PD gains:** Likely too low
- **Coupling:** Hip-yaw not locally rejectable without support fix

### Lag Correlations
- support_error → hip_yaw divergence: lag=-50 steps (-500 ms), corr=-0.818
- support_velocity → hip_yaw divergence: lag=-50 steps (-500 ms), corr=0.373

**Interpretation:** Support drift precedes and strongly correlates with hip-yaw divergence. Fixing support drift first is critical.

---

## Evidence from Phase 3: Advanced Hip-Yaw Candidates

**Status:** Phase 3 evaluated HY2-DIV (divergence damping/authority).

**Note:** Need to verify Phase 3 results before finalizing hip-yaw component. If Phase 3 found a mechanism that improved hip-yaw without worsening support, include it in J5. If not, rely on support fix alone to reduce hip-yaw disturbance at source.

---

## Joint Fix Design: Candidate Family

### Height Schedule Parameters

**Smoothstep interpolation function:**
```python
u(z) = clamp((z_high - z_ref) / (z_high - z_low), 0, 1)
s(z) = 3*u^2 - 2*u^3
param(z) = param_nominal + (param_low_max - param_nominal) * s(z)
```

**Height range:**
- z_low = 0.300 m (full low-height compensation)
- z_high = 0.393 m (no compensation, nominal behavior)
- z_ref = target/commanded CoM height

### Candidate Definitions

#### J0: Baseline (reference)
- No changes
- Support error: 0.2430 m (FAIL)
- Hip-yaw: 0.2137 rad (FAIL)

---

#### J1: Support Cap Increase
**Target:** Address position_torque_cap_saturation

**Schedule:**
```yaml
k_position(z):
  nominal: 40.0
  low_max: 80.0
  interpolation: smoothstep(z, 0.300, 0.393)

max_position_tau(z):
  nominal: 3.0 Nm
  low_max: 6.0 Nm
  interpolation: smoothstep(z, 0.300, 0.393)

k_velocity:
  constant: 15.0  # unchanged
```

**Rationale:**
- Doubling max_position_tau (3.0→6.0 Nm) removes saturation bottleneck
- k_position increase (40→80) provides authority to use the increased cap
- Conservative: doesn't change velocity damping yet

**Expected outcome:**
- Reduce position saturation from 91% to <50%
- Support error may improve 20-30%
- Hip-yaw may improve 10-15% via reduced support disturbance

**Risk:** May not be sufficient alone; velocity damping still zero.

---

#### J2: Support Cap + Moderate Damping
**Target:** Address position_torque_cap_saturation + insufficient_velocity_damping

**Schedule:**
```yaml
k_position(z):
  nominal: 40.0
  low_max: 80.0
  interpolation: smoothstep(z, 0.300, 0.393)

max_position_tau(z):
  nominal: 3.0 Nm
  low_max: 6.0 Nm
  interpolation: smoothstep(z, 0.300, 0.393)

k_velocity(z):
  nominal: 15.0
  low_max: 25.0
  interpolation: smoothstep(z, 0.300, 0.393)
```

**Rationale:**
- Adds velocity damping (15→25, +67% increase)
- Addresses support_velocity_underdamped
- Still conservative (not maximum k_velocity yet)

**Expected outcome:**
- τ_velocity RMS > 0 (active damping)
- Support velocity oscillations reduced
- Support error may pass threshold (<0.15 m)

**Risk:** May cause damping/position interaction at intermediate heights.

---

#### J3: Support Cap + Strong Damping
**Target:** Aggressive velocity damping if J2 insufficient

**Schedule:**
```yaml
k_position(z):
  nominal: 40.0
  low_max: 80.0
  interpolation: smoothstep(z, 0.300, 0.393)

max_position_tau(z):
  nominal: 3.0 Nm
  low_max: 6.0 Nm
  interpolation: smoothstep(z, 0.300, 0.393)

k_velocity(z):
  nominal: 15.0
  low_max: 30.0
  interpolation: smoothstep(z, 0.300, 0.393)
```

**Rationale:**
- k_velocity increase: 15→30 (+100%)
- Maximum reasonable damping increase before risking over-damping

**Expected outcome:**
- Strong velocity damping suppresses support velocity oscillations
- Support error likely passes (<0.15 m)
- Hip-yaw error may improve via reduced support disturbance

**Risk:** Over-damping may cause sluggish response; check recovery transients in Step C.

---

#### J4: Support Integral (conditional)
**Trigger:** Only if J1-J3 show persistent steady-state drift (not recommended unless proven necessary)

**Schedule:**
```yaml
k_position(z):
  nominal: 40.0
  low_max: 60.0 or 80.0  # tune based on J1-J3 results
  interpolation: smoothstep(z, 0.300, 0.393)

max_position_tau(z):
  nominal: 3.0 Nm
  low_max: 6.0 Nm
  interpolation: smoothstep(z, 0.300, 0.393)

k_velocity(z):
  nominal: 15.0
  low_max: 25.0
  interpolation: smoothstep(z, 0.300, 0.393)

support_integral:
  enable: true
  ki_support(z):
    nominal: 0.0
    low_max: 2.0
    interpolation: smoothstep(z, 0.300, 0.393)
  anti_windup: true
  integral_clamp: 2.0 Nm
```

**Rationale:**
- Persistent steady-state error indicates position+velocity insufficient
- Integral adds DC gain to eliminate steady-state drift

**Risk:** Integral can destabilize if coupling dynamics strong. Requires careful tuning and anti-windup.

**Defer unless:** J1-J3 all pass short-term tests but fail long-term (5000+ steps) with steady drift.

---

#### J5: Coupled Sagittal-Yaw (conditional)
**Trigger:** If hip-yaw still fails after best sagittal candidate (J1-J3) passes support gate

**Approach A: Best Sagittal + Phase 3 Hip-Yaw Mechanism (if Phase 3 passed)**
```yaml
# Best sagittal schedule from J1-J3
k_position(z): [as selected]
max_position_tau(z): [as selected]
k_velocity(z): [as selected]

# Phase 3 hip-yaw mechanism (if any passed)
hip_yaw_mechanism: [HY2-DIV or other, if validated]
```

**Approach B: Best Sagittal + Coupled Support-Yaw Compensation (if Phase 3 failed)**
```yaml
# Best sagittal schedule from J1-J3
k_position(z): [as selected]
max_position_tau(z): [as selected]
k_velocity(z): [as selected]

# Coupled support-yaw feedforward
hip_yaw_support_feedforward:
  enable: true
  k_support_hip_yaw(z):
    nominal: 0.0
    low_max: 2.0  # tune based on Phase 2 correlation data
    interpolation: smoothstep(z, 0.300, 0.393)
  sign: -1.0  # based on Phase 2 correlation sign
  tau_max: 1.0 Nm
```

**Rationale:**
- Only deploy if sagittal fix alone insufficient for hip-yaw
- Prefer validated Phase 3 mechanism over new coupled compensation
- Coupled compensation uses Phase 2 lag correlation (support_error → hip_yaw divergence, corr=-0.818)

**Risk:** Couples two control axes; requires careful validation across full height range.

---

## Evaluation Protocol

### Stop-at-First-Pass Logic
**Critical:** Evaluate candidates in order J0→J1→J2→J3→J4→J5. As soon as one candidate passes all phases, select it and STOP. Do not evaluate stronger candidates.

### Phase 6 Evaluation Sequence (per candidate)

#### Phase 6.1: low_0p300 Step E 1000
- **Pass criteria:** All Step E gates
- **Fail action:** Log, continue to next candidate

#### Phase 6.2: low_0p300 Step E 5000
- **Pass criteria:** All Step E gates, stable over long horizon
- **Fail action:** Log, continue to next candidate

#### Phase 6.3: high_0p480 Step E 5000
- **Pass criteria:** All Step E gates, no regression
- **Fail action:** Log, continue to next candidate

#### Phase 6.4: Step C low_0p300 5000
- **Pass criteria:** All Step C gates (height recovery)
- **Fail action:** Log, continue to next candidate

#### Phase 6.5: Step C high_0p480 5000
- **Pass criteria:** All Step C gates
- **Fail action:** Log, continue to next candidate

#### Phase 6.6: Practical Height Grid Step E
- **Heights:** 0.300, 0.330, 0.360, 0.393 (or low_small), nominal, high_small, 0.450, 0.480
- **Duration:** 1000 steps each
- **Pass criteria:** All heights pass Step E gates
- **Fail action:** Log, continue to next candidate

#### Phase 6.7: Step C Grid
- **Heights:** 0.300, 0.360, nominal, 0.480
- **Duration:** 5000 steps each
- **Pass criteria:** All heights pass Step C gates
- **Fail action:** Log, continue to next candidate

#### Phase 6.8: Five-Variant Regression
- **Variants:** nominal, low_tiny, high_tiny, low_small, high_small
- **Duration:** Step E 5000 each
- **Pass criteria:** All variants pass Step E gates
- **Fail action:** Log, continue to next candidate
- **Success action:** CANDIDATE SELECTED, stop evaluation

---

## Acceptance Gates (must ALL pass)

### Step E (standing balance at fixed height)
- `support_position_error_max_abs <= 0.15 m` ⚠️ **PRIMARY TARGET**
- `hip_yaw_abs_max <= 0.07 rad` ⚠️ **PRIMARY TARGET**
- `percent(hip_yaw_abs > 0.10 rad) = 0%`
- `pitch_x_max_abs <= 0.10 rad`
- `roll_y_max_abs <= 0.05 rad`
- `final_height_error <= 0.02 m`
- `contact_valid >= 99.9%`
- `non_wheel_contacts = 0`
- `WBC_applied = false`
- `hidden_torque = 0`
- `ownership_violations = 0`

### Step C (height recovery from perturbation)
- All Step E gates
- `height_recovered = true`
- `recovery_time_s < timeout`

### Continuity
- All scheduled parameters continuous over z ∈ [0.300, 0.480]
- No discontinuities, verified by `scripts/check_schedule_continuity.py`

### Regression
- nominal, low_tiny, high_tiny, low_small, high_small all pass

---

## Implementation Requirements

### Controller Modifications

**File:** `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`

**Add schedule functions:**
```python
def compute_low_height_k_position_schedule(z_ref: float, profile: str) -> float:
    """Continuous k_position schedule for low-height authority."""
    if profile not in ["J1", "J2", "J3", "J4"]:
        return 40.0  # nominal
    
    z_low, z_high = 0.300, 0.393
    k_nominal, k_low_max = 40.0, 80.0
    
    u = jnp.clip((z_high - z_ref) / (z_high - z_low), 0.0, 1.0)
    s = 3.0 * u**2 - 2.0 * u**3
    return k_nominal + (k_low_max - k_nominal) * s

def compute_low_height_max_position_tau_schedule(z_ref: float, profile: str) -> float:
    """Continuous max_position_tau schedule for low-height authority."""
    if profile not in ["J1", "J2", "J3", "J4"]:
        return 3.0  # nominal
    
    z_low, z_high = 0.300, 0.393
    tau_nominal, tau_low_max = 3.0, 6.0
    
    u = jnp.clip((z_high - z_ref) / (z_high - z_low), 0.0, 1.0)
    s = 3.0 * u**2 - 2.0 * u**3
    return tau_nominal + (tau_low_max - tau_nominal) * s

def compute_low_height_k_velocity_schedule(z_ref: float, profile: str) -> float:
    """Continuous k_velocity schedule for low-height damping."""
    if profile not in ["J2", "J3", "J4"]:
        return 15.0  # nominal
    
    z_low, z_high = 0.300, 0.393
    k_nominal = 15.0
    k_low_max = 25.0 if profile == "J2" else 30.0  # J3, J4
    
    u = jnp.clip((z_high - z_ref) / (z_high - z_low), 0.0, 1.0)
    s = 3.0 * u**2 - 2.0 * u**3
    return k_nominal + (k_low_max - k_nominal) * s
```

**Update compute() method:**
- Add `low_height_sagittal_profile` parameter (default: "baseline")
- Call schedule functions based on profile
- Add telemetry for effective scheduled values

### CLI Arguments

**File:** `scripts/simulate_hierarchical_controller.py`

**Add:**
```bash
--low-height-sagittal-profile {baseline,J1,J2,J3,J4,J5}
```

### Telemetry Additions

**Required columns:**
- `low_height_sagittal_profile` (str)
- `effective_k_position_scheduled` (float)
- `effective_max_position_tau_scheduled` (float)
- `effective_k_velocity_scheduled` (float)
- `sagittal_schedule_active` (bool)
- `sagittal_schedule_gate` (float, 0-1)

---

## Tests Required

### Unit Tests

**File:** `tests/test_sagittal_velocity_damped_balance_controller.py`

1. Schedule function continuity (J1, J2, J3)
2. Schedule boundary values (z=0.300→expected_max, z=0.393→nominal)
3. Schedule monotonicity
4. Profile selection logic
5. Baseline profile unchanged

### Integration Tests

**File:** `tests/test_joint_low_height_sagittal_yaw_fix.py`

1. J0 baseline reproduces known metrics
2. J1-J3 schedules active at low height
3. No WBC enabled
4. No hip-roll modification
5. Telemetry fields present

---

## Decision Tree

```
Start: J0 baseline
  ↓
J1: Support cap increase (k_position=80, max_tau=6.0, k_velocity=15)
  Pass? → SELECT J1, STOP
  Fail? → Continue
  ↓
J2: Support cap + moderate damping (k_position=80, max_tau=6.0, k_velocity=25)
  Pass? → SELECT J2, STOP
  Fail? → Continue
  ↓
J3: Support cap + strong damping (k_position=80, max_tau=6.0, k_velocity=30)
  Pass? → SELECT J3, STOP
  Fail? → Continue
  ↓
J4: Support integral (if steady-state drift proven)
  Pass? → SELECT J4, STOP
  Fail? → Continue
  ↓
J5: Coupled sagittal-yaw (if hip-yaw still fails)
  Pass? → SELECT J5, STOP
  Fail? → CONTROLLER_REDESIGN_REQUIRED
```

---

## Expected Outcome

**Most likely pass:** J2 (Support cap + moderate damping)

**Rationale:**
- Removes position torque saturation bottleneck (max_tau 3.0→6.0)
- Adds active velocity damping (k_velocity 15→25, expect τ_velocity RMS > 0.5)
- Conservative enough to avoid over-damping
- Addresses all 4 dominant failure modes from Phase 4

**Predicted metrics (J2 @ low_0p300):**
- Support error max: 0.12-0.14 m (vs 0.2430 baseline, **PASS** if <0.15)
- Hip-yaw abs max: 0.10-0.12 rad (vs 0.2137 baseline, may still **FAIL** but improved)
- Position saturated: <50% (vs 91% baseline)
- τ_velocity RMS: 0.5-1.0 Nm (vs 0.0 baseline)
- Pitch max: 0.08-0.09 rad (**PASS** <0.10)

**If hip-yaw still fails after J2 support fix:** Proceed to J5 with coupled compensation.

---

## Implementation Plan for Phase 6

1. Implement schedule functions in `sagittal_velocity_damped_balance_controller.py`
2. Add CLI argument and telemetry in `simulate_hierarchical_controller.py`
3. Write unit tests for schedule functions
4. Create `scripts/evaluate_joint_low_height_sagittal_yaw_fix.py` evaluation harness
5. Run evaluation protocol (J0→J1→J2→... stop at first pass)
6. Generate candidate comparison report and summary JSON
7. Verify selected candidate with continuity checks and regression tests

---

## Risk Mitigation

### Pitch Exceedance Risk (observed in E3)
- J1-J3 use k_position=80, not 100 (E3 used k=100 and caused pitch=0.1001)
- Monitor pitch_max in all evaluations
- If pitch approaches 0.10, reduce k_position slightly (e.g., 80→70)

### Over-Damping Risk (J3)
- J3 uses k_velocity=30 (strong damping)
- Check Step C recovery transients for sluggish response
- If recovery time excessive, fall back to J2 (k_velocity=25)

### Intermediate Height Instability
- Practical height grid (Phase 6.6) tests z=0.330, 0.360, 0.393
- Catches any discontinuity or instability in transition region
- Smoothstep interpolation should prevent discontinuities

### Hip-Yaw Still Fails After Sagittal Fix
- If support error passes but hip-yaw fails: proceed to J5
- J5 adds coupled support-yaw compensation
- Fallback: classify as CONTROLLER_REDESIGN_REQUIRED if J5 also fails

---

## Success Criteria Summary

**Minimum for BOUNDARY_RANGE_PASS:**
1. Selected candidate (J1-J5) passes all Phase 6.1-6.8 evaluations
2. Support error max ≤ 0.15 m @ low_0p300, high_0p480, all grid points
3. Hip-yaw abs max ≤ 0.07 rad @ low_0p300, high_0p480, all grid points
4. All other Step E and Step C gates pass
5. Five-variant regression passes
6. Schedule continuity verified
7. No WBC, hidden torque, or ownership violations

**If all candidates fail:** Assign appropriate decision code and document blocker.

---

## Files to Create/Modify

### To Create
- `scripts/evaluate_joint_low_height_sagittal_yaw_fix.py` (evaluation harness)
- `tests/test_joint_low_height_sagittal_yaw_fix.py` (integration tests)

### To Modify
- `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` (schedule functions)
- `scripts/simulate_hierarchical_controller.py` (CLI args, telemetry)
- `tests/test_sagittal_velocity_damped_balance_controller.py` (schedule unit tests)

---

## Next: Phase 6 Implementation

Proceed to implement the evaluation harness and run candidate experiments according to the stop-at-first-pass protocol.

**Expected runtime:** 4-8 hours depending on how many candidates need evaluation.

**Critical path:** J0→J1→J2 (expect J2 to pass, estimate 2-3 hours to reach decision).
