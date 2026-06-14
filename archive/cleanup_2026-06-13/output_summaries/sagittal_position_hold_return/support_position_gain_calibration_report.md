# Support Position Gain Calibration Report

**Date:** 2026-05-30
**Status:** FALLBACK FINAL GATE PASS — TRANSIENT PEAK EXCEEDS LIMITS
**Do NOT proceed to Step C until reviewed.**

---

## Changes Made

**Controller modification:** Added term-level position authority limiting to `SagittalVelocityDampedBalanceController`

**File:** `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`

**New parameter:** `max_position_tau` (Nm)

**Clipping formula:**
```python
tau_position_raw = -k_position * sagittal_position_error_m
tau_position = clip(tau_position_raw, -max_position_tau, +max_position_tau)
```

**Rationale:** Position hold is secondary to pitch balance. Clipping `tau_position` before summing into `tau_total` prevents the position-return term from saturating wheel torque on its own.

**New telemetry fields:** `tau_position_raw`, `tau_position_clipped`, `max_position_tau`, `tau_position_saturation_flag`

**New CLI argument:** `--vd-max-position-tau`

---

## Candidates Tested

| Candidate | k_position | max_tau | Steps | Max error | Final error | Gate +-0.15m | Gate final <=0.10m | Survived |
|-----------|-----------|---------|-------|-----------|-------------|-------------|-------------------|---------|
| A | 10.0 | 2.0 | 1000 | 0.234 m | 0.234 m | FAIL | FAIL | Yes |
| B | 15.0 | 2.5 | 1000 | 0.165 m | 0.165 m | FAIL | FAIL | Yes |
| C | 20.0 | 3.0 | 1000 | 0.133 m | 0.128 m | PASS | FAIL | Yes |
| **C** | **20.0** | **3.0** | **5000** | **0.595 m** | **0.053 m** | **FAIL** | **PASS** | **Yes** |

---

### Candidate A: k_position=10.0, max_position_tau=2.0 (1000 steps)

- support_position_error: min=-0.007, max=0.234, final=0.234 m
- pitch_x: 0.0 to 2.93 deg
- com_z: 0.4038 to 0.4086 m
- tau_position_raw: -2.34 to 0.067 Nm, RMS=1.699 Nm
- tau_position_clipped: -2.0 to 0.067 Nm (saturation: 132 steps, 13.2%)
- **Classification:** position_gain_too_low
- **Reason:** Steady-state formula: (50/10) x 0.051 = 0.255 m. Position-return authority too weak.

---

### Candidate B: k_position=15.0, max_position_tau=2.5 (1000 steps)

- support_position_error: min=-0.007, max=0.165, final=0.165 m
- pitch_x: 0.0 to 3.26 deg
- com_z: 0.4038 to 0.4086 m
- tau_position_raw: -2.473 to 0.103 Nm, RMS=1.793 Nm
- tau_position_clipped: -2.473 to 0.103 Nm (saturation: 0 steps, 0.0%)
- **Classification:** position_gain_too_low
- **Reason:** Steady-state formula: (50/15) x 0.057 = 0.190 m. Still too weak.

---

### Candidate C: k_position=20.0, max_position_tau=3.0

#### 1000-step run

- support_position_error: min=-0.007, max=0.133, final=0.128 m
- pitch_x: 0.0 to 3.57 deg
- com_z: 0.4038 to 0.4086 m
- tau_position_raw: -2.656 to 0.139 Nm, RMS=1.853 Nm
- tau_position_clipped: -2.656 to 0.139 Nm (saturation: 0 steps, 0.0%)
- Gate +-0.15m: **PASS** | Gate final <=0.10m: FAIL
- Note: final error equals max — still drifting at step 1000, not yet at steady-state

#### 5000-step run

- support_position_error: min=-0.007, max=**0.595**, final=**0.053** m
- Steady-state (last 1000 steps): mean=0.0527 m, std=0.0 m
- pitch_x: 0.0 to 7.19 deg | pitch_error mean (last 1000): 1.207 deg
- roll: -2.6 to 0.33 deg | yaw: -11.99 to 4.31 deg
- com_z: 0.3623 to 0.4086 m
- tau_position_raw: -11.9 to 0.139 Nm, RMS=2.805 Nm
- tau_position_clipped: -3.0 to 0.139 Nm, RMS=1.596 Nm
- Saturation: 564 steps (11.3%)

**Transient peak at step 1360:**
- Position error: 0.595 m
- Pitch error: 5.64 deg
- COM height: 0.365 m (dropped 0.04 m from nominal 0.404 m)
- tau_position_raw: -11.9 Nm (clipped to -3.0 Nm)

**Position error by window:**

| Steps | Max error | Final error |
|-------|-----------|-------------|
| 0-1000 | 0.133 m | 0.128 m |
| 1000-2000 | **0.595 m** | 0.039 m |
| 2000-3000 | 0.053 m | 0.053 m |
| 3000-4000 | 0.053 m | 0.053 m |
| 4000-5000 | 0.053 m | 0.053 m |

**Steady-state formula verification:**
```
pitch_ss = 1.207 deg = 0.0211 rad
predicted = (kp_pitch / k_position) x pitch_ss = (50 / 20) x 0.0211 = 0.0525 m
measured  = 0.0527 m
Match: YES
```

**Acceptance gates:**

| Gate | Threshold | Result | Status |
|------|-----------|--------|--------|
| Preferred max error | <=0.10 m | 0.595 m | FAIL |
| Fallback max error | <=0.15 m | 0.595 m | FAIL |
| Preferred final drift | <=0.05 m | 0.053 m | FAIL |
| **Fallback final drift** | **<=0.10 m** | **0.053 m** | **PASS** |
| Completed 5000 steps | Yes | Yes | **PASS** |

---

## Height Variant Regression (Candidate C)

### High +5cm (500 steps)

- support_position_error: max=0.224 m, final=0.220 m
- pitch_x: 0.0 to 5.20 deg
- com_z: 0.4539 to 0.4623 m
- tau_position_raw: -4.482 to 0.178 Nm
- tau_position_clipped: -3.0 to 0.178 Nm (saturation: 392 steps, 78.4%)
- Survived: Yes | Gate +-0.15m: **FAIL**
- Note: High saturation (78.4%) indicates max_position_tau is too low for this height. Robot survived but position hold is weak.

### Low -5cm (500 steps)

- support_position_error: max=0.083 m, final=0.041 m
- pitch_x: 0.0 to 2.25 deg
- com_z: 0.3518 to 0.3538 m
- tau_position_raw: -1.665 to 0.083 Nm
- tau_position_clipped: -1.665 to 0.083 Nm (saturation: 0 steps, 0.0%)
- Survived: Yes | Gate +-0.15m: **PASS**

---

## State Separation Verification

| Check | Status |
|-------|--------|
| Support-center position error used for position hold | CONFIRMED |
| COM error logged separately (diagnostic only) | CONFIRMED |
| Pitch reference enabled (pitch_x_error = pitch_x - pitch_ref) | CONFIRMED |
| Pitch stays near zero in steady-state | CONFIRMED |

---

## Verification

| Check | Status |
|-------|--------|
| No WBC changes | CONFIRMED |
| No E0b/E0c/E0d reintroduced | CONFIRMED |
| Torque ownership unchanged | CONFIRMED |
| Sagittal controllers mutually exclusive | CONFIRMED |
| balance-core mode only | CONFIRMED |
| velocity-damped controller only | CONFIRMED |
| kp_cp = 0.0 throughout | CONFIRMED |
| Term-level position authority limiting | CONFIRMED |

---

## Fundamental Limitation

The position-return approach cannot simultaneously achieve tight position hold during large pitch transients AND preserve pitch balance authority.

- Increasing k_position: reduces steady-state drift but worsens transient peaks
- Increasing max_position_tau: risks saturating wheel torque and preventing pitch balance
- Decreasing k_position: worsens steady-state drift (Candidates A and B)

The transient peak is the cost of prioritizing pitch balance over position hold during disturbances.

---

## Options for User

**Option 1 (Recommended): Accept Fallback Gate**

Accept Candidate C as passing the fallback final-drift gate (<=0.10 m). Document transient peak as known limitation. Steady-state performance is excellent and formula-verified. Robot recovers from disturbances without falling.

**Option 2: Pitch-Dependent Position Authority**

Reduce `max_position_tau` during large pitch excursions to preserve pitch balance authority while allowing tighter position hold during nominal operation. Requires additional logic and testing.

**Option 3: Redesign Position-Hold Strategy**

Replace position-return term with velocity-based hold, pitch-reference adjustment, or separate position-hold mode. Significant redesign required.

---

## Step C Recommendation

**DO NOT PROCEED** to Step C until the user reviews this report and decides which option to take.

Transient peak (0.595 m) exceeds both max-error gates. User must decide whether to accept fallback gate or pursue further improvements.
