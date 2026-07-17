# zero_crossing_support_recenter — Design

**Classification:** `ZERO_CROSSING_RECENTER_DESIGN_READY`

**Based on:** `adaptive_support_centering_trim`

**Created:** 2026-06-14

---

## 1. Problem Statement

The `adaptive_support_centering_trim` profile is stable and prevents saturation, but does **NOT** satisfy the required centering principle:

- Drift is overwhelmingly positive (92.2% of samples)
- Symmetry ratio = 102.9 (positive area 403.6 vs negative area 3.9)
- Mean signed drift = +0.080 m (far from zero)
- No negative violation episodes (e < -0.08) ever detected
- Only 26 zero crossings in 5000 steps

The robot drifts mostly on one side. It does NOT oscillate around zero.

## 2. Required Principle

If support drift goes too far positive (e > +0.08), the controller **MUST** command the wheels/support in the negative direction and **HOLD** that correction until the drift returns to zero or slightly crosses into the negative side.

If support drift goes too far negative (e < -0.08), the controller **MUST** command the opposite direction and **HOLD** until the drift returns to zero or slightly crosses into the positive side.

This is a **hysteresis / zero-crossing recenter controller**, NOT a simple proportional bias trim.

## 3. Design: zero_crossing_support_recenter

### 3.1 Base Configuration

Inherit ALL settings from `ADAPTIVE_SUPPORT_CENTERING_TRIM`:
- APCR1nD direct support recenter
- APCR1nD tuned parameters
- T6I phase-aware release
- T6F architecture fix (budget cap raise)
- Adaptive centering bias trim (existing, but may be supplemented)

### 3.2 New State Machine: Zero-Crossing Recenter (ZC)

**States:**
1. `CENTER_IDLE` — Normal operation, no recentering needed
2. `RECENTER_FROM_POSITIVE` — Drift > +zc_enter_m, apply negative correction
3. `RECENTER_FROM_NEGATIVE` — Drift < -zc_enter_m, apply positive correction
4. `HOLD_THROUGH_ZERO` — Correction held to ensure crossing
5. `SAFETY_DECAY` — Safety gate failed, decay correction to zero

**State IDs:**
```python
ZC_STATE_IDLE = 0
ZC_STATE_RECENTER_POSITIVE = 1
ZC_STATE_RECENTER_NEGATIVE = 2
ZC_STATE_HOLD_THROUGH_ZERO = 3
ZC_STATE_SAFETY_DECAY = 4
```

### 3.3 Configuration Parameters

```python
# Zero-crossing recenter (ZC) parameters
enable_zero_crossing_recenter: bool = False

# Entry/exit thresholds
zc_enter_m: float = 0.08           # Enter recenter when |e| > this
zc_exit_m: float = 0.025           # Exit recenter when |e| <= this (with dwell)
zc_cross_target_m: float = 0.02    # Target overshoot into opposite side
zc_near_zero_band_m: float = 0.03   # Error considered "near zero"

# Hold duration constraints
zc_min_hold_steps: int = 50        # Minimum hold before considering release
zc_max_hold_steps: int = 600       # Force exit after this many steps

# Torque authority
zc_base_tau_nm: float = 0.20      # Base correction torque
zc_max_tau_nm: float = 0.65        # Maximum correction torque
zc_rate_nm_per_step: float = 0.01  # Rate limit: increase toward target
zc_decay_nm_per_step: float = 0.02 # Decay rate: return to zero

# Error-proportional gain
zc_error_gain_nm_per_m: float = 3.0  # Nm per meter of error

# Optional velocity damping
zc_velocity_gain: float = 0.0      # Damping term (0.0 = disabled)

# Safety gates (absolute disable conditions)
zc_disable_if_abs_error_gt_m: float = 0.25
zc_disable_if_pitch_gt_deg: float = 12.0
zc_disable_if_roll_gt_deg: float = 5.0
zc_disable_if_hip_yaw_gt_rad: float = 0.25  # if telemetry exists

# Dwell time for exit (converging signal)
zc_dwell_steps_for_exit: int = 30
zc_dwell_target_within_m: float = 0.015  # Must be within this of exit threshold
```

### 3.4 State Transition Logic

```
CENTER_IDLE:
  if e > +zc_enter_m:
    state = RECENTER_FROM_POSITIVE
    direction = -1 (negative correction)
    hold_steps = 0
    enter_event++
    enter_error = e
    episode_id++
  if e < -zc_enter_m:
    state = RECENTER_FROM_NEGATIVE
    direction = +1 (positive correction)
    hold_steps = 0
    enter_event++
    enter_error = e
    episode_id++

RECENTER_FROM_POSITIVE:
  correction = direction * target_tau  # negative
  hold_steps++

  # Continue until:
  if e <= -zc_cross_target_m:
    # Crossed to opposite side
    state = HOLD_THROUGH_ZERO
    crossed_zero = True
    cross_target_reached = True
    hold_steps = 0
    exit_event++
  elif hold_steps >= zc_max_hold_steps:
    # Max hold reached
    state = SAFETY_DECAY
    exit_event++
    exit_reason = "max_hold"
  elif e <= zc_near_zero_band_m and hold_steps >= zc_min_hold_steps:
    # In near-zero band after min hold — start dwell
    dwell_steps++
    if dwell_steps >= zc_dwell_steps_for_exit:
      # Dwell satisfied
      state = SAFETY_DECAY
      exit_event++
      exit_reason = "dwell_exit"
  elif safety_gate_fails:
    state = SAFETY_DECAY
    exit_event++
    exit_reason = "safety_gate"

RECENTER_FROM_NEGATIVE:
  symmetric to RECENTER_FROM_POSITIVE

HOLD_THROUGH_ZERO:
  # Continue reduced correction to ensure crossing
  correction = direction * min(target_tau, zc_base_tau_nm)
  hold_steps++

  if abs(e) <= zc_exit_m and hold_steps >= zc_min_hold_steps:
    state = SAFETY_DECAY
    exit_event++
    exit_reason = "hold_through_zero"
  elif hold_steps >= zc_max_hold_steps:
    state = SAFETY_DECAY
    exit_event++
    exit_reason = "max_hold"

SAFETY_DECAY:
  # Decay correction toward zero
  correction = direction * max(0, abs(correction) - zc_decay_nm_per_step)
  direction = 0 when abs(correction) < 1e-9

  if abs(correction) < 1e-9:
    state = CENTER_IDLE
```

### 3.5 Correction Law

```python
# Target torque computation
target_tau = zc_base_tau_nm + zc_error_gain_nm_per_m * abs(e)
target_tau = min(target_tau, zc_max_tau_nm)
target_tau = max(target_tau, zc_base_tau_nm)

# Signed correction
correction = direction * target_tau

# Rate limiting
rate = zc_rate_nm_per_step if correction > current_correction else zc_decay_nm_per_step
delta = target_tau - current_correction
if abs(delta) > rate:
    delta = rate if delta > 0 else -rate
new_correction = current_correction + delta
```

### 3.6 Safety Gate

```python
def zc_safety_gate_pass(e, pitch, roll, hip_yaw=None):
    if abs(e) > zc_disable_if_abs_error_gt_m:
        return False
    if abs(pitch) > zc_disable_if_pitch_gt_deg * DEG2RAD:
        return False
    if abs(roll) > zc_disable_if_roll_gt_deg * DEG2RAD:
        return False
    if hip_yaw is not None and abs(hip_yaw) > zc_disable_if_hip_yaw_gt_rad:
        return False
    return True
```

### 3.7 Interaction with Existing Adaptive Trim

The ZC recenter operates **alongside** the existing `adaptive_bias_trim`:
- `adaptive_bias_trim` provides proportional centering with 0.50 Nm max
- ZC recenter provides aggressive hold-through-zero recentering with 0.65 Nm max
- Both add to tau_position (additive correction)
- ZC recenter can be temporarily dominant when drift exceeds threshold

**Alternative:** ZC recenter can **replace** adaptive_bias_trim if `zc_replace_adaptive = True`

### 3.8 Telemetry Fields

```python
zc_state_id                          # 0=IDLE, 1=POS, 2=NEG, 3=HOLD, 4=DECAY
zc_state_name                        # string name
zc_active                            # True if not IDLE
zc_direction                         # -1, 0, +1
zc_enter_event                       # cumulative
zc_exit_event                        # cumulative
zc_hold_steps                        # current hold duration
zc_target_tau_nm                     # target before rate limit
zc_tau_nm                            # rate-limited output
zc_crossed_zero                      # True if episode crossed zero
zc_cross_target_reached              # True if crossed to opposite side
zc_near_zero_dwell_steps             # dwell counter
zc_safety_gate_pass                  # True/False
zc_block_reason                      # "none", "pitch", "roll", "hip_yaw", "error"
zc_episode_id                        # episode counter
zc_episode_start_error               # error at episode start
zc_episode_min_error                 # min error in episode (for positive)
zc_episode_max_error                 # max error in episode (for negative)
zc_episode_crossed_zero_bool         # True if crossed zero in this episode
zc_expected_direction_correct        # sign(correction) == -sign(error)
```

---

## 4. Implementation Plan

### 4.1 Files to Modify

1. **`wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`**
   - Add new fields to `SagittalAuthoritySchedule` dataclass
   - Add `ZERO_CROSSING_SUPPORT_RECENTER` profile constant
   - Add ZC state machine variables to controller state
   - Add ZC correction computation in `compute` method
   - Add ZC telemetry logging

2. **`scripts/simulate_hierarchical_controller.py`**
   - Add `zero_crossing_support_recenter` to `SAGITTAL_AUTHORITY_PROFILES` registry
   - Add CLI argument for profile selection

### 4.2 Files to Create

1. **`tests/test_zero_crossing_support_recenter.py`**
   - Unit tests for profile existence
   - Unit tests for state transitions
   - Unit tests for correction law
   - Unit tests for safety gates
   - Unit tests for telemetry fields

### 4.3 NOT Modified

- `adaptive_support_centering_trim` (unchanged)
- `support_centering_bias_trim` (unchanged)
- `phase_aware_authority_release` (unchanged)
- `emergency_budget_cap_raise` (unchanged)

---

## 5. Behavioral Contract

### 5.1 Entry Condition
When `|e| > 0.08`, controller enters RECENTER state within 1 step.

### 5.2 Hold-Through-Zero Guarantee
Once in RECENTER state, controller:
- MUST NOT release correction just because error is decreasing
- MUST hold correction until error crosses to opposite side OR dwell satisfied
- MUST NOT allow correction sign to flip while still in RECENTER

### 5.3 Exit Condition
Exit RECENTER when:
- Error crosses to opposite side (e.g., e < -0.02 for positive recenter) — **OR**
- Dwell satisfied: error inside near-zero band for 30+ consecutive steps — **OR**
- Max hold steps (600) reached — **OR**
- Safety gate fails

### 5.4 Max Authority
Max ZC correction torque: 0.65 Nm (additive with existing tau_position)

---

## 6. Comparison: Adaptive Trim vs ZC Recenter

| Aspect | adaptive_support_centering_trim | zero_crossing_support_recenter |
|--------|--------------------------------|-------------------------------|
| Approach | Proportional bias trim | Hysteresis state machine |
| Entry | Above threshold | Above threshold |
| Exit | Hysteresis (near zero) | Hold through zero OR dwell |
| Sign reversal | 100-step hold before flip | Hold through zero |
| Max torque | 0.50 Nm | 0.65 Nm |
| Symmetry | No guarantee | Enforced by state machine |
| Oscillation | May not cross zero | Must cross zero |
| Complexity | Simple | Medium |

---

## 7. Validation Plan

### Phase 5: 500-step screening
- No fall
- Posture stable
- Zero crossings increase from 26/5000 to meaningful count
- Positive % decreases below 80%
- Negative % increases above 15%

### Phase 6: Staged validation
- 1200, 2000, 5000 steps
- Compare drift statistics vs adaptive
- Ensure P2P bounded
- Ensure posture safe

### Phase 7: Height ladder
- 2000 steps across height variants
- No regression at lower heights

---

## 8. Decision

**`ZERO_CROSSING_RECENTER_DESIGN_READY`**

The design is complete. Implementation can proceed with:
1. New profile constant in controller
2. State machine with 5 states
3. ZC recenter correction added to tau_position
4. Telemetry for debugging and validation
5. Safety gates to prevent destabilization

The key innovation is the **hold-through-zero** behavior: the state machine explicitly prevents early correction release and enforces that drift must cross to the opposite side before correction is reduced.