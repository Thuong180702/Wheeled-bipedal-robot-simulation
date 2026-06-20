# early_zero_crossing_recenter — Design

**Classification:** `EARLY_ZERO_CROSSING_RECENTER_DESIGN_READY`

**Based on:** `zero_crossing_support_recenter`

**Created:** 2026-06-15

---

## 1. Problem Statement

The `zero_crossing_support_recenter` profile improved drift symmetry vs `adaptive_support_centering_trim` but has two issues:

### Issue 1: Exit Target Too Deep

The old ZC logic requires `e <= -0.02` to exit RECENTER_FROM_POSITIVE. Evidence:

- **0/22 episodes reached -0.02 exit target**
- **0/22 episodes reached -0.025 exit target**
- **Max negative drift: -0.0413 m (deeper than needed)**
- **P2P increased from 0.2241 m to 0.2395 m (+6.9%)**

The controller keeps pushing past zero but can't reach the deep -0.02 target, causing:
- Correction holds longer than needed
- P2P inflates from overshoot
- Oscillation is exaggerated

### Issue 2: Entry Threshold Too Late

Entry threshold of 0.08 m means:
- Significant drift accumulates before correction begins
- The "early" recenter principle is not achieved
- Drift has more time to grow larger

### Required Behavior (from task spec)

```
If support drift exceeds +0.05 m:
    command correction in the negative direction.
    hold correction until the drift reaches or crosses 0.
    once e <= 0, stop recenter correction and decay to zero.

If support drift exceeds -0.05 m:
    command correction in the positive direction.
    hold correction until the drift reaches or crosses 0.
    once e >= 0, stop recenter correction and decay to zero.

Do NOT require crossing to -0.02 or +0.02.
Do NOT use zc_cross_target_m = 0.02 anymore as an exit target.
Do NOT exit at ±0.025 before zero.
The exit condition is zero crossing.
```

---

## 2. Design: early_zero_crossing_recenter

### 2.1 Base Configuration

Inherit ALL settings from `ZERO_CROSSING_SUPPORT_RECENTER`:
- Same adaptive trim base
- Same safety gates
- Same telemetry structure

Override only the ZC-specific parameters.

### 2.2 Key Differences from Old ZC

| Parameter | Old ZC | Early ZC | Rationale |
|-----------|--------|----------|-----------|
| Entry threshold | 0.08 m | **0.05 m** | Earlier intervention |
| Exit condition | e <= -0.02 | **e <= 0** | Exit at zero, not past it |
| Opposite-side target | -0.02 | **None** | Don't push past zero |
| Cross target | 0.02 m | **Removed** | Not needed |
| Re-entry threshold | 0.08 m | **0.05 m** | Same as entry |
| Decay after crossing | Slower | **Faster** | Stop correcting sooner |

### 2.3 New State Machine

**States:**
1. `CENTER_IDLE` — Normal operation, no recentering needed
2. `RECENTER_FROM_POSITIVE` — Drift > +0.05 m, apply negative correction
3. `RECENTER_FROM_NEGATIVE` — Drift < -0.05 m, apply positive correction
4. `ZERO_CROSSED_DECAY` — Correction decays to zero after crossing
5. `SAFETY_DECAY` — Safety gate failed, decay correction to zero

**State IDs:**
```python
EZC_STATE_IDLE = 0
EZC_STATE_RECENTER_POSITIVE = 1
EZC_STATE_RECENTER_NEGATIVE = 2
EZC_STATE_ZERO_CROSSED_DECAY = 3
EZC_STATE_SAFETY_DECAY = 4
```

### 2.4 Configuration Parameters

```python
# Early Zero-crossing recenter (EZC) parameters
ezc_enabled: bool = True

# Entry/exit thresholds
ezc_enter_m: float = 0.05              # Enter when |e| > 0.05
ezc_exit_at_zero: bool = True          # Exit at zero crossing, not opposite side
ezc_zero_dwell_steps: int = 3          # Dwell at zero before decay starts
ezc_reentry_m: float = 0.05            # Re-enter when |e| > 0.05 again

# Hold duration constraints
ezc_min_hold_steps: int = 0           # No minimum hold (exit immediately at zero)
ezc_max_hold_steps: int = 500         # Force exit after 500 steps

# Torque authority
ezc_base_tau_nm: float = 0.18         # Base correction torque
ezc_max_tau_nm: float = 0.55          # Maximum correction torque
ezc_rate_nm_per_step: float = 0.012   # Rate limit: increase toward target
ezc_decay_nm_per_step: float = 0.025  # Decay rate: return to zero

# Error-proportional gain
ezc_error_gain_nm_per_m: float = 3.0   # Nm per meter of error

# Safety gates (absolute disable conditions)
ezc_disable_if_abs_error_gt_m: float = 0.25
ezc_disable_if_pitch_gt_deg: float = 12.0
ezc_disable_if_roll_gt_deg: float = 5.0
ezc_disable_if_hip_yaw_gt_rad: float = 0.25  # if telemetry exists
```

### 2.5 State Transition Logic

```
CENTER_IDLE:
  if e > +ezc_enter_m:
    state = RECENTER_FROM_POSITIVE
    direction = -1 (negative correction)
    hold_steps = 0
    enter_event++
    enter_error = e
    episode_id++
  elif e < -ezc_enter_m:
    state = RECENTER_FROM_NEGATIVE
    direction = +1 (positive correction)
    hold_steps = 0
    enter_event++
    enter_error = e
    episode_id++

RECENTER_FROM_POSITIVE:
  correction = direction * target_tau  # negative
  hold_steps++

  # Exit conditions:
  if e <= 0:
    # CROSSED ZERO - exit immediately
    state = ZERO_CROSSED_DECAY
    zero_crossed = True
    zero_cross_exit_event++
    exit_reason = "zero_cross"
    hold_steps = 0
  elif hold_steps >= ezc_max_hold_steps:
    # Max hold reached
    state = SAFETY_DECAY
    exit_event++
    exit_reason = "max_hold"
  elif safety_gate_fails:
    state = SAFETY_DECAY
    exit_event++
    exit_reason = "safety_gate"

RECENTER_FROM_NEGATIVE:
  correction = direction * target_tau  # positive
  hold_steps++

  # Exit conditions:
  if e >= 0:
    # CROSSED ZERO - exit immediately
    state = ZERO_CROSSED_DECAY
    zero_crossed = True
    zero_cross_exit_event++
    exit_reason = "zero_cross"
    hold_steps = 0
  elif hold_steps >= ezc_max_hold_steps:
    state = SAFETY_DECAY
    exit_event++
    exit_reason = "max_hold"
  elif safety_gate_fails:
    state = SAFETY_DECAY
    exit_event++
    exit_reason = "safety_gate"

ZERO_CROSSED_DECAY:
  # Decay correction toward zero smoothly
  correction = direction * max(0, abs(correction) - ezc_decay_nm_per_step)
  hold_steps++
  zero_dwell_steps++

  # When correction is near zero, return to IDLE
  if abs(correction) < 1e-9:
    state = CENTER_IDLE
    direction = 0
  elif hold_steps >= ezc_max_hold_steps:
    # Safety limit
    state = CENTER_IDLE
    direction = 0
    correction = 0

SAFETY_DECAY:
  # Decay correction toward zero if safety gate failed
  correction = direction * max(0, abs(correction) - ezc_decay_nm_per_step)

  if abs(correction) < 1e-9:
    state = CENTER_IDLE
    direction = 0
```

### 2.6 Correction Law

```python
# Target torque computation (while in RECENTER state)
target_tau = ezc_base_tau_nm + ezc_error_gain_nm_per_m * abs(e)
target_tau = min(target_tau, ezc_max_tau_nm)
target_tau = max(target_tau, ezc_base_tau_nm)

# Signed correction
correction = direction * target_tau

# Rate limiting
if correction > current_correction:
    delta = min(target_tau - current_correction, ezc_rate_nm_per_step)
else:
    delta = -min(current_correction - target_tau, ezc_rate_nm_per_step)
new_correction = current_correction + delta

# In ZERO_CROSSED_DECAY state:
# target_tau = 0
# correction = direction * max(0, abs(correction) - ezc_decay_nm_per_step)
```

### 2.7 Comparison: Old ZC vs Early ZC

| Aspect | zero_crossing_support_recenter | early_zero_crossing_recenter |
|--------|--------------------------------|------------------------------|
| Entry threshold | 0.08 m | **0.05 m** |
| Exit condition | e <= -0.02 (opposite side) | **e <= 0 (zero crossing)** |
| Opposite-side target | -0.02 m | None |
| P2P impact | Increases (+6.9%) | **Should decrease** |
| Decay after crossing | Slower | **Faster** |
| Symmetry | 50.5 | **Expected ~10-30** |
| Oscillation | Exaggerated | **Controlled** |

---

## 3. Telemetry Fields

```python
ezc_state_id                          # 0=IDLE, 1=POS, 2=NEG, 3=DECAY, 4=SAFETY
ezc_state_name                        # string name
ezc_active                            # True if not IDLE
ezc_direction                         # -1, 0, +1
ezc_enter_event                       # cumulative
ezc_zero_cross_exit_event             # cumulative exits at zero
ezc_safety_exit_event                 # cumulative exits due to safety
ezc_hold_steps                        # current hold duration
ezc_target_tau_nm                     # target before rate limit
ezc_tau_nm                            # rate-limited output
ezc_crossed_zero                      # True if episode crossed zero
ezc_zero_dwell_steps                  # dwell counter after zero crossing
ezc_safety_gate_pass                  # True/False
ezc_block_reason                      # "none", "pitch", "roll", "hip_yaw", "error"
ezc_episode_id                        # episode counter
ezc_episode_start_error               # error at episode start
ezc_episode_min_error                 # min error in episode (for positive)
ezc_episode_max_error                 # max error in episode (for negative)
ezc_expected_direction_correct        # sign(correction) == -sign(error)
ezc_exit_reason                        # "zero_cross", "max_hold", "safety_gate"
```

---

## 4. Implementation Plan

### 4.1 Files to Modify

1. **`wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`**
   - Add EZC fields to `SagittalAuthoritySchedule` dataclass
   - Add `EARLY_ZERO_CROSSING_RECENTER` profile constant
   - Add EZC state machine variables to controller state
   - Add EZC correction computation in `compute()` method
   - Add EZC telemetry fields

2. **`scripts/simulate_hierarchical_controller.py`**
   - Add `early_zero_crossing_recenter` to `SAGITTAL_AUTHORITY_PROFILES` registry
   - Add CLI option for `--vd-sagittal-authority-profile`

### 4.2 Files to Create

1. **`tests/test_early_zero_crossing_recenter.py`**
   - Unit tests for profile existence
   - Unit tests for state transitions
   - Unit tests for correction law
   - Unit tests for exit at zero (not -0.02)
   - Unit tests for safety gates

### 4.3 NOT Modified

- `zero_crossing_support_recenter` (unchanged)
- `adaptive_support_centering_trim` (unchanged)
- `support_centering_bias_trim` (unchanged)

---

## 5. Behavioral Contract

### 5.1 Entry Condition
When `|e| > 0.05`, controller enters RECENTER state within 1 step.

### 5.2 Exit Condition (KEY DIFFERENCE)
Exit RECENTER when:
- **Error crosses zero** (e <= 0 for positive recenter, e >= 0 for negative recenter) — **OR**
- Max hold steps (500) reached — **OR**
- Safety gate fails

**Do NOT wait for e <= -0.02 or e >= +0.02**

### 5.3 Decay After Zero Crossing
Once error crosses zero:
- Correction begins immediate decay
- No minimum hold required
- Correction decays at 0.025 Nm/step until near zero
- Return to IDLE when correction ≈ 0

### 5.4 Re-entry
Re-enter RECENTER only if |e| exceeds 0.05 again (after decaying to IDLE).

---

## 6. Expected Improvements

| Metric | Old ZC | Expected Early ZC |
|--------|--------|-------------------|
| P2P | 0.2395 m | **~0.22 m** (lower) |
| Max negative | -0.0413 m | **~0.03 m** (shallower) |
| Positive % | 86.4% | **~75%** (lower) |
| Negative % | 13.6% | **~25%** (higher) |
| Symmetry ratio | 50.5 | **~10-20** (better) |
| Zero crossings | 36 | **~40-50** (more) |

---

## 7. Decision

**`EARLY_ZERO_CROSSING_RECENTER_DESIGN_READY`**

The design addresses the root cause:
1. **Exit at zero** — not -0.02, prevents overshoot
2. **Earlier entry** — 0.05 vs 0.08, prevents drift accumulation
3. **Faster decay** — stop correcting sooner after crossing

Implementation can proceed with the state machine as specified.