# K2 JAX — Python ABS Trim Source Trace (Phase 1)

**Date:** 2026-06-28
**Source file:** `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`

---

## A. Activation and Profile Fields

All fields belong to the `SagittalControllerAuthoritySchedule` dataclass.
The K2 profile `K2_NOTCH_LOW_Q_V1` inherits all defaults and overrides `adaptive_bias_trim_enabled=True`.

### A.1 Profile Parameters (line: default value → K2 active value)

| # | Field | Type | Default | K2 Value | File:Line | Control-Affecting? | Notes |
|---|-------|------|---------|----------|-----------|-------------------|-------|
| 1 | `adaptive_bias_trim_enabled` | bool | False | **True** | :845 | YES — master gate | Only True in K2 profiles (T5, APCR1, APCR1N, APCR1ND). False in T5J. |
| 2 | `adaptive_bias_trim_replace_t6j` | bool | True | **True** | :846 | YES — path selection | True = adaptive replaces T6J bang-bang trim entirely |
| 3 | `adaptive_bias_window_steps` | int | 300 | **300** | :847 | YES — slow window size | Controls slow error history length |
| 4 | `adaptive_bias_fast_window_steps` | int | 100 | **100** | :848 | YES — fast window size | Controls fast error history length |
| 5 | `adaptive_bias_enter_threshold_m` | float | 0.035 | **0.035** | :849 | YES — hysteresis enter | Error must exceed this to activate proportional trim |
| 6 | `adaptive_bias_exit_threshold_m` | float | 0.012 | **0.012** | :850 | YES — hysteresis exit | Below this, target = 0 (near_zero) |
| 7 | `adaptive_bias_relief_hysteresis_m` | float | 0.005 | **0.005** | :851 | YES — hysteresis band | Within exit_th + relief, holds current trim |
| 8 | `adaptive_bias_k_tau_per_m` | float | 5.0 | **5.0** | :852 | YES — proportional gain | Nm of trim per meter of error beyond threshold |
| 9 | `adaptive_bias_max_tau_low_nm` | float | 0.35 | **0.35** | :853 | YES — ceiling low | Max trim at h <= 0.38m |
| 10 | `adaptive_bias_max_tau_high_nm` | float | 0.50 | **0.50** | :854 | YES — ceiling high | Max trim at h = 0.48m |
| 11 | `adaptive_bias_max_tau_extreme_nm` | float | 0.55 | **0.55** | :855 | YES — ceiling extreme | Max trim at h >= 0.52m |
| 12 | `adaptive_bias_height_low_m` | float | 0.38 | **0.38** | :856 | YES — schedule pt | Low height breakpoint |
| 13 | `adaptive_bias_height_high_m` | float | 0.48 | **0.48** | :857 | YES — schedule pt | High height breakpoint |
| 14 | `adaptive_bias_height_extreme_m` | float | 0.52 | **0.52** | :858 | YES — schedule pt | Extreme height breakpoint |
| 15 | `adaptive_bias_rate_nm_per_step` | float | 0.006 | **0.006** | :859 | YES — rate limit | Max increase per step (0.01s) = 0.6 Nm/s |
| 16 | `adaptive_bias_fast_rate_nm_per_step` | float | 0.012 | **0.012** | :860 | **UNUSED** | Defined but never referenced in compute() |
| 17 | `adaptive_bias_decay_rate_nm_per_step` | float | 0.018 | **0.018** | :861 | YES — decay rate | Max decrease per step = 1.8 Nm/s (faster decay) |
| 18 | `adaptive_bias_only_when_upright` | bool | True | **True** | :862 | YES — safety gate | Disables trim when pitch > 12° or roll > 5° |
| 19 | `adaptive_bias_only_when_contact_stable` | bool | True | **True** | :863 | YES — safety gate | Disables trim when contact_valid = False |
| 20 | `adaptive_bias_disable_if_pitch_gt_deg` | float | 12.0 | **12.0** | :864 | YES — safety gate | Pitch threshold for upright check |
| 21 | `adaptive_bias_disable_if_roll_gt_deg` | float | 5.0 | **5.0** | :865 | YES — safety gate | Roll threshold for upright check |
| 22 | `adaptive_bias_disable_if_abs_error_gt_m` | float | 0.24 | **0.24** | :866 | YES — safety gate | Max absolute sagittal error |
| 23 | `adaptive_bias_disable_if_hip_yaw_gt_rad` | float | 0.25 | **0.25** | :867 | YES — safety gate | Max hip yaw divergence |
| 24 | `adaptive_bias_zero_crossing_guard_enabled` | bool | True | **True** | :868 | YES — guard gate | Enables zero-crossing guard |
| 25 | `adaptive_bias_zero_crossing_window_steps` | int | 500 | **500** | :869 | YES — ZC window | Window for counting zero crossings |
| 26 | `adaptive_bias_zero_crossing_limit` | int | 8 | **8** | :870 | YES — ZC threshold | Crossings above this trigger guard |
| 27 | `adaptive_bias_zero_crossing_max_scale` | float | 0.5 | **0.5** | :871 | YES — guard scale | Reduces max_tau by 50% when guard active |
| 28 | `adaptive_bias_sign_reversal_hold_steps` | int | 100 | **100** | :872 | YES — hold duration | Steps to hold trim after sign reversal (1.0s at 100Hz) |

### A.2 Gate Dependencies

```
adaptive_bias_trim_enabled = True (K2 profile)
    ├── contact_ok = (not only_when_contact_stable) OR contact_valid  → True requires contact_valid
    ├── upright_ok = pitch ≤ 12° AND roll ≤ 5°  (only_when_upright=True)
    ├── hip_yaw_ok = max_hip_yaw_error ≤ 0.25 rad
    ├── abs_error_ok = |sagittal_error| ≤ 0.24 m
    └── safety_pass = contact_ok AND upright_ok AND hip_yaw_ok AND abs_error_ok
        └── If safety_pass is False → trim NOT applied (trim_to_apply = 0 in effect)
```

### A.3 APCR1ND/Profile Inheritance

`K2_NOTCH_LOW_Q_V1` = APCR1ND profile. All T5/APCR1/APCR1N/APCR1ND profiles set `adaptive_bias_trim_enabled=True`.

ZC-related profiles (`enable_zero_crossing_recenter=True`) and EZC profiles (`enable_enhanced_zero_crossing_recenter=True`) supplement ABS trim with additional hold-through-zero logic — these are separate mechanisms, not part of core ABS trim.

---

## B. Input Signals

| Signal | Source | Python Location | Notes |
|--------|--------|----------------|-------|
| `sagittal_position_error_m` | `compute()` arg | :4373, :5574 | Signed sagittal displacement from equilibrium [m]. Positive = forward drift. |
| `com_z_m` | `compute()` arg | :5594 | CoM height for height-scheduled max_tau |
| `pitch_x_rad` | `compute()` arg | :5658 | Body pitch for safety gate |
| `roll_y_rad` | `compute()` arg | :5659 | Body roll for safety gate |
| `contact_valid` | `compute()` arg | :5660 | MuJoCo contact force validity |
| `hip_yaw_abs_max_tracking` | sagittal controller state | :5671 | Running max hip yaw divergence (tracked externally) |

**Sign convention**: `sagittal_position_error_m > 0` means forward drift (robot shifted forward relative to support center). Trim is NEGATIVE (pushes wheels backward) to correct.

---

## C. Ring Buffer / History

### C.1 Data Structures

```python
self._adaptive_bias_slow_error_history = []   # list[float], oldest-first, max 300 entries
self._adaptive_bias_fast_error_history = []   # list[float], oldest-first, max 100 entries
self._adaptive_bias_zero_crossing_history = [] # list[float], oldest-first, max 500 entries
```

### C.2 Update Timing (per step)

```
1. APPEND:  slow_history.append(signed_error)
            fast_history.append(signed_error)
            zc_history.append(signed_error)
2. TRUNCATE: if len(slow) > 300 → pop(0)
             if len(fast) > 100 → pop(0)
             if len(zc) > 500 → pop(0)
3. COMPUTE MEANS: from the (possibly truncated) lists
4. COMPUTE ZC: from zc_history
5. COMPUTE TRIM: using means
6. APPLY TRIM: to tau_position
```

**Critical**: Current sample IS included in means before computing trim. This means step N's trim uses step N's error.

### C.3 Mean Computation

```python
slow_n = max(1, len(slow_history))
fast_n = max(1, len(fast_history))
mean_err = sum(slow_history) / slow_n        # slow mean — used for proportional target
fast_mean_err = sum(fast_history) / fast_n   # fast mean — diagnostic only (NOT used for control)
```

Fast mean is **diagnostic-only** — it is computed and logged but never used to compute trim. The slow mean drives all control decisions.

### C.4 Zero-Crossing Computation

```python
zc_window = zc_history  # up to 500 entries
zc_count = 0
for i in range(1, len(zc_window)):
    if (zc_window[i-1] < 0) != (zc_window[i] < 0):
        zc_count += 1
```

Walks chronologically (oldest→newest), counts sign flips. ZC guard activates when `zc_count > 8` in 500-entry window.

### C.5 Partial Fill Behavior

- Steps 1-99: fast and slow histories contain identical entries (all appended, none popped)
- Steps 100-299: fast history capped at 100; slow continues growing
- Steps 300+: slow history capped at 300; fast remains at 100
- ZC history grows to 500 by step 500, then capped

During partial fill (count < window), means use all available entries. ZC counts all sign changes in available entries.

---

## D. Trim Formula

### D.1 Complete Formula

```
// Height scheduling
t = interpolate(com_z, [z_low, z_high, z_extreme])    // ∈ [0, 2]
max_tau_current = interpolate(t, [max_low, max_high, max_extreme])

// Zero-crossing guard
if zc_count > zc_limit:
    guard_scale = zc_max_scale (= 0.5)
    guard_trigger += 1
    if guard_trigger >= 3: guard_trigger = 0
else:
    guard_trigger = 0
max_tau_g = max_tau_current * guard_scale

// Sign-reversal hold
sign_err = sign(mean_err)
if sign_err != prev_err_sign AND sign_err != 0:
    hold_steps = sign_reversal_hold_steps (= 100)
    prev_err_sign = sign_err
elif hold_steps > 0:
    hold_steps -= 1
    prev_err_sign = sign_err

// Hysteresis target
near_zero = |mean_err| <= exit_th (= 0.012 m)
in_hysteresis = |mean_err| <= exit_th + relief (= 0.017 m)
sign_rev_blocked = (hold_steps > 0) AND err_sign_changed (at THIS step)

if near_zero:
    raw_target = 0.0                           // inside exit threshold
elif sign_rev_blocked:
    raw_target = 0.0                           // sign reversal hold
elif in_hysteresis:
    raw_target = current_trim_tau              // hold current
else:
    eff_err = mean_err - sign_err * exit_th    // error beyond threshold
    raw_target = -k_tau * eff_err             // proportional with deadband

// Clamp to guard-adjusted ceiling
clipped_target = clamp(raw_target, -max_tau_g, max_tau_g)

// Asymmetric rate limiting
is_decay = |clipped_target| < |current_trim|
rate = decay_rate (= 0.018) if is_decay else ramp_rate (= 0.006)
trim_delta = clamp(clipped_target - current_trim, -rate, rate)
new_trim = clamp(current_trim + trim_delta, -max_tau_g, max_tau_g)

// Safety gate
if safety_pass:
    tau_position += new_trim      // apply to position torque
    tau_position = clamp(tau_position, -effective_max_pos_tau, effective_max_pos_tau)
active = (not near_zero) AND (not sign_rev_blocked) AND |new_trim| > 1e-9
```

### D.2 Sign Convention

- Positive `sagittal_position_error_m` (forward drift) → negative trim (pushes back)
- Negative `sagittal_position_error_m` (backward drift) → positive trim (pushes forward)
- The proportional term uses `-k_tau * eff_err`, achieving this correction

### D.3 Clamp Order

1. `raw_target` → clipped to `[-max_tau_g, max_tau_g]`
2. Rate-limited delta → clipped to `[-rate, rate]`
3. Result → clipped to `[-max_tau_g, max_tau_g]`
4. Final `tau_position + trim` → clipped to `[-effective_max_pos_tau, effective_max_pos_tau]`

### D.4 Interaction with Position Cap

ABS trim is added to `tau_position` BEFORE the final position cap clip. This means:
- Trim is constrained by `effective_max_position_tau` (4.0 Nm nominal, up to 7.0 Nm with boost)
- During APCR1ND boost, trim + kpos * error is capped at boosted level
- Trim does NOT affect APCR1ND band selection directly
- Trim does NOT affect position cap boost computation

---

## E. Timing

### E.1 Per-Step Sequence (within `compute()`)

```
1. sagittal_position_error_m is computed (from support center vs COM)
2. ABS trim updates histories (append + truncate)
3. ABS trim computes means from updated histories
4. ABS trim computes target, rate-limits, applies safety gate
5. ABS trim is added to tau_position (before position cap clip)
6. tau_position goes through position cap, torque budget, final position clip
7. tau_position enters sagittal assembly → wheel torque
8. Diagnostics are written
```

### E.2 State Update vs Torque Computation

- Histories are updated BEFORE trim computation (current sample included)
- `_adaptive_bias_trim_tau` is updated INLINE during compute()
- State is NOT separately flushed — it's updated as part of compute

### E.3 Reset Behavior

- `__init__`: All histories = [], trim_tau = 0.0, hold_steps = 0, prev_err_sign = 0
- No explicit reset method — re-initialization happens via controller reconstruction
- On first step: histories have 1 entry, means = that entry, trim starts from zero

---

## F. State

### F.1 Python State Fields

| Field | Type | Init | File:Line | Used In | Control-Affecting? |
|-------|------|------|-----------|---------|-------------------|
| `_adaptive_bias_trim_tau` | float | 0.0 | :4226 | trim output | **YES** — applied to tau_position |
| `_adaptive_bias_trim_target_tau` | float | 0.0 | :4227 | rate limiting | **YES** — target before rate-limit |
| `_adaptive_bias_slow_error_history` | list[float] | [] | :4228 | slow mean | **YES** — mean_err drives target |
| `_adaptive_bias_fast_error_history` | list[float] | [] | :4229 | fast mean | DIAGNOSTIC ONLY |
| `_adaptive_bias_zero_crossing_history` | list[float] | [] | :4230 | ZC count | **YES** — gates guard |
| `_adaptive_bias_crossing_count` | int | 0 | :4231 | counts | REDUNDANT (zc_count recomputed) |
| `_adaptive_bias_guard_trigger_count` | int | 0 | :4232 | guard cycling | **YES** — resets at 3 |
| `_adaptive_bias_prev_trim_sign` | int | 0 | :4233 | unused | DIAGNOSTIC ONLY |
| `_adaptive_bias_prev_error_sign` | int | 0 | :4234 | sign-reversal | **YES** — gating |
| `_adaptive_bias_hold_steps` | int | 0 | :4235 | sign-reversal hold | **YES** — blocking |
| `_adaptive_bias_positive_area` | float | 0.0 | :4236 | pos drift accumulation | DIAGNOSTIC ONLY |
| `_adaptive_bias_negative_area` | float | 0.0 | :4237 | neg drift accumulation | DIAGNOSTIC ONLY |

### F.2 Telemetry Variables (compute() local)

| Variable | Source | Initialized At |
|----------|--------|---------------|
| `adaptive_bias_trim_enabled` | `bool(self.authority_schedule.adaptive_bias_trim_enabled)` | :4510 |
| `adaptive_bias_trim_active` | `bool(...)` | :4511, set at :5778 |
| `adaptive_bias_mean_error_m` | `mean_err` from slow history | :5588 |
| `adaptive_bias_fast_mean_error_m` | `fast_mean_err` from fast history | :5591 |
| `adaptive_bias_effective_error_m` | `mean_err - sign * exit_th` or 0 | :5736 |
| `adaptive_bias_target_tau_nm` | `clipped_target` | :5740 |
| `adaptive_bias_tau_nm` | `updated_trim_a` (after rate-limit) | :5756 |
| `adaptive_bias_max_tau_current_nm` | height-scheduled max | :5619 |
| `adaptive_bias_height_scale` | height fraction ∈ [0,1] | :5618 |
| `adaptive_bias_rate_used_nm_per_step` | decay or ramp rate | :5746 |
| `adaptive_bias_zero_crossing_count` | zc_count | :5631 |
| `adaptive_bias_zero_crossing_guard_active` | zc_guard_active | :5641 |
| `adaptive_bias_near_zero_relief_active` | near_zero | :5720 |
| `adaptive_bias_sign_reversal_blocked` | sign_rev_blocked | :5715 |
| `adaptive_bias_safety_gate_pass` | safety_pass | :5682 |
| `adaptive_bias_block_reason` | string enum | :5685-5695 |
| `adaptive_bias_expected_direction_correct` | sign check | :5759-5764 |
| `adaptive_bias_positive_area` | sum positive entries | :5650 |
| `adaptive_bias_negative_area` | abs sum negative entries | :5651 |
| `adaptive_bias_symmetry_ratio` | |pos-neg|/total | :5655 |
| `adaptive_bias_hip_yaw_gate_pass` | hy_ok | :5675 |
| `adaptive_bias_hip_yaw_abs_max` | hy_val | :5676 |

---

## Source-of-Truth Table

| Python scalar | file:line | formula/update | active for K2? | control-affecting? | required JAX destination |
|---|---|---|---|---|---|
| `adaptive_bias_trim_enabled` | :845 | True from profile | YES | YES (gate) | params or profile constant |
| `adaptive_bias_trim_replace_t6j` | :846 | True = replaces T6J | YES | YES (path) | params or profile constant |
| `adaptive_bias_window_steps` | :847 | 300 | YES | YES (slow window) | `_ABS_SLOW_WINDOW` |
| `adaptive_bias_fast_window_steps` | :848 | 100 | YES | DIAGNOSTIC | `_ABS_FAST_WINDOW` |
| `adaptive_bias_enter_threshold_m` | :849 | 0.035 | YES | YES (hysteresis) | profile constant |
| `adaptive_bias_exit_threshold_m` | :850 | 0.012 | YES | YES (near_zero) | profile constant |
| `adaptive_bias_relief_hysteresis_m` | :851 | 0.005 | YES | YES (hold band) | profile constant |
| `adaptive_bias_k_tau_per_m` | :852 | 5.0 | YES | YES (gain) | profile constant |
| `adaptive_bias_max_tau_low_nm` | :853 | 0.35 | YES | YES (ceiling) | profile constant |
| `adaptive_bias_max_tau_high_nm` | :854 | 0.50 | YES | YES (ceiling) | profile constant |
| `adaptive_bias_max_tau_extreme_nm` | :855 | 0.55 | YES | YES (ceiling) | profile constant |
| `adaptive_bias_height_low_m` | :856 | 0.38 | YES | YES (schedule) | profile constant |
| `adaptive_bias_height_high_m` | :857 | 0.48 | YES | YES (schedule) | profile constant |
| `adaptive_bias_height_extreme_m` | :858 | 0.52 | YES | YES (schedule) | profile constant |
| `adaptive_bias_rate_nm_per_step` | :859 | 0.006 | YES | YES (rate limit) | profile constant |
| `adaptive_bias_fast_rate_nm_per_step` | :860 | 0.012 | YES | **UNUSED** | **NOT REQUIRED** |
| `adaptive_bias_decay_rate_nm_per_step` | :861 | 0.018 | YES | YES (decay rate) | profile constant |
| `adaptive_bias_only_when_upright` | :862 | True | YES | YES (safety) | profile constant |
| `adaptive_bias_only_when_contact_stable` | :863 | True | YES | YES (safety) | **→ JAX hardcodes True** |
| `adaptive_bias_disable_if_pitch_gt_deg` | :864 | 12.0 | YES | YES (safety) | profile constant |
| `adaptive_bias_disable_if_roll_gt_deg` | :865 | 5.0 | YES | YES (safety) | profile constant |
| `adaptive_bias_disable_if_abs_error_gt_m` | :866 | 0.24 | YES | YES (safety) | profile constant |
| `adaptive_bias_disable_if_hip_yaw_gt_rad` | :867 | 0.25 | YES | YES (safety) | profile constant |
| `adaptive_bias_zero_crossing_guard_enabled` | :868 | True | YES | YES (guard) | profile constant |
| `adaptive_bias_zero_crossing_window_steps` | :869 | 500 | YES | YES (ZC window) | **→ JAX uses 300** |
| `adaptive_bias_zero_crossing_limit` | :870 | 8 | YES | YES (ZC threshold) | profile constant |
| `adaptive_bias_zero_crossing_max_scale` | :871 | 0.5 | YES | YES (guard scale) | profile constant |
| `adaptive_bias_sign_reversal_hold_steps` | :872 | 100 | YES | YES (hold) | profile constant |
| `_adaptive_bias_trim_tau` | :4226 | rate-limited trim | YES | **YES** | `_ABS_TRIM_TAU` |
| `_adaptive_bias_trim_target_tau` | :4227 | target before rate-limit | YES | YES (rate limit) | DIAGNOSTIC |
| `_adaptive_bias_slow_error_history` | :4228 | list, max 300 | YES | **YES** (slow mean) | ring buffer [28:328] |
| `_adaptive_bias_fast_error_history` | :4229 | list, max 100 | YES | DIAGNOSTIC | `_ABS_FAST_SUM` (diagnostic) |
| `_adaptive_bias_zero_crossing_history` | :4230 | list, max 500 | YES | **YES** (ZC count) | **→ JAX uses ring buffer (max 300)** |
| `_adaptive_bias_guard_trigger_count` | :4232 | 0; +=1 when guard; =0 when not; =0 after 3 | YES | **YES** (cycling) | `_ABS_GUARD_TRIGGER` |
| `_adaptive_bias_prev_error_sign` | :4234 | -1/0/1; updated on err_sign_changed or hold>0 | YES | **YES** (sign-reversal) | `_ABS_PREV_ERR_SIGN` |
| `_adaptive_bias_hold_steps` | :4235 | HOLD_STEPS when sign changes; decrement each step | YES | **YES** (blocking) | `_ABS_HOLD_STEPS` |
| `_adaptive_bias_positive_area` | :4236 | sum of positive entries in slow window | YES | DIAGNOSTIC | NOT REQUIRED |
| `_adaptive_bias_negative_area` | :4237 | abs sum of negative entries in slow window | YES | DIAGNOSTIC | NOT REQUIRED |
| `sagittal_position_error_m` | :5574 | input to compute() | YES | **YES** (signal) | `_I_SAG_POS_ERR` |
| `com_z_m` | :5594 | CoM height | YES | YES (schedule) | `_I_COM_Z` |
| `contact_valid` | :5660 | MuJoCo contact force valid | YES | YES (safety) | **→ JAX hardcodes True** |

---

## Critical Findings

1. **`adaptive_bias_fast_rate_nm_per_step` (0.012) is defined but NEVER USED.** Only `adaptive_bias_rate_nm_per_step` (ramp up) and `adaptive_bias_decay_rate_nm_per_step` (ramp down) are active.

2. **Fast mean is DIAGNOSTIC ONLY.** The slow mean drives all control decisions. Fast mean is computed and logged but never affects trim.

3. **Zero-crossing window: Python uses 500 entries, JAX uses 300 (ring buffer limit).** This is a confirmed discrepancy that affects ZC guard activation in long scenarios (>300 steps).

4. **Contact safety gate: Python checks `contact_valid`, JAX hardcodes `True`.** During push-induced contact breaks, Python disables trim while JAX continues. This is the likely root cause of the 0.98 Nm push_fwd diff.

5. **Hip-yaw safety gate: Python uses `hip_yaw_abs_max_tracking` (running max), JAX computes instantaneous.** These could diverge during transients.

6. **`guard_trigger` reset: Python resets to 0 inside the True branch at >=3; JAX only increments without the >=3 reset.** This is a diagnostic difference only — guard_scale is computed from zc_guard_active, not guard_trigger.

7. **Current sample is included in mean before computing trim** (both Python and JAX do this correctly).
