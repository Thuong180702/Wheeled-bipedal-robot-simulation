# adaptive_support_centering_trim — Design

**Date:** 2026-06-14
**Based on:** Root-cause audit `SUPPORT_TRIM_TOO_WEAK_BUT_POSTURE_STABLE`
**Classification:** `ADAPTIVE_SUPPORT_CENTERING_TRIM_DESIGN_READY`

---

## Problem statement

The existing `support_centering_bias_trim` (T6J) bias trim:
- Saturated **93.2%** of the time at its **−0.35 Nm cap**
- Mean signed error still at **+0.0787 m** (positive, not centered)
- Positive area **172×** negative area (bias ratio 0.988)
- **Bang-bang** — jumps to full cap instead of proportional modulation
- Direction **100% correct** — the fix is not wrong, it is just not strong enough at high heights

Posture is stable throughout. Hip-yaw is bounded (max 0.30 rad). The only problem is that the trim cannot generate enough counteracting torque to pull the drift toward zero.

---

## Design philosophy

> Keep the stable posture controller intact. Improve support-centering only through a **slow, bounded, adaptive trim** with proportional gain, height-aware ceiling, near-zero relief, sign-reversal guard, oscillation guard, and hip-yaw safety gate.

Do NOT:
- Suppress pitch
- Suppress damping
- Flip torque signs
- Add WBC
- Modify `support_centering_bias_trim` itself
- Make this profile the default

---

## New profile name

```
adaptive_support_centering_trim
```

Registered in `JOINT_FIX_PROFILES` alongside `support_centering_bias_trim`, both as opt-in only.

---

## Base configuration

Inherits all settings from `SUPPORT_CENTERING_BIAS_TRIM`:
- All T6I phase-aware settings
- All recenter priority settings
- All APCR1nD tuned settings
- All arch fix settings
- All `t6j_bias_trim_*` settings (unchanged)

The adaptive trim is **additional and separate** from the existing `t6j_bias_trim` path. It is gated by the new `adaptive_bias_trim_enabled` flag which is only `True` in the adaptive profile.

---

## New dataclass fields in SagittalAuthoritySchedule

### Control logic

| Field | Default | Description |
|-------|---------|-------------|
| `adaptive_bias_trim_enabled` | `False` | Master enable for this feature |
| `adaptive_bias_window_steps` | `300` | Slow window for mean error |
| `adaptive_bias_fast_window_steps` | `100` | Fast window for early detection |
| `adaptive_bias_enter_threshold_m` | `0.035` | Enter trim when mean error exceeds this |
| `adaptive_bias_exit_threshold_m` | `0.012` | Exit/decay trim when mean error below this |
| `adaptive_bias_k_tau_per_m` | `5.0` | Proportional gain: Nm per meter of error |
| `adaptive_bias_max_tau_low_nm` | `0.35` | Max trim at low heights (≤ 0.38 m) |
| `adaptive_bias_max_tau_high_nm` | `0.50` | Max trim at high heights (≥ 0.48 m) |
| `adaptive_bias_max_tau_extreme_nm` | `0.55` | Max trim at extreme heights (≥ 0.52 m) |
| `adaptive_bias_height_low_m` | `0.38` | Height below which use low ceiling |
| `adaptive_bias_height_high_m` | `0.48` | Height above which use high ceiling |
| `adaptive_bias_height_extreme_m` | `0.52` | Height above which use extreme ceiling |
| `adaptive_bias_rate_nm_per_step` | `0.006` | Rate to ramp up trim |
| `adaptive_bias_fast_rate_nm_per_step` | `0.012` | Rate to ramp up in fast-response mode |
| `adaptive_bias_decay_rate_nm_per_step` | `0.018` | Rate to decay trim toward zero |
| `adaptive_bias_relief_hysteresis_m` | `0.005` | Near-zero deadband (exit + this value) |

### Safety gates

| Field | Default | Description |
|-------|---------|-------------|
| `adaptive_bias_disable_if_pitch_gt_deg` | `12.0` | Block trim if pitch beyond this |
| `adaptive_bias_disable_if_roll_gt_deg` | `5.0` | Block trim if roll beyond this |
| `adaptive_bias_disable_if_abs_error_gt_m` | `0.24` | Block trim if error too large |
| `adaptive_bias_disable_if_hip_yaw_gt_rad` | `0.25` | Block trim if hip_yaw_abs_max above this |
| `adaptive_bias_only_when_contact_stable` | `True` | Require valid contact |
| `adaptive_bias_only_when_upright` | `True` | Require pitch/roll in safe range |

### Zero-crossing guard

| Field | Default | Description |
|-------|---------|-------------|
| `adaptive_bias_zero_crossing_guard_enabled` | `True` | Enable oscillation guard |
| `adaptive_bias_zero_crossing_window_steps` | `500` | Window to count crossings |
| `adaptive_bias_zero_crossing_limit` | `8` | Block trim if crossings exceed this |
| `adaptive_bias_zero_crossing_max_scale` | `0.5` | Reduce max trim by this fraction when guard active |

### Proportional trim logic

The adaptive trim computes a **proportional target** instead of a bang-bang cap:

```
effective_error = mean_error - sign(mean_error) * exit_threshold
raw_tau_target = -k_tau_per_m * effective_error
raw_tau_target = clip(raw_tau_target, -max_tau_current, +max_tau_current)
```

Where `max_tau_current` is height-scheduled:
- at `z <= 0.38 m`: `adaptive_bias_max_tau_low_nm` (0.35 Nm — no regression from T6J)
- at `z >= 0.48 m`: `adaptive_bias_max_tau_high_nm` (0.50 Nm — more authority at high height)
- at `z >= 0.52 m`: `adaptive_bias_max_tau_extreme_nm` (0.55 Nm — even more at extreme height)
- linear interpolation between these points

### Near-zero relief

```
if abs(mean_error) <= exit_threshold:
    target_tau = 0.0
    # decay existing trim toward zero
else if abs(mean_error) <= exit_threshold + relief_hysteresis:
    target_tau = 0.0
    # hold existing trim (don't let it decay prematurely)
else:
    # normal proportional target
```

### Sign-reversal guard

- Maintain `_adaptive_bias_prev_trim_sign` state
- When `mean_error` sign flips:
  - Do NOT immediately apply opposite trim
  - First decay current trim toward zero at the normal decay rate
  - Only after 100+ steps of consistent opposite mean error, begin applying new sign
- This prevents the trim from fighting itself during zero crossings

### Zero-crossing oscillation guard

- Track `_adaptive_bias_zero_crossing_history`: circular buffer of signed error (length `zero_crossing_window_steps`)
- Count crossings per window
- If crossings exceed `zero_crossing_limit`:
  - Reduce `max_tau_current` by `zero_crossing_max_scale` factor
  - Increase decay rate
  - Log guard active in telemetry
- Reset counter after guard triggers 3 times in a row (prevents permanent reduction)

### Rate limiting

```
current_trim = _adaptive_bias_trim_tau
max_step = decay_rate if abs(target_tau) < abs(current_trim) else rate

trim_delta = target_tau - current_trim
if abs(trim_delta) > max_step:
    trim_delta = sign(trim_delta) * max_step

updated_trim = current_trim + trim_delta
updated_trim = clip(updated_trim, -max_tau_current, +max_tau_current)
```

Apply the trim to the hip joints' yaw support torque, same channel as T6J trim.

---

## New controller state (per instance)

```python
self._adaptive_bias_trim_tau: float = 0.0           # current applied trim torque
self._adaptive_bias_trim_target_tau: float = 0.0    # current target before rate limiting
self._adaptive_bias_slow_error_history: list[float] = []  # slow window
self._adaptive_bias_fast_error_history: list[float] = []  # fast window
self._adaptive_bias_zero_crossing_history: list[float] = []  # zero-crossing detection
self._adaptive_bias_crossing_count: int = 0          # crossings in current window
self._adaptive_bias_guard_trigger_count: int = 0    # consecutive guard triggers
self._adaptive_bias_prev_trim_sign: int = 0         # +1/-1/0
self._adaptive_bias_prev_error_sign: int = 0        # +1/-1/0
self._adaptive_bias_hold_steps: int = 0             # steps in sign-reversal hold
```

---

## Telemetry fields (new)

```
adaptive_bias_trim_enabled
adaptive_bias_trim_active
adaptive_bias_mean_error_m         # slow window mean
adaptive_bias_fast_mean_error_m     # fast window mean
adaptive_bias_effective_error_m     # mean - sign*exit_threshold
adaptive_bias_target_tau_nm         # target before rate limiting
adaptive_bias_tau_nm                # current applied trim (after rate limiting)
adaptive_bias_max_tau_current_nm    # current height-scheduled ceiling
adaptive_bias_height_scale          # 0.0=low, 0.5=mid, 1.0=high/extreme
adaptive_bias_rate_used_nm_per_step # current rate (normal or fast)
adaptive_bias_zero_crossing_count    # crossings in window
adaptive_bias_zero_crossing_guard_active
adaptive_bias_near_zero_relief_active
adaptive_bias_sign_reversal_blocked  # true when in sign-reversal hold
adaptive_bias_safety_gate_pass
adaptive_bias_block_reason           # "pitch_unsafe", "roll_unsafe", "hip_yaw_unsafe",
                                     # "contact_unstable", "abs_error_too_large",
                                     # "zero_crossing_guard", "ok"
adaptive_bias_expected_direction_correct
adaptive_bias_positive_area         # accumulated positive drift area (slow window)
adaptive_bias_negative_area         # accumulated negative drift area (slow window)
adaptive_bias_symmetry_ratio         # abs(pos-neg)/(pos+neg) for slow window
adaptive_bias_hip_yaw_gate_pass
adaptive_bias_hip_yaw_abs_max       # current hip_yaw_abs_max value at each step
```

---

## Placement in the control pipeline

Applied **after** the existing `t6j_bias_trim` (no change to T6J path). Both trims apply to hip-yaw support torque. The adaptive trim adds to the existing T6J trim value (or replaces it — to be validated).

```
...
tau_position_with_t6j_trim = tau_position + t6j_bias_trim_tau   # existing path, unchanged
tau_position_with_total_trim = tau_position_with_t6j_trim + adaptive_bias_trim_tau  # new
tau_position = clip(tau_position_with_total_trim, -effective_max, +effective_max)
```

The adaptive trim is applied as an additional signed torque on top of the T6J trim, targeting the same channel (hip yaw support torque). If both are enabled, they compose additively. If the adaptive profile is selected, T6J bias trim remains enabled in the profile config but should be kept for safety fallback — the adaptive trim is the primary mechanism.

---

## Interaction with T6J bias trim

Since `adaptive_support_centering_trim` inherits all profile settings from `SUPPORT_CENTERING_BIAS_TRIM`, it also inherits `t6j_bias_trim_enabled=True`. 

**Decision:** Keep both active. The T6J bang-bang trim provides a hard fallback; the adaptive proportional trim provides finer control. If both apply their trim, the effective trim = t6j_trim + adaptive_trim. With `adaptive_bias_max_tau_high_nm = 0.50`, at extreme heights the combined max ≈ −0.35 + (−0.50) = −0.85 Nm, which is still bounded (within typical position torque range of ±4–6 Nm).

**Alternative decision (if combined is too strong):** Disable `t6j_bias_trim_enabled` when adaptive is active, letting adaptive be the sole trim. This will be evaluated empirically. The design should support both modes via a `adaptive_bias_replace_t6j: bool = True` field.

For initial implementation: `adaptive_bias_replace_t6j = True` (adaptive replaces T6J, simpler and cleaner).

---

## Validation criteria (Phase 5–8)

| Criterion | Threshold |
|-----------|-----------|
| Max abs error | ≤ 0.20 m (not worse than support_centering + 0.02) |
| P2P | not worse than support_centering + 15% |
| Mean signed error | closer to zero than +0.0787 m |
| Positive % | lower than 89.4% |
| Outside ±0.08 | lower than 87.4% |
| Outside ±0.10 | lower than 71.3% |
| Outside ±0.15 | not worse than 14.1% + 2 pp |
| Saturation % (adaptive) | < 60% (vs T6J 93%) |
| Zero crossings | ≥ 10 (vs T6J 18 — may reduce, not critical) |
| Accumulation ratio | < 1.5 |
| No fall, WBC, hidden, ownership | True |
| Posture safe | pitch < 15 deg, roll < 8 deg |
| Hip-yaw safe | hip_yaw_abs_max < 0.35 rad |

---

## JSON output

`outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/adaptive_support_centering_trim_design.json`

---

## Classification

**`ADAPTIVE_SUPPORT_CENTERING_TRIM_DESIGN_READY`**

All design questions answered. No blocking issues. Proceed to Phase 3 implementation.

**Do not proceed if:**
- Root cause audit classified differently
- New telemetry reveals additional problems not covered here