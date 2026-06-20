# tau_pitch Code Path Audit

**Date:** 2026-06-15
**Scope:** Phase 2 of pitch-bias compensation investigation
**Source:** `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`

---

## 1. Code Path Trace

### 1.1 Pitch measurement input

`SagittalVelocityDampedBalanceController.compute(...)` receives `pitch_x_rad`
and `pitch_rate_x_rad_s` directly from the simulator. There is **no
intermediate filter** applied to pitch inside this controller.

```python
def compute(...):
    ...
    pitch_x_rad: float,
    pitch_rate_x_rad_s: float,
    ...
```

The pitch sign convention is +x = forward lean (verified by Phase 1
audit: pitch correlates +1.000 with `pitch_x_error_rad` whose reference
is 0).

### 1.2 Pitch reference

There is **no explicit pitch reference**: the controller treats
`pitch_x_rad` itself as the pitch error. This corresponds to a hard-coded
reference of 0. Telemetry confirms: `pitch_x_ref_rad = -0.0` for all
profiles, all 5000 steps.

### 1.3 tau_pitch raw computation (line 2832, 2843)

```python
tau_pitch_raw_orig = self.kp_pitch * pitch_x_rad
...
tau_pitch_raw = self.kp_pitch * pitch_x_rad
tau_pitch_scheduled = tau_pitch_raw * effective_pitch_scale
if effective_pitch_tau_cap is None:
    tau_pitch = tau_pitch_scheduled
else:
    tau_pitch = float(jnp.clip(tau_pitch_scheduled, -effective_pitch_tau_cap, effective_pitch_tau_cap))
```

- `kp_pitch = 50.0` (default, line 2325)
- `effective_pitch_scale = 1.0` (telemetry confirms; this is the
  high-height schedule scale, defaults to 1.0 unless schedule active)
- `effective_pitch_tau_cap` is None for all current profiles (telemetry
  shows `nan` which the script reads as None)

So `tau_pitch = 50.0 * pitch_x_rad` is **a clean linear function of
pitch**, no hidden offset.

### 1.4 Pitch suppression / blend (lines 2825–2918)

There are two optional suppression paths:

1. **Hard suppression (`apc_hysteresis_pitch_suppress_in_recenter`)**:
   forces `tau_pitch = 0.0` during APC RECENTER state. **Not enabled**
   in any of the four profiles audited.
2. **Conditional blend (`apc_pitch_blend_enabled`)**: scales `tau_pitch`
   by 0.0 / 0.25 / 0.5 / 1.0 depending on error band, with safety gates.
   Enabled in EZC profile family but only fires inside RECENTER state.

Telemetry verification (5000-step ezc_v2): `apcr1m_pitch_blend_active = 0`
and `apcr1l_pitch_suppress_active = 0` for all 4999 steps. Neither
mechanism is firing — `tau_pitch` is exactly `kp_pitch * pitch_x_rad`
in this scenario.

### 1.5 Pitch rate term (line 2927)

```python
tau_pitch_rate = self.kd_pitch * pitch_rate_x_rad_s
```

- `kd_pitch = 10.0` (default, line 2326)
- Treated as a **separate term** in the wheel torque sum. NOT bundled
  into `tau_pitch`.

This is important: `tau_pitch` and `tau_pitch_rate` are decoupled in
the diagnostics dict. Phase 1 audit confirmed `tau_pitch_rate` mean ≈ 0.

### 1.6 Final wheel summation (line 5679)

```python
tau_common_unclipped = (
    tau_pitch + tau_pitch_rate + tau_sagittal_velocity +
    tau_support_velocity + tau_position + tau_cp + tau_com_vy
)
tau_common_unclipped += recenter_tau_clipped       # APCR1l
tau_common_unclipped += hyst_tau_clipped           # F2 hysteresis
tau_common_unclipped += bias_tau_clipped           # G1 bias-cancel (off in our profiles)
tau_common_unclipped += apc_tau_clipped            # APC active pitch crossing
```

`tau_pitch` enters the sum directly. There is **no asymmetric handling**:
positive and negative tau_pitch values are added the same way.

### 1.7 Position bound clip (line 3043)

```python
tau_position_lower_bound = -self.max_tau_wheel - tau_balance_before_position
tau_position_upper_bound =  self.max_tau_wheel - tau_balance_before_position
tau_position = jnp.clip(tau_position_before_clip, lower, upper)
```

`tau_balance_before_position = tau_pitch + tau_pitch_rate + ...
+ 0.5*(tau_wheel_vel_left + tau_wheel_vel_right)` (line 3032).

When `tau_pitch ≈ +3.4 Nm` and `max_tau_wheel = 7 Nm`, the lower bound
becomes `-7 - 3.4 = -10.4`, but `tau_position_lower_bound` telemetry
shows -7.0 in the file. That's because `enable_torque_budget_aware_position`
is False for these profiles (verified: telemetry column shows constant
`enable_torque_budget_aware_position = False`). The fallback path uses
`±max_position_tau` directly, which `arch_fix` raises to 7.0 at high
heights.

### 1.8 No DC bias detected in code

Searching the controller for additive constants on `tau_pitch`:
- `self.kp_pitch * pitch_x_rad` — pure proportional, no offset.
- No `+ pitch_bias`, no `+ pitch_offset`, no feedforward term.
- No asymmetric clipping (`jnp.clip` with symmetric bounds).
- `tau_pitch` is not multiplied by anything sign-asymmetric.

The controller **does not inject any DC bias into tau_pitch**.

---

## 2. Where the Bias Comes From

Combined with the Phase 1 audit, the chain is:

```
robot dynamics  →  pitch settles around +3 to +5 deg
                     (forward equilibrium at h=0.480 m)
                            ↓
              kp_pitch * pitch_x_rad = +3.3 Nm (correct gain)
                            ↓
            tau_position pulls -3.7 Nm (saturated 31% of time)
                            ↓
              final wheel torque ≈ 0  (stalemate)
                            ↓
                  drift remains positive
```

The robot is **stuck at a forward-pitched equilibrium**. The pitch
controller correctly pushes back, the position controller saturates
trying to recenter, and the two cancel — leaving negligible final
wheel torque to actually move the wheels backward and recenter.

The forward-pitched equilibrium is independent of the controller; it
is determined by:

- Hip-pitch and knee posture references at height 0.480 m.
- CoM position relative to wheel-contact line in that posture.
- Static feedforward terms (legs forward angle).

---

## 3. tau_pitch Combination With Other Terms

Combined with EZC, ZC, adaptive trim, and APC:

| Term                          | Sign during +drift | Magnitude (5000-step ezc_v2) |
|-------------------------------|--------------------|------------------------------|
| `tau_pitch`                   | +                  | mean +3.380 Nm               |
| `tau_pitch_rate`              | small              | ~0                           |
| `tau_position`                | -                  | mean -3.737 Nm (saturated 31%)|
| `tau_support_velocity`        | small              | ~0                           |
| `tau_sagittal_velocity`       | small              | ~0                           |
| `tau_wheel_velocity_avg`      | +                  | mean +0.394 Nm               |
| `adaptive_bias_tau_nm`        | -                  | small (V2 disables old ZC)   |
| `ezc_tau_nm`                  | -                  | active 510 steps             |
| `apc_tau_clipped`             | -                  | only in RECENTER             |
| **Sum (final wheel torque)**  | small +            | mean +0.012 Nm               |

`tau_pitch` is the **dominant positive contributor**. Removing even a
small DC component (~0.2–0.3 Nm) shifts the equilibrium toward more
negative final wheel torque during +drift, which is exactly what we
need.

---

## 4. Symmetry Check

Code is symmetric:

- `kp_pitch * pitch_x_rad` works for both signs.
- `jnp.clip(..., -cap, +cap)` is symmetric.
- No conditional branches that treat positive pitch differently from
  negative pitch in the `tau_pitch` computation.

The asymmetry observed in telemetry (pitch>0 in 83% of steps,
pitch<0 in 3%) is **purely from the dynamics**, not the code.

---

## 5. Conclusions

1. `tau_pitch` is computed as a pure proportional term with no DC offset,
   no asymmetric gain, no sign error.
2. The forward-pitched equilibrium is the source of the apparent bias.
3. The most surgical, opt-in correction is to **estimate the slow DC
   component of tau_pitch during quiescent posture windows**, then
   subtract a bounded fraction of it from `tau_pitch` upstream of the
   wheel summation.
4. The compensation should:
   - Operate on `tau_pitch` directly (after `kp_pitch * pitch_x_rad`,
     before suppression/blend logic at line 2862).
   - Be gated by posture/contact safety (only when stable).
   - Be bounded (≤ 0.6 Nm) and rate-limited.
   - Slowly converge toward the moving average of `tau_pitch` measured
     during low-pitch windows.
   - Not interact with EZC, ZC, adaptive trim, APC, or any other path.

---

## 6. Insertion Point for Compensation

The compensation should be inserted **between line 2850 (post-clip
`tau_pitch`) and line 2862 (`tau_pitch_before_blend = tau_pitch`)**,
specifically:

```python
# After line 2850 (tau_pitch is clipped/scheduled)
tau_pitch_before_bias_comp = tau_pitch       # NEW telemetry

# NEW: compute bias estimate during quiescent windows
# NEW: rate-limit toward bounded compensation
# NEW: apply: tau_pitch -= pitch_bias_comp_tau

tau_pitch_after_bias_comp = tau_pitch         # NEW telemetry
# Existing line 2862 follows
tau_pitch_before_blend = tau_pitch
```

This keeps the existing blend/suppress paths intact and acts purely
as a slow-DC remover on the post-schedule, post-clip `tau_pitch`.

---

## 7. Files / lines referenced

- `sagittal_velocity_damped_balance_controller.py:2325-2326` — kp_pitch, kd_pitch defaults
- `sagittal_velocity_damped_balance_controller.py:2832-2850` — tau_pitch raw + clip
- `sagittal_velocity_damped_balance_controller.py:2862-2918` — pitch blend
- `sagittal_velocity_damped_balance_controller.py:2927` — tau_pitch_rate
- `sagittal_velocity_damped_balance_controller.py:3030-3050` — torque-budget bounds
- `sagittal_velocity_damped_balance_controller.py:5679-5697` — final summation
- `sagittal_velocity_damped_balance_controller.py:580-599` — EZC dataclass fields
