# Support-Position Outer-Loop Pitch Reference — Phase B Design

**Profile name:** `support_position_outer_loop_pitch_ref`
**Base profile:** `height_scheduled_pitch_equilibrium_trim` (Phase A, committed `befb874`)
**Status classification:** `OUTER_LOOP_DESIGN_READY`
**Date:** 2026-06-16

---

## 1. Motivation

Phase A (`height_scheduled_pitch_equilibrium_trim`) fixed the **static** height-dependent
pitch-equilibrium mismatch: each fixed height settles at a distinct forward-pitched
equilibrium, so a per-height `pitch_ref` offset (piecewise-linear schedule) centers the
signed support drift. It passed the 10-height fixed-height ladder and the hip-yaw / leg-yaw
audit.

But the scheduled offset is a **constant per height**. It cannot react to:

- real-time support drift that accumulates within an episode,
- random / changing height commands (Step C),
- external push disturbances (Step D).

Phase B adds a **dynamic outer loop**: a bounded real-time correction to `pitch_ref`
driven by the live support-position error, layered **on top of** the frozen Phase A
schedule. The schedule supplies the height-dependent DC operating point; the outer loop
supplies the slow centering feedback around it.

This is **not** a replacement for Phase A and **not** a new inner controller. It only
nudges the existing `pitch_ref` setpoint. Pitch gain, damping, WBC state, and HY2-DIV
state are untouched.

---

## 2. Control law

```text
pitch_ref_total_deg =
      pitch_ref_offset_scheduled_deg          # Phase A, height-scheduled, constant per height
    + dynamic_support_pitch_ref_deg           # Phase B, per-step outer loop

dynamic_support_pitch_ref_deg =
      Kp_support * support_error_m             # PD: proportional
    + Kd_support * support_error_rate_m_s      # PD: derivative (damping)
    + Ki_support * integral_support_error_m_s  # I: DISABLED initially (Ki = 0)
```

Then, in order:

1. **Deadband**: if `abs(support_error) < deadband_m`, dynamic term contribution from the
   proportional path is zeroed (no nudging while already centered).
2. **Saturation**: clamp `dynamic_support_pitch_ref_deg` to `[-theta_ref_max, +theta_ref_max]`.
3. **Rate limit**: limit per-step change to `theta_ref_rate_limit_deg_per_step`.
4. **Low-pass**: `out = (1-alpha)*prev + alpha*target` with `theta_ref_lowpass_alpha`.

The smoothed, limited dynamic term is added to the scheduled base, and the sum replaces
`vd_pitch_ref_offset_deg` at the existing application site:

```text
pitch_x_ref = pitch_x_eq + radians(pitch_ref_total_deg)
pitch_x_error = body_pitch_x - pitch_x_ref
```

(See `scripts/simulate_hierarchical_controller.py:4732`. Today the RHS uses the static
`vd_pitch_ref_offset_deg`; Phase B replaces that scalar with `pitch_ref_total_deg`
computed per step.)

### 2.1 Support-error input

`support_error_m` is the live `sag_pos_error` already computed in the control loop at
`scripts/simulate_hierarchical_controller.py:4719` — the yaw-compensated support-center
displacement projected on the initial heading axis. **Positive = forward of reference.**
This is the same physical drift the centering metrics target
(`active_pitch_crossing_signed_error_m` tracks it closely inside the controller).

The outer loop uses the **unscaled** `sag_pos_error`, NOT `sag_pos_error_scaled`
(line 4813). The scaled version is reduced by `position_authority_scale` during pitch
transients; the outer loop must see the true steady-state drift, and its own safety gates
(pitch / contact / error magnitude) handle transients independently.

### 2.2 Support-error rate

```text
support_error_rate_m_s = (support_error_m - prev_support_error_m) / control_dt
```
Low-passed with `support_velocity_lowpass_alpha` to suppress numerical-derivative noise.
`prev_support_error_m` is per-run loop state initialized to the first-step error.

### 2.3 Integral (disabled initially)

`outer_loop_integral_enabled = False` and `Ki = 0` for all Phase B validation. The field
and accumulator exist so the I-term can be screened **later** (after PD passes Steps C/D),
but it contributes nothing in this phase. When enabled, the accumulator is clamped to
`±integral_clamp_m_s` and only integrates inside the deadband-cleared, safety-passed band.

---

## 3. Sign hypothesis (to be PROVEN in Phase 4, not assumed)

**Hypothesis:** `Kp_support > 0` is the restoring direction.

Reasoning from Phase A evidence: at `high_0p480` the robot drifts **forward** (support
error positive, 85–92% positive before the fix). Phase A applied a **positive** scheduled
offset (+3 deg) which centered it to 57.2% positive. The commit message records: *"Positive
offset makes controller target backward lean, reducing forward drift."*

Mechanism: positive offset → larger `pitch_x_ref` → smaller (more negative) `pitch_x_error`
→ the wheel-balance law commands the robot to settle further back → forward drift reduces.

Therefore, for the dynamic loop, a **positive** support error (forward drift) should add a
**positive** dynamic offset to push back. Restoring ⇒ `Kp_support > 0`.

**This sign is a hypothesis only.** Phase 4 runs both `Kp > 0` and `Kp < 0` at
`high_0p480` and selects the sign that (a) reduces support drift and (b) does not
destabilize posture / hip-yaw. If the sweep contradicts the hypothesis, the sweep wins
and the design doc is corrected. No sign is hard-coded as "correct" in the controller —
the sign lives entirely in the configured `Kp` value of the profile.

---

## 4. Configuration fields (added to `SagittalAuthoritySchedule`)

All default to disabled / inert so **every** existing profile is byte-for-byte unchanged
in behavior. Only `support_position_outer_loop_pitch_ref` sets `outer_loop_enabled = True`.

| Field | Default | Phase B initial |
|---|---|---|
| `outer_loop_enabled` | `False` | `True` |
| `outer_loop_kp_deg_per_m` | `0.0` | candidate from Phase 4 sign+gain sweep |
| `outer_loop_kd_deg_per_mps` | `0.0` | candidate from Phase 4 |
| `outer_loop_ki_deg_per_m_s` | `0.0` | `0.0` (disabled) |
| `outer_loop_integral_enabled` | `False` | `False` |
| `outer_loop_integral_clamp_m_s` | `0.05` | `0.05` |
| `outer_loop_theta_ref_max_deg` | `3.0` | `3.0` |
| `outer_loop_theta_ref_rate_limit_deg_per_step` | `0.03` | `0.03` |
| `outer_loop_theta_ref_lowpass_alpha` | `0.15` | `0.15` |
| `outer_loop_support_error_deadband_m` | `0.015` | `0.015` |
| `outer_loop_support_velocity_lowpass_alpha` | `0.20` | `0.20` |
| `outer_loop_disable_if_abs_error_gt_m` | `0.25` | `0.25` |
| `outer_loop_disable_if_pitch_gt_deg` | `12.0` | `12.0` |
| `outer_loop_disable_if_roll_gt_deg` | `5.0` | `5.0` |
| `outer_loop_contact_required` | `True` | `True` |
| `outer_loop_height_schedule_required` | `True` | `True` |

`outer_loop_height_schedule_required = True` means the loop only activates when the base
profile also has `pitch_ref_height_schedule_enabled = True`. This binds the dynamic loop
to the Phase A base and prevents it from acting on a profile with no height schedule.

---

## 5. Safety gates

The dynamic term is forced to **decay toward 0** (rate-limited) whenever ANY gate fails.
It never jumps to zero (avoids a step in `pitch_ref`).

| Gate | Condition to ALLOW | Block reason string |
|---|---|---|
| Enabled | `outer_loop_enabled` | `disabled` |
| Schedule bound | base `pitch_ref_height_schedule_enabled` | `no_height_schedule` |
| Contact | `contact_valid` (if `contact_required`) | `contact_invalid` |
| Error magnitude | `abs(support_error) <= disable_if_abs_error_gt_m` | `error_too_large` |
| Pitch | `abs(pitch_deg) <= disable_if_pitch_gt_deg` | `pitch_unsafe` |
| Roll | `abs(roll_deg) <= disable_if_roll_gt_deg` | `roll_unsafe` |

When all gates pass: `outer_loop_gate_pass = True`, `block_reason = "active"`.

These gates are **additive** safety on top of the inner controller's existing gates. They
never relax any existing threshold and never disable pitch / damping / WBC.

---

## 6. Telemetry (added to CSV)

Per-step fields:

- `outer_loop_active` (bool)
- `outer_loop_support_error_m`
- `outer_loop_support_error_rate_mps`
- `outer_loop_pitch_ref_dynamic_deg`        (PD+I raw, after deadband+clamp, before rate/lowpass)
- `outer_loop_pitch_ref_total_deg`          (scheduled + dynamic, before rate/lowpass)
- `outer_loop_pitch_ref_limited_deg`        (after saturation)
- `outer_loop_pitch_ref_rate_limited_deg`   (after rate limit + lowpass — the value actually applied)
- `outer_loop_integral_m_s`
- `outer_loop_gate_pass` (bool)
- `outer_loop_block_reason` (str)
- `outer_loop_sign_selected` (str: `"positive"` / `"negative"` / `"none"`)
- `pitch_ref_offset_scheduled_deg`          (Phase A base, echoed for diffing)
- `pitch_ref_total_after_outer_loop_deg`    (final deg applied = scheduled + smoothed dynamic)
- `pitch_x_error_after_outer_loop_rad`      (resulting inner-loop pitch error)

Telemetry is **diagnostic** and present for all profiles (zeros / `"disabled"` when the
loop is off), so the CSV schema is stable.

---

## 7. Implementation plan (Phase 2)

1. **Dataclass fields** — add the 16 `outer_loop_*` fields to `SagittalAuthoritySchedule`
   with the defaults in §4. (`wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` ~line 225, next to the existing `pitch_ref_*` fields.)

2. **Profile constant** — `SUPPORT_POSITION_OUTER_LOOP_PITCH_REF = replace(HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM, profile_name="support_position_outer_loop_pitch_ref", outer_loop_enabled=True, outer_loop_kp_deg_per_m=<TBD Phase4>, outer_loop_kd_deg_per_mps=<TBD>, ...)`. Inherits the full Phase A schedule + all safety machinery. Placed right after `HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM` (~line 2441).

3. **Registry** — add `"support_position_outer_loop_pitch_ref": SUPPORT_POSITION_OUTER_LOOP_PITCH_REF` to `JOINT_FIX_PROFILES` (~line 2501) and to the `--vd-sagittal-authority-profile` CLI `choices` list (~line 2648).

4. **Pure helper** — in the controller module (next to `interpolate_pitch_ref_offset`):
   ```python
   def compute_outer_loop_pitch_ref(
       support_error_m: float,
       support_error_rate_m_s: float,
       integral_error_m_s: float,
       kp_deg_per_m: float,
       kd_deg_per_mps: float,
       ki_deg_per_m_s: float,
       deadband_m: float,
       theta_ref_max_deg: float,
   ) -> float:
       """Raw dynamic pitch_ref (deg): PD(+I), deadband on P-term, clamped to ±max.
       Pure float; no JAX. Returns 0.0 if |error| < deadband AND no rate/integral."""
   ```
   Plus two tiny pure helpers `apply_rate_limit(prev, target, max_delta)` and
   `apply_lowpass(prev, target, alpha)` for testability.

5. **Sim-loop wiring** — in `scripts/simulate_hierarchical_controller.py`:
   - Before the loop: read `outer_loop_*` from the resolved profile; init per-run state
     `prev_support_error_m`, `outer_loop_pitch_ref_smoothed_deg = 0.0`,
     `outer_loop_integral_m_s = 0.0`.
   - Inside the loop, after `sag_pos_error` (line 4719) and before line 4732:
     compute rate (lowpassed), evaluate safety gates, call the helper, apply
     rate-limit + lowpass (or decay toward 0 when gated), form
     `pitch_ref_total_deg = pitch_ref_offset_scheduled_deg + smoothed_dynamic_deg`.
   - Change line 4732 to use `pitch_ref_total_deg` (equals scheduled when loop disabled).
   - Append the §6 telemetry fields.
   - Add CSV columns in the telemetry init + per-step append blocks.

6. **CSV writer** — register the new columns (`telemetry.setdefault(...)` init + per-step
   append, matching the existing `vd_sagittal_authority_profile` pattern at line 5202).

**Invariants preserved:** when `outer_loop_enabled = False` (every legacy profile),
`pitch_ref_total_deg == pitch_ref_offset_scheduled_deg == vd_pitch_ref_offset_deg`, so the
applied torque is bit-identical to Phase A. `height_scheduled_pitch_equilibrium_trim` keeps
`outer_loop_enabled = False` and is therefore unchanged.

---

## 8. Risk points

- **Sign error** → drift amplification / runaway. Mitigated by Phase 4 explicit two-sign
  sweep before any longer run, plus `theta_ref_max = 3 deg` hard clamp and the
  `disable_if_abs_error_gt = 0.25 m` gate.
- **Oscillation** from too-high `Kp` or `Kd` lag. Mitigated by conservative gain ladder
  (0.5 / 1.0 / 1.5 deg/m), Kd screening, rate-limit + lowpass, and oscillation indicators
  (zero-crossing rate, P2P) in the screening report.
- **Transient coupling** with `position_authority_scale`. Mitigated by using unscaled
  `sag_pos_error` and independent pitch/contact gates.
- **Height-transition discontinuity** (Step C): the scheduled base offset can jump between
  heights. Mitigated by the existing `pitch_ref_offset_rate_limit_deg_per_step` /
  `pitch_ref_offset_lowpass_alpha` fields on the base schedule (enabled in Step C if
  needed) plus the outer-loop lowpass.

---

## 9. Classification rationale

`OUTER_LOOP_DESIGN_READY`:

- Mechanism is a bounded, gated, opt-in additive nudge to an existing setpoint — no new
  inner controller, no change to pitch/damping/WBC/HY2-DIV.
- Application site (`pitch_x_ref` at line 4732) and input signal (`sag_pos_error` at
  line 4719) are both confirmed present in the loop.
- All required fields, telemetry, and safety gates are specified concretely.
- Sign is left to empirical Phase 4 screening, not assumed.
- Phase A base profile remains the fallback and is not modified.

No blocking unknowns. Proceed to Phase 2 (implementation) and Phase 3 (tests).
