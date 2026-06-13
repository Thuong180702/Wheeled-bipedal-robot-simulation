# T6J Centering Bias Trim Design

Date: 2026-06-13  
Base profile: `T6I_phase_aware_release`  
New profile: `T6J_centering_bias_trim`

## 1. Design objective
T6I is already validated as stable and bounded, but the Phase 1 audit showed that at `high_0p480` it settles into a persistent positive offset instead of centering around zero.

The design objective for T6J is therefore:
- preserve T6I stabilization behavior,
- preserve pitch and damping authority,
- preserve opt-in-only semantics,
- add only a **small, slow, bounded** centering correction,
- correct long-horizon signed support bias rather than emergency transients,
- avoid destabilizing high_0p480 or causing low-height regression.

## 2. Phase 1 evidence driving the design
From [t6i_positive_bias_root_cause_audit.md](docs/validation/t6i_positive_bias_root_cause_audit.md):
- mean error: **+0.0953 m**
- final error: **+0.1309 m**
- positive occupancy: **95.6%**
- late-run positive occupancy: **100.0%**
- late-run mean error: **+0.1005 m**
- `t6i_converging_pct`: **6.46%**
- classification: **T6I_BIAS_FROM_MISSING_CENTERING_INTEGRAL**

Interpretation:
- T6I keeps the robot alive, but not centered.
- The residual issue is a persistent signed bias, not a runaway instability.
- T6I cap decay is too sparse to act as a centering term.
- The fix should be a slow bias-correction path, not more aggressive emergency authority.

## 3. T6J mechanism summary
T6J will add a **support-centering bias trim** on top of T6I.

The trim is:
- opt-in only,
- stateful,
- bounded,
- rate-limited,
- gated by upright/contact/roll/wheel-speed/error magnitude safety checks,
- applied to the **support recenter / position torque path after T6I cap selection but before final torque composition**.

It is explicitly **not**:
- emergency recovery,
- pitch suppression,
- damping suppression,
- sign inversion,
- authority cap escalation,
- WBC,
- hard switching.

## 4. Control law concept
Let:
- `e` = selected physical signed support drift error
- `e_mean` = moving average of `e` over a long window
- `tau_trim_target` = desired centering trim torque
- `tau_trim` = actual bounded/rate-limited trim torque

### Rule
- If `e_mean > enter_threshold`, apply a **negative** trim torque.
- If `e_mean < -enter_threshold`, apply a **positive** trim torque.
- If `|e_mean| < exit_threshold`, decay trim torque toward zero.
- If any safety gate fails, decay trim torque toward zero.
- If `|e|` is too large, stop integrating further and decay or hold neutral so T6I/T6F emergency logic remains primary.

### Sign rule
Use the established `tau_position` sign convention:
- positive drift error → corrective negative trim torque
- negative drift error → corrective positive trim torque

This rule must be enforced directly from the signed error and must **not** be inferred from final wheel torque.

## 5. Proposed parameterization
Initial T6J parameters:
- `bias_trim_enabled = True`
- `bias_trim_window_steps = 200`
- `bias_trim_enter_threshold_m = 0.04`
- `bias_trim_exit_threshold_m = 0.015`
- `bias_trim_max_tau_nm = 0.35`
- `bias_trim_rate_nm_per_step = 0.01`
- `bias_trim_decay_rate_nm_per_step = 0.02`
- `bias_trim_only_when_upright = True`
- `bias_trim_only_when_contact_stable = True`
- `bias_trim_disable_if_pitch_gt_deg = 8.0`
- `bias_trim_disable_if_roll_gt_deg = 3.0`
- `bias_trim_disable_if_wheel_vel_gt_rad_s = 7.0`
- `bias_trim_disable_if_abs_error_gt_m = 0.22`

### Why these values are conservative
- **0.35 Nm max** is intentionally far below T6I emergency authority, so trim cannot dominate balance.
- **200-step window** makes the mechanism slow enough to respond to persistent bias rather than fast oscillation.
- **0.04 m enter / 0.015 m exit** creates hysteresis so the trim does not chatter.
- **0.01 Nm/step rate limit** keeps the correction gradual.
- **0.02 Nm/step decay** lets the trim wash out quickly when unsafe or no longer needed.
- **7 rad/s wheel-velocity gate** avoids acting during fast recovery motion.
- **0.22 m abs-error block** ensures the trim does not interfere with large-error rescue behavior.

## 6. Safety gates
The trim will only be eligible when all relevant safety conditions pass.

### Required gates
1. **Upright gate**
   - pitch within threshold
   - roll within threshold
2. **Contact-stable gate**
   - valid contact / stable support state
3. **Wheel-speed gate**
   - `|wheel_vel_mean| <= 7.0 rad/s`
4. **Magnitude gate**
   - `|e| <= 0.22 m`
5. **Height / variant continuity**
   - use the same high-level operating regime as T6I; do not create a separate controller mode

### Safety behavior
If any gate fails:
- do not integrate further,
- decay `tau_trim` toward zero,
- emit telemetry with a block reason.

## 7. Internal state needed
T6J requires a small amount of additional controller state:
- rolling bias-error history buffer,
- current mean error,
- current trim target torque,
- current applied trim torque,
- positive bias duration counter,
- negative bias duration counter.

These state variables are isolated to T6J and default-off so existing profiles remain unchanged.

## 8. Injection point
T6J trim will be applied:
1. after T6I cap logic determines the effective position/recenter authority,
2. before final wheel torque composition.

This preserves:
- T6I cap-decay semantics,
- pitch term semantics,
- damping term semantics,
- final clipping and safety behavior.

It also satisfies the explicit user requirement: *apply trim to the same support recenter position torque path after T6I cap selection but before final torque composition.*

## 9. Telemetry design
T6J will add the following telemetry fields:
- `t6j_bias_trim_enabled`
- `t6j_bias_trim_active`
- `t6j_bias_mean_error_m`
- `t6j_bias_window_steps`
- `t6j_bias_trim_tau_nm`
- `t6j_bias_trim_target_tau_nm`
- `t6j_bias_trim_rate_limited`
- `t6j_bias_positive_duration_steps`
- `t6j_bias_negative_duration_steps`
- `t6j_bias_safety_gate_pass`
- `t6j_bias_block_reason`
- `t6j_bias_applied_to_final_tau`
- `t6j_bias_expected_direction_correct`

### Telemetry intent
These fields will let validation answer:
- whether the trim actually engaged when persistent positive bias existed,
- whether it acted in the intended direction,
- whether it was being blocked by safety gates,
- whether it remained small and bounded,
- whether it introduced oscillatory behavior.

## 10. Expected behavior
### If T6J works as intended
Compared with T6I on `high_0p480`, T6J should:
- reduce positive occupancy,
- reduce mean error magnitude toward zero,
- reduce final error,
- reduce outside ±0.08 and ±0.10 occupancy,
- preserve survival and boundedness,
- preserve pitch/roll/contact stability,
- preserve no-WBC/no-hidden-torque/ownership invariants.

### What T6J should not do
- It should **not** materially reduce pitch authority.
- It should **not** materially reduce damping authority.
- It should **not** flip torque signs.
- It should **not** raise global emergency authority.
- It should **not** become active mainly during large transient rescue phases.

## 11. Main design risks
1. **Overcorrection / induced oscillation**
   - mitigated by long averaging window, low max tau, and rate limits.
2. **Interference with recovery**
   - mitigated by large-error block and wheel-speed safety gate.
3. **Low-height regression**
   - mitigated by staged validation and mandatory height-ladder sanity.
4. **Silent pitch/damping suppression**
   - mitigated by placing trim only on the position/recenter path and explicitly testing no pitch/damping suppression.

## 12. Classification
**T6J_CENTERING_BIAS_TRIM_DESIGN_READY**

The Phase 1 audit supports adding a dedicated slow centering trim, and the proposed T6J mechanism is conservative, opt-in, bounded, safety-gated, and aligned with all stated restrictions.
