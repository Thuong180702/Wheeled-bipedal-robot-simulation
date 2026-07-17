# Sagittal Structural Fix Decision

**Date:** 2026-06-15
**Phase:** 4 — Choose structural fix path
**Depends on:** `full_sagittal_control_logic_audit.md` (Phase 1),
`sagittal_equilibrium_state_audit.md` (Phase 2),
`sagittal_causal_ablation_report.md` (Phase 3)

---

## Classification

**`STRUCTURAL_FIX_OUTER_LOOP_PITCH_REF`** (implemented as a fixed equilibrium
pitch-reference offset, the static special case of the outer-loop concept).

---

## Why this path

The Phase 3 causal ablations isolated the true mechanism behind one-sided
positive support drift:

1. **Ablation A (kp_pitch sweep).** Halving `kp_pitch` (50→25) dropped positive
   drift from 80.8% to 62.1%; quartering it (50→12.5) gave 49.9% (symmetric)
   with `tau_pitch` mean collapsing from 2.87 Nm to 0.11 Nm. Drift one-sidedness
   is a **monotonic function of the standing `tau_pitch` DC level**, not of any
   recenter trim. This is the smoking gun: the pitch loop, not the recenter
   logic, sets the drift bias.

2. **Why not just lower kp_pitch.** The 1200-step run at kp_pitch=12.5 **fell**
   (`orientation_fail_pitch_x_0.79` at step 787, pitch swing −22°→+45°). Lowering
   the gain removes the forward bias but also removes the dynamic pitch authority
   that keeps the robot upright. Gain reduction is therefore rejected — it trades
   a drift bias for a fall risk.

3. **The bias is a setpoint problem, not a gain problem.** Phase 1/2 confirmed
   `pitch_x_ref = 0` exactly, `tau_pitch ↔ pitch_error` correlation = +1.000, no
   sign error, no injected DC. The robot settles at a forward-pitched equilibrium
   (+3 to +5°) because the height-0.48 leg geometry places the CoM slightly
   forward of the wheel contact line. With the reference pinned at 0, the pitch
   loop reads that equilibrium lean as a persistent error and pushes the wheels
   forward forever, fighting `tau_position` to a near-zero-torque stalemate
   biased forward.

4. **Ablation C (positive pitch_ref offset).** Moving the *reference* to match
   the equilibrium lean — instead of weakening the gain — keeps full dynamic
   authority while removing the DC bias:

   | offset | pos_drift% | tau_pitch mean | max_drift |
   |--------|-----------|----------------|-----------|
   | 0 deg  | 80.8%     | +2.865 Nm      | +0.157 m  |
   | +1 deg | 68.3%     | +1.894 Nm      | +0.157 m  |
   | +2 deg | 63.7%     | +1.029 Nm      | +0.115 m  |
   | +3 deg | 61.1%     | +0.219 Nm      | +0.042 m  |
   | +4 deg | **38.9%** | **−0.496 Nm**  | +0.035 m  |

   At +4° the standing `tau_pitch` crosses through zero and drift centers, with
   the dynamic gain fully intact.

   (Negative offsets in Ablation B made it strictly worse — 90.6% positive —
   confirming the sign of the correction.)

## Paths considered and rejected

- **Fix Path A — equilibrium posture (regenerate setup geometry).** Viable but
  invasive: requires regenerating validated height-setup JSONs and re-running
  static feasibility for every variant. The pitch-reference offset achieves the
  same equilibrium shift in the controller without touching the validated robot
  setup files. Deferred unless the offset proves height-fragile.
- **Fix Path C — unified sagittal LQR/state feedback.** A full rebuild is not
  justified: the additive architecture is not the root cause (Ablation A shows a
  single scalar — the pitch setpoint — controls the bias). Reserved as a future
  option if the offset cannot generalize.
- **Fix Path D — mixed posture + outer loop.** Not needed at this stage; the
  single-parameter setpoint fix already centers drift.

## What was implemented

A fixed `pitch_ref_offset_deg` is the **static (DC) special case** of the
support-position outer-loop pitch reference. The dynamic outer loop computes
`theta_ref = clamp(-kx·e_support - kd·ė_support, ±theta_max)`; here the support
error sits at a near-constant positive equilibrium, so the optimal `theta_ref`
is a constant ≈ +4°. We implement that constant directly, which is simpler,
has no extra state, and cannot introduce an integrator wind-up failure mode.
If a single constant proves height-fragile, the per-height schedule (below) or
the full dynamic outer loop is the documented next step.

- Field `pitch_ref_offset_deg` added to `SagittalAuthoritySchedule` (default
  `0.0` → every existing profile is byte-for-byte unchanged).
- Opt-in profile `pitch_equilibrium_trim` = `replace(ADAPTIVE_SUPPORT_CENTERING_TRIM,
  pitch_ref_offset_deg=4.0)`, inheriting all safety gates, recenter machinery,
  and authority scheduling.
- Applied at `pitch_x_ref = pitch_x_eq + radians(offset)` in the sim control
  loop; profile value overrides the CLI default when nonzero.
- No WBC path change, no HY2-DIV enablement, no default change.

## Per-height offsets (for the ladder, not yet baked into the profile)

A best-offset sweep across variants shows the optimum is height-dependent:

| variant   | best offset | pos% at best |
|-----------|-------------|--------------|
| high_0p480 | +4 deg | 38.9% |
| high_0p465 | +2 deg | 49.7% |
| high_0p450 | +4 deg | 57.1% |
| high_0p430 | +4 deg | 38.3% |
| low_0p360  | 0 deg  | 14.4% |
| low_0p330  | +6 deg | 4.4%  |
| low_0p300  | +2 deg | 61.1% |

The current profile ships a single +4° constant tuned for the primary target
(high_0p480). Low variants already run without falling at +4° (height-ladder
screen), but a height-scheduled offset is the documented refinement if symmetry
at low heights becomes a target.
