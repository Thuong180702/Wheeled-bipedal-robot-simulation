# Sagittal Structural Controller Rebuild — Final Report

**Date:** 2026-06-15
**Branch:** repo-cleanup-t6j (uncommitted)
**Primary target:** high_0p480 support drift centered around zero
**Profile delivered:** `pitch_equilibrium_trim` (opt-in, not default)

---

## Final Classification

**`STRUCTURAL_REBUILD_PASS_WITH_MONITORING`**

Drift at high_0p480 is centered (46.7% positive over 5000 steps vs 84–92%
for every prior fallback profile), with no fall and safe posture. Classified
**with monitoring** rather than unconditional pass because:

1. The fixed +4° offset is tuned for high_0p480. The best-offset sweep shows the
   optimum is height-dependent, and at a fixed +4° the low variants over-correct
   into negative-biased drift (low_0p330 ≈ 3–4% positive). They remain safe and
   upright, but symmetry there is not yet centered.
2. Positive-% drifts upward with horizon (38.9% → 41.5% → 43.7% → 46.7% across
   500/1200/2000/5000) and stabilizes near 47%; peak-to-peak grows to ~0.34 m at
   5000 steps. No runaway, but the long-horizon trend should be watched.

This is the first profile to break the forward-drift stalemate at root cause
rather than fighting it with recenter trims.

---

## Root Cause (Phases 1–3)

The one-sided positive support drift was **not** a recenter-logic failure. It is
a **pitch-reference / equilibrium-posture mismatch**:

- The height-0.48 leg geometry settles the robot at a forward-pitched
  equilibrium of +3 to +5°.
- The sagittal controller pins `pitch_x_ref = 0` exactly. It therefore reads the
  equilibrium lean as a persistent positive pitch error and produces a persistent
  positive `tau_pitch` (mean +3.3 Nm), driving wheels forward.
- `tau_position` saturates pulling backward (−7 Nm, clipped 13–31% of steps) but
  cannot win; the two net to ~0 final wheel torque and the robot freezes at a
  forward-biased support position.
- `tau_pitch ↔ pitch_error` correlation is +1.000: no sign error, no injected DC,
  no asymmetric gain. The bias is entirely the non-zero mean of pitch against a
  zero reference.

Causal ablation confirmed **`ROOT_CAUSE_PITCH_GAIN_TOO_HIGH`** relative to the
equilibrium requirement: positive-drift % is a monotonic function of the standing
`tau_pitch` DC level, controlled by a single scalar (the pitch setpoint), not by
any recenter parameter.

---

## Fix (Phases 4–5)

**`STRUCTURAL_FIX_OUTER_LOOP_PITCH_REF`**, implemented as its static DC special
case: a fixed equilibrium pitch-reference offset.

```
pitch_x_ref = pitch_x_eq + radians(pitch_ref_offset_deg)   # offset = +4.0
pitch_x_error = body_pitch_x - pitch_x_ref
tau_pitch = kp_pitch * pitch_x_error      # full gain preserved
```

Moving the *reference* to the equilibrium lean (rather than weakening the gain)
removes the DC bias while keeping full dynamic pitch authority. Lowering
`kp_pitch` instead was rejected: it centers drift but the robot **fell** at
1200 steps (kp=12.5, orientation_fail at step 787).

- New field `pitch_ref_offset_deg` (default `0.0`) on `SagittalAuthoritySchedule`
  → all existing profiles byte-for-byte unchanged.
- New opt-in profile `pitch_equilibrium_trim` = `replace(
  ADAPTIVE_SUPPORT_CENTERING_TRIM, pitch_ref_offset_deg=4.0)`, inheriting every
  safety gate, recenter mechanism, and authority schedule.
- No WBC path change. No HY2-DIV. No default change. No thresholds relaxed.

---

## Validation Results

### high_0p480 staged (Phases 6–7)

| steps | pos% | neg% | min drift | max drift | P2P | fall |
|-------|------|------|-----------|-----------|-----|------|
| 500   | 38.9 | 61.1 | −0.065 | +0.035 | 0.10 | no |
| 1200  | 41.5 | 58.5 | −0.171 | +0.119 | 0.29 | no |
| 2000  | 43.7 | 56.3 | −0.182 | +0.137 | 0.32 | no |
| 5000  | 46.7 | 53.3 | −0.182 | +0.155 | 0.34 | no |

Fallback (`pitch_bias_compensated_zero_crossing_recenter` and all predecessors):
84–92% positive at 5000 steps. The fix moves drift from one-sided-positive to
centered. pitch range at 5000 steps: −4.1° to +11.1°; roll −0.2° to +0.3°; CoM
height 0.481–0.491 m; both wheels in contact; no hip-yaw violation.

### Target check (5000-step, high_0p480)

| target | result | verdict |
|--------|--------|---------|
| positive % ≤ 70 | 46.7% | PASS |
| negative % ≥ 25 | 53.3% | PASS |
| symmetry ≫ better than fallback | 46.7% vs 84–92% | PASS |
| sustained zero crossings | both sides substantial | PASS |
| min/max roughly balanced | −0.182 / +0.155 | PASS |
| no drift accumulation | stabilizes ~47%, bounded | PASS |
| posture / hip-yaw / contact / height / roll safe | all safe | PASS |
| no fall | no fall | PASS |

The desired band was "positive % close to 50–65%". 46.7% is marginally below 50
(slightly negative-leaning) but unambiguously centered and far from the 84–92%
failure regime.

### Height ladder (Phase 8) — fixed +4° offset

| variant | fall | pitch range | pos% |
|---------|------|-------------|------|
| low_0p300 | no | 0.0° → 6.0° | 30.9% |
| low_0p330 | no | −5.6° → 2.2° | 3.4% |
| low_0p360 | no | −4.0° → 2.3° | 3.8% |

No falls at any height. Low variants over-correct (negative-biased) under the
high-tuned +4° offset — safe but not centered there. Best-offset sweep:
high_0p480→+4, high_0p465→+2, high_0p450→+4, high_0p430→+4, low_0p360→0,
low_0p330→+6, low_0p300→+2. A height-scheduled offset is the documented next
refinement.

---

## Tests

- `tests/test_pitch_equilibrium_trim.py`: 22 passed (profile exists, offset
  correct/bounded, opt-in confirmed across all other profiles, parent safety
  inherited, no WBC fields, pitch gain not suppressed, CLI accepts, no-NaN
  rollout).
- Full suite: 368 passed, 1 skipped (no regressions).
- `py_compile` clean on both modified files.

---

## Deliverables

Docs: `full_sagittal_control_logic_audit.md`, `sagittal_equilibrium_state_audit.md`,
`sagittal_causal_ablation_report.md`, `sagittal_root_cause_final_report.md`,
`sagittal_structural_fix_decision.md`, this report.
Code: `pitch_ref_offset_deg` field + `pitch_equilibrium_trim` profile in
`sagittal_velocity_damped_balance_controller.py`; CLI plumbing in
`simulate_hierarchical_controller.py`; `scripts/run_sagittal_causal_ablation.py`.

## Status / next steps

- **Not committed** (per instruction: commit only on final pass; this is
  PASS_WITH_MONITORING). Hold for user decision.
- Recommended refinement if low-height symmetry becomes a target: height-scheduled
  `pitch_ref_offset_deg` using the per-variant offsets above, or promote the static
  offset to the full dynamic support-position outer loop.
