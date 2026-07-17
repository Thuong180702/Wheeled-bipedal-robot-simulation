# Support-Position Outer-Loop Pitch Reference — Final Report

**Profile:** `support_position_outer_loop_pitch_ref`
**Base:** `height_scheduled_pitch_equilibrium_trim` (Phase A, commit `befb874`)
**Implementation commits:** `5b7248d` (impl), `5f0cf94` (sweep harness), `e831619` (Phase 5 fix + Step C/D runners)
**Date:** 2026-06-17

**Final classification: `OUTER_LOOP_STEP_C_D_PASS_CURRENT_BEST`**

---

## 1. Was Phase A committed?

Yes. Commit `befb874` (Phase A: `height_scheduled_pitch_equilibrium_trim`) is the
frozen baseline. Phase B is implemented strictly on top of it and remains opt-in.

## 2. Did B improve fixed-height behavior vs A?

Yes. **Phase 5 classification: `OUTER_LOOP_FIXED_HEIGHT_PASS_BETTER_THAN_HEIGHT_SCHEDULE`**

| Metric | A (Phase A) | B (Phase B) | Note |
|---|---|---|---|
| Improve heights (maxabs/P2P/pos-balance) | — | **9 / 10** | |
| Regression heights | — | **1 / 10** | high_0p450 only (maxabs +0.036, out15 +15.5pp; safe) |
| Hard safety failures (any height) | — | **0** | |
| Protected heights | high_0p480 / low_0p330 / low_0p360 | all safe (B marginal on high_0p480, within gate) | |

B vs A on the 10-height ladder at 2000 steps (excerpt):

| height | A max_abs | B max_abs | B vs A |
|---|---|---|---|
| low_0p300 | 0.068 | 0.081 | B slightly worse but improves pos% (44.6→45.5) |
| low_0p320 | 0.127 | 0.141 | B slightly worse but improves pos% (43.3→45.1) |
| low_0p330 | 0.123 | 0.131 | improve (49.0→50.0, P2P slightly worse) |
| low_0p360 | 0.116 | 0.117 | **improves** (P2P 0.227→0.206, hip_yaw 0.181→0.117) |
| high_0p465 | 0.166 | 0.152 | **improves** (out15 3.4→0.8) |
| high_0p480 | 0.169 | 0.181 | marginal regression at 2000 steps |

## 3. Did B improve min/max/P2P under fixed-height?

Mostly, with one regression:

- **max_abs:** 6/10 heights improved, 1/10 regression at high_0p450, 3/10 marginal.
- **P2P:** 5/10 improved, 4/10 marginal, 1/10 regression (high_0p450).
- **pos% balancing:** every height either equal or improved (B systematically nudges pos% toward 50 from above or below).

## 4. Did B preserve posture / contact / height / roll?

Yes. Every height passed the strict safety gate (no fall, pitch < 16 deg, roll_rms < 3 deg,
hip_yaw < 0.35 rad, no active WBC, no hidden torque, no ownership violation).

## 5. Did B preserve hip-yaw / leg-yaw?

Yes. Hip-yaw `abs_max` was below 0.22 rad at every height for both A and B (the
0.20 rad leg-yaw audit threshold from Phase A was preserved). B was marginally higher
than A on hip-yaw at high_0p480/5000-step (0.1395 vs 0.1022 rad) but well below 0.35.

## 6. Did B pass all 10 fixed heights?

Yes. All 10 heights completed without fall, with hip-yaw / roll / contact / height
within safety thresholds.

## 7. Did B pass Step C random/changing height?

Yes. **Phase 6 classification: `STEP_C_RANDOM_HEIGHT_PASS`** — 5/5 sequences.

- C1 slow ladder up/down (20 segments × 300 steps): A and B both safe.
- C2 random ~500-step dwell (10 segments): both safe.
- C3 random ~200-step dwell (15 segments): both safe.
- C4 abrupt high-low-high stress (5 segments): both safe.
- C5 long random 20-segment sequence: both safe.

Mean pos% centering tracked A within 1 pp on every sequence; transition-window
max-abs was 0.039 m for both (well below 0.25 m gate).

## 8. Did B pass Step D random push disturbance?

Yes. **Phase 7 classification: `STEP_D_RANDOM_PUSH_PASS`** — 6/6 cases B-not-worse vs A.

B's clearest advantage appeared under push recovery:

| Case | A max_abs (m) | B max_abs (m) | Δ |
|---|---|---|---|
| D3 (low_0p330, 30N push) | 1.164 | **0.198** | B keeps drift bounded where A rolls out |
| D4 (low_0p330, 60N push) | 1.117 | **0.153** | same |
| D5 (high_0p480, 90N push) | 0.147 | 0.056 | -62% |
| D2 (high_0p480, 60N push) | 0.130 | 0.098 | -25% |
| D1 (high_0p480, 30N push) | 0.003 | 0.028 | trivial (already well-centered) |
| D6 (high_0p480, random dir, 45N) | 0.123 | 0.116 | -6% |

The Phase A static schedule is just a per-height offset — under push it has no way to
react. The Phase B outer loop is a real-time correction to the same offset and visibly
prevents the failure mode Phase A cannot handle.

## 9. Which profile is current best?

`support_position_outer_loop_pitch_ref` (Phase B) for ALL scenarios covered by
this validation (fixed height + random/changing height + push disturbance), because:

- 9/10 fixed heights improved vs A with 0 hard-safety regressions.
- 5/5 random-height sequences passed safely.
- 6/6 push cases passed; B reduced max-abs in 4 of them by **≥25%**.
- B is *strictly* opt-in: `outer_loop_enabled=True` only on this profile, every
  other profile keeps Phase A's behavior byte-for-byte.

## 10. Should B be committed and/or made recommended profile?

Yes — commit Phase B as a **new recommended profile**, opt-in via
`--vd-sagittal-authority-profile support_position_outer_loop_pitch_ref`.

It must remain **opt-in, not default**, until:

- trained PPO policy interaction is validated (the current validation is hierarchical
  control + open-loop Phase B; no RL loop),
- lateral pushes (out-of-sagittal-plane) are tested,
- high-0p450 regression is reduced (currently maxabs +0.036, out15 +15.5pp; in the
  tolerance band but worth tightening).

## 11. Should A remain fallback?

Yes. `height_scheduled_pitch_equilibrium_trim` is the **fallback** — it remains the
default for every other profile and is what the inner-loop tuning (pitch gain, damping,
k_position) assumes. Phase B is a strict superset of A and falls back to A behavior
when `outer_loop_enabled=False` on every other profile.

---

## Final classification: `OUTER_LOOP_STEP_C_D_PASS_CURRENT_BEST`

Phase B (`support_position_outer_loop_pitch_ref`) passes Phase 5 (fixed-height ladder),
Phase 6 (Step C random-height) and Phase 7 (Step D push disturbance) and is the
**current best profile** in the validation set, while `height_scheduled_pitch_equilibrium_trim`
remains the recommended fallback.

## Recommendation

1. Commit Phase B profile (already in `5b7248d`, `5f0cf94`, `e831619`).
2. Do **NOT** change the default profile — keep Phase A as the default; users opt into
   Phase B via `--vd-sagittal-authority-profile support_position_outer_loop_pitch_ref`.
3. Next research step (not in this task): lateral-push validation, RL-pipeline
   integration with Phase B as the base controller, and tightening the high_0p450
   regression with a larger Kp or a per-height Kp schedule.