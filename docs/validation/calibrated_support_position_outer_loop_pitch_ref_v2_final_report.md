# calibrated_support_position_outer_loop_pitch_ref_v2 — Final Validation Report

**Profile:** `calibrated_support_position_outer_loop_pitch_ref_v2` (short name: **B2v2**)
**Current best:** `support_position_outer_loop_pitch_ref` (short name: **B**)
**Fallback:** `height_scheduled_pitch_equilibrium_trim` (short name: **A**)
**Date:** 2026-06-19

---

## Final Classification

**`CALIBRATED_OUTER_LOOP_V2_EXPERIMENTAL_ONLY`**

B2v2 is safe and equal-or-better than B in aggregate across all validation phases, but does
not *clearly* beat B due to a real (though safe) low_0p320 regression visible under 30N push
(D7: +0.039 maxabs). **B remains current best.** B2v2 is committed as an experimental opt-in.

---

## Evidence Summary

### Phase 6 — Fixed-height (prior, 10 heights)
- 6/10 heights: B2v2 better
- 2/10 heights: B2v2 minor regression (low_0p320, high_0p480)
- 0 hard safety failures
- high_0p465 critical regression from v1 **eliminated**
- high_0p450 major win **preserved**

### Step C — Random/changing height (8 scenarios, classification: PASS)
- B2v2 better or equal: 8/8 cases
- Hard failures: 0
- Clear wins: C3 (out15 16.4%→0.7%), C5 (out15 18.6%→13.5%)
- Focused low_0p320 transitions: ≈equal (regressions do not amplify)
- Focused high_0p480 transitions: ≈equal
- No fall, no WBC, no parameter discontinuity (pitch_ref_disc=0 all)

### Step D — Push validation (12 scenarios, classification: PASS_WITH_MONITORING)
- B2v2 improves or matches B: **6/6 original D1-D6 cases** (net improve)
- B2v2 preserves low_0p330 push recovery:
  - D3 (30N): maxabs 0.2560→**0.2404** (−6%), out25 1.7%→**0.0%**
  - D4 (60N): maxabs 0.3416→**0.3228** (−5.5%)
- Genuine monitoring signal: D7 low_0p320 30N (+0.039 maxabs) — safe, does not amplify at 60N (D8)
- Shared hip-yaw architecture limit at 60N low-height push — B and B2v2 hit identical levels

### Safety
- No fall in any of 20 dynamic runs
- No WBC authority, no WBC owners, no hidden torque, no ownership violation
- Hip-yaw: equal-or-better (single monitoring signal: D7, safe)
- Roll < 1.1° all push cases
- pitch_ref continuous: 0 discontinuities all Step C cases

---

## Known regressions and status

| Regression | Phase 6 | Step C | Step D |
|---|---|---|---|
| low_0p320 | minor score | harmless (C6) | **visible at 30N** (D7, +0.039, safe), harmless at 60N (D8) |
| high_0p480 | minor score | harmless (C7) | harmless (D1/D2/D5/D10/D11/D12) |

---

## Why NOT STEP_C_D_PASS_CURRENT_BEST

The criterion for `CALIBRATED_OUTER_LOOP_V2_STEP_C_D_PASS_CURRENT_BEST` requires B2v2 to
"clearly beat B in dynamic conditions." B2v2 does not clearly beat B because:

1. D7 (30N low_0p320 push): maxabs +0.039 above B. While safe, this real regression prevents
   an unqualified "better" assessment.
2. Step C C1/C2/C6/C7/C8 cases are ≈equal (not improvements).

If the D7 regression were absent or smaller, the classification would be current-best.

---

## Commit decision

**Classification: CALIBRATED_OUTER_LOOP_V2_EXPERIMENTAL_ONLY**

Commit B2v2 as an experimental opt-in using:

```
git add -A
git commit -m "exp: add calibrated outer-loop v2 candidate" \
  -m "Adds opt-in calibrated_support_position_outer_loop_pitch_ref_v2 profile.
Validated against support_position_outer_loop_pitch_ref (B) across:
  - Fixed-height Phase 6: 6/10 heights better, minor regressions at low_0p320 and high_0p480.
  - Step C random/changing height: PASS (8/8 gate cases, 2 clear wins).
  - Step D push: PASS_WITH_MONITORING (6/6 original cases meet threshold, D7 monitoring signal).

Remains secondary to support_position_outer_loop_pitch_ref because dynamic validation
did not clearly outperform the current best (D7 low_0p320 30N push regression, safe).
No default profile changed. WBC/HY2-DIV state unchanged."
```

---

## Files produced

- `docs/validation/calibrated_outer_loop_v2_step_c_report.md`
- `docs/validation/calibrated_outer_loop_v2_step_d_report.md`
- `docs/validation/calibrated_outer_loop_v2_consolidated_decision.md`
- `docs/validation/calibrated_support_position_outer_loop_pitch_ref_v2_final_report.md` (this file)
- `outputs/.../calibrated_outer_loop_v2_step_c_metrics.csv`
- `outputs/.../calibrated_outer_loop_v2_step_d_metrics.csv`
- `scripts/run_calibrated_outer_loop_v2_step_c.py`
- `scripts/run_calibrated_outer_loop_v2_step_d.py`
- `tests/test_height_scheduled_pitch_equilibrium_trim.py` (updated: B2v2 in SCHEDULE_ENABLED_PROFILES)
- `tests/test_support_position_outer_loop_pitch_ref.py` (updated: B2v2 in OUTER_LOOP_ENABLED_PROFILES)
