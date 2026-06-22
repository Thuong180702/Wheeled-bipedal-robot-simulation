## Task 5: Fixed-Height & Step C Regression Re-Check

**Goal:** Verify that Step C (fixed-height validation) results from the v2 tuning are still valid for the low-band v2 profile, confirming no regression at protected low heights (low_0p320, low_0p330, low_0p360) and high heights (high_0p480).

**Requirements:**
1. Read the existing Step C artifacts from `outputs/physics_ff_low_band_support_v2_tuning/`:
   - `full_fixed_height_metrics.csv`
   - `full_step_c_segment_metrics.csv`
2. Verify that the low-band v2 profile:
   - Had no falls in any fixed-height or Step C case
   - Had no hip-yaw hard fail (hip_yaw_abs_max_rad < 0.35)
   - Had no WBC rows, hidden torque, or ownership violations
   - maxabs at low_0p320 is < 0.15 m for focused case
   - maxabs at high_0p480 matches current PFF (within tolerance)
   - out15% = 0 for all Step C cases
3. Check if the current git state matches the state when artifacts were generated (compare git log). If the directory has uncommitted changes that affect controllers, flag this as a concern.
4. Write a verification report to `docs/validation/step_c_regression_recheck.md` containing:
   - Summary table with the 7 Step C cases + fixed-height low_0p320/high_0p480
   - Pass/fail for each gate
   - Overall verdict: STEP_C_RECHECK_PASS or STEP_C_RECHECK_FAIL
5. Add a pytest `tests/test_step_c_recheck.py` that validates the report file exists and contains "STEP_C_RECHECK_PASS".

**Commit message:** `test: add Step C recheck verification and report`

**Report file:** `docs/superpowers/plans/2024-06-21-physics-ff-low-band-support-v2-step-d-push-validation-and-promotion_task5_report.md`
