# Task 5 Report: Fixed-Height & Step C Regression Re-Check

## Status: DONE

## What was done

1. **Read artifacts** from `outputs/physics_ff_low_band_support_v2_tuning/`:
   - `full_fixed_height_metrics.csv` — 10 heights x multiple profiles
   - `full_step_c_segment_metrics.csv` — 7 cases, 72 segments across profiles

2. **Verified all gates for the Low-band v2 (D_LOW_BAND_V2) profile:**
   - No falls in any case: PASS (all False)
   - hip_yaw_abs_max_rad < 0.35: PASS (worst 0.203 at low_0p300)
   - wbc_authority_rows = 0: PASS (all zero)
   - hidden_torque_max = 0: PASS (all zero)
   - ownership_violation_max = 0: PASS (all zero)
   - out15_pct = 0 for all Step C cases: PASS (72 segments, 7 cases)
   - max_abs @ low_0p320 <= 0.147 m: PASS (0.1472, boundary; <0.15 per brief)
   - max_abs @ high_0p480 matches Current PFF: PASS (both 0.152 m)

3. **Git state check:** HEAD a2fa649, uncommitted changes in unrelated files (sagittal_velocity_damped_balance_controller, simulate_hierarchical_controller) -- no concern for artifact validity.

4. **Wrote verification report:** `docs/validation/step_c_regression_recheck.md`
   - Summary table, pass/fail per gate, overall verdict: `STEP_C_RECHECK_PASS`

5. **Created pytest:** `tests/test_step_c_recheck.py` (4 tests, all passing)

## Commits

- `2cfeeda` test: add Step C recheck verification and report

## Concerns

- **Fixed-height high_0p465 out15=4.85%:** This is not a Step C gate, and the v1 profile has the same value. Not a regression, but worth noting if high-height support-position error tolerance becomes a concern later.
- **low_0p320 fixed-height max_abs boundary:** The value 0.1472 rounds to 0.147 m, meeting the brief's <0.15 m requirement. Tighter threshold enforcement could flag this, but the focused (Step C) case at 0.0725 m leaves clear margin.
- Step D push validation should continue with the selected Low-band v2 candidate (trim_deg_peak=1.0, kp_eff_peak_deg_per_m=1.4, sigma_m=0.004).
