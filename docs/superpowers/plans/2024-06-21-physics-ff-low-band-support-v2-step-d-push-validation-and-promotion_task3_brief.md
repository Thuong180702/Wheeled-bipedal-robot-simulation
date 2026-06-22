## Task 3: Run Step D for All Profiles

**Goal:** Execute the Step D push‑disturbance validation for the three controller profiles (A = baseline `height_scheduled_pitch_equilibrium_trim`, B = candidate `support_position_outer_loop_pitch_ref`, C = low‑band v2 `physics_equilibrium_feedforward_outer_loop_low_band_support_v2`).

**Requirements:**
1. Use the existing `scripts/run_outer_loop_step_d_push.py` which already runs cases D1‑D6 for profiles A and B.
2. Extend it to also run profile C without modifying the original script's logic.
3. Create a wrapper script `scripts/run_step_d_all.py` that:
   - Imports the original runner as a module.
   - Defines `LOW_BAND_PROFILE = "physics_equilibrium_feedforward_outer_loop_low_band_support_v2"`.
   - Iterates over all `PUSH_CASES` and runs `run_sim` for each profile (A, B, C).
   - Collects all telemetry files and analysis results.
   - Writes a combined CSV `outputs/step_d_all_metrics.csv` with a row for each case/profile (fields: case_id, height, steps, push_mag_N, push_dur, push_int, profile, plus all metrics returned by `analyze`).
4. The wrapper must create its own output directory under `outputs/step_d_all/` (e.g., `step_d_D1_small_push_high_A`).
5. After the wrapper finishes, run the analysis function from the original script on each telemetry file to produce per‑case metrics.
6. The wrapper should be executable via `python scripts/run_step_d_all.py`.

**Testing:**
- Provide a quick sanity test that runs the wrapper with a dummy push magnitude of 0 (no pushes) for a single case to ensure it completes without error and produces a CSV with at least one row.
- The test file should be placed at `tests/test_step_d_all.py`.

**Commit:**
- Add the new wrapper script.
- Add the test file.
- Commit with message `feat: wrapper to run Step D for all three profiles`.

**Report:** Write a short report to:
`docs/superpowers/plans/2024-06-21-physics-ff-low-band-support-v2-step-d-push-validation-and-promotion_task3_report.md`
Include:
- Status: DONE (or appropriate)
- Commits made
- Any concerns
