## Task 2: Design Step D Validation Matrix

**Goal:** Produce a markdown document that lists the push test cases (D1‑D6) with their parameters (height, steps, push magnitude, duration, interval) and indicates which controller profiles (A = baseline `height_scheduled_pitch_equilibrium_trim`, B = candidate `support_position_outer_loop_pitch_ref`, C = low‑band v2 `physics_equilibrium_feedforward_outer_loop_low_band_support_v2`) will be evaluated.

**Required file:** `docs/validation/step_d_validation_matrix.md`

**Content outline:**
```markdown
# Step D Validation Matrix

| Case ID | Height | Steps | Push (N) | Duration (steps) | Interval (steps) | Profiles |
|---|---|---|---|---|---|---|
| D1_small_push_high | high_0p480 | 1000 | 30 | 5 | 150 | A, B, C |
| D2_medium_push_high | high_0p480 | 1000 | 60 | 5 | 150 | A, B, C |
| D3_small_push_low | low_0p330 | 1000 | 30 | 5 | 150 | A, B, C |
| D4_medium_push_low | low_0p330 | 1000 | 60 | 5 | 150 | A, B, C |
| D5_large_push_high | high_0p480 | 1000 | 90 | 5 | 200 | A, B, C |
| D6_random_push_high | high_0p480 | 1000 | 45 | 5 | 150 | A, B, C |
```

**Steps for implementer:**
1. Create the file with the table above.
2. Ensure the markdown renders correctly (no syntax errors).
3. Commit the new file with message `doc: add Step D validation matrix`.

**Report file:** `docs/superpowers/plans/2024-06-21-physics-ff-low-band-support-v2-step-d-push-validation-and-promotion_task2_report.md`
