# Step D Validation Matrix

**Profiles:**
- **A (B2v2 baseline):** `calibrated_support_position_outer_loop_pitch_ref_v2`
- **B (current PFF):** `physics_equilibrium_feedforward_outer_loop`
- **C (candidate):** `physics_equilibrium_feedforward_outer_loop_low_band_support_v2`

All profiles use `centered_posture_height_schedule`.

| Case ID | Height | Steps | Push (N) | Duration (steps) | Interval (steps) | Profiles |
|---|---|---|---|---|---|---|
| D1_small_push_high | high_0p480 | 1000 | 30 | 5 | 150 | A, B, C |
| D2_medium_push_high | high_0p480 | 1000 | 60 | 5 | 150 | A, B, C |
| D3_small_push_low | low_0p330 | 1000 | 30 | 5 | 150 | A, B, C |
| D4_medium_push_low | low_0p330 | 1000 | 60 | 5 | 150 | A, B, C |
| D5_large_push_high | high_0p480 | 1000 | 90 | 5 | 200 | A, B, C |
| D6_random_push_high | high_0p480 | 1000 | 45 | 5 | 150 | A, B, C |