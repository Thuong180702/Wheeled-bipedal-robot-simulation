# Step D: Random Push Disturbance Validation

**A:** `height_scheduled_pitch_equilibrium_trim`
**B:** `support_position_outer_loop_pitch_ref`
**Classification:** `STEP_D_RANDOM_PUSH_PASS`

---

## Case Results

| Case | Height | Push(N) | Prof | Fell | max_abs | P2P | out25% | hip_yaw | safe |
|------|--------|---------|------|------|---------|-----|--------|---------|------|
| D1_small_push_high | high_0p480 | 30 | A | False | 0.003 | 0.006 | 0.0 | 0.003 | True |
| D1_small_push_high | high_0p480 | 30 | B | False | 0.028 | 0.031 | 0.0 | 0.005 | True |
| D2_medium_push_high | high_0p480 | 60 | A | False | 0.130 | 0.165 | 0.0 | 0.058 | True |
| D2_medium_push_high | high_0p480 | 60 | B | False | 0.098 | 0.135 | 0.0 | 0.067 | True |
| D3_small_push_low | low_0p330 | 30 | A | False | 1.164 | 1.173 | 83.8 | 0.220 | True |
| D3_small_push_low | low_0p330 | 30 | B | False | 0.198 | 0.200 | 0.0 | 0.105 | True |
| D4_medium_push_low | low_0p330 | 60 | A | False | 1.117 | 1.127 | 82.2 | 0.221 | True |
| D4_medium_push_low | low_0p330 | 60 | B | False | 0.153 | 0.155 | 0.0 | 0.043 | True |
| D5_large_push_high | high_0p480 | 90 | A | False | 0.147 | 0.165 | 0.0 | 0.024 | True |
| D5_large_push_high | high_0p480 | 90 | B | False | 0.056 | 0.097 | 0.0 | 0.034 | True |
| D6_random_push_high | high_0p480 | 45 | A | False | 0.123 | 0.157 | 0.0 | 0.016 | True |
| D6_random_push_high | high_0p480 | 45 | B | False | 0.116 | 0.140 | 0.0 | 0.017 | True |

## Decision

- **STEP_D_RANDOM_PUSH_PASS**

- B passed Step D push disturbance validation.
- Proceed to Phase 8 final report and commit decision.
