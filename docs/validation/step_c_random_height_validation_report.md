# Step C: Random/Changing Height Validation

**A:** `height_scheduled_pitch_equilibrium_trim`
**B:** `support_position_outer_loop_pitch_ref`
**Classification:** `STEP_C_RANDOM_HEIGHT_PASS`

---

## Sequence Summaries

| Sequence | Profile | n_seg | any_fell | any_unsafe | mean_pos% | max_maxabs | max_trans |
|----------|---------|-------|----------|------------|-----------|------------|-----------|
| C1_slow_ladder_up_down | A | 20 | False | False | 55.0 | 0.1165 | 0.0389 |
| C1_slow_ladder_up_down | B | 20 | False | False | 54.5 | 0.1178 | 0.039 |
| C2_random_500dwell | A | 10 | False | False | 56.3 | 0.116 | 0.0339 |
| C2_random_500dwell | B | 10 | False | False | 56.7 | 0.1171 | 0.034 |
| C3_random_200dwell | A | 15 | False | False | 64.1 | 0.1165 | 0.0389 |
| C3_random_200dwell | B | 15 | False | False | 63.5 | 0.1178 | 0.039 |
| C4_abrupt_stress | A | 5 | False | False | 51.5 | 0.1165 | 0.0389 |
| C4_abrupt_stress | B | 5 | False | False | 51.4 | 0.1178 | 0.039 |
| C5_long_random | A | 20 | False | False | 57.4 | 0.1165 | 0.0389 |
| C5_long_random | B | 20 | False | False | 57.0 | 0.1178 | 0.039 |

## Decision

- **STEP_C_RANDOM_HEIGHT_PASS**

- B passed all Step C sequences.
- Proceed to Step D random push disturbance.
