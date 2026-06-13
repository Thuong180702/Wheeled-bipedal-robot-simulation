# T6J Centering Bias Trim Implementation Tests Report

Date: 2026-06-13  
Profile: `T6J_centering_bias_trim`

## Scope
This report covers Phase 4 implementation verification for the new opt-in T6J profile.

Implemented items:
- new schedule fields for slow centering bias trim
- new profile definition `T6J_centering_bias_trim`
- registry + CLI exposure
- internal controller state for rolling mean / trim torque / duration counters
- bounded, rate-limited trim injection on the support-position torque path
- telemetry fields for T6J bias trim
- dedicated unit tests
- regression verification against existing T6/T6I/APCR/telemetry/WBC suites

## New dedicated test file
- [tests/test_t6j_centering_bias_trim.py](tests/test_t6j_centering_bias_trim.py)

### Dedicated T6J results
`26 passed`

Verified behaviors:
1. T6J profile exists and is opt-in.
2. T6J inherits T6I settings.
3. T6I remains unchanged.
4. T6F remains unchanged.
5. T5 remains unchanged.
6. Bias trim activates for persistent positive mean error.
7. Bias trim activates for persistent negative mean error.
8. Positive mean error produces corrective negative trim.
9. Negative mean error produces corrective positive trim.
10. Bias trim does not activate below enter threshold.
11. Bias trim decays inside exit threshold.
12. Bias trim is bounded by max tau.
13. Bias trim is rate limited.
14. Bias trim disabled when pitch safety fails.
15. Bias trim disabled when roll safety fails.
16. Bias trim disabled when wheel velocity safety fails.
17. Bias trim disabled when contact unstable.
18. Bias trim disabled when abs(error) too large.
19. Bias trim does not suppress pitch.
20. Bias trim does not suppress damping.
21. T6I cap decay still works.
22. Final motor cap still respected.
23. T6J telemetry fields exist.
24. CSV writer logs T6J fields.
25. No WBC path change.
26. No HY2-DIV default change.

## Regression suite
Required regression suite re-run after implementation:
- `tests/test_t6h_t6i_variants.py`
- `tests/test_t6_high_height_variants.py`
- `tests/test_t6f_torque_sign_convention.py`
- `tests/test_apcr1nd_tuned_variants.py`
- `tests/test_sagittal_velocity_damped_balance_controller.py`
- `tests/test_simulation_telemetry_csv_writer.py`
- `tests/test_low_height_setup_initialization.py`
- `tests/test_step_e_wbc_gate_validator.py`

### Regression results
`428 passed`

This confirms:
- no regression to T6H/T6I behavior contracts,
- no regression to T6F sign-convention tests,
- no regression to APCR1nD tuned variants,
- no regression to broad sagittal controller invariants,
- telemetry CSV support remains valid,
- no regression to low-height setup initialization,
- no regression to Step E WBC gate validation.

## Implementation notes
### T6J mechanism placement
The bias trim is applied on the support-position torque path after T6I cap selection and before final torque composition, consistent with the requested design.

### Preserved restrictions
Confirmed by code path + tests:
- T6I not modified directly as a profile target
- T6F not modified directly as a profile target
- T5 not modified directly as a profile target
- no pitch suppression
- no damping suppression
- no sign flips
- no WBC path change
- no HY2-DIV default change

## Phase 4 classification
**T6J_IMPLEMENTED_TESTS_PASS**
