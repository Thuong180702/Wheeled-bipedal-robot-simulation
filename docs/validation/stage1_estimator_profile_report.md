# Stage 1: Estimator Profile and Duplicate Call Analysis

**Date:** 2026-06-26
**Profile:** K2_NOTCH_LOW_Q_V1, backend=python
**Scenario:** high_0p480, 100 steps headless

---

## Per-Component Timing (mean ms/step)

| Component | ms/step | % of Total |
|-----------|---------|-----------|
| centroidal_control (pre-physics) | 4.33 | 3.9% |
| capture_control (pre-physics) | 1.10 | 1.0% |
| balance_core_block (controller compute) | 27.91 | 25.3% |
| centroidal_log (post-physics, DUPLICATE) | 4.51 | 4.1% |
| capture_log (post-physics, DUPLICATE) | 1.10 | 1.0% |
| telemetry (dict construction) | 11.23 | 10.2% |
| **Total per step (measured)** | **110.34** | 100% |

**Unaccounted (~60 ms):** Non-balance-core code (WBC, inverse dynamics, gravity computation),
MuJoCo physics stepping, Python overhead.

---

## Duplicate Call Analysis

### Call Sites

1. **Control-time** (pre-physics, line ~5430):
   ```python
   centroidal_state_control, control_com_pos = centroidal_estimator.estimate(
       jnp.zeros(42), mj_data, prev_control_com_pos
   )
   centroidal_state_control = capture_estimator.update(centroidal_state_control)
   ```

2. **Logging-time** (post-physics, line ~6637):
   ```python
   centroidal_state_log, logged_com_pos = centroidal_estimator.estimate(
       jnp.zeros(42), mj_data, control_com_pos
   )
   centroidal_state_log = capture_estimator.update(centroidal_state_log)
   ```

### Why They Differ

Both calls sample `mj_data` at different times relative to `mj_step()`:
- Control call: BEFORE physics stepping (pre-step state → feeds controller)
- Logging call: AFTER physics stepping (post-step state → feeds termination check + telemetry)

The `prev_com_pos` argument also differs:
- Control: `prev_control_com_pos` (previous step's control CoM)
- Logging: `control_com_pos` (current step's control CoM)

These produce different velocity estimates via finite-difference.

### Duplicate Overhead

| Call | ms/step |
|------|---------|
| centroidal_log (duplicate estimate) | 4.51 |
| capture_log (duplicate update) | 1.10 |
| **Total duplicate overhead** | **5.61 ms/step (5.1%)** |

### Why Duplicates Are NOT Removed

The `capture_estimator.update()` calls are NOT simply redundant with the capture-point
computation inside `centroidal_estimator.estimate()`. They use a different `min_height`:

| Component | min_height clamp |
|-----------|-----------------|
| `centroidal_estimator.estimate()` | `jnp.maximum(com_height, 0.1)` (hardcoded) |
| `capture_estimator.update()` | `max(com_height, config.min_height)` where `min_height=0.35` |

At K2's lowest operating height (0.33m), the omega computation differs:
- Centroidal: omega = sqrt(9.81 / 0.33) = 5.45
- Capture estimator: omega = sqrt(9.81 / 0.35) = 5.29

Removing the duplicate `capture_estimator.update()` calls would change the `capture_point`
and `divergence` values at low heights, affecting telemetry fields that feed into
K2 validation metrics.

### Future Option

If the min_height discrepancy is resolved (e.g., by unifying to the same value),
the duplicate capture_estimator.update() calls could be removed, saving ~1.1 ms/step
each. The duplicate centroidal_estimator.estimate() call is harder to eliminate because
it samples post-physics state needed for termination checking.

---

## Key Bottlenecks (for JAX port target)

1. **balance_core_block (27.9 ms):** Main controller compute — the primary JAX port target
2. **centroidal_estimate (4.3 + 4.5 = 8.8 ms):** Contact-force loop + orientation computation — candidate for JAX port or optimization
3. **telemetry (11.2 ms):** 1100+ column dict construction — plan to move to flat diag array in JAX path
4. **Unaccounted (~60 ms):** WBC, inverse dynamics, MuJoCo stepping — external to controller

---

## Test Results

| Test Suite | Tests | Result |
|-----------|-------|--------|
| `test_k2_best_current_promotion.py` | 20 | 20 passed |
| `test_current_best_controller_profile.py` | 9 | 9 passed |
| `test_stage1_no_duplicate_calls.py` | 7 | 7 passed |
| `test_stage1_behavior_unchanged.py` | 11 | 11 passed |
| **Total** | **47** | **47 passed, 0 failed** |

---

## Stage 1 Gate Status

- [x] Duplicate calls identified and profiled
- [x] Duplicate calls NOT removed (behavior would change at low heights)
- [x] Per-component profile breakdown published
- [x] Python K2 behavior unchanged (29 existing tests pass, 3-height smoke tests pass)
- [x] `--profile-controller` flag works and produces JSON report
- [x] Profile report at: `outputs/profile/stage1_controller_profile_breakdown.json`

**Stage 1 gate: PASSED.** Ready for Stage 2 (JAX notch + torque limiter parity).

---

*Report generated 2026-06-26. Stage 1 implementation complete.*
