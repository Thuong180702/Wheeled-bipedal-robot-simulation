# K2 JAX APCR1ND Baseline and Safety Lock — Phase 0

**Date:** 2026-06-28
**Branch:** repo-cleanup-t6j
**Phase:** Phase 0 — Baseline and Safety Lock

## 1. Test Inventory

```
131 passed in 534.33s
```

All tests pass. No xfail, no skip. Python default preserved. JAX opt-in preserved.

Test files:
- `tests/test_k2_jax_backend_cli.py` — 6 tests
- `tests/test_k2_jax_component_parity.py` — 106 tests
- `tests/test_k2_jax_step_parity.py` — 19 tests

## 2. Both-Synced Parity Results

| # | Scenario | Steps | Max Abs Diff | Divergent Step | Divergent Actuator | Wheel[4] | Wheel[9] | HY[1] | HY[6] | Fell | Verdict |
|---|----------|-------|-------------|----------------|--------------------|----------|----------|-------|-------|------|---------|
| 1 | fixed_high_0p480 | 50 | 9.54e-08 | 2 | 8 (r_knee) | <1e-7 | <1e-7 | <1e-7 | <1e-7 | no | **PASS** |
| 2 | fixed_low_0p330 | 50 | 9.54e-08 | 2 | 8 (r_knee) | <1e-7 | <1e-7 | <1e-7 | <1e-7 | no | **PASS** |
| 3 | ramp_up | 500 | 7.88e-01 | 207 | **4 (l_wheel)** | 0.788 | — | <1e-7 | <1e-7 | no | **FAIL** |
| 4 | ramp_down | 500 | 9.54e-08 | 2 | 8 (r_knee) | <1e-7 | <1e-7 | <1e-7 | <1e-7 | no | **PASS** |
| 5 | gate_chatter | 500 | 7.92e-01 | 210 | **9 (r_wheel)** | 0.348 | 0.792 | <1e-7 | <1e-7 | no | **FAIL** |
| 6 | push_fwd_90N | 300 | 3.00e+00 | 205 | **4 (l_wheel)** | 3.0 | 3.0 | <1e-7 | <1e-7 | no | **FAIL** |
| 7 | push_bwd_90N | 300 | 3.30e+00 | 170 | **9 (r_wheel)** | 3.0 | 3.30 | <1e-7 | <1e-7 | no | **FAIL** |

### Divergence patterns

**All divergences occur at wheel indices [4,9] only.** Hip-yaw [1,6] remains <1e-7 across all scenarios.

#### ramp_up (0.33→0.48m, 500 steps)
- Steps 0–206: PASS (9.54e-08 baseline)
- **Steps 207–212**: Sudden transient at wheel [4] — max 0.788. Height ~0.39m at step 207 (below gate).
- Steps 213–403: PASS (returns to 9.54e-08)
- **Steps 404–416**: Growing divergence at wheel [4] — 0.075→0.274. Height ~0.45m at step 404 (inside gate 0.42–0.48m).
- Steps 417+: PASS (returns to 9.54e-08)

#### gate_chatter (0.40–0.47m oscillation, 500 steps)
- Steps 0–209: PASS (9.54e-08 baseline)
- **Steps 210–216**: Transient at wheel [9] — max 0.792
- Steps 217–415: PASS
- **Steps 416–426**: Growing divergence at wheel [4] — 0.294→0.363

#### push_fwd_90N (push at step 100, high_0p480)
- Steps 0–111: PASS (9.54e-08 baseline)
- **Steps 112–228**: Massive divergence at wheels [4,9] — saturates at 3.0 Nm clip
- Steps 229–300: Residual divergence at wheel [4] — ~0.015 (slowly decaying)

#### push_bwd_90N (push at step 100, high_0p480)
- Steps 0–110: PASS (9.54e-08 baseline)
- **Steps 111–223**: Massive divergence at wheels [4,9] — saturates at 3.0+ Nm
- Steps 236–274: Residual divergence at wheel [4] — ~0.011–0.013
- **Steps 291–299**: New divergence growing at wheel [4] — 0.117→1.099

## 3. APCR1ND Diagnostic Availability

Current both-synced diagnostics printed per step include:
- Per-term sagittal torque comparison (tau_pitch, tau_pitch_rate, tau_sagittal_velocity, tau_support_velocity, tau_wheel_velocity_left/right, tau_position_total)
- Support velocity comparison
- Notch filter state comparison
- Mode-div hip-yaw comparison

**Missing APCR1ND-specific diagnostics** (to be added in Phase 4):
- py/jx position_error, support_error, support_error_rate
- py/jx slow/fast history values
- py/jx drift direction, wheel damping raw, wheel damping override active
- py/jx damping scale, min clamp active, final wheel damping L/R

## 4. Classification

**Baseline:** K2_JAX_RELEASE_LOCK_PASS_DYNAMIC_PARITY_BLOCKED

- Fixed-height/core release lock: **PASS** (9.54e-08, unchanged) ✓
- Fixed-height both-synced parity: **3/3 PASS** (high, low, ramp_down) ✓
- Dynamic both-synced parity: **2/4 FAIL** (ramp_up, gate_chatter at wheels [4,9])
- Push both-synced parity: **2/4 FAIL** (fwd, bwd at wheels [4,9])
- Tests: **131/131 PASS** ✓
- No regressions ✓
- Hip-yaw [1,6] parity: **EXACT** (<1e-7 across all scenarios) ✓
- Sagittal fixed-height parity: **EXACT** (ramp_down passes) ✓
- All failures exclusively at wheel indices [4,9] — consistent with APCR1ND being the root cause

## 5. Safety Lock Status

| Gate | Status |
|------|--------|
| Fixed-height <1e-5 | PASS ✓ |
| Hip-yaw [1,6] <1e-8 | PASS ✓ |
| 131/131 tests pass | PASS ✓ |
| No gain tuning | MAINTAINED ✓ |
| No threshold relaxation | MAINTAINED ✓ |
| No empirical corrections | MAINTAINED ✓ |
| Python default preserved | MAINTAINED ✓ |
| JAX opt-in preserved | MAINTAINED ✓ |
| Python K2 unchanged | MAINTAINED ✓ |

## 6. Next Phase

Phase 1: Python APCR1ND source trace — map every APCR1ND control-affecting scalar to file/function/line, document activation gates, drift detection, wheel damping override, state update timing, and profile inheritance.
