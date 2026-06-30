# K2 JAX Full Semantic Closure — Phase 0 Baseline

**Date:** 2026-06-28
**Branch:** repo-cleanup-t6j
**Profile:** K2_NOTCH_LOW_Q_V1
**Backend status:** Python = default, JAX = opt-in

## Executive Summary

Baseline frozen before any code changes. Results confirm:

- **Fixed-height both-synced parity: PASS** (max diff ~1e-7 Nm, numerical precision)
- **Dynamic-height both-synced parity: PASS** (verified in prior validation gates)
- **Push both-synced parity: FAIL** (max diff 3.0 Nm at wheel indices [4,9])
- **Tests: pending** (running in background)

## Both-Synced Parity Results

### Fixed Height (high_0p480, 10 steps)

| Metric | Value |
|--------|-------|
| Max 10-dim abs_diff | 9.536743e-08 |
| Divergent actuator | 8 (r_knee) |
| Classification | K2_JAX_STATE_SYNCED_PARITY_PASS |

### Dynamic Height (ramp_up, ramp_down, gate_chatter)

| Scenario | Status |
|----------|--------|
| ramp_up (0.33→0.48) | PASS (prior gate) |
| ramp_down (0.48→0.33) | PASS (prior gate) |
| gate_chatter | PASS (prior gate) |

### Push Both-Synced

| Scenario | Max Diff | Divergent Actuator | First Divergent Step | Classification |
|----------|----------|---------------------|-----------------------|----------------|
| push_fwd_90N | 3.000000e+00 | 4 (l_wheel) / 9 (r_wheel) | 62 | FAIL |
| push_bwd_90N | 3.000000e+00 | 4 (l_wheel) / 9 (r_wheel) | ~62 | FAIL |

#### Push First-Divergence Trace (push_fwd_90N, push at steps 50-55)

| Step | Diff [4] (l_wheel) | Diff [9] (r_wheel) | tau_position_total py | tau_position_total jx |
|------|---------------------|---------------------|------------------------|------------------------|
| 62 | 3.0e-01 | 3.0e-01 | -4.44 Nm | -4.00 Nm |
| 63 | 6.0e-01 | 6.0e-01 | -4.94 Nm | -4.00 Nm |
| 64 | 9.0e-01 | 9.0e-01 | -5.43 Nm | -4.00 Nm |
| 65 | 1.2e+00 | 1.2e+00 | -5.91 Nm | -4.00 Nm |
| 89 | 3.0e+00 | 3.0e+00 | -11.64 Nm | -4.00 Nm |

Key observations:
1. Diff is SYMMETRIC at both wheels [4,9] — same magnitude, same sign
2. Diff grows linearly by 0.3 Nm per step until saturating at 3.0 Nm
3. Wheel diff is zero for all non-wheel joints — issue is specific to wheel torque path
4. Sagittal terms (tau_pitch, tau_pitch_rate, tau_sag_vel, tau_wheel_vel) are IDENTICAL between Python and JAX
5. tau_position_total differs because Python diag stores raw (pre-clip) value while JAX stores clipped value
6. APCR1ND state shows `recenter_held=0.0` at all divergent steps — APCR1ND gating is NOT activating

### Hip-Yaw Parity

| Scenario | Max Diff [1] | Max Diff [6] | Status |
|----------|-------------|-------------|--------|
| fixed_high_0p480 | ~0.0 | ~6.9e-18 | PASS |
| push_fwd_90N (pre-divergence) | ~0.0 | ~0.0 | PASS |

## Test Suite Status

Running 125 tests across:
- `test_k2_jax_step_parity.py` (17 tests)
- `test_k2_jax_component_parity.py` (108 tests including parametrized)
- `test_k2_jax_backend_cli.py`

Results pending — running in background.

## Code Freeze Verification

- No code changes made in this phase
- Both-synced parity testing uses `--controller-backend both-synced`
- Python remains default backend
- JAX remains opt-in
- Push parity failure confirmed at wheel indices [4,9] with 3.0 Nm max diff

## Known Current Issues (Pre-Fix)

1. **Push both-synced parity FAIL** — 3.0 Nm max diff at wheels [4,9], growing 0.3 Nm/step
2. **tau_position_total diagnostic mismatch** — Python diag stores raw value, JAX diag stores clipped value (diagnostic only, not a torque error)
3. **APCR1ND not activating during push** — `recenter_held=0.0` at all divergent steps (may be expected at step 62 given startup_guard=100)

## Next Steps

- Complete test suite baseline
- Phase 1: Build complete active K2 mechanism inventory
- Phase 2: Trace exact first-divergence scalar in push path
- Phase 3: Build push path gap matrix
- Phase 4: Implement proven fix
