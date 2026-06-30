# K2 JAX Release Hardening — Active Mechanism & Code Hygiene Audit

**Date:** 2026-06-28
**Phase:** 5
**Classification:** K2_JAX_RELEASE_HARDENING_ACTIVE_MECHANISM_AUDIT_PASS

---

## Active Mechanism Classification

| # | Mechanism | Classification | Notes |
|---|-----------|---------------|-------|
| 1 | effective_max_position_tau state passthrough | **PASS** | Index 834, gate correct: uses Python value when >0, falls back to JAX max_pos_tau |
| 2 | T6I/APCR1ND position cap behavior | **PASS** | `k2_jax_compute_boosted_position_cap` correct; two-stage clipping verified |
| 3 | APCR1ND wheel damping override | **PASS** | Band-based damping scale + minimum clamp; gated by module-level enabled flag |
| 4 | APCR1ND gating | **PASS** | Full state machine: startup guard, drift detection, safety gates, converging release |
| 5 | ABS trim | **PASS** | Ring buffer (300+100+500) correctly ported; sliding window, ZC guard, asymmetric rate limit |
| 6 | ABS height source com_z | **PASS** | Uses `com_z` directly; K2 profile `adaptive_bias_trim_replace_t6j=True` — com_z is correct for K2 |
| 7 | two-stage tau_position clipping | **PASS** | Stage 1: clip to `effective_max_position_tau` inside assembly; Stage 2: clip to `_boosted_cap` |
| 8 | torque composer clip + rate-limit | **PASS** | Correctly clips to torque_limit then rate-limits using max_torque_rate |
| 9 | prev_tau state sync | **PASS** | Indices 4-13 read from state, passed to composer, updated with tau_final |
| 10 | dynamic height scheduling | **PASS** | `continuous_max_position_tau=True` correctly scheduled; others (k_position, k_velocity, kd_pitch, k_wheel_velocity) correctly NOT scheduled per K2 profile |
| 11 | mode-div | **PASS** | Anti-symmetric hip-yaw with height gate; ref_source guard prevents crash on zero_only_for_debug |
| 12 | yaw | **PASS** | Anti-symmetric hip-yaw PD with clip |
| 13 | support FF | **INACTIVE_PROVEN** | Defined but correctly excluded (intentionally not added to tau_sum; comment explains Python has no equivalent) |
| 14 | hidden torque/WBC inactive path | **PASS** | No WBC code paths; confirmed by branch audit test |

**All required classifications achieved:**
- ✅ No PARTIAL
- ✅ No MISSING
- ✅ No WRONG
- ✅ No UNTESTED
- ✅ No UNKNOWN

---

## Code Hygiene Findings

### STATE LAYOUT (835 fields)

| Range | Indices | Description | Status |
|-------|---------|-------------|--------|
| 0-3 | 4 | Notch filter state | ✅ Active |
| 4-13 | 10 | prev_tau (composer) | ✅ Active |
| 14 | 1 | filtered_com_z | ✅ Active |
| 15 | 1 | prev_support_error | ⚠️ Read but never used in computation |
| 16-18 | 3 | Outer loop state | ✅ Active |
| 19-27 | 9 | ABS core fields | ✅ Active |
| 28-327 | 300 | ABS slow ring buffer | ✅ Active |
| 328 | 1 | ZC buf count | ✅ Active |
| 329 | 1 | ZC buf ptr | ✅ Active |
| 330-829 | 500 | ZC ring buffer | ✅ Active |
| 830-833 | 4 | APCR1ND gating state | ✅ Active |
| 834 | 1 | effective_max_pos_tau_py | ✅ Active |

### INPUT LAYOUT (42 fields)

| Index | Field | Status |
|-------|-------|--------|
| 0 | pitch_x_rad | ✅ Active |
| 1 | pitch_rate_rad_s | ✅ Active |
| 2 | roll_y_rad | ✅ Active |
| 3 | roll_rate_rad_s | ✅ Active |
| 4 | yaw_err_rad | ✅ Active |
| 5 | yaw_rate_rad_s | ✅ Active |
| 6 | com_z_m | ✅ Active |
| 7 | com_vy_m_s | ⚠️ Unpacked but never consumed |
| 8 | sagittal_velocity_m_s | ✅ Active |
| 9 | sagittal_position_error_m | ✅ Active |
| 10 | wheel_vel_left_rad_s | ✅ Active |
| 11 | wheel_vel_right_rad_s | ✅ Active |
| 12 | support_velocity_m_s | ✅ Active |
| 13 | commanded_height_ref_m | ✅ Active |
| 14 | hip_yaw_div_error | ✅ Active |
| 15 | hip_yaw_div_rate | ✅ Active |
| 16-23 | q (8 joints) | ✅ Active |
| 24-31 | qd (8 joints) | ✅ Active |
| 32-39 | q_ref (8 joints) | ✅ Active |
| 40 | support_pos_err | ✅ Active |
| 41 | contact_valid | ✅ Active |

### PARAMS LAYOUT (41 STAGE2 + 7 EXT = 48)

- All 41 STAGE2 params actively used
- Extended indices 41-47 (APCR1ND caps/bands): defined as module constants but read from K2 profile dataclass at module load time, not from params_flat. These indices are unused in the step function — they exist for future config-driven initialization.

### CODE HYGIENE ISSUES FOUND

| # | Finding | Severity | Recommendation |
|---|---------|----------|----------------|
| 1 | Stale comments: three places say "(332,)" but actual state size is 835 | Low | Fix comments to say "(835,)" |
| 2 | `prev_support_error` (index 15) read from state but never used in computation | Medium | Remove or repurpose; currently writes support_pos_err redundantly |
| 3 | `com_vy_m_s` (input index 7) unpacked but never consumed | Medium | Remove from input or use for future COM velocity terms |
| 4 | Orphaned param indices 41-47 defined but not read from params_flat | Low | Either populate in pack_params or remove indices |
| 5 | Multiple K2-specific values hardcoded (kp_pitch=50.0, kd_com_vy=5.0, kpos=40.0, kwheel=0.5, kd_pitch=10.0) instead of reading from K2 profile | Medium | Read from profile dataclass for self-documenting code |
| 6 | `pack_params_k2` referenced in comment but not defined anywhere | Trivial | Fix comment to reference `pack_params_stage2` |

### POSITIVE FINDINGS

- ✅ No state overlap (all 835 indices are non-overlapping)
- ✅ No unused state fields beyond the one identified
- ✅ No stale index constants (all constants match layout)
- ✅ No scenario-specific branch (no if-else on scenario name)
- ✅ No Python final torque copied to JAX (JAX computes independently)
- ✅ No Python behavior changes
- ✅ Python default preserved
- ✅ JAX opt-in preserved
- ✅ No hidden torque/WBC active path

---

## Verdict

**Classification: K2_JAX_RELEASE_HARDENING_ACTIVE_MECHANISM_AUDIT_PASS**

All 14 active mechanisms are PASS or INACTIVE_PROVEN. No blocking code hygiene issues. The 6 findings are all Low/Medium severity, cosmetic or minor, and none affect control correctness.
