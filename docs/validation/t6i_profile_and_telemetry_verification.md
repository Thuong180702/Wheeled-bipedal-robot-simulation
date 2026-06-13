# T6I Profile and Telemetry Verification

**Date:** 2026-06-13
**Classification:** T6I_PROFILE_TELEMETRY_READY

## 1. Profile Existence

| Item | Status |
|------|--------|
| Profile `T6I_phase_aware_release` in controller registry | ✅ Found at `sagittal_velocity_damped_balance_controller.py:1658` |
| Profile `T6I_phase_aware_release` in CLI registry | ✅ Found at `simulate_hierarchical_controller.py:1332` |
| CLI argument `--vd-sagittal-authority-profile` accepts `T6I_phase_aware_release` | ✅ Confirmed at `simulate_hierarchical_controller.py:2630` |

## 2. Profile Identity Telemetry Fields

| Field | Status | Location |
|-------|--------|----------|
| `vd_sagittal_authority_profile` | ✅ Written at step level | `simulate_hierarchical_controller.py:5106` |
| `controller_mode` | ✅ Written at step level | `simulate_hierarchical_controller.py:5104` |
| `sagittal_controller` | ✅ Written at step level | via `--sagittal-controller velocity-damped` |
| `height_variant_setup_name` | ✅ Written at step level | `simulate_hierarchical_controller.py:5109-5113` |

## 3. T6I-Specific Telemetry Fields

| Field | Status | Location |
|-------|--------|----------|
| `t6i_error_converging` | ✅ | `sagittal_velocity_damped_balance_controller.py:4278` |
| `t6i_error_trend` | ✅ | `sagittal_velocity_damped_balance_controller.py:4279` |
| `t6i_target_cap` | ✅ | `sagittal_velocity_damped_balance_controller.py:4280` |
| `t6i_current_cap` | ✅ | `sagittal_velocity_damped_balance_controller.py:4281` |
| `t6i_cap_delta_this_step` | ✅ | `sagittal_velocity_damped_balance_controller.py:4282` |
| `t6i_cap_change_rate_limited` | ✅ | `sagittal_velocity_damped_balance_controller.py:4283` |
| `t6i_release_reason` | ✅ | `sagittal_velocity_damped_balance_controller.py:4284` |

## 4. Common Torque and Drift Fields

| Field | Status | Location |
|-------|--------|----------|
| `active_pitch_crossing_signed_error_m` | ✅ Column priority used in analysis | `sagittal_velocity_damped_balance_controller.py:2730` |
| `sagittal_position_error_m` | ✅ | `sagittal_velocity_damped_balance_controller.py:4190` |
| `support_position_error_m` | ✅ | Present in telemetry dict |
| `hip_yaw_comp_support_error_m` | ✅ | Present in telemetry dict |
| `final_wheel_tau_with_apc` | ✅ | `sagittal_velocity_damped_balance_controller.py:4518` |
| `arch_fix_active` | ✅ | `sagittal_velocity_damped_balance_controller.py:4229` |
| `effective_max_position_tau_after_arch_fix` | ✅ | `sagittal_velocity_damped_balance_controller.py:4236` |
| `ownership_violation_count_max` | ✅ | `simulate_hierarchical_controller.py:3953` |
| `hidden_torque_norm_max` | ✅ | `simulate_hierarchical_controller.py:3954` |
| WBC flag | ✅ `per_actuator_wbc_authority_enabled` tracked | `simulate_hierarchical_controller.py:1712` |

## 5. Setup File

| File | Status |
|------|--------|
| `outputs/physical_target_height_setups/high_0p480_setup.json` | ✅ Exists |

## 6. Test Suite

| Suite | Result |
|-------|--------|
| `test_t6h_t6i_variants.py` | 38 passed |
| `test_t6_high_height_variants.py` | 36 passed |
| `test_t6f_torque_sign_convention.py` | 16 passed |
| `test_apcr1nd_tuned_variants.py` | 31 passed |
| `test_sagittal_velocity_damped_balance_controller.py` | 285 passed |
| `test_simulation_telemetry_csv_writer.py` | 9 passed |
| `test_low_height_setup_initialization.py` | 9 passed |
| `test_step_e_wbc_gate_validator.py` | 4 passed |
| **Total** | **428 passed, 0 failed** |

## 7. Decision

**T6I_PROFILE_TELEMETRY_READY** — All required telemetry fields verified present in code, profile registered in both controller and CLI registries, setup file exists, all tests pass. Proceed to Phase 2.
