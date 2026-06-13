# Dependency and Usage Audit

## Date: 2026-06-13

---

## 1. T6J_centering_bias_trim Dependencies

### 1.1 Runtime Dependencies

| Symbol | Location | Used By | Keep? |
|--------|----------|---------|-------|
| `T6J_CENTERING_BIAS_TRIM` | sagittal_velocity_damped_balance_controller.py | simulate_hierarchical_controller.py, test_t6j_centering_bias_trim.py | **YES** |
| `T6I_PHASE_AWARE_RELEASE` | sagittal_velocity_damped_balance_controller.py | T6J (inherits fields), test_t6j_centering_bias_trim.py | **YES** |
| `T6F_BUDGET_CAP_RAISE` | sagittal_velocity_damped_balance_controller.py | T6I (arch_fix base), test_t6j_centering_bias_trim.py | **YES** |
| `APCR1ND_T1-T5` profiles | sagittal_velocity_damped_balance_controller.py | simulate_hierarchical_controller.py | **YES** |
| `JOINT_FIX_PROFILES` | sagittal_velocity_damped_balance_controller.py | test_t6j_centering_bias_trim.py | **YES** |
| `SagittalAuthoritySchedule` | sagittal_velocity_damped_balance_controller.py | All T6* profiles | **YES** |
| `SagittalVelocityDampedBalanceController` | sagittal_velocity_damped_balance_controller.py | simulate_hierarchical_controller.py | **YES** |
| `balance_core_types` | balance_core_types.py | simulate_hierarchical_controller.py | **YES** |
| `position_hold_capture_gate` | position_hold_capture_gate.py | sagittal_velocity_damped_balance_controller.py | **YES** |

### 1.2 Test Dependencies

| Symbol | Location | Tests | Keep? |
|--------|----------|-------|-------|
| `T6J_CENTERING_BIAS_TRIM` | sagittal_velocity_damped_balance_controller.py | test_t6j_centering_bias_trim.py | **YES** |
| `T6I_PHASE_AWARE_RELEASE` | sagittal_velocity_damped_balance_controller.py | test_t6j_centering_bias_trim.py, test_t6h_t6i_variants.py | **YES** |
| `T6F_BUDGET_CAP_RAISE` | sagittal_velocity_damped_balance_controller.py | test_t6j_centering_bias_trim.py | **YES** |
| `T6H_SOFT_BLEND_ARCH_FIX` | sagittal_velocity_damped_balance_controller.py | test_t6h_t6i_variants.py | **YES** |
| `JOINT_FIX_PROFILES` | sagittal_velocity_damped_balance_controller.py | test_t6j_centering_bias_trim.py | **YES** |

### 1.3 Documentation Dependencies

| Document | References | Keep? |
|----------|------------|-------|
| `t6j_centering_bias_trim_final_report.md` | T6J | **YES** |
| `t6j_height_ladder_validation_report.md` | T6J | **YES** |
| `t6j_centering_bias_trim_design.md` | T6J design | **YES** |
| `t6i_full_staged_validation_final_report.md` | T6I (T6J base) | **YES** |
| `t6i_height_ladder_validation_summary.md` | T6I | **YES** |
| `t6f_budget_cap_raise_implementation_report.md` | T6F | **ARCHIVE** |
| `t6h_t6i_*.md` | T6H (superseded) | **ARCHIVE** |

---

## 2. Obsolete Controller Dependencies

### 2.1 APCR1 Variants (APCR1a through APCR1nD)

**Status**: All superseded by APCR1ND + T6 series

**Usage Analysis**:
- Only referenced in `sagittal_velocity_damped_balance_controller.py`
- Not imported by any active scripts
- Validation reports in `docs/validation/apcr1*.md`

**Decision**: Keep code profiles (no harm), archive documentation.

### 2.2 T6F_sign_corrected

**Status**: Failed design

**Usage Analysis**:
- Referenced in `sagittal_velocity_damped_balance_controller.py`
- Imported by `simulate_hierarchical_controller.py` (but never actively used)
- Test: `analyze_t6f_sign_corrected_500.py`

**Decision**: Keep profile in code, archive all related docs.

### 2.3 T6H_soft_blend_arch_fix

**Status**: Failed design, superseded by T6I

**Usage Analysis**:
- Referenced in `sagittal_velocity_damped_balance_controller.py`
- Imported by `simulate_hierarchical_controller.py` (registry only)
- Test: `test_t6h_t6i_variants.py`

**Decision**: Keep profile in code, keep test, archive docs.

---

## 3. Script Usage Analysis

### 3.1 Production Scripts (Keep)
```
scripts/
├── train.py                        # Training - used
├── evaluate.py                     # Evaluation - used
├── eval_balance.py                 # Research eval - used
├── visualize.py                   # Viewer - used
├── validate_checkpoint.py          # Validation - used
├── export_results.py              # Export - used
├── compare_baseline.py             # Comparison - used
├── analyze_residual.py             # Residual analysis - used
├── simulate_hierarchical_controller.py  # Main sim - used
├── run_t6j_height_ladder.py        # T6J runner - used
└── analyze_t6j_height_ladder.py     # T6J analysis - used
```

### 3.2 Archive Candidates (No active usage)

| Pattern | Count | Reason |
|---------|-------|--------|
| `analyze_apcr*.py` | ~15 | APCR experiments superseded |
| `analyze_t6f*.py` | ~6 | T6F experiments superseded |
| `analyze_t6h*.py` | ~1 | T6H experiments superseded |
| `audit_apcr*.py` | ~5 | APCR audits superseded |
| `audit_t6f*.py` | ~2 | T6F audits superseded |
| `audit_t6*.py` | ~5 | T6 audits superseded |
| `debug_*.py` | ~30 | One-off debug scripts |
| `diagnose_*.py` | ~15 | One-off diagnosis |
| `check_*.py` | ~25 | One-off checks |
| `compare_*.py` | ~15 | Comparison scripts (keep essential) |
| `evaluate_*.py` | ~10 | Eval scripts (consolidate) |
| `find_*.py` | ~15 | Equilibrium finding (keep if useful) |
| `phase_b9_*.py` | ~25 | Phase B9 development |
| `screening_*.py` | ~5 | One-off screening |
| `validate_*.py` | ~15 | Validation (consolidate) |
| `verify_*.py` | ~10 | Verification (consolidate) |
| `compute_*.py` | ~10 | One-off computations |
| `boundary_*.py` | ~5 | One-off boundary analysis |
| `render_*.py` | ~1 | One-off rendering |
| `monitor_*.py` | ~1 | One-off monitoring |
| `regenerate_*.py` | ~1 | One-off regeneration |
| `smoke_test_*.py` | ~1 | One-off smoke test |

**Total archive candidates**: ~220 scripts

---

## 4. Test Usage Analysis

### 4.1 Tests by Controller
```
test_t6j_centering_bias_trim.py     # T6J unit tests
test_t6h_t6i_variants.py           # T6H, T6I tests
test_t6f_torque_sign_convention.py # T6F sign tests
```

### 4.2 Tests by Category
```
Core functionality (keep):
├── test_action_codec.py
├── test_residual_balance_env.py
├── test_ppo_trainer.py
├── test_curriculum.py
├── test_benchmark.py
├── test_unified_controller.py
├── test_model.py
├── test_rewards.py
├── test_sim_helpers.py
├── test_noise_and_dr.py
├── test_standing_quality.py
├── test_baseline.py
├── test_latex_table.py
├── test_eval_balance.py
├── test_telemetry.py
└── test_validate_checkpoint.py

Controller-specific (keep):
├── test_lqr_controller.py
├── test_height_scheduled_com_lqr.py
├── test_lqr_ik_prior.py
├── test_lqr_ik_prior_variants.py
├── test_qp_allocator.py
├── test_capture_point_estimator.py
├── test_centroidal_integration.py
├── test_centroidal_balance_controller.py
├── test_momentum_coordinator.py
├── test_unified_force_distributor.py
├── test_integrated_wbc.py
├── test_dual_rate_balance_controller.py
├── test_sim_hierarchical_wheel_torque.py
├── test_posture_regularizer.py
├── test_leg_position_controller.py
├── test_controller_root_cause_regressions.py
├── test_hierarchical_controller_integration.py
├── test_hierarchical_vmc_lqr.py
├── test_centroidal_state_estimator.py
├── test_simple_force_distributor.py
├── test_centroidal_wrench_computer.py
├── test_contact_jacobian.py
├── test_support_pipeline_regressions.py
├── test_com_velocity_lifecycle.py
├── test_initial_contact_calibration.py
├── test_contact_geometry.py
├── test_actuator_signs.py
├── test_static_balance_controller.py
├── test_static_balance_simulation.py
├── test_robot_mass_consistency.py
├── test_stage1_equilibrium_corrections.py
├── test_static_posture_holding_controller.py
├── test_static_feedforward_controller.py
├── test_stage2b_a_parity.py
├── test_validate_checkpoint_residual.py
├── test_b9_static_posture.py
├── test_phase_b9_step*.py (17 files)
└── test_simulation_telemetry_csv_writer.py

Slow tests (mark for manual run only):
├── test_env.py (full MJX JIT compile)
└── test_smoke_train.py (marked slow)
```

**All 129 test files should be kept** - they test various controller configurations and evolution. The tests for superseded controllers serve as regression tests and historical record.

---

## 5. Dependency Tables

### 5.1 Keep Required by Runtime
```
wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py
wheeled_biped/controllers/balance_core_types.py
wheeled_biped/controllers/position_hold_capture_gate.py
wheeled_biped/validation/telemetry_adapter.py
wheeled_biped/validation/study_aggregator.py
wheeled_biped/validation/balance_core_validator.py
```

### 5.2 Keep Required by Tests
```
tests/test_t6j_centering_bias_trim.py
tests/test_t6h_t6i_variants.py
tests/test_t6f_torque_sign_convention.py
(all other test files)
```

### 5.3 Keep Required by Documentation
```
docs/validation/t6j_*.md
docs/validation/t6i_*.md
docs/validation/current_balance_core_roadmap_status.md
docs/validation/step_c_*.md
docs/validation/step_e_*.md
docs/validation/hip_yaw_*.md
```

### 5.4 Archive Candidate
```
docs/validation/apcr1*.md (50+ files)
docs/validation/t5_*.md
docs/validation/t6f_sign_corrected_*.md
docs/validation/t6h_*.md
docs/validation/e1_*.md
docs/validation/e2_*.md
docs/validation/f1_*.md
docs/validation/f2_*.md
docs/validation/g1_*.md
docs/phase_b*.md
docs/superpowers/**/*.md
```

### 5.5 Delete Candidate
```
scripts/analyze_t6f_sign_fix_500_diagnostic.py  # T6F sign fix failed
scripts/analyze_t6f_sign_corrected_500.py       # T6F sign corrected failed
scripts/analyze_t6h_t6i_500.py                  # T6H superseded
scripts/analyze_t6j_500_diagnostic.py            # Obsolete - superseded by final
scripts/analyze_t6j_1200_diagnostic.py          # Obsolete - superseded by final
scripts/analyze_t6j_2000_diagnostic.py          # Obsolete - superseded by final
scripts/analyze_t6j_5000_diagnostic.py          # Obsolete - superseded by final
scripts/analyze_t6j_height_ladder.py             # Keep - still actively used
scripts/analyze_apcr1b_*.py                      # APCR1b superseded
scripts/analyze_apcr1e_*.py                      # APCR1e superseded
scripts/analyze_apcr1l_*.py                      # APCR1l superseded
scripts/analyze_apcr1*.py                        # All APCR1 superseded
scripts/audit_apcr*.py                           # Superseded
scripts/audit_t6f*.py                            # Superseded
scripts/audit_t6*.py                             # Superseded
scripts/debug_t6f*.py                            # Superseded
scripts/audit_t6f_sign_fix_*.py                  # Failed design
scripts/audit_t6f_pitch_*.py                     # Failed design
scripts/check_*.py                               # One-off checks
scripts/comprehensive_*.py                      # One-off comprehensive
scripts/debug_axis_*.py                          # One-off debug
scripts/debug_contact_*.py                       # One-off debug
scripts/debug_force_*.py                         # One-off debug
scripts/debug_gate_*.py                          # One-off debug
scripts/debug_geometry_*.py                      # One-off debug
scripts/debug_stage2*.py                         # One-off debug
scripts/debug_static_*.py                        # One-off debug
scripts/debug_wbc_*.py                           # One-off debug
scripts/debug_wheel_*.py                         # One-off debug
scripts/identify_*.py                            # One-off sysid
scripts/sweep_*.py                               # One-off sweep
```

---

## 6. Classification Summary

| Classification | Count | Action |
|----------------|-------|--------|
| KEEP: Required by runtime | 7 files | Keep |
| KEEP: Required by tests | 129 files | Keep |
| KEEP: Required by documentation | 50 files | Keep |
| KEEP: Production scripts | 15 files | Keep |
| ARCHIVE: Obsolete experiment scripts | ~220 files | Move to archive |
| ARCHIVE: Obsolete documentation | ~200 files | Move to archive |
| DELETE: Generated/temp files | TBD | Delete |

---

## 7. Key Findings

### 7.1 Controller Code is Cohesive
All T6* and APCR* profiles live in a single file (`sagittal_velocity_damped_balance_controller.py`), making removal or archiving simple - just remove the profile definitions and update the registry.

### 7.2 Tests are Comprehensive
129 test files test the full evolution of controller variants, serving as both validation and regression tests.

### 7.3 Scripts are the Main Cleanup Target
336 scripts with ~220 archive candidates is the largest cleanup opportunity.

### 7.4 Documentation Bloat is Significant
339 markdown files with ~200 archive candidates.

### 7.5 No Dangerous Dependencies
No controller profiles are actively used by production code outside of their test files and the simulation script. Profiles can be safely archived without breaking runtime.

---

*Generated: 2026-06-13*