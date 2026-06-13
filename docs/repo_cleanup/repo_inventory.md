# Repository Inventory

## Date: 2026-06-13

---

## A. Core Runtime Code

### A.1 Package Modules (`wheeled_biped/`)
**Total: ~70 files**

```
wheeled_biped/
├── __init__.py
├── controllers/
│   ├── __init__.py
│   ├── action_codec.py              # Residual action composition
│   ├── balance_core_torque_composer.py
│   ├── balance_core_types.py         # Shared types/constants
│   ├── balance_ik.py
│   ├── capture_point_estimator.py
│   ├── centroidal_balance_controller.py
│   ├── centroidal_state_estimator.py
│   ├── centroidal_wrench_computer.py
│   ├── contact_jacobian.py
│   ├── contact_supervisor.py
│   ├── dual_rate_balance_controller.py
│   ├── force_distribution.py
│   ├── hierarchical_vmc_lqr.py
│   ├── height_ik.py
│   ├── integrated_wbc.py
│   ├── lateral_roll_balance_controller.py
│   ├── leg_position_controller.py
│   ├── lqr_balance.py               # Legacy LQR controller
│   ├── lqr_ik_prior.py              # Height-scheduled LQR/IK prior
│   ├── momentum_coordinator.py
│   ├── orientation_utils.py
│   ├── pitch_rate_consistency_estimator.py
│   ├── position_hold_capture_gate.py
│   ├── posture_regularizer.py
│   ├── qp_allocator.py
│   ├── robot_model_utils.py
│   ├── sagittal_balance_state.py
│   ├── sagittal_velocity_damped_balance_controller.py  # T6J lives here
│   ├── sagittal_wheel_balance_controller.py
│   ├── shape_posture_controller.py
│   ├── simple_force_distributor.py
│   ├── stage2b_roll_direct_controller.py
│   ├── stage2b_sagittal_wheel_controller.py
│   ├── stage2c_sagittal_state_feedback_controller.py
│   ├── stage2d_sagittal_lqr_controller.py
│   ├── static_balance_controller.py
│   ├── static_feedforward_controller.py
│   ├── static_posture_holding_controller.py
│   ├── support_feedforward_controller.py
│   ├── torque_ownership_validator.py
│   ├── unified_force_distributor.py
│   ├── wbc_balance_controller.py
│   ├── wheel_balancing_controller.py
│   ├── yaw_controller.py
│   └── ... (more)
├── envs/
│   ├── __init__.py
│   ├── balance_env.py               # Pure PPO balance (42-dim obs)
│   ├── base_env.py                  # MJX base environment
│   ├── locomotion_env.py
│   ├── residual_balance_env.py      # Residual PPO (52-dim obs)
│   ├── standup_env.py
│   ├── stair_env.py
│   ├── terrain_env.py
│   └── walking_env.py
├── eval/
│   ├── __init__.py
│   ├── baseline.py
│   ├── benchmark.py
│   ├── latex_table.py
│   └── standing_quality.py
├── evaluation/
│   ├── __init__.py
│   └── controller_eval.py
├── inference/
│   ├── __init__.py
│   └── unified_controller.py        # Multi-skill controller
├── rewards/
│   ├── __init__.py
│   └── reward_functions.py
├── sim/
│   ├── __init__.py
│   ├── domain_randomization.py
│   ├── hierarchical_torque_fusion.py
│   ├── low_level_control.py
│   ├── push_disturbance.py
│   ├── terrain_generator.py
│   └── torque_wbc.py
├── training/
│   ├── __init__.py
│   ├── curriculum.py
│   ├── live_viewer.py
│   ├── networks.py
│   └── ppo.py
├── utils/
│   ├── __init__.py
│   ├── config.py
│   ├── logger.py
│   ├── math_utils.py
│   └── telemetry.py
└── validation/
    ├── __init__.py
    ├── balance_core_validator.py
    ├── classification_report.py
    ├── failure_classifier.py
    ├── fix_cycle_reporter.py
    ├── physical_standing_height_envelope.py
    ├── step_c_height_recovery.py
    ├── step_e_audit_helpers.py
    ├── structural_invariant_checker.py
    ├── study_aggregator.py
    ├── telemetry_adapter.py
    └── telemetry_schema_checker.py
```

### A.2 Config Files (`configs/`)
```
configs/
├── robot.yaml                      # Robot physics parameters
├── curriculum.yaml                 # Multi-stage pipeline
├── baseline_lqr.yaml               # Classical LQR baseline
├── controllers/
│   ├── balanced_posture_table.yaml
│   ├── gain_scheduled_lqr.yaml
│   ├── height_scheduled_com_lqr.yaml
│   ├── height_scheduled_com_lqr_tuned.yaml
│   ├── height_scheduled_dynamic_lqr.yaml
│   ├── prior_variants.yaml
│   ├── b9_balanced_root_init_table.yaml
│   ├── dual_rate_balance_controller_b9.yaml
│   ├── dual_rate_balance_controller_b9_fixed.yaml
│   ├── dual_rate_balance_controller_b9_pitch_only.yaml
│   ├── height_ik_wheel_lqr_only_b8.yaml
│   ├── hierarchical_vmc_lqr.yaml
│   ├── hierarchical_vmc_lqr_v2.yaml
│   ├── hierarchical_vmc_lqr_v3.yaml
│   ├── momentum_coordinator.yaml
│   ├── posture_regularizer.yaml
│   ├── step5_26_candidate_1_com.yaml
│   ├── step5_26_candidate_2_com_cp.yaml
│   └── (7 ablation configs)
└── training/
    ├── balance.yaml                 # Pure PPO baseline
    ├── balance_robust.yaml          # Push recovery fine-tune
    ├── balance_residual.yaml        # Residual PPO
    ├── balance_residual_robust.yaml
    ├── stand_up.yaml
    ├── wheeled_locomotion.yaml
    ├── walking.yaml
    ├── stair_climbing.yaml
    └── rough_terrain.yaml
```

### A.3 CLI Entry Points (`scripts/`)
**336 script files total. Key production scripts:**
```
scripts/
├── train.py                        # Training entry point
├── evaluate.py                     # Evaluation (4 modes)
├── eval_balance.py                 # Research evaluation
├── visualize.py                    # MuJoCo viewer/render
├── validate_checkpoint.py          # Standing validation
├── export_results.py               # CSV/PNG/LaTeX export
├── compare_baseline.py             # Regression comparison
├── analyze_residual.py             # Residual diagnostics
├── simulate_hierarchical_controller.py  # Main simulation script (uses T6J)
└── run_t6j_height_ladder.py        # T6J height ladder runner
```

---

## B. Current Best Controller: T6J_centering_bias_trim

### B.1 Required Files for T6J

| File | Purpose | Status |
|------|---------|--------|
| `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` | T6J profile + controller class | **KEEP** |
| `wheeled_biped/controllers/balance_core_types.py` | Shared types/constants | **KEEP** |
| `wheeled_biped/controllers/position_hold_capture_gate.py` | Capture gate | **KEEP** |
| `scripts/simulate_hierarchical_controller.py` | Main simulation runner | **KEEP** |
| `scripts/run_t6j_height_ladder.py` | T6J validation runner | **KEEP** |
| `scripts/analyze_t6j_height_ladder.py` | T6J analysis | **KEEP** |
| `tests/test_t6j_centering_bias_trim.py` | T6J unit tests | **KEEP** |

### B.2 T6J Dependencies (internal)
- `T6I_PHASE_AWARE_RELEASE` (for inheritance test)
- `T6F_BUDGET_CAP_RAISE` (for architecture fix)
- `APCR1ND_*` profiles
- `JOINT_FIX_PROFILES` registry

### B.3 T6J Telemetry Fields
All fields defined in `SagittalAuthoritySchedule` dataclass:
- `t6j_bias_trim_*` (12 fields)
- `t6i_*` (6 fields)
- `apcr1nd_*` (18 fields)
- `arch_fix_*` (6 fields)
- Standard velocity damping / position / pitch fields

---

## C. Required Baselines/Fallbacks

### C.1 T6I_phase_aware_release
**Reason**: T6J inherits from T6I for several fields
**Files Required**:
- `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` (contains T6I definition)
- `docs/validation/t6i_*.md` (validation reports)
- `tests/test_t6h_t6i_variants.py` (tests both T6H and T6I)

**Decision**: Keep T6I in the same file. Cannot remove without refactoring T6J.

### C.2 T6F_budget_cap_raise
**Reason**: T6I and T6J inherit `arch_fix` from T6F
**Files Required**:
- Same controller file
- `docs/validation/t6f_*.md` (validation reports)

**Decision**: Keep in same file. Architecture fix is foundational.

### C.3 APCR1nd_* profiles
**Reason**: T6I and T6J reference APCR1nd profiles
**Decision**: Keep in same controller file.

---

## D. Failed/Obsolete Controller Experiments

### D.1 T6H_soft_blend_arch_fix
- **Status**: Failed design
- **Evidence**: Superseded by T6I_phase_aware_release
- **Files**: Only in `sagittal_velocity_damped_balance_controller.py`
- **Tests**: `test_t6h_t6i_variants.py`
- **Docs**: `docs/validation/t6h_*.md`

**Decision**: Archive docs, keep profile in code (no harm, used in tests).

### D.2 T6F_sign_corrected
- **Status**: Failed design
- **Evidence**: Superseded by architecture fix approach
- **Files**: Only in `sagittal_velocity_damped_balance_controller.py`
- **Docs**: `docs/validation/t6f_sign_corrected_*.md`

**Decision**: Archive docs, keep profile in code.

### D.3 Old APCR1 Variants (APCR1a through APCR1nD)
- **Status**: All superseded by APCR1nd + T6 series
- **Docs**: 50+ validation reports in `docs/validation/`
- **Decision**: Archive all APCR1* docs except latest summaries.

### D.4 Old T1-T6 Variants
- T6A, T6B, T6C, T6D, T6E: Keep in controller file (historical reference)
- T5 variants: Archive docs

---

## E. Tests

### E.1 Essential Tests (Must Keep)
```
tests/
├── test_action_codec.py            # Residual action composition
├── test_residual_balance_env.py    # Residual environment
├── test_validate_checkpoint.py      # Checkpoint validation
├── test_validate_checkpoint_residual.py
├── test_ppo_trainer.py             # PPO invariants
├── test_curriculum.py              # Curriculum logic
├── test_benchmark.py               # Benchmark semantics
├── test_lqr_controller.py          # LQR controller
├── test_height_scheduled_com_lqr.py
├── test_lqr_ik_prior.py
├── test_lqr_ik_prior_variants.py
├── test_unified_controller.py      # Multi-skill controller
├── test_model.py                   # MuJoCo model integrity
├── test_rewards.py                 # Reward functions
├── test_sim_helpers.py              # Push/PID helpers
├── test_noise_and_dr.py             # Sensor noise + DR
├── test_standing_quality.py        # Quality signals
├── test_baseline.py                # Baseline comparison
├── test_latex_table.py              # LaTeX table generator
├── test_eval_balance.py             # eval_balance.py structures
├── test_telemetry.py               # Telemetry schema
├── test_t6j_centering_bias_trim.py # T6J unit tests
├── test_t6h_t6i_variants.py        # T6H/T6I tests
├── test_t6f_torque_sign_convention.py
├── test_qp_allocator.py
├── test_simulation_telemetry_csv_writer.py
├── test_b9_static_posture.py
├── test_phase_b9_step4_slow_loop_gating.py
├── test_phase_b9_step5_10_logic.py
├── test_phase_b9_step5_11_corrective_path_audit.py
├── test_phase_b9_step5_13_reset_equilibrium_fix.py
├── test_phase_b9_step5_14_lateral_balance_layer.py
├── test_phase_b9_step5_15_vmc_whole_body_layer.py
├── test_phase_b9_step5_16_jacobian_wbc_vmc.py
├── test_phase_b9_step5_17_torque_level_wbc.py
├── test_phase_b9_step5_18_deployable_motor_torque_interface.py
├── test_phase_b9_step5_18b_hybrid_pid_torque_rollout.py
├── test_phase_b9_step5_18c_torque_gain_saturation_calibration.py
├── test_phase_b9_step5_19_controller_authority_reallocation.py
├── test_phase_b9_step5_20_low_stiffness_dynamic_balance.py
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
└── test_stage2b_a_parity.py
```

### E.2 Obsolete Tests (Consider Archive)
```
tests/test_env.py                   # Full MJX rollout (slow, excluded from CI)
tests/test_smoke_train.py           # Smoke train (marked slow)
```
Note: These are NOT obsolete - they are production tests but excluded from fast CI.

### E.3 Total Test Count
- **129 test files**
- **1856 test functions**

---

## F. Validation Reports

### F.1 Keep (Final Reports)
```
docs/validation/
├── t6j_centering_bias_trim_final_report.md    # T6J final validation
├── t6j_height_ladder_validation_report.md     # T6J height ladder
├── t6j_centering_bias_trim_design.md           # T6J design doc
├── t6j_centering_bias_trim_implementation_tests_report.md
├── t6i_full_staged_validation_final_report.md  # T6I final
├── t6i_height_ladder_validation_summary.md
├── current_balance_core_roadmap_status.md
├── step_c_height_recovery_done.md
├── step_e_done_2026-06-01.md
├── step_e_height_variant_robustness_done.md
├── physical_standing_height_envelope_validation.md
├── hip_yaw_disturbance_rejection_final_report_v2.md
├── hip_yaw_integration_fix_complete.md
└── (other current active controller reports)
```

### F.2 Archive (Obsolete/Failed Experiments)
```
docs/validation/
├── apcr1*_*                          # 50+ obsolete APCR1 variant reports
├── t5_*                              # T5 series reports
├── t6_*_old_versions                 # Old version reports
├── t6f_sign_corrected_*              # Failed sign corrected design
├── t6h_t6i_*                         # Superseded by T6I final
├── e1_*, e2_*, f1_*, f2_*, g1_*      # Strategy iteration reports
└── (other superseded experiment reports)
```

**Total validation reports: ~250**
**Keep: ~50 (current active reports)**
**Archive: ~200 (obsolete experiments)**

### F.3 Failed Experiment Reports (Archive)
- `t6f_sign_corrected_*.md` - Failed design
- `t6h_*.md` - Superseded
- `apcr1a_*.md` through `apcr1n_*.md` - All superseded by APCR1nd/T6J
- `e1_*`, `e2_*` early strategy iteration reports

---

## G. Outputs/Data

### G.1 Gitignored Outputs (No Action Needed)
```
outputs/                  # All training outputs (in .gitignore)
outputs_residual_smoke/  # Smoke test outputs (in .gitignore)
outputs_residual_test/   # Test outputs (in .gitignore)
runs/                    # TensorBoard (in .gitignore)
wandb/                   # WandB logs (in .gitignore)
logs/                    # Logs (in .gitignore)
MUJOCO_LOG.TXT           # MuJoCo log (in .gitignore)
```

### G.2 Tracked Output Directories
```
backup_checkpoints/      # Day1, Day2 backups - check if needed
baselines/               # Empty directory
```

### G.3 CSV/Telemetry Files
**Find tracked CSV files:**
- Check if any CSV files are tracked (should be in outputs)

---

## H. Training

### H.1 Training Code (Keep)
```
wheeled_biped/training/
├── ppo.py                # PPO + obs normalisation
├── networks.py            # Actor-Critic (Flax)
├── curriculum.py          # Curriculum manager
└── live_viewer.py         # Real-time viewer

configs/training/
├── balance.yaml
├── balance_robust.yaml
├── balance_residual.yaml
├── balance_residual_robust.yaml
└── (other stage configs)

scripts/train.py
scripts/evaluate.py
scripts/eval_balance.py
scripts/validate_checkpoint.py
scripts/export_results.py
scripts/compare_baseline.py
```

### H.2 Training Configs (Keep)
- `balance.yaml` - Pure PPO baseline
- `balance_robust.yaml` - Push recovery
- `balance_residual.yaml` - Residual PPO (Phase C)
- `balance_residual_robust.yaml`
- `stand_up.yaml`
- Other stage stubs

### H.3 Training Data (Gitignored)
```
outputs/balance/rl/seed42/...
outputs/balance/rl/seed113/...
outputs/balance/rl/seed999/...
outputs/balance_robust/...
outputs/<stage>/...
```
All training outputs are gitignored.

---

## I. Scripts

### I.1 Production Scripts (Keep)
```
scripts/train.py                    # Main training entry
scripts/evaluate.py                 # 4-mode evaluation
scripts/eval_balance.py              # Research evaluation
scripts/visualize.py                # MuJoCo viewer/render
scripts/validate_checkpoint.py      # Standing validation
scripts/export_results.py           # CSV/PNG/LaTeX
scripts/compare_baseline.py         # Regression
scripts/analyze_residual.py          # Residual diagnostics
scripts/simulate_hierarchical_controller.py  # Main sim (uses T6J)
scripts/run_t6j_height_ladder.py     # T6J height ladder
scripts/analyze_t6j_height_ladder.py # T6J analysis
```

### I.2 Diagnostic/Analysis Scripts (Archive)
**336 total scripts, many are one-off diagnostics:**

#### Archive candidates:
- `analyze_apcr1*.py` - APCR1 variant analysis
- `analyze_t6f*.py` - T6F analysis
- `analyze_t6h*.py` - T6H analysis  
- `audit_apcr*.py` - APCR audits
- `audit_t6f*.py` - T6F audits
- `check_*.py` - One-off checks
- `debug_*.py` - Debug scripts
- `diagnose_*.py` - Diagnosis scripts
- `compare_*.py` - Comparisons (keep essential ones)
- `evaluate_*.py` - Old eval scripts (consolidate)
- `find_*.py` - Equilibrium finding (keep if still useful)
- `phase_b9_*.py` - Phase B9 development
- `screening_*.py` - Screening scripts
- `validate_*.py` - Validation (consolidate)
- `verify_*.py` - Verification (consolidate)

### I.3 Scripts Count Breakdown
- **Keep (production)**: ~10-15 scripts
- **Archive (one-off diagnostics)**: ~300+ scripts
- **Unknown/Review**: ~20 scripts

---

## J. Documentation

### J.1 Keep (Core Documentation)
```
README.md                           # Main README
CLAUDE.md                           # Project instructions
docs/baseline_workflow.md           # Baseline workflow
docs/paper_claims_checklist.md      # Paper claims
docs/results_todo_checklist.md       # Results TODO
docs/experiment_plan.md             # Experiment plan
docs/validation/current_balance_core_roadmap_status.md
```

### J.2 Keep (T6J Documentation)
```
docs/validation/t6j_centering_bias_trim_final_report.md
docs/validation/t6j_height_ladder_validation_report.md
docs/validation/t6j_centering_bias_trim_design.md
docs/validation/t6j_centering_bias_trim_implementation_tests_report.md
```

### J.3 Keep (T6I Documentation - for T6J dependency)
```
docs/validation/t6i_full_staged_validation_final_report.md
docs/validation/t6i_height_ladder_validation_summary.md
docs/validation/t6i_high_0p480_full_validation_summary.md
```

### J.4 Archive (Obsolete Development Docs)
```
docs/phase_b*.md                    # Old phase reports (B2-B9)
docs/task_*.md                      # Task reports
docs/upstream_controller_debug_plan.md
docs/superpowers/                   # All superpowers docs (development)
docs/validation/apcr1*_*             # 50+ APCR1 reports
docs/validation/t5_*                 # T5 reports
docs/validation/t6f_sign_corrected_* # Failed design
docs/validation/t6h_*                # Superseded by T6I
docs/validation/e1_*, e2_*, f1_*    # Early strategy iterations
docs/validation/phase_*              # Phase reports
```

### J.5 Documentation Count
- **Total**: 339 markdown files (excluding .claude/worktrees)
- **Keep**: ~30 (core + T6J + T6I + current active)
- **Archive**: ~300+ (obsolete experiments, development docs)

---

## K. Summary Statistics

| Category | Count | Keep | Archive | Delete |
|----------|-------|------|---------|--------|
| Controllers (`.py`) | ~70 | ~70 | 0 | 0 |
| Test files | 129 | 129 | 0 | 0 |
| Scripts | 336 | ~15 | ~300 | ~20 |
| Docs (`.md`) | 339 | ~50 | ~280 | 9 |
| Config files | ~35 | ~35 | 0 | 0 |
| Validation reports | ~250 | ~50 | ~200 | 0 |

---

*Generated: 2026-06-13*