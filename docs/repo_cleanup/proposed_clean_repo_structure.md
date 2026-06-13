# Proposed Final Clean Repository Structure

## Date: 2026-06-13

---

## Target State Overview

After cleanup, the repository will have:
- **~832 tracked files** (down from 1072)
- Clean separation between production code and archived experiments
- No run logs or generated artifacts
- Well-organized documentation

---

## Proposed Directory Structure

```
Wheeled-bipedal-robot-simulation/
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── .gitignore                    # Updated (add export.log exclusion)
│
├── README.md                      # Main documentation
│├── CLAUDE.md                     # Project instructions
│├── pyproject.toml                # Package configuration
│├── pytest.ini                    # Test configuration
│
├── assets/
│   ├── robot/
│   │   └── wheeled_biped_real.xml
│   └── robot-urdf/
│       ├── urdf/
│       │   ├── HOANTHIEN_TEST.urdf
│       │   └── HOANTHIEN_TEST.csv
│       ├── meshes/               # 11 STL files
│       ├── config/
│       ├── launch/
│       ├── CMakeLists.txt
│       └── package.xml
│
├── configs/
│   ├── robot.yaml
│   ├── curriculum.yaml
│   ├── baseline_lqr.yaml
│   ├── training/
│   │   ├── balance.yaml
│   │   ├── balance_robust.yaml
│   │   ├── balance_residual.yaml
│   │   ├── balance_residual_robust.yaml
│   │   ├── stand_up.yaml
│   │   ├── wheeled_locomotion.yaml
│   │   ├── walking.yaml
│   │   ├── stair_climbing.yaml
│   │   └── rough_terrain.yaml
│   └── controllers/
│       ├── gain_scheduled_lqr.yaml
│       ├── height_scheduled_com_lqr.yaml
│       ├── height_scheduled_dynamic_lqr.yaml
│       ├── hierarchical_vmc_lqr.yaml
│       ├── momentum_coordinator.yaml
│       ├── posture_regularizer.yaml
│       ├── dual_rate_balance_controller_b9.yaml
│       ├── balanced_posture_table.yaml
│       ├── b9_balanced_root_init_table.yaml
│       └── (ablation configs)
│
├── wheeled_biped/                # Main package (~70 files)
│   ├── __init__.py
│   ├── controllers/
│   │   ├── __init__.py
│   │   ├── action_codec.py
│   │   ├── balance_core_types.py
│   │   ├── balance_core_torque_composer.py
│   │   ├── balance_ik.py
│   │   ├── capture_point_estimator.py
│   │   ├── centroidal_balance_controller.py
│   │   ├── centroidal_state_estimator.py
│   │   ├── centroidal_wrench_computer.py
│   │   ├── contact_jacobian.py
│   │   ├── contact_supervisor.py
│   │   ├── dual_rate_balance_controller.py
│   │   ├── force_distribution.py
│   │   ├── hierarchical_vmc_lqr.py
│   │   ├── height_ik.py
│   │   ├── integrated_wbc.py
│   │   ├── lateral_roll_balance_controller.py
│   │   ├── leg_position_controller.py
│   │   ├── lqr_balance.py
│   │   ├── lqr_ik_prior.py
│   │   ├── momentum_coordinator.py
│   │   ├── orientation_utils.py
│   │   ├── pitch_rate_consistency_estimator.py
│   │   ├── position_hold_capture_gate.py
│   │   ├── posture_regularizer.py
│   │   ├── qp_allocator.py
│   │   ├── robot_model_utils.py
│   │   ├── sagittal_balance_state.py
│   │   ├── sagittal_velocity_damped_balance_controller.py  # T6J here
│   │   ├── sagittal_wheel_balance_controller.py
│   │   ├── shape_posture_controller.py
│   │   ├── simple_force_distributor.py
│   │   ├── stage2b_*.py
│   │   ├── stage2c_*.py
│   │   ├── stage2d_*.py
│   │   ├── static_balance_controller.py
│   │   ├── static_feedforward_controller.py
│   │   ├── static_posture_holding_controller.py
│   │   ├── support_feedforward_controller.py
│   │   ├── torque_ownership_validator.py
│   │   ├── unified_force_distributor.py
│   │   ├── wbc_balance_controller.py
│   │   ├── wheel_balancing_controller.py
│   │   └── yaw_controller.py
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── base_env.py
│   │   ├── balance_env.py
│   │   ├── residual_balance_env.py
│   │   ├── standup_env.py
│   │   ├── locomotion_env.py
│   │   ├── walking_env.py
│   │   ├── stair_env.py
│   │   └── terrain_env.py
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── baseline.py
│   │   ├── benchmark.py
│   │   ├── latex_table.py
│   │   └── standing_quality.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── controller_eval.py
│   ├── inference/
│   │   ├── __init__.py
│   │   └── unified_controller.py
│   ├── rewards/
│   │   ├── __init__.py
│   │   └── reward_functions.py
│   ├── sim/
│   │   ├── __init__.py
│   │   ├── domain_randomization.py
│   │   ├── hierarchical_torque_fusion.py
│   │   ├── low_level_control.py
│   │   ├── push_disturbance.py
│   │   ├── terrain_generator.py
│   │   └── torque_wbc.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── curriculum.py
│   │   ├── live_viewer.py
│   │   ├── networks.py
│   │   └── ppo.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── logger.py
│   │   ├── math_utils.py
│   │   └── telemetry.py
│   └── validation/
│       ├── __init__.py
│       ├── balance_core_validator.py
│       ├── classification_report.py
│       ├── failure_classifier.py
│       ├── fix_cycle_reporter.py
│       ├── physical_standing_height_envelope.py
│       ├── step_c_height_recovery.py
│       ├── step_e_audit_helpers.py
│       ├── structural_invariant_checker.py
│       ├── study_aggregator.py
│       ├── telemetry_adapter.py
│       └── telemetry_schema_checker.py
│
├── scripts/                       # Production scripts only (~11 files)
│   ├── train.py
│   ├── evaluate.py
│   ├── eval_balance.py
│   ├── visualize.py
│   ├── validate_checkpoint.py
│   ├── export_results.py
│   ├── compare_baseline.py
│   ├── analyze_residual.py
│   ├── simulate_hierarchical_controller.py  # Main simulation
│   ├── run_t6j_height_ladder.py
│   ├── analyze_t6j_height_ladder.py
│   ├── convert_urdf.py
│   ├── train.py
│   └── visualize.py
│   # NOTE: __init__.py also kept
│
├── tests/                         # All tests preserved (~129 files)
│   ├── __init__.py
│   ├── test_*.py                  # All test files kept
│   # Note: test_env.py and test_smoke_train.py excluded from CI
│   # but kept for manual/slow runs
│
├── docs/
│   ├── baseline_workflow.md
│   ├── paper_claims_checklist.md
│   ├── results_todo_checklist.md
│   ├── experiment_plan.md
│   ├── validation/                 # T6J/T6I validation reports
│   │   ├── t6j_centering_bias_trim_final_report.md
│   │   ├── t6j_height_ladder_validation_report.md
│   │   ├── t6j_centering_bias_trim_design.md
│   │   ├── t6j_centering_bias_trim_implementation_tests_report.md
│   │   ├── t6i_full_staged_validation_final_report.md
│   │   ├── t6i_height_ladder_validation_summary.md
│   │   ├── t6i_high_0p480_full_validation_summary.md
│   │   ├── current_balance_core_roadmap_status.md
│   │   ├── step_c_height_recovery_done.md
│   │   ├── step_c_height_recovery_plan.md
│   │   ├── step_c_height_recovery_spec.md
│   │   ├── step_e_done_2026-06-01.md
│   │   ├── step_e_height_variant_robustness_done.md
│   │   ├── hip_yaw_disturbance_rejection_final_report_v2.md
│   │   ├── hip_yaw_integration_fix_complete.md
│   │   ├── physical_standing_height_envelope_validation.md
│   │   └── physical_standing_height_envelope_definition.md
│   └── repo_cleanup/               # Audit documentation
│       ├── repo_audit_baseline.md
│       ├── repo_inventory.md
│       ├── repo_inventory.json
│       ├── dependency_usage_audit.md
│       ├── dependency_usage_audit.json
│       ├── gitignore_generated_artifact_audit.md
│       ├── cleanup_manifest.md
│       ├── cleanup_manifest.json
│       └── proposed_clean_repo_structure.md
│   # NOTE: 280 obsolete docs moved to archive/
│
├── paper/
│   ├── main.tex
│   ├── refs.bib.backup
│   ├── notes/
│   │   └── phase_b6_stronger_classical_prior.md
│   └── figures/
│       └── annotated_robot_joints.png
│
├── archive/                        # NEW: Archived experiments
│   └── cleanup_2026-06-13/
│       ├── scripts_archive/
│       │   ├── analyze_apcr1*.py
│       │   ├── analyze_t6f*.py
│       │   ├── audit_apcr*.py
│       │   ├── debug_*.py
│       │   ├── diagnose_*.py
│       │   ├── phase_b9_*.py
│       │   └── ... (other obsolete scripts)
│       └── docs_archive/
│           ├── validation_archive/
│           │   ├── apcr1*.md
│           │   ├── t5*.md
│           │   ├── t6f_sign_corrected*.md
│           │   ├── t6h*.md
│           │   ├── e1*.md
│           │   ├── e2*.md
│           │   └── ...
│           ├── phase_b_reports/
│           │   ├── phase_b2_*.md
│           │   ├── phase_b4_*.md
│           │   └── ...
│           ├── task_reports/
│           │   └── task_*.md
│           └── superpowers/
│               └── (all superpowers development docs)
│
├── outputs/                       # Gitignored
├── outputs_residual_smoke/        # Gitignored
├── outputs_residual_test/         # Gitignored
├── .claude/                       # Worktrees ignored
│
├── backup_checkpoints/            # Keep or review
│   ├── Day1/
│   └── Day2/
│
├── baselines/                    # Empty, can be removed
│
└── (no root-level run logs)
```

---

## Key Changes from Current

### Removed from Root Level
- `t6a_run.log` through `t6h_t6i_500_T6I_run.log` (10 files)
- `repo_tracked_files.txt`, `repo_status_ignored.txt` (audit outputs)
- `analyze_t6j_*.py` (superseded by final)
- `analyze_t6h_t6i_500.py`
- `analyze_t6f_*.py`
- `audit_apcr1i_*.py`
- `audit_t6f_*.py`
- `check_arch_fix_gates.py`
- `check_band_state.py`
- `audit_apcr1nd_*.py`

### Archived (Moved to archive/)
- ~220 obsolete scripts in `scripts/`
- ~280 obsolete docs in `docs/validation/`, `docs/phase_b*.md`, `docs/task_*.md`, `docs/superpowers/`

### Kept in Place
- All 70 controller modules
- All 129 test files
- All 35 config files
- 11 production scripts
- 30 core documentation files
- T6J and T6I validation reports

### Gitignore Updates
```gitignore
# Add this line:
assets/robot-urdf/export.log
```

---

## File Count Comparison

| Category | Before | After | Change |
|----------|--------|-------|--------|
| Controllers | 70 | 70 | 0 |
| Tests | 129 | 129 | 0 |
| Scripts | 336 | 11 | -325 |
| Docs | 339 | 59 | -280 |
| Configs | 35 | 35 | 0 |
| Assets | 20 | 20 | 0 |
| Paper | 6 | 6 | 0 |
| Other | 10 | 0 | -10 |
| **Total** | **1072** | **~832** | **-240** |

---

## Archive Structure Details

### archive/cleanup_2026-06-13/scripts_archive/
Organized by experiment type:
```
scripts_archive/
├── apcr_experiments/
│   ├── analyze_apcr1b_*.py
│   ├── analyze_apcr1e_*.py
│   ├── analyze_apcr1l_*.py
│   └── audit_apcr*.py
├── t6f_experiments/
│   ├── analyze_t6f_*.py
│   └── audit_t6f_*.py
├── t6h_experiments/
│   └── analyze_t6h_t6i_500.py
├── debug_scripts/
│   ├── debug_*.py
│   ├── diagnose_*.py
│   └── check_*.py
├── phase_b9_development/
│   ├── phase_b9_*.py
│   └── ...
└── one_off_scripts/
    ├── screening_*.py
    ├── compute_*.py
    └── ...
```

### archive/cleanup_2026-06-13/docs_archive/
Organized by experiment:
```
docs_archive/
├── validation_archive/
│   ├── apcr1_experiments/
│   │   └── apcr1*.md (50+ files)
│   ├── t5_experiments/
│   │   └── t5*.md
│   ├── t6f_sign_corrected/
│   │   └── t6f_sign_corrected*.md
│   ├── t6h_experiments/
│   │   └── t6h*.md
│   ├── early_strategies/
│   │   ├── e1*.md
│   │   ├── e2*.md
│   │   ├── f1*.md
│   │   ├── f2*.md
│   │   └── g1*.md
│   └── superseded_reports/
│       └── phase_*.md
├── phase_b_reports/
│   ├── phase_b2_*.md
│   ├── phase_b4_*.md
│   ├── phase_b5_*.md
│   ├── phase_b6_*.md
│   ├── phase_b7_*.md
│   ├── phase_b8_*.md
│   └── phase_b9_*.md
├── task_reports/
│   ├── task_11_*.md
│   └── task_*.md
└── superpowers/
    ├── plans/
    ├── specs/
    ├── reports/
    └── all .md files
```

---

## Verification After Cleanup

### 1. Test Commands
```bash
# Fast test suite
pytest tests/ --ignore=tests/test_env.py -m "not slow" -v

# T6J specific test
pytest tests/test_t6j_centering_bias_trim.py -v

# T6H/T6I variant tests
pytest tests/test_t6h_t6i_variants.py -v
```

### 2. Simulation Test
```bash
python scripts/simulate_hierarchical_controller.py --controller-mode balance-core --sagittal-controller velocity-damped --vd-sagittal-authority-profile T6J_centering_bias_trim --steps 500
```

### 3. Count Verification
```bash
git ls-files | wc -l
# Should be ~832
```

---

*Generated: 2026-06-13*