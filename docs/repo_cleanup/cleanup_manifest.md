# Cleanup Manifest

## Date: 2026-06-13

---

## Class K: KEEP - Required for Current Functionality

### K.1 Core Runtime (wheeled_biped/)

| Path | Category | Reason | Risk Level |
|------|----------|--------|------------|
| wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py | Core | T6J lives here | LOW |
| wheeled_biped/controllers/balance_core_types.py | Core | Shared types | LOW |
| wheeled_biped/controllers/position_hold_capture_gate.py | Core | Capture gate | LOW |
| wheeled_biped/controllers/*.py | Core | All controller modules | LOW |
| wheeled_biped/envs/*.py | Core | All env modules | LOW |
| wheeled_biped/eval/*.py | Core | Evaluation modules | LOW |
| wheeled_biped/inference/*.py | Core | Inference modules | LOW |
| wheeled_biped/rewards/*.py | Core | Reward functions | LOW |
| wheeled_biped/sim/*.py | Core | Simulation helpers | LOW |
| wheeled_biped/training/*.py | Core | PPO/training | LOW |
| wheeled_biped/utils/*.py | Core | Utilities | LOW |
| wheeled_biped/validation/*.py | Core | Validation modules | LOW |

### K.2 Tests (129 files)

| Path | Category | Reason | Risk Level |
|------|----------|--------|------------|
| tests/*.py | Tests | All tests serve as regression tests | LOW |
| tests/test_t6j_centering_bias_trim.py | Tests | T6J validation | LOW |
| tests/test_t6h_t6i_variants.py | Tests | T6H/T6I validation | LOW |

**Note**: All 129 test files are kept - they provide regression coverage for the evolution of controller variants.

### K.3 Scripts - Production (11 files)

| Path | Category | Reason | Risk Level |
|------|----------|--------|------------|
| scripts/train.py | Production | Main training entry | LOW |
| scripts/evaluate.py | Production | 4-mode evaluation | LOW |
| scripts/eval_balance.py | Production | Research evaluation | LOW |
| scripts/visualize.py | Production | MuJoCo viewer | LOW |
| scripts/validate_checkpoint.py | Production | Checkpoint validation | LOW |
| scripts/export_results.py | Production | CSV/PNG/LaTeX | LOW |
| scripts/compare_baseline.py | Production | Regression comparison | LOW |
| scripts/analyze_residual.py | Production | Residual analysis | LOW |
| scripts/simulate_hierarchical_controller.py | Production | Main simulation (T6J) | LOW |
| scripts/run_t6j_height_ladder.py | Production | T6J validation runner | LOW |
| scripts/analyze_t6j_height_ladder.py | Production | T6J analysis | LOW |

### K.4 Configs (35 files)

| Path | Category | Reason | Risk Level |
|------|----------|--------|------------|
| configs/robot.yaml | Config | Robot parameters | LOW |
| configs/curriculum.yaml | Config | Multi-stage pipeline | LOW |
| configs/baseline_lqr.yaml | Config | LQR baseline | LOW |
| configs/training/*.yaml | Config | All training configs | LOW |
| configs/controllers/*.yaml | Config | All controller configs | LOW |

### K.5 Documentation - Core (30 files)

| Path | Category | Reason | Risk Level |
|------|----------|--------|------------|
| README.md | Docs | Main README | LOW |
| CLAUDE.md | Docs | Project instructions | LOW |
| docs/baseline_workflow.md | Docs | Baseline workflow | LOW |
| docs/paper_claims_checklist.md | Docs | Paper claims | LOW |
| docs/results_todo_checklist.md | Docs | Results TODO | LOW |
| docs/experiment_plan.md | Docs | Experiment plan | LOW |
| docs/validation/t6j_centering_bias_trim_final_report.md | Docs | T6J final report | LOW |
| docs/validation/t6j_height_ladder_validation_report.md | Docs | T6J validation | LOW |
| docs/validation/t6j_centering_bias_trim_design.md | Docs | T6J design | LOW |
| docs/validation/t6j_centering_bias_trim_implementation_tests_report.md | Docs | T6J tests | LOW |
| docs/validation/t6i_full_staged_validation_final_report.md | Docs | T6I final (base) | LOW |
| docs/validation/t6i_height_ladder_validation_summary.md | Docs | T6I ladder | LOW |
| docs/validation/t6i_high_0p480_full_validation_summary.md | Docs | T6I high | LOW |
| docs/validation/current_balance_core_roadmap_status.md | Docs | Roadmap status | LOW |
| docs/validation/step_c_height_recovery_done.md | Docs | Step C done | LOW |
| docs/validation/step_c_height_recovery_plan.md | Docs | Step C plan | LOW |
| docs/validation/step_c_height_recovery_spec.md | Docs | Step C spec | LOW |
| docs/validation/step_e_done_2026-06-01.md | Docs | Step E done | LOW |
| docs/validation/step_e_height_variant_robustness_done.md | Docs | Step E done | LOW |
| docs/validation/hip_yaw_disturbance_rejection_final_report_v2.md | Docs | Hip yaw fix | LOW |
| docs/validation/hip_yaw_integration_fix_complete.md | Docs | Hip yaw fix | LOW |
| docs/validation/physical_standing_height_envelope_validation.md | Docs | Height envelope | LOW |
| docs/validation/physical_standing_height_envelope_definition.md | Docs | Height envelope | LOW |

### K.6 Assets

| Path | Category | Reason | Risk Level |
|------|----------|--------|------------|
| assets/robot-urdf/urdf/HOANTHIEN_TEST.urdf | Assets | Robot URDF | LOW |
| assets/robot-urdf/urdf/HOANTHIEN_TEST.csv | Assets | Joint config | LOW |
| assets/robot-urdf/meshes/*.STL | Assets | Robot meshes | LOW |
| assets/robot-urdf/config/*.yaml | Assets | ROS config | LOW |
| assets/robot-urdf/CMakeLists.txt | Assets | ROS CMake | LOW |
| assets/robot-urdf/package.xml | Assets | ROS package | LOW |
| assets/robot/wheeled_biped_real.xml | Assets | Main MJCF | LOW |

### K.7 Other Essential Files

| Path | Category | Reason | Risk Level |
|------|----------|--------|------------|
| paper/main.tex | Paper | Main paper | LOW |
| paper/refs.bib.backup | Paper | Bib backup | LOW |
| paper/figures/annotated_robot_joints.png | Paper | Figure | LOW |
| pyproject.toml | Config | Package config | LOW |
| pytest.ini | Config | Pytest config | LOW |
| .github/workflows/ci.yml | CI | GitHub Actions | LOW |

---

## Class B: BACKUP_THEN_REMOVE - Archive for History

### B.1 Obsolete Script Analysis (220 files)

These are one-off diagnostic scripts that are no longer needed but contain historical information about past experiments.

**Backup Directory**: `archive/cleanup_2026-06-13/scripts_archive/`

| Pattern | Count | Reason | Evidence |
|---------|-------|--------|----------|
| analyze_apcr*.py | ~15 | APCR experiments superseded | Not referenced by active code |
| analyze_t6f*.py | ~6 | T6F experiments superseded | Not referenced by active code |
| analyze_t6h*.py | ~1 | T6H experiments superseded | Not referenced by active code |
| audit_apcr*.py | ~5 | APCR audits | Not referenced by active code |
| audit_t6f*.py | ~2 | T6F audits | Not referenced by active code |
| debug_*.py | ~30 | One-off debug | Not referenced by active code |
| diagnose_*.py | ~15 | One-off diagnosis | Not referenced by active code |
| check_*.py | ~25 | One-off checks | Not referenced by active code |
| phase_b9_*.py | ~25 | Phase B9 development | Superseded by current T6J |
| screening_*.py | ~5 | One-off screening | Not referenced by active code |
| compute_*.py | ~10 | One-off computations | Not referenced by active code |
| boundary_*.py | ~5 | Boundary analysis | Not referenced by active code |
| find_*.py | ~15 | Equilibrium finding | May keep if still useful |
| compare_*.py | ~10 | Comparisons | Keep if still useful |
| evaluate_*.py | ~8 | Eval scripts | Keep if still useful |
| validate_*.py | ~10 | Validation | Keep if still useful |
| verify_*.py | ~8 | Verification | Keep if still useful |
| other_one_off | ~25 | One-off scripts | Not referenced |

**Deletion Command**:
```bash
mkdir -p archive/cleanup_2026-06-13/scripts_archive
# Move scripts to archive first (for history)
git mv scripts/analyze_apcr*.py archive/cleanup_2026-06-13/scripts_archive/
git mv scripts/audit_apcr*.py archive/cleanup_2026-06-13/scripts_archive/
# ... etc
```

### B.2 Obsolete Documentation (280 files)

**Backup Directory**: `archive/cleanup_2026-06-13/docs_archive/validation_archive/`

| Pattern | Count | Reason | Evidence |
|---------|-------|--------|----------|
| validation/apcr1*.md | ~50 | APCR1 experiments | Superseded by APCR1ND/T6J |
| validation/t5*.md | ~10 | T5 variants | Superseded by T6 series |
| validation/t6f_sign_corrected*.md | ~10 | Failed design | T6F sign corrected failed |
| validation/t6h*.md | ~5 | Superseded | T6H superseded by T6I |
| validation/e1*.md | ~5 | Early strategy | Superseded |
| validation/e2*.md | ~5 | Early strategy | Superseded |
| validation/f1*.md | ~5 | Early strategy | Superseded |
| validation/f2*.md | ~5 | Early strategy | Superseded |
| validation/g1*.md | ~5 | Early strategy | Superseded |
| validation/phase_*.md | ~10 | Phase reports | Superseded |
| phase_b*.md | ~7 | Old phase docs | Superseded |
| task_*.md | ~3 | Task reports | Superseded |
| superpowers/**/*.md | ~100+ | Development docs | Superseded by T6J |

### B.3 Obsolete Controller Validation Reports

| File | Reason | Backup |
|------|--------|--------|
| docs/validation/t6i_500_diagnostic_report.md | Superseded by final | Keep if history needed |
| docs/validation/t6i_height_ladder_2000_report.md | Superseded by final | Keep if history needed |
| docs/validation/t6i_high_0p480_1200_validation_report.md | Superseded | Keep if history needed |
| docs/validation/t6i_high_0p480_2000_validation_report.md | Superseded | Keep if history needed |
| docs/validation/t6i_high_0p480_5000_validation_report.md | Superseded | Keep if history needed |
| docs/validation/t6i_profile_and_telemetry_verification.md | Superseded | Keep if history needed |
| docs/validation/t6i_positive_bias_root_cause_audit.md | Superseded | Keep if history needed |
| docs/validation/t6h_500_diagnostic_report.md | Failed design | Archive |
| docs/validation/t6h_t6i_500_comparative_diagnostic_report.md | Failed design | Archive |
| docs/validation/t6h_t6i_implementation_plan.md | Failed design | Archive |
| docs/validation/t6h_t6i_implementation_summary.md | Failed design | Archive |
| docs/validation/t6h_t6i_safe_next_candidates_design.md | Failed design | Archive |

---

## Class D: DELETE - Generated/Duplicated/Useless

### D.1 Root Directory Log Files

| Path | Category | Reason | Evidence | Deletion Command |
|------|----------|--------|----------|------------------|
| t6a_run.log | Generated | Run log | Ignored by git | `rm t6a_run.log` |
| t6b_5000_run.log | Generated | Run log | Ignored by git | `rm t6b_5000_run.log` |
| t6b_run.log | Generated | Run log | Ignored by git | `rm t6b_run.log` |
| t6c_run.log | Generated | Run log | Ignored by git | `rm t6c_run.log` |
| t6d_run.log | Generated | Run log | Ignored by git | `rm t6d_run.log` |
| t6e_run.log | Generated | Run log | Ignored by git | `rm t6e_run.log` |
| t6h_t6i_500_T5_run.log | Generated | Run log | Ignored by git | `rm t6h_t6i_500_T5_run.log` |
| t6h_t6i_500_T6F_run.log | Generated | Run log | Ignored by git | `rm t6h_t6i_500_T6F_run.log` |
| t6h_t6i_500_T6H_run.log | Generated | Run log | Ignored by git | `rm t6h_t6i_500_T6H_run.log` |
| t6h_t6i_500_T6I_run.log | Generated | Run log | Ignored by git | `rm t6h_t6i_500_T6I_run.log` |

### D.2 Audit Output Files (if any)

| Path | Category | Reason | Evidence | Deletion Command |
|------|----------|--------|----------|------------------|
| repo_tracked_files.txt | Generated | Audit output | Created by audit | `rm repo_tracked_files.txt` |
| repo_status_ignored.txt | Generated | Audit output | Created by audit | `rm repo_status_ignored.txt` |
| docs/repo_cleanup/*.md | Generated | Audit output | Keep for reference | Keep |
| docs/repo_cleanup/*.json | Generated | Audit output | Keep for reference | Keep |

### D.3 Obsolete T6J Diagnostic Files (Root level)

| Path | Category | Reason | Evidence | Deletion Command |
|------|----------|--------|----------|------------------|
| analyze_t6j_500_diagnostic.py | Analysis | Superseded by final | T6J 5000 final exists | `rm analyze_t6j_500_diagnostic.py` |
| analyze_t6j_1200_diagnostic.py | Analysis | Superseded by final | T6J 5000 final exists | `rm analyze_t6j_1200_diagnostic.py` |
| analyze_t6j_2000_diagnostic.py | Analysis | Superseded by final | T6J 5000 final exists | `rm analyze_t6j_2000_diagnostic.py` |
| analyze_t6j_5000_diagnostic.py | Analysis | Superseded by final | T6J 5000 final exists | `rm analyze_t6j_5000_diagnostic.py` |

**Note**: The final T6J reports are in `docs/validation/` and should be kept.

### D.4 T6H/T6F Analysis Files (Root level)

| Path | Category | Reason | Evidence | Deletion Command |
|------|----------|--------|----------|------------------|
| analyze_t6h_t6i_500.py | Analysis | Superseded | T6H/T6I superseded | `rm analyze_t6h_t6i_500.py` |
| analyze_t6f_sign_fix_500_diagnostic.py | Analysis | Failed design | T6F sign failed | `rm analyze_t6f_sign_fix_500_diagnostic.py` |
| analyze_t6f_sign_corrected_500.py | Analysis | Failed design | T6F sign corrected failed | `rm analyze_t6f_sign_corrected_500.py` |
| analyze_t6f_torque_transmission.py | Analysis | Obsolete | T6F superseded | `rm analyze_t6f_torque_transmission.py` |
| analyze_t6f_2000_screening.py | Analysis | Obsolete | T6F superseded | `rm analyze_t6f_2000_screening.py` |
| analyze_t6f_degradation_root_cause.py | Analysis | Obsolete | T6F superseded | `rm analyze_t6f_degradation_root_cause.py` |
| analyze_t6f_component_sign_audit.py | Analysis | Obsolete | T6F superseded | `rm analyze_t6f_component_sign_audit.py` |
| audit_t6f_sign_fix_telemetry.py | Audit | Failed design | T6F sign fix failed | `rm audit_t6f_sign_fix_telemetry.py` |
| audit_t6f_pitch_suppression.py | Audit | Failed design | T6F sign fix failed | `rm audit_t6f_pitch_suppression.py` |
| audit_t6f_high_authority.py | Audit | Failed design | T6F sign fix failed | `rm audit_t6f_high_authority.py` |
| debug_t6f_sign_zero_torque.py | Debug | Failed design | T6F sign fix failed | `rm debug_t6f_sign_zero_torque.py` |

### D.5 Tracked Log File to Untrack

| Path | Category | Reason | Evidence | Deletion Command |
|------|----------|--------|----------|------------------|
| assets/robot-urdf/export.log | Tracked Log | Generated | Not needed in repo | `git rm assets/robot-urdf/export.log` |

### D.6 Obsolete Root-level Analysis Files

| Path | Category | Reason | Evidence | Deletion Command |
|------|----------|--------|----------|------------------|
| audit_apcr1i_1000_episodes.py | Audit | Superseded | APCR1 superseded | `rm audit_apcr1i_1000_episodes.py` |
| audit_apcr1i_1000_torque_authority.py | Audit | Superseded | APCR1 superseded | `rm audit_apcr1i_1000_torque_authority.py` |
| audit_apcr1nd_band_control_failure.py | Audit | Superseded | APCR1 superseded | `rm audit_apcr1nd_band_control_failure.py` |
| check_arch_fix_gates.py | Check | Debug | One-off debug | `rm check_arch_fix_gates.py` |
| check_band_state.py | Check | Debug | One-off debug | `rm check_band_state.py` |

---

## Summary Statistics

| Class | Files | Action |
|-------|-------|--------|
| K (Keep) | ~200 | No action |
| B (Backup then Remove) | ~500 | Move to archive, then remove |
| D (Delete) | ~40 | Delete directly |

### Detailed Breakdown

| Category | Keep | Archive | Delete |
|----------|------|---------|--------|
| Controllers | 70 | 0 | 0 |
| Tests | 129 | 0 | 0 |
| Scripts (336) | 11 | ~220 | ~5 |
| Docs (339) | 50 | ~280 | 0 |
| Configs | 35 | 0 | 0 |
| Assets | ~20 | 0 | 1 |
| Generated/Root logs | 0 | 0 | ~15 |

---

*Generated: 2026-06-13*