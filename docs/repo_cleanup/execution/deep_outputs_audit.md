# Deep Outputs Audit

Total first-level `outputs/*` directories: **172**

## PROTECT (2 dirs, 273.9M)

| Directory | Size | Refs | Summary | Reason |
|---|---|---|---|---|
| `outputs/balance` | 273.8M | 31 | yes | Trained checkpoints / setup JSONs — never delete. |
| `outputs/physical_target_height_setups` | 19.1K | 44 | yes | Trained checkpoints / setup JSONs — never delete. |

## SUMMARIZE_THEN_DELETE (154 dirs, 7.8G)

| Directory | Size | Refs | Summary | Reason |
|---|---|---|---|---|
| `outputs/step_e_extreme_support_fix_eval` | 3.3G | 42 | yes | Large (3.3G) generated diagnostic/raw output; referenced by 42 kept file(s); regenerable. Extract small summaries first. |
| `outputs/hierarchical_controller_sim` | 713.3M | 55 | yes | Large (713.3M) generated diagnostic/raw output; referenced by 55 kept file(s); regenerable. Extract small summaries first. |
| `outputs/step_e_height_variant_position_hold_audit_v2` | 464.1M | 1 | yes | Large (464.1M) generated diagnostic/raw output; referenced by 1 kept file(s); regenerable. Extract small summaries first. |
| `outputs/step_e_height_variant_position_hold_final` | 416.8M | 2 | yes | Large (416.8M) generated diagnostic/raw output; referenced by 2 kept file(s); regenerable. Extract small summaries first. |
| `outputs/step_c_sagittal_schedule_fix` | 272.0M | 1 | yes | Large (272.0M) generated diagnostic/raw output; referenced by 1 kept file(s); regenerable. Extract small summaries first. |
| `outputs/hip_yaw_divergence_after_sign_fix_audit` | 256.1M | 4 | yes | Large (256.1M) generated diagnostic/raw output; referenced by 4 kept file(s); regenerable. Extract small summaries first. |
| `outputs/step_c_height_recovery_after_step_e_hv_fix` | 200.4M | 3 | yes | Large (200.4M) generated diagnostic/raw output; referenced by 3 kept file(s); regenerable. Extract small summaries first. |
| `outputs/balance_core_validation` | 172.9M | 1 | yes | Large (172.9M) generated diagnostic/raw output; referenced by 1 kept file(s); regenerable. Extract small summaries first. |
| `outputs/posture_feasibility` | 171.2M | 1 | no | Large (171.2M) generated diagnostic/raw output; referenced by 1 kept file(s); regenerable. Extract small summaries first. |
| `outputs/hip_yaw_disturbance_rejection_audit` | 168.8M | 5 | yes | Large (168.8M) generated diagnostic/raw output; referenced by 5 kept file(s); regenerable. Extract small summaries first. |
| `outputs/hip_yaw_hy_ff_evaluation` | 147.9M | 7 | yes | Large (147.9M) generated diagnostic/raw output; referenced by 7 kept file(s); regenerable. Extract small summaries first. |
| `outputs/step_e_best_current_profile_5000_eval` | 137.6M | 8 | yes | Large (137.6M) generated diagnostic/raw output; referenced by 8 kept file(s); regenerable. Extract small summaries first. |
| `outputs/hip_yaw_sign_convention_fix` | 130.7M | 7 | yes | Large (130.7M) generated diagnostic/raw output; referenced by 7 kept file(s); regenerable. Extract small summaries first. |
| `outputs/step_e_height_variant_sagittal_schedule_fix` | 112.2M | 1 | yes | Large (112.2M) generated diagnostic/raw output; referenced by 1 kept file(s); regenerable. Extract small summaries first. |
| `outputs/step_c_height_recovery` | 104.3M | 7 | yes | Large (104.3M) generated diagnostic/raw output; referenced by 7 kept file(s); regenerable. Extract small summaries first. |
| `outputs/step_e_wbc_gate_fix` | 85.8M | 1 | yes | Large (85.8M) generated diagnostic/raw output; referenced by 1 kept file(s); regenerable. Extract small summaries first. |
| `outputs/step_e_extreme_height_d2_official_check` | 85.8M | 13 | yes | Large (85.8M) generated diagnostic/raw output; referenced by 13 kept file(s); regenerable. Extract small summaries first. |
| `outputs/balance_core_position_containment` | 85.7M | 1 | yes | Large (85.7M) generated diagnostic/raw output; referenced by 1 kept file(s); regenerable. Extract small summaries first. |
| `outputs/pitch_safe_joint_sagittal_yaw_fix` | 85.3M | 3 | yes | Large (85.3M) generated diagnostic/raw output; referenced by 3 kept file(s); regenerable. Extract small summaries first. |
| `outputs/step_c_extreme_height_recovery` | 80.4M | 2 | yes | Large (80.4M) generated diagnostic/raw output; referenced by 2 kept file(s); regenerable. Extract small summaries first. |
| `outputs/step_e_extreme_height_position_hold` | 80.4M | 2 | yes | Large (80.4M) generated diagnostic/raw output; referenced by 2 kept file(s); regenerable. Extract small summaries first. |
| `outputs/boundary_step_e_step_c_log_verification` | 74.5M | 2 | yes | Large (74.5M) generated diagnostic/raw output; referenced by 2 kept file(s); regenerable. Extract small summaries first. |
| `outputs/sagittal_position_hold_return` | 72.2M | 1 | yes | Large (72.2M) generated diagnostic/raw output; referenced by 1 kept file(s); regenerable. Extract small summaries first. |
| `outputs/sagittal_position_aware_balance` | 69.9M | 1 | yes | Large (69.9M) generated diagnostic/raw output; referenced by 1 kept file(s); regenerable. Extract small summaries first. |
| `outputs/root_z_recheck` | 61.7M | 1 | yes | Large (61.7M) generated diagnostic/raw output; referenced by 1 kept file(s); regenerable. Extract small summaries first. |
| `outputs/visual_inspection_low_0p300` | 52.1M | 3 | yes | Large (52.1M) generated diagnostic/raw output; referenced by 3 kept file(s); regenerable. Extract small summaries first. |
| `outputs/step_c_height_recovery_rich` | 39.4M | 1 | yes | Small (39.4M) generated output; has summary artifacts; referenced by 1 kept file(s); extract summaries then delete. |
| `outputs/balance_core_position_containment_e0b` | 39.3M | 0 | yes | Small (39.3M) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/step_e_height_variant_position_hold_audit` | 31.9M | 2 | yes | Small (31.9M) generated output; has summary artifacts; referenced by 2 kept file(s); extract summaries then delete. |
| `outputs/step_e_hip_yaw_authority_fix` | 26.2M | 0 | yes | Small (26.2M) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/hip_yaw_boundary_audit` | 24.9M | 3 | yes | Small (24.9M) generated output; has summary artifacts; referenced by 3 kept file(s); extract summaries then delete. |
| `outputs/step_e_root_cause_diagnostics` | 23.9M | 1 | yes | Small (23.9M) generated output; has summary artifacts; referenced by 1 kept file(s); extract summaries then delete. |
| `outputs/step_c_height_recovery_variant_short` | 17.5M | 0 | yes | Small (17.5M) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/joint_profile_smoke_tests` | 17.0M | 3 | no | Small (17.0M) generated output; referenced by 3 kept file(s); extract summaries then delete. |
| `outputs/step_c_high_tiny_failure_audit` | 7.1M | 0 | yes | Small (7.1M) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/step_c_height_recovery_short_v2` | 7.0M | 0 | yes | Small (7.0M) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/stage2b_diagnostics` | 4.7M | 2 | yes | Small (4.7M) generated output; has summary artifacts; referenced by 2 kept file(s); extract summaries then delete. |
| `outputs/hip_yaw_root_cause_audit` | 4.0M | 2 | yes | Small (4.0M) generated output; has summary artifacts; referenced by 2 kept file(s); extract summaries then delete. |
| `outputs/step_c_height_recovery_short` | 3.5M | 1 | yes | Small (3.5M) generated output; has summary artifacts; referenced by 1 kept file(s); extract summaries then delete. |
| `outputs/continuous_low_height_sagittal_fix` | 3.4M | 2 | yes | Small (3.4M) generated output; has summary artifacts; referenced by 2 kept file(s); extract summaries then delete. |
| `outputs/step_e_official_validation` | 2.8M | 2 | yes | Small (2.8M) generated output; has summary artifacts; referenced by 2 kept file(s); extract summaries then delete. |
| `outputs/low_height_pitch_position_interaction_audit` | 2.4M | 3 | yes | Small (2.4M) generated output; has summary artifacts; referenced by 3 kept file(s); extract summaries then delete. |
| `outputs/step_c_high_tiny_rich_audit` | 2.0M | 1 | yes | Small (2.0M) generated output; has summary artifacts; referenced by 1 kept file(s); extract summaries then delete. |
| `outputs/hip_yaw_yaw_architecture_audit` | 1.9M | 7 | yes | Small (1.9M) generated output; has summary artifacts; referenced by 7 kept file(s); extract summaries then delete. |
| `outputs/phase_b9_geometry_check` | 1.3M | 2 | yes | Small (1.3M) generated output; has summary artifacts; referenced by 2 kept file(s); extract summaries then delete. |
| `outputs/d2_height_tracking_and_hiproll_audit` | 1.3M | 2 | yes | Small (1.3M) generated output; has summary artifacts; referenced by 2 kept file(s); extract summaries then delete. |
| `outputs/phase_b9_posture_fixed` | 1.2M | 1 | yes | Small (1.2M) generated output; has summary artifacts; referenced by 1 kept file(s); extract summaries then delete. |
| `outputs/stage2_diagnostics` | 1015.6K | 0 | yes | Small (1015.6K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/phase_b9_task6_empirical_ik_corrected` | 753.8K | 2 | yes | Small (753.8K) generated output; has summary artifacts; referenced by 2 kept file(s); extract summaries then delete. |
| `outputs/phase_b9_task6_empirical_ik_standing_only` | 751.4K | 0 | yes | Small (751.4K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/phase_b9_step5_5_roll_tilt_fix` | 687.7K | 1 | yes | Small (687.7K) generated output; has summary artifacts; referenced by 1 kept file(s); extract summaries then delete. |
| `outputs/advanced_hip_yaw_rejection_audit` | 546.2K | 2 | yes | Small (546.2K) generated output; has summary artifacts; referenced by 2 kept file(s); extract summaries then delete. |
| `outputs/phase_b9_task6_empirical_ik` | 519.9K | 3 | yes | Small (519.9K) generated output; has summary artifacts; referenced by 3 kept file(s); extract summaries then delete. |
| `outputs/step_c_initialization_method_audit` | 440.4K | 1 | yes | Small (440.4K) generated output; has summary artifacts; referenced by 1 kept file(s); extract summaries then delete. |
| `outputs/operational_height_envelope_search` | 385.9K | 2 | yes | Small (385.9K) generated output; has summary artifacts; referenced by 2 kept file(s); extract summaries then delete. |
| `outputs/phase_b8_diagnostics` | 313.4K | 2 | no | Small (313.4K) generated output; referenced by 2 kept file(s); extract summaries then delete. |
| `outputs/low_0p300_initialization_contact_audit` | 258.8K | 2 | yes | Small (258.8K) generated output; has summary artifacts; referenced by 2 kept file(s); extract summaries then delete. |
| `outputs/phase_b9_task4_telemetry` | 198.6K | 2 | yes | Small (198.6K) generated output; has summary artifacts; referenced by 2 kept file(s); extract summaries then delete. |
| `outputs/phase_b9_baseline_comparison` | 172.9K | 1 | yes | Small (172.9K) generated output; has summary artifacts; referenced by 1 kept file(s); extract summaries then delete. |
| `outputs/controller_system_root_cause_audit` | 160.0K | 5 | yes | Small (160.0K) generated output; has summary artifacts; referenced by 5 kept file(s); extract summaries then delete. |
| `outputs/step_e_second_stage_diagnostics` | 146.7K | 1 | yes | Small (146.7K) generated output; has summary artifacts; referenced by 1 kept file(s); extract summaries then delete. |
| `outputs/hierarchical_controller_fix_phase7` | 142.5K | 0 | yes | Small (142.5K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/hierarchical_controller_fix_phase6` | 139.2K | 0 | yes | Small (139.2K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/hierarchical_controller_fix_phase4` | 114.9K | 0 | yes | Small (114.9K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/physical_standing_height_envelope_search` | 113.4K | 3 | yes | Small (113.4K) generated output; has summary artifacts; referenced by 3 kept file(s); extract summaries then delete. |
| `outputs/hierarchical_controller_fix_phase5` | 112.2K | 0 | yes | Small (112.2K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/hierarchical_controller_fix_baseline_step_a` | 105.5K | 0 | yes | Small (105.5K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/hierarchical_controller_fix_phase3` | 100.2K | 0 | yes | Small (100.2K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/hierarchical_controller_fix_step_b` | 95.7K | 0 | yes | Small (95.7K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/hierarchical_controller_fix_phase2_selected` | 94.8K | 0 | yes | Small (94.8K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/phase_b9_task7_gain_tuning` | 78.9K | 2 | yes | Small (78.9K) generated output; has summary artifacts; referenced by 2 kept file(s); extract summaries then delete. |
| `outputs/step_e_extreme_failure_root_cause_audit` | 76.5K | 4 | yes | Small (76.5K) generated output; has summary artifacts; referenced by 4 kept file(s); extract summaries then delete. |
| `outputs/hierarchical_controller_fix_phase2` | 74.8K | 0 | yes | Small (74.8K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/phase_b9_step5_10_early_transient_fix` | 70.3K | 1 | yes | Small (70.3K) generated output; has summary artifacts; referenced by 1 kept file(s); extract summaries then delete. |
| `outputs/phase_b9_step5_20_low_stiffness_dynamic_balance` | 64.6K | 2 | yes | Small (64.6K) generated output; has summary artifacts; referenced by 2 kept file(s); extract summaries then delete. |
| `outputs/continuous_low_height_sagittal_authority_fix` | 60.5K | 0 | yes | Small (60.5K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/stage2_gain_sweep` | 58.5K | 1 | yes | Small (58.5K) generated output; has summary artifacts; referenced by 1 kept file(s); extract summaries then delete. |
| `outputs/hip_yaw_divergence_fix_authority_eval` | 54.4K | 2 | yes | Small (54.4K) generated output; has summary artifacts; referenced by 2 kept file(s); extract summaries then delete. |
| `outputs/balance_core_true_height_variants` | 54.4K | 19 | yes | Small (54.4K) generated output; has summary artifacts; referenced by 19 kept file(s); extract summaries then delete. |
| `outputs/phase_b9_slow_loop_gating` | 51.7K | 1 | no | Small (51.7K) generated output; referenced by 1 kept file(s); extract summaries then delete. |
| `outputs/prior_tuning` | 50.3K | 1 | yes | Small (50.3K) generated output; has summary artifacts; referenced by 1 kept file(s); extract summaries then delete. |
| `outputs/height_range_extension_strategy_audit` | 46.8K | 7 | yes | Small (46.8K) generated output; has summary artifacts; referenced by 7 kept file(s); extract summaries then delete. |
| `outputs/phase_b9_lqr_gain_strengthening` | 44.9K | 5 | yes | Small (44.9K) generated output; has summary artifacts; referenced by 5 kept file(s); extract summaries then delete. |
| `outputs/sagittal_velocity_damped_balance` | 44.2K | 0 | yes | Small (44.2K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/balance_core_extended_height_range` | 43.7K | 1 | yes | Small (43.7K) generated output; has summary artifacts; referenced by 1 kept file(s); extract summaries then delete. |
| `outputs/phase_b9_step5_25_hierarchical_torque_fusion` | 43.4K | 1 | yes | Small (43.4K) generated output; has summary artifacts; referenced by 1 kept file(s); extract summaries then delete. |
| `outputs/five_variant_step_e_step_c_baseline_audit` | 41.9K | 1 | yes | Small (41.9K) generated output; has summary artifacts; referenced by 1 kept file(s); extract summaries then delete. |
| `outputs/boundary_yaw_position_coupling_fix` | 31.3K | 5 | yes | Small (31.3K) generated output; has summary artifacts; referenced by 5 kept file(s); extract summaries then delete. |
| `outputs/phase_b9_step5_21_dualrate_controller_debug` | 28.4K | 0 | yes | Small (28.4K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/phase_b9_step5_23_equivalence_audit` | 25.6K | 0 | yes | Small (25.6K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/step_c_contact_invalid_audit` | 23.0K | 0 | yes | Small (23.0K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/phase_b9_step5_17_torque_level_wbc_prototype` | 21.5K | 0 | yes | Small (21.5K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/phase_b9_independent_standability_audit` | 18.9K | 0 | yes | Small (18.9K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/phase_b9_step5_24_torque_first_stabilized_wbc` | 17.4K | 0 | yes | Small (17.4K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/boundary_deep_root_cause_audit` | 16.9K | 3 | yes | Small (16.9K) generated output; has summary artifacts; referenced by 3 kept file(s); extract summaries then delete. |
| `outputs/phase_b9_step5_20b_canonical_eval_rebuild` | 16.5K | 0 | yes | Small (16.5K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/joint_low_height_sagittal_yaw_fix` | 16.4K | 3 | yes | Small (16.4K) generated output; has summary artifacts; referenced by 3 kept file(s); extract summaries then delete. |
| `outputs/phase_b9_step5_8_roll_redesign` | 16.1K | 1 | yes | Small (16.1K) generated output; has summary artifacts; referenced by 1 kept file(s); extract summaries then delete. |
| `outputs/phase_b9_step5_18_deployable_motor_torque_interface` | 13.9K | 0 | yes | Small (13.9K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/balance_core_longevity_height_sweep` | 12.8K | 1 | yes | Small (12.8K) generated output; has summary artifacts; referenced by 1 kept file(s); extract summaries then delete. |
| `outputs/height_range_extension_experiment_0` | 12.8K | 4 | yes | Small (12.8K) generated output; has summary artifacts; referenced by 4 kept file(s); extract summaries then delete. |
| `outputs/step_e_official_validation_v2` | 12.5K | 2 | yes | Small (12.5K) generated output; has summary artifacts; referenced by 2 kept file(s); extract summaries then delete. |
| `outputs/phase_b9_step5_22_torque_first_wbc` | 12.4K | 0 | yes | Small (12.4K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/posture_standing_validation_a0_5000` | 12.2K | 2 | yes | Small (12.2K) generated output; has summary artifacts; referenced by 2 kept file(s); extract summaries then delete. |
| `outputs/classical_prior_eval` | 11.8K | 1 | yes | Small (11.8K) generated output; has summary artifacts; referenced by 1 kept file(s); extract summaries then delete. |
| `outputs/classical_prior_comparison` | 11.5K | 1 | yes | Small (11.5K) generated output; has summary artifacts; referenced by 1 kept file(s); extract summaries then delete. |
| `outputs/phase_b9_step5_18c_torque_gain_saturation_calibration` | 11.4K | 3 | yes | Small (11.4K) generated output; has summary artifacts; referenced by 3 kept file(s); extract summaries then delete. |
| `outputs/phase_b9_step5_11_corrective_path_audit` | 10.3K | 1 | yes | Small (10.3K) generated output; has summary artifacts; referenced by 1 kept file(s); extract summaries then delete. |
| `outputs/low_height_sagittal_authority_transmission_audit` | 9.1K | 2 | yes | Small (9.1K) generated output; has summary artifacts; referenced by 2 kept file(s); extract summaries then delete. |
| `outputs/b9_diagnostic_pitch_only` | 9.0K | 0 | yes | Small (9.0K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/b9_diagnostic_step2_full` | 9.0K | 0 | yes | Small (9.0K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/phase_b9_step5_16_jacobian_wbc_vmc` | 8.8K | 0 | yes | Small (8.8K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/phase_b6_tuning_sanity` | 8.4K | 0 | yes | Small (8.4K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/tune_validation_smoke` | 8.4K | 0 | yes | Small (8.4K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/b9_diagnostic_obsfix` | 8.2K | 0 | yes | Small (8.2K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/phase_b7_eval` | 8.1K | 1 | yes | Small (8.1K) generated output; has summary artifacts; referenced by 1 kept file(s); extract summaries then delete. |
| `outputs/b9_diagnostic` | 7.7K | 0 | yes | Small (7.7K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/b9_diagnostic_fixed` | 7.7K | 0 | yes | Small (7.7K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/b9_diagnostic_fixed_smoke` | 7.7K | 0 | yes | Small (7.7K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/b9_diagnostic_fixed_corrected` | 7.5K | 0 | yes | Small (7.5K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/boundary_forensic_root_cause` | 7.1K | 1 | yes | Small (7.1K) generated output; has summary artifacts; referenced by 1 kept file(s); extract summaries then delete. |
| `outputs/phase_b9_fast_loop_only` | 6.9K | 1 | no | Small (6.9K) generated output; referenced by 1 kept file(s); extract summaries then delete. |
| `outputs/b9_diagnostic_rollfix` | 5.7K | 0 | yes | Small (5.7K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/b9_diagnostic_rollfix_flipped` | 5.7K | 0 | yes | Small (5.7K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/b9_diagnostic_corrected` | 5.7K | 0 | yes | Small (5.7K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/b9_diagnostic_resetfix` | 5.7K | 0 | yes | Small (5.7K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/b9_diagnostic_rollfix_strong` | 5.3K | 0 | yes | Small (5.3K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/hip_yaw_divergence_fix_eval` | 5.1K | 1 | yes | Small (5.1K) generated output; has summary artifacts; referenced by 1 kept file(s); extract summaries then delete. |
| `outputs/hip_yaw_sign_convention_audit` | 4.9K | 2 | yes | Small (4.9K) generated output; has summary artifacts; referenced by 2 kept file(s); extract summaries then delete. |
| `outputs/phase_b9_step5_18b_hybrid_pid_torque_rollout_validation` | 4.9K | 0 | yes | Small (4.9K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/phase_b9_step5_7_early_roll_stabilizer` | 4.8K | 1 | yes | Small (4.8K) generated output; has summary artifacts; referenced by 1 kept file(s); extract summaries then delete. |
| `outputs/phase_b_baseline_check` | 4.6K | 0 | yes | Small (4.6K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/phase_b8_ablation` | 4.6K | 1 | no | Small (4.6K) generated output; referenced by 1 kept file(s); extract summaries then delete. |
| `outputs/phase_b9_balanced_root_solver` | 4.1K | 0 | yes | Small (4.1K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/phase_b9_step5_9_roll_authority_audit` | 4.0K | 1 | yes | Small (4.0K) generated output; has summary artifacts; referenced by 1 kept file(s); extract summaries then delete. |
| `outputs/phase_b9_audit_gate` | 3.9K | 0 | yes | Small (3.9K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/phase_b_ppo_check` | 3.8K | 0 | yes | Small (3.8K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/balance_core_extended_longevity` | 3.3K | 1 | yes | Small (3.3K) generated output; has summary artifacts; referenced by 1 kept file(s); extract summaries then delete. |
| `outputs/phase_b9_task5_equilibrium` | 2.8K | 2 | yes | Small (2.8K) generated output; has summary artifacts; referenced by 2 kept file(s); extract summaries then delete. |
| `outputs/visual_environment_check` | 2.8K | 2 | yes | Small (2.8K) generated output; has summary artifacts; referenced by 2 kept file(s); extract summaries then delete. |
| `outputs/phase_b9_task6_corrected_ik_eval` | 2.5K | 0 | yes | Small (2.5K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/phase_b9_task6_standing_ik_eval` | 2.4K | 0 | yes | Small (2.4K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/phase_b_narrow_height` | 2.2K | 0 | yes | Small (2.2K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/diagnostics` | 2.2K | 0 | yes | Small (2.2K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/phase_b_no_roll` | 2.2K | 0 | yes | Small (2.2K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/phase_b_tune1` | 2.2K | 0 | yes | Small (2.2K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/phase_b_antisymmetric` | 2.2K | 0 | yes | Small (2.2K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/phase_b_verification` | 2.1K | 0 | yes | Small (2.1K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/b9_subprocess_smoke` | 2.0K | 0 | yes | Small (2.0K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/step_e_boundary_height_hold_0p300_0p480` | 1.8K | 1 | yes | Small (1.8K) generated output; has summary artifacts; referenced by 1 kept file(s); extract summaries then delete. |
| `outputs/phase_b9_step5_19_controller_authority_reallocation` | 1.7K | 3 | no | Small (1.7K) generated output; referenced by 3 kept file(s); extract summaries then delete. |
| `outputs/prior_eval` | 1.6K | 0 | yes | Small (1.6K) generated output; has summary artifacts; extract summaries then delete. |
| `outputs/phase_b9_task6_true_manifold` | 1.4K | 1 | yes | Small (1.4K) generated output; has summary artifacts; referenced by 1 kept file(s); extract summaries then delete. |
| `outputs/classical_prior_eval_test` | 812.0B | 0 | yes | Small (812.0B) generated output; has summary artifacts; extract summaries then delete. |

## DELETE_DIRECT (16 dirs, 78.9M)

| Directory | Size | Refs | Summary | Reason |
|---|---|---|---|---|
| `outputs/balance_core_position_aware_precheck_5000` | 31.8M | 0 | no | Small (31.8M) generated output, no summary files, no kept reference. |
| `outputs/balance_core_e0_cleanup_validation_5000` | 31.8M | 0 | no | Small (31.8M) generated output, no summary files, no kept reference. |
| `outputs/balance_core_e0_cleanup_validation` | 6.4M | 0 | no | Small (6.4M) generated output, no summary files, no kept reference. |
| `outputs/balance_core_position_aware_precheck_1000` | 6.4M | 0 | no | Small (6.4M) generated output, no summary files, no kept reference. |
| `outputs/phase_b9_posture_balanced_root` | 1.2M | 0 | no | Small (1.2M) generated output, no summary files, no kept reference. |
| `outputs/phase_b9_candidate_postures` | 916.0K | 0 | no | Small (916.0K) generated output, no summary files, no kept reference. |
| `outputs/phase_b9_camera_tests` | 285.9K | 0 | no | Small (285.9K) generated output, no summary files, no kept reference. |
| `outputs/phase_b9_balanced_root_contact_test` | 150.8K | 0 | no | Small (150.8K) generated output, no summary files, no kept reference. |
| `outputs/hip_yaw_sign_fix_roll_collapse_audit` | 113.2K | 0 | no | Small (113.2K) generated output, no summary files, no kept reference. |
| `outputs/phase_b9_slow_loop_gating_full_sweep` | 55.2K | 0 | no | Small (55.2K) generated output, no summary files, no kept reference. |
| `outputs/phase_b9_slow_loop_gating_fixed_baseline` | 10.6K | 0 | no | Small (10.6K) generated output, no summary files, no kept reference. |
| `outputs/balance_core_height_recovery` | 0.0B | 0 | no | Empty directory. |
| `outputs/hip_yaw_hy2_div_evaluation` | 0.0B | 0 | no | Empty directory. |
| `outputs/hip_yaw_sign_fix_smoke` | 0.0B | 0 | no | Empty directory. |
| `outputs/stage2d_sysid` | 0.0B | 3 | no | Empty directory. |
| `outputs/telemetry` | 0.0B | 1 | no | Empty directory. |

## REVIEW (0 dirs, 0.0B)

