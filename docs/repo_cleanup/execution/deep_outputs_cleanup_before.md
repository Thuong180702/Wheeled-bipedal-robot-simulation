# Deep Outputs Cleanup — Before State

**Date:** 2026-06-13
**Branch:** `repo-cleanup-t6j`
**Scope:** Deep clean of remaining `outputs/` bulk (~8.2 G) with summary extraction before deletion. Protected checkpoints and setup JSONs preserved.

---

## Total size

| Path | Size |
|---|---|
| `outputs/` | **8.2 G** |
| `backup_checkpoints/` (tracked) | 19 M |

`outputs/` holds **172** first-level subdirectories.

## First-level `outputs/*` (sorted, largest tail)

| Dir | Size |
|---|---|
| `outputs/step_e_extreme_support_fix_eval` | 3.4 G |
| `outputs/hierarchical_controller_sim` | 714 M |
| `outputs/step_e_height_variant_position_hold_audit_v2` | 465 M |
| `outputs/step_e_height_variant_position_hold_final` | 417 M |
| `outputs/balance` | 275 M (**PROTECTED**) |
| `outputs/step_c_sagittal_schedule_fix` | 273 M |
| `outputs/hip_yaw_divergence_after_sign_fix_audit` | 257 M |
| `outputs/step_c_height_recovery_after_step_e_hv_fix` | 201 M |
| `outputs/balance_core_validation` | 173 M |
| `outputs/posture_feasibility` | 172 M |
| `outputs/hip_yaw_disturbance_rejection_audit` | 169 M |
| `outputs/hip_yaw_hy_ff_evaluation` | 148 M |
| `outputs/step_e_best_current_profile_5000_eval` | 138 M |
| `outputs/hip_yaw_sign_convention_fix` | 131 M |
| `outputs/step_e_height_variant_sagittal_schedule_fix` | 113 M |
| `outputs/step_c_height_recovery` | 105 M |
| (+ ~157 smaller dirs) | … |

Full sorted listing: [deep_outputs_before_sizes.txt](deep_outputs_before_sizes.txt).

## Protected assets (must survive)

| Asset | Size |
|---|---|
| `outputs/balance/rl/seed42` | 119 M |
| `outputs/balance/rl/seed113` | 94 M |
| `outputs/balance/rl/seed999` | 62 M |
| `outputs/physical_target_height_setups` | 56 K |
| `backup_checkpoints/` | 19 M |

## Note on working-tree state

`git status` showed 4 previously-tracked root `.txt` files staged as deletions (`f1_run_output.txt`, `fast_suite_final.txt`, `fast_suite_results.txt`, `hierarchical_sim_output.txt`) — these were restored (`git checkout`) as out-of-scope for this task. Only the new audit docs under `docs/repo_cleanup/execution/` are untracked.
