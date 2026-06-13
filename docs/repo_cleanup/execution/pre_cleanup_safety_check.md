# Pre-Cleanup Safety Check

**Date:** 2026-06-13
**Branch:** repo-cleanup-t6j
**Baseline tracked files:** 1072

---

## 1. Baseline verification (Phase 0)

- Compile: `simulate_hierarchical_controller.py`, `sagittal_velocity_damped_balance_controller.py`, `shape_posture_controller.py` → **COMPILE OK**
- `pytest tests/test_t6j_centering_bias_trim.py` + `test_t6h_t6i_variants.py` + `test_sagittal_velocity_damped_balance_controller.py` + `test_simulation_telemetry_csv_writer.py` + `test_low_height_setup_initialization.py` + `test_step_e_wbc_gate_validator.py` → **349 passed (19.69s)**
- Second targeted run → **22 passed (4.14s)**

Baseline is GREEN. Cleanup may proceed.

---

## 2. Paper citation check

- `grep "docs/validation" paper/ docs/ README.md` → **no references**
- `paper/main.tex` does **not** cite any report scheduled for archive.
- Only self-references found are inside `paper/notes/phase_b6_stronger_classical_prior.md` pointing to `paper/notes/phase_b4_*` / `phase_b5_*` (all under `paper/notes/`, which is **KEPT** — not part of cleanup scope).

**Verdict:** No archived doc is referenced by the manuscript. Safe to archive `docs/phase_b*.md`, `docs/task_*.md`, `docs/validation/{apcr,t5,t6f_sign,t6h,e1,e2,f1,f2,g1,phase_*}*.md`, `docs/superpowers/**`.

See [paper_reference_check.md](paper_reference_check.md).

---

## 3. Setup JSON safety

`outputs/physical_target_height_setups/` exists and contains the height-variant setup JSONs consumed by `run_t6j_height_ladder.py` and the smoke simulation:

```
high_0p430_setup.json   high_0p450_setup.json   high_0p465_setup.json   high_0p480_setup.json
low_0p300_setup.json    low_0p320_setup.json    low_0p330_setup.json    low_0p340_setup.json
low_0p360_setup.json    low_0p380_setup.json
ladder_setup_validation_summary.json            static_validation_summary.json
physical_target_height_setup_report.md
```

**Decision:** PRESERVED. Not deleted, not archived. Required for height validation.

---

## 4. Dependency-driven exclusions from archive

The following were pattern-matched as "obsolete" but are **kept in place** because they are imported by collected tests or are mainline controller infrastructure:

### 4.1 Scripts imported by tests/other scripts (kept)
- `scripts/debug_stage2b_a_parity.py`
- `scripts/diagnose_step_e_root_causes.py`, `scripts/diagnose_step_e_second_stage.py`
- `scripts/phase_b9_posture_symmetry_fix.py`, `scripts/phase_b9_posture_geometry_inspection.py`
- `scripts/phase_b9_step4_slow_loop_gating.py`
- `scripts/phase_b9_step5_13_reset_equilibrium_fix.py`
- `scripts/phase_b9_step5_14_lateral_balance_layer.py`
- `scripts/phase_b9_step5_16_jacobian_wbc_vmc.py`
- `scripts/phase_b9_step5_18b_hybrid_pid_torque_rollout_validation.py`
- `scripts/phase_b9_step5_5_roll_tilt_fix.py`
- `scripts/phase_b9_step5_lqr_gain_strengthening.py`

### 4.2 phase_b9_* mainline controller infrastructure (kept)
Per project memory `project-phase-b9-mainline-controller-infrastructure`: Step 5.14/5.15 (`lateral_balance`, `vmc_whole_body`) and the surrounding phase_b9 step scripts are official mainline controller infrastructure, not failed history. **All `scripts/phase_b9_*` are kept in place**, not archived, to preserve the controller/telemetry/diagnostic chain referenced by the phase_b9 test suite.

---

## 5. Archive plan summary

| Class | Count | Destination |
|---|---|---|
| scripts/ one-offs (analyze/audit/check/compute/debug/diagnose/boundary/screening, non-imported) | 85 | `archive/cleanup_2026-06-13/scripts_archive/` |
| root-level loose analysis `.py` (none imported) | 58 | `archive/cleanup_2026-06-13/root_analysis_archive/` |
| obsolete docs (apcr/t5/t6f_sign/t6h/e/f/g/phase_*/task_*/superpowers) | 191 | `archive/cleanup_2026-06-13/docs_archive/` (relative paths preserved) |

| Class | Count | Action |
|---|---|---|
| root run logs + MUJOCO_LOG.TXT | 11 | delete (gitignored) |
| caches (`__pycache__`, `.pytest_cache`, `.ruff_cache`, `.mypy_cache`, egg-info) | n/a | delete (gitignored) |
| audit temp (`repo_tracked_files.txt`, `repo_status_ignored.txt`) | 2 | delete |
| `assets/robot-urdf/export.log` | 1 | `git rm --cached` + gitignore |

All archive moves use `git mv` (tracked). Branch `repo-cleanup-t6j` + git history provide full rollback.
