# Phase B.9 Step 5.13 Reset Equilibrium Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair B9 reset/static equilibrium before any controller tuning, then revalidate the existing Step 5 best controller from a physically valid initial state.

**Architecture:** Add a focused Step 5.13 diagnostic/repair script that verifies the current bug, generates and validates a physically plausible balanced-root table, and writes artifacts under `outputs/phase_b9_step5_13_reset_equilibrium_fix/`. Patch the Step 5 evaluator initializer to apply full freejoint root pose as MuJoCo quaternion plus joint targets, while keeping controller gains unchanged.

**Tech Stack:** Python, MuJoCo CPU API, JAX/MJX for `BalanceEnv` stepping, pytest, YAML/CSV/JSON artifacts.

---

### Task 1: Add failing tests for full-root initialization

**Files:**
- Create: `tests/test_phase_b9_step5_13_reset_equilibrium_fix.py`
- Modify: `scripts/phase_b9_step5_lqr_gain_strengthening.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_phase_b9_step5_13_reset_equilibrium_fix.py` with tests that import `apply_balanced_root_init` from `scripts.phase_b9_step5_lqr_gain_strengthening` and assert it applies root position, quaternion orientation, joint targets, and zero velocities. Include a sentinel test that reads `configs/training/balance_residual.yaml` before/after and verifies unchanged content.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_phase_b9_step5_13_reset_equilibrium_fix.py`

Expected: FAIL because current Step 5 initializer ignores full root pose.

- [ ] **Step 3: Implement full-root init in Step 5**

Modify `scripts/phase_b9_step5_lqr_gain_strengthening.py`:
- add `rpy_to_quat(roll, pitch, yaw)` if not present;
- update `apply_balanced_root_init(mjx_data, height, init_table)` to set `qpos[0]`, `qpos[2]`, `qpos[3:7]`, `qpos[7:17]`, and zero `qvel`;
- preserve `qpos[1]` unless table specifies root_y;
- keep function name to avoid touching call sites.

- [ ] **Step 4: Run full-root tests**

Run: `pytest -q tests/test_phase_b9_step5_13_reset_equilibrium_fix.py`

Expected: PASS for full-root application tests that do not depend on new table validity.

---

### Task 2: Add Step 5.13 repair/validation script

**Files:**
- Create: `scripts/phase_b9_step5_13_reset_equilibrium_fix.py`

- [ ] **Step 1: Implement diagnostic script skeleton**

Create a script with:
- constants for output directory and heights;
- MuJoCo model loader via `get_model_path()`;
- YAML loader/writer;
- `contact_forces_by_wheel`, `wheel_bottom_heights`, `body_com` reuse from existing B9 diagnostic utilities;
- `quat_to_rpy` and `rpy_to_quat` helpers;
- CSV/JSON/Markdown writers.

- [ ] **Step 2: Implement Phase 1 verification**

Add functions that compare:
- old table full-root state;
- old Step 5 joint-only state;
- contact metrics before `mj_forward`, after `mj_forward`, and after short settling.

Write:
- `reset_bug_verification.json`
- `reset_bug_verification.md`

- [ ] **Step 3: Implement candidate pose generation**

For each height:
- use table hip_pitch/knee as initial joint target;
- compute a root z that places wheel bottom near ground without penetration;
- search a small root_x/root_roll/root_pitch grid around zero/table values;
- score candidates by clearance, total contact force after settling, left/right imbalance, and roll/pitch after 100 ms;
- read expected weight from `sum(model.body_mass) * abs(model.opt.gravity[2])`.

- [ ] **Step 4: Implement validation gates**

For each candidate, compute:
- wheel clearances;
- left/right/total force;
- total force / expected weight;
- imbalance / total force;
- roll/pitch angle and rate;
- termination-like tilt/height flag.

Accept if:
- abs(clearance) <= 1 mm before dynamic settling or documented otherwise;
- after 100 ms, total force is plausible relative to weight;
- no wheel is fully unloaded in PID-hold settling;
- roll/pitch remain below small diagnostic thresholds.

- [ ] **Step 5: Write outputs**

Write:
- `new_balanced_root_table.yaml`
- `reset_equilibrium_validation.csv`
- `reset_equilibrium_summary.json`
- `full_root_application_trace.csv`
- `passive_settling.csv`
- `pid_hold_settling.csv`
- `step5_after_reset_fix_smoke.csv`

- [ ] **Step 6: Backup and replace table only if gates pass**

If all heights pass validation:
- copy `configs/controllers/b9_balanced_root_init_table.yaml` to `outputs/phase_b9_step5_13_reset_equilibrium_fix/b9_balanced_root_init_table.before_step5_13.yaml`;
- replace `configs/controllers/b9_balanced_root_init_table.yaml` with the new table.

If validation fails:
- do not replace config;
- write failure summary and stop before Step 5 revalidation.

---

### Task 3: Revalidate Step 5 baseline

**Files:**
- Modify: `scripts/phase_b9_step5_13_reset_equilibrium_fix.py`

- [ ] **Step 1: Add revalidation runner**

Use existing Step 5 best controller from `outputs/phase_b9_lqr_gain_strengthening/best_lqr_config.yaml`, with full-root reset and fixed controller targets per height.

- [ ] **Step 2: Log metrics**

For 5 episodes per height, log:
- survival time;
- fall flag and reason;
- pitch RMS;
- roll RMS;
- action saturation;
- first failure variable;
- contact force and clearance stats.

- [ ] **Step 3: Write revalidation outputs**

Write:
- `step5_revalidation_after_reset_fix.csv`
- `step5_revalidation_after_reset_fix_summary.json`

---

### Task 4: Update reports

**Files:**
- Modify: `docs/phase_b9_best_standalone_controller_report.md`
- Modify: `docs/phase_b9_audit_gate_report.md`

- [ ] **Step 1: Add Step 5.13 report section**

Add concise sections covering:
- whether old table was invalid;
- whether Step 5 used joint-only reset;
- what was fixed;
- whether contact/settling gates passed;
- new Step 5 baseline;
- Step 5 pass/fail;
- Step 6 status.

- [ ] **Step 2: Verify report references artifact paths**

Ensure both docs point to `outputs/phase_b9_step5_13_reset_equilibrium_fix/` artifacts.

---

### Task 5: Verification

**Files:**
- Test: `tests/test_phase_b9_step5_13_reset_equilibrium_fix.py`
- Test: `tests/test_phase_b9_step5_10_logic.py`
- Test: `tests/test_phase_b9_step5_11_corrective_path_audit.py`

- [ ] **Step 1: Run targeted tests**

Run:

```bash
pytest -q tests/test_phase_b9_step5_13_reset_equilibrium_fix.py
pytest -q tests/test_phase_b9_step5_10_logic.py
pytest -q tests/test_phase_b9_step5_11_corrective_path_audit.py
```

- [ ] **Step 2: Run repair script**

Run:

```bash
python scripts/phase_b9_step5_13_reset_equilibrium_fix.py
```

- [ ] **Step 3: Inspect final artifacts**

Read:
- `reset_equilibrium_summary.json`
- `step5_revalidation_after_reset_fix_summary.json` if produced
- docs sections updated in Task 4

- [ ] **Step 4: Decide final outcome**

Choose exactly one:
- `A. RESET_BUG_NOT_CONFIRMED`
- `B. RESET_TABLE_INVALID_BUT_NOT_FIXED`
- `C. FULL_ROOT_INIT_BUG_FIXED_TABLE_STILL_INVALID`
- `D. RESET_EQUILIBRIUM_FIXED_BASELINE_REVALIDATED`
- `E. RESET_FIXED_AND_STEP5_NOW_PASSES`
