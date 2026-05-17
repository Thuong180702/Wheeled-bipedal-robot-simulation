# Phase B.9 Step 5.16 Mainline Jacobian WBC/VMC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate Step 5.14/5.15 as maintained mainline controller infrastructure, then implement and evaluate Step 5.16 Jacobian-informed WBC/VMC target-offset control.

**Architecture:** First restore the prior worktree Step 5.14/5.15 controller/config/script/test infrastructure into current main without changing default baseline behavior. Then add a disabled-by-default `wbc_vmc` controller namespace that computes desired virtual wrench/contact redistribution and maps it to bounded normalized target offsets using MuJoCo Jacobian diagnostics and conservative runtime mappings. Evaluation remains reset-fixed/post-reset only, and Step 6 remains blocked unless explicitly authorized later.

**Tech Stack:** Python, NumPy, JAX/MJX, MuJoCo CPU diagnostics, YAML configs, pytest, CSV/JSON/Markdown artifacts.

---

## File Structure

- Modify `wheeled_biped/controllers/dual_rate_balance_controller.py`: restore `lateral_balance` and `vmc_whole_body`; add `wbc_vmc` config fields, telemetry, and bounded action-offset logic.
- Modify `configs/controllers/dual_rate_balance_controller_b9.yaml`: restore `lateral_balance` and `vmc_whole_body` blocks; add disabled `wbc_vmc` block.
- Create `scripts/phase_b9_step5_14_lateral_balance_layer.py`: copy prior Step 5.14 evaluation script.
- Create `scripts/phase_b9_step5_15_vmc_whole_body_layer.py`: copy prior Step 5.15 evaluation script.
- Create `scripts/phase_b9_step5_16_jacobian_wbc_vmc.py`: interface audit, Jacobian audit, response validation, candidate eval, optional full validation.
- Create `tests/test_phase_b9_step5_14_lateral_balance_layer.py`: copy prior Step 5.14 tests.
- Create `tests/test_phase_b9_step5_15_vmc_whole_body_layer.py`: copy prior Step 5.15 tests.
- Create `tests/test_phase_b9_step5_16_jacobian_wbc_vmc.py`: WBC disabled/default safety, telemetry, bounds, Jacobian sign helpers.
- Modify `docs/phase_b9_best_standalone_controller_report.md`: append Step 5.16 result.
- Modify `docs/phase_b9_audit_gate_report.md`: append Step 5.16 gate result.
- Do not modify `configs/training/balance_residual.yaml`.

---

### Task 1: Restore Step 5.14/5.15 mainline files

**Files:**
- Modify: `wheeled_biped/controllers/dual_rate_balance_controller.py`
- Modify: `configs/controllers/dual_rate_balance_controller_b9.yaml`
- Create: `scripts/phase_b9_step5_14_lateral_balance_layer.py`
- Create: `scripts/phase_b9_step5_15_vmc_whole_body_layer.py`
- Create: `tests/test_phase_b9_step5_14_lateral_balance_layer.py`
- Create: `tests/test_phase_b9_step5_15_vmc_whole_body_layer.py`

- [ ] **Step 1: Copy prior worktree files into main**

Run:
```bash
cp ".claude/worktrees/phase-b9-step5-13-reset-equilibrium/wheeled_biped/controllers/dual_rate_balance_controller.py" "wheeled_biped/controllers/dual_rate_balance_controller.py" && cp ".claude/worktrees/phase-b9-step5-13-reset-equilibrium/configs/controllers/dual_rate_balance_controller_b9.yaml" "configs/controllers/dual_rate_balance_controller_b9.yaml" && cp ".claude/worktrees/phase-b9-step5-13-reset-equilibrium/scripts/phase_b9_step5_14_lateral_balance_layer.py" "scripts/phase_b9_step5_14_lateral_balance_layer.py" && cp ".claude/worktrees/phase-b9-step5-13-reset-equilibrium/scripts/phase_b9_step5_15_vmc_whole_body_layer.py" "scripts/phase_b9_step5_15_vmc_whole_body_layer.py" && cp ".claude/worktrees/phase-b9-step5-13-reset-equilibrium/tests/test_phase_b9_step5_14_lateral_balance_layer.py" "tests/test_phase_b9_step5_14_lateral_balance_layer.py" && cp ".claude/worktrees/phase-b9-step5-13-reset-equilibrium/tests/test_phase_b9_step5_15_vmc_whole_body_layer.py" "tests/test_phase_b9_step5_15_vmc_whole_body_layer.py"
```
Expected: command exits 0.

- [ ] **Step 2: Verify default config keeps restored modules disabled**

Run:
```bash
pytest -q tests/test_phase_b9_step5_14_lateral_balance_layer.py::test_lateral_balance_disabled_by_default_config tests/test_phase_b9_step5_15_vmc_whole_body_layer.py::test_vmc_disabled_by_default_config
```
Expected: both tests pass.

- [ ] **Step 3: Run required integration tests**

Run:
```bash
pytest -q tests/test_phase_b9_step5_13_reset_equilibrium_fix.py && pytest -q tests/test_phase_b9_step5_14_lateral_balance_layer.py && pytest -q tests/test_phase_b9_step5_15_vmc_whole_body_layer.py && pytest -q tests/test_phase_b9_step5_10_logic.py && pytest -q tests/test_phase_b9_step5_11_corrective_path_audit.py
```
Expected: all pass. If a test file is missing in current main, inspect whether it exists in the prior worktree or implement the expected test file before proceeding.

---

### Task 2: Add WBC/VMC config and telemetry shell

**Files:**
- Modify: `wheeled_biped/controllers/dual_rate_balance_controller.py`
- Modify: `configs/controllers/dual_rate_balance_controller_b9.yaml`
- Test: `tests/test_phase_b9_step5_16_jacobian_wbc_vmc.py`

- [ ] **Step 1: Write tests for disabled default and telemetry**

Create `tests/test_phase_b9_step5_16_jacobian_wbc_vmc.py` with tests that instantiate `DualRateConfig.from_yaml`, assert `wbc_vmc_enabled is False`, enable it manually, call `compute_action`, and assert telemetry contains `tau_roll_des`, `Fy_des`, `Fz_des`, `delta_Fz_des`, `Fz_left_des`, `Fz_right_des`, `hip_roll_offset_left`, `hip_roll_offset_right`, `hip_pitch_offset_left`, `hip_pitch_offset_right`, `knee_offset_left`, `knee_offset_right`, `wheel_diff_cmd`, `clamped`, `wheel_unload_flag`, and `mapping_mode`.

- [ ] **Step 2: Run tests to verify failure before implementation**

Run:
```bash
pytest -q tests/test_phase_b9_step5_16_jacobian_wbc_vmc.py
```
Expected: fails because `wbc_vmc` fields and telemetry do not exist.

- [ ] **Step 3: Add config block**

Add to `configs/controllers/dual_rate_balance_controller_b9.yaml`:
```yaml
wbc_vmc:
  enabled: false
  mode: "jacobian_combined"
  update_rate_hz: 50
  use_mujoco_jacobian: true
  use_finite_difference_fallback: true
  compose_with_lateral_balance: false
  compose_with_vmc_whole_body: false

  gains:
    k_roll: 0.0
    k_roll_rate: 0.0
    k_com_y: 0.0
    k_com_y_rate: 0.0
    k_height: 0.0
    k_height_rate: 0.0
    k_force_balance: 0.0

  limits:
    max_delta_fz: 0.0
    max_hip_roll_offset: 0.0
    max_hip_pitch_offset: 0.0
    max_knee_offset: 0.0
    max_wheel_diff_cmd: 0.0
    max_correction_rate: 0.0

  mappings:
    use_hip_roll: true
    use_hip_pitch: true
    use_knee: true
    use_wheel_diff: false

  safety:
    disable_on_wheel_unload: true
    disable_on_large_pitch: true
    large_pitch_deg: 8.0
    disable_on_large_contact_impulse: true
    large_contact_impulse_n: 2000.0
```

- [ ] **Step 4: Extend `DualRateConfig`**

Add dataclass fields for every scalar above with names prefixed `wbc_vmc_`, parse them in `from_yaml` with disabled-safe defaults.

- [ ] **Step 5: Add default telemetry dict**

In `DualRateBalanceController.__init__`, initialize `self.last_wbc_vmc_terms` with all required telemetry keys and safe zero values.

- [ ] **Step 6: Add telemetry to `get_telemetry()`**

Return `"wbc_vmc": self.last_wbc_vmc_terms` in the controller telemetry dict.

- [ ] **Step 7: Run tests**

Run:
```bash
pytest -q tests/test_phase_b9_step5_16_jacobian_wbc_vmc.py
```
Expected: disabled-default and telemetry-shell tests pass.

---

### Task 3: Implement bounded WBC/VMC target-offset action logic

**Files:**
- Modify: `wheeled_biped/controllers/dual_rate_balance_controller.py`
- Test: `tests/test_phase_b9_step5_16_jacobian_wbc_vmc.py`

- [ ] **Step 1: Add tests for bounds and allowed indices**

Add tests that enable `wbc_vmc`, set large gains and small limits, call `compute_action` with roll perturbation, then assert only indices `{0, 2, 3, 4, 5, 7, 8, 9}` can differ from a baseline action and indices `{1, 6}` remain zero.

- [ ] **Step 2: Implement WBC computation after base action and before final clip**

In `compute_action`, after `vmc_whole_body`/`lateral_balance` handling is integrated, add `wbc_vmc` logic that computes:
```python
roll = float(np.arcsin(np.clip(gravity_body[1], -1.0, 1.0)))
roll_rate = float(obs[7])
height = current_height_norm * (self.config.height_max - self.config.height_min) + self.config.height_min
height_rate = float(obs[5]) if len(obs) > 5 else 0.0
support_width = 0.23
mass = 8.1
gravity = 9.81
tau_roll_des = -(self.config.wbc_vmc_k_roll * roll + self.config.wbc_vmc_k_roll_rate * roll_rate)
Fy_des = -(self.config.wbc_vmc_k_com_y * com_y_error + self.config.wbc_vmc_k_com_y_rate * com_y_dot)
Fz_des = mass * gravity - self.config.wbc_vmc_k_height * (height - height_cmd_m) - self.config.wbc_vmc_k_height_rate * height_rate
delta_Fz_des = tau_roll_des / max(support_width, 1e-6)
delta_Fz_des = np.clip(delta_Fz_des, -self.config.wbc_vmc_max_delta_fz, self.config.wbc_vmc_max_delta_fz)
Fz_left_des = max(0.0, 0.5 * Fz_des + delta_Fz_des)
Fz_right_des = max(0.0, 0.5 * Fz_des - delta_Fz_des)
```
Then map normalized bounded offsets:
```python
hip_roll_offset = np.clip(delta_Fz_des / max(self.config.wbc_vmc_max_delta_fz, 1e-6), -1.0, 1.0) * self.config.wbc_vmc_max_hip_roll_offset
leg_offset = np.clip(delta_Fz_des / max(self.config.wbc_vmc_max_delta_fz, 1e-6), -1.0, 1.0)
hip_pitch_offset = leg_offset * self.config.wbc_vmc_max_hip_pitch_offset
knee_offset = leg_offset * self.config.wbc_vmc_max_knee_offset
wheel_diff_cmd = np.clip(Fy_des / 80.0, -self.config.wbc_vmc_max_wheel_diff_cmd, self.config.wbc_vmc_max_wheel_diff_cmd)
```
Apply signs conservatively and update telemetry. Keep final `clip_normalized_action(action)`.

- [ ] **Step 3: Run bounds tests**

Run:
```bash
pytest -q tests/test_phase_b9_step5_16_jacobian_wbc_vmc.py
```
Expected: tests pass.

---

### Task 4: Add Step 5.16 diagnostic/evaluation script

**Files:**
- Create: `scripts/phase_b9_step5_16_jacobian_wbc_vmc.py`
- Test: `tests/test_phase_b9_step5_16_jacobian_wbc_vmc.py`

- [ ] **Step 1: Add script helper tests**

Test pure helper functions only: `normalized_action_offset_from_joint_delta`, `write_json`, `write_csv`, `site_jacobian_rows`, and candidate list contains the required six candidates.

- [ ] **Step 2: Implement output constants and candidate dataclass**

Create constants:
```python
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase_b9_step5_16_jacobian_wbc_vmc"
BEST_LQR_PATH = PROJECT_ROOT / "outputs" / "phase_b9_lqr_gain_strengthening" / "best_lqr_config.yaml"
CONTROLLER_CONFIG_PATH = PROJECT_ROOT / "configs" / "controllers" / "dual_rate_balance_controller_b9.yaml"
BALANCE_RESIDUAL_PATH = PROJECT_ROOT / "configs" / "training" / "balance_residual.yaml"
```
Add `WbcCandidate` with fields for mode, gains, limits, mappings, wheel assist, and composition flags.

- [ ] **Step 3: Implement interface audit**

Write JSON/Markdown with qpos/qvel indices, action indices, actuator semantics, model sites/bodies/geoms, contact names, and statement that current action interface is target-offset VMC, not direct torque WBC.

- [ ] **Step 4: Implement Jacobian audit**

Use `mujoco.mj_jacSite(model, data, jacp, jacr, site_id)` for `l_wheel_contact` and `r_wheel_contact`, extract translational Jacobian columns for hip_roll/hip_pitch/knee dofs, and save CSV/JSON signs.

- [ ] **Step 5: Implement response validation**

At h=0.60, apply reset-fixed balanced root init, perturb roll by ±2 deg, optionally y offset by ±0.01 m, run one control step, measure force redistribution, roll change, pitch disturbance, height disturbance, wheel unload, action saturation, and clamp telemetry.

- [ ] **Step 6: Implement small eval and optional full validation**

Reuse Step 5.15 evaluation patterns. Compare h=0.60 against supplied reset-fixed h=0.60 baseline. Only full-validate best 1–2 candidates if keep rule passes.

- [ ] **Step 7: Run script**

Run:
```bash
python scripts/phase_b9_step5_16_jacobian_wbc_vmc.py
```
Expected: output artifacts are written under `outputs/phase_b9_step5_16_jacobian_wbc_vmc/`; final decision is one of the allowed Step 5.16 decisions.

---

### Task 5: Add report updates

**Files:**
- Modify: `docs/phase_b9_best_standalone_controller_report.md`
- Modify: `docs/phase_b9_audit_gate_report.md`

- [ ] **Step 1: Append Step 5.16 report section**

Add a concise section with baseline used, Step 5.14/5.15 mainline integration status, torque-WBC feasibility, mapping result, response validation, candidate/full validation, final decision, current best controller, and Step 6 status.

- [ ] **Step 2: Verify docs contain Step 5.16 and BLOCKED status**

Run:
```bash
python - <<'PY'
from pathlib import Path
for p in [Path('docs/phase_b9_best_standalone_controller_report.md'), Path('docs/phase_b9_audit_gate_report.md')]:
    text = p.read_text(encoding='utf-8')
    assert 'Step 5.16' in text
    assert 'BLOCKED' in text
print('docs ok')
PY
```
Expected: `docs ok`.

---

### Task 6: Final verification

**Files:**
- Test: Step 5.13/5.14/5.15/5.16/5.10/5.11 tests

- [ ] **Step 1: Run required test suite**

Run:
```bash
pytest -q tests/test_phase_b9_step5_13_reset_equilibrium_fix.py && pytest -q tests/test_phase_b9_step5_14_lateral_balance_layer.py && pytest -q tests/test_phase_b9_step5_15_vmc_whole_body_layer.py && pytest -q tests/test_phase_b9_step5_16_jacobian_wbc_vmc.py && pytest -q tests/test_phase_b9_step5_10_logic.py && pytest -q tests/test_phase_b9_step5_11_corrective_path_audit.py
```
Expected: all pass.

- [ ] **Step 2: Verify residual config unchanged by content hash or git diff**

Run:
```bash
git diff -- configs/training/balance_residual.yaml
```
Expected: no diff.

- [ ] **Step 3: Verify current best controller remains unchanged unless full validation beats baseline**

Run:
```bash
git diff -- outputs/phase_b9_lqr_gain_strengthening/best_lqr_config.yaml
```
Expected: no diff.

---

## Self-review

- Spec coverage: plan covers mainline Step 5.14/5.15 integration, disabled-by-default config policy, Step 5.16 Jacobian/WBC extension, output directory, tests, reports, and Step 6/PPO constraints.
- Placeholder scan: no TBD/TODO placeholders remain.
- Type consistency: `wbc_vmc_*` config names, `last_wbc_vmc_terms`, and telemetry key names are used consistently across tasks.
