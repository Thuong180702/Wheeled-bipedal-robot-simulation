# Phase B.9 Step 5.17 Torque-Level WBC Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a disabled-by-default diagnostic torque/generalized-force WBC prototype to test whether torque authority can stabilize roll/lateral dynamics better than position-target WBC/VMC.

**Architecture:** Keep the current PID action path and residual PPO semantics untouched. Add a small diagnostic torque-WBC helper that computes bounded generalized joint-force injection (`qfrc_applied`) for allowed actuated joints, then add a Step 5.17 script that audits actuator semantics, writes the design, runs diagnostic response/candidate artifacts, and stops before full validation unless a candidate passes the small gate.

**Tech Stack:** Python, NumPy, JAX/MJX, MuJoCo CPU/MJX model/data, YAML configs, pytest, CSV/JSON/Markdown artifacts.

---

## File Structure

- Create `wheeled_biped/sim/torque_wbc.py`: pure helper functions for diagnostic torque-WBC command computation and `qfrc_applied` injection. It does not alter PPO action semantics.
- Modify `configs/controllers/dual_rate_balance_controller_b9.yaml`: add disabled-by-default `torque_wbc` config namespace with `diagnostic_only: true`.
- Create `scripts/phase_b9_step5_17_torque_level_wbc_prototype.py`: actuator/interface audit, design doc, response validation, small candidate summary, optional full validation gate.
- Create `tests/test_phase_b9_step5_17_torque_level_wbc.py`: required default-disabled, bounds, allowed indices, telemetry, and config-safety tests.
- Modify `docs/phase_b9_best_standalone_controller_report.md`: append Step 5.17 result.
- Modify `docs/phase_b9_audit_gate_report.md`: append Step 5.17 gate result.
- Do not modify `configs/training/balance_residual.yaml`.
- Do not modify `outputs/phase_b9_lqr_gain_strengthening/best_lqr_config.yaml` unless a full validation beats the reset-fixed baseline.

---

### Task 1: Add failing torque-WBC tests

**Files:**
- Create: `tests/test_phase_b9_step5_17_torque_level_wbc.py`

- [ ] **Step 1: Write failing tests**

Create tests that expect:
- `DualRateConfig.from_yaml("configs/controllers/dual_rate_balance_controller_b9.yaml")` has `torque_wbc_enabled is False` and `torque_wbc_diagnostic_only is True`.
- baseline controller action is identical before/after loading disabled torque config.
- action constants still have dimension/order 10.
- `compute_diagnostic_torque_wbc()` returns bounded commands.
- `apply_qfrc_applied_torque()` only writes qvel joint dof indices 6:16 and leaves root dofs 0:6 zero.
- telemetry contains `enabled`, `diagnostic_only`, `mode`, `tau_roll_des`, `Fy_des`, `delta_Fz_des`, `joint_torque_commands`, `qfrc_applied_indices`, `torque_clamped`, `contact_force_response`, `roll_response`.
- `configs/training/balance_residual.yaml` is unchanged by helper calls.

- [ ] **Step 2: Run tests to verify RED**

Run:
```bash
pytest -q tests/test_phase_b9_step5_17_torque_level_wbc.py
```
Expected: fails because `torque_wbc` config fields and `wheeled_biped.sim.torque_wbc` do not exist.

---

### Task 2: Add disabled torque_wbc config parsing

**Files:**
- Modify: `wheeled_biped/controllers/dual_rate_balance_controller.py`
- Modify: `configs/controllers/dual_rate_balance_controller_b9.yaml`
- Test: `tests/test_phase_b9_step5_17_torque_level_wbc.py`

- [ ] **Step 1: Add config namespace to YAML**

Add:
```yaml
torque_wbc:
  enabled: false
  diagnostic_only: true
  mode: "qfrc_applied"
  gains:
    k_roll: 0.0
    k_roll_rate: 0.0
    k_com_y: 0.0
    k_com_y_rate: 0.0
    k_height: 0.0
    k_height_rate: 0.0
  limits:
    max_joint_torque: 0.0
    max_wheel_torque: 0.0
    max_body_wrench: 0.0
    max_torque_rate: 0.0
  safety:
    disable_on_contact_loss: true
    disable_on_large_pitch: true
    large_pitch_deg: 8.0
    disable_on_large_roll: true
    large_roll_deg: 8.0
```

- [ ] **Step 2: Extend `DualRateConfig` fields and YAML parser**

Add `torque_wbc_*` fields to `DualRateConfig`, parsed with safe defaults matching the YAML.

- [ ] **Step 3: Run config tests**

Run:
```bash
pytest -q tests/test_phase_b9_step5_17_torque_level_wbc.py::test_torque_wbc_disabled_by_default_config tests/test_phase_b9_step5_17_torque_level_wbc.py::test_diagnostic_only_flag_true_by_default
```
Expected: pass.

---

### Task 3: Implement diagnostic torque-WBC helper

**Files:**
- Create: `wheeled_biped/sim/torque_wbc.py`
- Test: `tests/test_phase_b9_step5_17_torque_level_wbc.py`

- [ ] **Step 1: Implement helper dataclasses and pure functions**

Create:
```python
@dataclass(frozen=True)
class TorqueWbcGains:
    k_roll: float = 0.0
    k_roll_rate: float = 0.0
    k_com_y: float = 0.0
    k_com_y_rate: float = 0.0
    k_height: float = 0.0
    k_height_rate: float = 0.0

@dataclass(frozen=True)
class TorqueWbcLimits:
    max_joint_torque: float = 0.0
    max_wheel_torque: float = 0.0
    max_body_wrench: float = 0.0
    max_torque_rate: float = 0.0
```

Implement `compute_diagnostic_torque_wbc(obs, gains, limits, mode="torque_roll_plus_lateral", diagnostic_only=True)` returning `(joint_torque_commands, telemetry)`.

- [ ] **Step 2: Implement allowed qfrc injection**

Implement `apply_qfrc_applied_torque(mjx_data, joint_torque_commands, allowed_action_indices=None)`:
```python
allowed_action_indices = [0, 2, 3, 4, 5, 7, 8, 9]
qfrc = jnp.zeros_like(mjx_data.qfrc_applied)
for action_idx in allowed_action_indices:
    qfrc_idx = 6 + action_idx
    qfrc = qfrc.at[qfrc_idx].set(joint_torque_commands[action_idx])
return mjx_data.replace(qfrc_applied=qfrc), qfrc
```
Never write root qfrc indices `0:6` or hip-yaw action indices `1,6`.

- [ ] **Step 3: Run helper tests**

Run:
```bash
pytest -q tests/test_phase_b9_step5_17_torque_level_wbc.py
```
Expected: torque helper tests pass.

---

### Task 4: Add Step 5.17 diagnostic script

**Files:**
- Create: `scripts/phase_b9_step5_17_torque_level_wbc_prototype.py`
- Test: `tests/test_phase_b9_step5_17_torque_level_wbc.py`

- [ ] **Step 1: Implement output paths and candidate list**

Use:
```python
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase_b9_step5_17_torque_level_wbc_prototype"
```
Candidates:
- `torque_roll_only`
- `torque_lateral_com_only`
- `torque_roll_plus_lateral`
- `hybrid_pid_plus_torque_roll`
- `conservative_torque_wbc`

- [ ] **Step 2: Implement actuator/interface audit**

Read MuJoCo model and write JSON/Markdown answering the nine audit questions. Required conclusions from observed files:
- MJCF uses `<motor>` actuators with `ctrlrange` and `forcerange`.
- actuator `ctrl` is torque-like motor command at MJCF level.
- current env PID path writes motor torques to `ctrl` after interpreting actions as leg position / wheel velocity targets.
- `qfrc_applied` and `xfrc_applied` are MJX data fields usable for diagnostics.
- qfrc diagnostic injection can be added without changing PPO action semantics if disabled and script/helper-only.

- [ ] **Step 3: Implement design doc output**

Write `torque_wbc_design.md` explaining selected diagnostic mode: hybrid PID posture controller + diagnostic `qfrc_applied` torque residual for roll/lateral stabilization.

- [ ] **Step 4: Implement response/candidate artifacts**

Generate:
- `response_validation.csv`
- `response_validation_summary.json`
- `candidate_results.csv`
- `candidate_summary.json`

If no stabilizing response is validated in the diagnostic artifact gate, set decision `TORQUE_WBC_NO_STABILIZING_AUTHORITY`. If diagnostic authority exists but cannot pass deployable gate because it depends on `qfrc_applied`, set decision `LOW_LEVEL_CONTROL_REDESIGN_REQUIRED`.

- [ ] **Step 5: Run script**

Run:
```bash
python scripts/phase_b9_step5_17_torque_level_wbc_prototype.py
```
Expected: artifacts are written under `outputs/phase_b9_step5_17_torque_level_wbc_prototype/`, and final printed decision is one of the allowed Step 5.17 decisions.

---

### Task 5: Update reports

**Files:**
- Modify: `docs/phase_b9_best_standalone_controller_report.md`
- Modify: `docs/phase_b9_audit_gate_report.md`

- [ ] **Step 1: Append Step 5.17 sections**

Include torque feasibility, qfrc/xfrc usage, diagnostic-only status, response validation result, candidate result, full validation status, Step 5 status, Step 6 status, final decision, and current best controller.

- [ ] **Step 2: Verify docs**

Run:
```bash
python - <<'PY'
from pathlib import Path
for p in [Path('docs/phase_b9_best_standalone_controller_report.md'), Path('docs/phase_b9_audit_gate_report.md')]:
    text = p.read_text(encoding='utf-8')
    assert 'Step 5.17' in text
    assert 'BLOCKED' in text
print('docs ok')
PY
```
Expected: `docs ok`.

---

### Task 6: Final verification

**Files:**
- Test: Step 5.13/5.14/5.15/5.16/5.17/5.10/5.11 tests

- [ ] **Step 1: Run required tests**

Run:
```bash
pytest -q tests/test_phase_b9_step5_13_reset_equilibrium_fix.py && pytest -q tests/test_phase_b9_step5_14_lateral_balance_layer.py && pytest -q tests/test_phase_b9_step5_15_vmc_whole_body_layer.py && pytest -q tests/test_phase_b9_step5_16_jacobian_wbc_vmc.py && pytest -q tests/test_phase_b9_step5_17_torque_level_wbc.py && pytest -q tests/test_phase_b9_step5_10_logic.py && pytest -q tests/test_phase_b9_step5_11_corrective_path_audit.py
```
Expected: all pass.

- [ ] **Step 2: Verify protected files unchanged**

Run:
```bash
git diff -- configs/training/balance_residual.yaml
```
Expected: no diff.

Run:
```bash
git diff -- outputs/phase_b9_lqr_gain_strengthening/best_lqr_config.yaml
```
Expected: no diff.

---

## Self-review

- Spec coverage: plan covers actuator audit, disabled config, diagnostic-only qfrc path, telemetry, response/candidate artifacts, optional full-validation gate, tests, report updates, protected config, Step 6 blocked status.
- Placeholder scan: no TBD/TODO placeholders remain.
- Type consistency: `torque_wbc_*`, `TorqueWbcGains`, `TorqueWbcLimits`, `compute_diagnostic_torque_wbc`, and `apply_qfrc_applied_torque` names are consistent.
