# Phase B.9 Step 5.18 Deployable Motor-Torque Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in deployable actuator-ctrl motor torque path for WBC experiments while preserving the default position-PID / velocity-PID baseline path.

**Architecture:** Keep the current `pid_position_velocity` behavior as the default. Add low-level helper functions that map normalized torque commands directly to MJCF actuator `ctrl` and combine bounded torque residuals with PID output for hybrid mode. Wire an opt-in `low_level_control` namespace into `BalanceEnv` without changing action dimension, action order, residual PPO config, or current best controller artifacts.

**Tech Stack:** Python, JAX/MJX, MuJoCo actuator metadata, YAML-style config dicts, pytest, CSV/JSON/Markdown diagnostic artifacts.

---

## File Structure

- Modify `wheeled_biped/sim/low_level_control.py`: add `normalized_motor_torque_control`, `hybrid_pid_plus_torque_control`, and `actuator_saturation_flags` helpers.
- Modify `wheeled_biped/envs/balance_env.py`: parse opt-in `low_level_control` config; route `pid_position_velocity`, `motor_torque`, and `hybrid_pid_plus_torque`; add numeric telemetry in `state.info`.
- Create `tests/test_phase_b9_step5_18_deployable_motor_torque_interface.py`: TDD coverage for default mode, direct motor torque, hybrid residual, clamps, mapping, telemetry, protected config.
- Create `scripts/phase_b9_step5_18_deployable_motor_torque_interface.py`: motor interface audit, design doc, response/candidate/full-validation gate artifacts.
- Modify `docs/phase_b9_best_standalone_controller_report.md`: append Step 5.18 result.
- Modify `docs/phase_b9_audit_gate_report.md`: append Step 5.18 gate result.
- Do not modify `configs/training/balance_residual.yaml`.
- Do not modify `outputs/phase_b9_lqr_gain_strengthening/best_lqr_config.yaml` unless full validation beats the reset-fixed baseline.

---

### Task 1: Write Step 5.18 failing tests

**Files:**
- Create: `tests/test_phase_b9_step5_18_deployable_motor_torque_interface.py`

- [ ] **Step 1: Add tests**

Create tests that assert:
- `BalanceEnv({})._low_level_mode == "pid_position_velocity"`.
- Default `BalanceEnv` step output matches pre-existing PID/direct behavior when torque mode is not enabled.
- `normalized_motor_torque_control(jnp.array([-1, 0, 1]), ctrl_min, ctrl_max, max_ctrl_fraction=0.5)` writes direct actuator ctrl values at half range.
- `hybrid_pid_plus_torque_control(pid_ctrl, torque_residual, ctrl_min, ctrl_max, max_ctrl_fraction=0.25)` adds bounded residual and clamps.
- action constants remain `[0..9]` and `ACTION_DIM == 10`.
- actuator index equals action index for all ten MJCF actuators.
- `BalanceEnv` motor-torque opt-in writes `mjx_data.ctrl` directly from normalized action.
- `BalanceEnv` hybrid opt-in emits info keys `raw_pid_ctrl`, `torque_residual_ctrl`, `final_actuator_ctrl`, `actuator_saturation_flags`, `low_level_mode_code`, `torque_control_enabled`.
- `configs/training/balance_residual.yaml` content is unchanged by helper calls.

- [ ] **Step 2: Run RED**

Run:
```bash
pytest -q tests/test_phase_b9_step5_18_deployable_motor_torque_interface.py
```
Expected: fails because the new helper functions and env config fields do not exist.

---

### Task 2: Implement motor torque helpers

**Files:**
- Modify: `wheeled_biped/sim/low_level_control.py`
- Test: `tests/test_phase_b9_step5_18_deployable_motor_torque_interface.py`

- [ ] **Step 1: Add helper functions**

Add:
```python
def normalized_motor_torque_control(normalized_torque, ctrl_min, ctrl_max, max_ctrl_fraction=1.0, allow_mask=None):
    normalized_torque = jnp.clip(normalized_torque, -1.0, 1.0)
    max_ctrl_fraction = jnp.clip(max_ctrl_fraction, 0.0, 1.0)
    ctrl_limit = jnp.minimum(jnp.abs(ctrl_min), jnp.abs(ctrl_max)) * max_ctrl_fraction
    ctrl = normalized_torque * ctrl_limit
    if allow_mask is not None:
        ctrl = ctrl * allow_mask
    return jnp.clip(ctrl, ctrl_min, ctrl_max)
```

Add:
```python
def hybrid_pid_plus_torque_control(pid_ctrl, normalized_torque_residual, ctrl_min, ctrl_max, max_ctrl_fraction=1.0, allow_mask=None):
    residual = normalized_motor_torque_control(normalized_torque_residual, ctrl_min, ctrl_max, max_ctrl_fraction, allow_mask)
    final = jnp.clip(pid_ctrl + residual, ctrl_min, ctrl_max)
    return final, residual
```

Add:
```python
def actuator_saturation_flags(ctrl, ctrl_min, ctrl_max, eps=1e-6):
    return (ctrl <= ctrl_min + eps) | (ctrl >= ctrl_max - eps)
```

- [ ] **Step 2: Run helper tests**

Run:
```bash
pytest -q tests/test_phase_b9_step5_18_deployable_motor_torque_interface.py::test_motor_torque_helper_maps_normalized_action_to_direct_ctrl tests/test_phase_b9_step5_18_deployable_motor_torque_interface.py::test_hybrid_helper_adds_bounded_torque_residual_to_pid_ctrl
```
Expected: pass.

---

### Task 3: Wire opt-in BalanceEnv mode switch

**Files:**
- Modify: `wheeled_biped/envs/balance_env.py`
- Test: `tests/test_phase_b9_step5_18_deployable_motor_torque_interface.py`

- [ ] **Step 1: Import new helpers**

Change import from `wheeled_biped.sim.low_level_control` to include:
```python
actuator_saturation_flags,
hybrid_pid_plus_torque_control,
normalized_motor_torque_control,
pid_control,
```

- [ ] **Step 2: Parse `low_level_control` config in `BalanceEnv.__init__`**

Add after PID config parsing:
```python
ll_cfg = self.config.get("low_level_control", {})
torque_cfg = ll_cfg.get("torque_control", {})
self._low_level_mode = str(ll_cfg.get("mode", "pid_position_velocity"))
self._torque_control_enabled = bool(torque_cfg.get("enabled", False))
self._torque_max_ctrl_fraction = float(torque_cfg.get("max_ctrl_fraction", 1.0))
self._torque_allow_leg = bool(torque_cfg.get("allow_leg_torque", True))
self._torque_allow_wheel = bool(torque_cfg.get("allow_wheel_torque", True))
self._torque_allow_hip_yaw = bool(torque_cfg.get("allow_hip_yaw_torque", False))
```
Then build `self._torque_allow_mask` with zeros for hip yaw unless explicitly allowed, legs/wheels based on config.

- [ ] **Step 3: Add telemetry defaults in reset info**

Add to `info`:
```python
"last_actuator_ctrl": jnp.zeros(self.num_actions, dtype=jnp.float32),
"raw_pid_ctrl": jnp.zeros(self.num_actions, dtype=jnp.float32),
"torque_residual_ctrl": jnp.zeros(self.num_actions, dtype=jnp.float32),
"final_actuator_ctrl": jnp.zeros(self.num_actions, dtype=jnp.float32),
"actuator_saturation_flags": jnp.zeros(self.num_actions, dtype=jnp.bool_),
"low_level_mode_code": jnp.int32(0),
"torque_control_enabled": jnp.bool_(False),
"torque_safety_disabled": jnp.bool_(False),
"torque_residual_action": jnp.zeros(self.num_actions, dtype=jnp.float32),
```

- [ ] **Step 4: Replace Step 3 low-level action conversion block**

Use static Python branches on `self._low_level_mode`:
- default `pid_position_velocity`: existing PID/direct code unchanged.
- `motor_torque` when enabled: `scaled_action = normalized_motor_torque_control(control_action, self._ctrl_min, self._ctrl_max, self._torque_max_ctrl_fraction, self._torque_allow_mask)` and `pid_integral` unchanged.
- `hybrid_pid_plus_torque` when enabled: compute PID as before, read `state.info["torque_residual_action"]`, call `hybrid_pid_plus_torque_control(...)`.

- [ ] **Step 5: Add telemetry to `new_info`**

Add `raw_pid_ctrl`, `torque_residual_ctrl`, `final_actuator_ctrl`, `actuator_saturation_flags`, `low_level_mode_code`, `torque_control_enabled`, `torque_safety_disabled`, `last_actuator_ctrl`, and carry `torque_residual_action`.

- [ ] **Step 6: Run env tests**

Run:
```bash
pytest -q tests/test_phase_b9_step5_18_deployable_motor_torque_interface.py
```
Expected: pass.

---

### Task 4: Add Step 5.18 diagnostic script and artifacts

**Files:**
- Create: `scripts/phase_b9_step5_18_deployable_motor_torque_interface.py`

- [ ] **Step 1: Add constants and candidates**

Use output dir:
```python
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase_b9_step5_18_deployable_motor_torque_interface"
```
Candidates:
- `motor_torque_roll_only`
- `motor_torque_lateral_com_only`
- `motor_torque_roll_plus_lateral`
- `hybrid_pid_plus_torque_roll`
- `hybrid_pid_plus_torque_wbc`
- `conservative_motor_torque_wbc`

- [ ] **Step 2: Write motor audit artifacts**

Load MuJoCo model and write `motor_interface_audit.md/json` with all ten actuator names, action index mapping, ctrlrange, forcerange, gear, and conclusion that actuator `ctrl` is deployable motor torque command in MJCF while current baseline uses PID output as `ctrl`.

- [ ] **Step 3: Write design artifact**

Write `low_level_torque_design.md` documenting modes:
- `pid_position_velocity`
- `motor_torque`
- `hybrid_pid_plus_torque`

- [ ] **Step 4: Write response/candidate/full gate artifacts**

Generate CSV/JSON files. If static deployable motor torque response exists but no survival rollout is executed in this diagnostic patch, set final decision `F. HYBRID_PID_TORQUE_REQUIRED` when hybrid is the safest deployable path, or `D. MOTOR_TORQUE_IMPROVES_BUT_DOES_NOT_PASS_GATE` if candidate artifacts show static authority but no full gate pass.

- [ ] **Step 5: Run script**

Run:
```bash
python scripts/phase_b9_step5_18_deployable_motor_torque_interface.py
```
Expected: output artifacts are created and final decision is one of the allowed Step 5.18 decisions.

---

### Task 5: Update reports

**Files:**
- Modify: `docs/phase_b9_best_standalone_controller_report.md`
- Modify: `docs/phase_b9_audit_gate_report.md`

- [ ] **Step 1: Append Step 5.18 sections**

Report motor torque deployability, modes added, default PID unchanged, response validation, candidate results, full validation status, final decision, current best controller, Step 6 blocked.

- [ ] **Step 2: Verify docs**

Run:
```bash
python - <<'PY'
from pathlib import Path
for p in [Path('docs/phase_b9_best_standalone_controller_report.md'), Path('docs/phase_b9_audit_gate_report.md')]:
    text = p.read_text(encoding='utf-8')
    assert 'Step 5.18' in text
    assert 'BLOCKED' in text
print('docs ok')
PY
```
Expected: `docs ok`.

---

### Task 6: Final verification

**Files:**
- Test: Step 5.10/5.11/5.13/5.14/5.15/5.16/5.17/5.18 tests

- [ ] **Step 1: Run required tests**

Run:
```bash
pytest -q tests/test_phase_b9_step5_13_reset_equilibrium_fix.py && pytest -q tests/test_phase_b9_step5_14_lateral_balance_layer.py && pytest -q tests/test_phase_b9_step5_15_vmc_whole_body_layer.py && pytest -q tests/test_phase_b9_step5_16_jacobian_wbc_vmc.py && pytest -q tests/test_phase_b9_step5_17_torque_level_wbc.py && pytest -q tests/test_phase_b9_step5_18_deployable_motor_torque_interface.py && pytest -q tests/test_phase_b9_step5_10_logic.py && pytest -q tests/test_phase_b9_step5_11_corrective_path_audit.py
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

- Spec coverage: plan covers motor audit, low-level mode switch, deployable direct motor torque, hybrid PID+torque, telemetry, artifacts, tests, report updates, protected files, and Step 6 BLOCKED.
- Placeholder scan: no TBD/TODO placeholders remain.
- Type consistency: `pid_position_velocity`, `motor_torque`, `hybrid_pid_plus_torque`, helper names, telemetry keys, and output paths are consistent.
