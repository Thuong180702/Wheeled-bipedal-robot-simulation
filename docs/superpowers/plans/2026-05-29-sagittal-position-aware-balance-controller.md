# Sagittal Position-Aware Balance Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a sagittal position-aware balance controller that integrates sagittal position regulation into the core balance state while preserving the validated balance-core architecture, WBC-off behavior, and torque ownership invariants.

**Architecture:** Keep the validated four-source balance-core stack structure, but allow the sagittal wheel source to be supplied by a new `SagittalPositionAwareBalanceController` instead of the current pitch/CP wheel regulator. Design the new controller from a closed-loop identified discrete-time sagittal model, run it as an explicit experimental mode first, and only promote it to default after validation shows substantially reduced drift without breaking balance invariants.

**Execution policy:** Do not execute the full plan as one batch. Each task below has an explicit stop gate. Proceed to the next task only if the current task passes its gate.

**Tech Stack:** Python, JAX/jax.numpy, MuJoCo telemetry scripts, existing balance-core validation workflow, pytest

---

## File Structure

### Files to Modify

- `wheeled_biped/controllers/sagittal_wheel_balance_controller.py`
  - Keep as validated baseline controller.
- `scripts/simulate_hierarchical_controller.py`
  - Add explicit sagittal controller selection and route telemetry for baseline vs position-aware controller.
- `scripts/validate_balance_core.py`
  - Reuse existing validation entrypoint and add sagittal-controller selection passthrough if needed.
- `tests/test_balance_core_components.py`
  - Add state-construction and compatibility tests for the new sagittal controller.
- `tests/test_balance_core_mode_isolation.py`
  - Verify WBC remains off and controller selection does not activate forbidden paths.
- `tests/test_balance_core_validation_workflow.py`
  - Add workflow tests for controller-selection forwarding if CLI changes.

### Files to Create

- `wheeled_biped/controllers/sagittal_position_aware_balance_controller.py`
  - New controller implementation for integrated sagittal balance state regulation.
- `wheeled_biped/controllers/sagittal_balance_state.py`
  - Pure helper(s) for constructing sagittal state and reference values from telemetry/state inputs.
- `scripts/collect_sagittal_balance_sysid_data.py`
  - Closed-loop trajectory collection for sagittal identification.
- `scripts/identify_sagittal_balance_dynamics.py`
  - Fit discrete-time local sagittal model and emit report/artifacts.
- `scripts/validate_position_aware_balance.py`
  - Orchestrate nominal and height-variant validation runs plus summary output.
- `tests/test_sagittal_position_aware_balance_controller.py`
  - Dedicated controller tests including sign, damping, and mutual-exclusion checks.
- `tests/test_sagittal_balance_state.py`
  - Frame/sign/state-construction tests, including nonzero-yaw initial-heading checks.

### Documentation / Output Paths

- `outputs/sagittal_position_aware_balance/`
  - Validation outputs.
- `outputs/sagittal_position_aware_balance/sysid/`
  - Identification data and model report.
- `outputs/sagittal_position_aware_balance/validation/`
  - Validation summaries and comparisons.

---

### Task 1: Cleanup Verification Snapshot

**Files:**
- Check: `wheeled_biped/controllers/sagittal_wheel_balance_controller.py`
- Check: `scripts/simulate_hierarchical_controller.py`
- Check: `configs/controllers/hierarchical_vmc_lqr_v3.yaml`
- Test: `tests/test_sagittal_wheel_position_containment.py` *(rewritten cleanup/absence test only)*
- Test: `tests/test_e0d_phase_aware_position_containment.py` *(rewritten cleanup/absence test only)*

- [ ] **Step 1: Run rewritten cleanup/absence regression tests**

```bash
pytest tests/test_sagittal_wheel_position_containment.py tests/test_e0d_phase_aware_position_containment.py -q
```

Expected: PASS, confirming the old position-containment experiments are absent from runtime, old telemetry no longer drives control, and only cleanup/report-preservation behavior remains under test.

- [ ] **Step 2: Re-run baseline component and workflow tests**

```bash
pytest tests/test_balance_core_components.py -q
pytest tests/test_balance_core_validation_workflow.py -q
```

Expected: PASS, confirming cleanup did not break baseline balance-core behavior.

- [ ] **Step 3: Run baseline validation durations**

```bash
python scripts/validate_balance_core.py --single-duration 1000 --output-dir outputs/balance_core_position_aware_precheck_1000
python scripts/validate_balance_core.py --single-duration 5000 --output-dir outputs/balance_core_position_aware_precheck_5000
```

Expected: both pass before any new controller work starts.

**Stop gate:** If any cleanup regression or baseline validation fails, stop here and do not begin sagittal state helpers.

- [ ] **Step 4: Commit cleanup checkpoint**

```bash
git add wheeled_biped/controllers/sagittal_wheel_balance_controller.py scripts/simulate_hierarchical_controller.py configs/controllers/hierarchical_vmc_lqr_v3.yaml tests/test_sagittal_wheel_position_containment.py tests/test_e0d_phase_aware_position_containment.py tests/test_balance_core_components.py tests/test_balance_core_mode_isolation.py
git commit -m "refactor: remove failed position containment runtime"
```

---

### Task 2: Sagittal State and Frame Audit

**Files:**
- Create: `wheeled_biped/controllers/sagittal_balance_state.py`
- Test: `tests/test_sagittal_balance_state.py`
- Modify: `scripts/simulate_hierarchical_controller.py`

- [ ] **Step 1: Write failing state-construction test for initial-heading frame displacement**

```python
from wheeled_biped.controllers.sagittal_balance_state import project_sagittal_displacement


def test_project_sagittal_displacement_uses_initial_heading_frame():
    origin_xy = (0.0, 0.0)
    sagittal_axis_xy = (0.0, 1.0)
    current_xy = (0.2, 0.5)

    displacement = project_sagittal_displacement(
        origin_xy=origin_xy,
        sagittal_axis_xy=sagittal_axis_xy,
        current_xy=current_xy,
    )

    assert displacement == 0.5
```

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_sagittal_balance_state.py::test_project_sagittal_displacement_uses_initial_heading_frame -q
```

Expected: FAIL with import or function-missing error.

- [ ] **Step 3: Write failing sign test for sagittal velocity projection**

```python
from wheeled_biped.controllers.sagittal_balance_state import project_sagittal_velocity


def test_project_sagittal_velocity_matches_initial_heading_axis():
    sagittal_axis_xy = (0.0, 1.0)
    velocity_xy = (0.3, -0.4)

    velocity = project_sagittal_velocity(
        sagittal_axis_xy=sagittal_axis_xy,
        velocity_xy=velocity_xy,
    )

    assert velocity == -0.4
```

- [ ] **Step 4: Verify RED**

```bash
pytest tests/test_sagittal_balance_state.py::test_project_sagittal_velocity_matches_initial_heading_axis -q
```

Expected: FAIL.

- [ ] **Step 5: Write failing state-bundle test**

```python
from wheeled_biped.controllers.sagittal_balance_state import build_sagittal_balance_state


def test_build_sagittal_balance_state_orders_required_terms():
    state = build_sagittal_balance_state(
        sagittal_position_error=0.1,
        sagittal_velocity=-0.2,
        pitch_x=0.03,
        pitch_rate_x=-0.04,
        wheel_velocity_mean=1.5,
    )

    assert tuple(state) == (0.1, -0.2, 0.03, -0.04, 1.5)
```

- [ ] **Step 6: Verify RED**

```bash
pytest tests/test_sagittal_balance_state.py::test_build_sagittal_balance_state_orders_required_terms -q
```

Expected: FAIL.

- [ ] **Step 7: Write minimal implementation**

```python
import jax.numpy as jnp


def project_sagittal_displacement(origin_xy, sagittal_axis_xy, current_xy):
    dx = current_xy[0] - origin_xy[0]
    dy = current_xy[1] - origin_xy[1]
    return dx * sagittal_axis_xy[0] + dy * sagittal_axis_xy[1]


def project_sagittal_velocity(sagittal_axis_xy, velocity_xy):
    return velocity_xy[0] * sagittal_axis_xy[0] + velocity_xy[1] * sagittal_axis_xy[1]


def build_sagittal_balance_state(
    sagittal_position_error,
    sagittal_velocity,
    pitch_x,
    pitch_rate_x,
    wheel_velocity_mean,
):
    return jnp.array([
        sagittal_position_error,
        sagittal_velocity,
        pitch_x,
        pitch_rate_x,
        wheel_velocity_mean,
    ])
```

- [ ] **Step 8: Verify GREEN**

```bash
pytest tests/test_sagittal_balance_state.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add wheeled_biped/controllers/sagittal_balance_state.py tests/test_sagittal_balance_state.py
git commit -m "feat: add sagittal balance state helpers"
```

- [ ] **Step 10: Add nonzero-yaw initial-heading frame test**

```python
def test_project_sagittal_displacement_remains_correct_with_nonzero_yaw():
    import math
    yaw_rad = math.radians(30)
    sagittal_axis_xy = (math.sin(yaw_rad), math.cos(yaw_rad))
    current_xy = (0.1, 0.1732)
    origin_xy = (0.0, 0.0)

    displacement = project_sagittal_displacement(
        origin_xy=origin_xy,
        sagittal_axis_xy=sagittal_axis_xy,
        current_xy=current_xy,
    )

    assert abs(displacement - 0.2) < 1e-6
```

- [ ] **Step 11: Verify GREEN**

```bash
pytest tests/test_sagittal_balance_state.py -q
```

Expected: PASS.

**Stop gate:** If state-construction or frame tests fail, stop here and do not begin sysid data collection.

---

### Task 3: Closed-Loop System Identification Data Collection

**Files:**
- Create: `scripts/collect_sagittal_balance_sysid_data.py`
- Modify: `scripts/simulate_hierarchical_controller.py`
- Test: `tests/test_balance_core_validation_workflow.py`

- [ ] **Step 1: Write failing test for output path naming**

```python
from pathlib import Path
from scripts.collect_sagittal_balance_sysid_data import resolve_sysid_output_dir


def test_resolve_sysid_output_dir_uses_position_aware_namespace():
    path = resolve_sysid_output_dir(Path("outputs"))
    assert path == Path("outputs/sagittal_position_aware_balance/sysid")
```

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_balance_core_validation_workflow.py::test_resolve_sysid_output_dir_uses_position_aware_namespace -q
```

Expected: FAIL.

- [ ] **Step 3: Write failing test for metadata summary row**

```python
from scripts.collect_sagittal_balance_sysid_data import build_sysid_run_metadata


def test_build_sysid_run_metadata_marks_closed_loop_collection():
    metadata = build_sysid_run_metadata(
        scenario="nominal",
        duration_steps=5000,
        controller_mode="balance-core",
    )

    assert metadata["collection_mode"] == "closed_loop"
    assert metadata["controller_mode"] == "balance-core"
    assert metadata["duration_steps"] == 5000
```

- [ ] **Step 4: Verify RED**

```bash
pytest tests/test_balance_core_validation_workflow.py::test_build_sysid_run_metadata_marks_closed_loop_collection -q
```

Expected: FAIL.

- [ ] **Step 5: Write minimal collector implementation**

```python
from pathlib import Path


def resolve_sysid_output_dir(root: Path) -> Path:
    return root / "sagittal_position_aware_balance" / "sysid"


def build_sysid_run_metadata(scenario, duration_steps, controller_mode):
    return {
        "collection_mode": "closed_loop",
        "scenario": scenario,
        "duration_steps": duration_steps,
        "controller_mode": controller_mode,
    }
```

- [ ] **Step 6: Verify GREEN**

```bash
pytest tests/test_balance_core_validation_workflow.py::test_resolve_sysid_output_dir_uses_position_aware_namespace tests/test_balance_core_validation_workflow.py::test_build_sysid_run_metadata_marks_closed_loop_collection -q
```

Expected: PASS.

- [ ] **Step 7: Add collector CLI path**

```bash
python scripts/collect_sagittal_balance_sysid_data.py --scenario nominal --steps 1000 --output-root outputs
```

Expected: writes metadata and collected closed-loop telemetry under `outputs/sagittal_position_aware_balance/sysid`.

- [ ] **Step 8: Commit**

```bash
git add scripts/collect_sagittal_balance_sysid_data.py tests/test_balance_core_validation_workflow.py
git commit -m "feat: add closed-loop sagittal sysid collection scaffold"
```

**Stop gate:** If sysid collection fails or produces no valid trajectory data, stop here and do not begin dynamics identification. Collect better data before retrying.

---

### Task 4: Local Dynamics Identification

**Files:**
- Create: `scripts/identify_sagittal_balance_dynamics.py`
- Test: `tests/test_sagittal_position_aware_balance_controller.py`

- [ ] **Step 1: Write failing test for identified model schema**

```python
from scripts.identify_sagittal_balance_dynamics import build_identified_model_payload


def test_build_identified_model_payload_includes_state_space_keys():
    payload = build_identified_model_payload(
        A=[[1, 0], [0, 1]],
        B=[[0], [1]],
        state_names=["pos", "vel"],
        input_name="wheel_torque",
    )

    assert sorted(payload.keys()) == ["A", "B", "input_name", "state_names"]
```

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_sagittal_position_aware_balance_controller.py::test_build_identified_model_payload_includes_state_space_keys -q
```

Expected: FAIL.

- [ ] **Step 3: Write failing test for explicit model-quality gate**

```python
from scripts.identify_sagittal_balance_dynamics import model_is_usable


def test_model_is_usable_requires_all_quality_gates():
    assert model_is_usable(
        one_step_r2=0.85,
        rollout_r2=0.65,
        residual_mean_abs=0.05,
        sign_response_ok=True,
        nominal_fit_ok=True,
        height_variant_fit_ok=True,
    ) is True

    assert model_is_usable(
        one_step_r2=0.79,
        rollout_r2=0.65,
        residual_mean_abs=0.05,
        sign_response_ok=True,
        nominal_fit_ok=True,
        height_variant_fit_ok=True,
    ) is False
```

- [ ] **Step 4: Verify RED**

```bash
pytest tests/test_sagittal_position_aware_balance_controller.py::test_model_is_usable_requires_all_quality_gates -q
```

Expected: FAIL.

- [ ] **Step 5: Write minimal implementation**

```python
def build_identified_model_payload(A, B, state_names, input_name):
    return {
        "A": A,
        "B": B,
        "state_names": state_names,
        "input_name": input_name,
    }


def model_is_usable(
    one_step_r2,
    rollout_r2,
    residual_mean_abs,
    sign_response_ok,
    nominal_fit_ok,
    height_variant_fit_ok,
):
    return (
        one_step_r2 >= 0.80
        and rollout_r2 >= 0.60
        and residual_mean_abs <= 0.10
        and sign_response_ok is True
        and nominal_fit_ok is True
        and height_variant_fit_ok is True
    )
```

- [ ] **Step 6: Verify GREEN**

```bash
pytest tests/test_sagittal_position_aware_balance_controller.py::test_build_identified_model_payload_includes_state_space_keys tests/test_sagittal_position_aware_balance_controller.py::test_model_is_usable_requires_all_quality_gates -q
```

Expected: PASS.

- [ ] **Step 7: Run identification script on collected data**

```bash
python scripts/identify_sagittal_balance_dynamics.py --input outputs/sagittal_position_aware_balance/sysid --output outputs/sagittal_position_aware_balance/sysid/identified_model.json
```

Expected: writes identified model plus fit metrics only if all gates pass.

Required gates before controller design:
- one-step prediction quality threshold passed
- short-horizon rollout quality threshold passed
- residual sanity check passed
- correct qualitative response sign passed
- usable fit on nominal and, if available, ±5 cm data passed

If any gate fails, report:
- `model_identification_failed`

Do not design LQR/state-feedback from an unusable model.

- [ ] **Step 8: Commit**

```bash
git add scripts/identify_sagittal_balance_dynamics.py tests/test_sagittal_position_aware_balance_controller.py
git commit -m "feat: add sagittal dynamics identification scaffold"
```

**Stop gate:** If identification fails any quality gate, stop here, report `model_identification_failed`, and do not begin controller design.

---

### Task 5: Controller Design

**Files:**
- Create: `wheeled_biped/controllers/sagittal_position_aware_balance_controller.py`
- Test: `tests/test_sagittal_position_aware_balance_controller.py`

- [ ] **Step 1: Write failing test for wheel-only output ownership**

```python
import jax.numpy as jnp
from wheeled_biped.controllers.sagittal_position_aware_balance_controller import SagittalPositionAwareBalanceController


def test_position_aware_controller_outputs_only_on_wheels():
    controller = SagittalPositionAwareBalanceController(
        gain_vector=jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        max_tau_wheel=5.0,
    )

    tau, diagnostics = controller.compute_from_state(
        jnp.array([0.2, -0.1, 0.03, -0.02, 1.5])
    )

    assert tau[4] != 0.0 or tau[9] != 0.0
    assert all(tau[i] == 0.0 for i in [0, 1, 2, 3, 5, 6, 7, 8])
```

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_sagittal_position_aware_balance_controller.py::test_position_aware_controller_outputs_only_on_wheels -q
```

Expected: FAIL.

- [ ] **Step 3: Write failing saturation test**

```python
import jax.numpy as jnp
from wheeled_biped.controllers.sagittal_position_aware_balance_controller import SagittalPositionAwareBalanceController


def test_position_aware_controller_clips_to_wheel_limit():
    controller = SagittalPositionAwareBalanceController(
        gain_vector=jnp.array([100.0, 100.0, 100.0, 100.0, 100.0]),
        max_tau_wheel=3.0,
    )

    tau, diagnostics = controller.compute_from_state(
        jnp.array([1.0, 1.0, 1.0, 1.0, 10.0])
    )

    assert abs(tau[4]) <= 3.0
    assert abs(tau[9]) <= 3.0
    assert diagnostics["saturated"] is True
```

- [ ] **Step 4: Verify RED**

```bash
pytest tests/test_sagittal_position_aware_balance_controller.py::test_position_aware_controller_clips_to_wheel_limit -q
```

Expected: FAIL.

- [ ] **Step 5: Write failing sign tests**

```python
import jax.numpy as jnp
from wheeled_biped.controllers.sagittal_position_aware_balance_controller import SagittalPositionAwareBalanceController


def test_positive_position_error_produces_corrective_tendency_toward_reference():
    controller = SagittalPositionAwareBalanceController(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0]), 5.0)
    tau, _ = controller.compute_from_state(jnp.array([0.2, 0.0, 0.0, 0.0, 0.0]))
    assert tau[4] < 0.0 and tau[9] < 0.0


def test_negative_position_error_produces_corrective_tendency_toward_reference():
    controller = SagittalPositionAwareBalanceController(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0]), 5.0)
    tau, _ = controller.compute_from_state(jnp.array([-0.2, 0.0, 0.0, 0.0, 0.0]))
    assert tau[4] > 0.0 and tau[9] > 0.0


def test_positive_velocity_away_from_reference_is_damped():
    controller = SagittalPositionAwareBalanceController(jnp.array([0.0, 1.0, 0.0, 0.0, 0.0]), 5.0)
    tau, _ = controller.compute_from_state(jnp.array([0.0, 0.3, 0.0, 0.0, 0.0]))
    assert tau[4] < 0.0 and tau[9] < 0.0


def test_pitch_and_pitch_rate_signs_are_restoring_and_damping():
    controller = SagittalPositionAwareBalanceController(jnp.array([0.0, 0.0, 1.0, 1.0, 0.0]), 5.0)
    tau_pitch, _ = controller.compute_from_state(jnp.array([0.0, 0.0, 0.1, 0.0, 0.0]))
    tau_rate, _ = controller.compute_from_state(jnp.array([0.0, 0.0, 0.0, 0.1, 0.0]))
    assert tau_pitch[4] < 0.0 and tau_pitch[9] < 0.0
    assert tau_rate[4] < 0.0 and tau_rate[9] < 0.0


def test_wheel_velocity_mean_correction_damps_wheel_runaway():
    controller = SagittalPositionAwareBalanceController(jnp.array([0.0, 0.0, 0.0, 0.0, 1.0]), 5.0)
    tau, _ = controller.compute_from_state(jnp.array([0.0, 0.0, 0.0, 0.0, 2.0]))
    assert tau[4] < 0.0 and tau[9] < 0.0
```

- [ ] **Step 6: Verify RED**

```bash
pytest tests/test_sagittal_position_aware_balance_controller.py -q
```

Expected: FAIL before implementation.

- [ ] **Step 7: Write minimal implementation**

```python
import jax.numpy as jnp


class SagittalPositionAwareBalanceController:
    def __init__(self, gain_vector, max_tau_wheel):
        self.gain_vector = gain_vector
        self.max_tau_wheel = max_tau_wheel

    def compute_from_state(self, state):
        tau_cmd = -jnp.dot(self.gain_vector, state)
        tau_cmd = jnp.clip(tau_cmd, -self.max_tau_wheel, self.max_tau_wheel)
        tau = jnp.zeros(10)
        tau = tau.at[4].set(tau_cmd)
        tau = tau.at[9].set(tau_cmd)
        diagnostics = {
            "tau_wheel_raw": float(-jnp.dot(self.gain_vector, state)),
            "tau_wheel_clipped": float(tau_cmd),
            "saturated": bool(abs(float(tau_cmd)) >= self.max_tau_wheel),
        }
        return tau, diagnostics
```

- [ ] **Step 8: Verify GREEN**

```bash
pytest tests/test_sagittal_position_aware_balance_controller.py::test_position_aware_controller_outputs_only_on_wheels tests/test_sagittal_position_aware_balance_controller.py::test_position_aware_controller_clips_to_wheel_limit -q
```

Expected: PASS.

- [ ] **Step 9: Add smoothness/rate-limit test**

```python
import jax.numpy as jnp
from wheeled_biped.controllers.sagittal_position_aware_balance_controller import clip_wheel_torque_rate


def test_clip_wheel_torque_rate_limits_delta_per_step():
    clipped = clip_wheel_torque_rate(prev_tau=1.0, next_tau=5.0, max_delta=0.5)
    assert clipped == 1.5
```

- [ ] **Step 10: Verify RED, implement, verify GREEN**

```bash
pytest tests/test_sagittal_position_aware_balance_controller.py::test_clip_wheel_torque_rate_limits_delta_per_step -q
```

Expected sequence: FAIL, implement helper, PASS.

- [ ] **Step 11: Commit**

```bash
git add wheeled_biped/controllers/sagittal_position_aware_balance_controller.py tests/test_sagittal_position_aware_balance_controller.py
git commit -m "feat: add sagittal position-aware wheel controller scaffold"
```

**Stop gate:** If any sign, ownership, or saturation test fails after implementation, stop here and do not begin integration.

---

### Task 6: Controller Integration

**Files:**
- Modify: `scripts/simulate_hierarchical_controller.py`
- Modify: `scripts/validate_balance_core.py`
- Modify: `tests/test_balance_core_mode_isolation.py`

- [ ] **Step 1: Write failing CLI test for controller selection**

```python
from types import SimpleNamespace
from scripts.simulate_hierarchical_controller import is_balance_core_mode


def test_position_aware_sagittal_controller_requires_balance_core_mode():
    args = SimpleNamespace(controller_mode="balance-core", sagittal_controller="position-aware")
    assert is_balance_core_mode(args) is True
```

- [ ] **Step 2: Verify RED if selection field missing**

```bash
pytest tests/test_balance_core_mode_isolation.py::test_position_aware_sagittal_controller_requires_balance_core_mode -q
```

Expected: FAIL or missing-field behavior.

- [ ] **Step 3: Add explicit CLI flag**

```python
parser.add_argument(
    "--sagittal-controller",
    type=str,
    default="baseline",
    choices=["baseline", "position-aware"],
    help="Select sagittal wheel controller implementation",
)
```

- [ ] **Step 4: Add failing routing test**

```python
from scripts.simulate_hierarchical_controller import resolve_sagittal_controller_name


def test_resolve_sagittal_controller_name_returns_position_aware_when_requested():
    assert resolve_sagittal_controller_name("position-aware") == "position-aware"
```

- [ ] **Step 5: Add failing mutual-exclusion test**

```python
from scripts.simulate_hierarchical_controller import resolve_active_sagittal_controller_set


def test_baseline_and_position_aware_controllers_are_mutually_exclusive():
    assert resolve_active_sagittal_controller_set("baseline") == {"baseline"}
    assert resolve_active_sagittal_controller_set("position-aware") == {"position-aware"}
```

- [ ] **Step 6: Verify RED, implement, verify GREEN**

```bash
pytest tests/test_balance_core_mode_isolation.py::test_resolve_sagittal_controller_name_returns_position_aware_when_requested tests/test_balance_core_mode_isolation.py::test_baseline_and_position_aware_controllers_are_mutually_exclusive -q
```

Expected: FAIL, then PASS.

- [ ] **Step 7: Keep baseline as default**

```python
if args.sagittal_controller == "baseline":
    sagittal_controller = SagittalWheelBalanceController(...)
else:
    sagittal_controller = SagittalPositionAwareBalanceController(...)
```

Requirement:
- in baseline mode, `SagittalWheelBalanceController` active and `SagittalPositionAwareBalanceController` inactive
- in position-aware mode, `SagittalPositionAwareBalanceController` active and `SagittalWheelBalanceController` inactive
- they must never both contribute torque simultaneously

- [ ] **Step 8: Verify integration tests**

```bash
pytest tests/test_balance_core_mode_isolation.py -q
pytest tests/test_balance_core_validation_workflow.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add scripts/simulate_hierarchical_controller.py scripts/validate_balance_core.py tests/test_balance_core_mode_isolation.py tests/test_balance_core_validation_workflow.py
git commit -m "feat: add selectable sagittal controller integration"
```

**Stop gate:** If mutual exclusion or routing tests fail, stop here and do not begin validation runs.

---

### Task 7: Validation Script and Output Reporting

**Files:**
- Create: `scripts/validate_position_aware_balance.py`
- Test: `tests/test_balance_core_validation_workflow.py`

- [ ] **Step 1: Write failing output-directory test**

```python
from pathlib import Path
from scripts.validate_position_aware_balance import resolve_validation_output_dir


def test_resolve_validation_output_dir_uses_position_aware_namespace():
    assert resolve_validation_output_dir(Path("outputs")) == Path("outputs/sagittal_position_aware_balance/validation")
```

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_balance_core_validation_workflow.py::test_resolve_validation_output_dir_uses_position_aware_namespace -q
```

Expected: FAIL.

- [ ] **Step 3: Write failing comparison-row test**

```python
from scripts.validate_position_aware_balance import build_comparison_row


def test_build_comparison_row_records_baseline_and_candidate_drift():
    row = build_comparison_row(
        scenario="nominal_5000",
        baseline_max_drift_m=35.22,
        candidate_max_drift_m=0.48,
    )

    assert row["scenario"] == "nominal_5000"
    assert row["baseline_max_drift_m"] == 35.22
    assert row["candidate_max_drift_m"] == 0.48
```

- [ ] **Step 4: Verify RED**

```bash
pytest tests/test_balance_core_validation_workflow.py::test_build_comparison_row_records_baseline_and_candidate_drift -q
```

Expected: FAIL.

- [ ] **Step 5: Write minimal implementation**

```python
from pathlib import Path


def resolve_validation_output_dir(root: Path) -> Path:
    return root / "sagittal_position_aware_balance" / "validation"


def build_comparison_row(scenario, baseline_max_drift_m, candidate_max_drift_m):
    return {
        "scenario": scenario,
        "baseline_max_drift_m": baseline_max_drift_m,
        "candidate_max_drift_m": candidate_max_drift_m,
    }
```

- [ ] **Step 6: Verify GREEN**

```bash
pytest tests/test_balance_core_validation_workflow.py::test_resolve_validation_output_dir_uses_position_aware_namespace tests/test_balance_core_validation_workflow.py::test_build_comparison_row_records_baseline_and_candidate_drift -q
```

Expected: PASS.

- [ ] **Step 7: Run validation sequence**

```bash
python scripts/validate_position_aware_balance.py --sagittal-controller position-aware --output-root outputs
```

Expected runs:
- nominal 1000
- nominal 5000
- nominal 10000 if 5000 passes
- high_5cm 500
- low_5cm 500

- [ ] **Step 8: Commit**

```bash
git add scripts/validate_position_aware_balance.py tests/test_balance_core_validation_workflow.py
git commit -m "feat: add position-aware balance validation workflow"
```

**Stop gate:** If the first integration smoke test or validation report generation fails, stop here and do not proceed to any promotion/default-switch work.

Required validation report outcome:
- baseline reference recorded at 35.22 m / 5000 steps
- minimum acceptable improvement gate checked at <= 17.6 m
- E0b comparison at 15.98 m (historical context only, not an active mode)
- target gate checked at <= 5.0 m
- preferred gate checked at <= 0.50 m max drift and <= 0.20 m final drift
- if preferred target is not reached, report the best stable tradeoff and do not claim full position hold

---

### Task 8: Acceptance and Rollback Checks

**Files:**
- Modify: `tests/test_balance_core_mode_isolation.py`
- Modify: `tests/test_balance_core_components.py`

- [ ] **Step 1: Write failing WBC-off regression test for new controller path**

```python
from pathlib import Path


def test_position_aware_path_does_not_introduce_wbc_runtime_hook():
    content = Path("scripts/simulate_hierarchical_controller.py").read_text(encoding="utf-8")
    assert "tau_wbc + tau_position_aware" not in content
```

- [ ] **Step 2: Verify RED if needed**

```bash
pytest tests/test_balance_core_mode_isolation.py::test_position_aware_path_does_not_introduce_wbc_runtime_hook -q
```

Expected: FAIL only if forbidden coupling appears.

- [ ] **Step 3: Write failing ownership regression test**

```python
from wheeled_biped.controllers.sagittal_position_aware_balance_controller import SagittalPositionAwareBalanceController
import jax.numpy as jnp


def test_position_aware_controller_preserves_wheel_only_ownership():
    controller = SagittalPositionAwareBalanceController(jnp.ones(5), 4.0)
    tau, _ = controller.compute_from_state(jnp.ones(5))
    assert all(tau[i] == 0.0 for i in [0, 1, 2, 3, 5, 6, 7, 8])
```

- [ ] **Step 4: Verify RED/green cycle**

```bash
pytest tests/test_balance_core_mode_isolation.py::test_position_aware_path_does_not_introduce_wbc_runtime_hook tests/test_sagittal_position_aware_balance_controller.py::test_position_aware_controller_preserves_wheel_only_ownership -q
```

Expected: PASS after implementation.

- [ ] **Step 5: Final validation suite**

```bash
pytest tests/test_sagittal_balance_state.py tests/test_sagittal_position_aware_balance_controller.py tests/test_balance_core_components.py tests/test_balance_core_mode_isolation.py tests/test_balance_core_validation_workflow.py -q
python scripts/validate_balance_core.py --single-duration 1000 --output-dir outputs/position_aware_post_1000
python scripts/validate_balance_core.py --single-duration 5000 --output-dir outputs/position_aware_post_5000
```

Expected: all pass before any promotion to default.

- [ ] **Step 6: Rollback procedure documentation**

```text
Rollback = run with --sagittal-controller baseline, keep WBC off, keep four-source stack unchanged, disable position-aware path by default.
```

- [ ] **Step 7: Commit**

```bash
git add tests/test_balance_core_mode_isolation.py tests/test_sagittal_position_aware_balance_controller.py
git commit -m "test: lock in position-aware safety and rollback invariants"
```

**Overall acceptance gate:** If the position-aware controller does not beat the minimum 17.6 m drift gate on nominal 5000 steps, stop here, report the best result, and do not switch from baseline as default.

---

## Commands Summary

### Targeted tests

```bash
pytest tests/test_sagittal_wheel_position_containment.py tests/test_e0d_phase_aware_position_containment.py -q
pytest tests/test_balance_core_components.py -q
pytest tests/test_balance_core_mode_isolation.py -q
pytest tests/test_balance_core_validation_workflow.py -q
pytest tests/test_sagittal_balance_state.py -q
pytest tests/test_sagittal_position_aware_balance_controller.py -q
```

Note: `test_sagittal_wheel_position_containment.py` and `test_e0d_phase_aware_position_containment.py` are rewritten cleanup/absence tests that verify removed runtime behavior and preserved reports only. They do not assert active position-containment behavior.

### Validation commands

```bash
python scripts/validate_balance_core.py --single-duration 1000 --output-dir outputs/balance_core_position_aware_precheck_1000
python scripts/validate_balance_core.py --single-duration 5000 --output-dir outputs/balance_core_position_aware_precheck_5000
python scripts/collect_sagittal_balance_sysid_data.py --scenario nominal --steps 1000 --output-root outputs
python scripts/identify_sagittal_balance_dynamics.py --input outputs/sagittal_position_aware_balance/sysid --output outputs/sagittal_position_aware_balance/sysid/identified_model.json
python scripts/validate_position_aware_balance.py --sagittal-controller position-aware --output-root outputs
```

---

## Acceptance Criteria

Implementation is complete only if:

- baseline cleanup tests still pass
- new sagittal state/frame tests pass
- identified model passes minimum fit gate
- position-aware controller outputs wheel torque only
- wheel torque/rate limits are enforced
- WBC remains off
- ownership violation count remains zero
- baseline controller remains available
- position-aware mode is disabled by default until validated
- nominal 1000 and 5000 step runs are stable
- candidate controller substantially beats 35.22 m drift baseline
- nominal and ±5 cm runs are reported quantitatively

---

## Rollback Plan

If any stage fails:

1. Stop promotion work immediately.
2. Keep `--sagittal-controller baseline` as default.
3. Do not remove `SagittalWheelBalanceController`.
4. Preserve collected sysid and validation outputs for diagnosis.
5. Re-run baseline 1000/5000 validation before the next attempt.

---

## Naming Rules

Allowed production names:

- `SagittalPositionAwareBalanceController`
- `SagittalBalanceState`
- `SagittalBalanceReference`
- `SagittalPositionRegulationConfig`
- `PositionAwareBalanceValidator`

Forbidden production names:

- `E0Controller`
- `Stage2E`
- `temp_position_fix`
- `position_patch`
- `quick_fix`
- `hack_containment`

---

## Self-Review

Plan coverage check:

- cleanup verification covered in Task 1
- frame/state audit covered in Task 2
- closed-loop data collection covered in Task 3
- local dynamics identification covered in Task 4
- controller design covered in Task 5
- controller integration covered in Task 6
- tests covered in Tasks 2, 4, 5, 6, 8
- validation/reporting covered in Task 7
- rollback / no-WBC / no-hidden-torque checks covered in Tasks 1 and 8

Placeholder scan:

- no TBD/TODO placeholders remain
- every task includes explicit files and concrete commands
- code steps include concrete snippets rather than references to unnamed helpers

Type consistency check:

- controller name used consistently as `SagittalPositionAwareBalanceController`
- state helper naming used consistently as `SagittalBalanceState` helpers in `sagittal_balance_state.py`
- output namespace used consistently as `outputs/sagittal_position_aware_balance/`
