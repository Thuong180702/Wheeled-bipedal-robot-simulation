# Physical Standing Height Envelope Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable static physical-feasibility utility and a coarse-to-fine static standing-height search that finds `physical_min_height` and `physical_max_height` from static/kinematic feasibility only, serializes the selected extrema, reloads them, and revalidates them before any dynamic validation.

**Architecture:** Put all wheel-contact support geometry, CoM projection, perpendicular support-offset, static gate logic, and robot-CoM convention handling in `wheeled_biped/validation/physical_standing_height_envelope.py`. Keep `scripts/search_physical_standing_height_envelope.py` as an orchestration layer only: verify joint conventions from the model, generate calibrated symmetric standing candidates, call the shared utility, write artifacts, select extrema from static feasibility only, then reload and revalidate them. Static support feasibility must use whole-robot CoM, or exactly the same CoM definition already used by the existing balance-core / centroidal telemetry; do not silently substitute torso-only CoM unless equivalence is explicitly verified and documented. Dynamic Step E / Step C are explicitly out of scope for this plan.

**Tech Stack:** Python 3, MuJoCo, NumPy, pandas, dataclasses, pytest, JSON/CSV/Markdown artifact writing

---

## File map

- Create: `wheeled_biped/validation/physical_standing_height_envelope.py`
  - Reusable contact-geometry extraction, support-segment math, static feasibility gates, and JSON-serializable result objects.
- Create: `scripts/search_physical_standing_height_envelope.py`
  - Coarse-to-fine search driver, joint convention resolution, root_z calibration, candidate evaluation, extrema selection, serialization, and static revalidation.
- Create: `tests/test_physical_standing_height_envelope.py`
  - Geometry, morphology independence, root-z-only rejection, CoM-convention verification, calibrated-contact extraction, selection independence from dynamic failure, artifact serialization/reload, and script-uses-shared-utility tests.
- Create: `docs/validation/physical_standing_height_envelope_definition.md`
  - Static physical-envelope definition and corrected support-geometry interpretation.
- Create: `docs/validation/physical_standing_height_envelope_validation.md`
  - Search result summary generated from artifacts after the static run.
- Read during implementation, do not modify:
  - `assets/robot/wheeled_biped_real.xml`
  - `scripts/validate_balance_core_height_variants_v2.py`
  - `tests/test_balance_core_height_variant_setup.py`
  - `tests/test_balance_core_height_variant_setup_gates.py`
  - `outputs/balance_core_true_height_variants/true_height_variant_setup_report.json`

## Invariants this plan must preserve

- `physical_min_height` and `physical_max_height` are selected from static feasibility only.
- Dynamic Step E / Step C success or failure must not affect physical extrema selection.
- Root-z-only candidates are rejected.
- Static support feasibility uses whole-robot CoM, or exactly the same CoM definition used by the existing balance-core / centroidal telemetry; torso-only CoM must not be used unless explicitly verified equivalent and documented.
- Shared support geometry lives only in `wheeled_biped/validation/physical_standing_height_envelope.py`.
- No controller behavior changes.
- No torque logic changes.
- No WBC changes.
- Do not modify `scripts/simulate_hierarchical_controller.py` or `scripts/run_step_c_height_recovery.py` in this implementation phase.

### Task 1: Build support-segment geometry primitives

**Files:**
- Create: `wheeled_biped/validation/physical_standing_height_envelope.py`
- Test: `tests/test_physical_standing_height_envelope.py`

- [ ] **Step 1: Write the failing geometry tests**

```python
import math

from wheeled_biped.validation.physical_standing_height_envelope import (
    PhysicalStandingThresholds,
    build_support_segment_geometry,
)


def test_left_right_wheel_segment_contains_centered_com():
    geometry = build_support_segment_geometry(
        left_wheel_contact_xy=(-0.10, 0.00),
        right_wheel_contact_xy=(0.10, 0.00),
        com_xy=(0.00, 0.01),
        thresholds=PhysicalStandingThresholds(
            projection_tolerance=1e-6,
            preferred_sagittal_offset_m=0.01,
            max_sagittal_offset_m=0.02,
        ),
    )
    assert geometry.com_projection_inside_wheel_segment is True
    assert math.isclose(geometry.com_projection_fraction_on_wheel_segment, 0.5, abs_tol=1e-9)
    assert math.isclose(geometry.com_lateral_offset_from_support_center_m, 0.0, abs_tol=1e-9)
    assert math.isclose(geometry.com_sagittal_offset_from_support_center_m, 0.01, abs_tol=1e-9)


def test_front_back_wheel_segment_uses_same_projection_method():
    geometry = build_support_segment_geometry(
        left_wheel_contact_xy=(0.00, -0.12),
        right_wheel_contact_xy=(0.00, 0.12),
        com_xy=(0.01, 0.00),
        thresholds=PhysicalStandingThresholds(),
    )
    assert geometry.com_projection_inside_wheel_segment is True
    assert math.isclose(geometry.com_projection_fraction_on_wheel_segment, 0.5, abs_tol=1e-9)
    assert math.isclose(abs(geometry.com_sagittal_offset_from_support_center_m), 0.01, abs_tol=1e-9)


def test_projection_outside_segment_fails_containment():
    geometry = build_support_segment_geometry(
        left_wheel_contact_xy=(-0.10, 0.00),
        right_wheel_contact_xy=(0.10, 0.00),
        com_xy=(0.25, 0.00),
        thresholds=PhysicalStandingThresholds(projection_tolerance=1e-6),
    )
    assert geometry.com_projection_inside_wheel_segment is False
    assert geometry.com_projection_fraction_on_wheel_segment > 1.0


def test_degenerate_wheel_segment_is_rejected():
    geometry = build_support_segment_geometry(
        left_wheel_contact_xy=(0.00, 0.00),
        right_wheel_contact_xy=(1e-9, 1e-9),
        com_xy=(0.00, 0.00),
        thresholds=PhysicalStandingThresholds(),
    )
    assert geometry.valid is False
    assert "degenerate_wheel_support_segment" in geometry.rejection_reasons
```

- [ ] **Step 2: Run the geometry tests to verify they fail**

Run: `pytest tests/test_physical_standing_height_envelope.py -k "segment or geometry" -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'wheeled_biped.validation.physical_standing_height_envelope'`

- [ ] **Step 3: Write the minimal support-geometry implementation**

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math


@dataclass(frozen=True)
class PhysicalStandingThresholds:
    projection_tolerance: float = 1e-6
    preferred_sagittal_offset_m: float = 0.01
    max_sagittal_offset_m: float = 0.02
    max_pitch_abs_rad: float = 0.10
    max_roll_abs_rad: float = 0.05
    max_yaw_abs_rad: float = 0.10
    min_joint_limit_margin_rad: float = 0.05
    degenerate_segment_length_m: float = 1e-6


@dataclass
class SupportSegmentGeometry:
    valid: bool
    rejection_reasons: list[str] = field(default_factory=list)
    support_center_xy: tuple[float, float] = (0.0, 0.0)
    wheel_line_direction_xy: tuple[float, float] = (0.0, 0.0)
    support_error_direction_xy: tuple[float, float] = (0.0, 0.0)
    com_projection_fraction_on_wheel_segment: float = 0.0
    com_projection_inside_wheel_segment: bool = False
    com_lateral_offset_from_support_center_m: float = 0.0
    com_sagittal_offset_from_support_center_m: float = 0.0
    segment_length_m: float = 0.0
    min_endpoint_margin_m: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def build_support_segment_geometry(
    *,
    left_wheel_contact_xy: tuple[float, float],
    right_wheel_contact_xy: tuple[float, float],
    com_xy: tuple[float, float],
    thresholds: PhysicalStandingThresholds,
) -> SupportSegmentGeometry:
    lx, ly = left_wheel_contact_xy
    rx, ry = right_wheel_contact_xy
    cx, cy = com_xy

    dx = rx - lx
    dy = ry - ly
    segment_length = math.hypot(dx, dy)
    if segment_length <= thresholds.degenerate_segment_length_m:
        return SupportSegmentGeometry(
            valid=False,
            rejection_reasons=["degenerate_wheel_support_segment", "support_geometry_invalid"],
            segment_length_m=segment_length,
        )

    ux = dx / segment_length
    uy = dy / segment_length
    px = -uy
    py = ux

    center_x = 0.5 * (lx + rx)
    center_y = 0.5 * (ly + ry)

    vx = cx - lx
    vy = cy - ly
    fraction = (vx * ux + vy * uy) / segment_length
    inside = -thresholds.projection_tolerance <= fraction <= 1.0 + thresholds.projection_tolerance

    rel_x = cx - center_x
    rel_y = cy - center_y
    lateral_offset = rel_x * ux + rel_y * uy
    sagittal_offset = rel_x * px + rel_y * py
    endpoint_margin = 0.5 * segment_length - abs(lateral_offset)

    rejection_reasons: list[str] = []
    if not inside:
        rejection_reasons.append("projection_outside_wheel_segment")
    if abs(sagittal_offset) > thresholds.max_sagittal_offset_m:
        rejection_reasons.append("sagittal_support_offset_too_large")

    return SupportSegmentGeometry(
        valid=len(rejection_reasons) == 0,
        rejection_reasons=rejection_reasons,
        support_center_xy=(center_x, center_y),
        wheel_line_direction_xy=(ux, uy),
        support_error_direction_xy=(px, py),
        com_projection_fraction_on_wheel_segment=fraction,
        com_projection_inside_wheel_segment=inside,
        com_lateral_offset_from_support_center_m=lateral_offset,
        com_sagittal_offset_from_support_center_m=sagittal_offset,
        segment_length_m=segment_length,
        min_endpoint_margin_m=endpoint_margin,
    )
```

- [ ] **Step 4: Run the geometry tests to verify they pass**

Run: `pytest tests/test_physical_standing_height_envelope.py -k "segment or geometry" -v`

Expected: PASS for the four new geometry tests.

- [ ] **Step 5: Report progress and stop at the checkpoint if complete**

Report:
- changed files,
- tests run and results,
- whether the task is complete.

Stop at this checkpoint if the task is complete. Do not run `git commit` unless explicitly requested by the user.

### Task 2: Add MuJoCo contact extraction and static-feasibility evaluation

**Files:**
- Modify: `wheeled_biped/validation/physical_standing_height_envelope.py`
- Test: `tests/test_physical_standing_height_envelope.py`

- [ ] **Step 1: Write the failing contact and feasibility tests**

```python
import mujoco
import numpy as np

from wheeled_biped.utils.config import get_model_path
from wheeled_biped.validation.physical_standing_height_envelope import (
    ROBOT_COM_CONVENTION,
    PhysicalStandingThresholds,
    compute_robot_com_xy,
    evaluate_static_standing_pose,
    extract_wheel_floor_contact_points,
)


def test_extract_wheel_contacts_from_calibrated_static_pose():
    from scripts.search_physical_standing_height_envelope import calibrate_root_z_from_wheel_geometry

    model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    calibrate_root_z_from_wheel_geometry(model, data, target_contact_depth_m=-5e-4)
    mujoco.mj_forward(model, data)

    contact_points = extract_wheel_floor_contact_points(model, data)
    assert contact_points.left_wheel_contact_xy is not None
    assert contact_points.right_wheel_contact_xy is not None
    assert contact_points.left_wheel_contact is True
    assert contact_points.right_wheel_contact is True
    assert contact_points.non_wheel_floor_contact_count == 0


def test_compute_robot_com_xy_uses_documented_project_convention():
    model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    com_xy = compute_robot_com_xy(model, data)
    assert len(com_xy) == 2
    assert np.isfinite(com_xy[0])
    assert np.isfinite(com_xy[1])
    assert ROBOT_COM_CONVENTION in {"whole_robot", "balance_core_telemetry"}


def test_missing_wheel_contact_geometry_is_reported():
    result = evaluate_static_standing_pose(
        left_wheel_contact_xy=None,
        right_wheel_contact_xy=None,
        com_xy=(0.0, 0.0),
        pitch_x_rad=0.0,
        roll_y_rad=0.0,
        yaw_z_rad=0.0,
        left_wheel_contact=False,
        right_wheel_contact=False,
        non_wheel_floor_contact_count=0,
        joint_limit_margin_rad=0.2,
        thresholds=PhysicalStandingThresholds(),
        candidate_source="unit_test",
        candidate_is_root_z_only=False,
    )
    assert result.static_feasible is False
    assert "missing_wheel_floor_contact_geometry" in result.rejection_reasons


def test_root_z_only_candidates_are_rejected():
    result = evaluate_static_standing_pose(
        left_wheel_contact_xy=(-0.1, 0.0),
        right_wheel_contact_xy=(0.1, 0.0),
        com_xy=(0.0, 0.0),
        pitch_x_rad=0.0,
        roll_y_rad=0.0,
        yaw_z_rad=0.0,
        left_wheel_contact=True,
        right_wheel_contact=True,
        non_wheel_floor_contact_count=0,
        joint_limit_margin_rad=0.2,
        thresholds=PhysicalStandingThresholds(),
        candidate_source="unit_test",
        candidate_is_root_z_only=True,
    )
    assert result.static_feasible is False
    assert "root_z_only_candidate_not_allowed" in result.rejection_reasons


def test_large_pitch_roll_yaw_are_reported():
    result = evaluate_static_standing_pose(
        left_wheel_contact_xy=(-0.1, 0.0),
        right_wheel_contact_xy=(0.1, 0.0),
        com_xy=(0.0, 0.0),
        pitch_x_rad=0.2,
        roll_y_rad=0.06,
        yaw_z_rad=0.2,
        left_wheel_contact=True,
        right_wheel_contact=True,
        non_wheel_floor_contact_count=0,
        joint_limit_margin_rad=0.2,
        thresholds=PhysicalStandingThresholds(),
        candidate_source="unit_test",
        candidate_is_root_z_only=False,
    )
    assert "pitch_roll_yaw_out_of_bounds" in result.rejection_reasons
```

- [ ] **Step 2: Run the contact and feasibility tests to verify they fail**

Run: `pytest tests/test_physical_standing_height_envelope.py -k "contact or static or root_z_only or pitch_roll_yaw" -v`

Expected: FAIL with `ImportError` or `AttributeError` for missing symbols.

- [ ] **Step 3: Implement contact extraction and static-feasibility evaluation**

```python
from dataclasses import asdict, dataclass, field
import numpy as np
import mujoco


ROBOT_COM_CONVENTION = "whole_robot"


@dataclass
class WheelContactPoints:
    left_wheel_contact_xy: tuple[float, float] | None
    right_wheel_contact_xy: tuple[float, float] | None
    left_wheel_contact: bool
    right_wheel_contact: bool
    non_wheel_floor_contact_count: int
    wheel_contact_force_z_n: float
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StaticStandingFeasibilityResult:
    setup_valid: bool
    static_feasible: bool
    rejection_reasons: list[str]
    candidate_source: str
    candidate_is_root_z_only: bool
    support_geometry: dict
    contact_metrics: dict
    posture_metrics: dict
    joint_limit_margin_rad: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_robot_com_xy(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float]:
    # Use whole-robot CoM unless an existing shared balance-core / centroidal telemetry helper is adopted instead.
    masses = np.asarray(model.body_mass, dtype=float)
    positions_xy = np.asarray(data.xipos[:, :2], dtype=float)
    total_mass = float(np.sum(masses))
    weighted_xy = np.sum(masses[:, None] * positions_xy, axis=0)
    return tuple((weighted_xy / total_mass).tolist())


def extract_wheel_floor_contact_points(model: mujoco.MjModel, data: mujoco.MjData) -> WheelContactPoints:
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    left_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    right_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")

    left_points: list[tuple[float, float]] = []
    right_points: list[tuple[float, float]] = []
    total_fz = 0.0
    non_wheel_floor_contacts = 0

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if floor_geom_id not in (geom1, geom2):
            continue
        if left_geom_id in (geom1, geom2):
            left_points.append((float(contact.pos[0]), float(contact.pos[1])))
        elif right_geom_id in (geom1, geom2):
            right_points.append((float(contact.pos[0]), float(contact.pos[1])))
        else:
            non_wheel_floor_contacts += 1
            continue
        force_contact = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force_contact)
        frame = np.array(contact.frame).reshape(3, 3)
        force_world = frame.T @ force_contact[:3]
        total_fz += float(force_world[2])

    left_xy = tuple(np.mean(np.array(left_points), axis=0).tolist()) if left_points else None
    right_xy = tuple(np.mean(np.array(right_points), axis=0).tolist()) if right_points else None
    rejection_reasons: list[str] = []
    if left_xy is None or right_xy is None:
        rejection_reasons.append("missing_wheel_floor_contact_geometry")

    return WheelContactPoints(
        left_wheel_contact_xy=left_xy,
        right_wheel_contact_xy=right_xy,
        left_wheel_contact=left_xy is not None,
        right_wheel_contact=right_xy is not None,
        non_wheel_floor_contact_count=non_wheel_floor_contacts,
        wheel_contact_force_z_n=total_fz,
        rejection_reasons=rejection_reasons,
    )


def evaluate_static_standing_pose(
    *,
    left_wheel_contact_xy: tuple[float, float] | None,
    right_wheel_contact_xy: tuple[float, float] | None,
    com_xy: tuple[float, float],
    pitch_x_rad: float,
    roll_y_rad: float,
    yaw_z_rad: float,
    left_wheel_contact: bool,
    right_wheel_contact: bool,
    non_wheel_floor_contact_count: int,
    joint_limit_margin_rad: float,
    thresholds: PhysicalStandingThresholds,
    candidate_source: str,
    candidate_is_root_z_only: bool,
) -> StaticStandingFeasibilityResult:
    rejection_reasons: list[str] = []
    if candidate_is_root_z_only:
        rejection_reasons.append("root_z_only_candidate_not_allowed")
    if not left_wheel_contact or not right_wheel_contact:
        rejection_reasons.append("missing_wheel_floor_contact_geometry")
    if non_wheel_floor_contact_count > 0:
        rejection_reasons.append("non_wheel_floor_contact")
    if joint_limit_margin_rad < thresholds.min_joint_limit_margin_rad:
        rejection_reasons.append("joint_limit_margin_too_small")
    if abs(pitch_x_rad) > thresholds.max_pitch_abs_rad or abs(roll_y_rad) > thresholds.max_roll_abs_rad or abs(yaw_z_rad) > thresholds.max_yaw_abs_rad:
        rejection_reasons.append("pitch_roll_yaw_out_of_bounds")

    if left_wheel_contact_xy is None or right_wheel_contact_xy is None:
        geometry = SupportSegmentGeometry(
            valid=False,
            rejection_reasons=["missing_wheel_floor_contact_geometry", "support_geometry_invalid"],
        )
    else:
        geometry = build_support_segment_geometry(
            left_wheel_contact_xy=left_wheel_contact_xy,
            right_wheel_contact_xy=right_wheel_contact_xy,
            com_xy=com_xy,
            thresholds=thresholds,
        )
        rejection_reasons.extend(geometry.rejection_reasons)

    deduped_reasons = list(dict.fromkeys(rejection_reasons))
    static_feasible = len(deduped_reasons) == 0 and geometry.valid
    return StaticStandingFeasibilityResult(
        setup_valid=static_feasible,
        static_feasible=static_feasible,
        rejection_reasons=deduped_reasons,
        candidate_source=candidate_source,
        candidate_is_root_z_only=candidate_is_root_z_only,
        support_geometry=geometry.to_dict(),
        contact_metrics={
            "left_wheel_contact": left_wheel_contact,
            "right_wheel_contact": right_wheel_contact,
            "non_wheel_floor_contact_count": non_wheel_floor_contact_count,
        },
        posture_metrics={
            "pitch_x_rad": pitch_x_rad,
            "roll_y_rad": roll_y_rad,
            "yaw_z_rad": yaw_z_rad,
        },
        joint_limit_margin_rad=joint_limit_margin_rad,
    )
```

- [ ] **Step 4: Run the contact and feasibility tests to verify they pass**

Run: `pytest tests/test_physical_standing_height_envelope.py -k "contact or static or root_z_only or pitch_roll_yaw or robot_com" -v`

Expected: PASS for the new contact, CoM-convention, and feasibility tests.

- [ ] **Step 5: Report progress and stop at the checkpoint if complete**

Report:
- changed files,
- tests run and results,
- whether the task is complete.

Stop at this checkpoint if the task is complete. Do not run `git commit` unless explicitly requested by the user.

### Task 3: Add search helpers, joint convention resolution, and root_z calibration

**Files:**
- Create: `scripts/search_physical_standing_height_envelope.py`
- Test: `tests/test_physical_standing_height_envelope.py`

- [ ] **Step 1: Write the failing search-helper tests**

```python
from scripts.search_physical_standing_height_envelope import (
    calibrate_root_z_from_wheel_geometry,
    resolve_standing_joint_addresses,
    select_physical_extrema,
)


def test_select_physical_extrema_ignores_dynamic_failure_fields():
    candidates = [
        {"achieved_com_z_m": 0.38, "static_feasible": True, "dynamic_verdict": "FAIL"},
        {"achieved_com_z_m": 0.40, "static_feasible": True, "dynamic_verdict": "PASS"},
        {"achieved_com_z_m": 0.43, "static_feasible": True, "dynamic_verdict": "FAIL"},
    ]
    selected = select_physical_extrema(candidates)
    assert selected["physical_min_height"]["achieved_com_z_m"] == 0.38
    assert selected["physical_max_height"]["achieved_com_z_m"] == 0.43


def test_resolve_joint_addresses_reads_model_names_not_hardcoded_signs():
    import mujoco
    from wheeled_biped.utils.config import get_model_path

    model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    addresses = resolve_standing_joint_addresses(model)
    assert addresses["l_hip_pitch"] != addresses["r_hip_pitch"]
    assert addresses["l_knee"] != addresses["r_knee"]
    assert "l_hip_pitch_axis" in addresses
    assert "r_hip_pitch_axis" in addresses


def test_search_script_imports_shared_utility():
    source = open("scripts/search_physical_standing_height_envelope.py", "r", encoding="utf-8").read()
    assert "from wheeled_biped.validation.physical_standing_height_envelope import" in source
    assert "def build_support_segment_geometry" not in source
```

- [ ] **Step 2: Run the search-helper tests to verify they fail**

Run: `pytest tests/test_physical_standing_height_envelope.py -k "extrema or addresses or imports_shared_utility" -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.search_physical_standing_height_envelope'`

- [ ] **Step 3: Implement the search helpers and calibration path**

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import mujoco
import numpy as np

from wheeled_biped.utils.config import get_model_path
from wheeled_biped.validation.physical_standing_height_envelope import (
    ROBOT_COM_CONVENTION,
    PhysicalStandingThresholds,
    compute_robot_com_xy,
    evaluate_static_standing_pose,
    extract_wheel_floor_contact_points,
)


@dataclass(frozen=True)
class SearchConfig:
    coarse_target_step_m: float = 0.005
    fine_target_step_m: float = 0.001
    initial_target_span_m: float = 0.06
    outward_expand_step_m: float = 0.01
    max_outward_expansions: int = 6
    hip_pitch_grid_steps: int = 17
    knee_grid_steps: int = 17
    target_contact_depth_m: float = -5e-4


def resolve_standing_joint_addresses(model: mujoco.MjModel) -> dict[str, int | tuple[float, float, float]]:
    joint_names = ["l_hip_pitch", "l_knee", "r_hip_pitch", "r_knee", "l_hip_yaw", "r_hip_yaw", "l_hip_roll", "r_hip_roll"]
    mapping: dict[str, int | tuple[float, float, float]] = {}
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        mapping[name] = int(model.jnt_qposadr[joint_id])
        mapping[f"{name}_axis"] = tuple(float(v) for v in model.jnt_axis[joint_id])
    return mapping


def calibrate_root_z_from_wheel_geometry(model: mujoco.MjModel, data: mujoco.MjData, *, target_contact_depth_m: float) -> float:
    left_geom = model.geom("l_wheel_collision")
    right_geom = model.geom("r_wheel_collision")
    left_body_id = model.body("l_wheel_link").id
    right_body_id = model.body("r_wheel_link").id
    left_radius = float(model.geom_size[left_geom.id][0])
    right_radius = float(model.geom_size[right_geom.id][0])

    root_z_min = 0.25
    root_z_max = 0.75
    for _ in range(30):
        data.qpos[2] = 0.5 * (root_z_min + root_z_max)
        mujoco.mj_forward(model, data)
        left_bottom = float(data.xpos[left_body_id, 2]) - left_radius
        right_bottom = float(data.xpos[right_body_id, 2]) - right_radius
        average_bottom = 0.5 * (left_bottom + right_bottom)
        if average_bottom > target_contact_depth_m:
            root_z_max = float(data.qpos[2])
        else:
            root_z_min = float(data.qpos[2])
        if abs(average_bottom - target_contact_depth_m) < 1e-5:
            break
    mujoco.mj_forward(model, data)
    return float(data.qpos[2])


def select_physical_extrema(candidates: list[dict]) -> dict[str, dict | None]:
    valid_candidates = [candidate for candidate in candidates if candidate.get("static_feasible") is True]
    if not valid_candidates:
        return {"physical_min_height": None, "physical_max_height": None}
    return {
        "physical_min_height": min(valid_candidates, key=lambda item: item["achieved_com_z_m"]),
        "physical_max_height": max(valid_candidates, key=lambda item: item["achieved_com_z_m"]),
    }
```

- [ ] **Step 4: Run the search-helper tests to verify they pass**

Run: `pytest tests/test_physical_standing_height_envelope.py -k "extrema or addresses or imports_shared_utility" -v`

Expected: PASS for the new search-helper tests.

- [ ] **Step 5: Report progress and stop at the checkpoint if complete**

Report:
- changed files,
- tests run and results,
- whether the task is complete.

Stop at this checkpoint if the task is complete. Do not run `git commit` unless explicitly requested by the user.

### Task 4: Implement candidate serialization, artifact writing, and true static revalidation

**Files:**
- Modify: `scripts/search_physical_standing_height_envelope.py`
- Test: `tests/test_physical_standing_height_envelope.py`

- [ ] **Step 1: Write the failing serialization and revalidation tests**

```python
import json
from pathlib import Path

from scripts.search_physical_standing_height_envelope import (
    revalidate_saved_extrema,
    serialize_candidate_setup,
    write_candidate_artifacts,
)


def test_rejection_reasons_are_preserved_in_artifacts(tmp_path: Path):
    valid_candidates = [{"candidate_id": "ok", "static_feasible": True, "rejection_reasons": []}]
    invalid_candidates = [{"candidate_id": "bad", "static_feasible": False, "rejection_reasons": ["projection_outside_wheel_segment"]}]
    write_candidate_artifacts(tmp_path, valid_candidates, invalid_candidates, {"physical_min_height": None, "physical_max_height": None}, {"status": "PHYSICAL_ENVELOPE_INCONCLUSIVE", "results": []}, [])
    payload = json.loads((tmp_path / "physical_height_invalid_candidates.json").read_text(encoding="utf-8"))
    assert payload[0]["rejection_reasons"] == ["projection_outside_wheel_segment"]


def test_serialize_candidate_setup_keeps_required_schema_fields():
    candidate = {
        "requested_target_com_z_m": 0.401,
        "achieved_com_z_m": 0.401,
        "calibrated_root_z_m": 0.534,
        "hip_pitch_ref": 0.93,
        "knee_ref": 1.75,
        "support_geometry": {"com_projection_inside_wheel_segment": True},
        "contact_metrics": {"left_wheel_contact": True, "right_wheel_contact": True},
        "joint_limit_margin_rad": 0.2,
        "candidate_source": "coarse_to_fine_search",
        "candidate_is_root_z_only": False,
        "joint_qpos": {"l_hip_pitch": 0.93, "l_knee": 1.75, "r_hip_pitch": 0.93, "r_knee": 1.75},
        "rejection_reasons": [],
    }
    payload = serialize_candidate_setup(candidate)
    assert payload["candidate_is_root_z_only"] is False
    assert payload["requested_target_com_z_m"] == 0.401
    assert payload["joint_qpos"]["l_hip_pitch"] == 0.93


def test_revalidate_saved_extrema_recomputes_static_feasibility(tmp_path: Path):
    candidate = {
        "requested_target_com_z_m": 0.401,
        "achieved_com_z_m": 0.401,
        "calibrated_root_z_m": 0.534,
        "hip_pitch_ref": 0.926052,
        "knee_ref": 1.748364,
        "joint_qpos": {"l_hip_roll": 0.0, "l_hip_yaw": 0.0, "l_hip_pitch": 0.926052, "l_knee": 1.748364, "r_hip_roll": 0.0, "r_hip_yaw": 0.0, "r_hip_pitch": 0.926052, "r_knee": 1.748364},
        "candidate_source": "coarse_to_fine_search",
        "candidate_is_root_z_only": False,
        "support_geometry": {},
        "contact_metrics": {},
        "joint_limit_margin_rad": 0.2,
        "rejection_reasons": [],
    }
    setup_path = tmp_path / "physical_min_height_setup.json"
    setup_path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
    results = revalidate_saved_extrema([setup_path])
    assert results[0]["setup_path"].endswith("physical_min_height_setup.json")
    assert "static_feasible" in results[0]
```

- [ ] **Step 2: Run the serialization and revalidation tests to verify they fail**

Run: `pytest tests/test_physical_standing_height_envelope.py -k "preserved_in_artifacts or schema_fields or recomputes_static_feasibility" -v`

Expected: FAIL with `ImportError` or `AttributeError` for missing serialization/revalidation helpers.

- [ ] **Step 3: Implement setup serialization and artifact writing**

```python
def serialize_candidate_setup(candidate: dict) -> dict:
    return {
        "requested_target_com_z_m": candidate["requested_target_com_z_m"],
        "achieved_com_z_m": candidate["achieved_com_z_m"],
        "calibrated_root_z_m": candidate["calibrated_root_z_m"],
        "hip_pitch_ref": candidate["hip_pitch_ref"],
        "knee_ref": candidate["knee_ref"],
        "joint_qpos": candidate["joint_qpos"],
        "support_geometry": candidate["support_geometry"],
        "contact_metrics": candidate["contact_metrics"],
        "joint_limit_margin_rad": candidate["joint_limit_margin_rad"],
        "candidate_source": candidate["candidate_source"],
        "candidate_is_root_z_only": False,
        "rejection_reasons": candidate.get("rejection_reasons", []),
    }


def write_candidate_artifacts(
    output_dir: Path,
    valid_candidates: list[dict],
    invalid_candidates: list[dict],
    extrema: dict,
    static_revalidation: dict,
    search_grid_rows: list[dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "physical_height_search_grid.csv").write_text(pd.DataFrame(search_grid_rows).to_csv(index=False), encoding="utf-8")
    (output_dir / "physical_height_valid_candidates.json").write_text(json.dumps(valid_candidates, indent=2), encoding="utf-8")
    (output_dir / "physical_height_invalid_candidates.json").write_text(json.dumps(invalid_candidates, indent=2), encoding="utf-8")
    (output_dir / "physical_height_envelope_summary.json").write_text(json.dumps({
        "physical_min_height": extrema["physical_min_height"],
        "physical_max_height": extrema["physical_max_height"],
        "static_revalidation": static_revalidation,
        "search_grid_count": len(search_grid_rows),
    }, indent=2), encoding="utf-8")
    if extrema["physical_min_height"] is not None:
        (output_dir / "physical_min_height_setup.json").write_text(json.dumps(serialize_candidate_setup(extrema["physical_min_height"]), indent=2), encoding="utf-8")
    if extrema["physical_max_height"] is not None:
        (output_dir / "physical_max_height_setup.json").write_text(json.dumps(serialize_candidate_setup(extrema["physical_max_height"]), indent=2), encoding="utf-8")
```

- [ ] **Step 4: Implement true static revalidation from saved setup JSON**

```python
def revalidate_saved_extrema(setup_paths: list[Path]) -> list[dict]:
    model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    addresses = resolve_standing_joint_addresses(model)
    thresholds = PhysicalStandingThresholds()
    results: list[dict] = []

    for setup_path in setup_paths:
        payload = json.loads(setup_path.read_text(encoding="utf-8"))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.qpos[2] = payload["calibrated_root_z_m"]
        data.qpos[int(addresses["l_hip_roll"])] = payload["joint_qpos"]["l_hip_roll"]
        data.qpos[int(addresses["l_hip_yaw"])] = payload["joint_qpos"]["l_hip_yaw"]
        data.qpos[int(addresses["l_hip_pitch"])] = payload["joint_qpos"]["l_hip_pitch"]
        data.qpos[int(addresses["l_knee"])] = payload["joint_qpos"]["l_knee"]
        data.qpos[int(addresses["r_hip_roll"])] = payload["joint_qpos"]["r_hip_roll"]
        data.qpos[int(addresses["r_hip_yaw"])] = payload["joint_qpos"]["r_hip_yaw"]
        data.qpos[int(addresses["r_hip_pitch"])] = payload["joint_qpos"]["r_hip_pitch"]
        data.qpos[int(addresses["r_knee"])] = payload["joint_qpos"]["r_knee"]
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        mujoco.mj_forward(model, data)

        contact_points = extract_wheel_floor_contact_points(model, data)
        com_xy = compute_robot_com_xy(model, data)
        result = evaluate_static_standing_pose(
            left_wheel_contact_xy=contact_points.left_wheel_contact_xy,
            right_wheel_contact_xy=contact_points.right_wheel_contact_xy,
            com_xy=com_xy,
            pitch_x_rad=0.0,
            roll_y_rad=0.0,
            yaw_z_rad=0.0,
            left_wheel_contact=contact_points.left_wheel_contact,
            right_wheel_contact=contact_points.right_wheel_contact,
            non_wheel_floor_contact_count=contact_points.non_wheel_floor_contact_count,
            joint_limit_margin_rad=payload["joint_limit_margin_rad"],
            thresholds=thresholds,
            candidate_source=payload["candidate_source"],
            candidate_is_root_z_only=payload["candidate_is_root_z_only"],
        )
        results.append({
            "setup_path": str(setup_path).replace("\\", "/"),
            "static_feasible": result.static_feasible,
            "rejection_reasons": result.rejection_reasons,
        })

    return results
```

- [ ] **Step 5: Run the serialization and revalidation tests to verify they pass**

Run: `pytest tests/test_physical_standing_height_envelope.py -k "preserved_in_artifacts or schema_fields or recomputes_static_feasibility" -v`

Expected: PASS for the new serialization and revalidation tests.

- [ ] **Step 6: Report progress and stop at the checkpoint if complete**

Report:
- changed files,
- tests run and results,
- whether the task is complete.

Stop at this checkpoint if the task is complete. Do not run `git commit` unless explicitly requested by the user.

### Task 5: Run the actual static envelope search and generate documentation

**Files:**
- Modify: `scripts/search_physical_standing_height_envelope.py`
- Create: `docs/validation/physical_standing_height_envelope_definition.md`
- Create: `docs/validation/physical_standing_height_envelope_validation.md`

- [ ] **Step 1: Write the definition document before the search run**

```markdown
# Physical Standing Height Envelope Definition

## Physical / kinematic envelope

A standing pose is in the physical envelope only if it is statically feasible based on actual wheel-floor contact geometry, support-segment projection, perpendicular support offset, posture bounds, joint-limit margin, and calibrated non-root-z-only pose provenance.

## Controller-stable envelope

The controller-stable envelope is a later dynamic subset of the physical envelope. Dynamic failure must not shrink the physical envelope.
```

- [ ] **Step 2: Implement the full search loop with coarse-to-fine refinement**

```python
# coarse pass
coarse_targets = np.arange(min_target, max_target + 0.5 * coarse_step, coarse_step)
# if a valid candidate exists at the boundary, expand outward by outward_expand_step_m
# fine pass around the last-valid / first-invalid band with fine_target_step_m
# choose the best candidate at each achieved height by:
#   1) |com_sagittal_offset_from_support_center_m|
#   2) min_endpoint_margin_m descending
#   3) joint_limit_margin_rad descending
#   4) orientation error ascending
```

- [ ] **Step 3: Run the search script and verify artifacts are created**

Run: `python scripts/search_physical_standing_height_envelope.py`

Expected:
- exit code `0` if static extrema are found and revalidated,
- or exit code `1` with `PHYSICAL_ENVELOPE_INCONCLUSIVE` clearly written in the summary if no trustworthy extrema are found.

Expected artifact paths:
- `outputs/physical_standing_height_envelope_search/physical_height_search_grid.csv`
- `outputs/physical_standing_height_envelope_search/physical_height_valid_candidates.json`
- `outputs/physical_standing_height_envelope_search/physical_height_invalid_candidates.json`
- `outputs/physical_standing_height_envelope_search/physical_height_envelope_summary.json`
- `outputs/physical_standing_height_envelope_search/physical_height_envelope_report.md`
- `outputs/physical_standing_height_envelope_search/physical_min_height_setup.json`
- `outputs/physical_standing_height_envelope_search/physical_max_height_setup.json`
- `outputs/physical_standing_height_envelope_search/static_physical_extrema_validation.json`

- [ ] **Step 4: Write the validation document from the generated artifacts**

```markdown
# Physical Standing Height Envelope Validation

## Search coverage

- Coarse target step: 0.005 m
- Fine target step: 0.001 m
- Initial nominal-centered span: 0.06 m
- Outward expansion increment: 0.01 m

## Selected physical extrema

- physical_min_height: <fill from summary artifact>
- physical_max_height: <fill from summary artifact>

## Static revalidation

- Status: <PASS or PHYSICAL_ENVELOPE_INCONCLUSIVE>

## Dynamic status

Dynamic Step E / Step C validation not yet performed in Part 3.
```

- [ ] **Step 5: Report progress and stop at the checkpoint if complete**

Report:
- changed files,
- tests run and results,
- whether the task is complete.

Stop at this checkpoint if the task is complete. Do not run `git commit` unless explicitly requested by the user.

### Task 6: Run the final regression suite and stop before dynamic validation

**Files:**
- Modify: `tests/test_physical_standing_height_envelope.py`
- Read: `outputs/physical_standing_height_envelope_search/physical_height_envelope_summary.json`

- [ ] **Step 1: Add the final guard-rail tests**

```python
import json
from pathlib import Path


def test_selected_extrema_come_from_static_feasibility_only(tmp_path: Path):
    summary = {
        "physical_min_height": {"candidate_id": "low", "static_feasible": True, "dynamic_verdict": "FAIL"},
        "physical_max_height": {"candidate_id": "high", "static_feasible": True, "dynamic_verdict": "FAIL"},
    }
    assert summary["physical_min_height"]["candidate_id"] == "low"
    assert summary["physical_max_height"]["candidate_id"] == "high"


def test_validation_summary_has_no_step_e_or_step_c_fields():
    summary_path = Path("outputs/physical_standing_height_envelope_search/physical_height_envelope_summary.json")
    if not summary_path.exists():
        return
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "step_e_verdict" not in payload
    assert "step_c_verdict" not in payload
```

- [ ] **Step 2: Run the new physical-envelope suite**

Run: `pytest tests/test_physical_standing_height_envelope.py -v`

Expected: PASS for geometry, contact extraction, root-z-only rejection, morphology independence, selection independence from dynamic failure, serialization/reload, artifact reason preservation, and shared-utility import coverage.

- [ ] **Step 3: Run the existing standing-variant regression tests**

Run: `pytest tests/test_balance_core_height_variant_setup.py tests/test_balance_core_height_variant_setup_gates.py -v`

Expected: PASS with no regressions to the existing Step B validation path.

- [ ] **Step 4: Verify the stop point explicitly**

Run: `python -c "import json; from pathlib import Path; p = Path('outputs/physical_standing_height_envelope_search/physical_height_envelope_summary.json'); print(json.loads(p.read_text(encoding='utf-8')).get('static_revalidation', {}))"`

Expected:
- summary exists,
- static revalidation status is present,
- no Step E / Step C execution is included.

- [ ] **Step 5: Report progress and stop at the checkpoint if complete**

Report:
- changed files,
- tests run and results,
- whether the task is complete.

Stop at this checkpoint if the task is complete. Do not run `git commit` unless explicitly requested by the user.

## Spec coverage checklist

- Shared reusable utility: covered by Tasks 1-2.
- Actual wheel-floor contact extraction: covered by Task 2.
- Degenerate segment handling: covered by Task 1.
- No hardcoded X/Y semantics in geometry logic: covered by Task 1 tests, including synthetic front/back morphology.
- Root-z-only rejection: covered by Task 2 tests and search metadata.
- Joint convention verification from model/XML: covered by Task 3.
- Explicit root_z calibration method: covered by Task 3.
- Coarse-to-fine strategy with reported limits and resolution: covered by Task 5.
- Static-only extrema selection independent of dynamic failure: covered by Tasks 3, 4, and 6.
- Serialization to `physical_min_height_setup.json` / `physical_max_height_setup.json`: covered by Task 4.
- Static reload/revalidation before dynamic validation: covered by Tasks 4-6.
- Docs for definition and validation: covered by Task 5.
- Explicit stop point before Step E / Step C: covered by Task 6.

## Stop point

Stop immediately after Task 6 completes. At that point, report:
- files created/updated,
- tests run and results,
- `physical_min_height` found,
- `physical_max_height` found,
- whether static revalidation passed,
- artifacts created,
- `controller behavior changed: false`,
- `WBC added/applied: false`.

Do not start Step E or Step C dynamic validation in this implementation cycle.
