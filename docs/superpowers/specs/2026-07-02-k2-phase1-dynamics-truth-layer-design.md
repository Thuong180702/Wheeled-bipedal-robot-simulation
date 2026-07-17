# Phase 1: K2 Dynamics Truth Layer — Design

**Date:** 2026-07-02
**Status:** Approved → Implementation
**Scope:** Diagnostics / infrastructure only — no controller changes

## Purpose

Build a read-only dynamics truth layer that extracts and validates physical quantities from the MuJoCo model for future QP-WBC development. This ensures that any future WBC is based on correct physical quantities (model state, Jacobians, contacts, actuator limits, sign conventions), not heuristic tuning.

## Non-goals

- No QP-WBC implementation
- No controller changes
- No profile changes to `K2_JAX_DEDICATED_DEFAULT_V3`
- No gain/sign/profile tuning
- No promotion

## Architecture

```
wheeled_biped/dynamics/          # NEW package
├── __init__.py                  # Public exports
├── model_inspector.py           # Model index report + state snapshot
├── jacobian_checks.py           # Analytic + FD Jacobian validation
├── contact_inspector.py         # Contact pair/force inspection
└── torque_sign_checks.py        # Per-actuator sign convention probe

scripts/
└── phase1_dynamics_truth_audit.py  # NEW: orchestrates all checks → Markdown report

docs/validation/
└── k2_phase1_dynamics_truth_audit.md  # NEW: generated report

tests/
└── test_phase1_dynamics_truth_layer.py  # NEW: lightweight unit tests
```

## Design decisions

### 1. MuJoCo CPU API only

MJX (`mjx.Model`, `mjx.Data`) does not expose:
- `mj_jac` / `mj_jacBody` / `mj_jacSite` (Jacobian computation)
- `mj_contactForce` (contact force extraction)
- `mj_fullM` (mass matrix)
- Contact struct fields beyond what MJX exposes via `data.contact`

All diagnostics use CPU `mujoco.MjModel` / `mujoco.MjData`. The report documents which quantities are CPU-only vs. also available in MJX.

### 2. No controller involvement

All rollouts for torque sign probes use direct `mj_data.ctrl[:] = value` with zero controller. No profile is loaded, no controller step function is called.

### 3. Reuse existing utilities

- `get_model_path()` from `wheeled_biped.utils.config`
- `get_total_robot_mass()` from `wheeled_biped.controllers.robot_model_utils`
- Existing `test_actuator_signs.py` is referenced but not duplicated

### 4. Jacobian FD validation

- Perturb each actuated joint qpos by ε = 1e-4
- Measure Δx/Δq in world frame
- Compare to analytic Jacobian columns for actuated joints only
- Free-joint columns (0-5 in v) are skipped for FD
- Thresholds: PASS < 1e-3, WARN < 1e-2, FAIL ≥ 1e-2

### 5. Torque sign convention

- Apply ±1.0 Nm to each actuator individually
- Measure resulting qacc sign
- Report as MEASURED (sign consistent), AMBIGUOUS (sign flips unexpectedly), or MISSING
- Left/right mirrored joints are NOT flagged as failures for differing signs

## Module specifications

### model_inspector.py

```python
build_model_index_report(model) -> dict
extract_state_snapshot(model, data) -> dict
```

Returns name/index maps for joints, actuators, bodies, geoms, sites. Includes qpos/qvel/actuator dimensions and actuator ctrlrange/forcerange.

### jacobian_checks.py

```python
compute_task_jacobian(model, data, target_name, target_type) -> dict
finite_difference_jacobian_check(model, data, target_name, target_type) -> dict
```

Supports `target_type` = "body" or "site". Returns analytic Jacobian shape, norms, rank estimate, actuated-column FD errors.

### contact_inspector.py

```python
inspect_contacts(model, data) -> dict
```

Returns contact count, geom pairs, body pairs, positions, normals, and estimated contact forces (via `mj_contactForce`).

### torque_sign_checks.py

```python
torque_sign_probe(model, data, joint_name, actuator_name) -> dict
```

Runs short non-destructive rollout (±1 Nm, 1 step), reports qacc sign per actuator.

## Files NOT touched

- `wheeled_biped/controllers/k2_jax_controller.py`
- `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
- `scripts/run_k2_jax_realtime.py`
- All controller profile definitions
- All config YAML files
- All promotion scripts
- `K2_JAX_DEDICATED_DEFAULT_V3` or any other profile

## Phase 2 readiness criteria

- `READY_FOR_QP_WBC`: model mapping, actuator mapping, body IDs, actuator limits, state snapshot, Jacobian checks, contact inspection, torque sign probes all structurally valid
- `PARTIAL_READY`: all structural checks pass, but key quantities (mass matrix, contact forces) only available via CPU MuJoCo, not integrated into realtime/JAX path
- `NOT_READY`: mapping/sign/Jacobian/contact failures that make QP-WBC unsafe
