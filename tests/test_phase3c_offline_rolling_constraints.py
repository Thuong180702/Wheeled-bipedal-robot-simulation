"""Tests for Phase 3C — Offline Rolling Constraints.

Validates:
  - Rolling module imports
  - Rolling constants build
  - Wheel geometry (radius, qvel indices)
  - Wheel contact classifier
  - Rolling basis vectors
  - Velocity residuals
  - Lateral/forward constraint row building
  - Rolling modes (soft/hard)
  - QP build with rolling constraints
  - Hard constraint regression
  - No controller imports

CPU-only, offline only, no training, no visual mode.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Phase 3C modules must be imported first to register in sys.modules
# before other wheeled_biped imports interact with the editable finder.
import wheeled_biped.wbc.offline_rolling_constraints  # noqa: F401
import wheeled_biped.wbc.phase3c_rolling_qp  # noqa: F401


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def mj_model():
    import mujoco
    from wheeled_biped.utils.config import get_model_path
    return mujoco.MjModel.from_xml_path(str(get_model_path()))


@pytest.fixture(scope="module")
def mj_data(mj_model):
    import mujoco
    data = mujoco.MjData(mj_model)
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, data, 0)
    mujoco.mj_forward(mj_model, data)
    return data


@pytest.fixture(scope="module")
def rolling_constants(mj_model):
    from wheeled_biped.wbc.offline_rolling_constraints import build_wheel_rolling_constants
    return build_wheel_rolling_constants(mj_model)


@pytest.fixture(scope="module")
def qp_constants(mj_model):
    from wheeled_biped.wbc.offline_qp_wbc import build_qp_wbc_constants
    from wheeled_biped.dynamics.jax_mass_matrix import build_mass_matrix_constants
    from wheeled_biped.dynamics.jax_bias_forces import build_bias_force_constants
    from wheeled_biped.dynamics.jax_contact_dynamics import build_contact_dynamics_constants
    from wheeled_biped.dynamics.jax_kinematics import build_kinematic_tree_constants
    from wheeled_biped.wbc.offline_rolling_constraints import build_wheel_rolling_constants

    mass_c = build_mass_matrix_constants(mj_model)
    bias_c = build_bias_force_constants(mj_model, mass_matrix_constants=mass_c)
    contact_c = build_contact_dynamics_constants(mj_model, kinematics_constants=bias_c)
    qp_c = build_qp_wbc_constants(mj_model, dynamics_constants=bias_c, contact_constants=contact_c)
    kin_c = build_kinematic_tree_constants(mj_model)
    qp_c["_kinematics_constants"] = kin_c
    rolling_c = build_wheel_rolling_constants(mj_model, contact_constants=contact_c)
    qp_c["_rolling_constants"] = rolling_c
    return qp_c


@pytest.fixture(scope="module")
def sample_state(mj_model, mj_data):
    import mujoco
    qpos = mj_data.qpos.copy()
    qvel = np.zeros(mj_model.nv)
    qvel[0] = 0.1
    qvel[10] = 0.5
    qvel[15] = 0.5

    data = mujoco.MjData(mj_model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(mj_model, data)

    from wheeled_biped.dynamics.jax_mass_matrix import build_mass_matrix_constants
    from wheeled_biped.dynamics.jax_bias_forces import build_bias_force_constants
    from wheeled_biped.dynamics.jax_contact_dynamics import build_contact_dynamics_constants

    mass_c = build_mass_matrix_constants(mj_model)
    bias_c = build_bias_force_constants(mj_model, mass_matrix_constants=mass_c)
    contact_c = build_contact_dynamics_constants(mj_model, kinematics_constants=bias_c)

    contacts = _extract_contacts_from_data(mj_model, data, contact_c)
    return qpos, qvel, contacts


def _extract_contacts_from_data(model, data, contact_constants):
    wheel_body_ids = contact_constants["wheel_body_ids"]
    wheel_ids_set = set(int(v) for v in wheel_body_ids.values() if v >= 0)
    contacts = []
    for contact_id in range(data.ncon):
        c = data.contact[contact_id]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        wheel_body = None
        if b1 in wheel_ids_set:
            wheel_body = b1
        elif b2 in wheel_ids_set:
            wheel_body = b2
        if wheel_body is None:
            continue
        pos = np.array(c.pos, dtype=np.float64)
        frame = np.array(c.frame, dtype=np.float64).reshape(3, 3)
        body_xpos = np.array(data.xpos[wheel_body], dtype=np.float64)
        body_xmat = np.array(data.xmat[wheel_body], dtype=np.float64).reshape(3, 3)
        local_point = body_xmat.T @ (pos - body_xpos)
        contacts.append({"body_id": int(wheel_body), "position": pos, "frame": frame,
                          "local_point": local_point, "distance": float(c.dist)})
    return contacts


@pytest.fixture(scope="module")
def snapshot(sample_state, qp_constants):
    from wheeled_biped.wbc.phase3b_cached_stack import prepare_phase3b_snapshot
    qpos, qvel, contacts = sample_state
    return prepare_phase3b_snapshot("test_snapshot", qpos, qvel, contacts, qp_constants)


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_module_imports():
    from wheeled_biped.wbc.offline_rolling_constraints import ROLLING_MODES
    assert len(ROLLING_MODES) == 5
    assert "normal_only" in ROLLING_MODES


def test_rolling_qp_imports():
    from wheeled_biped.wbc.phase3c_rolling_qp import build_phase3c_qp_from_snapshot, solve_phase3c_offline_qp, validate_phase3c_solution
    assert callable(build_phase3c_qp_from_snapshot)


def test_rolling_constants_build(rolling_constants):
    for key in ["l_wheel_body_id", "r_wheel_radius", "l_wheel_qvel_index", "constants_version"]:
        assert key in rolling_constants
    assert rolling_constants["constants_version"] == "phase3c_offline_rolling_constraints"


def test_wheel_radius_finite_positive(rolling_constants):
    assert rolling_constants["l_wheel_radius"] > 0
    assert rolling_constants["r_wheel_radius"] > 0
    assert np.isfinite(rolling_constants["l_wheel_radius"])


def test_wheel_qvel_indices_valid(rolling_constants):
    nv = rolling_constants["nv"]
    assert 0 <= rolling_constants["l_wheel_qvel_index"] < nv
    assert 0 <= rolling_constants["r_wheel_qvel_index"] < nv
    assert rolling_constants["l_wheel_qvel_index"] != rolling_constants["r_wheel_qvel_index"]


def test_wheel_contact_classifier(sample_state, rolling_constants):
    qpos, qvel, contacts = sample_state
    from wheeled_biped.wbc.offline_rolling_constraints import classify_wheel_contacts
    result = classify_wheel_contacts(contacts, rolling_constants)
    assert isinstance(result["left_active"], bool)


def test_rolling_basis_finite(sample_state, rolling_constants):
    qpos, qvel, contacts = sample_state
    from wheeled_biped.wbc.offline_rolling_constraints import compute_wheel_contact_basis
    basis = compute_wheel_contact_basis(qpos, contacts, rolling_constants)
    for side in ["left", "right"]:
        for key in ["normal_world", "t_roll_world", "t_lat_world", "axis_world"]:
            assert np.all(np.isfinite(basis[side][key]))


def test_rolling_basis_orthogonal(sample_state, rolling_constants):
    qpos, qvel, contacts = sample_state
    from wheeled_biped.wbc.offline_rolling_constraints import compute_wheel_contact_basis
    basis = compute_wheel_contact_basis(qpos, contacts, rolling_constants)
    for side in ["left", "right"]:
        n = basis[side]["normal_world"]
        t_roll = basis[side]["t_roll_world"]
        t_lat = basis[side]["t_lat_world"]
        assert abs(np.dot(n, t_roll)) < 0.1
        assert abs(np.dot(n, t_lat)) < 0.1


def test_lateral_slip_finite(sample_state, rolling_constants):
    qpos, qvel, contacts = sample_state
    from wheeled_biped.wbc.offline_rolling_constraints import compute_rolling_velocity_residual
    vel_res = compute_rolling_velocity_residual(qpos, qvel, contacts, rolling_constants)
    assert np.isfinite(vel_res["max_abs_lateral_slip"])


def test_forward_rolling_finite(sample_state, rolling_constants):
    qpos, qvel, contacts = sample_state
    from wheeled_biped.wbc.offline_rolling_constraints import compute_rolling_velocity_residual
    vel_res = compute_rolling_velocity_residual(qpos, qvel, contacts, rolling_constants)
    assert np.isfinite(vel_res["max_abs_forward_rolling_residual"])


def test_lateral_soft_qp_builds(snapshot, qp_constants):
    from wheeled_biped.wbc.phase3c_rolling_qp import build_phase3c_qp_from_snapshot
    qp_mats = build_phase3c_qp_from_snapshot(snapshot, "balanced_default", "lateral_soft", qp_constants)
    assert qp_mats["H"].shape[0] == qp_mats["nz"]
    assert qp_mats["rolling_mode"] == "lateral_soft"


def test_full_rolling_soft_qp_builds(snapshot, qp_constants):
    from wheeled_biped.wbc.phase3c_rolling_qp import build_phase3c_qp_from_snapshot
    qp_mats = build_phase3c_qp_from_snapshot(snapshot, "balanced_default", "full_rolling_soft", qp_constants)
    assert qp_mats["rolling_mode"] == "full_rolling_soft"


def test_lateral_hard_qp_builds(snapshot, qp_constants):
    from wheeled_biped.wbc.phase3c_rolling_qp import build_phase3c_qp_from_snapshot
    qp_mats = build_phase3c_qp_from_snapshot(snapshot, "balanced_default", "lateral_hard", qp_constants)
    assert qp_mats["rolling_mode"] == "lateral_hard"


def test_normal_only_reproduces_phase3b1_gates(snapshot, qp_constants):
    from wheeled_biped.wbc.phase3c_rolling_qp import build_phase3c_qp_from_snapshot, solve_phase3c_offline_qp
    qp_mats = build_phase3c_qp_from_snapshot(snapshot, "balanced_default", "normal_only", qp_constants)
    solution = solve_phase3c_offline_qp(qp_mats, qp_constants)
    assert solution["max_dynamics_residual"] < 1e-5


def test_balanced_lateral_soft_solves(snapshot, qp_constants):
    from wheeled_biped.wbc.phase3c_rolling_qp import build_phase3c_qp_from_snapshot, solve_phase3c_offline_qp
    qp_mats = build_phase3c_qp_from_snapshot(snapshot, "balanced_default", "lateral_soft", qp_constants)
    solution = solve_phase3c_offline_qp(qp_mats, qp_constants)
    assert solution["success"]


def test_balanced_full_rolling_soft_solves(snapshot, qp_constants):
    from wheeled_biped.wbc.phase3c_rolling_qp import build_phase3c_qp_from_snapshot, solve_phase3c_offline_qp
    qp_mats = build_phase3c_qp_from_snapshot(snapshot, "balanced_default", "full_rolling_soft", qp_constants)
    solution = solve_phase3c_offline_qp(qp_mats, qp_constants)
    assert solution["success"]


def test_hard_constraints_pass_all_modes(snapshot, qp_constants):
    from wheeled_biped.wbc.phase3c_rolling_qp import build_phase3c_qp_from_snapshot, solve_phase3c_offline_qp
    for rm in ["normal_only", "lateral_soft", "lateral_hard", "full_rolling_soft", "full_rolling_hard"]:
        qp_mats = build_phase3c_qp_from_snapshot(snapshot, "balanced_default", rm, qp_constants)
        solution = solve_phase3c_offline_qp(qp_mats, qp_constants)
        if solution["success"]:
            assert solution["max_dynamics_residual"] < 1e-5, f"{rm}: dyn={solution['max_dynamics_residual']:.2e}"


def test_rolling_residual_metrics_reported(sample_state, rolling_constants):
    qpos, qvel, contacts = sample_state
    from wheeled_biped.wbc.offline_rolling_constraints import compute_rolling_velocity_residual
    vel_res = compute_rolling_velocity_residual(qpos, qvel, contacts, rolling_constants)
    for side in ["left", "right"]:
        assert "v_lat_slip" in vel_res[side]
        assert "v_roll_residual" in vel_res[side]


def test_no_controller_modules_imported():
    forbidden = ["wheeled_biped.controllers.k2_jax_controller",
                  "wheeled_biped.controllers.sagittal_velocity_damped_balance_controller"]
    for mod_name in forbidden:
        assert mod_name not in sys.modules


def test_no_qp_torque_injection_path():
    import inspect
    from wheeled_biped.wbc import offline_rolling_constraints as orc
    from wheeled_biped.wbc import phase3c_rolling_qp as prq
    for mod in [orc, prq]:
        source = inspect.getsource(mod)
        assert "data.ctrl" not in source
        assert "data.qfrc_applied" not in source


def test_wheel_center_jacobian_shape(sample_state, rolling_constants):
    qpos, qvel, contacts = sample_state
    from wheeled_biped.wbc.offline_rolling_constraints import compute_wheel_center_jacobian
    J_l = compute_wheel_center_jacobian(qpos, "left", rolling_constants)
    J_r = compute_wheel_center_jacobian(qpos, "right", rolling_constants)
    assert J_l.shape == (3, 16)
    assert J_r.shape == (3, 16)
    assert np.all(np.isfinite(J_l))
