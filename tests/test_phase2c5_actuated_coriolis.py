"""Tests for Phase 2C.5 — Actuated Coriolis Coupling / RNEA Compliance Fix.

Validates the free-joint Coriolis acceleration fix that resolves the
actuated bias residual in mixed base-velocity cases.

CPU-only, no GPU, no training, no visual mode.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PASS_TH = 1e-3
WARN_TH = 1e-2


@pytest.fixture(scope="module")
def mj_model():
    import mujoco
    from wheeled_biped.utils.config import get_model_path
    return mujoco.MjModel.from_xml_path(str(get_model_path()))


@pytest.fixture
def mj_data(mj_model):
    import mujoco
    data = mujoco.MjData(mj_model)
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, data, 0)
    mujoco.mj_forward(mj_model, data)
    return data


@pytest.fixture(scope="module")
def constants(mj_model):
    from wheeled_biped.dynamics.jax_bias_forces import build_bias_force_constants
    return build_bias_force_constants(mj_model)


@pytest.fixture
def qpos(mj_data):
    import jax.numpy as jnp
    return jnp.array(mj_data.qpos.copy(), dtype=jnp.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Import and version tests
# ═══════════════════════════════════════════════════════════════════════════

class TestImports:
    def test_all_public_api_importable(self):
        from wheeled_biped.dynamics.jax_bias_forces import (
            build_bias_force_constants,
            extract_jax_bias_arrays,
            jax_bias_forces,
            jax_bias_forces_fk_arrays,
            jax_gravity_forces,
            jax_velocity_bias_forces,
            compare_bias_forces_to_mujoco,
            rnea_body_local,
        )
        assert callable(jax_bias_forces)
        assert callable(jax_gravity_forces)

    def test_constants_version(self, constants):
        assert constants["constants_version"] == "phase2c5_actuated_coriolis"

    def test_no_controller_imports(self):
        """Verify no controller modules are imported in jax_bias_forces."""
        import ast
        src = (PROJECT_ROOT / "wheeled_biped" / "dynamics" / "jax_bias_forces.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        forbidden = ["k2_jax_controller", "sagittal_velocity_damped_balance_controller"]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not any(f in alias.name for f in forbidden), \
                        f"jax_bias_forces.py imports forbidden: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not any(f in node.module for f in forbidden), \
                        f"jax_bias_forces.py imports forbidden: {node.module}"


# ═══════════════════════════════════════════════════════════════════════════
# Gravity tests
# ═══════════════════════════════════════════════════════════════════════════

class TestGravityOnly:
    def test_gravity_keyframe(self, mj_model, mj_data, constants, qpos):
        import mujoco
        from wheeled_biped.dynamics.jax_bias_forces import jax_gravity_forces
        import jax.numpy as jnp

        jax_grav = np.array(jax_gravity_forces(qpos, constants), dtype=np.float64)
        d = mj_data
        d.qpos[:] = np.array(qpos)
        mujoco.mj_forward(mj_model, d)
        cpu_grav = np.array(d.qfrc_bias, dtype=np.float64)
        err = float(np.max(np.abs(jax_grav - cpu_grav)))
        assert err < PASS_TH, f"Gravity error {err:.2e} >= {PASS_TH}"

    def test_gravity_non_identity_orientations(self, mj_model, mj_data, constants):
        from wheeled_biped.dynamics.jax_bias_forces import jax_gravity_forces
        import jax.numpy as jnp
        from scipy.spatial.transform import Rotation
        import mujoco

        qpos_base = mj_data.qpos.copy()
        for rpy, name in [((0,0,0), "id"), ((0,10,0), "p10"), ((0,-10,0), "nm10"),
                           ((10,0,0), "r10"), ((0,0,15), "y15"), ((5,8,12), "comb")]:
            R = Rotation.from_euler('xyz', np.deg2rad(rpy)).as_matrix()
            quat = Rotation.from_matrix(R).as_quat()
            qp = qpos_base.copy()
            qp[3:7] = [quat[3], quat[0], quat[1], quat[2]]
            qp_j = jnp.array(qp, dtype=jnp.float32)
            jax_grav = np.array(jax_gravity_forces(qp_j, constants), dtype=np.float64)
            d2 = mujoco.MjData(mj_model); d2.qpos[:] = qp
            mujoco.mj_forward(mj_model, d2)
            cpu_grav = np.array(d2.qfrc_bias, dtype=np.float64)
            err = float(np.max(np.abs(jax_grav - cpu_grav)))
            assert err < PASS_TH, f"Gravity at {name}: {err:.2e} >= {PASS_TH}"


# ═══════════════════════════════════════════════════════════════════════════
# Pure velocity tests
# ═══════════════════════════════════════════════════════════════════════════

def _run_comparison(mj_model, constants, qpos_np, qvel_np):
    from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
    import jax.numpy as jnp
    import mujoco

    qpos_jax = jnp.array(qpos_np, dtype=jnp.float32)
    qvel_jax = jnp.array(qvel_np, dtype=jnp.float32)
    jax_full = np.array(jax_bias_forces(qpos_jax, qvel_jax, constants), dtype=np.float64)
    d = mujoco.MjData(mj_model); d.qpos[:] = qpos_np; d.qvel[:] = qvel_np
    mujoco.mj_forward(mj_model, d)
    cpu_full = np.array(d.qfrc_bias, dtype=np.float64)
    return float(np.max(np.abs(jax_full - cpu_full))), \
           float(np.max(np.abs(jax_full[6:16] - cpu_full[6:16]))), \
           float(np.max(np.abs(jax_full[0:6] - cpu_full[0:6])))


class TestPureVelocities:
    def test_pure_base_vx(self, mj_model, mj_data, constants):
        qvel = np.zeros(mj_model.nv); qvel[0] = 1.0
        _, act_err, _ = _run_comparison(mj_model, constants, mj_data.qpos.copy(), qvel)
        assert act_err < PASS_TH, f"act_err={act_err:.2e}"

    def test_pure_base_wz(self, mj_model, mj_data, constants):
        qvel = np.zeros(mj_model.nv); qvel[5] = 1.0
        _, act_err, _ = _run_comparison(mj_model, constants, mj_data.qpos.copy(), qvel)
        assert act_err < PASS_TH, f"act_err={act_err:.2e}"

    def test_pure_actuated_single_dof_all(self, mj_model, mj_data, constants):
        nv = mj_model.nv
        for j in range(6, 16):
            qvel = np.zeros(nv); qvel[j] = 1.0 if j not in [10, 15] else 5.0
            _, act_err, _ = _run_comparison(mj_model, constants, mj_data.qpos.copy(), qvel)
            assert act_err < PASS_TH, f"DOF {j}: act_err={act_err:.2e}"

    def test_symmetric_wheels(self, mj_model, mj_data, constants):
        nv = mj_model.nv
        qvel = np.zeros(nv); qvel[10] = 5.0; qvel[15] = 5.0
        _, act_err, _ = _run_comparison(mj_model, constants, mj_data.qpos.copy(), qvel)
        assert act_err < PASS_TH, f"act_err={act_err:.2e}"


# ═══════════════════════════════════════════════════════════════════════════
# Mixed velocity tests — THE CRITICAL ONES
# ═══════════════════════════════════════════════════════════════════════════

class TestMixedBaseVelocities:
    """These tests were FAIL in Phases 2C, 2C.1-2C.4. Phase 2C.5 fixes them."""

    def test_wz_plus_vx(self, mj_model, mj_data, constants):
        nv = mj_model.nv
        qvel = np.zeros(nv); qvel[5] = 1.0; qvel[0] = 1.0
        _, act_err, _ = _run_comparison(mj_model, constants, mj_data.qpos.copy(), qvel)
        assert act_err < PASS_TH, f"wz+vx act_err={act_err:.2e} (was ~0.251 in 2C.4)"

    def test_wx_plus_vy(self, mj_model, mj_data, constants):
        nv = mj_model.nv
        qvel = np.zeros(nv); qvel[3] = 1.0; qvel[1] = 1.0
        _, act_err, _ = _run_comparison(mj_model, constants, mj_data.qpos.copy(), qvel)
        assert act_err < PASS_TH, f"wx+vy act_err={act_err:.2e} (was ~0.105 in 2C.4)"

    def test_wy_plus_vz(self, mj_model, mj_data, constants):
        nv = mj_model.nv
        qvel = np.zeros(nv); qvel[4] = 1.0; qvel[2] = 1.0
        _, act_err, _ = _run_comparison(mj_model, constants, mj_data.qpos.copy(), qvel)
        assert act_err < PASS_TH, f"wy+vz act_err={act_err:.2e} (was ~0.317 in 2C.4)"

    def test_small_random(self, mj_model, mj_data, constants):
        nv = mj_model.nv
        qvel = np.random.default_rng(42).uniform(-0.1, 0.1, nv)
        full_err, act_err, _ = _run_comparison(mj_model, constants, mj_data.qpos.copy(), qvel)
        assert act_err < PASS_TH, f"small_random act_err={act_err:.2e}"
        assert full_err < PASS_TH, f"small_random full_err={full_err:.2e}"

    def test_moderate_random(self, mj_model, mj_data, constants):
        nv = mj_model.nv
        qvel = np.random.default_rng(42).uniform(-0.5, 0.5, nv)
        full_err, act_err, _ = _run_comparison(mj_model, constants, mj_data.qpos.copy(), qvel)
        assert act_err < PASS_TH, f"moderate_random act_err={act_err:.2e}"
        assert full_err < PASS_TH, f"moderate_random full_err={full_err:.2e}"


# ═══════════════════════════════════════════════════════════════════════════
# Cross-term tests: base + actuated
# ═══════════════════════════════════════════════════════════════════════════

class TestBaseActuatedCrossTerms:
    def test_base_wz_plus_l_hip_pitch(self, mj_model, mj_data, constants):
        nv = mj_model.nv
        qvel = np.zeros(nv); qvel[5] = 1.0; qvel[8] = 1.0
        _, act_err, _ = _run_comparison(mj_model, constants, mj_data.qpos.copy(), qvel)
        assert act_err < PASS_TH, f"wz+l_hp act_err={act_err:.2e}"

    def test_base_wz_plus_l_knee(self, mj_model, mj_data, constants):
        nv = mj_model.nv
        qvel = np.zeros(nv); qvel[5] = 1.0; qvel[9] = 1.0
        _, act_err, _ = _run_comparison(mj_model, constants, mj_data.qpos.copy(), qvel)
        assert act_err < PASS_TH, f"wz+l_kn act_err={act_err:.2e}"

    def test_base_wx_plus_l_hip_roll(self, mj_model, mj_data, constants):
        nv = mj_model.nv
        qvel = np.zeros(nv); qvel[3] = 1.0; qvel[6] = 1.0
        _, act_err, _ = _run_comparison(mj_model, constants, mj_data.qpos.copy(), qvel)
        assert act_err < PASS_TH, f"wx+l_hr act_err={act_err:.2e}"

    def test_base_vx_plus_l_hip_pitch(self, mj_model, mj_data, constants):
        nv = mj_model.nv
        qvel = np.zeros(nv); qvel[0] = 1.0; qvel[8] = 1.0
        _, act_err, _ = _run_comparison(mj_model, constants, mj_data.qpos.copy(), qvel)
        assert act_err < PASS_TH, f"vx+l_hp act_err={act_err:.2e}"

    def test_actuated_pair_l_hp_l_kn(self, mj_model, mj_data, constants):
        nv = mj_model.nv
        qvel = np.zeros(nv); qvel[8] = 1.0; qvel[9] = 1.0
        _, act_err, _ = _run_comparison(mj_model, constants, mj_data.qpos.copy(), qvel)
        assert act_err < PASS_TH, f"l_hp+l_kn act_err={act_err:.2e}"


# ═══════════════════════════════════════════════════════════════════════════
# Non-identity orientation tests
# ═══════════════════════════════════════════════════════════════════════════

class TestNonIdentityOrientation:
    def test_wz_vx_at_non_identity(self, mj_model, constants):
        from scipy.spatial.transform import Rotation
        import mujoco

        nv = mj_model.nv
        d = mujoco.MjData(mj_model)
        if mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(mj_model, d, 0)
        mujoco.mj_forward(mj_model, d)

        for rpy, name in [((0,10,0), "p10"), ((0,0,15), "y15"), ((5,8,12), "comb")]:
            R = Rotation.from_euler('xyz', np.deg2rad(rpy)).as_matrix()
            quat = Rotation.from_matrix(R).as_quat()
            qp = d.qpos.copy()
            qp[3:7] = [quat[3], quat[0], quat[1], quat[2]]
            qvel = np.zeros(nv); qvel[5] = 1.0; qvel[0] = 1.0
            _, act_err, _ = _run_comparison(mj_model, constants, qp, qvel)
            assert act_err < PASS_TH, f"wz+vx at {name}: act_err={act_err:.2e}"


# ═══════════════════════════════════════════════════════════════════════════
# JIT compatibility
# ═══════════════════════════════════════════════════════════════════════════

class TestJITCompatibility:
    def test_jit_gravity(self, mj_model, mj_data, constants):
        import jax
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_bias_forces import (
            extract_jax_fk_arrays, extract_jax_bias_arrays, jax_bias_forces_fk_arrays,
        )

        fk_arrays = extract_jax_fk_arrays(constants)
        bias_arrays_full = extract_jax_bias_arrays(constants)
        _, *bias_rest = bias_arrays_full
        bias_arrays = tuple(bias_rest)

        qpos_test = jnp.array(mj_data.qpos.copy(), dtype=jnp.float32)
        qvel_zero = jnp.zeros(mj_model.nv, dtype=jnp.float32)

        jit_fn = jax.jit(lambda q: jax_bias_forces_fk_arrays(q, qvel_zero, fk_arrays, bias_arrays))
        r_jit = np.array(jit_fn(qpos_test))
        r_nojit = np.array(jax_bias_forces_fk_arrays(qpos_test, qvel_zero, fk_arrays, bias_arrays))
        diff = float(np.max(np.abs(r_jit - r_nojit)))
        assert diff < 1e-5, f"JIT gravity diff={diff:.2e}"
        assert np.all(np.isfinite(r_jit)), "JIT gravity output has NaN/Inf"

    def test_jit_full_bias(self, mj_model, mj_data, constants):
        import jax
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_bias_forces import (
            extract_jax_fk_arrays, extract_jax_bias_arrays, jax_bias_forces_fk_arrays,
        )

        fk_arrays = extract_jax_fk_arrays(constants)
        bias_arrays_full = extract_jax_bias_arrays(constants)
        _, *bias_rest = bias_arrays_full
        bias_arrays = tuple(bias_rest)

        qpos_test = jnp.array(mj_data.qpos.copy(), dtype=jnp.float32)
        qvel_test = jnp.array(
            np.random.default_rng(99).uniform(-0.2, 0.2, mj_model.nv),
            dtype=jnp.float32,
        )

        jit_fn = jax.jit(lambda q, qv: jax_bias_forces_fk_arrays(q, qv, fk_arrays, bias_arrays))
        r_jit = np.array(jit_fn(qpos_test, qvel_test))
        r_nojit = np.array(jax_bias_forces_fk_arrays(qpos_test, qvel_test, fk_arrays, bias_arrays))
        diff = float(np.max(np.abs(r_jit - r_nojit)))
        assert diff < 1e-5, f"JIT full bias diff={diff:.2e}"
        assert np.all(np.isfinite(r_jit)), "JIT full bias output has NaN/Inf"


# ═══════════════════════════════════════════════════════════════════════════
# Strict threshold enforcement
# ═══════════════════════════════════════════════════════════════════════════

class TestStrictThresholds:
    """Verify that no test accepts large errors as PASS."""

    def test_actuated_bias_never_exceeds_1e3_threshold(self, mj_model, mj_data, constants):
        """Comprehensive test: actuated bias < 1e-3 for all key velocity patterns."""
        nv = mj_model.nv
        from scipy.spatial.transform import Rotation

        qpos_base = mj_data.qpos.copy()
        orient_cases = [
            ("id", qpos_base),
            ("p10", None), ("y15", None),
        ]
        for rpy, name in [((0,10,0), "p10"), ((0,0,15), "y15")]:
            R = Rotation.from_euler('xyz', np.deg2rad(rpy)).as_matrix()
            quat = Rotation.from_matrix(R).as_quat()
            qp = qpos_base.copy()
            qp[3:7] = [quat[3], quat[0], quat[1], quat[2]]
            orient_cases.append((name, qp))

        vel_patterns = [
            np.zeros(nv),
            np.eye(nv)[0], np.eye(nv)[5],  # vx, wz
            np.eye(nv)[5] + np.eye(nv)[0],  # wz+vx
            np.eye(nv)[3] + np.eye(nv)[1],  # wx+vy
            np.eye(nv)[4] + np.eye(nv)[2],  # wy+vz
            np.eye(nv)[8], np.eye(nv)[9],    # l_hp, l_kn
            np.eye(nv)[5] + np.eye(nv)[8],   # wz+l_hp
        ]
        vel_patterns.append(np.random.default_rng(42).uniform(-0.3, 0.3, nv))

        for oname, qp in orient_cases:
            if qp is None:
                continue
            for vi, qvel in enumerate(vel_patterns):
                full_err, act_err, _ = _run_comparison(mj_model, constants, qp, qvel)
                assert act_err < 1e-3, \
                    f"orient={oname} vel={vi}: act_err={act_err:.2e} >= 1e-3"
                assert full_err < 1e-3, \
                    f"orient={oname} vel={vi}: full_err={full_err:.2e} >= 1e-3"
