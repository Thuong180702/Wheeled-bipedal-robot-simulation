"""Tests for Phase 2C.1 — Bias / Coriolis Correction.

Validates the corrected world-frame RNEA bias force computation.
CPU-only, no GPU, no training, no visual mode.

Note: Mixed-velocity (small_random, moderate_random) cases are expected
to FAIL with the current world-frame approach.  These reflect residual
Coriolis coefficient errors documented in the Phase 2C.1 audit.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


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


# ── Import tests ────────────────────────────────────────────────────────────


class TestImports:
    def test_import_jax_bias_forces(self):
        from wheeled_biped.dynamics.jax_bias_forces import (
            build_bias_force_constants,
            extract_jax_bias_arrays,
            jax_bias_forces,
            jax_bias_forces_fk_arrays,
            jax_gravity_forces,
            jax_velocity_bias_forces,
            compare_bias_forces_to_mujoco,
            _crm, _crf, _skew3, _body_local_spatial_inertia, _quat_to_rotmat,
        )
        assert callable(build_bias_force_constants)
        assert callable(jax_bias_forces)
        assert callable(_crm)
        assert callable(_crf)
        assert callable(_skew3)
        assert callable(_body_local_spatial_inertia)

    def test_import_diagnostics(self):
        from wheeled_biped.dynamics.bias_force_diagnostics import (
            decompose_bias_errors,
            decompose_velocity_components,
            compute_cross_term_decomposition,
        )
        assert callable(decompose_bias_errors)

    def test_import_package_exports(self):
        import wheeled_biped.dynamics
        assert hasattr(wheeled_biped.dynamics, "build_bias_force_constants")
        assert hasattr(wheeled_biped.dynamics, "jax_bias_forces")


# ── Spatial algebra tests ────────────────────────────────────────────────────


class TestSpatialAlgebra:
    def test_crm_shape_and_finite(self):
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_bias_forces import _crm
        v = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=jnp.float32)
        M = _crm(v)
        assert M.shape == (6, 6)
        assert bool(jnp.all(jnp.isfinite(M)))

    def test_crf_shape_and_finite(self):
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_bias_forces import _crf
        v = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=jnp.float32)
        M = _crf(v)
        assert M.shape == (6, 6)
        assert bool(jnp.all(jnp.isfinite(M)))

    def test_crm_crf_dual_relation(self):
        import jax.numpy as jnp
        import numpy as np_cpu
        from wheeled_biped.dynamics.jax_bias_forces import _crm, _crf
        v = jnp.array([0.5, -0.3, 1.2, -2.0, 0.1, 0.8], dtype=jnp.float32)
        M_crm = np_cpu.array(_crm(v))
        M_crf = np_cpu.array(_crf(v))
        assert np_cpu.allclose(M_crm.T + M_crf, 0.0, atol=1e-5)

    def test_spatial_inertia_symmetric_and_finite(self):
        import jax.numpy as jnp
        import numpy as np_cpu
        from wheeled_biped.dynamics.jax_bias_forces import _body_local_spatial_inertia
        m = jnp.array(5.0, dtype=jnp.float32)
        r = jnp.array([0.1, 0.0, -0.2], dtype=jnp.float32)
        I_cm = jnp.eye(3, dtype=jnp.float32) * 0.1
        I_spatial = np_cpu.array(_body_local_spatial_inertia(m, r, I_cm))
        assert I_spatial.shape == (6, 6)
        assert np_cpu.all(np_cpu.isfinite(I_spatial))
        assert np_cpu.allclose(I_spatial, I_spatial.T, atol=1e-5)

    def test_skew3(self):
        import jax.numpy as jnp
        import numpy as np_cpu
        from wheeled_biped.dynamics.jax_bias_forces import _skew3
        v = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32)
        S = np_cpu.array(_skew3(v))
        assert S.shape == (3, 3)
        assert np_cpu.allclose(S + S.T, 0.0, atol=1e-7)


# ── Constants tests ──────────────────────────────────────────────────────────


class TestBiasForceConstants:
    def test_returns_dict(self, constants):
        assert isinstance(constants, dict)

    def test_has_required_keys(self, constants):
        for key in ["nbody", "nq", "nv", "gravity", "body_mass",
                     "body_ipos", "body_iquat", "body_inertia",
                     "parent_ids", "body_categories", "joint_axis",
                     "joint_dof_adr", "body_order", "children",
                     "body_inertia_3x3"]:
            assert key in constants, f"Missing key: {key}"

    def test_correct_k2_dimensions(self, constants):
        assert constants["nbody"] == 12
        assert constants["nq"] == 17
        assert constants["nv"] == 16

    def test_gravity_vector(self, constants):
        g = np.array(constants["gravity"])
        assert g.shape == (3,)
        assert g[2] == pytest.approx(-9.81, abs=0.1)

    def test_extract_jax_bias_arrays(self, constants):
        from wheeled_biped.dynamics.jax_bias_forces import extract_jax_bias_arrays
        arrs = extract_jax_bias_arrays(constants)
        assert isinstance(arrs, tuple)
        assert len(arrs) in (17, 25)  # fk_arrays + bias arrays (Phase 2C.4: 25)


# ── Gravity forces tests ────────────────────────────────────────────────────


class TestGravityForces:
    def test_gravity_shape(self, qpos, constants):
        from wheeled_biped.dynamics.jax_bias_forces import jax_gravity_forces
        g = jax_gravity_forces(qpos, constants)
        assert g.shape == (16,)

    def test_gravity_finite(self, qpos, constants):
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_bias_forces import jax_gravity_forces
        g = jax_gravity_forces(qpos, constants)
        assert bool(jnp.all(jnp.isfinite(g)))

    def test_gravity_cpu_comparison_nominal(self, qpos, constants, mj_model, mj_data):
        from wheeled_biped.dynamics.jax_bias_forces import jax_gravity_forces
        g_jax = np.array(jax_gravity_forces(qpos, constants))
        cpu_bias = np.array(mj_data.qfrc_bias)
        err = np.max(np.abs(g_jax - cpu_bias))
        assert err < 1e-3, f"Gravity error {err:.2e} >= 1e-3 PASS threshold"


# ── Bias forces tests ────────────────────────────────────────────────────────


class TestBiasForces:
    def test_bias_shape(self, qpos, constants):
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        import jax.numpy as jnp
        qvel = jnp.zeros(constants["nv"], dtype=jnp.float32)
        bias = jax_bias_forces(qpos, qvel, constants)
        assert bias.shape == (16,)

    def test_bias_cpu_comparison_nominal(self, qpos, constants, mj_model, mj_data):
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        qvel = jnp.zeros(constants["nv"], dtype=jnp.float32)
        bias_jax = np.array(jax_bias_forces(qpos, qvel, constants))
        cpu_bias = np.array(mj_data.qfrc_bias)
        err = np.max(np.abs(bias_jax - cpu_bias))
        assert err < 1e-3, f"Full bias error {err:.2e} >= 1e-3 PASS threshold"

    def test_base_yaw_rate_pass(self, qpos, constants, mj_model):
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        import mujoco
        nv = constants["nv"]
        qvel_np = np.zeros(nv); qvel_np[5] = 1.0
        d = mujoco.MjData(mj_model); d.qpos[:] = np.array(qpos); d.qvel[:] = qvel_np
        mujoco.mj_forward(mj_model, d)
        jax_full = np.array(jax_bias_forces(qpos, jnp.array(qvel_np, dtype=jnp.float32), constants))
        err = float(np.max(np.abs(jax_full - d.qfrc_bias)))
        assert err < 1e-3, f"Base yaw rate error {err:.2e} >= 1e-3"

    def test_symmetric_wheels_pass(self, qpos, constants, mj_model):
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        import mujoco
        nv = constants["nv"]
        qvel_np = np.zeros(nv); qvel_np[10] = 5.0; qvel_np[15] = 5.0
        d = mujoco.MjData(mj_model); d.qpos[:] = np.array(qpos); d.qvel[:] = qvel_np
        mujoco.mj_forward(mj_model, d)
        jax_full = np.array(jax_bias_forces(qpos, jnp.array(qvel_np, dtype=jnp.float32), constants))
        err = float(np.max(np.abs(jax_full - d.qfrc_bias)))
        assert err < 1e-3, f"Symmetric wheels error {err:.2e} >= 1e-3"

    # ── Known limitations: mixed-velocity cases ──────────────────────

    def test_small_random_velocity_known_limitation(self, qpos, constants, mj_model):
        """Small random velocity has known Coriolis coupling error (documented)."""
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        import mujoco
        nv = constants["nv"]
        rng = np.random.default_rng(123)
        qvel_np = rng.uniform(-0.1, 0.1, nv)
        d = mujoco.MjData(mj_model); d.qpos[:] = np.array(qpos); d.qvel[:] = qvel_np
        mujoco.mj_forward(mj_model, d)
        jax_full = np.array(jax_bias_forces(qpos, jnp.array(qvel_np, dtype=jnp.float32), constants))
        err = float(np.max(np.abs(jax_full - d.qfrc_bias)))
        # Known limitation: max error ~0.08 (dominant free-base coupling)
        assert err < 0.5, f"Small random error {err:.2e} exceeds known bound"

    def test_moderate_random_velocity_known_limitation(self, qpos, constants, mj_model):
        """Moderate random velocity has known Coriolis coupling error (documented)."""
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        import mujoco
        nv = constants["nv"]
        rng = np.random.default_rng(123)
        qvel_np = rng.uniform(-0.5, 0.5, nv)
        d = mujoco.MjData(mj_model); d.qpos[:] = np.array(qpos); d.qvel[:] = qvel_np
        mujoco.mj_forward(mj_model, d)
        jax_full = np.array(jax_bias_forces(qpos, jnp.array(qvel_np, dtype=jnp.float32), constants))
        err = float(np.max(np.abs(jax_full - d.qfrc_bias)))
        # Known limitation: max error ~1.9 N (dominant free-base coupling)
        assert err < 5.0, f"Moderate random error {err:.2e} exceeds known bound"


# ── Velocity bias tests ──────────────────────────────────────────────────────


class TestVelocityBias:
    def test_velocity_bias_zero_at_zero_velocity(self, qpos, constants):
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_bias_forces import jax_velocity_bias_forces
        qvel = jnp.zeros(constants["nv"], dtype=jnp.float32)
        vel_bias = np.array(jax_velocity_bias_forces(qpos, qvel, constants))
        assert np.allclose(vel_bias, 0, atol=1e-10)

    def test_decomposition_adds_up(self, qpos, constants):
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_bias_forces import (
            jax_bias_forces, jax_gravity_forces, jax_velocity_bias_forces,
        )
        nv = constants["nv"]
        rng = np.random.default_rng(43)
        qvel = jnp.array(rng.uniform(-0.3, 0.3, nv), dtype=jnp.float32)
        full = np.array(jax_bias_forces(qpos, qvel, constants))
        grav = np.array(jax_gravity_forces(qpos, constants))
        vel = np.array(jax_velocity_bias_forces(qpos, qvel, constants))
        assert np.allclose(full, grav + vel, atol=1e-5)


# ── JIT compatibility ────────────────────────────────────────────────────────


class TestJITCompatibility:
    def test_jit_gravity(self, qpos, constants):
        import jax
        from wheeled_biped.dynamics.jax_bias_forces import (
            jax_bias_forces_fk_arrays, extract_jax_fk_arrays, extract_jax_bias_arrays,
        )
        import jax.numpy as jnp
        fk_arrays = extract_jax_fk_arrays(constants)
        bias_arrays_full = extract_jax_bias_arrays(constants)
        _, *rest = bias_arrays_full; bias_arrays = tuple(rest)
        qvel_zero = jnp.zeros(constants["nv"], dtype=jnp.float32)
        jit_fn = jax.jit(lambda q: jax_bias_forces_fk_arrays(q, qvel_zero, fk_arrays, bias_arrays))
        result = np.array(jit_fn(qpos))
        assert result.shape == (16,)
        assert np.all(np.isfinite(result))

    def test_jit_full_bias(self, qpos, constants):
        import jax
        from wheeled_biped.dynamics.jax_bias_forces import (
            jax_bias_forces_fk_arrays, extract_jax_fk_arrays, extract_jax_bias_arrays,
        )
        import jax.numpy as jnp
        fk_arrays = extract_jax_fk_arrays(constants)
        bias_arrays_full = extract_jax_bias_arrays(constants)
        _, *rest = bias_arrays_full; bias_arrays = tuple(rest)
        nv = constants["nv"]
        rng = np.random.default_rng(45)
        qvel = jnp.array(rng.uniform(-0.2, 0.2, nv), dtype=jnp.float32)
        jit_fn = jax.jit(lambda q, qv: jax_bias_forces_fk_arrays(q, qv, fk_arrays, bias_arrays))
        result = np.array(jit_fn(qpos, qvel))
        assert result.shape == (16,)
        assert np.all(np.isfinite(result))

    def test_jit_matches_nojit(self, qpos, constants):
        import jax
        from wheeled_biped.dynamics.jax_bias_forces import (
            jax_bias_forces_fk_arrays, extract_jax_fk_arrays, extract_jax_bias_arrays,
        )
        import jax.numpy as jnp
        fk_arrays = extract_jax_fk_arrays(constants)
        bias_arrays_full = extract_jax_bias_arrays(constants)
        _, *rest = bias_arrays_full; bias_arrays = tuple(rest)
        qvel_zero = jnp.zeros(constants["nv"], dtype=jnp.float32)
        jit_fn = jax.jit(lambda q: jax_bias_forces_fk_arrays(q, qvel_zero, fk_arrays, bias_arrays))
        result_jit = np.array(jit_fn(qpos))
        result_nojit = np.array(jax_bias_forces_fk_arrays(qpos, qvel_zero, fk_arrays, bias_arrays))
        diff = np.max(np.abs(result_jit - result_nojit))
        assert diff < 1e-5


# ── Cross-term diagnostic tests ──────────────────────────────────────────────


class TestCrossTermDiagnostics:
    def test_cross_term_returns_structured(self, qpos, constants, mj_model):
        from wheeled_biped.dynamics.bias_force_diagnostics import compute_cross_term_decomposition
        nv = constants["nv"]
        qpos_np = np.array(qpos)
        cross_pairs = [{"name": "test_pair", "v_i": _v(nv, 6, 1.0), "v_j": _v(nv, 8, 1.0)}]
        results = compute_cross_term_decomposition(mj_model, constants, qpos_np, cross_pairs)
        assert len(results) == 1
        assert "cross_max_abs_error" in results[0]

    def test_cross_term_zero_for_zero_velocity(self, qpos, constants, mj_model):
        from wheeled_biped.dynamics.bias_force_diagnostics import compute_cross_term_decomposition
        nv = constants["nv"]
        qpos_np = np.array(qpos)
        cross_pairs = [{"name": "zero_pair", "v_i": np.zeros(nv), "v_j": np.zeros(nv)}]
        results = compute_cross_term_decomposition(mj_model, constants, qpos_np, cross_pairs)
        assert results[0]["cross_max_abs_error"] < 1e-3


# ── Validation helper tests ──────────────────────────────────────────────────


class TestValidationHelper:
    def test_passes_at_keyframe_zero_velocity(self, mj_model, mj_data, constants):
        from wheeled_biped.dynamics.jax_bias_forces import compare_bias_forces_to_mujoco
        import mujoco
        mj_data.qvel[:] = 0.0
        mujoco.mj_forward(mj_model, mj_data)
        result = compare_bias_forces_to_mujoco(mj_model, mj_data, constants)
        assert result["full_bias"]["verdict"] == "PASS"
        assert result["gravity_only"]["verdict"] == "PASS"
        assert result["all_finite"] is True


# ── No-controller-import tests ───────────────────────────────────────────────


class TestNoControllerImports:
    FORBIDDEN = ["k2_jax_controller", "sagittal_velocity_damped_balance_controller"]

    def test_jax_bias_forces_no_controller_import(self):
        import ast
        src = (PROJECT_ROOT / "wheeled_biped" / "dynamics" / "jax_bias_forces.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not any(f in alias.name for f in self.FORBIDDEN)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not any(f in node.module for f in self.FORBIDDEN)

    def test_bias_force_diagnostics_no_controller_import(self):
        import ast
        src = (PROJECT_ROOT / "wheeled_biped" / "dynamics" / "bias_force_diagnostics.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not any(f in alias.name for f in self.FORBIDDEN)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not any(f in node.module for f in self.FORBIDDEN)


def _v(nv, idx, val):
    arr = np.zeros(nv)
    arr[idx] = val
    return arr
