"""Tests for Phase 2B — JAX Mass Matrix / CRBA Port.

Lightweight, CPU-only tests. No GPU, no training, no visual mode.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mj_model():
    """Load MuJoCo model."""
    import mujoco
    from wheeled_biped.utils.config import get_model_path

    return mujoco.MjModel.from_xml_path(str(get_model_path()))


@pytest.fixture
def mj_data(mj_model):
    """Create fresh MuJoCo data at keyframe."""
    import mujoco

    data = mujoco.MjData(mj_model)
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, data, 0)
    mujoco.mj_forward(mj_model, data)
    return data


@pytest.fixture(scope="module")
def constants(mj_model):
    """Build mass matrix constants (shared across tests)."""
    from wheeled_biped.dynamics.jax_mass_matrix import build_mass_matrix_constants

    return build_mass_matrix_constants(mj_model)


@pytest.fixture
def qpos(mj_data):
    """JAX qpos array at keyframe."""
    import jax.numpy as jnp

    return jnp.array(mj_data.qpos.copy(), dtype=jnp.float32)


# ── Import tests ────────────────────────────────────────────────────

class TestImports:
    """Verify Phase 2B modules import successfully."""

    def test_import_jax_mass_matrix(self):
        from wheeled_biped.dynamics.jax_mass_matrix import (
            build_mass_matrix_constants,
            extract_jax_mm_arrays,
            jax_mass_matrix,
            jax_mass_matrix_fk_arrays,
            jax_actuated_mass_submatrix,
            jax_body_spatial_velocities,
            jax_compute_kinetic_energy,
            compare_mass_matrix_to_mujoco,
        )
        assert callable(build_mass_matrix_constants)
        assert callable(extract_jax_mm_arrays)
        assert callable(jax_mass_matrix)
        assert callable(jax_mass_matrix_fk_arrays)
        assert callable(jax_actuated_mass_submatrix)
        assert callable(jax_body_spatial_velocities)
        assert callable(jax_compute_kinetic_energy)
        assert callable(compare_mass_matrix_to_mujoco)

    def test_import_package_exports(self):
        import wheeled_biped.dynamics
        assert hasattr(wheeled_biped.dynamics, "build_mass_matrix_constants")
        assert hasattr(wheeled_biped.dynamics, "jax_mass_matrix")
        assert hasattr(wheeled_biped.dynamics, "jax_actuated_mass_submatrix")
        assert hasattr(wheeled_biped.dynamics, "jax_body_spatial_velocities")
        assert hasattr(wheeled_biped.dynamics, "compare_mass_matrix_to_mujoco")


# ── Constants tests ─────────────────────────────────────────────────

class TestMassMatrixConstants:
    """Verify build_mass_matrix_constants returns correct structure."""

    def test_returns_dict(self, constants):
        assert isinstance(constants, dict)

    def test_has_dimensions(self, constants):
        for key in ["nbody", "nq", "nv"]:
            assert key in constants
            assert isinstance(constants[key], int)

    def test_correct_k2_dimensions(self, constants):
        assert constants["nbody"] == 12
        assert constants["nq"] == 17
        assert constants["nv"] == 16

    def test_has_inertia_arrays(self, constants):
        for key in ["body_mass", "body_inertia", "body_ipos", "body_iquat"]:
            assert key in constants, f"Missing key: {key}"

    def test_body_inertia_shapes(self, constants):
        nbody = constants["nbody"]
        assert constants["body_mass"].shape == (nbody,)
        assert constants["body_inertia"].shape == (nbody, 3)
        assert constants["body_ipos"].shape == (nbody, 3)
        assert constants["body_iquat"].shape == (nbody, 4)

    def test_has_dof_armature(self, constants):
        assert "dof_armature" in constants
        assert constants["dof_armature"].shape == (constants["nv"],)

    def test_extract_jax_mm_arrays(self, constants):
        from wheeled_biped.dynamics.jax_mass_matrix import extract_jax_mm_arrays

        arrs = extract_jax_mm_arrays(constants)
        assert isinstance(arrs, tuple)
        assert len(arrs) == 8  # fk_arrays + 7 mm arrays


# ── Body spatial velocity tests ─────────────────────────────────────

class TestBodySpatialVelocities:
    """Verify JAX body velocities match CPU Jacobians."""

    def test_returns_structured(self, qpos, constants):
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_mass_matrix import jax_body_spatial_velocities

        nv = constants["nv"]
        qvel = jnp.zeros(nv, dtype=jnp.float32)
        result = jax_body_spatial_velocities(qpos, qvel, constants)
        assert "body_vel_world" in result
        assert "body_omega_world" in result
        assert "body_pos_world" in result
        assert "body_quat_world" in result

    def test_shapes(self, qpos, constants):
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_mass_matrix import jax_body_spatial_velocities

        nv = constants["nv"]
        nbody = constants["nbody"]
        qvel = jnp.zeros(nv, dtype=jnp.float32)
        result = jax_body_spatial_velocities(qpos, qvel, constants)
        assert result["body_vel_world"].shape == (nbody, 3)
        assert result["body_omega_world"].shape == (nbody, 3)

    def test_zero_qvel_gives_zero_velocity(self, qpos, constants):
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_mass_matrix import jax_body_spatial_velocities

        nv = constants["nv"]
        qvel = jnp.zeros(nv, dtype=jnp.float32)
        result = jax_body_spatial_velocities(qpos, qvel, constants)
        import numpy as np
        assert np.allclose(np.array(result["body_vel_world"]), 0, atol=1e-7)
        assert np.allclose(np.array(result["body_omega_world"]), 0, atol=1e-7)

    def test_free_base_velocity_pass_through(self, qpos, constants):
        """Torso velocity should equal qvel[0:6]."""
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_mass_matrix import jax_body_spatial_velocities

        nv = constants["nv"]
        qvel = jnp.zeros(nv, dtype=jnp.float32)
        qvel = qvel.at[0].set(1.0)  # base lin x
        qvel = qvel.at[5].set(2.0)  # base ang z
        result = jax_body_spatial_velocities(qpos, qvel, constants)
        import numpy as np
        torso_v = np.array(result["body_vel_world"][1])
        torso_w = np.array(result["body_omega_world"][1])
        assert np.allclose(torso_v, [1.0, 0.0, 0.0], atol=1e-6)
        assert np.allclose(torso_w, [0.0, 0.0, 2.0], atol=1e-6)


# ── Mass matrix tests ──────────────────────────────────────────────

class TestMassMatrix:
    """Verify mass matrix properties and CPU comparison."""

    def test_shape(self, qpos, constants):
        from wheeled_biped.dynamics.jax_mass_matrix import jax_mass_matrix

        M = jax_mass_matrix(qpos, constants)
        assert M.shape == (16, 16)

    def test_finite(self, qpos, constants):
        from wheeled_biped.dynamics.jax_mass_matrix import jax_mass_matrix
        import jax.numpy as jnp

        M = jax_mass_matrix(qpos, constants)
        assert bool(jnp.all(jnp.isfinite(M)))

    def test_symmetric(self, qpos, constants):
        from wheeled_biped.dynamics.jax_mass_matrix import jax_mass_matrix
        import numpy as np

        M = jax_mass_matrix(qpos, constants)
        M_np = np.array(M)
        asym = np.max(np.abs(M_np - M_np.T))
        assert asym < 1e-10, f"Mass matrix asymmetry {asym:.2e} exceeds 1e-10"

    def test_diagonal_positive(self, qpos, constants):
        from wheeled_biped.dynamics.jax_mass_matrix import jax_mass_matrix
        import numpy as np

        M = jax_mass_matrix(qpos, constants)
        diag = np.diag(np.array(M))
        assert np.all(diag > 0), f"Non-positive diagonal entries: {diag}"

    def test_cpu_comparison_nominal(self, qpos, constants, mj_model, mj_data):
        """JAX mass matrix matches CPU mj_fullM at keyframe within 1e-4."""
        from wheeled_biped.dynamics.jax_mass_matrix import jax_mass_matrix
        import numpy as np
        import mujoco

        M_jax = np.array(jax_mass_matrix(qpos, constants))
        nv = mj_model.nv
        cpu_M = np.zeros((nv, nv))
        mujoco.mj_fullM(mj_model, cpu_M, mj_data.qM)
        err = np.max(np.abs(M_jax - cpu_M))
        assert err < 1e-4, f"Mass matrix error {err:.2e} >= 1e-4 PASS threshold"

    def test_actuated_submatrix(self, qpos, constants, mj_model, mj_data):
        """Actuated sub-block M[6:16, 6:16] matches CPU."""
        from wheeled_biped.dynamics.jax_mass_matrix import jax_actuated_mass_submatrix
        import numpy as np
        import mujoco

        M_act = np.array(jax_actuated_mass_submatrix(qpos, constants))
        assert M_act.shape == (10, 10)

        nv = mj_model.nv
        cpu_M = np.zeros((nv, nv))
        mujoco.mj_fullM(mj_model, cpu_M, mj_data.qM)
        cpu_act = cpu_M[6:16, 6:16]
        err = np.max(np.abs(M_act - cpu_act))
        assert err < 1e-4, f"Actuated block error {err:.2e} >= 1e-4"

    def test_jit_compatible(self, qpos, constants):
        """JIT compilation succeeds and produces matching output."""
        import jax
        from wheeled_biped.dynamics.jax_mass_matrix import (
            jax_mass_matrix_fk_arrays,
            extract_jax_mm_arrays,
        )
        import numpy as np

        fk_arrays, body_mass, body_ipos, body_iquat, body_inertia, joint_dof_adr, body_order, dof_armature = extract_jax_mm_arrays(constants)
        mm_arrays = (body_mass, body_ipos, body_iquat, body_inertia, joint_dof_adr, body_order, dof_armature)

        jit_fn = jax.jit(lambda q: jax_mass_matrix_fk_arrays(q, fk_arrays, mm_arrays))
        M_jit = np.array(jit_fn(qpos))
        assert M_jit.shape == (16, 16)
        assert np.all(np.isfinite(M_jit))


# ── Validation helper tests ─────────────────────────────────────────

class TestValidationHelper:
    """Verify compare_mass_matrix_to_mujoco works correctly."""

    def test_returns_structured(self, mj_model, mj_data, constants):
        from wheeled_biped.dynamics.jax_mass_matrix import compare_mass_matrix_to_mujoco

        result = compare_mass_matrix_to_mujoco(mj_model, mj_data, constants)
        assert "full_matrix" in result
        assert "actuated_block" in result
        assert "symmetry" in result
        assert "diagonal" in result
        assert "all_finite" in result
        assert "condition_number" in result

    def test_passes_at_keyframe(self, mj_model, mj_data, constants):
        from wheeled_biped.dynamics.jax_mass_matrix import compare_mass_matrix_to_mujoco

        result = compare_mass_matrix_to_mujoco(mj_model, mj_data, constants)
        assert result["full_matrix"]["verdict"] == "PASS"
        assert result["actuated_block"]["verdict"] == "PASS"
        assert result["symmetry"]["verdict"] == "PASS"
        assert result["diagonal"]["all_positive"] is True
        assert result["all_finite"] is True


# ── No-controller-import tests ───────────────────────────────────────

class TestNoControllerImports:
    """Verify Phase 2B code does not import controller modules."""

    FORBIDDEN = [
        "k2_jax_controller",
        "sagittal_velocity_damped_balance_controller",
    ]

    def test_jax_mass_matrix_no_controller_import(self):
        import ast
        src = (PROJECT_ROOT / "wheeled_biped" / "dynamics" / "jax_mass_matrix.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not any(f in alias.name for f in self.FORBIDDEN), (
                        f"jax_mass_matrix.py imports forbidden: {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not any(f in node.module for f in self.FORBIDDEN), (
                        f"jax_mass_matrix.py imports forbidden: {node.module}"
                    )
