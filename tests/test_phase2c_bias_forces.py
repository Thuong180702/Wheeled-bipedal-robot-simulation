"""Tests for Phase 2C — JAX Bias Forces / Gravity / Coriolis Port.

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
    """Create fresh MuJoCo data at keyframe with zero velocity."""
    import mujoco
    data = mujoco.MjData(mj_model)
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, data, 0)
    mujoco.mj_forward(mj_model, data)
    return data


@pytest.fixture(scope="module")
def constants(mj_model):
    """Build bias force constants (shared across tests)."""
    from wheeled_biped.dynamics.jax_bias_forces import build_bias_force_constants
    return build_bias_force_constants(mj_model)


@pytest.fixture
def qpos(mj_data):
    """JAX qpos array at keyframe."""
    import jax.numpy as jnp
    return jnp.array(mj_data.qpos.copy(), dtype=jnp.float32)


# ── Import tests ────────────────────────────────────────────────────

class TestImports:
    """Verify Phase 2C modules import successfully."""

    def test_import_jax_bias_forces(self):
        from wheeled_biped.dynamics.jax_bias_forces import (
            build_bias_force_constants,
            extract_jax_bias_arrays,
            jax_bias_forces,
            jax_bias_forces_fk_arrays,
            jax_gravity_forces,
            jax_velocity_bias_forces,
            compare_bias_forces_to_mujoco,
        )
        assert callable(build_bias_force_constants)
        assert callable(extract_jax_bias_arrays)
        assert callable(jax_bias_forces)
        assert callable(jax_bias_forces_fk_arrays)
        assert callable(jax_gravity_forces)
        assert callable(jax_velocity_bias_forces)
        assert callable(compare_bias_forces_to_mujoco)

    def test_import_package_exports(self):
        import wheeled_biped.dynamics
        assert hasattr(wheeled_biped.dynamics, "build_bias_force_constants")
        assert hasattr(wheeled_biped.dynamics, "jax_bias_forces")
        assert hasattr(wheeled_biped.dynamics, "jax_gravity_forces")
        assert hasattr(wheeled_biped.dynamics, "jax_velocity_bias_forces")
        assert hasattr(wheeled_biped.dynamics, "compare_bias_forces_to_mujoco")


# ── Constants tests ─────────────────────────────────────────────────

class TestBiasForceConstants:
    """Verify build_bias_force_constants returns correct structure."""

    def test_returns_dict(self, constants):
        assert isinstance(constants, dict)

    def test_has_required_keys(self, constants):
        for key in ["nbody", "nq", "nv", "gravity", "body_mass",
                     "body_ipos", "body_iquat", "body_inertia",
                     "parent_ids", "body_categories", "joint_axis",
                     "joint_dof_adr", "body_order", "children"]:
            assert key in constants, f"Missing key: {key}"

    def test_correct_k2_dimensions(self, constants):
        assert constants["nbody"] == 12
        assert constants["nq"] == 17
        assert constants["nv"] == 16

    def test_gravity_vector(self, constants):
        import numpy as np
        g = np.array(constants["gravity"])
        assert g.shape == (3,)
        assert g[2] == pytest.approx(-9.81, abs=0.1)

    def test_extract_jax_bias_arrays(self, constants):
        from wheeled_biped.dynamics.jax_bias_forces import extract_jax_bias_arrays
        arrs = extract_jax_bias_arrays(constants)
        assert isinstance(arrs, tuple)
        assert len(arrs) == 25  # fk_arrays + 16 bias + 3 Phase 2C.3 + 5 Phase 2C.4

    def test_children_array(self, constants):
        children = np.array(constants["children"])
        assert children.shape[0] == constants["nbody"]


# ── Gravity forces tests ────────────────────────────────────────────

class TestGravityForces:
    """Verify gravity force computation."""

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
        """Gravity forces match CPU qfrc_bias at qvel=0."""
        from wheeled_biped.dynamics.jax_bias_forces import jax_gravity_forces
        import numpy as np

        g_jax = np.array(jax_gravity_forces(qpos, constants))
        cpu_bias = np.array(mj_data.qfrc_bias)
        err = np.max(np.abs(g_jax - cpu_bias))
        assert err < 1e-3, f"Gravity error {err:.2e} >= 1e-3 PASS threshold"

    def test_gravity_z_force_positive(self, qpos, constants):
        """Gravity z-force should be positive (opposing gravity)."""
        from wheeled_biped.dynamics.jax_bias_forces import jax_gravity_forces
        import numpy as np
        g = np.array(jax_gravity_forces(qpos, constants))
        assert g[2] > 0, f"Expected positive z-force for gravity opposition, got {g[2]:.2f}"


# ── Bias forces tests ───────────────────────────────────────────────

class TestBiasForces:
    """Verify full bias force computation."""

    def test_bias_shape(self, qpos, constants):
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        import jax.numpy as jnp
        nv = constants["nv"]
        qvel = jnp.zeros(nv, dtype=jnp.float32)
        bias = jax_bias_forces(qpos, qvel, constants)
        assert bias.shape == (16,)

    def test_bias_finite_zero_velocity(self, qpos, constants):
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        nv = constants["nv"]
        qvel = jnp.zeros(nv, dtype=jnp.float32)
        bias = jax_bias_forces(qpos, qvel, constants)
        assert bool(jnp.all(jnp.isfinite(bias)))

    def test_bias_equals_gravity_at_zero_velocity(self, qpos, constants):
        import jax.numpy as jnp
        import numpy as np
        from wheeled_biped.dynamics.jax_bias_forces import (
            jax_bias_forces, jax_gravity_forces,
        )
        nv = constants["nv"]
        qvel = jnp.zeros(nv, dtype=jnp.float32)
        bias = np.array(jax_bias_forces(qpos, qvel, constants))
        grav = np.array(jax_gravity_forces(qpos, constants))
        assert np.allclose(bias, grav, atol=1e-5)

    def test_bias_cpu_comparison_nominal(self, qpos, constants, mj_model, mj_data):
        """Full bias matches CPU at keyframe with zero velocity."""
        import jax.numpy as jnp
        import numpy as np
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces

        nv = constants["nv"]
        qvel = jnp.zeros(nv, dtype=jnp.float32)
        bias_jax = np.array(jax_bias_forces(qpos, qvel, constants))
        cpu_bias = np.array(mj_data.qfrc_bias)
        err = np.max(np.abs(bias_jax - cpu_bias))
        assert err < 1e-3, f"Full bias error {err:.2e} >= 1e-3 PASS threshold"

    def test_actuated_bias_cpu_comparison(self, qpos, constants, mj_model, mj_data):
        """Actuated part bias[6:16] matches CPU."""
        import jax.numpy as jnp
        import numpy as np
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces

        nv = constants["nv"]
        qvel = jnp.zeros(nv, dtype=jnp.float32)
        bias_jax = np.array(jax_bias_forces(qpos, qvel, constants))
        cpu_bias = np.array(mj_data.qfrc_bias)
        act_err = np.max(np.abs(bias_jax[6:16] - cpu_bias[6:16]))
        assert act_err < 1e-3, f"Actuated bias error {act_err:.2e} >= 1e-3"

    def test_bias_finite_nonzero_velocity(self, qpos, constants):
        """Bias forces are finite with random nonzero velocity."""
        import jax.numpy as jnp
        import numpy as np
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces

        nv = constants["nv"]
        rng = np.random.default_rng(42)
        qvel = jnp.array(rng.uniform(-0.2, 0.2, nv), dtype=jnp.float32)
        bias = jax_bias_forces(qpos, qvel, constants)
        assert bool(jnp.all(jnp.isfinite(bias))), "Bias contains NaN/Inf at nonzero velocity"


# ── Velocity bias tests ─────────────────────────────────────────────

class TestVelocityBias:
    """Verify velocity-dependent bias force decomposition."""

    def test_velocity_bias_zero_at_zero_velocity(self, qpos, constants):
        import jax.numpy as jnp
        import numpy as np
        from wheeled_biped.dynamics.jax_bias_forces import jax_velocity_bias_forces

        nv = constants["nv"]
        qvel = jnp.zeros(nv, dtype=jnp.float32)
        vel_bias = np.array(jax_velocity_bias_forces(qpos, qvel, constants))
        assert np.allclose(vel_bias, 0, atol=1e-10)

    def test_decomposition_adds_up(self, qpos, constants):
        """full = gravity + velocity_dependent."""
        import jax.numpy as jnp
        import numpy as np
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

    def test_velocity_bias_shape(self, qpos, constants):
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_bias_forces import jax_velocity_bias_forces

        nv = constants["nv"]
        rng = np.random.default_rng(44)
        qvel = jnp.array(rng.uniform(-0.1, 0.1, nv), dtype=jnp.float32)
        vel_bias = jax_velocity_bias_forces(qpos, qvel, constants)
        assert vel_bias.shape == (16,)


# ── JIT compatibility ───────────────────────────────────────────────

class TestJITCompatibility:
    """Verify JIT compilation works for bias force functions."""

    def test_jit_gravity(self, qpos, constants):
        import jax
        from wheeled_biped.dynamics.jax_bias_forces import (
            jax_bias_forces_fk_arrays, extract_jax_fk_arrays, extract_jax_bias_arrays,
        )
        import jax.numpy as jnp
        import numpy as np

        fk_arrays = extract_jax_fk_arrays(constants)
        bias_arrays_full = extract_jax_bias_arrays(constants)
        _, *rest = bias_arrays_full
        bias_arrays = tuple(rest)

        nv = constants["nv"]
        qvel_zero = jnp.zeros(nv, dtype=jnp.float32)

        jit_fn = jax.jit(lambda q: jax_bias_forces_fk_arrays(
            q, qvel_zero, fk_arrays, bias_arrays))
        result = np.array(jit_fn(qpos))
        assert result.shape == (16,)
        assert np.all(np.isfinite(result))

    def test_jit_full_bias(self, qpos, constants):
        import jax
        from wheeled_biped.dynamics.jax_bias_forces import (
            jax_bias_forces_fk_arrays, extract_jax_fk_arrays, extract_jax_bias_arrays,
        )
        import jax.numpy as jnp
        import numpy as np

        fk_arrays = extract_jax_fk_arrays(constants)
        bias_arrays_full = extract_jax_bias_arrays(constants)
        _, *rest = bias_arrays_full
        bias_arrays = tuple(rest)

        nv = constants["nv"]
        rng = np.random.default_rng(45)
        qvel = jnp.array(rng.uniform(-0.2, 0.2, nv), dtype=jnp.float32)

        jit_fn = jax.jit(lambda q, qv: jax_bias_forces_fk_arrays(
            q, qv, fk_arrays, bias_arrays))
        result = np.array(jit_fn(qpos, qvel))
        assert result.shape == (16,)
        assert np.all(np.isfinite(result))

    def test_jit_matches_nojit_gravity(self, qpos, constants):
        import jax
        from wheeled_biped.dynamics.jax_bias_forces import (
            jax_bias_forces_fk_arrays, extract_jax_fk_arrays, extract_jax_bias_arrays,
        )
        import jax.numpy as jnp
        import numpy as np

        fk_arrays = extract_jax_fk_arrays(constants)
        bias_arrays_full = extract_jax_bias_arrays(constants)
        _, *rest = bias_arrays_full
        bias_arrays = tuple(rest)

        nv = constants["nv"]
        qvel_zero = jnp.zeros(nv, dtype=jnp.float32)

        jit_fn = jax.jit(lambda q: jax_bias_forces_fk_arrays(
            q, qvel_zero, fk_arrays, bias_arrays))
        result_jit = np.array(jit_fn(qpos))
        result_nojit = np.array(jax_bias_forces_fk_arrays(
            qpos, qvel_zero, fk_arrays, bias_arrays))
        diff = np.max(np.abs(result_jit - result_nojit))
        assert diff < 1e-5, f"JIT-vs-noJIT difference {diff:.2e} exceeds threshold"


# ── Validation helper tests ─────────────────────────────────────────

class TestValidationHelper:
    """Verify compare_bias_forces_to_mujoco works correctly."""

    def test_returns_structured(self, mj_model, mj_data, constants):
        from wheeled_biped.dynamics.jax_bias_forces import compare_bias_forces_to_mujoco

        result = compare_bias_forces_to_mujoco(mj_model, mj_data, constants)
        for key in ["full_bias", "free_base_part", "actuated_part",
                     "gravity_only", "velocity_dependent"]:
            assert key in result, f"Missing key: {key}"
            assert "max_abs_error" in result[key]
            assert "max_rel_error" in result[key]
            assert "verdict" in result[key]
        assert "all_finite" in result
        assert "thresholds" in result

    def test_passes_at_keyframe_zero_velocity(self, mj_model, mj_data, constants):
        from wheeled_biped.dynamics.jax_bias_forces import compare_bias_forces_to_mujoco

        # Ensure zero velocity
        mj_data.qvel[:] = 0.0
        import mujoco
        mujoco.mj_forward(mj_model, mj_data)

        result = compare_bias_forces_to_mujoco(mj_model, mj_data, constants)
        assert result["full_bias"]["verdict"] == "PASS"
        assert result["gravity_only"]["verdict"] == "PASS"
        assert result["actuated_part"]["verdict"] == "PASS"
        assert result["all_finite"] is True

    def test_nonzero_velocity_returns_result(self, mj_model, mj_data, constants):
        import numpy as np
        import mujoco
        from wheeled_biped.dynamics.jax_bias_forces import compare_bias_forces_to_mujoco

        rng = np.random.default_rng(46)
        mj_data.qvel[:] = rng.uniform(-0.1, 0.1, mj_model.nv)
        mujoco.mj_forward(mj_model, mj_data)

        result = compare_bias_forces_to_mujoco(mj_model, mj_data, constants)
        # Should return result even if not all PASS
        assert "full_bias" in result
        assert "velocity_dependent" in result
        assert result["all_finite"] is True


# ── No-controller-import tests ───────────────────────────────────────

class TestNoControllerImports:
    """Verify Phase 2C code does not import controller modules."""

    FORBIDDEN = [
        "k2_jax_controller",
        "sagittal_velocity_damped_balance_controller",
    ]

    def test_jax_bias_forces_no_controller_import(self):
        import ast
        src = (PROJECT_ROOT / "wheeled_biped" / "dynamics" / "jax_bias_forces.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not any(f in alias.name for f in self.FORBIDDEN), (
                        f"jax_bias_forces.py imports forbidden: {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not any(f in node.module for f in self.FORBIDDEN), (
                        f"jax_bias_forces.py imports forbidden: {node.module}"
                    )
