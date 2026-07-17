"""Tests for Phase 2A — JAX Kinematics / COM / Jacobian port.

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
    """Build kinematic tree constants (shared across tests)."""
    from wheeled_biped.dynamics.jax_kinematics import build_kinematic_tree_constants

    return build_kinematic_tree_constants(mj_model)


@pytest.fixture(scope="module")
def fk_arrays(constants):
    """Extract JAX array tuple for JIT-safe FK."""
    from wheeled_biped.dynamics.jax_kinematics import extract_jax_fk_arrays

    return extract_jax_fk_arrays(constants)


@pytest.fixture
def qpos(mj_data):
    """JAX qpos array at keyframe."""
    import jax.numpy as jnp

    return jnp.array(mj_data.qpos.copy(), dtype=jnp.float32)


# ── Import tests ────────────────────────────────────────────────────

class TestImports:
    """Verify Phase 2A modules import successfully."""

    def test_import_jax_kinematics(self):
        from wheeled_biped.dynamics.jax_kinematics import (
            build_kinematic_tree_constants,
            extract_jax_fk_arrays,
            jax_forward_kinematics,
            jax_forward_kinematics_fk_arrays,
        )
        assert callable(build_kinematic_tree_constants)
        assert callable(extract_jax_fk_arrays)
        assert callable(jax_forward_kinematics)
        assert callable(jax_forward_kinematics_fk_arrays)

    def test_import_jax_com(self):
        from wheeled_biped.dynamics.jax_com import (
            jax_compute_com,
            jax_compute_body_com_positions,
            jax_compute_subtree_or_total_com,
        )
        assert callable(jax_compute_com)
        assert callable(jax_compute_body_com_positions)
        assert callable(jax_compute_subtree_or_total_com)

    def test_import_jax_jacobians(self):
        from wheeled_biped.dynamics.jax_jacobians import (
            jax_body_position_jacobian,
            jax_compute_all_target_jacobians,
            validate_jacobian_actuated_columns,
        )
        assert callable(jax_body_position_jacobian)
        assert callable(jax_compute_all_target_jacobians)
        assert callable(validate_jacobian_actuated_columns)

    def test_import_package_exports(self):
        import wheeled_biped.dynamics
        assert hasattr(wheeled_biped.dynamics, "build_kinematic_tree_constants")
        assert hasattr(wheeled_biped.dynamics, "jax_forward_kinematics")
        assert hasattr(wheeled_biped.dynamics, "jax_compute_com")
        assert hasattr(wheeled_biped.dynamics, "jax_body_position_jacobian")


# ── Constants tests ─────────────────────────────────────────────────

class TestKinematicTreeConstants:
    """Verify build_kinematic_tree_constants returns correct structure."""

    def test_returns_dict(self, constants):
        assert isinstance(constants, dict)

    def test_has_dimensions(self, constants):
        for key in ["nbody", "njnt", "nq", "nv"]:
            assert key in constants, f"Missing key: {key}"
            assert isinstance(constants[key], int), f"{key} should be int"

    def test_correct_k2_dimensions(self, constants):
        assert constants["nbody"] == 12
        assert constants["njnt"] == 11
        assert constants["nq"] == 17
        assert constants["nv"] == 16

    def test_has_required_jax_arrays(self, constants):
        for key in [
            "parent_ids", "body_jntadr", "body_pos_local", "body_quat_local",
            "joint_type", "joint_axis", "joint_qpos_adr", "joint_dof_adr",
            "body_mass", "body_categories",
        ]:
            assert key in constants, f"Missing key: {key}"

    def test_has_metadata(self, constants):
        for key in ["body_names", "joint_names", "target_body_ids"]:
            assert key in constants, f"Missing key: {key}"

    def test_target_body_ids_has_mandatory(self, constants):
        mandatory = [
            "torso", "l_wheel_link", "r_wheel_link",
            "l_knee_link", "r_knee_link", "l_thigh", "r_thigh",
        ]
        for name in mandatory:
            assert name in constants["target_body_ids"], f"Missing target: {name}"
            assert constants["target_body_ids"][name] >= 0, f"Target {name} not found in model"

    def test_extract_jax_fk_arrays_returns_tuple(self, constants):
        from wheeled_biped.dynamics.jax_kinematics import extract_jax_fk_arrays

        arrs = extract_jax_fk_arrays(constants)
        assert isinstance(arrs, tuple)
        assert len(arrs) == 8  # 8 JAX arrays


# ── FK tests ────────────────────────────────────────────────────────

class TestForwardKinematics:
    """Verify JAX FK matches CPU MuJoCo ground truth."""

    def test_fk_returns_dict(self, qpos, constants):
        from wheeled_biped.dynamics.jax_kinematics import jax_forward_kinematics

        result = jax_forward_kinematics(qpos, constants)
        assert isinstance(result, dict)
        assert "body_pos_world" in result
        assert "body_quat_world" in result

    def test_fk_shapes(self, qpos, constants):
        from wheeled_biped.dynamics.jax_kinematics import jax_forward_kinematics

        result = jax_forward_kinematics(qpos, constants)
        nbody = constants["nbody"]
        assert result["body_pos_world"].shape == (nbody, 3)
        assert result["body_quat_world"].shape == (nbody, 4)

    def test_fk_positions_finite(self, qpos, constants):
        from wheeled_biped.dynamics.jax_kinematics import jax_forward_kinematics
        import jax.numpy as jnp

        result = jax_forward_kinematics(qpos, constants)
        assert bool(jnp.all(jnp.isfinite(result["body_pos_world"])))
        assert bool(jnp.all(jnp.isfinite(result["body_quat_world"])))

    def test_torso_position_matches_cpu(self, qpos, constants, mj_data):
        from wheeled_biped.dynamics.jax_kinematics import jax_forward_kinematics

        result = jax_forward_kinematics(qpos, constants)
        jax_torso = np.array(result["body_pos_world"][1])
        cpu_torso = mj_data.xpos[1]
        err = np.max(np.abs(jax_torso - cpu_torso))
        assert err < 1e-4, f"Torso position error {err:.2e} exceeds 1e-4 PASS threshold"

    def test_all_mandatory_fk_pass(self, qpos, constants, mj_data):
        from wheeled_biped.dynamics.jax_kinematics import jax_forward_kinematics

        result = jax_forward_kinematics(qpos, constants)
        jax_xpos = np.array(result["body_pos_world"])
        cpu_xpos = mj_data.xpos.copy()

        mandatory = [
            "torso", "l_wheel_link", "r_wheel_link",
            "l_knee_link", "r_knee_link", "l_thigh", "r_thigh",
        ]
        for name in mandatory:
            bid = constants["target_body_ids"][name]
            err = np.max(np.abs(jax_xpos[bid] - cpu_xpos[bid]))
            assert err < 1e-4, f"{name}: FK position error {err:.2e} >= 1e-4 PASS threshold"

    def test_fk_jit_works(self, qpos, fk_arrays):
        import jax
        from wheeled_biped.dynamics.jax_kinematics import jax_forward_kinematics_fk_arrays

        jit_fn = jax.jit(jax_forward_kinematics_fk_arrays)
        result = jit_fn(qpos, fk_arrays)
        assert result["body_pos_world"].shape == (12, 3)
        import jax.numpy as jnp
        assert bool(jnp.all(jnp.isfinite(result["body_pos_world"])))


# ── COM tests ───────────────────────────────────────────────────────

class TestCenterOfMass:
    """Verify JAX COM matches CPU MuJoCo ground truth."""

    def test_com_returns_finite_vector(self, qpos, constants):
        from wheeled_biped.dynamics.jax_com import jax_compute_subtree_or_total_com

        result = jax_compute_subtree_or_total_com(qpos, constants)
        import jax.numpy as jnp
        assert result["com"].shape == (3,)
        assert bool(jnp.all(jnp.isfinite(result["com"])))

    def test_com_matches_cpu(self, qpos, constants, mj_data):
        from wheeled_biped.dynamics.jax_kinematics import jax_forward_kinematics
        from wheeled_biped.dynamics.jax_com import jax_compute_com

        fk = jax_forward_kinematics(qpos, constants)
        jax_com = jax_compute_com(
            fk["body_pos_world"], fk["body_quat_world"],
            constants["body_ipos"], constants["body_mass"],
        )
        cpu_com = mj_data.subtree_com[1]
        err = float(np.max(np.abs(np.array(jax_com) - cpu_com)))
        assert err < 1e-4, f"COM error {err:.2e} exceeds 1e-4 PASS threshold"

    def test_com_jit_works(self, qpos, fk_arrays, constants):
        import jax
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_kinematics import jax_forward_kinematics_fk_arrays
        from wheeled_biped.dynamics.jax_com import jax_compute_com

        def jit_fn(q, fa, ipos, mass):
            fk = jax_forward_kinematics_fk_arrays(q, fa)
            return jax_compute_com(fk["body_pos_world"], fk["body_quat_world"], ipos, mass)

        jit_compute = jax.jit(jit_fn)
        result = jit_compute(qpos, fk_arrays, constants["body_ipos"], constants["body_mass"])
        assert result.shape == (3,)
        assert bool(jnp.all(jnp.isfinite(result)))

    def test_body_com_positions_shape(self, qpos, constants):
        from wheeled_biped.dynamics.jax_kinematics import jax_forward_kinematics
        from wheeled_biped.dynamics.jax_com import jax_compute_body_com_positions

        fk = jax_forward_kinematics(qpos, constants)
        body_com = jax_compute_body_com_positions(
            fk["body_pos_world"], fk["body_quat_world"], constants["body_ipos"],
        )
        assert body_com.shape == (constants["nbody"], 3)


# ── Jacobian tests ──────────────────────────────────────────────────

class TestJacobians:
    """Verify JAX Jacobians match CPU MuJoCo ground truth."""

    def test_jacobian_returns_structured(self, qpos, constants):
        from wheeled_biped.dynamics.jax_jacobians import jax_body_position_jacobian

        result = jax_body_position_jacobian(qpos, constants, 1)  # torso
        assert "jac_full" in result
        assert "jac_actuated" in result
        assert "jac_full_shape" in result
        assert "jac_actuated_shape" in result
        assert result["jac_full_shape"] == [3, 17]
        assert result["jac_actuated_shape"] == [3, 10]

    def test_jacobian_actuated_finite(self, qpos, constants):
        from wheeled_biped.dynamics.jax_jacobians import jax_body_position_jacobian
        import jax.numpy as jnp

        result = jax_body_position_jacobian(qpos, constants, 1)
        assert bool(jnp.all(jnp.isfinite(result["jac_actuated"])))

    def test_torso_jacobian_matches_cpu(self, qpos, constants, mj_model, mj_data):
        from wheeled_biped.dynamics.jax_jacobians import jax_body_position_jacobian
        from wheeled_biped.dynamics.jacobian_checks import compute_task_jacobian
        import numpy as np

        jax_jac = jax_body_position_jacobian(qpos, constants, 1)
        cpu_jac = compute_task_jacobian(mj_model, mj_data, "torso", "body")
        cpu_jacp = np.array(cpu_jac["jacp"])
        jax_act = np.array(jax_jac["jac_actuated"])

        err = np.max(np.abs(jax_act - cpu_jacp[:, 6:16]))
        assert err < 1e-3, f"Torso Jacobian actuated-column error {err:.2e} >= 1e-3 PASS threshold"

    def test_all_mandatory_jacobians_pass(self, qpos, constants, mj_model, mj_data):
        from wheeled_biped.dynamics.jax_jacobians import (
            jax_body_position_jacobian,
            validate_jacobian_actuated_columns,
        )
        from wheeled_biped.dynamics.jacobian_checks import compute_task_jacobian
        import numpy as np

        mandatory = [
            "torso", "l_wheel_link", "r_wheel_link",
            "l_knee_link", "r_knee_link", "l_thigh", "r_thigh",
        ]
        for name in mandatory:
            bid = constants["target_body_ids"][name]
            jax_jac = jax_body_position_jacobian(qpos, constants, int(bid))
            cpu_jac = compute_task_jacobian(mj_model, mj_data, name, "body")
            cpu_jacp = np.array(cpu_jac["jacp"])

            result = validate_jacobian_actuated_columns(
                jax_jac["jac_actuated"], cpu_jacp, name,
            )
            assert result["verdict"] == "PASS", (
                f"{name}: Jacobian verdict {result['verdict']} "
                f"(max_abs_err={result['max_abs_error']:.2e})"
            )

    def test_jacobian_jit_works(self, qpos, constants):
        import jax
        from wheeled_biped.dynamics.jax_jacobians import jax_body_position_jacobian
        import jax.numpy as jnp

        jit_fn = jax.jit(lambda q: jax_body_position_jacobian(q, constants, 1))
        result = jit_fn(qpos)
        assert result["jac_actuated_shape"] == [3, 10]
        assert bool(jnp.all(jnp.isfinite(result["jac_actuated"])))

    def test_compute_all_target_jacobians(self, qpos, constants):
        from wheeled_biped.dynamics.jax_jacobians import jax_compute_all_target_jacobians

        results = jax_compute_all_target_jacobians(qpos, constants)
        mandatory = ["torso", "l_wheel_link", "r_wheel_link", "l_knee_link", "r_knee_link"]
        for name in mandatory:
            assert name in results, f"Missing Jacobian for {name}"
            assert "jac_actuated" in results[name]


# ── No-controller-import tests ───────────────────────────────────────

class TestNoControllerImports:
    """Verify Phase 2A code does not import controller modules."""

    FORBIDDEN = [
        "k2_jax_controller",
        "sagittal_velocity_damped_balance_controller",
    ]

    def test_jax_kinematics_no_controller_import(self):
        import ast
        src = (PROJECT_ROOT / "wheeled_biped" / "dynamics" / "jax_kinematics.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not any(f in alias.name for f in self.FORBIDDEN), (
                        f"jax_kinematics.py imports forbidden: {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not any(f in node.module for f in self.FORBIDDEN), (
                        f"jax_kinematics.py imports forbidden: {node.module}"
                    )

    def test_jax_com_no_controller_import(self):
        import ast
        src = (PROJECT_ROOT / "wheeled_biped" / "dynamics" / "jax_com.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not any(f in alias.name for f in self.FORBIDDEN)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not any(f in node.module for f in self.FORBIDDEN)

    def test_jax_jacobians_no_controller_import(self):
        import ast
        src = (PROJECT_ROOT / "wheeled_biped" / "dynamics" / "jax_jacobians.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not any(f in alias.name for f in self.FORBIDDEN)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not any(f in node.module for f in self.FORBIDDEN)
