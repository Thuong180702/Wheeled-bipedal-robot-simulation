"""Tests for Phase 2D — Contact Dynamics / Contact Jacobian / Constraint Force Validation.

Validates JAX contact kinematics, translational/rotational Jacobians, and
contact force → generalized force mapping against CPU MuJoCo ground truth.

CPU-only, no GPU, no training, no visual mode.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PASS_TH_JAC = 1e-5   # Jacobian threshold (strict, machine-precision nearby)
WARN_TH_JAC = 1e-4
PASS_TH_QFRC = 1e-4  # Force mapping threshold
WARN_TH_QFRC = 1e-3
PASS_TH_POINT = 1e-6  # Contact point reconstruction threshold


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
def constants(mj_model):
    from wheeled_biped.dynamics.jax_contact_dynamics import build_contact_dynamics_constants
    return build_contact_dynamics_constants(mj_model)


@pytest.fixture(scope="module")
def qpos_ref(mj_data):
    import jax.numpy as jnp
    return jnp.array(mj_data.qpos.copy(), dtype=jnp.float32)


@pytest.fixture(scope="module")
def l_wheel_id(constants):
    return constants["wheel_body_ids"]["l_wheel_link"]


@pytest.fixture(scope="module")
def r_wheel_id(constants):
    return constants["wheel_body_ids"]["r_wheel_link"]


def _np_quat_to_rotmat(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])


# ═══════════════════════════════════════════════════════════════════════════
# Import and version tests
# ═══════════════════════════════════════════════════════════════════════════

class TestImports:
    def test_all_public_api_importable(self):
        from wheeled_biped.dynamics.jax_contact_dynamics import (
            build_contact_dynamics_constants,
            contact_point_world_position,
            contact_point_translational_jacobian,
            contact_point_rotational_jacobian,
            contact_force_to_generalized_force,
            contact_wrench_to_generalized_force,
            CONSTANTS_VERSION,
        )
        assert CONSTANTS_VERSION == "phase2d_contact_dynamics"

    def test_constants_version(self, constants):
        assert constants["constants_version"] == "phase2d_contact_dynamics"

    def test_constants_required_keys(self, constants):
        required = [
            "nq", "nv", "nbody", "ngeom",
            "wheel_body_ids", "wheel_geom_ids", "floor_geom_ids",
            "body_to_root_padded", "body_path_len",
            "body_dof_adr", "joint_axis", "joint_type", "body_jntadr",
            "free_joint_convention",
        ]
        for key in required:
            assert key in constants, f"Missing required key: {key}"

    def test_no_controller_imports(self):
        """Verify no controller modules are imported in jax_contact_dynamics."""
        import ast
        src = (PROJECT_ROOT / "wheeled_biped" / "dynamics" / "jax_contact_dynamics.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        forbidden = ["k2_jax_controller", "sagittal_velocity_damped_balance_controller"]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not any(f in alias.name for f in forbidden), \
                        f"jax_contact_dynamics.py imports forbidden: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not any(f in node.module for f in forbidden), \
                        f"jax_contact_dynamics.py imports forbidden: {node.module}"


# ═══════════════════════════════════════════════════════════════════════════
# Contact point position tests
# ═══════════════════════════════════════════════════════════════════════════

class TestContactPointPosition:
    def test_returns_shape_3(self, qpos_ref, l_wheel_id, constants):
        from wheeled_biped.dynamics.jax_contact_dynamics import contact_point_world_position
        import jax.numpy as jnp
        local_pt = jnp.array([0.0, 0.0, -0.06], dtype=jnp.float32)
        p = contact_point_world_position(qpos_ref, l_wheel_id, local_pt, constants)
        assert p.shape == (3,)

    def test_returns_finite(self, qpos_ref, l_wheel_id, constants):
        from wheeled_biped.dynamics.jax_contact_dynamics import contact_point_world_position
        import jax.numpy as jnp
        local_pt = jnp.array([0.0, 0.0, -0.06], dtype=jnp.float32)
        p = contact_point_world_position(qpos_ref, l_wheel_id, local_pt, constants)
        assert np.all(np.isfinite(np.array(p)))

    def test_wheel_bottom_near_floor(self, mj_model, mj_data, qpos_ref, l_wheel_id, constants):
        """At default keyframe, wheel bottom should be near z=0 (on floor)."""
        from wheeled_biped.dynamics.jax_contact_dynamics import contact_point_world_position
        import jax.numpy as jnp
        # Wheel collision is cylinder with radius 0.06 at pos [-0.038, 0, 0]
        # Bottom point is at z ≈ -0.06 locally
        local_pt = jnp.array([0.0, 0.0, -0.06], dtype=jnp.float32)
        p = np.array(contact_point_world_position(qpos_ref, l_wheel_id, local_pt, constants))
        assert abs(p[2]) < 0.01, f"Wheel bottom z={p[2]:.4f}, expected near 0"

    def test_reconstruction_vs_cpu_keyframe(self, mj_model, mj_data, qpos_ref, l_wheel_id, r_wheel_id, constants):
        """Contact point reconstruction matches CPU MuJoCo body position + rotation."""
        from wheeled_biped.dynamics.jax_contact_dynamics import contact_point_world_position
        import jax.numpy as jnp
        for body_id in [l_wheel_id, r_wheel_id]:
            local_pt = jnp.array([0.0, 0.0, -0.06], dtype=jnp.float32)
            p_jax = np.array(contact_point_world_position(qpos_ref, body_id, local_pt, constants))
            # CPU: body xpos + R_body @ local_point
            R_cpu = _np_quat_to_rotmat(mj_data.xquat[body_id])
            p_cpu = mj_data.xpos[body_id] + R_cpu @ np.array([0.0, 0.0, -0.06])
            err = float(np.max(np.abs(p_jax - p_cpu)))
            assert err < PASS_TH_POINT, f"Body {body_id} point reconstruction error {err:.2e}"


# ═══════════════════════════════════════════════════════════════════════════
# Translational Jacobian tests
# ═══════════════════════════════════════════════════════════════════════════

class TestTranslationalJacobian:
    def test_returns_shape_3x16(self, qpos_ref, l_wheel_id, constants):
        from wheeled_biped.dynamics.jax_contact_dynamics import contact_point_translational_jacobian
        import jax.numpy as jnp
        local_pt = jnp.array([0.0, 0.0, -0.06], dtype=jnp.float32)
        Jp = contact_point_translational_jacobian(qpos_ref, l_wheel_id, local_pt, constants)
        assert Jp.shape == (3, 16)

    def test_finite(self, qpos_ref, l_wheel_id, constants):
        from wheeled_biped.dynamics.jax_contact_dynamics import contact_point_translational_jacobian
        import jax.numpy as jnp
        local_pt = jnp.array([0.0, 0.0, -0.06], dtype=jnp.float32)
        Jp = contact_point_translational_jacobian(qpos_ref, l_wheel_id, local_pt, constants)
        assert np.all(np.isfinite(np.array(Jp)))

    def test_base_linear_columns_identity(self, qpos_ref, l_wheel_id, constants):
        """qvel[0:3] = v_lin_world directly adds to point velocity → Jp[:,0:3] = I_3."""
        from wheeled_biped.dynamics.jax_contact_dynamics import contact_point_translational_jacobian
        import jax.numpy as jnp
        local_pt = jnp.array([0.0, 0.0, -0.06], dtype=jnp.float32)
        Jp = np.array(contact_point_translational_jacobian(qpos_ref, l_wheel_id, local_pt, constants))
        err = float(np.max(np.abs(Jp[:, 0:3] - np.eye(3))))
        assert err < 1e-12, f"Base linear columns not I_3: err={err:.2e}"

    def test_base_angular_columns_body_frame_convention(self, mj_model, mj_data, qpos_ref, l_wheel_id, constants):
        """qvel[3:6] = omega_BODY.  Contribution to point velocity from qvel[3:6]
        is (R_base @ omega_body) × r = -skew(r) @ R_base @ omega_body.

        So Jp[:, 3:6] should equal -skew(r) @ R_base_world.
        """
        from wheeled_biped.dynamics.jax_contact_dynamics import contact_point_translational_jacobian, contact_point_world_position
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_bias_forces import _quat_to_rotmat, _skew3

        local_pt = jnp.array([0.0, 0.0, -0.06], dtype=jnp.float32)
        Jp = np.array(contact_point_translational_jacobian(qpos_ref, l_wheel_id, local_pt, constants))

        p_w = np.array(contact_point_world_position(qpos_ref, l_wheel_id, local_pt, constants))
        base_origin = mj_data.xpos[1]  # torso position
        r = p_w - base_origin

        R_base_np = _np_quat_to_rotmat(mj_data.xquat[1])
        expected_Jp_ang = -np.array(_skew3(jnp.array(r, dtype=jnp.float32))) @ R_base_np
        err = float(np.max(np.abs(Jp[:, 3:6] - expected_Jp_ang)))
        assert err < 1e-5, f"Base angular columns convention error: {err:.2e}"

    def test_actuated_columns_finite(self, qpos_ref, l_wheel_id, constants):
        from wheeled_biped.dynamics.jax_contact_dynamics import contact_point_translational_jacobian
        import jax.numpy as jnp
        local_pt = jnp.array([0.0, 0.0, -0.06], dtype=jnp.float32)
        Jp = np.array(contact_point_translational_jacobian(qpos_ref, l_wheel_id, local_pt, constants))
        assert np.all(np.isfinite(Jp[:, 6:16])), "Actuated columns contain NaN/Inf"

    def test_jacobian_vs_cpu_mujoco_keyframe(self, mj_model, mj_data, qpos_ref, l_wheel_id, r_wheel_id, constants):
        """Full Jp comparison against CPU MuJoCo mj_jac at keyframe."""
        from wheeled_biped.dynamics.jax_contact_dynamics import contact_point_translational_jacobian
        import jax.numpy as jnp
        import mujoco

        for body_id in [l_wheel_id, r_wheel_id]:
            local_pt = jnp.array([0.0, 0.0, -0.06], dtype=jnp.float32)
            Jp_jax = np.array(
                contact_point_translational_jacobian(qpos_ref, body_id, local_pt, constants),
                dtype=np.float64,
            )

            # CPU Jacobian
            body_pos = mj_data.xpos[body_id].copy()
            body_quat = mj_data.xquat[body_id].copy()
            R_body = _np_quat_to_rotmat(body_quat)
            p_world = body_pos + R_body @ np.array([0.0, 0.0, -0.06])
            jacp_cpu = np.zeros((3, mj_model.nv), dtype=np.float64)
            jacr_cpu = np.zeros((3, mj_model.nv), dtype=np.float64)
            mujoco.mj_jac(mj_model, mj_data, jacp_cpu, jacr_cpu, p_world, body_id)

            full_err = float(np.max(np.abs(Jp_jax - jacp_cpu)))
            base_lin_err = float(np.max(np.abs(Jp_jax[:, 0:3] - jacp_cpu[:, 0:3])))
            base_ang_err = float(np.max(np.abs(Jp_jax[:, 3:6] - jacp_cpu[:, 3:6])))
            act_err = float(np.max(np.abs(Jp_jax[:, 6:16] - jacp_cpu[:, 6:16])))

            assert full_err < PASS_TH_JAC, \
                f"Body {body_id}: full Jp error {full_err:.2e} >= {PASS_TH_JAC}"
            assert base_lin_err < PASS_TH_JAC, \
                f"Body {body_id}: base linear error {base_lin_err:.2e}"
            assert base_ang_err < PASS_TH_JAC, \
                f"Body {body_id}: base angular error {base_ang_err:.2e}"
            assert act_err < PASS_TH_JAC, \
                f"Body {body_id}: actuated error {act_err:.2e}"

    def test_jacobian_vs_cpu_mujoco_non_identity_orientation(self, mj_model, mj_data, constants):
        """Jp comparison at non-identity base orientations."""
        from wheeled_biped.dynamics.jax_contact_dynamics import contact_point_translational_jacobian
        import jax.numpy as jnp
        import mujoco
        from scipy.spatial.transform import Rotation

        qpos_base = mj_data.qpos.copy()
        for rpy, name in [((0, 10, 0), "p10"), ((0, 0, 15), "y15"), ((5, 8, 12), "comb")]:
            R = Rotation.from_euler('xyz', np.deg2rad(rpy)).as_matrix()
            quat = Rotation.from_matrix(R).as_quat()
            qp = qpos_base.copy()
            qp[3:7] = [quat[3], quat[0], quat[1], quat[2]]
            d2 = mujoco.MjData(mj_model); d2.qpos[:] = qp
            mujoco.mj_forward(mj_model, d2)
            qpos_jax = jnp.array(qp, dtype=jnp.float32)

            for body_id in [constants["wheel_body_ids"]["l_wheel_link"],
                           constants["wheel_body_ids"]["r_wheel_link"]]:
                local_pt = jnp.array([0.0, 0.0, -0.06], dtype=jnp.float32)
                Jp_jax = np.array(
                    contact_point_translational_jacobian(qpos_jax, body_id, local_pt, constants),
                    dtype=np.float64,
                )
                body_pos = d2.xpos[body_id].copy()
                body_quat = d2.xquat[body_id].copy()
                R_body = _np_quat_to_rotmat(body_quat)
                p_world = body_pos + R_body @ np.array([0.0, 0.0, -0.06])
                jacp_cpu = np.zeros((3, mj_model.nv), dtype=np.float64)
                jacr_cpu = np.zeros((3, mj_model.nv), dtype=np.float64)
                mujoco.mj_jac(mj_model, d2, jacp_cpu, jacr_cpu, p_world, body_id)

                full_err = float(np.max(np.abs(Jp_jax - jacp_cpu)))
                assert full_err < PASS_TH_JAC, \
                    f"Body {body_id} at {name}: full Jp error {full_err:.2e}"


# ═══════════════════════════════════════════════════════════════════════════
# Rotational Jacobian tests
# ═══════════════════════════════════════════════════════════════════════════

class TestRotationalJacobian:
    def test_returns_shape_3x16(self, qpos_ref, l_wheel_id, constants):
        from wheeled_biped.dynamics.jax_contact_dynamics import contact_point_rotational_jacobian
        Jr = contact_point_rotational_jacobian(qpos_ref, l_wheel_id, constants)
        assert Jr.shape == (3, 16)

    def test_finite(self, qpos_ref, l_wheel_id, constants):
        from wheeled_biped.dynamics.jax_contact_dynamics import contact_point_rotational_jacobian
        Jr = contact_point_rotational_jacobian(qpos_ref, l_wheel_id, constants)
        assert np.all(np.isfinite(np.array(Jr)))

    def test_vs_cpu_mujoco_keyframe(self, mj_model, mj_data, qpos_ref, l_wheel_id, r_wheel_id, constants):
        """Full Jr comparison against CPU MuJoCo mj_jac at keyframe."""
        from wheeled_biped.dynamics.jax_contact_dynamics import contact_point_rotational_jacobian
        import mujoco

        for body_id in [l_wheel_id, r_wheel_id]:
            Jr_jax = np.array(
                contact_point_rotational_jacobian(qpos_ref, body_id, constants),
                dtype=np.float64,
            )
            # CPU: mj_jac at body origin
            body_pos = mj_data.xpos[body_id].copy()
            jacp_cpu = np.zeros((3, mj_model.nv), dtype=np.float64)
            jacr_cpu = np.zeros((3, mj_model.nv), dtype=np.float64)
            mujoco.mj_jac(mj_model, mj_data, jacp_cpu, jacr_cpu, body_pos, body_id)

            full_err = float(np.max(np.abs(Jr_jax - jacr_cpu)))
            assert full_err < PASS_TH_JAC, \
                f"Body {body_id}: Jr error {full_err:.2e} >= {PASS_TH_JAC}"


# ═══════════════════════════════════════════════════════════════════════════
# Contact force → generalized force mapping tests
# ═══════════════════════════════════════════════════════════════════════════

class TestContactForceToQfrc:
    def test_returns_shape_16(self, qpos_ref, l_wheel_id, constants):
        from wheeled_biped.dynamics.jax_contact_dynamics import contact_force_to_generalized_force
        import jax.numpy as jnp
        local_pt = jnp.array([0.0, 0.0, -0.06], dtype=jnp.float32)
        f_w = jnp.array([0.0, 0.0, 100.0], dtype=jnp.float32)
        qfrc = contact_force_to_generalized_force(qpos_ref, l_wheel_id, local_pt, f_w, constants)
        assert qfrc.shape == (16,)

    def test_returns_finite(self, qpos_ref, l_wheel_id, constants):
        from wheeled_biped.dynamics.jax_contact_dynamics import contact_force_to_generalized_force
        import jax.numpy as jnp
        local_pt = jnp.array([0.0, 0.0, -0.06], dtype=jnp.float32)
        f_w = jnp.array([0.0, 0.0, 100.0], dtype=jnp.float32)
        qfrc = contact_force_to_generalized_force(qpos_ref, l_wheel_id, local_pt, f_w, constants)
        assert np.all(np.isfinite(np.array(qfrc)))

    def test_vertical_force_produces_z_reaction(self, qpos_ref, l_wheel_id, constants):
        """A vertical force on a wheel should produce upward force on free-base Z."""
        from wheeled_biped.dynamics.jax_contact_dynamics import contact_force_to_generalized_force
        import jax.numpy as jnp
        local_pt = jnp.array([0.0, 0.0, -0.06], dtype=jnp.float32)
        f_w = jnp.array([0.0, 0.0, 100.0], dtype=jnp.float32)  # 100N upward
        qfrc = np.array(contact_force_to_generalized_force(qpos_ref, l_wheel_id, local_pt, f_w, constants))
        # Free-base Z force should equal the applied Z force (Jp[:,0:3] = I_3)
        assert abs(qfrc[2] - 100.0) < 1e-5, f"Z force {qfrc[2]:.6f} != 100.0"

    def test_vs_cpu_path_a_keyframe(self, mj_model, mj_data, qpos_ref, l_wheel_id, r_wheel_id, constants):
        """JAX qfrc mapping vs CPU jacp^T @ force_world."""
        from wheeled_biped.dynamics.jax_contact_dynamics import contact_force_to_generalized_force
        import jax.numpy as jnp
        import mujoco

        for body_id in [l_wheel_id, r_wheel_id]:
            local_pt = jnp.array([0.0, 0.0, -0.06], dtype=jnp.float32)
            # Test with various forces
            for f_test in [
                np.array([0.0, 0.0, 100.0]),
                np.array([10.0, 0.0, 100.0]),
                np.array([0.0, 20.0, 50.0]),
                np.array([-10.0, 15.0, 80.0]),
            ]:
                f_w_jax = jnp.array(f_test, dtype=jnp.float32)
                qfrc_jax = np.array(
                    contact_force_to_generalized_force(qpos_ref, body_id, local_pt, f_w_jax, constants),
                    dtype=np.float64,
                )
                # CPU reference: jacp^T @ f
                body_pos = mj_data.xpos[body_id].copy()
                body_quat = mj_data.xquat[body_id].copy()
                R_body = _np_quat_to_rotmat(body_quat)
                p_world = body_pos + R_body @ np.array([0.0, 0.0, -0.06])
                jacp_cpu = np.zeros((3, mj_model.nv), dtype=np.float64)
                jacr_cpu = np.zeros((3, mj_model.nv), dtype=np.float64)
                mujoco.mj_jac(mj_model, mj_data, jacp_cpu, jacr_cpu, p_world, body_id)
                qfrc_cpu = jacp_cpu.T @ f_test

                err = float(np.max(np.abs(qfrc_jax - qfrc_cpu)))
                assert err < PASS_TH_QFRC, \
                    f"Body {body_id} f={f_test}: qfrc error {err:.2e} >= {PASS_TH_QFRC}"


# ═══════════════════════════════════════════════════════════════════════════
# Contact wrench tests
# ═══════════════════════════════════════════════════════════════════════════

class TestContactWrenchToQfrc:
    def test_returns_shape_16(self, qpos_ref, l_wheel_id, constants):
        from wheeled_biped.dynamics.jax_contact_dynamics import contact_wrench_to_generalized_force
        import jax.numpy as jnp
        local_pt = jnp.array([0.0, 0.0, -0.06], dtype=jnp.float32)
        f_w = jnp.array([0.0, 0.0, 100.0], dtype=jnp.float32)
        tau_w = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
        qfrc = contact_wrench_to_generalized_force(qpos_ref, l_wheel_id, local_pt, f_w, tau_w, constants)
        assert qfrc.shape == (16,)

    def test_torque_adds_to_free_base(self, qpos_ref, l_wheel_id, constants):
        """Torque on a body should add to free-base torque DOFs."""
        from wheeled_biped.dynamics.jax_contact_dynamics import (
            contact_force_to_generalized_force,
            contact_wrench_to_generalized_force,
        )
        import jax.numpy as jnp
        local_pt = jnp.array([0.0, 0.0, -0.06], dtype=jnp.float32)
        f_w = jnp.array([0.0, 0.0, 100.0], dtype=jnp.float32)
        tau_w = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)  # torque about world X

        qfrc_f_only = np.array(contact_force_to_generalized_force(qpos_ref, l_wheel_id, local_pt, f_w, constants))
        qfrc_f_tau = np.array(contact_wrench_to_generalized_force(qpos_ref, l_wheel_id, local_pt, f_w, tau_w, constants))

        # The torque should affect the generalized force
        diff = float(np.max(np.abs(qfrc_f_tau - qfrc_f_only)))
        assert diff > 1e-10, "Torque addition had no effect on qfrc"


# ═══════════════════════════════════════════════════════════════════════════
# Contact frame tests
# ═══════════════════════════════════════════════════════════════════════════

class TestContactFrameHandling:
    def test_transform_identity(self):
        """Identity contact frame: world force = contact-frame force."""
        from wheeled_biped.dynamics.jax_contact_dynamics import transform_contact_force_to_world
        import jax.numpy as jnp
        f_contact = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        R_id = jnp.eye(3, dtype=jnp.float32)
        f_world = transform_contact_force_to_world(f_contact, R_id)
        assert float(jnp.max(jnp.abs(f_world - f_contact))) < 1e-6

    def test_transform_rotated(self):
        """Rotated contact frame: force should transform correctly."""
        from wheeled_biped.dynamics.jax_contact_dynamics import transform_contact_force_to_world
        import jax.numpy as jnp
        # 90-degree rotation about Z
        R_z90 = jnp.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=jnp.float32)
        f_contact = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)  # along X in contact frame
        f_world = transform_contact_force_to_world(f_contact, R_z90)
        expected = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32)  # along Y in world
        assert float(jnp.max(jnp.abs(f_world - expected))) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# JIT compatibility tests
# ═══════════════════════════════════════════════════════════════════════════

class TestJITCompatibility:
    def test_jit_contact_point_position(self, qpos_ref, l_wheel_id, constants):
        import jax
        from wheeled_biped.dynamics.jax_contact_dynamics import contact_point_world_position
        import jax.numpy as jnp
        local_pt = jnp.array([0.0, 0.0, -0.06], dtype=jnp.float32)
        jit_fn = jax.jit(lambda q: contact_point_world_position(q, l_wheel_id, local_pt, constants))
        r_jit = np.array(jit_fn(qpos_ref))
        r_nojit = np.array(contact_point_world_position(qpos_ref, l_wheel_id, local_pt, constants))
        diff = float(np.max(np.abs(r_jit - r_nojit)))
        assert diff < 1e-5, f"JIT contact point diff={diff:.2e}"
        assert np.all(np.isfinite(r_jit))

    def test_jit_translational_jacobian(self, qpos_ref, l_wheel_id, constants):
        import jax
        from wheeled_biped.dynamics.jax_contact_dynamics import contact_point_translational_jacobian
        import jax.numpy as jnp
        local_pt = jnp.array([0.0, 0.0, -0.06], dtype=jnp.float32)
        jit_fn = jax.jit(lambda q: contact_point_translational_jacobian(q, l_wheel_id, local_pt, constants))
        r_jit = np.array(jit_fn(qpos_ref))
        r_nojit = np.array(contact_point_translational_jacobian(qpos_ref, l_wheel_id, local_pt, constants))
        diff = float(np.max(np.abs(r_jit - r_nojit)))
        assert diff < 1e-5, f"JIT Jp diff={diff:.2e}"
        assert np.all(np.isfinite(r_jit))

    def test_jit_rotational_jacobian(self, qpos_ref, l_wheel_id, constants):
        import jax
        from wheeled_biped.dynamics.jax_contact_dynamics import contact_point_rotational_jacobian
        jit_fn = jax.jit(lambda q: contact_point_rotational_jacobian(q, l_wheel_id, constants))
        r_jit = np.array(jit_fn(qpos_ref))
        r_nojit = np.array(contact_point_rotational_jacobian(qpos_ref, l_wheel_id, constants))
        diff = float(np.max(np.abs(r_jit - r_nojit)))
        assert diff < 1e-5, f"JIT Jr diff={diff:.2e}"
        assert np.all(np.isfinite(r_jit))

    def test_jit_force_to_qfrc(self, qpos_ref, l_wheel_id, constants):
        import jax
        from wheeled_biped.dynamics.jax_contact_dynamics import contact_force_to_generalized_force
        import jax.numpy as jnp
        local_pt = jnp.array([0.0, 0.0, -0.06], dtype=jnp.float32)
        f_w = jnp.array([10.0, 0.0, 100.0], dtype=jnp.float32)
        jit_fn = jax.jit(lambda q, f: contact_force_to_generalized_force(q, l_wheel_id, local_pt, f, constants))
        r_jit = np.array(jit_fn(qpos_ref, f_w))
        r_nojit = np.array(contact_force_to_generalized_force(qpos_ref, l_wheel_id, local_pt, f_w, constants))
        diff = float(np.max(np.abs(r_jit - r_nojit)))
        assert diff < 1e-5, f"JIT qfrc diff={diff:.2e}"
        assert np.all(np.isfinite(r_jit))
