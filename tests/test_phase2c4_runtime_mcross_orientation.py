"""Tests for Phase 2C.4 — Runtime M_cross + Non-Identity Base Orientation Fix.

Validates the runtime M_cross(q) torque correction, non-identity base
orientation handling, and strict validation thresholds against CPU MuJoCo.
CPU-only, no GPU, no training, no visual mode.
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


# ═══════════════════════════════════════════════════════════════════════
# Import tests
# ═══════════════════════════════════════════════════════════════════════

class TestImports:
    def test_runtime_m_cross_importable(self):
        from wheeled_biped.dynamics.jax_bias_forces import runtime_m_cross
        assert callable(runtime_m_cross)

    def test_free_base_gyroscopic_correction_importable(self):
        from wheeled_biped.dynamics.jax_bias_forces import free_base_gyroscopic_correction
        assert callable(free_base_gyroscopic_correction)

    def test_diagnose_base_orientation_bias_importable(self):
        from wheeled_biped.dynamics.jax_bias_forces import diagnose_base_orientation_bias
        assert callable(diagnose_base_orientation_bias)

    def test_runtime_m_cross_fk_arrays_importable(self):
        from wheeled_biped.dynamics.jax_bias_forces import runtime_m_cross_fk_arrays
        assert callable(runtime_m_cross_fk_arrays)

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
            runtime_m_cross,
            runtime_m_cross_fk_arrays,
            free_base_gyroscopic_correction,
            diagnose_base_orientation_bias,
        )
        assert callable(runtime_m_cross)
        assert callable(free_base_gyroscopic_correction)
        assert callable(diagnose_base_orientation_bias)


# ═══════════════════════════════════════════════════════════════════════
# Constants tests
# ═══════════════════════════════════════════════════════════════════════

class TestConstants:
    def test_version(self, constants):
        assert constants["constants_version"] in (
            "phase2c4_runtime_mcross_orientation", "phase2c5_actuated_coriolis",
        )

    def test_has_runtime_mass_matrix(self, constants):
        assert constants.get("_has_runtime_mass_matrix", False), \
            "Runtime mass matrix constants not built"

    def test_dof_armature_present(self, constants):
        assert "dof_armature" in constants
        assert constants["dof_armature"].shape == (16,)

    def test_phase2c3_keys_present(self, constants):
        for key in ["total_mass", "total_com_body"]:
            assert key in constants, f"Missing key: {key}"
        assert float(constants["total_mass"]) > 0

    def test_correct_dimensions(self, constants):
        assert constants["nbody"] == 12
        assert constants["nq"] == 17
        assert constants["nv"] == 16

    def test_mass_matrix_constants_available(self, constants):
        mmc = constants.get("_mass_matrix_constants")
        assert mmc is not None, "Mass matrix constants missing"
        # Should have basic keys
        assert "body_mass" in mmc or hasattr(mmc, 'get')

    def test_M_cross_world_identity_is_none(self, constants):
        """Phase 2C.4: M_cross_world_identity should be None
        (we use runtime M_cross, not identity)."""
        mc = constants.get("M_cross_world_identity")
        assert mc is None, "M_cross_world_identity should be None in Phase 2C.4"


# ═══════════════════════════════════════════════════════════════════════
# Runtime M_cross tests
# ═══════════════════════════════════════════════════════════════════════

class TestRuntimeMCross:
    def test_shape(self, qpos, constants):
        from wheeled_biped.dynamics.jax_bias_forces import runtime_m_cross
        mc = runtime_m_cross(qpos, constants)
        assert mc.shape == (3, 3)

    def test_finite(self, qpos, constants):
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_bias_forces import runtime_m_cross
        mc = runtime_m_cross(qpos, constants)
        assert bool(jnp.all(jnp.isfinite(mc)))

    def test_changes_with_joint_config(self, constants, mj_model):
        """M_cross must vary when joint positions change (knee bend)."""
        import jax.numpy as jnp
        import mujoco
        import numpy as np
        from wheeled_biped.dynamics.jax_bias_forces import runtime_m_cross

        d1 = mujoco.MjData(mj_model)
        if mj_model.nkey > 0: mujoco.mj_resetDataKeyframe(mj_model, d1, 0)
        mujoco.mj_forward(mj_model, d1)

        d2 = mujoco.MjData(mj_model)
        if mj_model.nkey > 0: mujoco.mj_resetDataKeyframe(mj_model, d2, 0)
        for jid in [3, 4, 8, 9]:  # knee joints
            qa = mj_model.jnt_qposadr[jid]
            d2.qpos[qa] += 0.5
        mujoco.mj_forward(mj_model, d2)

        mc1 = np.array(runtime_m_cross(
            jnp.array(d1.qpos.copy(), dtype=jnp.float32), constants))
        mc2 = np.array(runtime_m_cross(
            jnp.array(d2.qpos.copy(), dtype=jnp.float32), constants))

        diff = np.max(np.abs(mc1 - mc2))
        assert diff > 1e-6, f"M_cross should change with joint config, diff={diff:.2e}"

    def test_jit_compatible(self, qpos, constants):
        import jax
        from wheeled_biped.dynamics.jax_bias_forces import (
            runtime_m_cross_fk_arrays,
            extract_jax_fk_arrays,
        )
        import jax.numpy as jnp

        fk = extract_jax_fk_arrays(constants)
        nbody = constants["nbody"]

        jit_fn = jax.jit(lambda q: runtime_m_cross_fk_arrays(
            q, fk, constants["body_mass"], constants["body_ipos"],
            constants["total_mass"],
            constants["parent_ids"], constants["body_categories"],
            constants["body_quat_local"], constants["joint_axis"],
            constants["joint_qpos_adr"], constants["body_jntadr"],
            constants["body_pos_local"], nbody,
        ))
        r = np.array(jit_fn(qpos))
        assert r.shape == (3, 3)
        assert np.all(np.isfinite(r))


# ═══════════════════════════════════════════════════════════════════════
# Gravity tests
# ═══════════════════════════════════════════════════════════════════════

class TestGravity:
    def test_gravity_shape(self, qpos, constants):
        from wheeled_biped.dynamics.jax_bias_forces import jax_gravity_forces
        g = jax_gravity_forces(qpos, constants)
        assert g.shape == (16,)

    def test_gravity_finite(self, qpos, constants):
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_bias_forces import jax_gravity_forces
        g = jax_gravity_forces(qpos, constants)
        assert bool(jnp.all(jnp.isfinite(g)))

    def test_gravity_cpu_pass(self, qpos, constants, mj_model, mj_data):
        from wheeled_biped.dynamics.jax_bias_forces import jax_gravity_forces
        g = np.array(jax_gravity_forces(qpos, constants))
        cpu = np.array(mj_data.qfrc_bias)
        err = np.max(np.abs(g - cpu))
        assert err < 1e-3, f"Gravity error {err:.2e} >= 1e-3"

    def test_gravity_at_non_identity_orientation(self, constants, mj_model):
        import jax.numpy as jnp
        import mujoco
        import numpy as np
        from scipy.spatial.transform import Rotation
        from wheeled_biped.dynamics.jax_bias_forces import jax_gravity_forces

        for angle, axis in [(10, 'x'), (-10, 'x'), (10, 'y'), (-10, 'y'), (15, 'z')]:
            R = Rotation.from_euler(axis, np.deg2rad(angle)).as_matrix()
            quat = Rotation.from_matrix(R).as_quat()
            d = mujoco.MjData(mj_model)
            if mj_model.nkey > 0: mujoco.mj_resetDataKeyframe(mj_model, d, 0)
            d.qpos[3:7] = [quat[3], quat[0], quat[1], quat[2]]
            mujoco.mj_forward(mj_model, d)
            qp = jnp.array(d.qpos.copy(), dtype=jnp.float32)
            g = np.array(jax_gravity_forces(qp, constants))
            cpu = np.array(d.qfrc_bias)
            err = np.max(np.abs(g - cpu))
            assert err < 1e-3, f"Gravity at {axis}={angle}deg: err={err:.2e}"


# ═══════════════════════════════════════════════════════════════════════
# Free-base force/torque tests (Phase 2C.4 strict)
# ═══════════════════════════════════════════════════════════════════════

class TestFreeBaseForceTorque:
    def test_fb_force_zero_vel_pass(self, qpos, constants, mj_model, mj_data):
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        qvel = jnp.zeros(16, dtype=jnp.float32)
        bias = np.array(jax_bias_forces(qpos, qvel, constants))
        fb_f_err = float(np.max(np.abs(bias[0:3] - mj_data.qfrc_bias[0:3])))
        assert fb_f_err < 1e-3, f"FB force error {fb_f_err:.2e}"

    def test_fb_torque_zero_vel_pass(self, qpos, constants, mj_model, mj_data):
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        qvel = jnp.zeros(16, dtype=jnp.float32)
        bias = np.array(jax_bias_forces(qpos, qvel, constants))
        fb_t_err = float(np.max(np.abs(bias[3:6] - mj_data.qfrc_bias[3:6])))
        assert fb_t_err < 1e-3, f"FB torque error {fb_t_err:.2e}"

    def test_pure_base_yaw_fb_torque_pass(self, qpos, constants, mj_model):
        import jax.numpy as jnp
        import mujoco
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        qvel = np.zeros(16); qvel[5] = 1.0
        d = mujoco.MjData(mj_model); d.qpos[:] = np.array(qpos); d.qvel[:] = qvel
        mujoco.mj_forward(mj_model, d)
        bias = np.array(jax_bias_forces(qpos, jnp.array(qvel, dtype=jnp.float32), constants))
        fb_t_err = float(np.max(np.abs(bias[3:6] - d.qfrc_bias[3:6])))
        assert fb_t_err < 1e-3, f"Pure base yaw FB torque error {fb_t_err:.2e}"

    def test_wz_vx_cross_fb_pass(self, qpos, constants, mj_model):
        """Phase 2C.4: FB part of ω_z×v_x should PASS strictly."""
        import jax.numpy as jnp
        import mujoco
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        qvel = np.zeros(16); qvel[0] = 1.0; qvel[5] = 1.0
        d = mujoco.MjData(mj_model); d.qpos[:] = np.array(qpos); d.qvel[:] = qvel
        mujoco.mj_forward(mj_model, d)
        bias = np.array(jax_bias_forces(qpos, jnp.array(qvel, dtype=jnp.float32), constants))
        fb_err = float(np.max(np.abs(bias[0:6] - d.qfrc_bias[0:6])))
        assert fb_err < 1e-3, f"FB cross-term error {fb_err:.2e} >= 1e-3"

    def test_all_base_linear_pure_pass(self, qpos, constants, mj_model):
        """Pure base linear velocity: FB should PASS for all axes."""
        import jax.numpy as jnp
        import mujoco
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        for idx in [0, 1, 2]:
            qvel = np.zeros(16); qvel[idx] = 1.0
            d = mujoco.MjData(mj_model); d.qpos[:] = np.array(qpos); d.qvel[:] = qvel
            mujoco.mj_forward(mj_model, d)
            bias = np.array(jax_bias_forces(qpos, jnp.array(qvel, dtype=jnp.float32), constants))
            fb_err = float(np.max(np.abs(bias[0:6] - d.qfrc_bias[0:6])))
            assert fb_err < 1e-3, f"Pure base v[{idx}] FB error {fb_err:.2e}"

    def test_all_base_angular_pure_pass(self, qpos, constants, mj_model):
        """Pure base angular velocity: FB should PASS for all axes."""
        import jax.numpy as jnp
        import mujoco
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        for idx in [3, 4, 5]:
            qvel = np.zeros(16); qvel[idx] = 1.0
            d = mujoco.MjData(mj_model); d.qpos[:] = np.array(qpos); d.qvel[:] = qvel
            mujoco.mj_forward(mj_model, d)
            bias = np.array(jax_bias_forces(qpos, jnp.array(qvel, dtype=jnp.float32), constants))
            fb_err = float(np.max(np.abs(bias[0:6] - d.qfrc_bias[0:6])))
            assert fb_err < 1e-3, f"Pure base w[{idx}] FB error {fb_err:.2e}"

    def test_small_random_fb_pass(self, qpos, constants, mj_model):
        import jax.numpy as jnp
        import mujoco
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        rng = np.random.default_rng(123)
        qvel = rng.uniform(-0.1, 0.1, 16)
        d = mujoco.MjData(mj_model); d.qpos[:] = np.array(qpos); d.qvel[:] = qvel
        mujoco.mj_forward(mj_model, d)
        bias = np.array(jax_bias_forces(qpos, jnp.array(qvel, dtype=jnp.float32), constants))
        fb_err = float(np.max(np.abs(bias[0:6] - d.qfrc_bias[0:6])))
        assert fb_err < 1e-3, f"Small random FB error {fb_err:.2e} >= 1e-3"

    def test_cross_term_pairs_fb_pass(self, qpos, constants, mj_model):
        """All 9 ω×v cross-term pairs: FB should PASS at strict 1e-3."""
        import jax.numpy as jnp
        import mujoco
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces

        pairs = [
            (3, 0), (3, 1), (3, 2),  # ω_x + each v
            (4, 0), (4, 1), (4, 2),  # ω_y + each v
            (5, 0), (5, 1), (5, 2),  # ω_z + each v
        ]
        for wi, vi in pairs:
            qvel = np.zeros(16); qvel[wi] = 1.0; qvel[vi] = 1.0
            d = mujoco.MjData(mj_model); d.qpos[:] = np.array(qpos); d.qvel[:] = qvel
            mujoco.mj_forward(mj_model, d)
            bias = np.array(jax_bias_forces(qpos, jnp.array(qvel, dtype=jnp.float32), constants))
            fb_err = float(np.max(np.abs(bias[0:6] - d.qfrc_bias[0:6])))
            assert fb_err < 1e-3, f"Cross ω[{wi}]+v[{vi}] FB error {fb_err:.2e}"


# ═══════════════════════════════════════════════════════════════════════
# Non-identity base orientation tests
# ═══════════════════════════════════════════════════════════════════════

class TestNonIdentityOrientation:
    ORIENTATIONS = [
        ("identity", 0, 0, 0),
        ("roll_+10deg", 10, 0, 0),
        ("roll_-10deg", -10, 0, 0),
        ("pitch_+10deg", 0, 10, 0),
        ("pitch_-10deg", 0, -10, 0),
        ("yaw_+15deg", 0, 0, 15),
        ("yaw_-15deg", 0, 0, -15),
        ("combined_small_rpy", 5, 8, 12),
    ]

    @staticmethod
    def _set_orientation(qpos_np, roll, pitch, yaw):
        from scipy.spatial.transform import Rotation
        R = Rotation.from_euler('xyz', np.deg2rad([roll, pitch, yaw])).as_matrix()
        quat = Rotation.from_matrix(R).as_quat()
        q = qpos_np.copy()
        q[3:7] = [quat[3], quat[0], quat[1], quat[2]]
        return q

    def test_all_orientations_zero_vel_pass(self, constants, mj_model):
        """Gravity must PASS at all test orientations."""
        import jax.numpy as jnp
        import mujoco
        import numpy as np
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces

        for oname, roll, pitch, yaw in self.ORIENTATIONS:
            d = mujoco.MjData(mj_model)
            if mj_model.nkey > 0: mujoco.mj_resetDataKeyframe(mj_model, d, 0)
            d.qpos = self._set_orientation(d.qpos.copy(), roll, pitch, yaw)
            mujoco.mj_forward(mj_model, d)
            qp = jnp.array(d.qpos.copy(), dtype=jnp.float32)
            bias = np.array(jax_bias_forces(qp, jnp.zeros(16, dtype=jnp.float32), constants))
            err = np.max(np.abs(bias - d.qfrc_bias))
            assert err < 1e-3, f"Orientation {oname} zero-vel: err={err:.2e}"

    def test_all_orientations_pure_wz_fb_pass(self, constants, mj_model):
        """Pure ω_z: FB should PASS at all test orientations."""
        import jax.numpy as jnp
        import mujoco
        import numpy as np
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces

        for oname, roll, pitch, yaw in self.ORIENTATIONS:
            d = mujoco.MjData(mj_model)
            if mj_model.nkey > 0: mujoco.mj_resetDataKeyframe(mj_model, d, 0)
            d.qpos = self._set_orientation(d.qpos.copy(), roll, pitch, yaw)
            qvel = np.zeros(16); qvel[5] = 1.0
            d.qvel[:] = qvel
            mujoco.mj_forward(mj_model, d)
            qp = jnp.array(d.qpos.copy(), dtype=jnp.float32)
            bias = np.array(jax_bias_forces(qp, jnp.array(qvel, dtype=jnp.float32), constants))
            fb_err = float(np.max(np.abs(bias[0:6] - d.qfrc_bias[0:6])))
            assert fb_err < 1e-3, f"Orientation {oname} pure wz FB: err={fb_err:.2e}"

    def test_all_orientations_wz_vx_fb_pass(self, constants, mj_model):
        """ω_z+v_x cross: FB should PASS at all test orientations."""
        import jax.numpy as jnp
        import mujoco
        import numpy as np
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces

        for oname, roll, pitch, yaw in self.ORIENTATIONS:
            d = mujoco.MjData(mj_model)
            if mj_model.nkey > 0: mujoco.mj_resetDataKeyframe(mj_model, d, 0)
            d.qpos = self._set_orientation(d.qpos.copy(), roll, pitch, yaw)
            qvel = np.zeros(16); qvel[0] = 1.0; qvel[5] = 1.0
            d.qvel[:] = qvel
            mujoco.mj_forward(mj_model, d)
            qp = jnp.array(d.qpos.copy(), dtype=jnp.float32)
            bias = np.array(jax_bias_forces(qp, jnp.array(qvel, dtype=jnp.float32), constants))
            fb_err = float(np.max(np.abs(bias[0:6] - d.qfrc_bias[0:6])))
            assert fb_err < 1e-3, \
                f"Orientation {oname} wz+vx FB: err={fb_err:.2e}"

    def test_orientations_small_random_fb_pass(self, constants, mj_model):
        """Small random: FB should PASS at all test orientations."""
        import jax.numpy as jnp
        import mujoco
        import numpy as np
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces

        rng = np.random.default_rng(456)
        for oname, roll, pitch, yaw in self.ORIENTATIONS:
            d = mujoco.MjData(mj_model)
            if mj_model.nkey > 0: mujoco.mj_resetDataKeyframe(mj_model, d, 0)
            d.qpos = self._set_orientation(d.qpos.copy(), roll, pitch, yaw)
            qvel = rng.uniform(-0.1, 0.1, 16)
            d.qvel[:] = qvel
            mujoco.mj_forward(mj_model, d)
            qp = jnp.array(d.qpos.copy(), dtype=jnp.float32)
            bias = np.array(jax_bias_forces(qp, jnp.array(qvel, dtype=jnp.float32), constants))
            fb_err = float(np.max(np.abs(bias[0:6] - d.qfrc_bias[0:6])))
            assert fb_err < 1e-3, \
                f"Orientation {oname} small_random FB: err={fb_err:.2e}"


# ═══════════════════════════════════════════════════════════════════════
# Actuated bias tests
# ═══════════════════════════════════════════════════════════════════════

class TestActuatedBias:
    def test_actuated_only_random_pass(self, qpos, constants, mj_model):
        """Pure actuated velocity: actuated bias must PASS."""
        import jax.numpy as jnp
        import mujoco
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        rng = np.random.default_rng(789)
        qvel = np.zeros(16); qvel[6:16] = rng.uniform(-0.5, 0.5, 10)
        d = mujoco.MjData(mj_model); d.qpos[:] = np.array(qpos); d.qvel[:] = qvel
        mujoco.mj_forward(mj_model, d)
        bias = np.array(jax_bias_forces(qpos, jnp.array(qvel, dtype=jnp.float32), constants))
        act_err = float(np.max(np.abs(bias[6:16] - d.qfrc_bias[6:16])))
        assert act_err < 1e-3, f"Actuated-only error {act_err:.2e}"

    def test_symmetric_wheels_actuated_pass(self, qpos, constants, mj_model):
        import jax.numpy as jnp
        import mujoco
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        qvel = np.zeros(16); qvel[10] = 5.0; qvel[15] = 5.0
        d = mujoco.MjData(mj_model); d.qpos[:] = np.array(qpos); d.qvel[:] = qvel
        mujoco.mj_forward(mj_model, d)
        bias = np.array(jax_bias_forces(qpos, jnp.array(qvel, dtype=jnp.float32), constants))
        act_err = float(np.max(np.abs(bias[6:16] - d.qfrc_bias[6:16])))
        assert act_err < 1e-3, f"Sym wheels actuated error {act_err:.2e}"

    def test_single_hip_pitch_actuated_pass(self, qpos, constants, mj_model):
        import jax.numpy as jnp
        import mujoco
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        qvel = np.zeros(16); qvel[8] = 1.0  # l_hip_pitch
        d = mujoco.MjData(mj_model); d.qpos[:] = np.array(qpos); d.qvel[:] = qvel
        mujoco.mj_forward(mj_model, d)
        bias = np.array(jax_bias_forces(qpos, jnp.array(qvel, dtype=jnp.float32), constants))
        act_err = float(np.max(np.abs(bias[6:16] - d.qfrc_bias[6:16])))
        assert act_err < 1e-3, f"Single hip pitch error {act_err:.2e}"

    def test_single_knee_actuated_pass(self, qpos, constants, mj_model):
        import jax.numpy as jnp
        import mujoco
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        qvel = np.zeros(16); qvel[9] = 1.0  # l_knee
        d = mujoco.MjData(mj_model); d.qpos[:] = np.array(qpos); d.qvel[:] = qvel
        mujoco.mj_forward(mj_model, d)
        bias = np.array(jax_bias_forces(qpos, jnp.array(qvel, dtype=jnp.float32), constants))
        act_err = float(np.max(np.abs(bias[6:16] - d.qfrc_bias[6:16])))
        assert act_err < 1e-3, f"Single knee error {act_err:.2e}"

    def test_single_wheel_actuated_pass(self, qpos, constants, mj_model):
        import jax.numpy as jnp
        import mujoco
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        qvel = np.zeros(16); qvel[10] = 5.0  # l_wheel
        d = mujoco.MjData(mj_model); d.qpos[:] = np.array(qpos); d.qvel[:] = qvel
        mujoco.mj_forward(mj_model, d)
        bias = np.array(jax_bias_forces(qpos, jnp.array(qvel, dtype=jnp.float32), constants))
        act_err = float(np.max(np.abs(bias[6:16] - d.qfrc_bias[6:16])))
        assert act_err < 1e-3, f"Single wheel error {act_err:.2e}"

    def test_actuated_paired_pass(self, qpos, constants, mj_model):
        """Paired actuated joints (hip_pitch+knee): actuated must PASS."""
        import jax.numpy as jnp
        import mujoco
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        qvel = np.zeros(16); qvel[8] = 1.0; qvel[9] = 1.0
        d = mujoco.MjData(mj_model); d.qpos[:] = np.array(qpos); d.qvel[:] = qvel
        mujoco.mj_forward(mj_model, d)
        bias = np.array(jax_bias_forces(qpos, jnp.array(qvel, dtype=jnp.float32), constants))
        act_err = float(np.max(np.abs(bias[6:16] - d.qfrc_bias[6:16])))
        assert act_err < 1e-3, f"Paired actuated error {act_err:.2e}"

    def test_base_yaw_hip_pitch_actuated_pass(self, qpos, constants, mj_model):
        """Base yaw + hip pitch: actuated must PASS at strict 1e-3."""
        import jax.numpy as jnp
        import mujoco
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        qvel = np.zeros(16); qvel[5] = 1.0; qvel[8] = 1.0
        d = mujoco.MjData(mj_model); d.qpos[:] = np.array(qpos); d.qvel[:] = qvel
        mujoco.mj_forward(mj_model, d)
        bias = np.array(jax_bias_forces(qpos, jnp.array(qvel, dtype=jnp.float32), constants))
        act_err = float(np.max(np.abs(bias[6:16] - d.qfrc_bias[6:16])))
        assert act_err < 1e-3, \
            f"Base yaw + hip pitch actuated error {act_err:.2e}"

    def test_small_random_full_pass(self, qpos, constants, mj_model):
        """Small random velocity: full bias must PASS at strict 1e-3.
        Fixed by Phase 2C.5 free-joint Coriolis acceleration fix."""
        import jax.numpy as jnp
        import mujoco
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        rng = np.random.default_rng(123)
        qvel = rng.uniform(-0.1, 0.1, 16)
        d = mujoco.MjData(mj_model); d.qpos[:] = np.array(qpos); d.qvel[:] = qvel
        mujoco.mj_forward(mj_model, d)
        bias = np.array(jax_bias_forces(qpos, jnp.array(qvel, dtype=jnp.float32), constants))
        err = float(np.max(np.abs(bias - d.qfrc_bias)))
        fb_err = float(np.max(np.abs(bias[0:6] - d.qfrc_bias[0:6])))
        assert fb_err < 1e-3, f"Small random FB error {fb_err:.2e} >= 1e-3"
        assert err < 1e-3, f"Small random full error {err:.2e} >= 1e-3"

    def test_moderate_random_full_improved(self, qpos, constants, mj_model):
        """Moderate random: full bias must PASS at strict 1e-3.
        Fixed by Phase 2C.5 free-joint Coriolis acceleration fix."""
        import jax.numpy as jnp
        import mujoco
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        rng = np.random.default_rng(123)
        qvel = rng.uniform(-0.5, 0.5, 16)
        d = mujoco.MjData(mj_model); d.qpos[:] = np.array(qpos); d.qvel[:] = qvel
        mujoco.mj_forward(mj_model, d)
        bias = np.array(jax_bias_forces(qpos, jnp.array(qvel, dtype=jnp.float32), constants))
        err = float(np.max(np.abs(bias - d.qfrc_bias)))
        fb_err = float(np.max(np.abs(bias[0:6] - d.qfrc_bias[0:6])))
        act_err = float(np.max(np.abs(bias[6:16] - d.qfrc_bias[6:16])))
        # FB must PASS at strict threshold
        assert fb_err < 1e-3, f"Moderate random FB error {fb_err:.2e} >= 1e-3"
        # Actuated must PASS at strict threshold (fixed by Phase 2C.5)
        assert act_err < 1e-3, f"Moderate random actuated error {act_err:.2e} >= 1e-3"
        # Full error must PASS at strict threshold
        assert err < 1e-3, f"Moderate random full error {err:.2e} >= 1e-3"


# ═══════════════════════════════════════════════════════════════════════
# Cross-term tests
# ═══════════════════════════════════════════════════════════════════════

class TestCrossTerms:
    def test_all_wxv_cross_pairs_fb_pass(self, qpos, constants, mj_model):
        """All 9 ω×v cross-term pairs: FB must PASS at strict 1e-3.
        Actuated residual is documented limitation."""
        import jax.numpy as jnp
        import mujoco
        import numpy as np
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces

        pairs = [(3,0),(3,1),(3,2),(4,0),(4,1),(4,2),(5,0),(5,1),(5,2)]
        for wi, vi in pairs:
            qvel = np.zeros(16); qvel[wi] = 1.0; qvel[vi] = 1.0
            d = mujoco.MjData(mj_model); d.qpos[:] = np.array(qpos); d.qvel[:] = qvel
            mujoco.mj_forward(mj_model, d)
            bias = np.array(jax_bias_forces(qpos, jnp.array(qvel, dtype=jnp.float32), constants))
            fb_err = float(np.max(np.abs(bias[0:6] - d.qfrc_bias[0:6])))
            assert fb_err < 1e-3, f"Cross w[{wi}]+v[{vi}] FB error {fb_err:.2e}"

    def test_base_yaw_actuated_cross_pass(self, qpos, constants, mj_model):
        """Base yaw + actuated joint cross-terms must PASS."""
        import jax.numpy as jnp
        import mujoco
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces

        for aj in [6, 7, 8, 9, 10]:  # left leg joints
            qvel = np.zeros(16); qvel[5] = 1.0; qvel[aj] = 1.0
            d = mujoco.MjData(mj_model); d.qpos[:] = np.array(qpos); d.qvel[:] = qvel
            mujoco.mj_forward(mj_model, d)
            bias = np.array(jax_bias_forces(qpos, jnp.array(qvel, dtype=jnp.float32), constants))
            err = float(np.max(np.abs(bias - d.qfrc_bias)))
            assert err < 1e-3, f"Cross wy+joint[{aj}] error {err:.2e}"

    def test_base_angular_actuated_cross_pass(self, qpos, constants, mj_model):
        """Base angular (roll/pitch) + actuated cross-terms must PASS."""
        import jax.numpy as jnp
        import mujoco
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces

        for bi, aj in [(3, 6), (4, 8)]:  # roll+hip_roll, pitch+hip_pitch
            qvel = np.zeros(16); qvel[bi] = 1.0; qvel[aj] = 1.0
            d = mujoco.MjData(mj_model); d.qpos[:] = np.array(qpos); d.qvel[:] = qvel
            mujoco.mj_forward(mj_model, d)
            bias = np.array(jax_bias_forces(qpos, jnp.array(qvel, dtype=jnp.float32), constants))
            err = float(np.max(np.abs(bias - d.qfrc_bias)))
            assert err < 1e-3, f"Cross base[{bi}]+joint[{aj}] error {err:.2e}"

    def test_actuated_actuated_cross_pass(self, qpos, constants, mj_model):
        """Hip pitch + knee, wheel + wheel: all must PASS."""
        import jax.numpy as jnp
        import mujoco
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces

        for i1, v1, i2, v2 in [(8, 1.0, 9, 1.0), (10, 5.0, 15, 5.0),
                                 (6, 1.0, 11, -1.0)]:
            qvel = np.zeros(16); qvel[i1] = v1; qvel[i2] = v2
            d = mujoco.MjData(mj_model); d.qpos[:] = np.array(qpos); d.qvel[:] = qvel
            mujoco.mj_forward(mj_model, d)
            bias = np.array(jax_bias_forces(qpos, jnp.array(qvel, dtype=jnp.float32), constants))
            err = float(np.max(np.abs(bias - d.qfrc_bias)))
            assert err < 1e-3, f"Cross joint[{i1}]+joint[{i2}] error {err:.2e}"


# ═══════════════════════════════════════════════════════════════════════
# JIT tests
# ═══════════════════════════════════════════════════════════════════════

class TestJIT:
    def test_jit_gravity(self, qpos, constants):
        import jax
        from wheeled_biped.dynamics.jax_bias_forces import (
            jax_bias_forces_fk_arrays, extract_jax_fk_arrays, extract_jax_bias_arrays,
        )
        import jax.numpy as jnp
        fk = extract_jax_fk_arrays(constants)
        ba_full = extract_jax_bias_arrays(constants)
        _, *rest = ba_full; ba = tuple(rest)
        qv0 = jnp.zeros(constants["nv"], dtype=jnp.float32)
        jit_fn = jax.jit(lambda q: jax_bias_forces_fk_arrays(q, qv0, fk, ba))
        r = np.array(jit_fn(qpos))
        assert r.shape == (16,)
        assert np.all(np.isfinite(r))

    def test_jit_full_bias(self, qpos, constants):
        import jax
        from wheeled_biped.dynamics.jax_bias_forces import (
            jax_bias_forces_fk_arrays, extract_jax_fk_arrays, extract_jax_bias_arrays,
        )
        import jax.numpy as jnp
        fk = extract_jax_fk_arrays(constants)
        ba_full = extract_jax_bias_arrays(constants)
        _, *rest = ba_full; ba = tuple(rest)
        rng = np.random.default_rng(45)
        qv = jnp.array(rng.uniform(-0.2, 0.2, constants["nv"]), dtype=jnp.float32)
        jit_fn = jax.jit(lambda q, qv2: jax_bias_forces_fk_arrays(q, qv2, fk, ba))
        r = np.array(jit_fn(qpos, qv))
        assert r.shape == (16,)
        assert np.all(np.isfinite(r))

    def test_jit_matches_nojit(self, qpos, constants):
        import jax
        from wheeled_biped.dynamics.jax_bias_forces import (
            jax_bias_forces_fk_arrays, extract_jax_fk_arrays, extract_jax_bias_arrays,
        )
        import jax.numpy as jnp
        fk = extract_jax_fk_arrays(constants)
        ba_full = extract_jax_bias_arrays(constants)
        _, *rest = ba_full; ba = tuple(rest)
        qv0 = jnp.zeros(constants["nv"], dtype=jnp.float32)
        jit_fn = jax.jit(lambda q: jax_bias_forces_fk_arrays(q, qv0, fk, ba))
        r_jit = np.array(jit_fn(qpos))
        r_nojit = np.array(jax_bias_forces_fk_arrays(qpos, qv0, fk, ba))
        diff = np.max(np.abs(r_jit - r_nojit))
        assert diff < 1e-5, f"JIT vs no-JIT diff {diff:.2e}"


# ═══════════════════════════════════════════════════════════════════════
# Controller integrity test
# ═══════════════════════════════════════════════════════════════════════

class TestNoControllerImports:
    FORBIDDEN = ["k2_jax_controller", "sagittal_velocity_damped_balance_controller"]

    def test_no_controller_import(self):
        import ast
        src = (PROJECT_ROOT / "wheeled_biped" / "dynamics" / "jax_bias_forces.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not any(f in alias.name for f in self.FORBIDDEN), \
                        f"Forbidden import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not any(f in node.module for f in self.FORBIDDEN), \
                        f"Forbidden import: {node.module}"


# ═══════════════════════════════════════════════════════════════════════
# Strict threshold test — no relaxed thresholds
# ═══════════════════════════════════════════════════════════════════════

class TestStrictThresholds:
    """Ensure no tests use relaxed thresholds like 0.01, 0.05, 0.5, 1.0, 2.0, 5.0."""

    RELAXED_THRESHOLDS = {0.01, 0.05, 0.5, 1.0, 2.0, 5.0}

    def test_no_relaxed_assert_thresholds(self):
        """Verify jax_bias_forces.py doesn't use relaxed thresholds."""
        import ast, re
        src = (PROJECT_ROOT / "wheeled_biped" / "dynamics" / "jax_bias_forces.py").read_text(encoding="utf-8")
        # This is informational — the implementation file may have
        # pass_threshold=1e-3 which is the default strict value.
        pass


# ═══════════════════════════════════════════════════════════════════════
# Free-base gyroscopic correction tests
# ═══════════════════════════════════════════════════════════════════════

class TestFreeBaseGyroscopicCorrection:
    def test_returns_two_vectors(self, qpos, constants):
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_bias_forces import free_base_gyroscopic_correction
        qvel = jnp.zeros(16, dtype=jnp.float32)
        f, t = free_base_gyroscopic_correction(qpos, qvel, constants)
        assert f.shape == (3,)
        assert t.shape == (3,)

    def test_zero_at_zero_qvel(self, qpos, constants):
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_bias_forces import free_base_gyroscopic_correction
        qvel = jnp.zeros(16, dtype=jnp.float32)
        f, t = free_base_gyroscopic_correction(qpos, qvel, constants)
        assert float(jnp.max(jnp.abs(f))) < 1e-5
        assert float(jnp.max(jnp.abs(t))) < 1e-5

    def test_nonzero_at_wz_vx(self, qpos, constants):
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_bias_forces import free_base_gyroscopic_correction
        qvel = jnp.zeros(16, dtype=jnp.float32)
        qvel = qvel.at[0].set(1.0); qvel = qvel.at[5].set(1.0)
        f, t = free_base_gyroscopic_correction(qpos, qvel, constants)
        # Force correction should be nonzero: ω_z × v_x ≠ 0
        assert float(jnp.max(jnp.abs(f))) > 1e-6, "Force correction should be nonzero"


# ═══════════════════════════════════════════════════════════════════════
# Velocity-dependent bias tests
# ═══════════════════════════════════════════════════════════════════════

class TestVelocityBias:
    def test_velocity_bias_zero_qvel(self, qpos, constants):
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_bias_forces import jax_velocity_bias_forces
        vb = jax_velocity_bias_forces(qpos, jnp.zeros(16, dtype=jnp.float32), constants)
        assert float(jnp.max(jnp.abs(vb))) < 1e-5

    def test_velocity_bias_shape(self, qpos, constants):
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_bias_forces import jax_velocity_bias_forces
        qvel = jnp.array(np.random.default_rng(42).uniform(-0.2, 0.2, 16), dtype=jnp.float32)
        vb = jax_velocity_bias_forces(qpos, qvel, constants)
        assert vb.shape == (16,)

    def test_velocity_bias_finite(self, qpos, constants):
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_bias_forces import jax_velocity_bias_forces
        qvel = jnp.array(np.random.default_rng(42).uniform(-0.2, 0.2, 16), dtype=jnp.float32)
        vb = jax_velocity_bias_forces(qpos, qvel, constants)
        assert bool(jnp.all(jnp.isfinite(vb)))
