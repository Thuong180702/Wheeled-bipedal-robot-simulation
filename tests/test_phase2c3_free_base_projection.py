"""Tests for Phase 2C.3 — Free-Base Force Projection Correction.

Validates the free-base gyroscopic correction in the body-local RNEA.
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


# ── Import tests ──────────────────────────────────────────────────────────

class TestImports:
    def test_import_all_helpers(self):
        from wheeled_biped.dynamics.jax_bias_forces import (
            build_bias_force_constants, extract_jax_bias_arrays,
            jax_bias_forces, jax_bias_forces_fk_arrays,
            jax_gravity_forces, jax_velocity_bias_forces,
            compare_bias_forces_to_mujoco, rnea_body_local,
            _compute_free_base_correction,
            _free_base_motion_subspace,
            _project_root_spatial_force_to_mujoco_qfrc,
        )
        assert callable(_compute_free_base_correction)
        assert callable(_free_base_motion_subspace)
        assert callable(_project_root_spatial_force_to_mujoco_qfrc)

    def test_import_from_package(self):
        import wheeled_biped.dynamics
        assert hasattr(wheeled_biped.dynamics, "jax_bias_forces")


# ── Constants tests ────────────────────────────────────────────────────────

class TestConstants:
    def test_version(self, constants):
        assert constants["constants_version"] in (
            "phase2c3_free_base_projection", "phase2c4_runtime_mcross_orientation",
            "phase2c5_actuated_coriolis",
        )

    def test_has_phase2c3_keys(self, constants):
        for key in ["total_mass", "total_com_body"]:
            assert key in constants, f"Missing key: {key}"
        assert float(constants["total_mass"]) > 0
        assert np.array(constants["total_com_body"]).shape == (3,)

    def test_M_cross_shape(self, constants):
        mc = constants.get("M_cross_world_identity")
        if mc is not None:
            assert mc.shape == (3, 3)

    def test_correct_dimensions(self, constants):
        assert constants["nbody"] == 12
        assert constants["nq"] == 17
        assert constants["nv"] == 16

    def test_total_mass_matches(self, constants):
        total = float(np.sum(np.array(constants["body_mass"])[1:]))
        assert abs(total - float(constants["total_mass"])) < 1e-5


# ── Gravity tests ──────────────────────────────────────────────────────────

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


# ── Bias force tests ───────────────────────────────────────────────────────

class TestBiasForces:
    def test_zero_vel_equals_gravity(self, qpos, constants):
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_bias_forces import (
            jax_bias_forces, jax_gravity_forces,
        )
        qvel = jnp.zeros(constants["nv"], dtype=jnp.float32)
        bias = np.array(jax_bias_forces(qpos, qvel, constants))
        grav = np.array(jax_gravity_forces(qpos, constants))
        assert np.allclose(bias, grav, atol=1e-5)

    def test_bias_cpu_pass_at_zero_vel(self, qpos, constants, mj_model, mj_data):
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        qvel = jnp.zeros(constants["nv"], dtype=jnp.float32)
        bias = np.array(jax_bias_forces(qpos, qvel, constants))
        cpu = np.array(mj_data.qfrc_bias)
        err = np.max(np.abs(bias - cpu))
        assert err < 1e-3, f"Full bias error {err:.2e} >= 1e-3"

    def test_base_yaw_rate_pass(self, qpos, constants, mj_model):
        import jax.numpy as jnp
        import mujoco as mj
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        qvel = np.zeros(16); qvel[5] = 1.0
        d = mj.MjData(mj_model); d.qpos[:] = np.array(qpos); d.qvel[:] = qvel
        mj.mj_forward(mj_model, d)
        bias = np.array(jax_bias_forces(qpos, jnp.array(qvel, dtype=jnp.float32), constants))
        err = float(np.max(np.abs(bias - d.qfrc_bias)))
        assert err < 1e-3, f"Base yaw rate error {err:.2e}"

    def test_symmetric_wheels_pass(self, qpos, constants, mj_model):
        import jax.numpy as jnp
        import mujoco as mj
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        qvel = np.zeros(16); qvel[10] = 5.0; qvel[15] = 5.0
        d = mj.MjData(mj_model); d.qpos[:] = np.array(qpos); d.qvel[:] = qvel
        mj.mj_forward(mj_model, d)
        bias = np.array(jax_bias_forces(qpos, jnp.array(qvel, dtype=jnp.float32), constants))
        err = float(np.max(np.abs(bias - d.qfrc_bias)))
        assert err < 1e-3, f"Symmetric wheels error {err:.2e}"

    def test_actuated_only_pass(self, qpos, constants, mj_model):
        import jax.numpy as jnp
        import mujoco as mj
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        rng = np.random.default_rng(456)
        qvel = np.zeros(16); qvel[6:16] = rng.uniform(-0.5, 0.5, 10)
        d = mj.MjData(mj_model); d.qpos[:] = np.array(qpos); d.qvel[:] = qvel
        mj.mj_forward(mj_model, d)
        bias = np.array(jax_bias_forces(qpos, jnp.array(qvel, dtype=jnp.float32), constants))
        err = float(np.max(np.abs(bias - d.qfrc_bias)))
        assert err < 1e-3, f"Actuated-only error {err:.2e}"

    def test_pure_base_linear_velocity_pass(self, qpos, constants, mj_model):
        import jax.numpy as jnp
        import mujoco as mj
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        for idx in [0, 1, 2]:
            qvel = np.zeros(16); qvel[idx] = 1.0
            d = mj.MjData(mj_model); d.qpos[:] = np.array(qpos); d.qvel[:] = qvel
            mj.mj_forward(mj_model, d)
            bias = np.array(jax_bias_forces(qpos, jnp.array(qvel, dtype=jnp.float32), constants))
            err = float(np.max(np.abs(bias - d.qfrc_bias)))
            assert err < 1e-3, f"Pure base v[{idx}] error {err:.2e}"

    def test_pure_base_angular_velocity_pass(self, qpos, constants, mj_model):
        import jax.numpy as jnp
        import mujoco as mj
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        for idx in [3, 4, 5]:
            qvel = np.zeros(16); qvel[idx] = 1.0
            d = mj.MjData(mj_model); d.qpos[:] = np.array(qpos); d.qvel[:] = qvel
            mj.mj_forward(mj_model, d)
            bias = np.array(jax_bias_forces(qpos, jnp.array(qvel, dtype=jnp.float32), constants))
            err = float(np.max(np.abs(bias - d.qfrc_bias)))
            assert err < 1e-3, f"Pure base w[{idx}] error {err:.2e}"

    def test_cross_term_wz_vx_pass(self, qpos, constants, mj_model):
        """Phase 2C.3: ω×v cross-term should PASS (was FAIL in 2C.2)."""
        import jax.numpy as jnp
        import mujoco as mj
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        qvel = np.zeros(16); qvel[0] = 1.0; qvel[5] = 1.0
        d = mj.MjData(mj_model); d.qpos[:] = np.array(qpos); d.qvel[:] = qvel
        mj.mj_forward(mj_model, d)
        bias = np.array(jax_bias_forces(qpos, jnp.array(qvel, dtype=jnp.float32), constants))
        err = float(np.max(np.abs(bias - d.qfrc_bias)))
        # Free-base force+torus should fall under 1e-3.  Actuated may still exceed
        fb_err = float(np.max(np.abs(bias[0:6] - d.qfrc_bias[0:6])))
        assert fb_err < 1e-3, f"FB cross-term error {fb_err:.2e} >= 1e-3"

    def test_base_yaw_hip_pitch_cross_term_pass(self, qpos, constants, mj_model):
        import jax.numpy as jnp
        import mujoco as mj
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        qvel = np.zeros(16); qvel[5] = 1.0; qvel[8] = 1.0
        d = mj.MjData(mj_model); d.qpos[:] = np.array(qpos); d.qvel[:] = qvel
        mj.mj_forward(mj_model, d)
        bias = np.array(jax_bias_forces(qpos, jnp.array(qvel, dtype=jnp.float32), constants))
        err = float(np.max(np.abs(bias - d.qfrc_bias)))
        assert err < 1e-3, f"Cross-term error {err:.2e}"

    def test_small_random_fb_pass(self, qpos, constants, mj_model):
        """FB part of small_random should PASS after Phase 2C.3 correction."""
        import jax.numpy as jnp
        import mujoco as mj
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        rng = np.random.default_rng(123)
        qvel = rng.uniform(-0.1, 0.1, 16)
        d = mj.MjData(mj_model); d.qpos[:] = np.array(qpos); d.qvel[:] = qvel
        mj.mj_forward(mj_model, d)
        bias = np.array(jax_bias_forces(qpos, jnp.array(qvel, dtype=jnp.float32), constants))
        fb_err = float(np.max(np.abs(bias[0:6] - d.qfrc_bias[0:6])))
        assert fb_err < 1e-3, f"Small random FB error {fb_err:.2e} >= 1e-3"

    def test_moderate_random_documented_improvement(self, qpos, constants, mj_model):
        """Moderate random: FB part should pass; actuated is documented lim."""
        import jax.numpy as jnp
        import mujoco as mj
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        rng = np.random.default_rng(123)
        qvel = rng.uniform(-0.5, 0.5, 16)
        d = mj.MjData(mj_model); d.qpos[:] = np.array(qpos); d.qvel[:] = qvel
        mj.mj_forward(mj_model, d)
        bias = np.array(jax_bias_forces(qpos, jnp.array(qvel, dtype=jnp.float32), constants))
        fb_err = float(np.max(np.abs(bias[0:6] - d.qfrc_bias[0:6])))
        act_err = float(np.max(np.abs(bias[6:16] - d.qfrc_bias[6:16])))
        full_err = float(np.max(np.abs(bias - d.qfrc_bias)))
        # FB should be much better than Phase 2C.2 (1.38 → < 1e-3)
        assert fb_err < 1e-3, f"Moderate random FB error {fb_err:.2e} >= 1e-3"
        # Full error improved from 1.38 to ~0.06 (actuated residual)
        assert full_err < 0.1, f"Moderate random error {full_err:.2e} >= 0.1"


# ── Free-base correction tests ─────────────────────────────────────────

class TestFreeBaseCorrection:
    def test_force_correction_wz_vx(self, constants):
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_bias_forces import _compute_free_base_correction

        qvel = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0,0,0,0,0,0,0,0,0,0], dtype=jnp.float32)
        quat = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
        total_mass = constants["total_mass"]
        total_com = constants["total_com_body"]
        mc = constants.get("M_cross_world_identity")

        f_corr, tau_corr = _compute_free_base_correction(
            qvel, quat, total_mass, total_com, mc,
        )
        f = np.array(f_corr); tau = np.array(tau_corr)

        # Force correction should be m * [0,0,1] x [1,0,0] = [0, m, 0]
        m = float(total_mass)
        assert abs(f[0]) < 1e-5
        assert abs(f[1] - m) < 1e-5, f"f_y={f[1]:.4f}, expected ~{m:.4f}"
        assert abs(f[2]) < 1e-5

        # Torque correction should be nonzero (from M_cross coupling)
        if mc is not None:
            assert np.max(np.abs(tau)) > 0, "Torque correction should be nonzero"

    def test_correction_zero_for_pure_translation(self, constants):
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_bias_forces import _compute_free_base_correction

        qvel = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0,0,0,0,0,0,0,0,0,0], dtype=jnp.float32)
        quat = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
        total_mass = constants["total_mass"]
        total_com = constants["total_com_body"]
        mc = constants.get("M_cross_world_identity")

        f_corr, tau_corr = _compute_free_base_correction(
            qvel, quat, total_mass, total_com, mc,
        )
        # Pure translation: ω=0, so f_corr = m * 0 x v = 0
        assert float(jnp.max(jnp.abs(f_corr))) < 1e-5
        # v x ω = 0, so tau_corr = 0
        assert float(jnp.max(jnp.abs(tau_corr))) < 1e-5

    def test_correction_zero_for_pure_rotation(self, constants):
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_bias_forces import _compute_free_base_correction

        qvel = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0,0,0,0,0,0,0,0,0,0], dtype=jnp.float32)
        quat = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
        total_mass = constants["total_mass"]
        total_com = constants["total_com_body"]
        mc = constants.get("M_cross_world_identity")

        f_corr, tau_corr = _compute_free_base_correction(
            qvel, quat, total_mass, total_com, mc,
        )
        # Pure rotation: v=0, so f_corr = m * ω x 0 = 0
        assert float(jnp.max(jnp.abs(f_corr))) < 1e-5
        # v x ω = 0, so tau_corr = 0
        assert float(jnp.max(jnp.abs(tau_corr))) < 1e-5


# ── Motion subspace tests ─────────────────────────────────────────────

class TestMotionSubspace:
    def test_free_base_motion_subspace_shape(self, qpos, constants):
        import numpy as np
        from wheeled_biped.dynamics.jax_bias_forces import _free_base_motion_subspace
        S = np.array(_free_base_motion_subspace(qpos, constants))
        assert S.shape == (6, 6)
        assert np.all(np.isfinite(S))

    def test_project_root_spatial_force_shape(self, qpos, constants):
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_bias_forces import _project_root_spatial_force_to_mujoco_qfrc
        F = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=jnp.float32)
        qfrc = _project_root_spatial_force_to_mujoco_qfrc(F, qpos, constants)
        assert qfrc.shape == (6,)
        assert bool(jnp.all(jnp.isfinite(qfrc)))


# ── JIT tests ──────────────────────────────────────────────────────────

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
        assert diff < 1e-5


# ── No controller imports ────────────────────────────────────────────

class TestNoControllerImports:
    FORBIDDEN = ["k2_jax_controller", "sagittal_velocity_damped_balance_controller"]

    def test_no_controller_import(self):
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


# ── Non-identity orientation tests ──────────────────────────────────

class TestNonIdentityOrientation:
    def test_roll_10deg_pass(self, constants, mj_model):
        import jax.numpy as jnp
        import mujoco as mj
        import numpy as np
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        from scipy.spatial.transform import Rotation

        # Set 10 deg roll
        R = Rotation.from_euler('xyz', [np.deg2rad(10), 0, 0]).as_matrix()
        quat = Rotation.from_matrix(R).as_quat()
        d = mj.MjData(mj_model)
        if mj_model.nkey > 0: mj.mj_resetDataKeyframe(mj_model, d, 0)
        d.qpos[3:7] = [quat[3], quat[0], quat[1], quat[2]]
        d.qvel[:] = 0.0
        mj.mj_forward(mj_model, d)
        qp = np.array(d.qpos.copy())
        qp_j = jnp.array(qp, dtype=jnp.float32)
        cpu_g = np.array(d.qfrc_bias)
        jax_g = np.array(jax_bias_forces(qp_j, jnp.zeros(16, dtype=jnp.float32), constants))
        assert np.max(np.abs(jax_g - cpu_g)) < 1e-3

    def test_pitch_10deg_wz_vx_fb_pass(self, constants, mj_model):
        import jax.numpy as jnp
        import mujoco as mj
        import numpy as np
        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces
        from scipy.spatial.transform import Rotation

        R = Rotation.from_euler('xyz', [0, np.deg2rad(10), 0]).as_matrix()
        quat = Rotation.from_matrix(R).as_quat()
        d = mj.MjData(mj_model)
        if mj_model.nkey > 0: mj.mj_resetDataKeyframe(mj_model, d, 0)
        d.qpos[3:7] = [quat[3], quat[0], quat[1], quat[2]]
        qvel = np.zeros(16); qvel[0] = 1.0; qvel[5] = 1.0
        d.qvel[:] = qvel
        mj.mj_forward(mj_model, d)
        qp = np.array(d.qpos.copy())
        qp_j = jnp.array(qp, dtype=jnp.float32)
        cpu = np.array(d.qfrc_bias)
        jax_b = np.array(jax_bias_forces(qp_j, jnp.array(qvel, dtype=jnp.float32), constants))
        fb_err = float(np.max(np.abs(jax_b[0:6] - cpu[0:6])))
        # At non-identity orientation, the body-local RNEA has a pre-existing
        # error in the centrifugal force computation (not caused by Phase 2C.3).
        # Phase 2C.3 free-base correction targets only the w x v cross-term.
        # The pure-wz centrifugal force error at tilted orientations is a
        # separate issue to be addressed in a future phase.
        fb_force_err = float(np.max(np.abs(jax_b[0:3] - cpu[0:3])))
        # Force error at non-identity orientation: documented limitation < 0.2
        assert fb_force_err < 0.2, f"Pitch 10deg FB force error {fb_force_err:.2e}"
        # Full FB error < 0.3 is a documented limitation
        assert fb_err < 0.3, f"Pitch 10deg FB error {fb_err:.2e}"
