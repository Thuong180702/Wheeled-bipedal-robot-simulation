"""Tests for Phase 3D.3-E JAX Dynamics Cache."""
import pytest
import numpy as np
import jax
import jax.numpy as jnp

from wheeled_biped.wbc.phase3d3e_jax_dynamics_cache import (
    JAXDynamicsCache,
    initialize_jax_dynamics_cache,
    DEFAULT_MAX_CONTACTS,
)


@pytest.fixture(scope="module")
def test_model_and_constants():
    """Load model and build constants once for all cache tests."""
    from wheeled_biped.utils.config import get_model_path
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(get_model_path()))

    from wheeled_biped.wbc.offline_qp_wbc import build_qp_wbc_constants
    constants = build_qp_wbc_constants(model)

    return model, constants


@pytest.fixture(scope="module")
def cached(test_model_and_constants):
    """Build a warmed-up cache once per test module."""
    model, constants = test_model_and_constants
    return initialize_jax_dynamics_cache(model, constants, warmup=True)


class TestJAXDynamicsCacheInit:
    """Tests for cache initialization."""

    def test_cache_initializes(self, cached):
        assert cached.initialized
        assert cached.max_contacts == DEFAULT_MAX_CONTACTS

    def test_cache_warmup_records_compile_time(self, cached):
        assert cached.compile_time_s > 0
        assert cached.warmup_time_s >= 0

    def test_cache_records_environment(self, cached):
        assert cached.jax_platform != ""
        assert cached.jax_backend != ""

    def test_mass_matrix_jit_compiled(self, cached):
        assert cached.mass_matrix_jit is not None

        qpos = jnp.zeros(17, dtype=jnp.float32)
        M = cached.mass_matrix_jit(qpos)
        assert M.shape == (16, 16)
        assert np.all(np.isfinite(np.array(M)))

    def test_bias_forces_jit_compiled(self, cached):
        assert cached.bias_forces_jit is not None

        qpos = jnp.zeros(17, dtype=jnp.float32)
        qvel = jnp.zeros(16, dtype=jnp.float32)
        h = cached.bias_forces_jit(qpos, qvel)
        assert h.shape == (16,)
        assert np.all(np.isfinite(np.array(h)))

    def test_com_jacobian_jit_compiled(self, cached):
        assert cached.com_jacobian_jit is not None

        qpos = jnp.zeros(17, dtype=jnp.float32)
        Jcom_qpos = cached.com_jacobian_jit(qpos)
        # Returns qpos-space Jacobian (3, 17)
        assert Jcom_qpos.shape == (3, 17)
        assert np.all(np.isfinite(np.array(Jcom_qpos)))

    def test_torso_ang_vel_jacobian_jit_compiled(self, cached):
        assert cached.torso_ang_vel_jacobian_jit is not None

        qpos = jnp.zeros(17, dtype=jnp.float32)
        Jquat_qpos = cached.torso_ang_vel_jacobian_jit(qpos)
        # Returns torso quaternion qpos-space Jacobian (4, 17)
        assert Jquat_qpos.shape == (4, 17)
        assert np.all(np.isfinite(np.array(Jquat_qpos)))

    def test_torso_orientation_error_jit_compiled(self, cached):
        assert cached.torso_orientation_error_jit is not None

        qpos = jnp.zeros(17, dtype=jnp.float32)
        result = cached.torso_orientation_error_jit(qpos)
        assert "e_R" in result
        assert result["e_R"].shape == (3,)
        assert result["current_rpy"].shape == (3,)

    def test_fk_arrays_are_tuples(self, cached):
        assert isinstance(cached.fk_arrays, tuple)
        assert isinstance(cached.mm_arrays, tuple)
        assert isinstance(cached.bias_arrays, tuple)
        assert len(cached.fk_arrays) > 0


class TestMassMatrixBiasForcesCorrectness:
    """Verify jitted M and h match the original non-jitted functions."""

    def _get_default_qpos_qvel(self, test_model_and_constants):
        model, _ = test_model_and_constants
        # Try "standing" first (K2 model), fall back to "default" if absent
        try:
            kf = model.keyframe("standing")
        except Exception:
            kf = model.keyframe("default")
        qpos0 = np.array(kf.qpos, dtype=np.float64)
        qvel0 = np.zeros(model.nv, dtype=np.float64)
        return qpos0, qvel0

    def _get_orig_constants(self, test_model_and_constants):
        """Ensure dynamics constants are built for original function comparisons."""
        _, constants = test_model_and_constants
        from wheeled_biped.wbc.offline_qp_wbc import _ensure_dynamics_constants
        _ensure_dynamics_constants(constants)
        return constants

    def test_mass_matrix_matches_original_default_pose(self, cached, test_model_and_constants):
        qpos0, _ = self._get_default_qpos_qvel(test_model_and_constants)
        constants = self._get_orig_constants(test_model_and_constants)

        from wheeled_biped.dynamics.jax_mass_matrix import jax_mass_matrix

        # Original
        M_orig = np.array(
            jax_mass_matrix(
                jnp.array(qpos0, dtype=jnp.float32),
                constants["_mass_matrix_constants"],
            ),
            dtype=np.float64,
        )
        # Cached (jitted)
        M_cache = np.array(
            cached.mass_matrix_jit(jnp.array(qpos0, dtype=jnp.float32)),
            dtype=np.float64,
        )

        max_diff = np.max(np.abs(M_orig - M_cache))
        assert max_diff < 1e-6, f"Mass matrix max diff: {max_diff}"
        assert M_orig.shape == M_cache.shape

    def test_bias_forces_matches_original_default_pose(self, cached, test_model_and_constants):
        qpos0, qvel0 = self._get_default_qpos_qvel(test_model_and_constants)
        constants = self._get_orig_constants(test_model_and_constants)

        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces

        # Original
        h_orig = np.array(
            jax_bias_forces(
                jnp.array(qpos0, dtype=jnp.float32),
                jnp.array(qvel0, dtype=jnp.float32),
                constants["_dynamics_constants"],
            ),
            dtype=np.float64,
        )
        # Cached (jitted)
        h_cache = np.array(
            cached.bias_forces_jit(
                jnp.array(qpos0, dtype=jnp.float32),
                jnp.array(qvel0, dtype=jnp.float32),
            ),
            dtype=np.float64,
        )

        max_diff = np.max(np.abs(h_orig - h_cache))
        assert max_diff < 1e-6, f"Bias forces max diff: {max_diff}"
        assert h_orig.shape == h_cache.shape

    def test_mass_matrix_matches_original_perturbed_pose(self, cached, test_model_and_constants):
        qpos0, _ = self._get_default_qpos_qvel(test_model_and_constants)
        constants = self._get_orig_constants(test_model_and_constants)

        from wheeled_biped.dynamics.jax_mass_matrix import jax_mass_matrix

        rng = np.random.RandomState(42)
        for trial in range(5):
            qpos_p = qpos0.copy()
            qpos_p[7:17] += rng.randn(10) * 0.05

            M_orig = np.array(
                jax_mass_matrix(
                    jnp.array(qpos_p, dtype=jnp.float32),
                    constants["_mass_matrix_constants"],
                ),
                dtype=np.float64,
            )
            M_cache = np.array(
                cached.mass_matrix_jit(jnp.array(qpos_p, dtype=jnp.float32)),
                dtype=np.float64,
            )
            max_diff = np.max(np.abs(M_orig - M_cache))
            assert max_diff < 1e-6, f"Trial {trial}: mass matrix max diff: {max_diff}"

    def test_bias_forces_matches_original_nonzero_qvel(self, cached, test_model_and_constants):
        qpos0, _ = self._get_default_qpos_qvel(test_model_and_constants)
        constants = self._get_orig_constants(test_model_and_constants)

        from wheeled_biped.dynamics.jax_bias_forces import jax_bias_forces

        rng = np.random.RandomState(42)
        for trial in range(5):
            qvel_p = rng.randn(16) * 0.1

            h_orig = np.array(
                jax_bias_forces(
                    jnp.array(qpos0, dtype=jnp.float32),
                    jnp.array(qvel_p, dtype=jnp.float32),
                    constants["_dynamics_constants"],
                ),
                dtype=np.float64,
            )
            h_cache = np.array(
                cached.bias_forces_jit(
                    jnp.array(qpos0, dtype=jnp.float32),
                    jnp.array(qvel_p, dtype=jnp.float32),
                ),
                dtype=np.float64,
            )
            max_diff = np.max(np.abs(h_orig - h_cache))
            assert max_diff < 1e-6, f"Trial {trial}: bias forces max diff: {max_diff}"

    def test_no_recompilation_on_same_shape(self, cached, test_model_and_constants):
        """Verify that repeated calls with different values but same shape
        do not trigger reinit."""
        qpos0, _ = self._get_default_qpos_qvel(test_model_and_constants)

        # Call 10 times with slightly different qpos
        rng = np.random.RandomState(99)
        for _ in range(10):
            qpos_p = qpos0.copy()
            qpos_p[7:17] += rng.randn(10) * 0.001
            _ = cached.mass_matrix_jit(jnp.array(qpos_p, dtype=jnp.float32))

        # No reinit should have been triggered
        assert cached.recompile_count == 0
        assert cached.fallback_count == 0


class TestCOMTorsoJdotQdotCorrectness:
    """Verify jitted COM and torso Jdot*qdot functions match originals."""

    @pytest.fixture(scope="module")
    def cache_and_refs(self, test_model_and_constants):
        model, constants = test_model_and_constants
        cache = initialize_jax_dynamics_cache(model, constants, warmup=True)
        return cache, constants

    def _get_default_qpos_qvel(self, test_model_and_constants):
        model, _ = test_model_and_constants
        try:
            kf = model.keyframe("standing")
        except Exception:
            kf = model.keyframe("default")
        qpos0 = np.array(kf.qpos, dtype=np.float64)
        qvel0 = np.zeros(model.nv, dtype=np.float64)
        return qpos0, qvel0

    def test_com_jdot_qdot_compiled(self, cache_and_refs):
        """Verify com_jdot_qdot_jit is compiled and returns correct shape."""
        cache, _ = cache_and_refs
        assert cache.com_jdot_qdot_jit is not None

        qpos = jnp.zeros(17, dtype=jnp.float32)
        qvel = jnp.zeros(16, dtype=jnp.float32)
        result = cache.com_jdot_qdot_jit(qpos, qvel)
        assert result.shape == (3,), f"Expected (3,), got {result.shape}"

    def test_torso_jdotw_qdot_compiled(self, cache_and_refs):
        """Verify torso_jdotw_qdot_jit is compiled and returns correct shape."""
        cache, _ = cache_and_refs
        assert cache.torso_jdotw_qdot_jit is not None

        qpos = jnp.zeros(17, dtype=jnp.float32)
        qvel = jnp.zeros(16, dtype=jnp.float32)
        result = cache.torso_jdotw_qdot_jit(qpos, qvel)
        assert result.shape == (4,), f"Expected (4,), got {result.shape}"

    def test_com_jdot_qdot_matches_original(self, cache_and_refs, test_model_and_constants):
        """COM Jdot*qdot agreement with original (float32 FD tolerance).

        The jitted cache uses float32 precision for qpos integration
        (_integrate_qpos_jax), while the original uses float64.
        This ~1e-7 qpos difference propagates through the finite-
        difference: error ~ 1e-7 / (2*1e-5) ~ 5e-3.  A 1e-2 tolerance
        accounts for the worst-case float32 FD noise.
        """
        cache, constants = cache_and_refs
        qpos0, _ = self._get_default_qpos_qvel(test_model_and_constants)

        from wheeled_biped.wbc.offline_task_stack import compute_com_jdot_qdot

        rng = np.random.RandomState(42)
        qvel_p = rng.randn(16) * 0.1
        kc = constants["_kinematics_constants"]

        jdq_orig = compute_com_jdot_qdot(qpos0, qvel_p, kc)  # (3,)
        jdq_cache = np.array(
            cache.com_jdot_qdot_jit(
                jnp.array(qpos0, dtype=jnp.float32),
                jnp.array(qvel_p, dtype=jnp.float32),
            ),
            dtype=np.float64,
        )
        assert jdq_cache.shape == (3,), f"Expected (3,), got {jdq_cache.shape}"

        max_diff = np.max(np.abs(jdq_orig - jdq_cache))
        assert max_diff < 1e-2, f"COM Jdot*qdot max diff: {max_diff}"

    def test_torso_jdotw_qdot_matches_original(self, cache_and_refs, test_model_and_constants):
        cache, constants = cache_and_refs
        qpos0, _ = self._get_default_qpos_qvel(test_model_and_constants)

        from wheeled_biped.wbc.offline_task_stack import compute_torso_jdotw_qdot

        rng = np.random.RandomState(42)
        qvel_p = rng.randn(16) * 0.1
        kc = constants["_kinematics_constants"]

        jdw_orig = compute_torso_jdotw_qdot(qpos0, qvel_p, kc)  # (3,) — angular acceleration
        jdw_cache_quat = np.array(
            cache.torso_jdotw_qdot_jit(
                jnp.array(qpos0, dtype=jnp.float32),
                jnp.array(qvel_p, dtype=jnp.float32),
            ),
            dtype=np.float64,
        )  # (4,) — quaternion-space
        assert jdw_cache_quat.shape == (4,), f"Expected (4,), got {jdw_cache_quat.shape}"

        # Convert quat-space to angular: alpha = 2 * G(q)^T @ jdw_quat
        q_torso = qpos0[3:7]
        w, x, y, z = q_torso[0], q_torso[1], q_torso[2], q_torso[3]
        G = np.array([[-x, -y, -z], [w, -z, y], [z, w, -x], [-y, x, w]], dtype=np.float64)
        jdw_cache_ang = 2.0 * G.T @ jdw_cache_quat  # (3,)

        max_diff = np.max(np.abs(jdw_orig - jdw_cache_ang))
        assert max_diff < 1e-6, f"Torso Jdotw*qdot max diff: {max_diff}"

    def test_com_jdot_qdot_multiple_poses(self, cache_and_refs, test_model_and_constants):
        """Verify COM Jdot*qdot agrees across 5 random poses and velocities.

        Uses float32 FD tolerance (1e-2) — see
        test_com_jdot_qdot_matches_original for rationale.
        """
        cache, constants = cache_and_refs
        qpos0, _ = self._get_default_qpos_qvel(test_model_and_constants)

        from wheeled_biped.wbc.offline_task_stack import compute_com_jdot_qdot
        kc = constants["_kinematics_constants"]

        rng = np.random.RandomState(123)
        for trial in range(5):
            qpos_p = qpos0.copy()
            qpos_p[7:17] += rng.randn(10) * 0.05
            qvel_p = rng.randn(16) * 0.2

            jdq_orig = compute_com_jdot_qdot(qpos_p, qvel_p, kc)
            jdq_cache = np.array(
                cache.com_jdot_qdot_jit(
                    jnp.array(qpos_p, dtype=jnp.float32),
                    jnp.array(qvel_p, dtype=jnp.float32),
                ),
                dtype=np.float64,
            )
            max_diff = np.max(np.abs(jdq_orig - jdq_cache))
            assert max_diff < 1e-2, f"Trial {trial}: COM Jdot*qdot max diff: {max_diff}"
