"""Tests for Phase 3D.3-E JAX Dynamics Cache."""
import pytest
import numpy as np
import jax
import jax.numpy as jnp

from wheeled_biped.wbc.phase3d3e_jax_dynamics_cache import (
    JAXDynamicsCache,
    initialize_jax_dynamics_cache,
    contacts_to_padded_arrays,
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


class TestContactJacobianCorrectness:
    """Verify jitted contact Jacobian and Jdot*qdot match originals."""

    @pytest.fixture(scope="module")
    def cache_and_contacts(self, test_model_and_constants):
        model, constants = test_model_and_constants
        import mujoco
        cache = initialize_jax_dynamics_cache(model, constants, warmup=True)

        # Extract contacts at default pose
        try:
            kf = model.keyframe("standing")
        except Exception:
            kf = model.keyframe("default")
        qpos0 = np.array(kf.qpos, dtype=np.float64)
        data = mujoco.MjData(model)
        data.qpos[:] = qpos0
        mujoco.mj_forward(model, data)
        contact_c = constants["_contact_constants"]
        wheel_body_ids = contact_c.get("wheel_body_ids", {})
        wheel_ids_set = set(int(v) for v in wheel_body_ids.values() if v >= 0)
        contacts = []
        for contact_id in range(data.ncon):
            c = data.contact[contact_id]
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
            wheel_body = b1 if b1 in wheel_ids_set else (b2 if b2 in wheel_ids_set else None)
            if wheel_body is None:
                continue
            pos = np.array(c.pos, dtype=np.float64)
            frame = np.array(c.frame, dtype=np.float64).reshape(3, 3)
            body_xpos = np.array(data.xpos[wheel_body], dtype=np.float64)
            body_xmat = np.array(data.xmat[wheel_body], dtype=np.float64).reshape(3, 3)
            local_point = body_xmat.T @ (pos - body_xpos)
            contacts.append({
                "body_id": int(wheel_body), "position": pos,
                "frame": frame, "local_point": local_point,
            })
        return cache, constants, model, contacts, qpos0

    def test_contact_jacobian_single_matches_original(self, cache_and_contacts):
        cache, constants, model, contacts, qpos0 = cache_and_contacts
        import jax.numpy as jnp
        from wheeled_biped.dynamics.jax_contact_dynamics import contact_point_translational_jacobian

        qpos_jax = jnp.array(qpos0, dtype=jnp.float32)
        contact_c = constants["_contact_constants"]

        for i, c in enumerate(contacts[:4]):  # test all contacts
            bid = int(c["body_id"])
            lp = jnp.array(c["local_point"], dtype=jnp.float32)

            Jp_orig = np.array(
                contact_point_translational_jacobian(qpos_jax, bid, lp, contact_c),
                dtype=np.float64,
            )
            Jp_cache = np.array(
                cache._contact_jacobian_single_jit(qpos_jax, bid, lp),
                dtype=np.float64,
            )

            max_diff = np.max(np.abs(Jp_orig - Jp_cache))
            assert max_diff < 1e-6, f"Contact {i}: Jp max diff: {max_diff}"
            assert Jp_orig.shape == (3, 16)

    def test_contact_jdot_qdot_single_matches_original(self, cache_and_contacts):
        cache, constants, model, contacts, qpos0 = cache_and_contacts
        import jax.numpy as jnp
        from wheeled_biped.wbc.offline_qp_wbc import compute_contact_jdot_qdot

        contact_c = constants["_contact_constants"]
        rng = np.random.RandomState(42)
        qvel_p = rng.randn(16) * 0.1

        # Original: computes all contacts at once
        jdq_all_orig = compute_contact_jdot_qdot(qpos0, qvel_p, contacts, contact_c)

        qpos_jax = jnp.array(qpos0, dtype=jnp.float32)
        qvel_jax = jnp.array(qvel_p, dtype=jnp.float32)

        for i, c in enumerate(contacts[:4]):
            bid = int(c["body_id"])
            lp = jnp.array(c["local_point"], dtype=jnp.float32)

            jdq_cache = np.array(
                cache._contact_jdot_qdot_single_jit(qpos_jax, qvel_jax, bid, lp),
                dtype=np.float64,
            )
            jdq_orig = jdq_all_orig[3*i:3*i+3]

            max_diff = np.max(np.abs(jdq_orig - jdq_cache))
            # Float32 FD noise floor. This is a PRECISION test, not a correctness
            # test: both methods compute the same J̇q̇, but the cache path does a
            # central FD of the analytic contact Jacobian with eps=1e-5 in float32,
            # which suffers catastrophic cancellation (~1e-2). The float64 FD-of-
            # Jacobian agrees with the position-double-FD `compute_contact_jdot_qdot`
            # to <1e-3, confirming both are correct. The floor rose from ~5e-3 to
            # ~1e-2 once the contact Jacobian's actuated (leg) columns became
            # nonzero — before the F1 fix they were identically zero, so those
            # columns contributed no cancellation noise (and the WBC dynamics model
            # was wrong). The production WBC path uses the float64 `compute_contact_
            # jdot_qdot`, not this float32 fallback.
            assert max_diff < 2e-2, f"Contact {i}: Jdot*qdot max diff: {max_diff}"

    def test_padded_contact_array_shape(self):
        """Verify contacts_to_padded_arrays produces correct shapes."""
        # Empty contacts
        empty = contacts_to_padded_arrays([], max_contacts=4)
        assert empty["active"].shape == (4,)
        assert empty["body_id"].shape == (4,)
        assert empty["local_point"].shape == (4, 3)
        assert empty["frame"].shape == (4, 3, 3)
        assert empty["num_contacts"] == 0
        assert np.all(empty["active"] == 0)

        # 2 contacts
        contacts_2 = [
            {"body_id": 5, "local_point": [1.0, 2.0, 3.0], "frame": np.eye(3), "position": [0.0, 0.0, 0.0]},
            {"body_id": 8, "local_point": [4.0, 5.0, 6.0], "frame": np.eye(3), "position": [1.0, 1.0, 1.0]},
        ]
        padded = contacts_to_padded_arrays(contacts_2, max_contacts=4)
        assert padded["num_contacts"] == 2
        assert np.all(padded["active"][:2] == 1)
        assert np.all(padded["active"][2:] == 0)

    def test_too_many_contacts_raises(self):
        """Verify ValueError when contact count exceeds max_contacts."""
        contacts_5 = [
            {"body_id": i, "local_point": [0,0,0], "frame": np.eye(3), "position": [0,0,0]}
            for i in range(5)
        ]
        with pytest.raises(ValueError, match="exceeds max_contacts"):
            contacts_to_padded_arrays(contacts_5, max_contacts=4)


class TestFullCachedSnapshot:
    """Verify prepare_phase3b_snapshot_cached matches original."""

    @pytest.fixture(scope="module")
    def cache_and_refs(self, test_model_and_constants):
        model, constants = test_model_and_constants
        import mujoco
        cache = initialize_jax_dynamics_cache(model, constants, warmup=True)

        try:
            kf = model.keyframe("standing")
        except Exception:
            kf = model.keyframe("default")
        qpos0 = np.array(kf.qpos, dtype=np.float64)
        qvel0 = np.zeros(model.nv, dtype=np.float64)

        # Contact extraction (same pattern as other tests)
        data = mujoco.MjData(model)
        data.qpos[:] = qpos0
        mujoco.mj_forward(model, data)
        contact_c = constants["_contact_constants"]
        wids = set(int(v) for v in contact_c.get("wheel_body_ids", {}).values() if v >= 0)
        contacts = []
        for ci in range(data.ncon):
            c = data.contact[ci]
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
            wb = b1 if b1 in wids else (b2 if b2 in wids else None)
            if wb is None:
                continue
            pos = np.array(c.pos, dtype=np.float64)
            fr = np.array(c.frame, dtype=np.float64).reshape(3, 3)
            bx = np.array(data.xpos[wb], dtype=np.float64)
            bm = np.array(data.xmat[wb], dtype=np.float64).reshape(3, 3)
            lp = bm.T @ (pos - bx)
            contacts.append({
                "body_id": int(wb), "position": pos,
                "frame": fr, "local_point": lp,
            })
        return cache, constants, model, qpos0, qvel0, contacts

    def test_full_snapshot_matches_original(self, cache_and_refs):
        cache, constants, model, qpos0, qvel0, contacts = cache_and_refs

        from wheeled_biped.wbc.phase3b_cached_stack import prepare_phase3b_snapshot
        from wheeled_biped.wbc.phase3d3e_jax_dynamics_cache import prepare_phase3b_snapshot_cached

        snap_orig = prepare_phase3b_snapshot("test", qpos0, qvel0, contacts, constants)
        snap_cache = prepare_phase3b_snapshot_cached(
            cache, "test", qpos0, qvel0, contacts, constants,
        )

        # Compare M and h (exact match expected — same jax_mass_matrix_fk_arrays)
        assert np.max(np.abs(snap_orig.M - snap_cache.M)) < 1e-6, "M mismatch"
        assert np.max(np.abs(snap_orig.h - snap_cache.h)) < 1e-6, "h mismatch"

        # Compare COM Jacobian
        assert np.max(np.abs(snap_orig.Jcom - snap_cache.Jcom)) < 1e-6, "Jcom mismatch"

        # Compare COM Jdot*qdot (float32 FD tolerance)
        assert np.max(np.abs(snap_orig.jdq_com - snap_cache.jdq_com)) < 1e-2, \
            f"jdq_com mismatch: {np.max(np.abs(snap_orig.jdq_com - snap_cache.jdq_com))}"

        # Compare torso Jacobian
        assert np.max(np.abs(snap_orig.Jr - snap_cache.Jr)) < 1e-6, "Jr mismatch"

        # Compare orientation error
        assert np.max(np.abs(snap_orig.e_R - snap_cache.e_R)) < 1e-6, "e_R mismatch"
        assert np.max(np.abs(snap_orig.current_rpy - snap_cache.current_rpy)) < 1e-6, "rpy mismatch"

        # Contact count
        assert snap_orig.m == snap_cache.m, f"contact count: {snap_orig.m} vs {snap_cache.m}"

        # Contact Jdot*qdot (float32 FD tolerance)
        if snap_orig.m > 0:
            jdq_orig = snap_orig.jdot_qdot[:3*snap_orig.m]
            jdq_cache = snap_cache.jdot_qdot[:3*snap_cache.m]
            max_jdq_diff = np.max(np.abs(jdq_orig - jdq_cache))
            assert max_jdq_diff < 1e-2, f"contact jdot_qdot mismatch: {max_jdq_diff}"

        # Mass info
        assert abs(snap_orig.total_mass - snap_cache.total_mass) < 1e-10
        assert abs(snap_orig.robot_weight - snap_cache.robot_weight) < 1e-10
