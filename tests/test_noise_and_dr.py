"""
Tests for Issue #2 (DR range widening) and Issue #3 (sensor noise).

Covers:
  - balance.yaml DR ranges are widened to target values
  - sensor_noise config loads correctly into WheeledBipedEnv
  - _extract_obs() adds noise when enabled + rng provided
  - noise is deterministic under fixed seed
  - noise respects channel masking (prev_action unchanged)
  - noise is zero when disabled
  - DR model randomization changes model parameters per episode
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wheeled_biped.utils.config import get_model_path, load_yaml  # noqa: E402

# ---------------------------------------------------------------------------
# Shared MuJoCo fixtures (same pattern as test_sim_helpers.py)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mj_model():
    import mujoco

    return mujoco.MjModel.from_xml_path(str(get_model_path()))


@pytest.fixture(scope="module")
def fake_mjx_data(mj_model):
    """Minimal MJX data in standing pose."""
    import mujoco
    from mujoco import mjx

    mj_data = mujoco.MjData(mj_model)
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mujoco.mj_forward(mj_model, mj_data)
    return mjx.put_data(mj_model, mj_data)


# ---------------------------------------------------------------------------
# Issue #2 — DR config ranges
# ---------------------------------------------------------------------------


class TestDRConfigRanges:
    """balance.yaml DR ranges should match the widened targets from the audit fix."""

    def _load_dr(self):
        config = load_yaml("configs/training/balance.yaml")
        return config["domain_randomization"]

    def test_mass_range_widened(self):
        dr = self._load_dr()
        lo, hi = dr["mass_range"]
        assert lo <= 0.85, f"mass_range[0]={lo} should be ≤ 0.85 (was 0.9)"
        assert hi >= 1.15, f"mass_range[1]={hi} should be ≥ 1.15 (was 1.1)"

    def test_friction_range_widened(self):
        dr = self._load_dr()
        lo, hi = dr["friction_range"]
        assert lo <= 0.6, f"friction_range[0]={lo} should be ≤ 0.6 (was 0.8)"
        assert hi >= 1.4, f"friction_range[1]={hi} should be ≥ 1.4 (was 1.2)"

    def test_damping_range_widened(self):
        dr = self._load_dr()
        lo, hi = dr["joint_damping_range"]
        assert lo <= 0.7, f"joint_damping_range[0]={lo} should be ≤ 0.7 (was 0.8)"
        assert hi >= 1.3, f"joint_damping_range[1]={hi} should be ≥ 1.3 (was 1.2)"

    def test_sensor_noise_section_present(self):
        config = load_yaml("configs/training/balance.yaml")
        assert "sensor_noise" in config, "balance.yaml must have a sensor_noise section"
        sn = config["sensor_noise"]
        for key in ("enabled", "ang_vel_std", "gravity_std", "joint_pos_std", "joint_vel_std"):
            assert key in sn, f"sensor_noise.{key} missing from balance.yaml"


# ---------------------------------------------------------------------------
# Issue #3 — Sensor noise config loading
# ---------------------------------------------------------------------------


class TestSensorNoiseConfig:
    """WheeledBipedEnv must load sensor_noise parameters from config."""

    def _make_env(self, noise_cfg=None):
        from wheeled_biped.envs.base_env import WheeledBipedEnv

        config = {}
        if noise_cfg is not None:
            config["sensor_noise"] = noise_cfg
        return WheeledBipedEnv(config=config)

    def test_noise_disabled_by_default(self):
        """No sensor_noise section → noise disabled."""
        env = self._make_env()
        assert env._noise_enabled is False

    def test_noise_params_loaded(self):
        """Explicit config → all std values stored correctly."""
        env = self._make_env(
            {
                "enabled": True,
                "ang_vel_std": 0.05,
                "gravity_std": 0.02,
                "joint_pos_std": 0.005,
                "joint_vel_std": 0.01,
            }
        )
        assert env._noise_enabled is True
        assert env._noise_ang_vel_std == pytest.approx(0.05)
        assert env._noise_gravity_std == pytest.approx(0.02)
        assert env._noise_joint_pos_std == pytest.approx(0.005)
        assert env._noise_joint_vel_std == pytest.approx(0.01)

    def test_noise_disabled_flag_respected(self):
        """enabled: false overrides non-zero std values."""
        env = self._make_env(
            {
                "enabled": False,
                "ang_vel_std": 1.0,
                "gravity_std": 1.0,
            }
        )
        assert env._noise_enabled is False


# ---------------------------------------------------------------------------
# Issue #3 — Sensor noise observation behavior
# ---------------------------------------------------------------------------


class TestSensorNoiseObservation:
    """_extract_obs() must add noise only when enabled and rng is provided."""

    NUM_JOINTS = 10

    def _make_noisy_env(self):
        from wheeled_biped.envs.base_env import WheeledBipedEnv

        return WheeledBipedEnv(
            config={
                "sensor_noise": {
                    "enabled": True,
                    "ang_vel_std": 0.1,
                    "gravity_std": 0.05,
                    "joint_pos_std": 0.01,
                    "joint_vel_std": 0.02,
                }
            }
        )

    def _make_clean_env(self):
        from wheeled_biped.envs.base_env import WheeledBipedEnv

        return WheeledBipedEnv(config={"sensor_noise": {"enabled": False}})

    def test_noise_changes_obs_when_enabled(self, fake_mjx_data):
        """Noisy obs (with rng + enabled) differs from clean obs (no rng)."""
        env = self._make_noisy_env()
        prev_action = jnp.zeros(self.NUM_JOINTS)
        rng = jax.random.PRNGKey(42)

        obs_noisy = env._extract_obs(fake_mjx_data, prev_action, rng)
        obs_clean = env._extract_obs(fake_mjx_data, prev_action)

        assert not np.allclose(np.array(obs_noisy), np.array(obs_clean)), (
            "Noise-enabled obs with rng must differ from no-rng obs"
        )

    def test_noise_deterministic_under_fixed_seed(self, fake_mjx_data):
        """Same rng key → identical noisy obs."""
        env = self._make_noisy_env()
        prev_action = jnp.zeros(self.NUM_JOINTS)
        rng = jax.random.PRNGKey(99)

        obs1 = env._extract_obs(fake_mjx_data, prev_action, rng)
        obs2 = env._extract_obs(fake_mjx_data, prev_action, rng)

        assert np.allclose(np.array(obs1), np.array(obs2)), (
            "Same RNG key must produce identical noisy obs (determinism)"
        )

    def test_different_seeds_produce_different_noise(self, fake_mjx_data):
        """Different rng keys → different noisy obs."""
        env = self._make_noisy_env()
        prev_action = jnp.zeros(self.NUM_JOINTS)

        obs1 = env._extract_obs(fake_mjx_data, prev_action, jax.random.PRNGKey(0))
        obs2 = env._extract_obs(fake_mjx_data, prev_action, jax.random.PRNGKey(1))

        assert not np.allclose(np.array(obs1), np.array(obs2)), (
            "Different RNG keys should produce different noise"
        )

    def test_prev_action_not_corrupted_by_noise(self, fake_mjx_data):
        """Last 10 dims (prev_action) must be identical between noisy and clean obs.

        Obs layout: gravity_body[0:3], lin_vel[3:6], ang_vel[6:9],
                    joint_pos[9:19], joint_vel[19:29], prev_action[29:39].
        Only gravity_body, ang_vel, joint_pos, joint_vel are noised.
        """
        env = self._make_noisy_env()
        sentinel = jnp.array([0.1 * (i + 1) for i in range(self.NUM_JOINTS)], dtype=jnp.float32)
        rng = jax.random.PRNGKey(7)

        obs_noisy = env._extract_obs(fake_mjx_data, sentinel, rng)
        obs_clean = env._extract_obs(fake_mjx_data, sentinel)

        # prev_action occupies last NUM_JOINTS dims
        assert np.allclose(
            np.array(obs_noisy[-self.NUM_JOINTS :]),
            np.array(obs_clean[-self.NUM_JOINTS :]),
        ), "prev_action channel must not be affected by obs noise"

    def test_noise_disabled_obs_unchanged(self, fake_mjx_data):
        """noise_enabled=False → obs identical with or without rng."""
        env = self._make_clean_env()
        prev_action = jnp.zeros(self.NUM_JOINTS)
        rng = jax.random.PRNGKey(7)

        obs_with_rng = env._extract_obs(fake_mjx_data, prev_action, rng)
        obs_no_rng = env._extract_obs(fake_mjx_data, prev_action)

        assert np.allclose(np.array(obs_with_rng), np.array(obs_no_rng)), (
            "noise_enabled=False must produce same obs regardless of rng"
        )

    def test_noised_channels_are_ang_vel_gravity_joints(self, fake_mjx_data):
        """Noise affects only gravity_body[0:3], ang_vel[6:9], joint_pos[9:19],
        joint_vel[19:29].  base_lin_vel[3:6] and prev_action[29:39] stay clean.
        """
        env = self._make_noisy_env()
        # Use a known prev_action and compare element-wise
        prev_action = jnp.zeros(self.NUM_JOINTS)
        rng = jax.random.PRNGKey(13)

        obs_noisy = np.array(env._extract_obs(fake_mjx_data, prev_action, rng))
        obs_clean = np.array(env._extract_obs(fake_mjx_data, prev_action))

        diff = np.abs(obs_noisy - obs_clean)

        # lin_vel [3:6] must be zero diff (not noised)
        assert np.allclose(diff[3:6], 0.0), (
            "base_lin_vel[3:6] should not be noised (not a direct sensor)"
        )
        # prev_action [29:39] must be zero diff
        assert np.allclose(diff[29:39], 0.0), "prev_action[29:39] should not be noised"
        # At least one of gravity[0:3] or ang_vel[6:9] should differ
        assert diff[0:3].max() > 0.0 or diff[6:9].max() > 0.0, (
            "gravity_body or ang_vel should have non-zero noise"
        )


# ---------------------------------------------------------------------------
# Issue #2 — per-episode DR model randomization
# ---------------------------------------------------------------------------


class TestDRModelRandomization:
    """randomize_mjx_model() must produce different model params per episode."""

    def test_different_keys_produce_different_mass(self):
        """Two calls with different rng keys yield different body_mass arrays."""
        import mujoco
        from mujoco import mjx

        from wheeled_biped.sim.domain_randomization import randomize_mjx_model

        mj_model = mujoco.MjModel.from_xml_path(str(get_model_path()))
        base_mjx = mjx.put_model(mj_model)
        dr_config = {
            "mass_range": [0.85, 1.15],
            "friction_range": [0.6, 1.4],
            "joint_damping_range": [0.7, 1.3],
        }

        model_a, _ = randomize_mjx_model(base_mjx, jax.random.PRNGKey(0), dr_config)
        model_b, _ = randomize_mjx_model(base_mjx, jax.random.PRNGKey(1), dr_config)

        assert not np.allclose(np.array(model_a.body_mass), np.array(model_b.body_mass)), (
            "Different rng keys must yield different mass distributions"
        )

    def test_same_key_deterministic(self):
        """Same rng key yields identical randomized model."""
        import mujoco
        from mujoco import mjx

        from wheeled_biped.sim.domain_randomization import randomize_mjx_model

        mj_model = mujoco.MjModel.from_xml_path(str(get_model_path()))
        base_mjx = mjx.put_model(mj_model)
        dr_config = {
            "mass_range": [0.85, 1.15],
            "friction_range": [0.6, 1.4],
            "joint_damping_range": [0.7, 1.3],
        }

        model_a, _ = randomize_mjx_model(base_mjx, jax.random.PRNGKey(42), dr_config)
        model_b, _ = randomize_mjx_model(base_mjx, jax.random.PRNGKey(42), dr_config)

        assert np.allclose(np.array(model_a.body_mass), np.array(model_b.body_mass)), (
            "Same rng key must produce identical model (determinism)"
        )

    def test_mass_within_specified_range(self):
        """Randomized mass must stay within the configured [lo, hi] * base_mass range."""
        import mujoco
        from mujoco import mjx

        from wheeled_biped.sim.domain_randomization import randomize_mjx_model

        mj_model = mujoco.MjModel.from_xml_path(str(get_model_path()))
        base_mjx = mjx.put_model(mj_model)
        base_mass = np.array(base_mjx.body_mass)
        dr_config = {"mass_range": [0.85, 1.15]}

        for i in range(10):
            model_r, _ = randomize_mjx_model(base_mjx, jax.random.PRNGKey(i), dr_config)
            rand_mass = np.array(model_r.body_mass)
            scales = rand_mass / np.where(base_mass > 0, base_mass, 1.0)
            valid = (base_mass == 0) | ((scales >= 0.85 - 1e-5) & (scales <= 1.15 + 1e-5))
            assert np.all(valid), (
                f"Seed {i}: some mass scales outside [0.85, 1.15]: {scales[~valid]}"
            )

    def test_balance_env_carries_dr_model_in_state(self):
        """BalanceEnv.reset() must store dr_mjx_model in state.info."""
        from wheeled_biped.envs.balance_env import BalanceEnv

        config = {
            "task": {"episode_length": 10, "initial_min_height": 0.68},
            "domain_randomization": {
                "enabled": True,
                "mass_range": [0.85, 1.15],
                "friction_range": [0.6, 1.4],
                "joint_damping_range": [0.7, 1.3],
                "push_magnitude": 0,
                "push_interval": 9999,
                "push_duration": 1,
            },
            "low_level_pid": {"enabled": False},
            "curriculum": {"enabled": False},
        }
        env = BalanceEnv(config=config)
        state = env.reset(jax.random.PRNGKey(0))

        assert "dr_mjx_model" in state.info, "state.info must contain dr_mjx_model after reset()"
        assert "noise_rng" in state.info, "state.info must contain noise_rng after reset()"

    def test_balance_env_dr_differs_between_episodes(self):
        """Two resets with different keys produce different dr_mjx_model mass."""
        from wheeled_biped.envs.balance_env import BalanceEnv

        config = {
            "task": {"episode_length": 10, "initial_min_height": 0.68},
            "domain_randomization": {
                "enabled": True,
                "mass_range": [0.85, 1.15],
                "friction_range": [0.6, 1.4],
                "joint_damping_range": [0.7, 1.3],
                "push_magnitude": 0,
                "push_interval": 9999,
                "push_duration": 1,
            },
            "low_level_pid": {"enabled": False},
            "curriculum": {"enabled": False},
        }
        env = BalanceEnv(config=config)
        state_a = env.reset(jax.random.PRNGKey(0))
        state_b = env.reset(jax.random.PRNGKey(1))

        mass_a = np.array(state_a.info["dr_mjx_model"].body_mass)
        mass_b = np.array(state_b.info["dr_mjx_model"].body_mass)

        assert not np.allclose(mass_a, mass_b), (
            "Different reset keys must produce different DR model mass"
        )
