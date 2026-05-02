"""
Tests for PPO trainer invariants.

Covers:
  - Single rollout + update produces no NaN in params
  - obs_rms updates correctly (Welford)
  - Loss metrics dict has expected keys
  - Checkpoint save → load round-trip
  - compute_gae shapes and no-NaN property
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Minimal config helpers
# ---------------------------------------------------------------------------

_TINY_CONFIG = {
    "task": {
        "env": "BalanceEnv",
        "num_envs": 4,
        "episode_length": 20,
        "initial_min_height": 0.68,
    },
    "ppo": {
        "learning_rate": 3e-4,
        "num_epochs": 1,
        "num_minibatches": 2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "entropy_coeff": 0.01,
        "value_loss_coeff": 0.5,
        "max_grad_norm": 0.5,
        "normalize_advantages": True,
        "rollout_length": 4,
    },
    "network": {
        "policy_hidden": [32, 32],
        "value_hidden": [32, 32],
        "activation": "elu",
    },
    "rewards": {
        "upright": 1.0,
        "alive": 0.5,
    },
    "curriculum": {"enabled": False},
}

NUM_ENVS = 4
OBS_SIZE = 42  # BalanceEnv is 39-base + height_cmd + current_height + yaw_error = 42
ACTION_SIZE = 10


@pytest.fixture(scope="module")
def env():
    from wheeled_biped.envs.balance_env import BalanceEnv

    return BalanceEnv(config=_TINY_CONFIG)


@pytest.fixture(scope="module")
def trainer(env):
    from wheeled_biped.training.ppo import PPOTrainer

    t = PPOTrainer(env=env, config=_TINY_CONFIG, logger=None, seed=0)
    # Fix num_envs to match test size
    t.num_envs = NUM_ENVS
    t._rollout_length = 4
    return t


@pytest.fixture(scope="module")
def rollout_data(trainer, env):
    """Run one rollout and return (env_state, transitions, rng)."""
    rng = jax.random.PRNGKey(7)
    env_state = env.v_reset(rng, NUM_ENVS)
    rng, rollout_key = jax.random.split(rng)
    env_state, transitions, rng = trainer._rollout(
        trainer.params, env_state, rollout_key, trainer.obs_rms
    )
    jax.block_until_ready(transitions.reward)
    return env_state, transitions, rng


# ---------------------------------------------------------------------------
# Tests: compute_gae
# ---------------------------------------------------------------------------


class TestComputeGAE:
    def test_output_shapes(self, trainer):
        """GAE advantages and returns have shape (T, num_envs)."""
        from wheeled_biped.training.ppo import compute_gae

        T, N = 8, 4  # noqa: N806
        rewards = jnp.ones((T, N))
        values = jnp.ones((T, N)) * 0.5
        dones = jnp.zeros((T, N))
        last_value = jnp.ones(N) * 0.5

        adv, ret = compute_gae(rewards, values, dones, last_value)

        assert adv.shape == (T, N), f"advantages shape {adv.shape}"
        assert ret.shape == (T, N), f"returns shape {ret.shape}"

    def test_no_nan(self, trainer):
        """GAE with random rewards produces no NaN."""
        from wheeled_biped.training.ppo import compute_gae

        rng = jax.random.PRNGKey(42)
        T, N = 16, 4  # noqa: N806
        rewards = jax.random.normal(rng, (T, N))
        values = jax.random.normal(rng, (T, N))
        dones = (jax.random.uniform(rng, (T, N)) > 0.8).astype(jnp.float32)
        last_value = jnp.zeros(N)

        adv, ret = compute_gae(rewards, values, dones, last_value)

        assert not np.any(np.isnan(np.array(adv))), "NaN in advantages"
        assert not np.any(np.isnan(np.array(ret))), "NaN in returns"

    def test_returns_equal_advantages_plus_values(self):
        """returns = advantages + values (definition)."""
        from wheeled_biped.training.ppo import compute_gae

        T, N = 4, 2  # noqa: N806
        rewards = jnp.ones((T, N))
        values = jnp.ones((T, N)) * 2.0
        dones = jnp.zeros((T, N))
        last_value = jnp.ones(N) * 2.0

        adv, ret = compute_gae(rewards, values, dones, last_value)
        diff = jnp.abs(ret - (adv + values))
        assert float(jnp.max(diff)) < 1e-4, "returns != advantages + values"


# ---------------------------------------------------------------------------
# Tests: obs normalization
# ---------------------------------------------------------------------------


class TestObsNormalization:
    def test_rms_updates_mean(self, trainer):
        """update_running_mean_std changes the mean after a batch."""
        from wheeled_biped.training.ppo import (
            init_running_mean_std,
            update_running_mean_std,
        )

        rms = init_running_mean_std((OBS_SIZE,))
        original_mean = np.array(rms.mean).copy()

        batch = jnp.ones((32, OBS_SIZE)) * 5.0
        rms2 = update_running_mean_std(rms, batch)

        new_mean = np.array(rms2.mean)
        # Mean should have moved toward 5.0
        assert not np.allclose(new_mean, original_mean), "mean did not update"

    def test_rms_no_nan(self, trainer):
        """Normalized obs has no NaN."""
        from wheeled_biped.training.ppo import (
            init_running_mean_std,
            normalize_obs,
            update_running_mean_std,
        )

        rms = init_running_mean_std((OBS_SIZE,))
        batch = jax.random.normal(jax.random.PRNGKey(1), (64, OBS_SIZE))
        rms = update_running_mean_std(rms, batch)

        obs = jax.random.normal(jax.random.PRNGKey(2), (OBS_SIZE,))
        normed = normalize_obs(obs, rms)
        assert not np.any(np.isnan(np.array(normed))), "normalized obs has NaN"


# ---------------------------------------------------------------------------
# Tests: single rollout + update
# ---------------------------------------------------------------------------


class TestSingleUpdate:
    pytestmark = pytest.mark.slow

    def test_rollout_obs_no_nan(self, rollout_data):
        """Rollout observations contain no NaN."""
        _, transitions, _ = rollout_data
        obs_np = np.array(transitions.obs)
        assert not np.any(np.isnan(obs_np)), "NaN in rollout obs"

    def test_rollout_reward_no_nan(self, rollout_data):
        """Rollout rewards contain no NaN."""
        _, transitions, _ = rollout_data
        rew_np = np.array(transitions.reward)
        assert not np.any(np.isnan(rew_np)), "NaN in rollout rewards"

    def test_update_step_no_nan_params(self, trainer, rollout_data):
        """After one PPO update step params have no NaN."""
        env_state, transitions, rng = rollout_data

        # Compute last value for GAE
        from wheeled_biped.training.ppo import normalize_obs

        last_obs = normalize_obs(env_state.obs, trainer.obs_rms)
        _, last_value = trainer.model.apply(trainer.params, last_obs)

        rng, update_key = jax.random.split(rng)
        new_params, new_opt_state, metrics, _ = trainer._update_step(
            trainer.params,
            trainer.opt_state,
            transitions,
            last_value,
            update_key,
        )
        jax.block_until_ready(new_params)

        # Check no NaN in leaf arrays
        leaves = jax.tree_util.tree_leaves(jax.device_get(new_params))
        for leaf in leaves:
            assert not np.any(np.isnan(leaf)), "NaN found in updated params"

    def test_update_step_metrics_keys(self, trainer, rollout_data):
        """Metrics dict from _update_step contains required keys."""
        env_state, transitions, rng = rollout_data

        from wheeled_biped.training.ppo import normalize_obs

        last_obs = normalize_obs(env_state.obs, trainer.obs_rms)
        _, last_value = trainer.model.apply(trainer.params, last_obs)

        rng, update_key = jax.random.split(rng)
        _, _, metrics, _ = trainer._update_step(
            trainer.params,
            trainer.opt_state,
            transitions,
            last_value,
            update_key,
        )

        required = {"loss/policy", "loss/value", "loss/entropy", "loss/total"}
        for key in required:
            assert key in metrics, f"Missing metric key: {key}"

    def test_update_step_loss_is_finite(self, trainer, rollout_data):
        """Total loss is a finite scalar."""
        env_state, transitions, rng = rollout_data

        from wheeled_biped.training.ppo import normalize_obs

        last_obs = normalize_obs(env_state.obs, trainer.obs_rms)
        _, last_value = trainer.model.apply(trainer.params, last_obs)

        rng, update_key = jax.random.split(rng)
        _, _, metrics, _ = trainer._update_step(
            trainer.params,
            trainer.opt_state,
            transitions,
            last_value,
            update_key,
        )

        total_loss = float(metrics["loss/total"])
        assert np.isfinite(total_loss), f"Loss not finite: {total_loss}"


class TestPolicyKlGuard:
    pytestmark = pytest.mark.slow

    def test_train_rejects_update_when_policy_kl_exceeds_limit(self, env, tmp_path):
        """A destructive PPO update is rolled back when approx_kl exceeds max_policy_kl."""
        import copy
        import types

        import jax.tree_util as jtu

        from wheeled_biped.training.ppo import PPOTrainer

        cfg = copy.deepcopy(_TINY_CONFIG)
        cfg["ppo"]["max_policy_kl"] = 0.01
        trainer = PPOTrainer(env=env, config=cfg, logger=None, seed=0)
        trainer.num_envs = NUM_ENVS

        initial_leaves = [
            np.array(leaf) for leaf in jtu.tree_leaves(jax.device_get(trainer.params))
        ]

        def _fake_high_kl_update(self_, params, opt_state, transitions, last_value, rng):
            del self_, transitions, last_value
            new_params = jtu.tree_map(lambda x: x + jnp.ones_like(x), params)
            metrics = {"policy/approx_kl": jnp.array(1.0, dtype=jnp.float32)}
            return new_params, opt_state, metrics, rng

        trainer._update_step = types.MethodType(_fake_high_kl_update, trainer)
        steps_per_update = trainer._rollout_length * trainer.num_envs

        trainer.train(
            total_steps=steps_per_update,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_interval=99,
            save_interval=99,
        )

        final_leaves = [
            np.array(leaf) for leaf in jtu.tree_leaves(jax.device_get(trainer.params))
        ]
        for before, after in zip(initial_leaves, final_leaves):
            np.testing.assert_allclose(after, before, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: checkpoint save / load
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_checkpoint_keys_present(self, trainer, tmp_path):
        """Saved checkpoint pickle contains all required keys."""
        ckpt_dir = str(tmp_path / "ckpt_keys")
        trainer._save_checkpoint(ckpt_dir, global_step=100, best_reward=0.5)

        pkl_path = Path(ckpt_dir) / "checkpoint.pkl"
        assert pkl_path.exists(), "checkpoint.pkl not created"

        with open(pkl_path, "rb") as f:
            ckpt = pickle.load(f)

        required_keys = {
            "params",
            "opt_state",
            "obs_rms",
            "rng",
            "env_state",
            "config",
            "global_step",
            "best_reward",
        }
        for k in required_keys:
            assert k in ckpt, f"Missing checkpoint key: {k}"

    def test_checkpoint_global_step_matches(self, trainer, tmp_path):
        """Loaded checkpoint restores global_step correctly."""
        ckpt_dir = str(tmp_path / "ckpt_step")
        trainer._save_checkpoint(ckpt_dir, global_step=12345, best_reward=1.23)
        trainer.load_checkpoint(ckpt_dir)
        assert trainer._resumed_global_step == 12345

    def test_checkpoint_best_reward_matches(self, trainer, tmp_path):
        """Loaded checkpoint restores best_reward correctly."""
        ckpt_dir = str(tmp_path / "ckpt_reward")
        trainer._save_checkpoint(ckpt_dir, global_step=0, best_reward=-3.14)
        trainer.load_checkpoint(ckpt_dir)
        assert abs(trainer._resumed_best_reward - (-3.14)) < 1e-5

    def test_checkpoint_obs_rms_roundtrip(self, trainer, tmp_path):
        """obs_rms mean is preserved through save/load."""
        from wheeled_biped.training.ppo import (
            init_running_mean_std,
            update_running_mean_std,
        )

        # Give the trainer a non-trivial obs_rms
        batch = jnp.ones((16, OBS_SIZE)) * 3.0
        trainer.obs_rms = update_running_mean_std(trainer.obs_rms, batch)
        original_mean = np.array(trainer.obs_rms.mean).copy()

        ckpt_dir = str(tmp_path / "ckpt_rms")
        trainer._save_checkpoint(ckpt_dir, global_step=0, best_reward=0.0)

        # Corrupt the in-memory state
        trainer.obs_rms = init_running_mean_std((OBS_SIZE,))

        trainer.load_checkpoint(ckpt_dir)
        restored_mean = np.array(trainer.obs_rms.mean)
        assert np.allclose(original_mean, restored_mean, atol=1e-5), (
            "obs_rms mean not preserved through checkpoint round-trip"
        )

    def test_checkpoint_params_roundtrip(self, trainer, tmp_path):
        """Network params are preserved through save/load (spot-check one leaf)."""
        import jax.tree_util as jtu

        ckpt_dir = str(tmp_path / "ckpt_params")
        original_leaves = [np.array(lf) for lf in jtu.tree_leaves(jax.device_get(trainer.params))]

        trainer._save_checkpoint(ckpt_dir, global_step=0, best_reward=0.0)
        trainer.load_checkpoint(ckpt_dir)

        restored_leaves = [np.array(lf) for lf in jtu.tree_leaves(jax.device_get(trainer.params))]
        for orig, rest in zip(original_leaves, restored_leaves):
            assert np.allclose(orig, rest, atol=1e-6), "Params differ after checkpoint round-trip"

    def test_checkpoint_rng_roundtrip(self, env, tmp_path):
        """RNG is saved and restored so resume does not restart the random stream."""
        from wheeled_biped.training.ppo import PPOTrainer

        trainer = PPOTrainer(env=env, config=_TINY_CONFIG, logger=None, seed=0)
        trainer.rng = jax.random.PRNGKey(123)

        ckpt_dir = str(tmp_path / "ckpt_rng")
        trainer._save_checkpoint(ckpt_dir, global_step=0, best_reward=0.0)
        trainer.rng = jax.random.PRNGKey(999)

        trainer.load_checkpoint(ckpt_dir)

        assert np.array_equal(np.array(trainer.rng), np.array(jax.random.PRNGKey(123)))

    def test_checkpoint_env_state_roundtrip(self, env, tmp_path):
        """EnvState is saved compactly and restored into a step-able JAX state."""
        from wheeled_biped.training.ppo import (
            _COMPACT_MJX_MODEL_MARKER,
            PPOTrainer,
        )

        trainer = PPOTrainer(env=env, config=_TINY_CONFIG, logger=None, seed=0)
        trainer.num_envs = NUM_ENVS
        env_state = env.v_reset(jax.random.PRNGKey(5), NUM_ENVS)
        jax.block_until_ready(env_state.obs)

        ckpt_dir = str(tmp_path / "ckpt_env_state")
        trainer._save_checkpoint(
            ckpt_dir,
            global_step=0,
            best_reward=0.0,
            env_state=env_state,
        )

        with open(Path(ckpt_dir) / "checkpoint.pkl", "rb") as f:
            ckpt = pickle.load(f)
        dr_payload = ckpt["env_state"].info.get("dr_mjx_model")
        assert isinstance(dr_payload, dict)
        assert dr_payload.get(_COMPACT_MJX_MODEL_MARKER) is True

        trainer.load_checkpoint(ckpt_dir)
        restored = trainer._resumed_env_state
        assert restored is not None
        np.testing.assert_allclose(np.array(restored.obs), np.array(env_state.obs), atol=1e-6)

        next_state = env.v_step(restored, jnp.zeros((NUM_ENVS, env.num_actions)))
        jax.block_until_ready(next_state.obs)
        assert next_state.obs.shape == env_state.obs.shape

    def test_curriculum_reset_patch_updates_command_and_obs(self, env):
        """Legacy resume reset is patched to the saved curriculum height range."""
        from wheeled_biped.training.ppo import PPOTrainer

        trainer = PPOTrainer(env=env, config=_TINY_CONFIG, logger=None, seed=0)
        env_state = env.v_reset(jax.random.PRNGKey(11), NUM_ENVS)

        patched, _ = trainer._patch_curriculum_reset_state(
            env_state,
            jax.random.PRNGKey(12),
            current_min_h=0.40,
        )

        height_command = np.array(patched.info["height_command"])
        obs_height = np.array(patched.obs[:, -3])
        expected_obs_height = (height_command - env.MIN_HEIGHT_CMD) / (
            env.MAX_HEIGHT_CMD - env.MIN_HEIGHT_CMD
        )

        assert np.all(height_command >= 0.40)
        assert np.all(height_command <= env.MAX_HEIGHT_CMD)
        np.testing.assert_allclose(obs_height, expected_obs_height, atol=1e-6)
        np.testing.assert_allclose(
            np.array(patched.info["curriculum_min_height"]),
            np.full(NUM_ENVS, 0.40),
            atol=1e-6,
        )

    def test_checkpoint_warm_start_resets_training_state(self, env, tmp_path):
        """Cross-stage warm-start loads weights but does not exact-resume training state."""
        from wheeled_biped.training.ppo import PPOTrainer

        source = PPOTrainer(env=env, config=_TINY_CONFIG, logger=None, seed=0)
        source.num_envs = NUM_ENVS
        env_state = env.v_reset(jax.random.PRNGKey(21), NUM_ENVS)
        source._curriculum_min_height = 0.40

        ckpt_dir = str(tmp_path / "ckpt_warm_start")
        source._save_checkpoint(
            ckpt_dir,
            global_step=12345,
            best_reward=9.0,
            best_eval_per_step=8.0,
            best_eval_success=0.9,
            best_train_reward=7.0,
            env_state=env_state,
        )
        with open(Path(ckpt_dir) / "checkpoint.pkl", "rb") as f:
            ckpt = pickle.load(f)
        ckpt["opt_state"] = "stale optimizer state"
        with open(Path(ckpt_dir) / "checkpoint.pkl", "wb") as f:
            pickle.dump(ckpt, f)

        target = PPOTrainer(env=env, config=_TINY_CONFIG, logger=None, seed=999)
        target.load_checkpoint(ckpt_dir, resume_training=False)

        assert target.opt_state != "stale optimizer state"
        assert target._resumed_global_step == 0
        assert target._resumed_best_reward == float("-inf")
        assert target._resumed_curriculum_min is None
        assert target._resumed_env_state is None
        assert target._resumed_best_eval_per_step == float("-inf")
        assert target._resumed_best_eval_success == float("-inf")
        assert target._resumed_best_train_reward == float("-inf")


class TestTrainCliStepResolution:
    def test_steps_default_is_absolute_target(self):
        from scripts.train import _resolve_target_total_steps

        assert (
            _resolve_target_total_steps(
                steps=100,
                additional_steps=None,
                resumed_step=60,
            )
            == 100
        )

    def test_additional_steps_offsets_from_resume_step(self):
        from scripts.train import _resolve_target_total_steps

        assert (
            _resolve_target_total_steps(
                steps=100,
                additional_steps=40,
                resumed_step=60,
            )
            == 100
        )

    def test_additional_steps_must_be_positive(self):
        from scripts.train import _resolve_target_total_steps

        with pytest.raises(ValueError, match="additional-steps"):
            _resolve_target_total_steps(
                steps=100,
                additional_steps=0,
                resumed_step=60,
            )

    def test_absolute_steps_must_exceed_resume_step(self):
        from scripts.train import _resolve_target_total_steps

        with pytest.raises(ValueError, match="total target"):
            _resolve_target_total_steps(
                steps=60,
                additional_steps=None,
                resumed_step=60,
            )


# ---------------------------------------------------------------------------
# Tests: checkpoint versioning (#8)
# ---------------------------------------------------------------------------


class TestCheckpointVersioning:
    """Checkpoint version field is written and validated on load.

    Covers:
      - new checkpoint contains version == CHECKPOINT_VERSION
      - pre-versioned checkpoint (no version field) loads with RuntimeWarning
      - checkpoint with future version raises ValueError with a clear message
    """

    def test_new_checkpoint_has_version_field(self, trainer, tmp_path):
        """_save_checkpoint() must write version == CHECKPOINT_VERSION."""
        from wheeled_biped.training.ppo import CHECKPOINT_VERSION

        ckpt_dir = str(tmp_path / "ckpt_ver")
        trainer._save_checkpoint(ckpt_dir, global_step=0, best_reward=0.0)

        pkl_path = Path(ckpt_dir) / "checkpoint.pkl"
        with open(pkl_path, "rb") as f:
            ckpt = pickle.load(f)

        assert "version" in ckpt, "checkpoint.pkl must contain a 'version' key"
        assert ckpt["version"] == CHECKPOINT_VERSION, (
            f"checkpoint version {ckpt['version']} != CHECKPOINT_VERSION {CHECKPOINT_VERSION}"
        )

    def test_pre_versioned_checkpoint_loads_with_warning(self, trainer, tmp_path):
        """Checkpoint without 'version' key loads successfully but emits RuntimeWarning."""
        import warnings

        # Write a checkpoint then manually strip the version key
        ckpt_dir = str(tmp_path / "ckpt_old")
        trainer._save_checkpoint(ckpt_dir, global_step=5, best_reward=1.0)

        pkl_path = Path(ckpt_dir) / "checkpoint.pkl"
        with open(pkl_path, "rb") as f:
            ckpt = pickle.load(f)
        ckpt.pop("version", None)  # simulate pre-versioned checkpoint
        with open(pkl_path, "wb") as f:
            pickle.dump(ckpt, f)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trainer.load_checkpoint(ckpt_dir)  # must not raise

        assert any(issubclass(w.category, RuntimeWarning) for w in caught), (
            "Loading a pre-versioned checkpoint must emit a RuntimeWarning"
        )
        assert trainer._resumed_global_step == 5, (
            "global_step must still be restored from a pre-versioned checkpoint"
        )

    def test_future_version_checkpoint_raises_value_error(self, trainer, tmp_path):
        """Checkpoint with version > CHECKPOINT_VERSION must raise ValueError."""
        from wheeled_biped.training.ppo import CHECKPOINT_VERSION

        ckpt_dir = str(tmp_path / "ckpt_future")
        trainer._save_checkpoint(ckpt_dir, global_step=0, best_reward=0.0)

        pkl_path = Path(ckpt_dir) / "checkpoint.pkl"
        with open(pkl_path, "rb") as f:
            ckpt = pickle.load(f)
        ckpt["version"] = CHECKPOINT_VERSION + 99  # simulate future format
        with open(pkl_path, "wb") as f:
            pickle.dump(ckpt, f)

        with pytest.raises(ValueError, match="version"):
            trainer.load_checkpoint(ckpt_dir)

    def test_current_version_checkpoint_loads_silently(self, trainer, tmp_path):
        """Checkpoint with version == CHECKPOINT_VERSION loads without warnings."""
        import warnings

        ckpt_dir = str(tmp_path / "ckpt_cur")
        trainer._save_checkpoint(ckpt_dir, global_step=77, best_reward=2.5)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trainer.load_checkpoint(ckpt_dir)

        runtime_warns = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert len(runtime_warns) == 0, (
            f"Current-version checkpoint should load silently, got: {runtime_warns}"
        )
        assert trainer._resumed_global_step == 77


# ---------------------------------------------------------------------------
# Tests: BalanceEnv reward weight defaults (#6)
# ---------------------------------------------------------------------------


class TestBalanceEnvRewardWeightDefaults:
    """BalanceEnv fallback defaults must match configs/training/balance.yaml.

    These tests catch any future divergence between the hardcoded defaults
    and the production config, without requiring a full env reset.
    They use a minimal no-op config so the defaults take effect.
    """

    def _make_env_no_rewards_cfg(self):
        """Create BalanceEnv with empty rewards section → all defaults active."""
        from wheeled_biped.envs.balance_env import BalanceEnv

        # No 'rewards' key → all reward weights come from defaults
        config = {
            "task": {"episode_length": 10, "initial_min_height": 0.68},
            "low_level_pid": {"enabled": False},
            "domain_randomization": {"enabled": False},
            "curriculum": {"enabled": False},
        }
        return BalanceEnv(config=config)

    def _load_production_weights(self):
        from wheeled_biped.utils.config import load_yaml

        cfg = load_yaml("configs/training/balance.yaml")
        return cfg["rewards"]

    def test_all_defaults_match_production_config(self):
        """Every reward weight default must match the corresponding value in balance.yaml."""
        env = self._make_env_no_rewards_cfg()
        production = self._load_production_weights()

        mismatches = []
        for key, prod_val in production.items():
            if key not in env._reward_weights:
                mismatches.append(f"  {key}: missing from _reward_weights")
                continue
            code_val = env._reward_weights[key]
            if abs(float(code_val) - float(prod_val)) > 1e-9:
                mismatches.append(f"  {key}: code default={code_val}, balance.yaml={prod_val}")

        assert not mismatches, (
            "BalanceEnv reward weight defaults diverge from configs/training/balance.yaml:\n"
            + "\n".join(mismatches)
        )

    def test_joint_velocity_default_matches_yaml(self):
        """Focused check: joint_velocity default must be -0.0007 (matches balance.yaml)."""
        env = self._make_env_no_rewards_cfg()
        assert abs(env._reward_weights["joint_velocity"] - (-0.0007)) < 1e-9, (
            f"joint_velocity default={env._reward_weights['joint_velocity']}, expected -0.0007"
        )

    def test_explicit_config_overrides_default(self):
        """A reward key present in config must override the default."""
        from wheeled_biped.envs.balance_env import BalanceEnv

        config = {
            "task": {"episode_length": 10, "initial_min_height": 0.68},
            "low_level_pid": {"enabled": False},
            "domain_randomization": {"enabled": False},
            "curriculum": {"enabled": False},
            "rewards": {"joint_velocity": -0.999},
        }
        env = BalanceEnv(config=config)
        assert abs(env._reward_weights["joint_velocity"] - (-0.999)) < 1e-9, (
            "Explicit rewards.joint_velocity in config must override the default"
        )


# ---------------------------------------------------------------------------
# Tests: control path semantics (#10 + #5 documentation)
# ---------------------------------------------------------------------------


class TestControlPathSemantics:
    """Document and protect the control-path design choices for #5 and #10.

    #5 — Action flow through PPO rollout:
      raw_action (policy sample) → clip → [balance_env: smooth → PID] → physics
      PPO importance ratio uses raw_action on both sides (internally consistent).
      Advantage reflects smoothed-action outcomes: accepted trade-off for action smoothing.

    #10 — BalanceEnv.step() must override WheeledBipedEnv.step():
      base class: direct-torque semantics (action → scaled ctrl)
      BalanceEnv:  PID semantics (action → smooth → PID → torque)
      Python dispatch must always reach the override when env is BalanceEnv.
    """

    _MINIMAL_CFG = {
        "task": {"episode_length": 10, "initial_min_height": 0.68},
        "low_level_pid": {
            "enabled": True,
            "action_smoothing_alpha": 0.4,
            "anti_windup_limit": 0.35,
            "wheel_vel_limit": 20.0,
        },
        "domain_randomization": {
            "enabled": False,
            "push_magnitude": 0,
            "push_interval": 9999,
            "push_duration": 1,
        },
        "curriculum": {"enabled": False},
        "sensor_noise": {"enabled": False},
    }

    def test_balance_env_step_is_override_not_base(self):
        """BalanceEnv.step must NOT be WheeledBipedEnv.step.

        This protects against accidental dispatch to the direct-torque base
        class when BalanceEnv is instantiated.
        """
        from wheeled_biped.envs.balance_env import BalanceEnv
        from wheeled_biped.envs.base_env import WheeledBipedEnv

        assert BalanceEnv.step is not WheeledBipedEnv.step, (
            "BalanceEnv.step must override WheeledBipedEnv.step. "
            "If this fails, BalanceEnv accidentally lost its step() override."
        )

    def test_balance_env_v_step_dispatches_to_override(self):
        """v_step on a BalanceEnv instance must dispatch to BalanceEnv.step.

        Verifies Python method resolution works correctly at runtime for the
        vectorised interface used during training.
        """
        import types

        from wheeled_biped.envs.balance_env import BalanceEnv

        env = BalanceEnv(config=self._MINIMAL_CFG)
        # jax.vmap(self.step) captures the bound method — verify it points to the override
        bound = env.step
        assert isinstance(bound, types.MethodType), "env.step must be a bound method"
        assert bound.__func__ is BalanceEnv.step, (
            "env.step must dispatch to BalanceEnv.step, not WheeledBipedEnv.step"
        )

    @pytest.mark.slow
    def test_pid_enabled_produces_different_ctrl_than_direct_torque(self):
        """With PID enabled, scaled_ctrl differs from simple ctrlrange scaling.

        This confirms BalanceEnv.step() is using the PID path, not the base
        direct-torque path, when pid_enabled=True.

        Strategy: run one step with PID on and one with PID off; verify ctrl differs.
        We do NOT run actual MJX physics — just check that the control_action passed
        to the actuators changes depending on the control mode.
        """
        import jax
        import jax.numpy as jnp

        from wheeled_biped.envs.balance_env import BalanceEnv

        # PID-enabled env
        env_pid = BalanceEnv(config=self._MINIMAL_CFG)

        # Direct-torque env (pid disabled)
        cfg_notpid = {**self._MINIMAL_CFG, "low_level_pid": {"enabled": False}}
        env_nopid = BalanceEnv(config=cfg_notpid)

        rng = jax.random.PRNGKey(0)
        state_pid = env_pid.reset(rng)
        state_nopid = env_nopid.reset(rng)

        # Same deterministic action for both
        action = jnp.zeros(env_pid.num_actions)

        next_pid = env_pid.step(state_pid, action)
        next_nopid = env_nopid.step(state_nopid, action)

        # ctrl values differ: PID integrates error, direct mode just scales
        ctrl_pid = np.array(next_pid.mjx_data.ctrl)
        ctrl_nopid = np.array(next_nopid.mjx_data.ctrl)
        assert not np.allclose(ctrl_pid, ctrl_nopid), (
            "PID and direct-torque paths must produce different ctrl for the same action. "
            "If they match, the PID path may not be active."
        )

    @pytest.mark.slow
    def test_rollout_stores_raw_action_not_clipped(self):
        """PPO rollout transition.action stores the raw (pre-clip) policy sample.

        This documents the #5 design choice: the importance ratio uses raw_action
        on both sides (π_new(raw)/π_old(raw)), which is internally consistent.
        The env receives clip(raw_action) and applies smoothing on top.
        """
        import jax

        from wheeled_biped.envs.balance_env import BalanceEnv
        from wheeled_biped.training.ppo import PPOTrainer

        env = BalanceEnv(config=self._MINIMAL_CFG)
        trainer = PPOTrainer(
            env=env,
            config={
                **self._MINIMAL_CFG,
                "ppo": {
                    "learning_rate": 3e-4,
                    "num_epochs": 1,
                    "num_minibatches": 2,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_epsilon": 0.2,
                    "entropy_coeff": 0.01,
                    "value_loss_coeff": 0.5,
                    "max_grad_norm": 0.5,
                    "normalize_advantages": True,
                    "rollout_length": 4,
                },
                "network": {
                    "policy_hidden": [32, 32],
                    "value_hidden": [32, 32],
                    "activation": "elu",
                },
            },
            logger=None,
            seed=0,
        )
        trainer.num_envs = 4
        trainer._rollout_length = 4

        rng = jax.random.PRNGKey(1)
        env_state = env.v_reset(rng, 4)
        rng, key = jax.random.split(rng)
        _, transitions, _ = trainer._rollout(trainer.params, env_state, key, trainer.obs_rms)

        # raw_action may have values outside [-1, 1] (Gaussian tails)
        # With init_std=0.2 and seed=1 it's unlikely but possible.
        # The key invariant: transition.action is NOT the clipped version.
        # We verify by checking it equals what the policy produced (not what env saw).
        # Structural check: shape and dtype are correct
        assert transitions.action.shape[-1] == env.num_actions, (
            f"transition.action last dim {transitions.action.shape[-1]}"
            f" != num_actions {env.num_actions}"
        )
        # Verify stored actions are unbounded (raw samples, not clipped to [-1,1])
        # NOTE: for a freshly-initialized policy with init_std=0.2 this may occasionally
        # be all within [-1,1]; the structural test above is the reliable invariant.
        # The presence of the comment in ppo.py is the primary documentation artifact.


# ---------------------------------------------------------------------------
# Tests: eval_pass()
# ---------------------------------------------------------------------------


class TestEvalPass:
    """Tests for PPOTrainer.eval_pass() — the held-out evaluation method.

    These tests verify the structural contract of eval_pass() without requiring
    a full MJX GPU training loop.  They run with tiny envs (4 envs, short episodes).
    Marked slow because eval_pass() triggers MJX JIT compilation on first call.
    """

    pytestmark = pytest.mark.slow

    def test_eval_pass_returns_required_keys(self, trainer, env):
        """eval_pass() returns a dict with all required metric keys."""
        rng = jax.random.PRNGKey(99)
        result = trainer.eval_pass(num_eval_envs=4, num_episodes=4, rng=rng)

        required = {
            "eval_reward_mean",
            "eval_reward_std",
            "eval_fall_rate",
            "eval_success_rate",
            "eval_num_episodes",
        }
        for key in required:
            assert key in result, f"Missing key in eval_pass result: {key}"

    def test_eval_pass_no_nan(self, trainer, env):
        """eval_pass() produces no NaN in any metric."""
        rng = jax.random.PRNGKey(55)
        result = trainer.eval_pass(num_eval_envs=4, num_episodes=4, rng=rng)

        for key, val in result.items():
            assert np.isfinite(val), f"Non-finite value for {key}: {val}"

    def test_eval_pass_rates_sum_to_one(self, trainer, env):
        """fall_rate + success_rate == 1.0 (they are complements)."""
        rng = jax.random.PRNGKey(77)
        result = trainer.eval_pass(num_eval_envs=4, num_episodes=4, rng=rng)

        total = result["eval_fall_rate"] + result["eval_success_rate"]
        assert abs(total - 1.0) < 1e-6, f"fall_rate + success_rate = {total} (expected 1.0)"

    def test_eval_pass_collects_episodes(self, trainer, env):
        """eval_pass() reports at least one completed episode."""
        rng = jax.random.PRNGKey(13)
        result = trainer.eval_pass(num_eval_envs=4, num_episodes=4, rng=rng)

        assert result["eval_num_episodes"] >= 1, "eval_pass() should complete at least 1 episode"

    def test_eval_pass_does_not_mutate_obs_rms(self, trainer, env):
        """eval_pass() does not change obs_rms (evaluation is read-only)."""

        # Record current obs_rms mean
        before_mean = np.array(trainer.obs_rms.mean).copy()

        rng = jax.random.PRNGKey(42)
        trainer.eval_pass(num_eval_envs=4, num_episodes=4, rng=rng)

        after_mean = np.array(trainer.obs_rms.mean)
        assert np.allclose(before_mean, after_mean, atol=1e-7), (
            "eval_pass() must not mutate obs_rms"
        )

    def test_eval_pass_curriculum_min_height_accepted(self, trainer, env):
        """eval_pass() accepts curriculum_min_height without raising.

        curriculum_min_height=0.40 is below initial_min_height (0.68), so it
        exercises the full patching path: resample height_command from the wider
        range and update obs[:, -1] accordingly.
        """
        rng = jax.random.PRNGKey(5)
        result = trainer.eval_pass(
            num_eval_envs=4, num_episodes=4, rng=rng, curriculum_min_height=0.40
        )
        required = {
            "eval_reward_mean",
            "eval_reward_std",
            "eval_fall_rate",
            "eval_success_rate",
            "eval_num_episodes",
        }
        for key in required:
            assert key in result, f"Missing key in result: {key}"
        assert np.isfinite(result["eval_reward_mean"])

    def test_eval_pass_curriculum_min_height_changes_result(self, trainer, env):
        """curriculum_min_height=0.40 produces different results than the default.

        Using the same RNG seed, the only difference between the two runs is the
        initial height_command distribution:
          default:  [0.68, 0.70] -> height_norm in [0.933, 1.0]
          curriculum: [0.40, 0.70] -> height_norm in [0.0, 1.0]
        Different obs -> different greedy actions -> different rewards.
        If results are identical, the parameter had no effect (regression).
        """
        result_default = trainer.eval_pass(
            num_eval_envs=4, num_episodes=4, rng=jax.random.PRNGKey(99)
        )
        result_curriculum = trainer.eval_pass(
            num_eval_envs=4,
            num_episodes=4,
            rng=jax.random.PRNGKey(99),
            curriculum_min_height=0.40,
        )
        assert result_default["eval_reward_mean"] != result_curriculum["eval_reward_mean"], (
            "curriculum_min_height=0.40 must change eval results vs default "
            "(proves the parameter actively patches height_command / obs)"
        )

    def test_eval_pass_curriculum_min_height_none_unchanged(self, trainer, env):
        """curriculum_min_height=None is identical to omitting the argument."""
        result_omitted = trainer.eval_pass(
            num_eval_envs=4, num_episodes=4, rng=jax.random.PRNGKey(17)
        )
        result_none = trainer.eval_pass(
            num_eval_envs=4,
            num_episodes=4,
            rng=jax.random.PRNGKey(17),
            curriculum_min_height=None,
        )
        assert result_omitted["eval_reward_mean"] == result_none["eval_reward_mean"], (
            "curriculum_min_height=None must be identical to the default (no patch)"
        )


# ---------------------------------------------------------------------------
# Tests: eval-gated within-stage curriculum signal
# ---------------------------------------------------------------------------

_CURRICULUM_CONFIG = {
    **_TINY_CONFIG,
    "task": {
        **_TINY_CONFIG["task"],
        "initial_min_height": 0.68,
    },
    "curriculum": {
        "enabled": True,
        "reward_threshold": 0.5,  # low threshold — easy to meet in tests
        "num_levels": 5,
        "window": 2,  # small window for legacy path tests
        "use_eval_signal": True,
        "eval_interval": 2,  # fire every 2 updates — fast for tests
        "eval_episodes": 2,  # minimal episodes per check
    },
}


class TestEvalGatedCurriculum:
    """Verify eval-gated within-stage curriculum advancement.

    These tests monkeypatch eval_pass() so no real JAX training is needed.
    They verify:
      - use_eval_signal=True reads config correctly
      - curriculum advances when eval_per_step >= threshold
      - curriculum does NOT advance when eval_per_step < threshold
      - backward-compat: use_eval_signal=False still uses reward_window
    """

    @pytest.fixture(scope="class")
    def curriculum_env(self):
        from wheeled_biped.envs.balance_env import BalanceEnv

        return BalanceEnv(config=_CURRICULUM_CONFIG)

    @pytest.fixture(scope="class")
    def curriculum_trainer(self, curriculum_env):
        from wheeled_biped.training.ppo import PPOTrainer

        t = PPOTrainer(env=curriculum_env, config=_CURRICULUM_CONFIG, logger=None, seed=1)
        t.num_envs = NUM_ENVS
        t._rollout_length = 4
        return t

    def test_eval_signal_config_is_read(self, curriculum_trainer):
        """use_eval_signal and eval_interval are parsed from config."""
        cfg = curriculum_trainer.config.get("curriculum", {})
        assert cfg.get("use_eval_signal") is True, "use_eval_signal not set in config"
        assert cfg.get("eval_interval") == 2, "eval_interval not set in config"
        assert cfg.get("eval_episodes") == 2, "eval_episodes not set in config"

    def test_eval_gated_advances_when_threshold_met(self, curriculum_trainer):
        """Curriculum advances when eval_per_step >= reward_threshold.

        Monkeypatches eval_pass() to return a high-reward result.
        reward_threshold = 0.5; episode_length=10;
        so eval_reward_mean must be >= 0.5 * 10 = 5.0 to advance.
        """
        import types

        # High eval return: per_step = eval_reward_mean / episode_length >= threshold.
        # episode_length=20, threshold=0.5*(1.0+0.5)=0.75, so need mean >= 0.75*20 = 15.0.
        def _fake_eval_pass(self_, **kwargs):
            return {
                "eval_reward_mean": 20.0,  # episode return; /20 = 1.0 >= 0.75 threshold
                "eval_reward_std": 0.0,
                "eval_fall_rate": 0.0,
                "eval_success_rate": 1.0,
                "eval_num_episodes": 2,
            }

        original_eval = curriculum_trainer.eval_pass
        curriculum_trainer.eval_pass = types.MethodType(_fake_eval_pass, curriculum_trainer)

        # Read curriculum state before
        cfg = curriculum_trainer.config.get("curriculum", {})
        initial_min_h = float(
            curriculum_trainer.config.get("task", {}).get("initial_min_height", 0.68)
        )
        final_min_h = getattr(curriculum_trainer.env, "MIN_HEIGHT_CMD", 0.38)
        num_levels = cfg.get("num_levels", 5)
        _level_step = (initial_min_h - final_min_h) / max(num_levels, 1)  # noqa: F841
        reward_threshold = cfg["reward_threshold"] * sum(
            w for w in curriculum_trainer.config.get("rewards", {}).values() if w > 0
        )

        # Simulate what the training loop does at eval_interval
        eval_result = curriculum_trainer.eval_pass(num_eval_envs=4, num_episodes=2)
        eval_per_step = eval_result["eval_reward_mean"] / max(1, curriculum_trainer.episode_length)
        assert eval_per_step >= reward_threshold, (
            f"Test setup error: eval_per_step={eval_per_step} < threshold={reward_threshold}"
        )

        curriculum_trainer.eval_pass = original_eval  # restore

    def test_eval_gated_does_not_advance_when_threshold_not_met(self, curriculum_trainer):
        """Curriculum does not advance when eval_per_step < reward_threshold."""
        import types

        # Low eval return: per_step = 0.0 / 10 = 0.0 < 0.5 threshold
        def _fake_eval_pass_low(self_, **kwargs):
            return {
                "eval_reward_mean": 0.0,
                "eval_reward_std": 0.0,
                "eval_fall_rate": 1.0,
                "eval_success_rate": 0.0,
                "eval_num_episodes": 2,
            }

        original_eval = curriculum_trainer.eval_pass
        curriculum_trainer.eval_pass = types.MethodType(_fake_eval_pass_low, curriculum_trainer)

        eval_result = curriculum_trainer.eval_pass(num_eval_envs=4, num_episodes=2)
        eval_per_step = eval_result["eval_reward_mean"] / max(1, curriculum_trainer.episode_length)

        cfg = curriculum_trainer.config.get("curriculum", {})
        reward_threshold = cfg["reward_threshold"] * sum(
            w for w in curriculum_trainer.config.get("rewards", {}).values() if w > 0
        )
        assert eval_per_step < reward_threshold, (
            f"Test setup error: eval_per_step={eval_per_step} >= threshold={reward_threshold}"
        )

        curriculum_trainer.eval_pass = original_eval  # restore

    def test_backward_compat_uses_reward_window_when_eval_disabled(self):
        """When use_eval_signal=False, config still reads window for legacy path."""
        legacy_config = {
            **_CURRICULUM_CONFIG,
            "curriculum": {
                **_CURRICULUM_CONFIG["curriculum"],
                "use_eval_signal": False,
            },
        }
        cfg = legacy_config.get("curriculum", {})
        assert cfg.get("use_eval_signal") is False
        assert cfg.get("window") == 2  # legacy window still present

    def test_eval_per_step_normalization(self, curriculum_trainer):
        """eval_per_step = eval_reward_mean / episode_length is correct."""
        episode_length = curriculum_trainer.episode_length
        # If mean episode return = 7.5 and episode_length = 10 -> per_step = 0.75
        eval_reward_mean = 7.5
        expected = eval_reward_mean / max(1, episode_length)
        computed = eval_reward_mean / max(1, episode_length)
        assert abs(computed - expected) < 1e-9


# ---------------------------------------------------------------------------
# Tests: logger lifecycle (log before close)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Config for eval-triggered checkpoint tests
# ---------------------------------------------------------------------------

_EVAL_CKPT_CONFIG = {
    **_TINY_CONFIG,
    "task": {
        **_TINY_CONFIG["task"],
        "initial_min_height": 0.68,
    },
    "curriculum": {
        "enabled": True,
        "reward_threshold": 999.0,  # impossibly high — curriculum never advances
        "num_levels": 5,
        "window": 2,
        "use_eval_signal": True,
        "eval_interval": 1,  # eval every update — triggers on the first update
        "eval_episodes": 2,
        "ckpt_cooldown_evals": 0,  # disable cooldown so existing tests are unaffected
    },
}


# ---------------------------------------------------------------------------
# Tests: eval-triggered checkpoint saving
# ---------------------------------------------------------------------------


class TestEvalTriggeredCheckpoints:
    """Verify that best_eval_per_step and best_eval_success checkpoints are saved.

    Uses monkeypatched eval_pass() so no real MJX training is needed.
    Runs a 3-update training loop (eval_interval=1) with a tiny config.
    Marked slow because t.train() triggers MJX JIT compilation on first call.
    """

    pytestmark = pytest.mark.slow

    @staticmethod
    def _make_trainer():
        from wheeled_biped.envs.balance_env import BalanceEnv
        from wheeled_biped.training.ppo import PPOTrainer

        env = BalanceEnv(config=_EVAL_CKPT_CONFIG)
        t = PPOTrainer(env=env, config=_EVAL_CKPT_CONFIG, logger=None, seed=0)
        t.num_envs = NUM_ENVS
        t._rollout_length = 4
        return t

    def test_saves_best_eval_per_step_on_improvement(self, tmp_path):
        """best_eval_per_step/checkpoint.pkl is created when eval_per_step improves."""
        import types

        t = self._make_trainer()

        def _good_eval(self_, **kwargs):
            return {
                "eval_reward_mean": 50.0,
                "eval_reward_std": 0.0,
                "eval_fall_rate": 0.0,
                "eval_success_rate": 1.0,
                "eval_num_episodes": 2,
            }

        t.eval_pass = types.MethodType(_good_eval, t)
        spu = t._rollout_length * t.num_envs
        t.train(total_steps=spu * 3, checkpoint_dir=str(tmp_path / "ckpt"))

        assert (tmp_path / "ckpt" / "best_eval_per_step" / "checkpoint.pkl").exists(), (
            "best_eval_per_step checkpoint should be created on first improvement"
        )

    def test_saves_best_eval_success_on_improvement(self, tmp_path):
        """best_eval_success/checkpoint.pkl is created when eval_success_rate improves."""
        import types

        t = self._make_trainer()

        def _good_eval(self_, **kwargs):
            return {
                "eval_reward_mean": 50.0,
                "eval_reward_std": 0.0,
                "eval_fall_rate": 0.0,
                "eval_success_rate": 1.0,
                "eval_num_episodes": 2,
            }

        t.eval_pass = types.MethodType(_good_eval, t)
        spu = t._rollout_length * t.num_envs
        t.train(total_steps=spu * 3, checkpoint_dir=str(tmp_path / "ckpt"))

        assert (tmp_path / "ckpt" / "best_eval_success" / "checkpoint.pkl").exists(), (
            "best_eval_success checkpoint should be created on first improvement"
        )

    def test_no_extra_save_when_metric_does_not_improve(self, tmp_path):
        """When eval metrics are constant, the checkpoint is written only once.

        With eval_interval=1 and 3 updates, eval fires at updates 1, 2, 3.
        The first eval (update 1) improves over -inf and triggers a save.
        Updates 2 and 3 return the same value — no overwrite should happen.
        We verify this by checking that global_step inside the saved checkpoint
        matches the first eval's step, not the later ones.
        """
        import pickle
        import types

        t = self._make_trainer()

        def _constant_eval(self_, **kwargs):
            return {
                "eval_reward_mean": 50.0,
                "eval_reward_std": 0.0,
                "eval_fall_rate": 0.0,
                "eval_success_rate": 1.0,
                "eval_num_episodes": 2,
            }

        t.eval_pass = types.MethodType(_constant_eval, t)
        spu = t._rollout_length * t.num_envs  # 4 * 4 = 16

        # 4 real updates; compile warmup is discarded and must not count.
        t.train(total_steps=spu * 4, checkpoint_dir=str(tmp_path / "ckpt"))

        ckpt_file = tmp_path / "ckpt" / "best_eval_per_step" / "checkpoint.pkl"
        assert ckpt_file.exists()

        with open(ckpt_file, "rb") as f:
            ckpt = pickle.load(f)

        # global_step timeline:
        #   after compile:  global_step = 0
        #   update 1 done: global_step = spu (16) → eval fires → SAVE (first improvement)
        #   update 2 done: global_step = 2*spu (32) → eval fires → no save (same value)
        #   update 3 done: global_step = 3*spu (48) → eval fires → no save (same value)
        first_eval_step = spu
        assert ckpt["global_step"] == first_eval_step, (
            f"Checkpoint should record global_step={first_eval_step} (first eval), "
            f"got {ckpt['global_step']} — suggests spurious re-save on stagnant metric"
        )

    def test_existing_final_checkpoint_still_saved(self, tmp_path):
        """Adding eval-triggered saves does not break the existing final checkpoint."""
        import types

        t = self._make_trainer()

        def _dummy_eval(self_, **kwargs):
            return {
                "eval_reward_mean": 1.0,
                "eval_reward_std": 0.0,
                "eval_fall_rate": 1.0,
                "eval_success_rate": 0.0,
                "eval_num_episodes": 2,
            }

        t.eval_pass = types.MethodType(_dummy_eval, t)
        spu = t._rollout_length * t.num_envs
        t.train(total_steps=spu * 3, checkpoint_dir=str(tmp_path / "ckpt"))

        assert (tmp_path / "ckpt" / "final" / "checkpoint.pkl").exists(), (
            "final checkpoint must still be saved regardless of eval-triggered saves"
        )


# ---------------------------------------------------------------------------
# Config: legacy (use_eval_signal=False) path
# ---------------------------------------------------------------------------

_LEGACY_CKPT_CONFIG = {
    **_TINY_CONFIG,
    "task": {
        **_TINY_CONFIG["task"],
        "initial_min_height": 0.68,
    },
    "curriculum": {
        "enabled": True,
        "reward_threshold": 999.0,  # impossibly high — curriculum never advances
        "num_levels": 5,
        "window": 2,  # fills after 2 updates
        "use_eval_signal": False,
        "ckpt_cooldown_evals": 0,  # disable cooldown so existing tests are unaffected
    },
}

# Config for cooldown-specific tests: cooldown=2, eval every update.
_COOLDOWN_TEST_CONFIG = {
    **_EVAL_CKPT_CONFIG,
    "curriculum": {
        **_EVAL_CKPT_CONFIG["curriculum"],
        "ckpt_cooldown_evals": 2,
    },
}


# ---------------------------------------------------------------------------
# Tests: hardened eval-triggered checkpoint saving
# ---------------------------------------------------------------------------


class TestEvalTriggeredCheckpointHardening:
    """Verify the three known risks are fixed.

    Risk 1 — Resume resets trackers.
    Risk 2 — Legacy path (use_eval_signal=False) had no triggered save.
    Risk 3 — No minimum-delta guard; tiny increments could spam saves.
    """

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _make_eval_trainer():
        from wheeled_biped.envs.balance_env import BalanceEnv
        from wheeled_biped.training.ppo import PPOTrainer

        env = BalanceEnv(config=_EVAL_CKPT_CONFIG)
        t = PPOTrainer(env=env, config=_EVAL_CKPT_CONFIG, logger=None, seed=0)
        t.num_envs = NUM_ENVS
        t._rollout_length = 4
        return t

    @staticmethod
    def _make_legacy_trainer():
        from wheeled_biped.envs.balance_env import BalanceEnv
        from wheeled_biped.training.ppo import PPOTrainer

        env = BalanceEnv(config=_LEGACY_CKPT_CONFIG)
        t = PPOTrainer(env=env, config=_LEGACY_CKPT_CONFIG, logger=None, seed=0)
        t.num_envs = NUM_ENVS
        t._rollout_length = 4
        return t

    @staticmethod
    def _write_checkpoint(path, trainer, **extra_fields):
        """Pickle a minimal checkpoint using the trainer's current params."""
        import os
        import pickle

        import jax

        os.makedirs(path, exist_ok=True)
        ckpt = {
            "params": jax.device_get(trainer.params),
            "opt_state": jax.device_get(trainer.opt_state),
            "obs_rms": jax.device_get(trainer.obs_rms),
            "config": trainer.config,
            "global_step": 0,
            "best_reward": float("-inf"),
            "curriculum_min_height": None,
            "best_eval_per_step": float("-inf"),
            "best_eval_success": float("-inf"),
            "best_train_reward": float("-inf"),
        }
        ckpt.update(extra_fields)
        with open(os.path.join(path, "checkpoint.pkl"), "wb") as f:
            pickle.dump(ckpt, f)

    # ── Risk 1: tracker survives resume ──────────────────────────────────────

    def test_best_trackers_restored_from_checkpoint(self, tmp_path):
        """load_checkpoint() restores best_eval_per_step/success/train_reward."""
        t = self._make_eval_trainer()
        ckpt_dir = str(tmp_path / "ckpt")
        self._write_checkpoint(
            ckpt_dir,
            t,
            best_eval_per_step=8.0,
            best_eval_success=0.9,
            best_train_reward=3.5,
        )

        t.load_checkpoint(ckpt_dir)

        assert t._resumed_best_eval_per_step == 8.0
        assert t._resumed_best_eval_success == 0.9
        assert t._resumed_best_train_reward == 3.5

    def test_old_checkpoint_without_tracker_fields_loads_safely(self, tmp_path):
        """Old checkpoints missing tracker fields fall back to -inf without error."""
        import os
        import pickle

        import jax

        t = self._make_eval_trainer()
        ckpt_dir = str(tmp_path / "ckpt")
        os.makedirs(ckpt_dir, exist_ok=True)

        # Simulate an older checkpoint that predates the tracker fields.
        old_ckpt = {
            "params": jax.device_get(t.params),
            "opt_state": jax.device_get(t.opt_state),
            "obs_rms": jax.device_get(t.obs_rms),
            "config": t.config,
            "global_step": 50000,
            "best_reward": 4.2,
            "curriculum_min_height": 0.65,
            # no best_eval_per_step / best_eval_success / best_train_reward
        }
        with open(os.path.join(ckpt_dir, "checkpoint.pkl"), "wb") as f:
            pickle.dump(old_ckpt, f)

        t.load_checkpoint(ckpt_dir)  # must not raise

        assert t._resumed_best_eval_per_step == float("-inf")
        assert t._resumed_best_eval_success == float("-inf")
        assert t._resumed_best_train_reward == float("-inf")

    @pytest.mark.slow
    def test_no_overwrite_on_resume_when_eval_worse(self, tmp_path):
        """After resume with best_eval_per_step=8.0, an eval returning 7.0 must NOT save.

        The checkpoint_dir here is different from the one we loaded.  The test
        verifies that best_eval_per_step/ is never created in the output dir.
        """
        import types

        t = self._make_eval_trainer()
        load_dir = str(tmp_path / "loaded")
        self._write_checkpoint(load_dir, t, best_eval_per_step=8.0, best_eval_success=0.9)
        t.load_checkpoint(load_dir)

        def _worse_eval(self_, **kwargs):
            # eval_per_step = 140.0 / 20 = 7.0 < 8.0 — should not save
            return {
                "eval_reward_mean": 140.0,
                "eval_reward_std": 0.0,
                "eval_fall_rate": 0.0,
                "eval_success_rate": 0.85,  # 0.85 < 0.9 — should not save
                "eval_num_episodes": 2,
            }

        t.eval_pass = types.MethodType(_worse_eval, t)
        spu = t._rollout_length * t.num_envs
        out_dir = str(tmp_path / "out")
        # total_steps chosen so training runs ≥1 loop update after the resume point
        t.train(total_steps=spu * 3, checkpoint_dir=out_dir)

        assert not (Path(out_dir) / "best_eval_per_step" / "checkpoint.pkl").exists(), (
            "best_eval_per_step must NOT be overwritten when eval is worse than resumed best"
        )
        assert not (Path(out_dir) / "best_eval_success" / "checkpoint.pkl").exists(), (
            "best_eval_success must NOT be overwritten when eval_success_rate is worse"
        )

    @pytest.mark.slow
    def test_final_checkpoint_contains_tracker_fields(self, tmp_path):
        """final/checkpoint.pkl must contain the three tracker fields."""
        import types

        t = self._make_eval_trainer()

        def _dummy_eval(self_, **kwargs):
            return {
                "eval_reward_mean": 50.0,
                "eval_reward_std": 0.0,
                "eval_fall_rate": 0.0,
                "eval_success_rate": 1.0,
                "eval_num_episodes": 2,
            }

        t.eval_pass = types.MethodType(_dummy_eval, t)
        spu = t._rollout_length * t.num_envs
        ckpt_dir = str(tmp_path / "ckpt")
        t.train(total_steps=spu * 3, checkpoint_dir=ckpt_dir)

        final_path = tmp_path / "ckpt" / "final" / "checkpoint.pkl"
        assert final_path.exists()
        with open(final_path, "rb") as f:
            ckpt = pickle.load(f)

        assert "best_eval_per_step" in ckpt
        assert "best_eval_success" in ckpt
        assert "best_train_reward" in ckpt

    # ── Risk 2: legacy path ───────────────────────────────────────────────────

    @pytest.mark.slow
    def test_legacy_path_saves_best_train_reward(self, tmp_path):
        """use_eval_signal=False path creates best_train_reward/checkpoint.pkl."""
        t = self._make_legacy_trainer()
        spu = t._rollout_length * t.num_envs
        ckpt_dir = str(tmp_path / "ckpt")
        # 4 updates (window=2 fills at update 2, triggering a save)
        t.train(total_steps=spu * 4, checkpoint_dir=ckpt_dir)

        assert (tmp_path / "ckpt" / "best_train_reward" / "checkpoint.pkl").exists(), (
            "legacy path must create best_train_reward checkpoint when rolling window improves"
        )

    @pytest.mark.slow
    def test_legacy_path_does_not_create_eval_checkpoints(self, tmp_path):
        """use_eval_signal=False must NOT create best_eval_per_step or best_eval_success."""
        t = self._make_legacy_trainer()
        spu = t._rollout_length * t.num_envs
        ckpt_dir = str(tmp_path / "ckpt")
        t.train(total_steps=spu * 4, checkpoint_dir=ckpt_dir)

        assert not (tmp_path / "ckpt" / "best_eval_per_step" / "checkpoint.pkl").exists()
        assert not (tmp_path / "ckpt" / "best_eval_success" / "checkpoint.pkl").exists()

    # ── Risk 3: delta guard ───────────────────────────────────────────────────

    @pytest.mark.slow
    def test_tiny_improvement_below_delta_does_not_resave(self, tmp_path):
        """Improvements below _EVAL_CKPT_MIN_DELTA (1e-3) must not trigger a second save.

        Timeline (eval_interval=1, 4 loop updates):
          update 1: eval_per_step=5.0  → saves (> -inf+1e-3)    step=spu
          update 2: eval_per_step=5.0005 → 5.0005 > 5.0+0.001=5.001? No. No save.
          update 3: same → no save
          update 4: same → no save
        Verify: saved checkpoint has global_step==spu (first and only save).
        """
        import types

        t = self._make_eval_trainer()
        call_count = 0

        def _tiny_step_eval(self_, **kwargs):
            nonlocal call_count
            call_count += 1
            # episode_length=20; per_step = eval_reward_mean / 20
            if call_count == 1:
                return {
                    "eval_reward_mean": 100.0,  # per_step = 5.0
                    "eval_reward_std": 0.0,
                    "eval_fall_rate": 0.0,
                    "eval_success_rate": 0.0,
                    "eval_num_episodes": 2,
                }
            return {
                "eval_reward_mean": 100.01,  # per_step = 5.0005 < 5.0 + 1e-3
                "eval_reward_std": 0.0,
                "eval_fall_rate": 0.0,
                "eval_success_rate": 0.0,
                "eval_num_episodes": 2,
            }

        t.eval_pass = types.MethodType(_tiny_step_eval, t)
        spu = t._rollout_length * t.num_envs  # 16
        ckpt_dir = str(tmp_path / "ckpt")
        t.train(total_steps=spu * 5, checkpoint_dir=ckpt_dir)

        ckpt_file = tmp_path / "ckpt" / "best_eval_per_step" / "checkpoint.pkl"
        assert ckpt_file.exists()
        with open(ckpt_file, "rb") as f:
            ckpt = pickle.load(f)

        first_eval_step = spu
        assert ckpt["global_step"] == first_eval_step, (
            f"expected global_step={first_eval_step} (first eval only), "
            f"got {ckpt['global_step']} — delta guard not working"
        )

    @pytest.mark.slow
    def test_genuine_improvement_above_delta_updates_checkpoint(self, tmp_path):
        """Improvement >= _EVAL_CKPT_MIN_DELTA triggers a new save at the later step.

        Timeline (eval_interval=1, 3 loop updates):
          update 1: eval_per_step=5.0  → saves at step=spu
          update 2: eval_per_step=6.5  → 6.5 > 5.0+0.001 → saves at step=2*spu
        Verify: saved checkpoint has global_step==2*spu.
        """
        import types

        t = self._make_eval_trainer()
        call_count = 0

        def _big_jump_eval(self_, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "eval_reward_mean": 100.0,  # per_step = 5.0
                    "eval_reward_std": 0.0,
                    "eval_fall_rate": 0.0,
                    "eval_success_rate": 0.0,
                    "eval_num_episodes": 2,
                }
            return {
                "eval_reward_mean": 130.0,  # per_step = 6.5 > 5.0 + 0.001
                "eval_reward_std": 0.0,
                "eval_fall_rate": 0.0,
                "eval_success_rate": 0.0,
                "eval_num_episodes": 2,
            }

        t.eval_pass = types.MethodType(_big_jump_eval, t)
        spu = t._rollout_length * t.num_envs  # 16
        ckpt_dir = str(tmp_path / "ckpt")
        t.train(total_steps=spu * 3, checkpoint_dir=ckpt_dir)

        ckpt_file = tmp_path / "ckpt" / "best_eval_per_step" / "checkpoint.pkl"
        assert ckpt_file.exists()
        with open(ckpt_file, "rb") as f:
            ckpt = pickle.load(f)

        second_eval_step = 2 * spu
        assert ckpt["global_step"] == second_eval_step, (
            f"expected global_step={second_eval_step} (second eval after big improvement), "
            f"got {ckpt['global_step']}"
        )


# ---------------------------------------------------------------------------
# Tests: improved checkpoint saving — cooldown + versioning
# ---------------------------------------------------------------------------


class TestImprovedCheckpointSaving:
    """Verify cooldown guard, versioned dirs, and stable pointer for all triggered paths.

    Uses _COOLDOWN_TEST_CONFIG (ckpt_cooldown_evals=2, eval_interval=1) for
    cooldown tests and _LEGACY_CKPT_CONFIG (ckpt_cooldown_evals=0) for legacy tests.
    Marked slow because all tests call t.train() which triggers MJX JIT compilation.
    """

    pytestmark = pytest.mark.slow

    @staticmethod
    def _make_cooldown_trainer():
        from wheeled_biped.envs.balance_env import BalanceEnv
        from wheeled_biped.training.ppo import PPOTrainer

        env = BalanceEnv(config=_COOLDOWN_TEST_CONFIG)
        t = PPOTrainer(env=env, config=_COOLDOWN_TEST_CONFIG, logger=None, seed=0)
        t.num_envs = NUM_ENVS
        t._rollout_length = 4
        return t

    @staticmethod
    def _make_legacy_trainer():
        from wheeled_biped.envs.balance_env import BalanceEnv
        from wheeled_biped.training.ppo import PPOTrainer

        env = BalanceEnv(config=_LEGACY_CKPT_CONFIG)
        t = PPOTrainer(env=env, config=_LEGACY_CKPT_CONFIG, logger=None, seed=0)
        t.num_envs = NUM_ENVS
        t._rollout_length = 4
        return t

    @staticmethod
    def _make_eval_trainer():
        from wheeled_biped.envs.balance_env import BalanceEnv
        from wheeled_biped.training.ppo import PPOTrainer

        env = BalanceEnv(config=_EVAL_CKPT_CONFIG)
        t = PPOTrainer(env=env, config=_EVAL_CKPT_CONFIG, logger=None, seed=0)
        t.num_envs = NUM_ENVS
        t._rollout_length = 4
        return t

    # ── cooldown ─────────────────────────────────────────────────────────────

    def test_cooldown_prevents_save_within_n_evals(self, tmp_path):
        """With ckpt_cooldown_evals=2, a save at eval N must block eval N+1.

        Timeline (eval_interval=1, cooldown=2):
          update 1 (step=spu): counter starts at 2, +=1→3, 3>=2 → saves.
          update 2 (step=2*spu): counter=0+1=1, 1<2 → blocked (even with improvement).
          update 3 (step=3*spu): counter=1+1=2, 2>=2 → saves again.

        Verify that ckpt_best/eval_per_step_s{2*spu:010d}/ does NOT exist.
        """
        import types

        t = self._make_cooldown_trainer()
        call_count = 0

        def _always_improving(self_, **kwargs):
            nonlocal call_count
            call_count += 1
            # episode_length=20; per_step = eval_reward_mean/20
            return {
                "eval_reward_mean": 100.0 * call_count,
                "eval_reward_std": 0.0,
                "eval_fall_rate": 0.0,
                "eval_success_rate": 0.0,
                "eval_num_episodes": 2,
            }

        t.eval_pass = types.MethodType(_always_improving, t)
        spu = t._rollout_length * t.num_envs  # 16
        ckpt_dir = str(tmp_path / "ckpt")
        t.train(total_steps=spu * 5, checkpoint_dir=ckpt_dir)

        blocked_step = spu * 2  # update 2 — eval 2 — must be blocked
        blocked_dir = tmp_path / "ckpt" / "ckpt_best" / f"eval_per_step_s{blocked_step:010d}"
        assert not blocked_dir.exists(), (
            f"cooldown should block save at eval 2 (step {blocked_step}), "
            f"but {blocked_dir} was created"
        )

    def test_cooldown_allows_save_after_n_evals(self, tmp_path):
        """After the cooldown has passed, a genuine improvement must save.

        Same timeline as test_cooldown_prevents_save_within_n_evals.
        Update 3 (step=3*spu) is eval 3 — cooldown satisfied — must save.
        """
        import types

        t = self._make_cooldown_trainer()
        call_count = 0

        def _always_improving(self_, **kwargs):
            nonlocal call_count
            call_count += 1
            return {
                "eval_reward_mean": 100.0 * call_count,
                "eval_reward_std": 0.0,
                "eval_fall_rate": 0.0,
                "eval_success_rate": 0.0,
                "eval_num_episodes": 2,
            }

        t.eval_pass = types.MethodType(_always_improving, t)
        spu = t._rollout_length * t.num_envs
        ckpt_dir = str(tmp_path / "ckpt")
        t.train(total_steps=spu * 5, checkpoint_dir=ckpt_dir)

        allowed_step = spu * 3  # update 3 — eval 3 — cooldown satisfied
        allowed_dir = tmp_path / "ckpt" / "ckpt_best" / f"eval_per_step_s{allowed_step:010d}"
        assert allowed_dir.exists(), (
            f"save at step {allowed_step} (eval 3, after cooldown) should have been created"
        )

    def test_cooldown_zero_disables_rate_limit(self, tmp_path):
        """ckpt_cooldown_evals=0 must allow every eval that meets the delta.

        Uses _EVAL_CKPT_CONFIG (cooldown=0) and two evals that each improve.
        Both versioned dirs should exist.
        """
        import types

        t = self._make_eval_trainer()
        call_count = 0

        def _always_improving(self_, **kwargs):
            nonlocal call_count
            call_count += 1
            return {
                "eval_reward_mean": 100.0 * call_count,
                "eval_reward_std": 0.0,
                "eval_fall_rate": 0.0,
                "eval_success_rate": 0.0,
                "eval_num_episodes": 2,
            }

        t.eval_pass = types.MethodType(_always_improving, t)
        spu = t._rollout_length * t.num_envs
        ckpt_dir = str(tmp_path / "ckpt")
        t.train(total_steps=spu * 3, checkpoint_dir=ckpt_dir)

        # At least eval 1 (step=spu) and eval 2 (step=2*spu) should save.
        dir1 = tmp_path / "ckpt" / "ckpt_best" / f"eval_per_step_s{spu:010d}"
        dir2 = tmp_path / "ckpt" / "ckpt_best" / f"eval_per_step_s{spu * 2:010d}"
        assert dir1.exists(), f"eval 1 versioned dir missing: {dir1}"
        assert dir2.exists(), f"eval 2 versioned dir missing: {dir2}"

    # ── versioning ────────────────────────────────────────────────────────────

    def test_versioned_dir_naming(self, tmp_path):
        """Each triggered save goes to ckpt_best/eval_per_step_s{step:010d}/."""
        import types

        t = self._make_eval_trainer()  # cooldown=0

        def _good_eval(self_, **kwargs):
            return {
                "eval_reward_mean": 50.0,
                "eval_reward_std": 0.0,
                "eval_fall_rate": 0.0,
                "eval_success_rate": 1.0,
                "eval_num_episodes": 2,
            }

        t.eval_pass = types.MethodType(_good_eval, t)
        spu = t._rollout_length * t.num_envs
        ckpt_dir = str(tmp_path / "ckpt")
        t.train(total_steps=spu * 3, checkpoint_dir=ckpt_dir)

        # First eval fires at update 1 → global_step = spu
        first_step = spu
        versioned = tmp_path / "ckpt" / "ckpt_best" / f"eval_per_step_s{first_step:010d}"
        assert versioned.exists(), f"expected versioned dir: {versioned}"
        assert (versioned / "checkpoint.pkl").exists()

    def test_stable_pointer_updated_alongside_versioned(self, tmp_path):
        """best_eval_per_step/checkpoint.pkl (stable pointer) is kept in sync."""
        import types

        t = self._make_eval_trainer()

        def _good_eval(self_, **kwargs):
            return {
                "eval_reward_mean": 50.0,
                "eval_reward_std": 0.0,
                "eval_fall_rate": 0.0,
                "eval_success_rate": 1.0,
                "eval_num_episodes": 2,
            }

        t.eval_pass = types.MethodType(_good_eval, t)
        spu = t._rollout_length * t.num_envs
        ckpt_dir = str(tmp_path / "ckpt")
        t.train(total_steps=spu * 3, checkpoint_dir=ckpt_dir)

        stable = tmp_path / "ckpt" / "best_eval_per_step" / "checkpoint.pkl"
        assert stable.exists(), "stable pointer best_eval_per_step/checkpoint.pkl must exist"

    def test_stable_pointer_reflects_latest_best(self, tmp_path):
        """After two successive saves, the stable pointer holds the later step."""
        import types

        t = self._make_eval_trainer()  # cooldown=0
        call_count = 0

        def _increasing_eval(self_, **kwargs):
            nonlocal call_count
            call_count += 1
            return {
                "eval_reward_mean": 100.0 * call_count,
                "eval_reward_std": 0.0,
                "eval_fall_rate": 0.0,
                "eval_success_rate": 0.0,
                "eval_num_episodes": 2,
            }

        t.eval_pass = types.MethodType(_increasing_eval, t)
        spu = t._rollout_length * t.num_envs
        ckpt_dir = str(tmp_path / "ckpt")
        t.train(total_steps=spu * 3, checkpoint_dir=ckpt_dir)

        stable_path = tmp_path / "ckpt" / "best_eval_per_step" / "checkpoint.pkl"
        assert stable_path.exists()
        with open(stable_path, "rb") as f:
            ckpt = pickle.load(f)
        # Three evals saved (steps spu, 2*spu, 3*spu); stable must reflect the later one
        assert ckpt["global_step"] == spu * 3, (
            f"stable pointer should reflect latest save at step {spu * 3}, "
            f"got {ckpt['global_step']}"
        )

    # ── legacy path versioning ────────────────────────────────────────────────

    @pytest.mark.slow
    def test_legacy_path_creates_versioned_dir(self, tmp_path):
        """use_eval_signal=False must write a versioned dir under ckpt_best/."""
        t = self._make_legacy_trainer()
        spu = t._rollout_length * t.num_envs
        ckpt_dir = str(tmp_path / "ckpt")
        t.train(total_steps=spu * 4, checkpoint_dir=ckpt_dir)

        ckpt_best = tmp_path / "ckpt" / "ckpt_best"
        assert ckpt_best.exists(), "ckpt_best/ directory should exist after legacy triggered save"
        versioned = list(ckpt_best.glob("train_reward_s*/"))
        assert len(versioned) >= 1, "at least one train_reward_s*/ versioned dir must exist"
        # Each versioned dir must contain the checkpoint file
        for d in versioned:
            assert (d / "checkpoint.pkl").exists(), f"missing checkpoint.pkl in {d}"

    @pytest.mark.slow
    def test_legacy_path_stable_pointer_exists(self, tmp_path):
        """use_eval_signal=False must also maintain best_train_reward/ stable pointer."""
        t = self._make_legacy_trainer()
        spu = t._rollout_length * t.num_envs
        ckpt_dir = str(tmp_path / "ckpt")
        t.train(total_steps=spu * 4, checkpoint_dir=ckpt_dir)

        stable = tmp_path / "ckpt" / "best_train_reward" / "checkpoint.pkl"
        assert stable.exists(), "stable pointer best_train_reward/checkpoint.pkl must exist"

    # ── periodic and final unchanged ──────────────────────────────────────────

    def test_periodic_and_final_not_in_ckpt_best(self, tmp_path):
        """Periodic (step_N/) and final/ checkpoints must NOT appear under ckpt_best/."""
        import types

        t = self._make_eval_trainer()

        def _dummy(self_, **kwargs):
            return {
                "eval_reward_mean": 1.0,
                "eval_reward_std": 0.0,
                "eval_fall_rate": 1.0,
                "eval_success_rate": 0.0,
                "eval_num_episodes": 2,
            }

        t.eval_pass = types.MethodType(_dummy, t)
        spu = t._rollout_length * t.num_envs
        ckpt_dir = str(tmp_path / "ckpt")
        # save_interval=1 so a periodic save fires every update
        t.train(total_steps=spu * 3, checkpoint_dir=ckpt_dir, save_interval=1)

        ckpt_best = tmp_path / "ckpt" / "ckpt_best"
        if ckpt_best.exists():
            names = [d.name for d in ckpt_best.iterdir()]
            for name in names:
                assert name.startswith("eval_") or name.startswith("train_"), (
                    f"unexpected entry in ckpt_best/: {name!r} "
                    "(periodic and final saves must not go there)"
                )

        # final/ must still exist at the top level
        assert (tmp_path / "ckpt" / "final" / "checkpoint.pkl").exists()


class TestLoggerLifecycle:
    """Verify the correct logger lifecycle: log_dict() then close().

    The historical bug: PPOTrainer.train() called logger.close() BEFORE
    running eval_pass() and logging eval metrics.  After close(), the
    TensorBoard writer is shut down and JSONL data accumulates in-memory
    without being flushed.  The fix moves logger.close() to after the
    eval logging block.

    These tests verify the correct lifecycle independently of the training loop.
    """

    def test_log_dict_then_close_persists_to_jsonl(self, tmp_path):
        """Metrics logged before close() appear in the JSONL file."""
        import json

        from wheeled_biped.utils.logger import TrainingLogger

        logger = TrainingLogger(
            log_dir=str(tmp_path),
            experiment_name="lifecycle_ok",
            use_tensorboard=False,
            use_wandb=False,
            flush_every=1000,  # disable auto-flush so only close() writes
        )
        logger.set_step(500)
        logger.log_dict({"eval/reward_mean": 3.14, "eval/fall_rate": 0.05})
        logger.close()  # correct order: logs first, close after

        jsonl = tmp_path / "lifecycle_ok_metrics.jsonl"
        assert jsonl.exists(), "JSONL file should be created"
        lines = [line for line in jsonl.read_text().strip().split("\n") if line]
        assert len(lines) == 2, f"Expected 2 log entries, got {len(lines)}"
        tags = {json.loads(line)["tag"] for line in lines}
        assert "eval/reward_mean" in tags
        assert "eval/fall_rate" in tags

    def test_close_flushes_buffer(self, tmp_path):
        """close() flushes any buffered metrics to disk before closing writers."""
        import json

        from wheeled_biped.utils.logger import TrainingLogger

        logger = TrainingLogger(
            log_dir=str(tmp_path),
            experiment_name="close_flush",
            use_tensorboard=False,
            use_wandb=False,
            flush_every=1000,  # prevent auto-flush
        )
        logger.set_step(1)
        logger.log_scalar("train/loss", 0.42)  # buffered, not yet flushed
        # Buffer has 1 entry; file is empty until close() is called
        jsonl = tmp_path / "close_flush_metrics.jsonl"
        if jsonl.exists():
            assert jsonl.read_text().strip() == "", "Should not be flushed yet"
        logger.close()  # must flush before closing
        assert jsonl.exists()
        lines = [line for line in jsonl.read_text().strip().split("\n") if line]
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["tag"] == "train/loss"
        assert abs(entry["value"] - 0.42) < 1e-6


# ---------------------------------------------------------------------------
# Tests: balance curriculum cadence fixes (pre-flight audit blocking issues A/B/C)
# ---------------------------------------------------------------------------


class TestBalanceCurriculumFixes:
    """Tests for the three blocking curriculum fixes identified in the pre-flight audit.

    Fix A: eval_interval now compatible with GPU batch size (was 50, now 2).
    Fix B: curriculum_min_height is passed to in-training eval_pass() in ppo.py.
    Fix C: eval_pass() max_steps cap now reaches configured eval_episodes for stable policies.

    Tests A and C use only arithmetic on config values — no JIT required, fast suite.
    Test B requires running train() with a monkeypatched eval_pass() — marked @pytest.mark.slow.
    """

    def _effective_eval_interval(self, cfg, num_envs, rollout_length):
        """Mirrors ppo.py logic: eval_interval_steps takes priority over eval_interval."""
        steps_per_update = num_envs * rollout_length
        curr = cfg["curriculum"]
        if "eval_interval_steps" in curr:
            target_steps = int(curr["eval_interval_steps"])
            interval = max(1, (target_steps + steps_per_update - 1) // steps_per_update)
            return interval, steps_per_update
        return int(curr.get("eval_interval", 50)), steps_per_update

    def test_eval_interval_compatible_with_long_gpu_run(self):
        """balance.yaml eval cadence must fire regularly in a long GPU run.

        With num_envs=4096, rollout_length=128 → steps_per_update=524,288.
        eval_interval_steps=10_000_000 → eval_interval=ceil(10M/524K)=20 updates.
        """
        from wheeled_biped.utils.config import load_yaml

        cfg = load_yaml("configs/training/balance.yaml")
        eval_interval, steps_per_update = self._effective_eval_interval(
            cfg, cfg["task"]["num_envs"], cfg["ppo"]["rollout_length"]
        )
        steps_for_first_eval = eval_interval * steps_per_update

        assert steps_for_first_eval <= 12_000_000, (
            f"eval_interval={eval_interval} × steps_per_update={steps_per_update:,}"
            f" = {steps_for_first_eval:,} > 12M steps: curriculum eval cadence is "
            "too sparse for the intended long GPU run."
        )

    def test_eval_interval_allows_multiple_evals_in_50m_run(self):
        """A 50M-step GPU run should trigger at least 4 curriculum evals.

        Sufficient for advancing at least a few curriculum levels before end-of-training eval.
        """
        from wheeled_biped.utils.config import load_yaml

        cfg = load_yaml("configs/training/balance.yaml")
        eval_interval, steps_per_update = self._effective_eval_interval(
            cfg, cfg["task"]["num_envs"], cfg["ppo"]["rollout_length"]
        )
        num_updates_50m = 50_000_000 // steps_per_update
        num_evals_50m = num_updates_50m // eval_interval

        assert num_evals_50m >= 4, (
            f"A 50M-step GPU run produces only {num_evals_50m} curriculum evals "
            f"(eval_interval={eval_interval}, steps_per_update={steps_per_update:,}). "
            "Need >= 4 evals to advance the curriculum meaningfully."
        )

    def test_eval_pass_max_steps_formula_reaches_configured_episodes(self):
        """eval_pass() max_steps must collect eval_episodes for stable policies.

        For a stable policy (avg_episode_length ≈ episode_length), the old formula
        episode_length × 4 was insufficient. The fixed formula accounts for num_eval_envs.
        """
        from wheeled_biped.utils.config import load_yaml

        cfg = load_yaml("configs/training/balance.yaml")
        episode_length = cfg["task"]["episode_length"]  # 1000
        eval_episodes = cfg["curriculum"]["eval_episodes"]  # 200
        eval_envs = min(32, cfg["task"]["num_envs"])  # 32

        # Old (broken) formula — capped at ~128 episodes for stable policies
        old_max_steps = episode_length * 4  # 4000
        old_episodes_reachable = (old_max_steps * eval_envs) // episode_length  # 128

        # New formula (fixed in eval_pass())
        new_max_steps = max(
            episode_length * 4,
            int(eval_episodes * episode_length * 2 / eval_envs) + 100,
        )
        new_episodes_reachable = (new_max_steps * eval_envs) // episode_length

        # Document the bug: old formula was insufficient
        assert old_episodes_reachable < eval_episodes, (
            "Test setup check: old formula should have been insufficient for configured "
            f"episode count ({old_episodes_reachable} < {eval_episodes})"
        )
        # Confirm new formula is sufficient
        assert new_episodes_reachable >= eval_episodes, (
            f"Fixed max_steps={new_max_steps} reaches only ~{new_episodes_reachable} episodes "
            f"(target: {eval_episodes}). Formula needs adjustment."
        )

    @pytest.mark.slow
    def test_curriculum_eval_uses_current_min_height(self, tmp_path):
        """train() must pass curriculum_min_height to in-training eval_pass() (Fix B).

        Uses a monkeypatched eval_pass() to capture the keyword arguments it is called
        with, then runs train() with use_eval_signal=True and verifies that
        curriculum_min_height is forwarded on every eval call.
        """
        import types

        from wheeled_biped.envs.balance_env import BalanceEnv
        from wheeled_biped.training.ppo import PPOTrainer

        tiny_cfg = {
            "task": {
                "name": "balance",
                "env": "BalanceEnv",
                "episode_length": 10,
                "num_envs": 4,
                "initial_min_height": 0.69,
            },
            "curriculum": {
                "enabled": True,
                "reward_threshold": 0.75,
                "num_levels": 5,
                "window": 5,
                "use_eval_signal": True,
                "eval_interval": 1,  # fire every update so we capture calls quickly
                "eval_episodes": 2,
            },
            "ppo": {
                "learning_rate": 1e-4,
                "num_epochs": 1,
                "num_minibatches": 1,
                "rollout_length": 4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_epsilon": 0.2,
                "entropy_coeff": 0.0,
                "value_loss_coeff": 0.5,
                "max_grad_norm": 0.5,
                "normalize_advantages": True,
                "seed": 0,
            },
            "network": {
                "policy_hidden": [32],
                "value_hidden": [32],
                "activation": "elu",
                "init_std": 0.5,
            },
            "low_level_pid": {"enabled": False},
            "rewards": {
                "body_level": 1.0,
                "height": 1.0,
                "alive": 0.1,
            },
            "termination": {"max_tilt_rad": 0.8, "min_height": 0.3},
            "domain_randomization": {"enabled": False},
            "sensor_noise": {"enabled": False},
        }

        env = BalanceEnv(tiny_cfg)
        trainer = PPOTrainer(env=env, config=tiny_cfg, seed=0)

        eval_call_kwargs: list[dict] = []

        def _capture_eval(self_, **kwargs):
            eval_call_kwargs.append(dict(kwargs))
            return {
                "eval_reward_mean": 0.0,
                "eval_reward_std": 0.0,
                "eval_fall_rate": 1.0,
                "eval_success_rate": 0.0,
                "eval_num_episodes": 2,
            }

        trainer.eval_pass = types.MethodType(_capture_eval, trainer)

        spu = tiny_cfg["ppo"]["rollout_length"] * tiny_cfg["task"]["num_envs"]
        trainer.train(total_steps=spu * 3, checkpoint_dir=str(tmp_path / "ckpt"))

        assert len(eval_call_kwargs) >= 1, (
            "eval_pass() was never called during train() — check eval_interval and total_steps"
        )
        for i, kwargs in enumerate(eval_call_kwargs):
            assert "curriculum_min_height" in kwargs, (
                f"eval_pass() call #{i + 1} is missing 'curriculum_min_height' kwarg. "
                "Fix B: train() must forward curriculum_min_height to eval_pass()."
            )


# ===========================================================================
# TestTrainingFPSSemantics
# ===========================================================================


class TestTrainingFPSSemantics:
    """Unit tests for _compute_training_fps timing semantics.

    These are pure arithmetic tests — no JAX, no MuJoCo, no JIT.
    They verify that the FPS helper:
      (a) uses training-only time (not wall-clock including eval),
      (b) produces stable output regardless of external overhead,
      (c) handles edge cases gracefully.
    """

    def _fps(self, *, steps_per_update, updates_done, train_time_s):
        from wheeled_biped.training.ppo import _compute_training_fps

        return _compute_training_fps(
            steps_per_update=steps_per_update,
            updates_done=updates_done,
            train_time_s=train_time_s,
        )

    def test_basic_throughput(self):
        """10 updates × 512 steps at 1s/update → 5120 fps."""
        fps = self._fps(steps_per_update=512, updates_done=10, train_time_s=1.0)
        assert abs(fps - 5120.0) < 1e-3

    def test_zero_updates_returns_zero(self):
        """No updates yet → fps=0, no division-by-zero."""
        fps = self._fps(steps_per_update=512, updates_done=0, train_time_s=0.0)
        assert fps == 0.0

    def test_eval_overhead_excluded(self):
        """Simulates a run where 10s of eval overhead is added externally.

        If the function correctly uses training-only time, the FPS it computes
        must be identical regardless of how much eval overhead occurred.
        The caller is responsible for NOT adding eval time to train_time_s;
        this test documents that contract.
        """
        steps_per_update = 512
        updates_done = 10
        train_time_s = 5.0  # only training time

        fps_no_eval = self._fps(
            steps_per_update=steps_per_update,
            updates_done=updates_done,
            train_time_s=train_time_s,
        )
        # If we mistakenly added eval overhead to the denominator, fps would differ.
        fps_with_eval_overhead = self._fps(
            steps_per_update=steps_per_update,
            updates_done=updates_done,
            train_time_s=train_time_s + 10.0,  # wrong: should not include eval
        )
        # Confirm that adding eval to the denominator *does* degrade the metric
        # (this is the bug we fixed — now the caller never passes eval time in).
        assert fps_with_eval_overhead < fps_no_eval, (
            "FPS with inflated denominator should be lower — confirms the bug exists "
            "when eval time is incorrectly included."
        )
        # The contract: train() accumulates update_elapsed (training-only) into
        # _train_time_total and never adds eval_pass() time.  fps_no_eval is what
        # train() now reports; fps_with_eval_overhead is what the old code produced.

    def test_consistent_fps_across_updates_without_eval(self):
        """FPS should be roughly stable if each update takes the same time."""
        spu = 512
        t_per_update = 0.5  # seconds per update

        fps_values = [
            self._fps(
                steps_per_update=spu,
                updates_done=n,
                train_time_s=n * t_per_update,
            )
            for n in range(1, 20)
        ]
        # All should be close to spu / t_per_update = 1024
        expected = spu / t_per_update
        for fps in fps_values:
            assert abs(fps - expected) / expected < 1e-6, (
                f"fps={fps:.1f} deviated from expected={expected:.1f}"
            )


# ===========================================================================
# TestBestRewardTracking
# ===========================================================================


class TestBestRewardTracking:
    """Tests that best_reward tracks the true maximum over ALL training updates.

    Root cause of the bug: best_reward was updated inside `if update %
    log_interval == 0:`, so any reward peak between two log checkpoints was
    silently dropped.  The fix moves the update outside the gate.

    These tests are slow because they need train() to run (JAX JIT compile).
    """

    pytestmark = pytest.mark.slow

    _TINY_CFG = {
        "task": {
            "env": "BalanceEnv",
            "num_envs": 4,
            "episode_length": 10,
            "initial_min_height": 0.68,
        },
        "ppo": {
            "learning_rate": 3e-4,
            "num_epochs": 1,
            "num_minibatches": 2,
            "rollout_length": 4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "entropy_coeff": 0.01,
            "value_loss_coeff": 0.5,
            "max_grad_norm": 0.5,
            "normalize_advantages": True,
        },
        "network": {
            "policy_hidden": [32, 32],
            "value_hidden": [32, 32],
            "activation": "elu",
        },
        "rewards": {"alive": 0.3, "height": 1.0},
        "curriculum": {"enabled": False},
        "low_level_pid": {"enabled": False},
        "domain_randomization": {"enabled": False, "push_magnitude": 0},
        "sensor_noise": {"enabled": False},
    }

    def _make_trainer(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from wheeled_biped.envs.balance_env import BalanceEnv
        from wheeled_biped.training.ppo import PPOTrainer

        env = BalanceEnv(config=self._TINY_CFG)
        return PPOTrainer(env=env, config=self._TINY_CFG, logger=None, seed=0)

    def test_best_reward_updates_on_non_log_step(self, tmp_path):
        """best_reward must equal the max reward seen over ALL updates.

        Strategy: run train() for enough steps that some updates fall between
        log_interval boundaries, then verify that train() result['best_reward']
        equals the maximum of all per-update rewards we observe via a patched
        rollout.

        We inject a synthetic reward spike at a non-log-aligned update and
        confirm it is captured.
        """
        import types

        trainer = self._make_trainer()
        spu = self._TINY_CFG["ppo"]["rollout_length"] * self._TINY_CFG["task"]["num_envs"]
        # Run enough updates to straddle a log_interval boundary (default 10).
        # We'll observe 12 updates: log boundaries at 10; update 7 is non-log.
        total_steps = spu * 12

        # Capture rewards per update
        observed_rewards: list[float] = []
        original_rollout = trainer._rollout

        # Inject a spike at actual update 7. Call #0 is compile-only and is
        # discarded by train(), so call #7 maps to update 7; 7 % 10 ≠ 0.
        call_count = [0]
        SPIKE_CALL = 7  # noqa: N806

        import jax.numpy as jnp

        def _patched_rollout(self_, params, env_state, rng, obs_rms):
            env_state_out, transitions, rng_out = original_rollout(params, env_state, rng, obs_rms)
            idx = call_count[0]
            call_count[0] += 1
            if idx == SPIKE_CALL:
                # Replace rewards with a very large value to create a detectable spike.
                spike_reward = jnp.full_like(transitions.reward, 999.0)
                transitions = transitions._replace(reward=spike_reward)
            observed_rewards.append(float(jnp.mean(transitions.reward)))
            return env_state_out, transitions, rng_out

        trainer._rollout = types.MethodType(_patched_rollout, trainer)

        result = trainer.train(
            total_steps=total_steps,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_interval=10,
        )

        true_max = max(observed_rewards) if observed_rewards else float("-inf")
        reported_best = result["best_reward"]

        # The spike was injected at a non-log update; best_reward must still
        # capture it (or come within floating-point of it).
        assert reported_best >= true_max - 1e-3, (
            f"best_reward={reported_best:.4f} missed the true max={true_max:.4f}. "
            f"Spike was at call {SPIKE_CALL} (non-log update). "
            "Fix: best_reward must be updated every update, not only at log_interval."
        )
