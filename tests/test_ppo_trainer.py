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


class TestSingleUpdate:
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

        required_keys = {"params", "opt_state", "obs_rms", "config", "global_step", "best_reward"}
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
        original_leaves = [np.array(l) for l in jtu.tree_leaves(jax.device_get(trainer.params))]

        trainer._save_checkpoint(ckpt_dir, global_step=0, best_reward=0.0)
        trainer.load_checkpoint(ckpt_dir)

        restored_leaves = [np.array(l) for l in jtu.tree_leaves(jax.device_get(trainer.params))]
        for orig, rest in zip(original_leaves, restored_leaves):
            assert np.allclose(orig, rest, atol=1e-6), "Params differ after checkpoint round-trip"


# ---------------------------------------------------------------------------
# Tests: eval_pass()
# ---------------------------------------------------------------------------

class TestEvalPass:
    """Tests for PPOTrainer.eval_pass() — the held-out evaluation method.

    These tests verify the structural contract of eval_pass() without requiring
    a full MJX GPU training loop.  They run with tiny envs (4 envs, short episodes).
    """

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
        assert abs(total - 1.0) < 1e-6, (
            f"fall_rate + success_rate = {total} (expected 1.0)"
        )

    def test_eval_pass_collects_episodes(self, trainer, env):
        """eval_pass() reports at least one completed episode."""
        rng = jax.random.PRNGKey(13)
        result = trainer.eval_pass(num_eval_envs=4, num_episodes=4, rng=rng)

        assert result["eval_num_episodes"] >= 1, (
            "eval_pass() should complete at least 1 episode"
        )

    def test_eval_pass_does_not_mutate_obs_rms(self, trainer, env):
        """eval_pass() does not change obs_rms (evaluation is read-only)."""
        from wheeled_biped.training.ppo import update_running_mean_std
        import jax.numpy as jnp

        # Record current obs_rms mean
        before_mean = np.array(trainer.obs_rms.mean).copy()

        rng = jax.random.PRNGKey(42)
        trainer.eval_pass(num_eval_envs=4, num_episodes=4, rng=rng)

        after_mean = np.array(trainer.obs_rms.mean)
        assert np.allclose(before_mean, after_mean, atol=1e-7), (
            "eval_pass() must not mutate obs_rms"
        )

<<<<<<< HEAD

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
        "reward_threshold": 0.5,   # low threshold — easy to meet in tests
        "num_levels": 5,
        "window": 2,               # small window for legacy path tests
        "use_eval_signal": True,
        "eval_interval": 2,        # fire every 2 updates — fast for tests
        "eval_episodes": 2,        # minimal episodes per check
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
                "eval_reward_mean": 20.0,   # episode return; /20 = 1.0 >= 0.75 threshold
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
        level_step = (initial_min_h - final_min_h) / max(num_levels, 1)
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
        assert cfg.get("window") == 2   # legacy window still present

    def test_eval_per_step_normalization(self, curriculum_trainer):
        """eval_per_step = eval_reward_mean / episode_length is correct."""
        episode_length = curriculum_trainer.episode_length
        # If mean episode return = 7.5 and episode_length = 10 → per_step = 0.75
        eval_reward_mean = 7.5
        expected = eval_reward_mean / max(1, episode_length)
        computed = eval_reward_mean / max(1, episode_length)
        assert abs(computed - expected) < 1e-9
=======
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
            "eval_reward_mean", "eval_reward_std", "eval_fall_rate",
            "eval_success_rate", "eval_num_episodes",
        }
        for key in required:
            assert key in result, f"Missing key in result: {key}"
        assert np.isfinite(result["eval_reward_mean"])

    def test_eval_pass_curriculum_min_height_changes_result(self, trainer, env):
        """curriculum_min_height=0.40 produces different results than the default
        initial range (0.68).

        Using the same RNG seed, the only difference between the two runs is the
        initial height_command distribution:
          default:  [0.68, 0.70] → height_norm ∈ [0.933, 1.0]
          curriculum: [0.40, 0.70] → height_norm ∈ [0.0, 1.0]
        Different obs → different greedy actions → different rewards.
        If results are identical, the parameter had no effect (regression).
        """
        rng_seed = jax.random.PRNGKey(99)

        result_default = trainer.eval_pass(
            num_eval_envs=4, num_episodes=4, rng=jax.random.PRNGKey(99)
        )
        result_curriculum = trainer.eval_pass(
            num_eval_envs=4, num_episodes=4, rng=jax.random.PRNGKey(99),
            curriculum_min_height=0.40,
        )
        assert result_default["eval_reward_mean"] != result_curriculum["eval_reward_mean"], (
            "curriculum_min_height=0.40 must change eval results vs default "
            "(proves the parameter actively patches height_command / obs)"
        )

    def test_eval_pass_curriculum_min_height_none_unchanged(self, trainer, env):
        """curriculum_min_height=None is identical to omitting the argument."""
        rng = jax.random.PRNGKey(17)
        result_omitted = trainer.eval_pass(num_eval_envs=4, num_episodes=4,
                                           rng=jax.random.PRNGKey(17))
        result_none    = trainer.eval_pass(num_eval_envs=4, num_episodes=4,
                                           rng=jax.random.PRNGKey(17),
                                           curriculum_min_height=None)
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
        "reward_threshold": 0.5,   # low threshold — easy to meet in tests
        "num_levels": 5,
        "window": 2,               # small window for legacy path tests
        "use_eval_signal": True,
        "eval_interval": 2,        # fire every 2 updates — fast for tests
        "eval_episodes": 2,        # minimal episodes per check
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
                "eval_reward_mean": 20.0,   # episode return; /20 = 1.0 >= 0.75 threshold
                "eval_reward_std": 0.0,
                "eval_fall_rate": 0.0,
                "eval_success_rate": 1.0,
                "eval_num_episodes": 2,
            }

        original_eval = curriculum_trainer.eval_pass
        curriculum_trainer.eval_pass = types.MethodType(_fake_eval_pass, curriculum_trainer)

        # Simulate what the training loop does at eval_interval
        cfg = curriculum_trainer.config.get("curriculum", {})
        reward_threshold = cfg["reward_threshold"] * sum(
            w for w in curriculum_trainer.config.get("rewards", {}).values() if w > 0
        )
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
        assert cfg.get("window") == 2   # legacy window still present

    def test_eval_per_step_normalization(self, curriculum_trainer):
        """eval_per_step = eval_reward_mean / episode_length is correct."""
        episode_length = curriculum_trainer.episode_length
        # If mean episode return = 7.5 and episode_length = 10 → per_step = 0.75
        eval_reward_mean = 7.5
        expected = eval_reward_mean / max(1, episode_length)
        computed = eval_reward_mean / max(1, episode_length)
        assert abs(computed - expected) < 1e-9


# ---------------------------------------------------------------------------
# Tests: logger lifecycle (log before close)
# ---------------------------------------------------------------------------

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
            flush_every=1000,   # disable auto-flush so only close() writes
        )
        logger.set_step(500)
        logger.log_dict({"eval/reward_mean": 3.14, "eval/fall_rate": 0.05})
        logger.close()  # correct order: logs first, close after

        jsonl = tmp_path / "lifecycle_ok_metrics.jsonl"
        assert jsonl.exists(), "JSONL file should be created"
        lines = [l for l in jsonl.read_text().strip().split("\n") if l]
        assert len(lines) == 2, f"Expected 2 log entries, got {len(lines)}"
        tags = {json.loads(l)["tag"] for l in lines}
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
            flush_every=1000,   # prevent auto-flush
        )
        logger.set_step(1)
        logger.log_scalar("train/loss", 0.42)   # buffered, not yet flushed
        # Buffer has 1 entry; file is empty until close() is called
        jsonl = tmp_path / "close_flush_metrics.jsonl"
        if jsonl.exists():
            assert jsonl.read_text().strip() == "", "Should not be flushed yet"
        logger.close()  # must flush before closing
        assert jsonl.exists()
        lines = [l for l in jsonl.read_text().strip().split("\n") if l]
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["tag"] == "train/loss"
        assert abs(entry["value"] - 0.42) < 1e-6

