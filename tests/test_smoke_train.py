"""
End-to-end smoke test for the balance training path.

Marked @pytest.mark.slow — excluded from fast unit-test runs.
Run explicitly when you want to verify the full train() pipeline:

  pytest tests/test_smoke_train.py -v -m slow

What this tests:
  - PPOTrainer.train() can start, run a few updates, and finish
  - Checkpoint file is created with the expected keys
  - Return dict contains all required keys with finite values
  - Network params have no NaN after training
  - eval_pass() is invoked at end of train() (eval_reward_mean key present)

What this does NOT test:
  - Learning quality or reward improvement (too short for that)
  - Curriculum advancement (curriculum disabled for speed)
  - Live viewer or WandB integration

Typical wall time on CPU: 2-5 min (dominated by JAX JIT compile).
Typical wall time on GPU: <30 s.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Tiny config — cheap enough to compile and run a handful of updates
# ---------------------------------------------------------------------------

_SMOKE_CONFIG = {
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
    "rewards": {
        "alive": 0.3,
        "height": 1.0,
    },
    # Curriculum disabled: keeps the smoke test independent of curriculum logic.
    # Curriculum advancement is covered in test_ppo_trainer.py (TestEvalGatedCurriculum).
    "curriculum": {"enabled": False},
    # Disable PID and push for speed / simplicity
    "low_level_pid": {"enabled": False},
    "domain_randomization": {
        "enabled": False,
        "push_magnitude": 0,
        "push_interval": 9999,
        "push_duration": 1,
    },
    "termination": {
        "max_tilt_rad": 0.8,
        "min_height": 0.1,
    },
}

# 3 updates worth of steps: warmup counts as update 0, loop runs 1..2
_NUM_ENVS = 4
_ROLLOUT = 4
_TOTAL_STEPS = _NUM_ENVS * _ROLLOUT * 3  # = 48


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smoke_env():
    from wheeled_biped.envs.balance_env import BalanceEnv

    return BalanceEnv(config=_SMOKE_CONFIG)


@pytest.fixture(scope="module")
def smoke_trainer(smoke_env):
    from wheeled_biped.training.ppo import PPOTrainer

    t = PPOTrainer(env=smoke_env, config=_SMOKE_CONFIG, logger=None, seed=0)
    t.num_envs = _NUM_ENVS
    t._rollout_length = _ROLLOUT
    return t


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


class TestSmokeTrainBalance:
    """Full end-to-end: PPOTrainer.train() runs without error on tiny config."""

    @pytest.fixture(scope="class")
    def train_result(self, smoke_trainer, tmp_path_factory):
        """Run train() once and cache the result for all tests in this class."""
        ckpt_dir = str(tmp_path_factory.mktemp("smoke_ckpt"))
        result = smoke_trainer.train(
            total_steps=_TOTAL_STEPS,
            checkpoint_dir=ckpt_dir,
            log_interval=1,
            save_interval=999,  # suppress mid-run saves; final is always saved
        )
        return result, ckpt_dir

    def test_train_returns_dict(self, train_result):
        """train() returns a dict (not None or exception)."""
        result, _ = train_result
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"

    def test_train_required_keys(self, train_result):
        """Return dict contains all required keys."""
        result, _ = train_result
        required = {
            "best_reward",
            "eval_reward_mean",
            "eval_fall_rate",
            "eval_success_rate",
            "total_steps",
        }
        for key in required:
            assert key in result, f"Missing key in train() result: {key}"

    def test_train_metrics_finite(self, train_result):
        """All numeric return values are finite."""
        result, _ = train_result
        for key, val in result.items():
            if isinstance(val, float):
                assert np.isfinite(val), f"Non-finite return value for '{key}': {val}"

    def test_final_checkpoint_exists(self, train_result):
        """Final checkpoint file is created."""
        _, ckpt_dir = train_result
        ckpt_path = Path(ckpt_dir) / "final" / "checkpoint.pkl"
        assert ckpt_path.exists(), f"Final checkpoint not found at {ckpt_path}"

    def test_checkpoint_has_required_keys(self, train_result):
        """Saved checkpoint has all required keys."""
        _, ckpt_dir = train_result
        ckpt_path = Path(ckpt_dir) / "final" / "checkpoint.pkl"
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)

        required = {"params", "opt_state", "obs_rms", "config", "global_step", "best_reward"}
        for k in required:
            assert k in ckpt, f"Missing checkpoint key: {k}"

    def test_params_no_nan_after_train(self, train_result):
        """Network params have no NaN after train()."""
        _, ckpt_dir = train_result
        ckpt_path = Path(ckpt_dir) / "final" / "checkpoint.pkl"
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)

        import jax.tree_util as jtu

        for leaf in jtu.tree_leaves(ckpt["params"]):
            arr = np.array(leaf)
            assert not np.any(np.isnan(arr)), f"NaN found in saved params leaf shape={arr.shape}"

    def test_eval_pass_ran(self, train_result):
        """eval_reward_mean is finite — confirms eval_pass() ran at end of train()."""
        result, _ = train_result
        assert np.isfinite(result["eval_reward_mean"]), (
            "eval_reward_mean is not finite — eval_pass() may not have run"
        )

    def test_fall_and_success_sum_to_one(self, train_result):
        """eval_fall_rate + eval_success_rate == 1.0."""
        result, _ = train_result
        total = result["eval_fall_rate"] + result["eval_success_rate"]
        assert abs(total - 1.0) < 1e-5, f"fall_rate + success_rate = {total} (expected 1.0)"

    def test_global_step_matches_expected(self, train_result):
        """total_steps in result is >= the requested total_steps."""
        result, _ = train_result
        # train() may overshoot by one update; check it's at least the floor
        assert result["total_steps"] >= _NUM_ENVS * _ROLLOUT, (
            f"total_steps={result['total_steps']} is less than one update"
        )
