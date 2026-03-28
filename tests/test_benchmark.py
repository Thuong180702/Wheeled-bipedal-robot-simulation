"""
Tests for wheeled_biped.eval.benchmark.

Strategy
--------
These tests cover:
  - BenchmarkResult creation and JSON serialisation
  - Mode registry / validation (run_benchmark raises on unknown mode)
  - _base_metrics() aggregation logic (pure numpy, no JAX)
  - Each mode function's extra metric keys (via stub env + model)
  - Mode-specific patching and restore behaviour

All heavy JAX / MuJoCo operations are replaced with minimal stubs so the
tests remain fast (< 5 s total) and CI-safe.
"""

from __future__ import annotations

import json
import math
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers: stubs for JAX / env / model
# ---------------------------------------------------------------------------


def _make_fake_jax_array(val: float, size: int = 1):
    """Return a tiny numpy array that behaves like a JAX array for our purposes."""
    return np.full(size, val, dtype=np.float32)


def _make_stub_env(
    num_envs: int = 4,
    episode_length: int = 10,
    fall_on_episode: int | None = None,
    has_height_command: bool = True,
):
    """Build a fake env that terminates episodes after ``episode_length`` steps."""
    env = MagicMock()
    env.MIN_HEIGHT_CMD = 0.40
    env.MAX_HEIGHT_CMD = 0.70
    env._initial_min_height = 0.40
    env._push_enabled = True
    env._push_magnitude = 20.0

    step_counter = {"n": 0}

    def _fake_v_reset(rng, n_env):
        step_counter["n"] = 0
        info: dict[str, Any] = {
            "is_fallen": np.zeros(n_env, dtype=bool),
            "time_limit": np.zeros(n_env, dtype=bool),
        }
        if has_height_command:
            info["height_command"] = np.full(n_env, 0.55, dtype=np.float32)
        state = MagicMock()
        state.obs = np.zeros((n_env, 40), dtype=np.float32)
        state.reward = np.zeros(n_env, dtype=np.float32)
        state.done = np.zeros(n_env, dtype=bool)
        state.info = info
        state.mjx_data = MagicMock()
        state.mjx_data.qpos = np.tile([0, 0, 0.55, 1, 0, 0, 0] + [0.0] * 10, (n_env, 1))
        return state

    def _fake_v_step(states, actions):
        step_counter["n"] += 1
        n_env = len(states.done)
        is_fallen = fall_on_episode is not None and step_counter["n"] % fall_on_episode == 0
        done = np.array([step_counter["n"] >= episode_length] * n_env)
        info: dict[str, Any] = {
            "is_fallen": np.full(n_env, is_fallen, dtype=bool),
            "time_limit": done & ~np.full(n_env, is_fallen, dtype=bool),
        }
        if has_height_command:
            info["height_command"] = np.full(n_env, 0.55, dtype=np.float32)
        state = MagicMock()
        state.obs = np.zeros((n_env, 40), dtype=np.float32)
        state.reward = np.ones(n_env, dtype=np.float32)
        state.done = done
        state.info = info
        state.mjx_data = MagicMock()
        state.mjx_data.qpos = np.tile([0, 0, 0.55, 1, 0, 0, 0] + [0.0] * 10, (n_env, 1))
        return state

    def _fake_v_reset_if_done(states, rng):
        return _fake_v_reset(rng, len(states.done))

    env.v_reset.side_effect = _fake_v_reset
    env.v_step.side_effect = _fake_v_step
    env.v_reset_if_done.side_effect = _fake_v_reset_if_done
    env.mj_model = MagicMock()
    env.mj_model.body_mass = np.array([1.0, 2.0, 3.0])
    env.mj_model.geom_friction = np.array([[0.5, 0.5, 0.5]])
    env.mjx_model = MagicMock()
    return env


def _make_stub_model():
    """Return a model whose .apply() returns deterministic zero actions."""
    model = MagicMock()
    dist = MagicMock()
    dist.loc = np.zeros((4, 10), dtype=np.float32)  # (num_envs, num_actions)
    value = np.zeros((4, 1), dtype=np.float32)
    model.apply.return_value = (dist, value)
    return model


def _fake_obs_rms():
    rms = MagicMock()
    rms.mean = np.zeros(40)
    rms.var = np.ones(40)
    return rms


# ---------------------------------------------------------------------------
# Patch helpers: replace JAX primitives in the benchmark module with numpy ops
# ---------------------------------------------------------------------------


def _patch_benchmark(monkeypatch):
    """Patch JAX-heavy imports inside benchmark.py with numpy equivalents."""

    # normalize_obs: just return obs unchanged for tests
    monkeypatch.setattr(
        "wheeled_biped.eval.benchmark.normalize_obs",
        lambda obs, rms: obs,
        raising=False,
    )

    # jax.random.split: return two copies of the same key (np array)
    fake_rng = np.array([0, 1], dtype=np.uint32)

    def _fake_split(key, *args):
        if args:
            return [np.array([i, i + 1], dtype=np.uint32) for i in range(args[0])]
        return fake_rng, fake_rng

    monkeypatch.setattr("wheeled_biped.eval.benchmark.jax", MagicMock(), raising=False)

    return fake_rng


# ---------------------------------------------------------------------------
# Module-level imports (done lazily to avoid JAX import at collection time)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_jax_in_benchmark(monkeypatch):
    """Patch module-level jax, jnp, mjx, normalize_obs in benchmark so tests run without real JAX."""  # noqa: E501
    import wheeled_biped.eval.benchmark as bm

    fake_jax = MagicMock()
    fake_rng = np.array([0, 1], dtype=np.uint32)
    fake_jax.random.split.return_value = (fake_rng, fake_rng)

    fake_jnp = MagicMock()
    fake_jnp.zeros = np.zeros
    fake_jnp.where = np.where
    fake_jnp.int32 = np.int32  # prevent MagicMock recursion when used as dtype

    fake_mjx = MagicMock()
    fake_mjx.put_model.return_value = MagicMock()

    monkeypatch.setattr(bm, "jax", fake_jax)
    monkeypatch.setattr(bm, "jnp", fake_jnp)
    monkeypatch.setattr(bm, "mjx", fake_mjx)
    monkeypatch.setattr(bm, "normalize_obs", lambda obs, rms: obs)
    return fake_rng


# ---------------------------------------------------------------------------
# BenchmarkResult tests
# ---------------------------------------------------------------------------


class TestBenchmarkResult:
    def test_to_dict_is_json_serialisable(self):
        from wheeled_biped.eval.benchmark import BenchmarkResult

        result = BenchmarkResult(
            mode="nominal",
            num_episodes=10,
            reward_mean=1.5,
            success_rate=0.8,
            fall_rate=0.2,
            mode_metrics={"foo": 1.0},
        )
        d = result.to_dict()
        # Must serialise without error
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        assert loaded["mode"] == "nominal"
        assert loaded["reward_mean"] == pytest.approx(1.5)
        assert loaded["mode_metrics"]["foo"] == pytest.approx(1.0)

    def test_default_fields_are_zero(self):
        from wheeled_biped.eval.benchmark import BenchmarkResult

        result = BenchmarkResult(mode="test", num_episodes=0)
        assert result.reward_mean == 0.0
        assert result.fall_rate == 0.0
        assert result.mode_metrics == {}

    def test_mode_metrics_with_nested_list(self):
        from wheeled_biped.eval.benchmark import BenchmarkResult

        result = BenchmarkResult(
            mode="command_tracking",
            num_episodes=5,
            mode_metrics={"per_command": [{"height_command": 0.55, "height_rmse": 0.02}]},
        )
        d = result.to_dict()
        assert d["mode_metrics"]["per_command"][0]["height_rmse"] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# Mode registry / validation
# ---------------------------------------------------------------------------


class TestModeRegistry:
    def test_modes_tuple_contains_all_four(self):
        from wheeled_biped.eval.benchmark import MODES

        assert "nominal" in MODES
        assert "push_recovery" in MODES
        assert "domain_randomized" in MODES
        assert "command_tracking" in MODES

    def test_run_benchmark_raises_on_unknown_mode(self):
        from wheeled_biped.eval.benchmark import run_benchmark

        with pytest.raises(ValueError, match="Unknown benchmark mode"):
            run_benchmark(
                "nonexistent_mode",
                env=MagicMock(),
                model=MagicMock(),
                params=MagicMock(),
                obs_rms=_fake_obs_rms(),
                rng=np.array([0, 1]),
                num_episodes=1,
                num_envs=1,
                max_steps=10,
            )


# ---------------------------------------------------------------------------
# _base_metrics helper
# ---------------------------------------------------------------------------


class TestBaseMetrics:
    def test_all_survive(self):
        from wheeled_biped.eval.benchmark import _base_metrics

        rewards = [1.0, 2.0, 3.0]
        lengths = [100, 100, 100]
        fallen = [False, False, False]
        timed_out = [True, True, True]  # all episodes reached time limit
        m = _base_metrics(rewards, lengths, fallen, timed_out)

        assert m["success_rate"] == pytest.approx(1.0)
        assert m["fall_rate"] == pytest.approx(0.0)
        assert m["timeout_rate"] == pytest.approx(1.0)
        assert m["reward_mean"] == pytest.approx(2.0)
        assert m["episode_length_max"] == 100

    def test_all_fallen(self):
        from wheeled_biped.eval.benchmark import _base_metrics

        rewards = [0.5, 0.3]
        lengths = [30, 50]
        fallen = [True, True]
        timed_out = [False, False]  # both fell; time_limit not reached
        m = _base_metrics(rewards, lengths, fallen, timed_out)

        assert m["fall_rate"] == pytest.approx(1.0)
        assert m["success_rate"] == pytest.approx(0.0)
        assert m["timeout_rate"] == pytest.approx(0.0)

    def test_percentiles(self):
        from wheeled_biped.eval.benchmark import _base_metrics

        rewards = list(range(101))  # 0..100
        lengths = [50] * 101
        fallen = [False] * 101
        timed_out = [True] * 101
        m = _base_metrics(rewards, lengths, fallen, timed_out)

        assert m["reward_p50"] == pytest.approx(50.0)
        assert m["reward_p5"] == pytest.approx(5.0)
        assert m["reward_p95"] == pytest.approx(95.0)


# ---------------------------------------------------------------------------
# Success semantics: regression tests for the old length >= max_steps bug
# ---------------------------------------------------------------------------


class TestSuccessSemantics:
    """Verify that success_rate uses env time_limit, NOT length >= benchmark max_steps.

    Old bug
    -------
    _base_metrics computed success_rate = mean(length >= max_steps).
    If the env's internal episode_length (e.g. 500 steps) was SMALLER than
    the benchmark's max_steps (e.g. 1000), a perfect policy that never fell
    and always timed out at step 500 would get success_rate = 0.0.

    New behaviour
    -------------
    success_rate = mean(time_limit) from info, so the episode's actual
    termination reason drives the metric regardless of max_steps.
    """

    def test_old_bug_scenario_now_correct(self):
        """Env episode_length=500 steps, benchmark max_steps=1000.

        Under the old logic: length(500) < max_steps(1000) → success=0.
        Under the new logic: time_limit=True  → success=1.
        """
        from wheeled_biped.eval.benchmark import _base_metrics

        rewards = [10.0, 12.0, 11.0]
        lengths = [500, 500, 500]  # env timed out at 500 steps
        fallen = [False, False, False]
        timed_out = [True, True, True]  # info["time_limit"] was True

        m = _base_metrics(rewards, lengths, fallen, timed_out)

        # New: correct — all episodes reached the env time limit
        assert m["success_rate"] == pytest.approx(1.0), (
            "success_rate should be 1.0 when all episodes reached time_limit, "
            "regardless of benchmark max_steps"
        )
        assert m["fall_rate"] == pytest.approx(0.0)

    def test_mixed_fall_and_timeout(self):
        """2 fell, 3 timed out → success_rate = 0.6, fall_rate = 0.4."""
        from wheeled_biped.eval.benchmark import _base_metrics

        rewards = [5.0] * 5
        lengths = [100, 200, 300, 400, 500]
        fallen = [True, True, False, False, False]
        timed_out = [False, False, True, True, True]

        m = _base_metrics(rewards, lengths, fallen, timed_out)

        assert m["success_rate"] == pytest.approx(0.6)
        assert m["fall_rate"] == pytest.approx(0.4)
        assert m["timeout_rate"] == pytest.approx(0.6)  # synonymous with success_rate

    def test_success_rate_equals_timeout_rate(self):
        """Invariant: success_rate == timeout_rate always."""
        import random

        from wheeled_biped.eval.benchmark import _base_metrics

        rng = random.Random(42)
        n = 20
        fallen = [rng.random() < 0.3 for _ in range(n)]
        timed_out = [not f for f in fallen]
        m = _base_metrics(
            episode_rewards=[1.0] * n,
            episode_lengths=[100] * n,
            episode_fallen=fallen,
            episode_timed_out=timed_out,
        )
        assert m["success_rate"] == pytest.approx(m["timeout_rate"])


# ---------------------------------------------------------------------------
# Nominal mode
# ---------------------------------------------------------------------------


class TestNominalMode:
    def test_returns_standard_fields(self, monkeypatch):
        """_run_nominal should return a BenchmarkResult with all base fields."""
        import wheeled_biped.eval.benchmark as bm
        from wheeled_biped.eval.benchmark import _run_nominal

        # Stub _rollout to return a controlled set of episodes without needing jnp.clip
        fake_ep_r = [1.0, 2.0, 3.0, 1.5]
        fake_ep_l = [100, 50, 200, 150]
        fake_ep_f = [False, True, False, False]
        fake_ep_t = [True, False, True, True]  # time_limit flags
        monkeypatch.setattr(  # noqa: E501
            bm, "_rollout", lambda *a, **kw: (fake_ep_r, fake_ep_l, fake_ep_f, fake_ep_t, [])
        )

        result = _run_nominal(
            env=MagicMock(),
            model=MagicMock(),
            params=MagicMock(),
            obs_rms=_fake_obs_rms(),
            rng=np.array([0, 1]),
            num_episodes=4,
            num_envs=2,
            max_steps=200,
        )

        assert result.mode == "nominal"
        assert result.num_episodes == 4
        assert result.fall_rate == pytest.approx(0.25)  # 1/4
        assert result.reward_mean == pytest.approx(1.875)  # mean([1,2,3,1.5])
        assert hasattr(result, "timeout_rate")
        assert hasattr(result, "success_rate")

    def test_result_is_json_serialisable(self):
        from wheeled_biped.eval.benchmark import BenchmarkResult

        result = BenchmarkResult(
            mode="nominal",
            num_episodes=10,
            reward_mean=2.5,
            fall_rate=0.1,
            timeout_rate=0.9,
        )
        d = result.to_dict()
        json.dumps(d)  # must not raise


# ---------------------------------------------------------------------------
# Push recovery mode — patching behaviour
# ---------------------------------------------------------------------------


class TestPushRecoveryMode:
    def test_env_push_attrs_restored_after_run(self, monkeypatch):
        """Env push attrs must be restored to their originals after run, even on exception."""
        import wheeled_biped.eval.benchmark as bm
        from wheeled_biped.eval.benchmark import _run_push_recovery

        env = MagicMock()
        env._push_enabled = False  # deliberately set to False
        env._push_magnitude = 5.0

        fake_ep_r = [1.0, 2.0]
        fake_ep_l = [50, 60]
        fake_ep_f = [True, False]

        # Stub _rollout; also verify push attrs were changed DURING the call
        push_enabled_during = []
        push_mag_during = []

        def _recording_rollout(*a, **kw):
            push_enabled_during.append(env._push_enabled)
            push_mag_during.append(env._push_magnitude)
            return fake_ep_r, fake_ep_l, fake_ep_f, [False, True], []

        monkeypatch.setattr(bm, "_rollout", _recording_rollout)

        result = _run_push_recovery(
            env=env,
            model=MagicMock(),
            params=MagicMock(),
            obs_rms=_fake_obs_rms(),
            rng=np.array([0, 1]),
            num_episodes=2,
            num_envs=2,
            max_steps=100,
            push_magnitude=120.0,
        )

        # Push was enabled and magnitude patched DURING the rollout
        assert push_enabled_during == [True]
        assert push_mag_during == [pytest.approx(120.0)]
        # Attrs restored AFTER the run
        assert env._push_enabled is False
        assert env._push_magnitude == pytest.approx(5.0)
        # Result has expected mode-specific keys
        assert result.mode_metrics["push_magnitude_used"] == pytest.approx(120.0)
        assert "fall_after_push_rate" in result.mode_metrics

    def test_mode_metrics_keys(self):
        from wheeled_biped.eval.benchmark import BenchmarkResult

        result = BenchmarkResult(
            mode="push_recovery",
            num_episodes=10,
            mode_metrics={
                "push_magnitude_used": 80.0,
                "fall_after_push_rate": 0.3,
                "mean_steps_to_fall": 42.0,
            },
        )
        assert "push_magnitude_used" in result.mode_metrics
        assert "fall_after_push_rate" in result.mode_metrics
        assert "mean_steps_to_fall" in result.mode_metrics
        json.dumps(result.to_dict())  # serialisable


# ---------------------------------------------------------------------------
# Domain randomized mode — patching behaviour
# ---------------------------------------------------------------------------


class TestDomainRandomizedMode:
    def test_mj_model_attrs_restored_after_run(self, monkeypatch):
        """mj_model mass/friction must be restored to original values after run."""
        import wheeled_biped.eval.benchmark as bm
        from wheeled_biped.eval.benchmark import _run_domain_randomized

        env = MagicMock()
        env.mj_model = MagicMock()
        env.mj_model.body_mass = np.array([1.0, 2.0, 3.0])
        env.mj_model.geom_friction = np.array([[0.5, 0.5, 0.5]])
        env.mjx_model = MagicMock()
        orig_mass = env.mj_model.body_mass.copy()
        orig_friction = env.mj_model.geom_friction.copy()

        # Track what mass/friction were during rollout
        mass_during = []
        friction_during = []

        def _recording_rollout(*a, **kw):
            mass_during.append(env.mj_model.body_mass.copy())
            friction_during.append(env.mj_model.geom_friction.copy())
            return [1.0, 2.0], [50, 60], [False, False], [True, True], []

        monkeypatch.setattr(bm, "_rollout", _recording_rollout)

        result = _run_domain_randomized(
            env=env,
            model=MagicMock(),
            params=MagicMock(),
            obs_rms=_fake_obs_rms(),
            rng=np.array([0, 1]),
            num_episodes=2,
            num_envs=2,
            max_steps=100,
        )

        # Mass/friction should have been perturbed DURING rollout
        assert not np.allclose(mass_during[0], orig_mass)  # perturbed
        # And restored AFTER rollout
        np.testing.assert_array_almost_equal(env.mj_model.body_mass, orig_mass)
        np.testing.assert_array_almost_equal(env.mj_model.geom_friction, orig_friction)
        assert result.mode == "domain_randomized"
        assert "mass_perturb_pct" in result.mode_metrics

    def test_mode_metrics_keys(self):
        from wheeled_biped.eval.benchmark import BenchmarkResult

        result = BenchmarkResult(
            mode="domain_randomized",
            num_episodes=10,
            mode_metrics={
                "mass_perturb_pct": 0.3,
                "friction_perturb_pct": 0.5,
                "height_error_mean": 0.05,
            },
        )
        assert "height_error_mean" in result.mode_metrics
        assert "mass_perturb_pct" in result.mode_metrics
        json.dumps(result.to_dict())


# ---------------------------------------------------------------------------
# Command tracking mode
# ---------------------------------------------------------------------------


class TestCommandTrackingMode:
    def test_per_command_results_present(self):
        from wheeled_biped.eval.benchmark import BenchmarkResult

        per_cmd = [
            {
                "height_command": h,
                "height_rmse": 0.02,
                "success_rate": 0.9,
                "fall_rate": 0.1,
                "reward_mean": 2.0,
                "num_episodes": 10,
            }
            for h in [0.40, 0.55, 0.70]
        ]
        result = BenchmarkResult(
            mode="command_tracking",
            num_episodes=30,
            mode_metrics={
                "height_commands": [0.40, 0.55, 0.70],
                "overall_height_rmse": 0.02,
                "per_command": per_cmd,
            },
        )
        assert len(result.mode_metrics["per_command"]) == 3
        assert result.mode_metrics["overall_height_rmse"] == pytest.approx(0.02)
        json.dumps(result.to_dict())

    def test_height_rmse_is_finite(self):
        per_cmd = [
            {
                "height_command": 0.55,
                "height_rmse": 0.03,
                "success_rate": 1.0,
                "fall_rate": 0.0,
                "reward_mean": 3.0,
                "num_episodes": 5,
            },
        ]
        from wheeled_biped.eval.benchmark import BenchmarkResult

        result = BenchmarkResult(
            mode="command_tracking",
            num_episodes=5,
            mode_metrics={"per_command": per_cmd, "overall_height_rmse": 0.03},
        )
        assert math.isfinite(result.mode_metrics["overall_height_rmse"])
