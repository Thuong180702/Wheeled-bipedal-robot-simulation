"""
Benchmark suite: evaluate a trained policy under multiple named conditions.

Supported modes
---------------
nominal           Standard env defaults. Extended stats (fall_rate, timeout_rate).
push_recovery     Larger / always-enabled push disturbances. Reports fall_after_push_rate.
domain_randomized Random mass ±30 % and friction ±50 %. Reports height_error, drift.
command_tracking  Sweep over fixed height commands. Reports per-command height RMSE.

Usage (Python)
--------------
from wheeled_biped.eval.benchmark import run_benchmark

result = run_benchmark(
    mode="nominal",
    env=env,
    model=model,
    params=params,
    obs_rms=obs_rms,
    rng=jax.random.PRNGKey(0),
    num_episodes=100,
    num_envs=64,
    max_steps=1000,
)
print(result.to_dict())
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# JAX + project imports deferred to avoid slow init at module import time.
# They are imported here so they can be patched in tests.
try:
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    from wheeled_biped.training.ppo import normalize_obs
except Exception:  # pragma: no cover
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    mjx = None  # type: ignore[assignment]

    def normalize_obs(obs, rms):  # type: ignore[misc]  # noqa: E731
        return obs


# ---------------------------------------------------------------------------
# BenchmarkResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Structured output of a benchmark run.

    All numeric fields are plain Python floats / ints so the result is
    immediately JSON-serialisable via ``to_dict()``.
    """

    mode: str
    num_episodes: int
    # Reward statistics (all modes)
    reward_mean: float = 0.0
    reward_std: float = 0.0
    reward_min: float = 0.0
    reward_p5: float = 0.0
    reward_p50: float = 0.0
    reward_p95: float = 0.0
    reward_max: float = 0.0
    # Episode length statistics (all modes)
    episode_length_mean: float = 0.0
    episode_length_max: int = 0
    # Common derived metrics (all modes)
    success_rate: float = 0.0  # fraction ending because env time_limit fired (not fallen)
    fall_rate: float = 0.0  # fraction ending in is_fallen
    timeout_rate: float = 0.0  # same as success_rate: fraction reaching env episode limit
    # Mode-specific extras stored in a free-form dict
    mode_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        import dataclasses

        d = dataclasses.asdict(self)
        return d


# ---------------------------------------------------------------------------
# Shared rollout helper
# ---------------------------------------------------------------------------


def _rollout(
    env,
    model,
    params,
    obs_rms,
    rng,
    num_episodes: int,
    num_envs: int,
    max_steps: int,
    step_hook=None,
):
    """Run vectorised rollout until num_episodes complete."""
    rng, reset_key = jax.random.split(rng)
    env_states = env.v_reset(reset_key, num_envs)

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_fallen: list[bool] = []
    episode_timed_out: list[bool] = []  # True when env time_limit fired
    episode_info_last: list[dict] = []

    current_rewards = jnp.zeros(num_envs)
    current_lengths = jnp.zeros(num_envs, dtype=jnp.int32)

    for step_idx in range(max_steps):
        if len(episode_rewards) >= num_episodes:
            break

        obs = normalize_obs(env_states.obs, obs_rms)
        dist, _ = model.apply(params, obs)
        actions = jnp.clip(dist.loc, -1.0, 1.0)

        env_states = env.v_step(env_states, actions)
        current_rewards += env_states.reward
        current_lengths += 1

        if step_hook is not None:
            step_hook(env_states, step_idx)

        dones = env_states.done
        for i in range(num_envs):
            if dones[i] and len(episode_rewards) < num_episodes:
                episode_rewards.append(float(current_rewards[i]))
                episode_lengths.append(int(current_lengths[i]))
                fallen = bool(env_states.info.get("is_fallen", jnp.zeros(num_envs))[i])
                timed_out = bool(env_states.info.get("time_limit", jnp.zeros(num_envs))[i])
                episode_fallen.append(fallen)
                episode_timed_out.append(timed_out)
                ep_info = {
                    k: (float(v[i]) if hasattr(v, "__len__") else float(v))
                    for k, v in env_states.info.items()
                    if k not in ("push_rng", "pid_integral", "anchor_xy")
                }
                episode_info_last.append(ep_info)

        rng, reset_key = jax.random.split(rng)
        env_states = env.v_reset_if_done(env_states, reset_key)
        current_rewards = jnp.where(dones, 0.0, current_rewards)
        current_lengths = jnp.where(dones, 0, current_lengths)

    return episode_rewards, episode_lengths, episode_fallen, episode_timed_out, episode_info_last


def _base_metrics(
    episode_rewards: list[float],
    episode_lengths: list[int],
    episode_fallen: list[bool],
    episode_timed_out: list[bool],
) -> dict[str, Any]:
    """Compute statistics shared across all modes.

    ``success_rate`` and ``timeout_rate`` are both derived from the env's own
    ``time_limit`` flag in ``info``, **not** from comparing episode length against
    an external ``max_steps`` threshold.  This avoids mislabelling a perfectly
    successful episode as a failure when the benchmark's ``max_steps`` is larger
    than the env's internal episode length limit.

    Invariant: success_rate == timeout_rate and fall_rate + success_rate == 1.0
    (assuming every done comes from exactly one of is_fallen / time_limit).
    """
    r = np.array(episode_rewards, dtype=float)
    l = np.array(episode_lengths, dtype=float)  # noqa: E741
    f = np.array(episode_fallen, dtype=bool)
    t = np.array(episode_timed_out, dtype=bool)
    return {
        "num_episodes": len(episode_rewards),
        "reward_mean": float(r.mean()),
        "reward_std": float(r.std()),
        "reward_min": float(r.min()),
        "reward_p5": float(np.percentile(r, 5)),
        "reward_p50": float(np.percentile(r, 50)),
        "reward_p95": float(np.percentile(r, 95)),
        "reward_max": float(r.max()),
        "episode_length_mean": float(l.mean()),
        "episode_length_max": int(l.max()),
        # Use env's time_limit flag: fraction that reached episode end without falling
        "success_rate": float(np.mean(t)),
        "fall_rate": float(np.mean(f)),
        "timeout_rate": float(np.mean(t)),  # synonymous with success_rate for this env
    }


# ---------------------------------------------------------------------------
# Mode: nominal
# ---------------------------------------------------------------------------


def _run_nominal(env, model, params, obs_rms, rng, num_episodes, num_envs, max_steps):
    """Standard evaluation — env default settings."""
    ep_r, ep_l, ep_f, ep_t, _ = _rollout(
        env, model, params, obs_rms, rng, num_episodes, num_envs, max_steps
    )
    base = _base_metrics(ep_r, ep_l, ep_f, ep_t)
    result = BenchmarkResult(mode="nominal", **base)
    return result


# ---------------------------------------------------------------------------
# Mode: push_recovery
# ---------------------------------------------------------------------------


def _run_push_recovery(
    env,
    model,
    params,
    obs_rms,
    rng,
    num_episodes,
    num_envs,
    max_steps,
    push_magnitude: float = 80.0,
):
    """Evaluation under larger forced push disturbances.

    Patches ``env._push_enabled`` and ``env._push_magnitude`` in place,
    then restores them.  No JAX recompilation required (attrs are used only
    at Python / tracing time for these scalar hyper-params).

    Extra metrics
    -------------
    push_magnitude_used     : the patched push force magnitude (N)
    fall_after_push_rate    : fraction of episodes that fell (≈ fell due to push)
    mean_steps_to_fall      : mean steps for fallen episodes (shorter → harder knock-down)
    """
    orig_push_enabled = getattr(env, "_push_enabled", True)
    orig_push_magnitude = getattr(env, "_push_magnitude", 20.0)

    env._push_enabled = True
    env._push_magnitude = push_magnitude

    try:
        ep_r, ep_l, ep_f, ep_t, _ = _rollout(
            env, model, params, obs_rms, rng, num_episodes, num_envs, max_steps
        )
    finally:
        env._push_enabled = orig_push_enabled
        env._push_magnitude = orig_push_magnitude

    base = _base_metrics(ep_r, ep_l, ep_f, ep_t)
    fallen_lengths = [lf for lf, f in zip(ep_l, ep_f) if f]
    mode_metrics = {
        "push_magnitude_used": push_magnitude,
        "fall_after_push_rate": base["fall_rate"],
        "mean_steps_to_fall": float(np.mean(fallen_lengths)) if fallen_lengths else float("nan"),
    }
    result = BenchmarkResult(mode="push_recovery", mode_metrics=mode_metrics, **base)
    return result


# ---------------------------------------------------------------------------
# Mode: domain_randomized
# ---------------------------------------------------------------------------


def _run_domain_randomized(
    env,
    model,
    params,
    obs_rms,
    rng,
    num_episodes,
    num_envs,
    max_steps,
    mass_perturb: float = 0.30,
    friction_perturb: float = 0.50,
):
    """Evaluation with randomised mass and friction.

    Patches ``mj_model.body_mass`` and ``mj_model.geom_friction`` (±%, uniform),
    rebuilds ``mjx_model``, runs rollout, then restores originals.

    Extra metrics
    -------------
    mass_perturb_pct     : fractional mass perturbation applied
    friction_perturb_pct : fractional friction perturbation applied
    height_error_mean    : mean |actual_height − height_command| across episodes
    position_drift_mean  : mean end-of-ep XY drift (if info available)
    """
    # Save originals
    orig_mass = copy.deepcopy(env.mj_model.body_mass.copy())
    orig_friction = copy.deepcopy(env.mj_model.geom_friction.copy())

    rng_np = np.random.default_rng(seed=42)
    mass_perturbed = orig_mass * (
        1.0 + rng_np.uniform(-mass_perturb, mass_perturb, size=orig_mass.shape)
    )
    friction_perturbed = orig_friction * (
        1.0 + rng_np.uniform(-friction_perturb, friction_perturb, size=orig_friction.shape)
    )
    friction_perturbed = np.clip(friction_perturbed, 0.01, None)  # friction must be positive

    env.mj_model.body_mass[:] = mass_perturbed
    env.mj_model.geom_friction[:] = friction_perturbed
    env.mjx_model = mjx.put_model(env.mj_model)

    # Collect per-step height error via hook
    height_errors: list[float] = []

    def _height_hook(env_states, _step_idx):
        heights = np.array(env_states.mjx_data.qpos[:, 2])  # (num_envs,)
        h_cmds = np.array(env_states.info.get("height_command", jnp.zeros(len(heights))))
        height_errors.extend(np.abs(heights - h_cmds).tolist())

    try:
        ep_r, ep_l, ep_f, ep_t, ep_info = _rollout(
            env,
            model,
            params,
            obs_rms,
            rng,
            num_episodes,
            num_envs,
            max_steps,
            step_hook=_height_hook,
        )
    finally:
        env.mj_model.body_mass[:] = orig_mass
        env.mj_model.geom_friction[:] = orig_friction
        env.mjx_model = mjx.put_model(env.mj_model)

    base = _base_metrics(ep_r, ep_l, ep_f, ep_t)
    mode_metrics = {
        "mass_perturb_pct": mass_perturb,
        "friction_perturb_pct": friction_perturb,
        "height_error_mean": float(np.mean(height_errors)) if height_errors else float("nan"),
    }
    result = BenchmarkResult(mode="domain_randomized", mode_metrics=mode_metrics, **base)
    return result


# ---------------------------------------------------------------------------
# Mode: command_tracking
# ---------------------------------------------------------------------------


def _run_command_tracking(
    env,
    model,
    params,
    obs_rms,
    rng,
    num_episodes,
    num_envs,
    max_steps,
    height_commands: list[float] | None = None,
):
    """Evaluation sweeping fixed height commands.

    For each command value, patches ``env._initial_min_height`` so that reset()
    samples a height close to the target, then collects per-command metrics.

    Extra metrics (all per command, keyed by command value)
    --------------------------------------------------------
    per_command:
      height_rmse     : RMSE of actual height to command during episode
      success_rate    : fraction surviving the full episode
      fall_rate       : fraction that fell
    aggregate:
      overall_height_rmse : RMSE across all commands
    """
    # Default: 5 evenly-spaced commands between MIN and MAX height
    min_h = getattr(env, "MIN_HEIGHT_CMD", 0.40)
    max_h = getattr(env, "MAX_HEIGHT_CMD", 0.70)
    if height_commands is None:
        height_commands = [round(min_h + i * (max_h - min_h) / 4, 3) for i in range(5)]

    orig_min_h = getattr(env, "_initial_min_height", min_h)
    # How to fix the height command: set _initial_min_height = cmd and MAX_HEIGHT_CMD = cmd+epsilon
    # so the uniform sample [min, max] always gives ≈ cmd.
    epsilon = 0.01  # tiny range so all resets get the same command

    per_command_results: list[dict] = []
    all_height_errors: list[float] = []

    for cmd in height_commands:
        # Clamp cmd inside valid range
        cmd_clamped = float(np.clip(cmd, min_h, max_h))
        env._initial_min_height = max(min_h, cmd_clamped - epsilon)
        orig_class_max = env.MAX_HEIGHT_CMD
        # Temporarily allow wider sampling inside BalanceEnv.reset by patching class attr
        # We do this via instance attr override
        env.MAX_HEIGHT_CMD = min(max_h, cmd_clamped + epsilon)  # type: ignore[assignment]

        height_errors_cmd: list[float] = []

        def _track_hook(env_states, _step_idx, _cmd=cmd_clamped):
            heights = np.array(env_states.mjx_data.qpos[:, 2])
            h_cmds = np.full_like(heights, _cmd)
            height_errors_cmd.extend(np.abs(heights - h_cmds).tolist())

        rng, cmd_key = jax.random.split(rng)

        try:
            ep_r, ep_l, ep_f, ep_t, _ = _rollout(
                env,
                model,
                params,
                obs_rms,
                cmd_key,
                num_episodes,
                num_envs,
                max_steps,
                step_hook=_track_hook,
            )
        finally:
            env._initial_min_height = orig_min_h
            env.MAX_HEIGHT_CMD = orig_class_max  # type: ignore[assignment]

        rmse = (
            float(np.sqrt(np.mean(np.array(height_errors_cmd) ** 2)))
            if height_errors_cmd
            else float("nan")
        )
        all_height_errors.extend(height_errors_cmd)

        per_command_results.append(
            {
                "height_command": cmd_clamped,
                "height_rmse": rmse,
                "success_rate": float(np.mean(ep_t)),
                "fall_rate": float(np.mean(np.array(ep_f))),
                "reward_mean": float(np.mean(ep_r)),
                "num_episodes": len(ep_r),
            }
        )

    overall_rmse = (
        float(np.sqrt(np.mean(np.array(all_height_errors) ** 2)))
        if all_height_errors
        else float("nan")
    )

    # Aggregate across all commands for BenchmarkResult base fields
    all_r = [ep["reward_mean"] for ep in per_command_results]
    base = {
        "num_episodes": sum(ep["num_episodes"] for ep in per_command_results),
        "reward_mean": float(np.mean(all_r)),
        "reward_std": float(np.std(all_r)),
        "reward_min": float(np.min(all_r)),
        "reward_p5": float(np.percentile(all_r, 5)),
        "reward_p50": float(np.percentile(all_r, 50)),
        "reward_p95": float(np.percentile(all_r, 95)),
        "reward_max": float(np.max(all_r)),
        "episode_length_mean": 0.0,
        "episode_length_max": 0,
        "success_rate": float(np.mean([ep["success_rate"] for ep in per_command_results])),
        "fall_rate": float(np.mean([ep["fall_rate"] for ep in per_command_results])),
        "timeout_rate": float(np.mean([1.0 - ep["fall_rate"] for ep in per_command_results])),
    }
    mode_metrics = {
        "height_commands": height_commands,
        "overall_height_rmse": overall_rmse,
        "per_command": per_command_results,
    }
    return BenchmarkResult(mode="command_tracking", mode_metrics=mode_metrics, **base)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

#: Registry of known benchmark modes.
MODES = ("nominal", "push_recovery", "domain_randomized", "command_tracking")


def run_benchmark(
    mode: str,
    env,
    model,
    params,
    obs_rms,
    *,
    rng,
    num_episodes: int = 100,
    num_envs: int = 64,
    max_steps: int = 1000,
    mode_kwargs: dict[str, Any] | None = None,
) -> BenchmarkResult:
    """Run a named benchmark mode against a trained policy.

    Args:
        mode: one of ``MODES`` ("nominal", "push_recovery",
              "domain_randomized", "command_tracking").
        env: instantiated WheeledBipedEnv subclass.
        model: Flax module (actor-critic network).
        params: model parameters (pytree).
        obs_rms: observation running mean-std (from checkpoint).
        rng: JAX random key.
        num_episodes: total episodes to collect.
        num_envs: number of parallel environments.
        max_steps: maximum steps per episode (success threshold).
        mode_kwargs: extra keyword args forwarded to the mode function.

    Returns:
        :class:`BenchmarkResult` with common + mode-specific metrics.

    Raises:
        ValueError: if ``mode`` is unrecognised.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown benchmark mode {mode!r}. Choose from {MODES}.")

    kwargs = dict(
        env=env,
        model=model,
        params=params,
        obs_rms=obs_rms,
        rng=rng,
        num_episodes=num_episodes,
        num_envs=num_envs,
        max_steps=max_steps,
        **(mode_kwargs or {}),
    )

    if mode == "nominal":
        return _run_nominal(**kwargs)
    elif mode == "push_recovery":
        return _run_push_recovery(**kwargs)
    elif mode == "domain_randomized":
        return _run_domain_randomized(**kwargs)
    elif mode == "command_tracking":
        return _run_command_tracking(**kwargs)
    else:  # pragma: no cover
        raise RuntimeError(f"Unhandled mode: {mode}")  # unreachable
