"""
PPO (Proximal Policy Optimization) triển khai bằng JAX.

Đặc điểm:
  - Hoàn toàn trên GPU (JAX JIT)
  - Hỗ trợ parallel environments (vectorized)
  - Generalized Advantage Estimation (GAE)
  - Gradient clipping
  - Observation normalization (running mean/var)

Reference: Schulman et al., "Proximal Policy Optimization Algorithms", 2017
"""

from __future__ import annotations

import functools
import time
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax

from wheeled_biped.envs.base_env import EnvState, WheeledBipedEnv
from wheeled_biped.training.networks import create_actor_critic
from wheeled_biped.utils.logger import TrainingLogger

# ============================================================
# Checkpoint versioning
# ============================================================

# Increment this when the checkpoint format changes in a backward-incompatible way.
# Version history:
#   1 — initial versioned format (params, opt_state, obs_rms, config,
#         global_step, best_reward, curriculum_min_height,
#         best_eval_per_step, best_eval_success, best_train_reward)
CHECKPOINT_VERSION: int = 1

# ============================================================
# Data Structures
# ============================================================


class Transition(NamedTuple):
    """Dữ liệu một bước transition."""

    obs: jnp.ndarray  # (obs_size,)
    action: jnp.ndarray  # (action_size,)
    reward: jnp.ndarray  # scalar
    done: jnp.ndarray  # bool
    value: jnp.ndarray  # V(s)
    log_prob: jnp.ndarray  # log π(a|s)


class RolloutBatch(NamedTuple):
    """Batch dữ liệu rollout cho PPO update."""

    obs: jnp.ndarray  # (batch, obs_size)
    action: jnp.ndarray  # (batch, action_size)
    advantage: jnp.ndarray  # (batch,)
    returns: jnp.ndarray  # (batch,)
    old_log_prob: jnp.ndarray  # (batch,)
    old_value: jnp.ndarray  # (batch,)


class RunningMeanStd(NamedTuple):
    """Running mean/std cho observation normalization."""

    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray


class PPOTrainState(NamedTuple):
    """Trạng thái training PPO."""

    params: Any  # Network params
    opt_state: Any  # Optimizer state
    obs_rms: RunningMeanStd  # Running obs stats
    rng: jax.Array  # Random key
    global_step: jnp.ndarray  # Tổng số bước
    env_state: EnvState  # Trạng thái env


# ============================================================
# PPO Core Functions
# ============================================================


def init_running_mean_std(shape: tuple[int, ...]) -> RunningMeanStd:
    """Khởi tạo running mean/std."""
    return RunningMeanStd(
        mean=jnp.zeros(shape),
        var=jnp.ones(shape),
        count=jnp.float32(1e-4),
    )


@jax.jit
def update_running_mean_std(rms: RunningMeanStd, batch: jnp.ndarray) -> RunningMeanStd:
    """Cập nhật running mean/std với batch mới (Welford's algorithm)."""
    batch_mean = jnp.mean(batch, axis=0)
    batch_var = jnp.var(batch, axis=0)
    batch_count = batch.shape[0]

    delta = batch_mean - rms.mean
    total_count = rms.count + batch_count
    new_mean = rms.mean + delta * batch_count / total_count

    m_a = rms.var * rms.count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + jnp.square(delta) * rms.count * batch_count / total_count
    new_var = m2 / total_count

    return RunningMeanStd(
        mean=new_mean,
        var=new_var,
        count=total_count,
    )


@jax.jit
def normalize_obs(obs: jnp.ndarray, rms: RunningMeanStd) -> jnp.ndarray:
    """Chuẩn hóa observation bằng running mean/std."""
    return (obs - rms.mean) / jnp.sqrt(rms.var + 1e-8)


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    last_value: jnp.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Tính Generalized Advantage Estimation.

    Args:
        rewards: (T, num_envs)
        values: (T, num_envs)
        dones: (T, num_envs)
        last_value: (num_envs,)
        gamma: discount factor.
        gae_lambda: GAE lambda.

    Returns:
        (advantages, returns) - mỗi cái shape (T, num_envs).
    """

    def _scan_fn(carry, transition):
        last_gae, next_value = carry
        reward, value, done = transition

        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * gae_lambda * (1 - done) * last_gae
        return (gae, value), gae

    # Scan ngược thời gian
    _, advantages = jax.lax.scan(
        _scan_fn,
        (jnp.zeros_like(last_value), last_value),
        (rewards[::-1], values[::-1], dones[::-1]),
    )
    advantages = advantages[::-1]
    returns = advantages + values

    return advantages, returns


# ============================================================
# PPO Trainer
# ============================================================


class PPOTrainer:
    """PPO Trainer cho Wheeled Bipedal Robot.

    Sử dụng:
      trainer = PPOTrainer(env, config)
      trainer.train(total_steps=10_000_000)

    Attributes:
        env: Wheeled Bipedal environment.
        config: training config dict.
        model: ActorCritic network.
    """

    def __init__(
        self,
        env: WheeledBipedEnv,
        config: dict[str, Any],
        logger: TrainingLogger | None = None,
        seed: int = 42,
    ):
        self.env = env
        self.config = config
        self.logger = logger
        self._stop_requested = False  # Flag để dừng training từ bên ngoài
        self._resumed_global_step = 0  # Sẽ được set khi load_checkpoint
        self._resumed_best_reward = float("-inf")
        self._resumed_curriculum_min = None  # Curriculum min height từ checkpoint
        self._curriculum_min_height = None  # Giá trị hiện tại (cập nhật trong train())
        # Eval-triggered checkpoint trackers — restored from checkpoint on resume.
        # Default to -inf so first eval always saves when no prior checkpoint exists.
        self._resumed_best_eval_per_step: float = float("-inf")
        self._resumed_best_eval_success: float = float("-inf")
        self._resumed_best_train_reward: float = float("-inf")

        # PPO hyperparams
        ppo_cfg = config.get("ppo", {})
        self.lr = ppo_cfg.get("learning_rate", 3e-4)
        self.num_epochs = ppo_cfg.get("num_epochs", 4)
        self.num_minibatches = ppo_cfg.get("num_minibatches", 32)
        self.gamma = ppo_cfg.get("gamma", 0.99)
        self.gae_lambda = ppo_cfg.get("gae_lambda", 0.95)
        self.clip_epsilon = ppo_cfg.get("clip_epsilon", 0.2)
        self.entropy_coeff = ppo_cfg.get("entropy_coeff", 0.01)
        self.value_coeff = ppo_cfg.get("value_loss_coeff", 0.5)
        self.max_grad_norm = ppo_cfg.get("max_grad_norm", 0.5)
        self.normalize_adv = ppo_cfg.get("normalize_advantages", True)

        # Task params
        task_cfg = config.get("task", {})
        self.num_envs = task_cfg.get("num_envs", 4096)
        self.episode_length = task_cfg.get("episode_length", 1000)

        # Rollout params: mỗi rollout dài bao nhiêu bước trước khi update
        self._rollout_length = ppo_cfg.get("rollout_length", 32)  # bước

        # Tạo network
        self.rng = jax.random.PRNGKey(seed)
        self.rng, init_key = jax.random.split(self.rng)
        self.model, self.params = create_actor_critic(
            obs_size=env.obs_size,
            action_size=env.num_actions,
            config=config,
            rng=init_key,
        )

        # Optimizer
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(self.lr, eps=1e-5),
        )
        self.opt_state = self.optimizer.init(self.params)

        # Observation normalization
        self.obs_rms = init_running_mean_std((env.obs_size,))

    def _ppo_loss(
        self,
        params: Any,
        batch: RolloutBatch,
    ) -> tuple[jnp.ndarray, dict]:
        """Tính PPO loss.

        Returns:
            (total_loss, metrics_dict)
        """
        log_prob, entropy, value = self.model.apply(
            params,
            batch.obs,
            method=self.model.evaluate_action,
            action=batch.action,
        )

        # Policy loss (clipped surrogate)
        ratio = jnp.exp(log_prob - batch.old_log_prob)
        adv = batch.advantage
        if self.normalize_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        surr1 = ratio * adv
        surr2 = jnp.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv
        policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

        # Value loss (clipped)
        value_pred_clipped = batch.old_value + jnp.clip(
            value - batch.old_value, -self.clip_epsilon, self.clip_epsilon
        )
        value_loss1 = jnp.square(value - batch.returns)
        value_loss2 = jnp.square(value_pred_clipped - batch.returns)
        value_loss = 0.5 * jnp.mean(jnp.maximum(value_loss1, value_loss2))

        # Entropy bonus
        entropy_loss = -jnp.mean(entropy)

        total_loss = policy_loss + self.value_coeff * value_loss + self.entropy_coeff * entropy_loss

        metrics = {
            "loss/total": total_loss,
            "loss/policy": policy_loss,
            "loss/value": value_loss,
            "loss/entropy": -entropy_loss,
            "policy/clip_fraction": jnp.mean(jnp.abs(ratio - 1.0) > self.clip_epsilon),
            "policy/approx_kl": jnp.mean(0.5 * jnp.square(log_prob - batch.old_log_prob)),
        }

        return total_loss, metrics

    @functools.partial(jax.jit, static_argnums=(0,))
    def _rollout(
        self,
        params: Any,
        env_state: EnvState,
        rng: jax.Array,
        obs_rms: RunningMeanStd,
    ) -> tuple[EnvState, Transition, jax.Array]:
        """Thu thập dữ liệu rollout từ num_envs environments.

        Returns:
            (final_env_state, transitions, new_rng)
        """

        def _env_step(carry, _):
            env_state, rng = carry
            rng, action_key, reset_key = jax.random.split(rng, 3)

            # Lưu raw obs cho obs_rms update
            raw_obs = env_state.obs

            # Normalize observation
            obs = normalize_obs(raw_obs, obs_rms)

            # Lấy action từ policy
            raw_action, log_prob, _, value = self.model.apply(
                params,
                obs,
                rng=action_key,
                method=self.model.get_action_and_value,
            )

            # The environment receives clip(raw_action).  Task envs such as
            # BalanceEnv further transform this into:
            #   control_action = alpha * prev_action + (1 - alpha) * clip(raw_action)
            # before applying PID and stepping physics.
            #
            # PPO importance ratio design:
            #   - transition.action stores raw_action (the policy sample)
            #   - old_log_prob = log π_old(raw_action | obs)
            #   - update recomputes log_prob_new(raw_action | obs)
            #   → ratio = π_new(raw_action) / π_old(raw_action)  — internally consistent
            #
            # Imprecision: the advantage A_t reflects outcomes caused by control_action
            # (smoothed), but the gradient pushes π_θ(raw_action).  raw_action has
            # ~0.6 weight in control_action and the policy observes prev_action in its
            # obs, so the signal is directionally correct.  This is an accepted
            # engineering trade-off for action smoothing in on-policy RL.
            action = jnp.clip(raw_action, -1.0, 1.0)

            # Step environment
            next_state = self.env.v_step(env_state, action)

            # Lưu done và reward TRƯỚC auto-reset
            # (reset_if_done thay thế toàn bộ state khi done=True,
            #  bao gồm done→False và reward→0, mất tín hiệu terminal)
            transition_done = next_state.done
            transition_reward = next_state.reward

            # Auto-reset done envs
            next_state = self.env.v_reset_if_done(next_state, reset_key)

            transition = Transition(
                obs=obs,
                action=raw_action,  # Policy sample (pre-clip, pre-smooth) — see comment above
                reward=transition_reward,
                done=transition_done,
                value=value,
                log_prob=log_prob,
            )

            return (next_state, rng), transition

        (final_state, rng), transitions = jax.lax.scan(
            _env_step,
            (env_state, rng),
            None,
            length=self._rollout_length,
        )

        return final_state, transitions, rng

    def eval_pass(
        self,
        num_eval_envs: int = 64,
        num_episodes: int = 50,
        rng: jax.Array | None = None,
        curriculum_min_height: float | None = None,
    ) -> dict[str, float]:
        """Run a held-out evaluation pass using the current policy.

        This produces a real evaluation metric suitable for curriculum gating:
          - Runs ``num_eval_envs`` independent environments in parallel.
          - Accumulates per-episode returns (sum of rewards) until ``num_episodes``
            complete episodes have been observed across all envs.
          - Policy acts *greedily* (mean action, no exploration noise).
          - Does NOT update ``obs_rms``, ``params``, or any training state.

        Args:
            num_eval_envs: parallel evaluation environments.
            num_episodes: minimum completed episodes to average over.
            rng: optional JAX key; defaults to a new split from self.rng.
            curriculum_min_height: when the in-env height curriculum is active,
                pass the current ``current_min_h`` from the training loop here.
                eval_pass() will resample the initial height_command (and obs)
                from ``[curriculum_min_height, MAX_HEIGHT_CMD]`` so the eval
                reflects the actual height range being trained, not the fixed
                initial_min_height baked into BalanceEnv.reset().

        Returns:
            Dict with keys:
              ``eval_reward_mean``   -- mean episode return.
              ``eval_reward_std``    -- std of episode returns.
              ``eval_fall_rate``     -- fraction of episodes that terminated early
                                       (is_fallen flag set, not a time-limit).
              ``eval_success_rate``  -- fraction that survived the full episode.
              ``eval_num_episodes``  -- actual number of completed episodes counted.
        """
        if rng is None:
            self.rng, rng = jax.random.split(self.rng)

        rng, reset_key = jax.random.split(rng)
        env_state = self.env.v_reset(reset_key, num_eval_envs)

        # If curriculum is active, patch the initial env state so the first
        # episode's height_command and obs reflect the current training range
        # rather than the fixed initial_min_height baked into reset().
        # Subsequent reset_if_done() calls will carry curriculum_min_height
        # forward automatically (BalanceEnv.reset_if_done preserves it).
        if (
            curriculum_min_height is not None
            and hasattr(self.env, "MAX_HEIGHT_CMD")
            and hasattr(self.env, "MIN_HEIGHT_CMD")
            and "curriculum_min_height" in env_state.info
        ):
            rng, h_key = jax.random.split(rng)
            abs_min = float(self.env.MIN_HEIGHT_CMD)
            max_h = float(self.env.MAX_HEIGHT_CMD)
            curr_min = float(curriculum_min_height)
            new_h_cmd = jax.random.uniform(
                h_key, shape=(num_eval_envs,), minval=curr_min, maxval=max_h
            )
            height_norm = (new_h_cmd - abs_min) / (max_h - abs_min)
            new_obs = env_state.obs.at[:, -1].set(height_norm)
            new_info = {
                **env_state.info,
                "height_command": new_h_cmd,
                "curriculum_min_height": jnp.full_like(
                    env_state.info["curriculum_min_height"], curr_min
                ),
            }
            env_state = env_state._replace(obs=new_obs, info=new_info)

        episode_returns: list[float] = []
        episode_falls: list[bool] = []
        # Accumulate per-env partial returns across steps
        running_return = [0.0] * num_eval_envs
        running_fallen = [False] * num_eval_envs

        max_steps = self.env._episode_length * 4  # hard safety cap
        obs_rms = self.obs_rms  # snapshot — not updated
        params = self.params

        for _ in range(max_steps):
            if len(episode_returns) >= num_episodes:
                break

            # Greedy action: use mode (mean) of the policy distribution (no sampling noise)
            # model.__call__(obs) returns (action_dist, value); dist.mode() = mean for Gaussian
            norm_obs = jax.vmap(lambda o: normalize_obs(o, obs_rms))(env_state.obs)
            dist, _ = self.model.apply(params, norm_obs)
            action = jnp.clip(dist.mode(), -1.0, 1.0)

            next_state = self.env.v_step(env_state, action)

            # Collect per-env reward and done BEFORE auto-reset
            rewards_np = list(jax.device_get(jnp.asarray(next_state.reward)))
            dones_np = list(jax.device_get(jnp.asarray(next_state.done)))

            # Check fallen flag if available, otherwise use done-before-time-limit
            if "is_fallen" in next_state.info:
                fallen_np = list(jax.device_get(jnp.asarray(next_state.info["is_fallen"])))
            else:
                fallen_np = dones_np  # conservative fallback

            for i in range(num_eval_envs):
                running_return[i] += float(rewards_np[i])
                if dones_np[i]:
                    episode_returns.append(running_return[i])
                    episode_falls.append(bool(fallen_np[i]))
                    running_return[i] = 0.0
                    running_fallen[i] = False

            rng, reset_key = jax.random.split(rng)
            env_state = self.env.v_reset_if_done(next_state, reset_key)

        if not episode_returns:
            # Fallback: use last partial return rather than returning nothing
            episode_returns = running_return[:]
            episode_falls = running_fallen[:]

        n = len(episode_returns)
        mean_ret = float(sum(episode_returns) / n)
        std_ret = (
            float((sum((r - mean_ret) ** 2 for r in episode_returns) / n) ** 0.5) if n > 1 else 0.0
        )
        fall_rate = float(sum(episode_falls) / n)

        return {
            "eval_reward_mean": mean_ret,
            "eval_reward_std": std_ret,
            "eval_fall_rate": fall_rate,
            "eval_success_rate": 1.0 - fall_rate,
            "eval_num_episodes": n,
        }

    @functools.partial(jax.jit, static_argnums=(0,))
    def _update_step(
        self,
        params: Any,
        opt_state: Any,
        transitions: Transition,
        last_value: jnp.ndarray,
        rng: jax.Array,
    ) -> tuple[Any, Any, dict, jax.Array]:
        """Thực hiện PPO update dựa trên dữ liệu rollout.

        Returns:
            (new_params, new_opt_state, metrics, new_rng)
        """
        # Tính GAE
        advantages, returns = compute_gae(
            transitions.reward,
            transitions.value,
            transitions.done,
            last_value,
            self.gamma,
            self.gae_lambda,
        )

        # Flatten: (T, num_envs, ...) → (T*num_envs, ...)
        batch_size = self._rollout_length * self.num_envs
        batch = RolloutBatch(
            obs=transitions.obs.reshape(batch_size, -1),
            action=transitions.action.reshape(batch_size, -1),
            advantage=advantages.reshape(batch_size),
            returns=returns.reshape(batch_size),
            old_log_prob=transitions.log_prob.reshape(batch_size),
            old_value=transitions.value.reshape(batch_size),
        )

        # Mini-batch training
        minibatch_size = batch_size // self.num_minibatches
        all_metrics = {}

        def _epoch_step(carry, _):
            params, opt_state, rng = carry
            rng, perm_key = jax.random.split(rng)
            perm = jax.random.permutation(perm_key, batch_size)

            def _minibatch_step(carry, start_idx):
                params, opt_state = carry
                idx = jax.lax.dynamic_slice(perm, (start_idx,), (minibatch_size,))
                mb = jax.tree.map(lambda x: x[idx], batch)

                grad_fn = jax.value_and_grad(self._ppo_loss, has_aux=True)
                (loss, metrics), grads = grad_fn(params, mb)

                updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
                new_params = optax.apply_updates(params, updates)

                return (new_params, new_opt_state), metrics

            starts = jnp.arange(0, batch_size, minibatch_size)[: self.num_minibatches]
            (params, opt_state), epoch_metrics = jax.lax.scan(
                _minibatch_step,
                (params, opt_state),
                starts,
            )

            return (params, opt_state, rng), epoch_metrics

        (new_params, new_opt_state, rng), all_metrics = jax.lax.scan(
            _epoch_step,
            (params, opt_state, rng),
            None,
            length=self.num_epochs,
        )

        # Trung bình metrics qua epochs và minibatches
        avg_metrics = jax.tree.map(lambda x: jnp.mean(x), all_metrics)

        return new_params, new_opt_state, avg_metrics, rng

    def train(
        self,
        total_steps: int = 10_000_000,
        log_interval: int = 10,
        save_interval: int = 100,
        checkpoint_dir: str = "checkpoints",
        live_view: bool = False,
        view_interval: int = 2,
        _external_viewer: Any = None,
    ) -> dict:
        """Vòng lặp training chính.

        Args:
            total_steps: tổng số bước training.
            log_interval: mỗi bao nhiêu update thì log.
            save_interval: mỗi bao nhiêu update thì save checkpoint.
            checkpoint_dir: thư mục lưu checkpoint.
            live_view: bật cửa sổ MuJoCo viewer real-time.
            view_interval: mỗi bao nhiêu update thì cập nhật viewer.
            _external_viewer: LiveTrainingViewer instance (internal, do
                run_training_with_viewer truyền vào).

        Returns:
            Dict metrics cuối cùng.
        """
        import os
        import shutil

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Live viewer — ưu tiên external viewer (đã chạy trên main thread)
        viewer = _external_viewer
        if viewer is None and live_view:
            try:
                from wheeled_biped.training.live_viewer import LiveTrainingViewer

                viewer = LiveTrainingViewer(self.env.mj_model, title="PPO Training")
                # Không gọi start() — sẽ warning vì không chạy trên main thread
                print("  ⚠️  live_view=True nhưng viewer cần chạy trên main thread.")
                print("     Hãy dùng:  python scripts/train.py single --stage balance --live-view")
                viewer = None
            except Exception as e:
                print(f"  ⚠️  Không mở được live viewer: {e}")
                viewer = None

        self._stop_requested = False

        # Detect CPU vs GPU
        backend = jax.default_backend()
        is_cpu = backend == "cpu"
        if is_cpu and self.num_envs > 256:
            print(f"  ⚠️  CPU + num_envs={self.num_envs}: JIT compile sẽ rất lâu (10-20 phút)!")
            print("      Trên CPU nên dùng --num-envs 64~128 (JIT ~1-2 phút)")
            print("      Tăng num_envs trên CPU không nhanh hơn: mỗi update lâu hơn tỉ lệ thuận")
            print("      GPU (jax[cuda12]) mới thật sự song song 4096 envs")
            print()

        # Reset environments
        self.rng, reset_key = jax.random.split(self.rng)
        env_state = self.env.v_reset(reset_key, self.num_envs)

        steps_per_update = self._rollout_length * self.num_envs

        # ====== Reward-based Curriculum ======
        curriculum_cfg = self.config.get("curriculum", {})
        curriculum_enabled = curriculum_cfg.get("enabled", False)
        threshold_ratio = float(curriculum_cfg.get("reward_threshold", 0.92))
        num_levels = int(curriculum_cfg.get("num_levels", 10))
        window_size = int(curriculum_cfg.get("window", 50))

        # Eval-gated curriculum signal (more trustworthy than rolling training reward).
        # When use_eval_signal=True, curriculum advances based on eval_pass() results
        # (greedy policy, complete episodes) instead of the noisy per-rollout avg_reward.
        use_eval_signal = bool(curriculum_cfg.get("use_eval_signal", False))
        _eval_interval = int(curriculum_cfg.get("eval_interval", window_size))
        _curriculum_eval_envs = min(32, self.num_envs)
        _curriculum_eval_episodes = int(curriculum_cfg.get("eval_episodes", 20))

        # Tính max reward có thể đạt (tổng trọng số dương)
        reward_weights = self.config.get("rewards", {})
        max_reward_possible = sum(w for w in reward_weights.values() if w > 0)
        reward_threshold = threshold_ratio * max_reward_possible

        # Curriculum state
        task_cfg = self.config.get("task", {})
        initial_min_h = float(task_cfg.get("initial_min_height", 0.68))
        final_min_h = getattr(self.env, "MIN_HEIGHT_CMD", 0.38)
        level_step = (initial_min_h - final_min_h) / max(num_levels, 1)

        # Khôi phục từ checkpoint hoặc dùng giá trị mặc định
        if self._resumed_curriculum_min is not None:
            current_min_h = self._resumed_curriculum_min
        else:
            current_min_h = initial_min_h
        self._curriculum_min_height = current_min_h

        # Set curriculum_min_height cho tất cả envs
        if curriculum_enabled and "curriculum_min_height" in env_state.info:
            new_min_arr = jnp.full_like(env_state.info["curriculum_min_height"], current_min_h)
            env_state = env_state._replace(
                info={**env_state.info, "curriculum_min_height": new_min_arr}
            )

        reward_window: list[float] = []  # Reward window cho curriculum
        curriculum_level = (
            round((initial_min_h - current_min_h) / level_step) if level_step > 0 else 0
        )

        # Tính số updates còn lại (trừ bước đã train trước đó nếu resume)
        resumed_step = self._resumed_global_step
        remaining_steps = max(steps_per_update, total_steps - resumed_step)
        num_updates = remaining_steps // steps_per_update

        print("═══ PPO Training ═══")
        print(f"  Backend: {backend.upper()}")
        print(f"  Envs: {self.num_envs}, Rollout: {self._rollout_length}")
        print(f"  Steps/update: {steps_per_update:,}")
        if resumed_step > 0:
            print(f"  ▶ Resumed from step: {resumed_step:,}")
            print(f"  ▶ Resumed best_reward: {self._resumed_best_reward:.4f}")
            print(f"  Remaining updates: {num_updates:,}")
            print(f"  Remaining steps: {remaining_steps:,}")
        else:
            print(f"  Total updates: {num_updates:,}")
        print(f"  Target total steps: {total_steps:,}")
        print("════════════════════")

        # Warmup: JIT compile lần đầu (rất lâu — thông báo cho user)
        print("  ⏳ Đang JIT compile (lần đầu có thể mất 1-3 phút)...")
        if viewer is not None:
            viewer.set_status("JIT compiling... (1-3 phút lần đầu)")

        compile_start = time.time()

        # Warmup rollout
        self.rng, warmup_key = jax.random.split(self.rng)
        env_state, warmup_transitions, self.rng = self._rollout(
            self.params,
            env_state,
            warmup_key,
            self.obs_rms,
        )
        # Force JAX to finish compilation
        jax.block_until_ready(warmup_transitions.reward)

        # Warmup update — un-normalize để lấy raw obs cho obs_rms
        # (transitions.obs đã normalized trong _rollout, cần raw data cho Welford)
        raw_obs = warmup_transitions.obs * jnp.sqrt(self.obs_rms.var + 1e-8) + self.obs_rms.mean
        all_obs = raw_obs.reshape(-1, self.env.obs_size)
        self.obs_rms = update_running_mean_std(self.obs_rms, all_obs)
        last_obs = normalize_obs(env_state.obs, self.obs_rms)
        _, last_value = self.model.apply(self.params, last_obs)

        self.rng, warmup_update_key = jax.random.split(self.rng)
        self.params, self.opt_state, _, self.rng = self._update_step(
            self.params,
            self.opt_state,
            warmup_transitions,
            last_value,
            warmup_update_key,
        )
        jax.block_until_ready(self.params)

        compile_time = time.time() - compile_start
        print(f"  ✅ JIT compile xong! ({compile_time:.1f}s)")
        if viewer is not None:
            viewer.set_status(f"Training bắt đầu! (compiled in {compile_time:.1f}s)")

        # Kiểm tra stop ngay sau warmup (có thể bị Ctrl+C trong lúc compile)
        if self._stop_requested:
            print("\n  🛑 Training dừng theo yêu cầu (trong lúc JIT compile)")
            gs = resumed_step + steps_per_update
            self._save_checkpoint(
                os.path.join(checkpoint_dir, "final"),
                global_step=gs,
                best_reward=self._resumed_best_reward,
                best_eval_per_step=self._resumed_best_eval_per_step,
                best_eval_success=self._resumed_best_eval_success,
                best_train_reward=self._resumed_best_train_reward,
            )
            if viewer is not None:
                viewer.request_stop()
            return {
                "best_reward": self._resumed_best_reward,
                "eval_reward_mean": self._resumed_best_reward,
                "total_steps": gs,
            }

        global_step = resumed_step + steps_per_update  # Tính cả bước đã resume
        start_time = time.time()
        best_reward = self._resumed_best_reward  # Giữ best_reward từ lần trước
        # train_reward_mean: rolling mean of per-step avg_rewards during training.
        # This is a TRAINING metric — used for logging only, NOT for curriculum gating.
        # Curriculum gating uses eval_reward_mean from eval_pass() at the end of train().
        _train_rewards: list[float] = []  # rolling window, last 50 updates
        _TRAIN_WINDOW = 50  # noqa: N806
        # ── Improvement-triggered checkpoint settings ─────────────────────────
        # Minimum improvement required before a triggered save fires.
        # Prevents noise-driven writes when the metric stagnates near a plateau.
        _EVAL_CKPT_MIN_DELTA: float = 1e-3  # noqa: N806
        _TRAIN_CKPT_MIN_DELTA: float = 1e-2  # noqa: N806  (legacy use_eval_signal=False path)
        # Minimum evals that must pass between triggered saves (both paths).
        # Prevents rapid-fire saves during a sustained improvement phase.
        # Configurable via curriculum.ckpt_cooldown_evals; default 5.
        # Cooldown is NOT persisted — resets on every (re)start.
        # After resume, first triggered save is always eligible (safe: never misses peaks).
        _EVAL_CKPT_COOLDOWN: int = int(curriculum_cfg.get("ckpt_cooldown_evals", 5))  # noqa: N806
        _TRAIN_CKPT_COOLDOWN: int = _EVAL_CKPT_COOLDOWN  # noqa: N806
        # Tracker values — restored from checkpoint metadata on resume.
        # Fall back to -inf for old checkpoints without these fields
        # (safe: first post-resume eval saves, then delta + cooldown kick in).
        _best_eval_per_step: float = self._resumed_best_eval_per_step
        _best_eval_success: float = self._resumed_best_eval_success
        _best_train_reward: float = self._resumed_best_train_reward
        # Per-metric eval counts since last triggered save.
        # Init to cooldown so the first eval is always eligible.
        _evals_since_per_step_save: int = _EVAL_CKPT_COOLDOWN
        _evals_since_success_save: int = _EVAL_CKPT_COOLDOWN
        _windows_since_train_save: int = _TRAIN_CKPT_COOLDOWN
        # ─────────────────────────────────────────────────────────────────────

        # Cập nhật viewer ngay sau warmup
        if viewer is not None:
            try:
                viewer.update(
                    env_state.mjx_data,
                    env_idx=0,
                    info={"step": str(global_step), "status": "warmup done"},
                )
            except Exception:
                pass

        try:
            for update in range(1, num_updates):  # Bắt đầu từ 1 (đã warmup update 0)
                # Kiểm tra stop flag (Ctrl+C hoặc viewer đóng)
                if self._stop_requested:
                    print(f"\n  🛑 Training dừng theo yêu cầu tại update {update}/{num_updates}")
                    break
                if viewer is not None and not viewer.is_running:
                    print(f"\n  🛑 Viewer đã đóng, dừng training tại update {update}/{num_updates}")
                    break

                update_start = time.time()

                # Rollout
                self.rng, rollout_key = jax.random.split(self.rng)
                env_state, transitions, self.rng = self._rollout(
                    self.params,
                    env_state,
                    rollout_key,
                    self.obs_rms,
                )
                # Force kết quả về CPU để đo thời gian chính xác
                jax.block_until_ready(transitions.reward)

                # Cập nhật observation normalization bằng RAW obs
                # (transitions.obs đã normalized, un-normalize trước khi update)
                raw_obs = transitions.obs * jnp.sqrt(self.obs_rms.var + 1e-8) + self.obs_rms.mean
                all_obs = raw_obs.reshape(-1, self.env.obs_size)
                self.obs_rms = update_running_mean_std(self.obs_rms, all_obs)

                # Tính last value cho GAE
                last_obs = normalize_obs(env_state.obs, self.obs_rms)
                _, last_value = self.model.apply(self.params, last_obs)

                # PPO Update
                self.rng, update_key = jax.random.split(self.rng)
                self.params, self.opt_state, metrics, self.rng = self._update_step(
                    self.params,
                    self.opt_state,
                    transitions,
                    last_value,
                    update_key,
                )
                jax.block_until_ready(self.params)

                global_step += steps_per_update
                update_elapsed = time.time() - update_start

                # Progress mỗi update (để người dùng biết còn chạy)
                avg_reward = float(jnp.mean(transitions.reward))
                # Accumulate for train_reward_mean (logging only)
                _train_rewards.append(avg_reward)
                if len(_train_rewards) > _TRAIN_WINDOW:
                    _train_rewards.pop(0)
                elapsed_total = time.time() - start_time
                fps = global_step / max(elapsed_total, 1)
                eta_s = (num_updates - update) * update_elapsed
                eta_m = eta_s / 60

                # Luôn in progress mỗi update
                # Thêm curriculum info nếu enabled
                range_str = ""
                if curriculum_enabled:
                    max_h = getattr(self.env, "MAX_HEIGHT_CMD", 0.72)
                    signal_label = "[eval]" if use_eval_signal else "[train]"
                    range_str = (
                        f" | range=[{current_min_h:.2f},{max_h:.2f}] "
                        f"L{curriculum_level}/{num_levels}{signal_label}"
                    )
                print(
                    f"  [{update}/{num_updates}] "
                    f"step={global_step:,} | "
                    f"reward={avg_reward:.4f} | "
                    f"fps={fps:.0f} | "
                    f"{update_elapsed:.1f}s/update | "
                    f"ETA {eta_m:.0f}m"
                    f"{range_str}",
                    flush=True,
                )

                # Logging chi tiết
                if update % log_interval == 0:
                    log_metrics = {
                        "reward/mean": avg_reward,
                        "reward/std": float(jnp.std(transitions.reward)),
                        "reward/min": float(jnp.min(transitions.reward)),
                        "reward/max": float(jnp.max(transitions.reward)),
                        "training/fps": fps,
                        "training/global_step": global_step,
                        "training/update_time_s": update_elapsed,
                        **{k: float(v) for k, v in metrics.items()},
                    }

                    # Curriculum metrics (enable sweeps over curriculum params)
                    if curriculum_enabled:
                        log_metrics["curriculum/level"] = float(curriculum_level)
                        log_metrics["curriculum/min_height"] = float(current_min_h)
                        # Log the gate values so any run's JSONL is self-contained:
                        # reward_threshold = threshold_ratio × max_reward_possible (per step).
                        # Required to reconstruct why curriculum did/did not advance.
                        log_metrics["curriculum/reward_threshold"] = float(reward_threshold)
                        log_metrics["curriculum/max_reward_possible"] = float(max_reward_possible)

                    if self.logger:
                        self.logger.set_step(global_step)
                        self.logger.log_dict(log_metrics)

                    # Save best
                    if avg_reward > best_reward:
                        best_reward = avg_reward

                # ====== Curriculum advancement ======
                if curriculum_enabled and current_min_h > final_min_h:
                    if use_eval_signal:
                        # ── Eval-gated path ──────────────────────────────────────────
                        # Runs eval_pass() every _eval_interval updates.
                        # eval_pass() uses a greedy policy over complete episodes, so
                        # the signal is cleaner than per-rollout avg_reward (which
                        # carries exploration noise and is mean over a single rollout).
                        #
                        # Threshold comparison: reward_threshold is calibrated in
                        # per-step units (fraction of max_reward_possible).
                        # eval_pass() returns mean episode *sum*, so we normalize:
                        #   eval_per_step = eval_reward_mean / episode_length
                        if update % _eval_interval == 0:
                            self.rng, _ceval_key = jax.random.split(self.rng)
                            _ceval = self.eval_pass(
                                num_eval_envs=_curriculum_eval_envs,
                                num_episodes=_curriculum_eval_episodes,
                                rng=_ceval_key,
                            )
                            _eval_per_step = _ceval["eval_reward_mean"] / max(
                                1, self.episode_length
                            )
                            if self.logger:
                                self.logger.set_step(global_step)
                                self.logger.log_dict(
                                    {
                                        "curriculum/eval_per_step": _eval_per_step,
                                        "curriculum/eval_success_rate": _ceval["eval_success_rate"],
                                        "curriculum/eval_fall_rate": _ceval["eval_fall_rate"],
                                    }
                                )
                            print(
                                f"  [Curriculum eval] per_step={_eval_per_step:.3f} "
                                f"threshold={reward_threshold:.3f} "
                                f"success={_ceval['eval_success_rate']:.2f} "
                                f"fall={_ceval['eval_fall_rate']:.2f}",
                                flush=True,
                            )
                            # ── Eval-triggered checkpoints ─────────────────────────────
                            # Guard: metric must improve by >= _EVAL_CKPT_MIN_DELTA AND
                            # at least _EVAL_CKPT_COOLDOWN evals must have passed since
                            # the last save for that metric.
                            # Each save writes a versioned dir
                            # (ckpt_best/eval_per_step_s{step:010d}/)
                            # and updates the stable pointer (best_eval_per_step/) via copy.
                            _evals_since_per_step_save += 1
                            _evals_since_success_save += 1
                            if (
                                _eval_per_step > _best_eval_per_step + _EVAL_CKPT_MIN_DELTA
                                and _evals_since_per_step_save >= _EVAL_CKPT_COOLDOWN
                            ):
                                _best_eval_per_step = _eval_per_step
                                _evals_since_per_step_save = 0
                                _v = os.path.join(
                                    checkpoint_dir,
                                    "ckpt_best",
                                    f"eval_per_step_s{global_step:010d}",
                                )
                                self._save_checkpoint(
                                    _v,
                                    global_step=global_step,
                                    best_reward=best_reward,
                                    best_eval_per_step=_best_eval_per_step,
                                    best_eval_success=_best_eval_success,
                                    best_train_reward=_best_train_reward,
                                )
                                _stable = os.path.join(checkpoint_dir, "best_eval_per_step")
                                os.makedirs(_stable, exist_ok=True)
                                shutil.copy2(
                                    os.path.join(_v, "checkpoint.pkl"),
                                    os.path.join(_stable, "checkpoint.pkl"),
                                )
                                if self.logger:
                                    self.logger.flush()
                            _success_improved = (
                                _ceval["eval_success_rate"]
                                > _best_eval_success + _EVAL_CKPT_MIN_DELTA
                            )
                            if (
                                _success_improved
                                and _evals_since_success_save >= _EVAL_CKPT_COOLDOWN
                            ):
                                _best_eval_success = _ceval["eval_success_rate"]
                                _evals_since_success_save = 0
                                _v = os.path.join(
                                    checkpoint_dir,
                                    "ckpt_best",
                                    f"eval_success_s{global_step:010d}",
                                )
                                self._save_checkpoint(
                                    _v,
                                    global_step=global_step,
                                    best_reward=best_reward,
                                    best_eval_per_step=_best_eval_per_step,
                                    best_eval_success=_best_eval_success,
                                    best_train_reward=_best_train_reward,
                                )
                                _stable = os.path.join(checkpoint_dir, "best_eval_success")
                                os.makedirs(_stable, exist_ok=True)
                                shutil.copy2(
                                    os.path.join(_v, "checkpoint.pkl"),
                                    os.path.join(_stable, "checkpoint.pkl"),
                                )
                                if self.logger:
                                    self.logger.flush()
                            # ───────────────────────────────────────────────────────────
                            if _eval_per_step >= reward_threshold:
                                current_min_h = max(
                                    round(current_min_h - level_step, 4), final_min_h
                                )
                                self._curriculum_min_height = current_min_h
                                curriculum_level = min(curriculum_level + 1, num_levels)
                                new_min_arr = jnp.full_like(
                                    env_state.info["curriculum_min_height"],
                                    current_min_h,
                                )
                                env_state = env_state._replace(
                                    info={
                                        **env_state.info,
                                        "curriculum_min_height": new_min_arr,
                                    }
                                )
                                max_h = getattr(self.env, "MAX_HEIGHT_CMD", 0.72)
                                print(
                                    f"  \U0001f4c8 Curriculum Level"
                                    f" {curriculum_level}/{num_levels}:"
                                    f" height range [{current_min_h:.2f}, {max_h:.2f}]"
                                    f" (eval_per_step={_eval_per_step:.3f}"
                                    f" >= {reward_threshold:.3f})"
                                )
                                if current_min_h <= final_min_h:
                                    print(
                                        f"  \u2705 Curriculum hoàn thành! Full range "
                                        f"[{final_min_h:.2f}, {max_h:.2f}]"
                                    )
                    else:
                        # ── Training-reward window (backward-compatible) ──────────────
                        # NOTE: avg_reward is a per-step mean across one rollout,
                        # including exploration noise.  This is the original behavior.
                        # Set curriculum.use_eval_signal: true for a cleaner signal.
                        reward_window.append(avg_reward)
                        if len(reward_window) > window_size:
                            reward_window.pop(0)
                        if len(reward_window) >= window_size:
                            window_avg = sum(reward_window) / len(reward_window)
                            # ── Train-reward–triggered checkpoint (legacy path) ──────
                            # Signal: rolling window of noisy per-rollout avg_reward.
                            # NOT an eval trigger — named best_train_reward to be explicit.
                            # Same delta + cooldown guards as eval path.
                            # Versioned dir: ckpt_best/train_reward_s{step:010d}/
                            # Stable pointer: best_train_reward/ (overwritten each improvement)
                            _windows_since_train_save += 1
                            if (
                                window_avg > _best_train_reward + _TRAIN_CKPT_MIN_DELTA
                                and _windows_since_train_save >= _TRAIN_CKPT_COOLDOWN
                            ):
                                _best_train_reward = window_avg
                                _windows_since_train_save = 0
                                _v = os.path.join(
                                    checkpoint_dir,
                                    "ckpt_best",
                                    f"train_reward_s{global_step:010d}",
                                )
                                self._save_checkpoint(
                                    _v,
                                    global_step=global_step,
                                    best_reward=best_reward,
                                    best_eval_per_step=_best_eval_per_step,
                                    best_eval_success=_best_eval_success,
                                    best_train_reward=_best_train_reward,
                                )
                                _stable = os.path.join(checkpoint_dir, "best_train_reward")
                                os.makedirs(_stable, exist_ok=True)
                                shutil.copy2(
                                    os.path.join(_v, "checkpoint.pkl"),
                                    os.path.join(_stable, "checkpoint.pkl"),
                                )
                                if self.logger:
                                    self.logger.flush()
                            # ─────────────────────────────────────────────────────────
                            if window_avg >= reward_threshold:
                                current_min_h = max(
                                    round(current_min_h - level_step, 4), final_min_h
                                )
                                self._curriculum_min_height = current_min_h
                                curriculum_level = min(curriculum_level + 1, num_levels)
                                new_min_arr = jnp.full_like(
                                    env_state.info["curriculum_min_height"],
                                    current_min_h,
                                )
                                env_state = env_state._replace(
                                    info={
                                        **env_state.info,
                                        "curriculum_min_height": new_min_arr,
                                    }
                                )
                                reward_window.clear()
                                max_h = getattr(self.env, "MAX_HEIGHT_CMD", 0.72)
                                print(
                                    f"  \U0001f4c8 Curriculum Level"
                                    f" {curriculum_level}/{num_levels}:"
                                    f" height range [{current_min_h:.2f}, {max_h:.2f}]"
                                    f" (avg={window_avg:.2f} >= {reward_threshold:.2f})"
                                )
                                if current_min_h <= final_min_h:
                                    print(
                                        f"  \u2705 Curriculum hoàn thành! Full range "
                                        f"[{final_min_h:.2f}, {max_h:.2f}]"
                                    )

                # Live viewer update
                if viewer is not None and update % view_interval == 0:
                    try:
                        viewer.update(
                            env_state.mjx_data,
                            env_idx=0,
                            info={
                                "step": f"{global_step:,}",
                                "reward": f"{avg_reward:.3f}",
                                "fps": f"{fps:.0f}",
                            },
                        )
                    except Exception:
                        pass

                # Checkpoint
                if update % save_interval == 0 and update > 0:
                    self._save_checkpoint(
                        os.path.join(checkpoint_dir, f"step_{global_step}"),
                        global_step=global_step,
                        best_reward=best_reward,
                        best_eval_per_step=_best_eval_per_step,
                        best_eval_success=_best_eval_success,
                        best_train_reward=_best_train_reward,
                    )
                    # Flush logger buffer to disk alongside checkpoint
                    if self.logger:
                        self.logger.flush()

        except KeyboardInterrupt:
            print("\n\n⚠️  Training bị dừng bởi người dùng (Ctrl+C)")
            print(f"   Đã train {global_step:,} steps, best_reward={best_reward:.4f}")
            print("   Đang lưu checkpoint cuối...")

        # Save final
        self._save_checkpoint(
            os.path.join(checkpoint_dir, "final"),
            global_step=global_step,
            best_reward=best_reward,
            best_eval_per_step=_best_eval_per_step,
            best_eval_success=_best_eval_success,
            best_train_reward=_best_train_reward,
        )

        # ====== Curriculum Report ======
        if curriculum_enabled:
            max_h = getattr(self.env, "MAX_HEIGHT_CMD", 0.72)
            print("\n  \U0001f4ca Curriculum Report:")
            print(f"     Height range đã train: [{current_min_h:.2f}, {max_h:.2f}] m")
            print(f"     Level: {curriculum_level}/{num_levels}")
            if current_min_h > final_min_h:
                print(
                    f"     \u26a0\ufe0f  Chưa hoàn thành! Range đầy đủ:"
                    f" [{final_min_h:.2f}, {max_h:.2f}] m"
                )
                print("     \u2192 Tiếp tục train với --resume để mở rộng range")
            else:
                print("     \u2705 Đã train đầy đủ full range!")

        if viewer is not None:
            viewer.request_stop()

        # Compute train_reward_mean (rolling window of step-level rewards, NOT eval)
        train_reward_mean = (
            float(sum(_train_rewards) / len(_train_rewards)) if _train_rewards else best_reward
        )

        # ── Real evaluation pass ──────────────────────────────────────────────
        # Run a lightweight held-out evaluation (no gradient updates) to produce
        # a trustworthy metric for curriculum gating.  64 envs × up to 50 episodes
        # is cheap relative to the training budget but gives a clean signal.
        print("  📊 Running end-of-stage eval pass...")
        self.rng, eval_key = jax.random.split(self.rng)
        eval_metrics = self.eval_pass(
            num_eval_envs=min(64, self.num_envs),
            num_episodes=50,
            rng=eval_key,
            curriculum_min_height=current_min_h if curriculum_enabled else None,
        )
        eval_reward_mean = eval_metrics["eval_reward_mean"]
        print(
            f"  📊 Eval: reward_mean={eval_reward_mean:.4f} "
            f"fall_rate={eval_metrics['eval_fall_rate']:.3f} "
            f"n={eval_metrics['eval_num_episodes']}"
        )
        if self.logger:
            self.logger.set_step(global_step)
            self.logger.log_dict(
                {
                    "eval/reward_mean": eval_reward_mean,
                    "eval/reward_std": eval_metrics["eval_reward_std"],
                    "eval/fall_rate": eval_metrics["eval_fall_rate"],
                    "eval/success_rate": eval_metrics["eval_success_rate"],
                    "eval/num_episodes": float(eval_metrics["eval_num_episodes"]),
                    "train/reward_mean_recent": train_reward_mean,
                }
            )
            self.logger.close()  # close after eval metrics are safely logged

        return {
            "best_reward": best_reward,
            "train_reward_mean": train_reward_mean,  # rolling train metric (logging only)
            "eval_reward_mean": eval_reward_mean,  # real eval pass — used for curriculum
            "eval_fall_rate": eval_metrics["eval_fall_rate"],
            "eval_success_rate": eval_metrics["eval_success_rate"],
            "total_steps": global_step,
            # episode_length is included so CurriculumManager can normalise
            # eval_reward_mean → per-step units for dimensionally correct gating.
            "episode_length": self.episode_length,
            "curriculum_min_height": current_min_h if curriculum_enabled else None,
            "curriculum_level": curriculum_level if curriculum_enabled else None,
            "curriculum_num_levels": num_levels if curriculum_enabled else None,
        }

    def _save_checkpoint(
        self,
        path: str,
        global_step: int = 0,
        best_reward: float = float("-inf"),
        best_eval_per_step: float = float("-inf"),
        best_eval_success: float = float("-inf"),
        best_train_reward: float = float("-inf"),
    ) -> None:
        """Lưu checkpoint (params + obs_rms + training state).

        best_eval_per_step / best_eval_success / best_train_reward are the
        eval-triggered checkpoint tracker values at save time.  Storing them
        here lets a resumed run restore the trackers instead of restarting
        from -inf and overwriting these directories on the first eval.
        """
        import os
        import pickle

        os.makedirs(path, exist_ok=True)

        checkpoint = {
            "version": CHECKPOINT_VERSION,
            "params": jax.device_get(self.params),
            "opt_state": jax.device_get(self.opt_state),
            "obs_rms": jax.device_get(self.obs_rms),
            "config": self.config,
            "global_step": int(global_step),
            "best_reward": float(best_reward),
            "curriculum_min_height": (
                float(self._curriculum_min_height)
                if self._curriculum_min_height is not None
                else None
            ),
            # Eval-triggered tracker values — read back by load_checkpoint()
            # so that a resumed run does not reset these to -inf.
            "best_eval_per_step": float(best_eval_per_step),
            "best_eval_success": float(best_eval_success),
            "best_train_reward": float(best_train_reward),
        }

        with open(os.path.join(path, "checkpoint.pkl"), "wb") as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(self, path: str) -> None:
        """Tải checkpoint."""
        import os
        import pickle
        import warnings

        with open(os.path.join(path, "checkpoint.pkl"), "rb") as f:
            checkpoint = pickle.load(f)

        # Version validation — fail fast on incompatible formats, warn on pre-versioned files.
        ckpt_version = checkpoint.get("version", None)
        if ckpt_version is None:
            warnings.warn(
                f"Checkpoint at '{path}' has no version field — written before versioning "
                f"was introduced. Loading with backward-compatible defaults. "
                f"Consider resaving to version {CHECKPOINT_VERSION}.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif ckpt_version > CHECKPOINT_VERSION:
            raise ValueError(
                f"Checkpoint at '{path}' has version {ckpt_version}, but this codebase "
                f"only supports up to version {CHECKPOINT_VERSION}. "
                f"Update the codebase or use the matching checkpoint."
            )
        # ckpt_version == CHECKPOINT_VERSION or None (pre-versioned, handled above): proceed.

        self.params = jax.device_put(checkpoint["params"])
        self.opt_state = jax.device_put(checkpoint["opt_state"])
        self.obs_rms = jax.device_put(checkpoint["obs_rms"])

        # Khôi phục training state (tương thích checkpoint cũ)
        self._resumed_global_step = checkpoint.get("global_step", 0)
        self._resumed_best_reward = checkpoint.get("best_reward", float("-inf"))
        self._resumed_curriculum_min = checkpoint.get("curriculum_min_height", None)
        # Eval-triggered tracker values — absent in older checkpoints; fall back
        # to -inf so the first post-resume eval saves unconditionally (safe).
        self._resumed_best_eval_per_step = checkpoint.get("best_eval_per_step", float("-inf"))
        self._resumed_best_eval_success = checkpoint.get("best_eval_success", float("-inf"))
        self._resumed_best_train_reward = checkpoint.get("best_train_reward", float("-inf"))
        print(
            f"  \U0001f4c2 Checkpoint: step={self._resumed_global_step:,},"
            f" best_reward={self._resumed_best_reward:.4f}"
        )
        if self._resumed_curriculum_min is not None:
            print(f"  \U0001f4c2 Curriculum min_height: {self._resumed_curriculum_min:.2f}")
