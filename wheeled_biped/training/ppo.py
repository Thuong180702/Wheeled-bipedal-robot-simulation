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
from wheeled_biped.training.networks import ActorCritic, create_actor_critic
from wheeled_biped.utils.logger import TrainingLogger


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
    ):
        self.env = env
        self.config = config
        self.logger = logger
        self._stop_requested = False  # Flag để dừng training từ bên ngoài
        self._resumed_global_step = 0  # Sẽ được set khi load_checkpoint
        self._resumed_best_reward = float("-inf")

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
        self.rng = jax.random.PRNGKey(42)
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

        total_loss = (
            policy_loss
            + self.value_coeff * value_loss
            + self.entropy_coeff * entropy_loss
        )

        metrics = {
            "loss/total": total_loss,
            "loss/policy": policy_loss,
            "loss/value": value_loss,
            "loss/entropy": -entropy_loss,
            "policy/clip_fraction": jnp.mean(jnp.abs(ratio - 1.0) > self.clip_epsilon),
            "policy/approx_kl": jnp.mean(
                0.5 * jnp.square(log_prob - batch.old_log_prob)
            ),
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

            # Clip action cho environment (nhưng giữ raw_action cho PPO ratio)
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
                action=raw_action,  # Unclipped action: PPO ratio consistent
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

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Live viewer — ưu tiên external viewer (đã chạy trên main thread)
        viewer = _external_viewer
        if viewer is None and live_view:
            try:
                from wheeled_biped.training.live_viewer import LiveTrainingViewer

                viewer = LiveTrainingViewer(self.env.mj_model, title="PPO Training")
                # Không gọi start() — sẽ warning vì không chạy trên main thread
                print("  ⚠️  live_view=True nhưng viewer cần chạy trên main thread.")
                print(
                    "     Hãy dùng:  python scripts/train.py single --stage balance --live-view"
                )
                viewer = None
            except Exception as e:
                print(f"  ⚠️  Không mở được live viewer: {e}")
                viewer = None

        self._stop_requested = False

        # Detect CPU vs GPU
        backend = jax.default_backend()
        is_cpu = backend == "cpu"
        if is_cpu and self.num_envs > 256:
            print(
                f"  ⚠️  CPU + num_envs={self.num_envs}: JIT compile sẽ rất lâu (10-20 phút)!"
            )
            print(f"      Trên CPU nên dùng --num-envs 64~128 (JIT ~1-2 phút)")
            print(
                f"      Tăng num_envs trên CPU không nhanh hơn: mỗi update lâu hơn tỉ lệ thuận"
            )
            print(f"      GPU (jax[cuda12]) mới thật sự song song 4096 envs")
            print()

        # Reset environments
        self.rng, reset_key = jax.random.split(self.rng)
        env_state = self.env.v_reset(reset_key, self.num_envs)

        steps_per_update = self._rollout_length * self.num_envs

        # Tính số updates còn lại (trừ bước đã train trước đó nếu resume)
        resumed_step = self._resumed_global_step
        remaining_steps = max(steps_per_update, total_steps - resumed_step)
        num_updates = remaining_steps // steps_per_update

        print(f"═══ PPO Training ═══")
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
        print(f"════════════════════")

        # Warmup: JIT compile lần đầu (rất lâu — thông báo cho user)
        print(f"  ⏳ Đang JIT compile (lần đầu có thể mất 1-3 phút)...")
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
        raw_obs = (
            warmup_transitions.obs * jnp.sqrt(self.obs_rms.var + 1e-8)
            + self.obs_rms.mean
        )
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
            )
            if viewer is not None:
                viewer.request_stop()
            return {"best_reward": self._resumed_best_reward, "total_steps": gs}

        global_step = resumed_step + steps_per_update  # Tính cả bước đã resume
        start_time = time.time()
        best_reward = self._resumed_best_reward  # Giữ best_reward từ lần trước

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

        last_update_time = time.time()

        try:
            for update in range(1, num_updates):  # Bắt đầu từ 1 (đã warmup update 0)
                # Kiểm tra stop flag (Ctrl+C hoặc viewer đóng)
                if self._stop_requested:
                    print(
                        f"\n  🛑 Training dừng theo yêu cầu tại update {update}/{num_updates}"
                    )
                    break
                if viewer is not None and not viewer.is_running:
                    print(
                        f"\n  🛑 Viewer đã đóng, dừng training tại update {update}/{num_updates}"
                    )
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
                raw_obs = (
                    transitions.obs * jnp.sqrt(self.obs_rms.var + 1e-8)
                    + self.obs_rms.mean
                )
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
                elapsed_total = time.time() - start_time
                fps = global_step / max(elapsed_total, 1)
                eta_s = (num_updates - update) * update_elapsed
                eta_m = eta_s / 60

                # Luôn in progress mỗi update
                print(
                    f"  [{update}/{num_updates}] "
                    f"step={global_step:,} | "
                    f"reward={avg_reward:.4f} | "
                    f"fps={fps:.0f} | "
                    f"{update_elapsed:.1f}s/update | "
                    f"ETA {eta_m:.0f}m",
                    flush=True,
                )

                # Logging chi tiết
                if update % log_interval == 0:
                    log_metrics = {
                        "reward/mean": avg_reward,
                        "reward/std": float(jnp.std(transitions.reward)),
                        "training/fps": fps,
                        "training/global_step": global_step,
                        **{k: float(v) for k, v in metrics.items()},
                    }

                    if self.logger:
                        self.logger.set_step(global_step)
                        self.logger.log_dict(log_metrics)

                    # Save best
                    if avg_reward > best_reward:
                        best_reward = avg_reward

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
                    )

        except KeyboardInterrupt:
            print(f"\n\n⚠️  Training bị dừng bởi người dùng (Ctrl+C)")
            print(f"   Đã train {global_step:,} steps, best_reward={best_reward:.4f}")
            print(f"   Đang lưu checkpoint cuối...")

        # Save final
        self._save_checkpoint(
            os.path.join(checkpoint_dir, "final"),
            global_step=global_step,
            best_reward=best_reward,
        )

        if self.logger:
            self.logger.close()

        if viewer is not None:
            viewer.request_stop()

        return {"best_reward": best_reward, "total_steps": global_step}

    def _save_checkpoint(
        self,
        path: str,
        global_step: int = 0,
        best_reward: float = float("-inf"),
    ) -> None:
        """Lưu checkpoint (params + obs_rms + training state)."""
        import os
        import pickle

        os.makedirs(path, exist_ok=True)

        checkpoint = {
            "params": jax.device_get(self.params),
            "opt_state": jax.device_get(self.opt_state),
            "obs_rms": jax.device_get(self.obs_rms),
            "config": self.config,
            "global_step": int(global_step),
            "best_reward": float(best_reward),
        }

        with open(os.path.join(path, "checkpoint.pkl"), "wb") as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(self, path: str) -> None:
        """Tải checkpoint."""
        import os
        import pickle

        with open(os.path.join(path, "checkpoint.pkl"), "rb") as f:
            checkpoint = pickle.load(f)

        self.params = jax.device_put(checkpoint["params"])
        self.opt_state = jax.device_put(checkpoint["opt_state"])
        self.obs_rms = jax.device_put(checkpoint["obs_rms"])

        # Khôi phục training state (tương thích checkpoint cũ)
        self._resumed_global_step = checkpoint.get("global_step", 0)
        self._resumed_best_reward = checkpoint.get("best_reward", float("-inf"))
        print(
            f"  📂 Checkpoint: step={self._resumed_global_step:,}, best_reward={self._resumed_best_reward:.4f}"
        )
