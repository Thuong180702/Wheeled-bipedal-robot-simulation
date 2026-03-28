"""
Actor-Critic Networks cho PPO, viết bằng Flax (JAX).

Kiến trúc:
  - Actor: MLP → Gaussian distribution (mean + log_std)
  - Critic: MLP → scalar value

Tất cả tương thích jax.jit.
"""

from __future__ import annotations

from collections.abc import Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp


class MLP(nn.Module):
    """Multi-Layer Perceptron với activation tùy chọn.

    Attributes:
        features: danh sách kích thước hidden layer.
        activation: tên hàm activation ("elu", "relu", "tanh").
        output_size: kích thước output cuối.
    """

    features: Sequence[int]
    activation: str = "elu"
    output_size: int | None = None

    def _get_activation(self):
        """Trả về hàm activation."""
        activations = {
            "elu": nn.elu,
            "relu": nn.relu,
            "tanh": nn.tanh,
            "silu": nn.silu,
        }
        return activations.get(self.activation, nn.elu)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        act_fn = self._get_activation()
        for feat in self.features:
            x = nn.Dense(feat, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
            x = nn.LayerNorm()(x)
            x = act_fn(x)
        if self.output_size is not None:
            x = nn.Dense(
                self.output_size,
                kernel_init=nn.initializers.orthogonal(0.01),
            )(x)
        return x


class ActorNetwork(nn.Module):
    """Actor network: output mean và log_std cho Gaussian policy.

    Attributes:
        hidden_sizes: kích thước hidden layers.
        action_size: số lượng action.
        activation: hàm activation.
        init_std: std khởi tạo ban đầu.
    """

    hidden_sizes: Sequence[int] = (256, 256, 128)
    action_size: int = 10
    activation: str = "elu"
    init_std: float = 1.0

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> distrax.Distribution:
        # MLP backbone
        x = MLP(
            features=self.hidden_sizes,
            activation=self.activation,
        )(obs)

        # Mean
        mean = nn.Dense(
            self.action_size,
            kernel_init=nn.initializers.orthogonal(0.01),
        )(x)

        # Log std (learnable parameter, không phụ thuộc obs)
        log_std = self.param(
            "log_std",
            nn.initializers.constant(jnp.log(self.init_std)),
            (self.action_size,),
        )

        # Clip log_std để tránh quá nhỏ/lớn
        log_std = jnp.clip(log_std, -5.0, 2.0)

        return distrax.MultivariateNormalDiag(
            loc=mean,
            scale_diag=jnp.exp(log_std),
        )


class CriticNetwork(nn.Module):
    """Critic network: ước lượng state-value V(s).

    Attributes:
        hidden_sizes: kích thước hidden layers.
        activation: hàm activation.
    """

    hidden_sizes: Sequence[int] = (256, 256, 128)
    activation: str = "elu"

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = MLP(
            features=self.hidden_sizes,
            activation=self.activation,
            output_size=1,
        )(obs)
        return x.squeeze(-1)  # (batch,) thay vì (batch, 1)


class ActorCritic(nn.Module):
    """Kết hợp Actor + Critic trong một module.

    Dùng shared observation nhưng separate backbones.

    Attributes:
        actor_hidden: hidden sizes cho actor.
        critic_hidden: hidden sizes cho critic.
        action_size: số action.
        activation: hàm activation.
        init_std: std khởi tạo.
    """

    actor_hidden: Sequence[int] = (256, 256, 128)
    critic_hidden: Sequence[int] = (256, 256, 128)
    action_size: int = 10
    activation: str = "elu"
    init_std: float = 1.0

    def setup(self):
        self.actor = ActorNetwork(
            hidden_sizes=self.actor_hidden,
            action_size=self.action_size,
            activation=self.activation,
            init_std=self.init_std,
        )
        self.critic = CriticNetwork(
            hidden_sizes=self.critic_hidden,
            activation=self.activation,
        )

    def __call__(self, obs: jnp.ndarray) -> tuple[distrax.Distribution, jnp.ndarray]:
        """Forward pass.

        Args:
            obs: observation (batch_size, obs_size).

        Returns:
            (action_dist, value)
        """
        return self.actor(obs), self.critic(obs)

    def get_action_and_value(
        self, obs: jnp.ndarray, rng: jax.Array
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Lấy action, log_prob, entropy, value.

        Dùng trong rollout phase.

        Args:
            obs: observation.
            rng: random key.

        Returns:
            (action, log_prob, entropy, value)
        """
        dist, value = self(obs)
        action = dist.sample(seed=rng)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def evaluate_action(
        self, obs: jnp.ndarray, action: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Đánh giá action đã cho (dùng trong PPO update).

        Args:
            obs: observation.
            action: action cần đánh giá.

        Returns:
            (log_prob, entropy, value)
        """
        dist, value = self(obs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, value


def create_actor_critic(
    obs_size: int,
    action_size: int,
    config: dict,
    rng: jax.Array,
) -> tuple[ActorCritic, dict]:
    """Tạo và khởi tạo ActorCritic model.

    Args:
        obs_size: kích thước observation.
        action_size: số lượng action.
        config: network config dict.
        rng: random key.

    Returns:
        (model, params)
    """
    net_cfg = config.get("network", {})

    model = ActorCritic(
        actor_hidden=tuple(net_cfg.get("policy_hidden", [256, 256, 128])),
        critic_hidden=tuple(net_cfg.get("value_hidden", [256, 256, 128])),
        action_size=action_size,
        activation=net_cfg.get("activation", "elu"),
        init_std=net_cfg.get("init_std", 1.0),
    )

    # Khởi tạo params
    dummy_obs = jnp.zeros(obs_size)
    params = model.init(rng, dummy_obs)

    return model, params
