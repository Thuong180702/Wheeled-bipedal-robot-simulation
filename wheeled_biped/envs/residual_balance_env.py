"""
Residual Balance Environment - Bounded residual PPO over LQR/IK prior.

Task: Learn dynamic balance corrections over a height-dependent LQR/IK prior.
The prior provides geometric height tracking and sagittal stabilization but cannot
achieve standalone static balance (Phase B.4 feasibility study: 0.0023% static
feasibility across height range). Residual policy must learn CoM correction and
dynamic balance.

Observation: 52 dims = base (42) + base_action_abs (10)
    Base 42: [g_body(3), ang_vel(3), lin_vel(3), qpos(10), qvel(10),
              prev_final_action(10), height_cmd(1), current_height(1), yaw_error(1)]
    base_action_abs (10): LQR/IK prior output (absolute normalized)

Policy action: residual_action ∈ [-1, 1]^10 (bounded correction)

Action composition:
    base_action_abs = LQR/IK prior(obs, height_cmd)
    final_action_abs = clip(base_action_abs + residual_scale * residual_action, -1, 1)
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from wheeled_biped.controllers.action_codec import compose_residual_action
from wheeled_biped.controllers.lqr_ik_prior import LQRIKConfig, LQRIKPrior
from wheeled_biped.envs.balance_env import BalanceEnv, EnvState
from wheeled_biped.rewards.reward_functions import (
    penalty_residual_magnitude,
    penalty_residual_rate,
    penalty_residual_saturation,
)


class ResidualBalanceEnv(BalanceEnv):
    """Residual balance environment with LQR/IK prior.

    Extends BalanceEnv with:
    - 52-dim observation (42 base + 10 base_action_abs)
    - Policy outputs residual_action only
    - LQR/IK prior computes base_action_abs
    - Action composition via action_codec.compose_residual_action()
    - Logging of all action components
    """

    def __init__(self, config: dict[str, Any] | None = None, **kwargs):
        super().__init__(config=config, **kwargs)

        # Load residual-specific config
        residual_cfg = self.config.get("residual", {})

        # Residual scale (per-joint or scalar)
        residual_scale_cfg = residual_cfg.get("residual_scale", None)
        if residual_scale_cfg is None:
            # Default: higher authority for wheels, moderate for hip_pitch/knee
            residual_scale_cfg = [
                0.10,  # l_hip_roll
                0.05,  # l_hip_yaw
                0.20,  # l_hip_pitch
                0.20,  # l_knee
                0.40,  # l_wheel
                0.10,  # r_hip_roll
                0.05,  # r_hip_yaw
                0.20,  # r_hip_pitch
                0.20,  # r_knee
                0.40,  # r_wheel
            ]

        if isinstance(residual_scale_cfg, list):
            if len(residual_scale_cfg) != self.num_actions:
                raise ValueError(
                    f"residual_scale must have {self.num_actions} elements, "
                    f"got {len(residual_scale_cfg)}"
                )
            self._residual_scale = jnp.array(residual_scale_cfg, dtype=jnp.float32)
        else:
            self._residual_scale = jnp.float32(residual_scale_cfg)

        # Initialize LQR/IK prior
        prior_config_path = residual_cfg.get(
            "prior_config",
            "configs/controllers/gain_scheduled_lqr.yaml",
        )
        prior_config_path = Path(prior_config_path)
        if not prior_config_path.is_absolute():
            # Resolve relative to repo root
            repo_root = Path(__file__).parent.parent.parent
            prior_config_path = repo_root / prior_config_path

        lqr_ik_config = LQRIKConfig.from_yaml(prior_config_path)
        self._lqr_ik_prior = LQRIKPrior(lqr_ik_config, self.mj_model)

    def _compute_obs_size(self) -> int:
        """Observation = base (42) + base_action_abs (10) = 52."""
        base_obs_size = super()._compute_obs_size()  # 42
        return base_obs_size + self.num_actions  # 52

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: jax.Array) -> EnvState:
        """Reset with base_action_abs appended to observation."""
        # Get base reset state (42-dim obs)
        base_state = super().reset(rng)

        # Compute initial base_action_abs from prior
        base_action_abs = self._compute_base_action(base_state.obs)

        # Append base_action_abs to observation
        obs = jnp.concatenate([base_state.obs, base_action_abs])

        # Initialize all residual info keys to ensure structure matches step()
        new_info = {
            **base_state.info,
            "base_action_abs": base_action_abs,
            "residual_action": jnp.zeros(self.num_actions),
            "residual_scaled": jnp.zeros(self.num_actions),
            "final_action_abs": base_action_abs,  # Initially same as base
            "residual_norm": jnp.float32(0.0),
            "residual_saturation_rate": jnp.float32(0.0),
        }

        return base_state._replace(obs=obs, info=new_info)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: EnvState, action: jnp.ndarray) -> EnvState:
        """Step with residual action composition.

        Args:
            state: Current state with 52-dim obs.
            action: Policy output (residual_action), shape (10,).

        Returns:
            New state with updated obs, reward, done, and action logging.
        """
        # Extract base observation (first 42 dims)
        base_obs_42 = state.obs[:42]

        # Compute base_action_abs from prior
        base_action_abs = self._compute_base_action(base_obs_42)

        # Compose residual action
        action_breakdown = compose_residual_action(
            base_action_abs=base_action_abs,
            residual_action=action,
            residual_scale=self._residual_scale,
            clip=True,
        )

        # Use final_action_abs for environment step
        final_action_abs = action_breakdown.final_action_abs

        # Create temporary state with base obs for parent step()
        # Parent step() expects 42-dim obs and will compute new 42-dim obs
        # Store current action breakdown in temp_state.info so _compute_reward()
        # can access current residual actions instead of stale prev_state.info
        temp_state = state._replace(
            obs=base_obs_42,
            info={
                **state.info,
                "current_base_action_abs": base_action_abs,
                "current_residual_action": action,
                "current_residual_scaled": action_breakdown.residual_scaled,
                "current_final_action_abs": final_action_abs,
            }
        )

        # Call parent step with final_action_abs
        new_base_state = super().step(temp_state, final_action_abs)

        # Compute new base_action_abs for next step
        new_base_action_abs = self._compute_base_action(new_base_state.obs)

        # Append new base_action_abs to observation
        new_obs = jnp.concatenate([new_base_state.obs, new_base_action_abs])

        # Add residual-specific logging to info
        new_info = {
            **new_base_state.info,
            "base_action_abs": new_base_action_abs,
            "residual_action": action,
            "residual_scaled": action_breakdown.residual_scaled,
            "final_action_abs": final_action_abs,
            "residual_norm": action_breakdown.residual_norm,
            "residual_saturation_rate": action_breakdown.residual_saturation_rate,
        }

        return new_base_state._replace(obs=new_obs, info=new_info)

    def _compute_base_action(self, base_obs_42: jnp.ndarray) -> jnp.ndarray:
        """Compute base_action_abs from LQR/IK prior.

        Args:
            base_obs_42: Base observation (42 dims).

        Returns:
            base_action_abs: Prior output, shape (10,).

        Performance Warning:
            This uses jax.pure_callback with vmap_method='sequential' to call
            a NumPy-based LQR/IK prior. This is a KNOWN PERFORMANCE BOTTLENECK:
            - Breaks JAX's parallelization across the batch dimension
            - Forces sequential execution for each environment
            - Prevents XLA optimization of the prior computation
            - May become the dominant cost in large-scale training (4096+ envs)

            Future optimization paths:
            1. Rewrite LQR/IK prior as pure JAX (preferred for performance)
            2. Profile actual training throughput impact before optimizing
            3. Consider caching if prior is deterministic for given obs
            4. Benchmark vmap_method='broadcast_all' if prior is vectorizable

            Current choice: Accept the performance cost for Phase C prototype.
            Revisit if training throughput becomes a blocker for 1M+ step runs.
        """
        # Use pure_callback to call NumPy-based prior from JIT context
        def _prior_callback(obs_array):
            import numpy as np
            obs_np = np.asarray(obs_array)
            action_np = self._lqr_ik_prior.compute_action(obs_np)
            return np.asarray(action_np, dtype=np.float32)

        base_action_abs = jax.pure_callback(
            _prior_callback,
            jax.ShapeDtypeStruct((self.num_actions,), jnp.float32),
            base_obs_42,
            vmap_method='sequential',
        )
        return base_action_abs

    @functools.partial(jax.jit, static_argnums=(0,))
    def _compute_reward(
        self,
        mjx_data: mjx.Data,
        action: jnp.ndarray,
        prev_state: EnvState,
    ) -> jnp.ndarray:
        """Compute reward with residual-specific terms.

        Args:
            mjx_data: MuJoCo data.
            action: final_action_abs (used by parent for action_rate).
            prev_state: Previous state with residual info.

        Returns:
            Total reward.
        """
        # Get base reward from parent
        base_reward = super()._compute_reward(mjx_data, action, prev_state)

        # Add residual-specific penalties if configured
        residual_weights = {
            "residual_magnitude": self._reward_weights.get("residual_magnitude", 0.0),
            "residual_rate": self._reward_weights.get("residual_rate", 0.0),
            "residual_saturation": self._reward_weights.get("residual_saturation", 0.0),
        }

        # Only compute residual penalties if any weight is non-zero
        if any(abs(w) > 1e-6 for w in residual_weights.values()):
            # Extract CURRENT residual info from prev_state.info
            # (step() stores current action breakdown in temp_state.info before calling parent)
            residual_scaled = prev_state.info.get("current_residual_scaled", jnp.zeros(self.num_actions))
            residual_action = prev_state.info.get("current_residual_action", jnp.zeros(self.num_actions))
            final_action_abs = prev_state.info.get("current_final_action_abs", action)

            # Compute residual penalties
            residual_components = {}

            if abs(residual_weights["residual_magnitude"]) > 1e-6:
                residual_components["residual_magnitude"] = penalty_residual_magnitude(
                    residual_scaled
                )

            if abs(residual_weights["residual_rate"]) > 1e-6:
                # Get PREVIOUS residual action for rate computation
                # For first step, use zeros
                prev_residual = jnp.zeros(self.num_actions)
                if "residual_action" in prev_state.info:
                    prev_residual = prev_state.info["residual_action"]
                residual_components["residual_rate"] = penalty_residual_rate(
                    residual_action, prev_residual
                )

            if abs(residual_weights["residual_saturation"]) > 1e-6:
                residual_components["residual_saturation"] = penalty_residual_saturation(
                    final_action_abs
                )

            # Add weighted residual penalties to base reward
            for name, value in residual_components.items():
                weight = residual_weights[name]
                base_reward = base_reward + weight * value

        return base_reward
