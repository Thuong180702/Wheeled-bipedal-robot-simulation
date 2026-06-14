"""Diagnose why prior evaluation fails immediately.

Minimal rollout with detailed logging to identify the root cause of
100% fall rate and ~0.4s survival time for all variants.
"""

import jax
import mujoco
import numpy as np
from pathlib import Path

from wheeled_biped.controllers.lqr_ik_prior import LQRIKConfig, LQRIKPrior
from wheeled_biped.envs.balance_env import BalanceEnv
from wheeled_biped.utils.config import get_model_path


def diagnose_single_rollout():
    """Run a single rollout with detailed logging."""
    print("\n=== Diagnostic Rollout ===\n")

    # Load model and create environment with PID enabled
    model_path = get_model_path()
    mj_model = mujoco.MjModel.from_xml_path(str(model_path))
    env_config = {
        "task": {"name": "balance"},
        "low_level_pid": {
            "enabled": True,
            "disable_pid_action_bias": True,
            "action_smoothing_alpha": 0.5,
            "anti_windup_limit": 0.4,
            "wheel_vel_limit": 20.0,
            "kp": [55.0, 40.0, 70.0, 70.0, 4.0, 55.0, 40.0, 70.0, 70.0, 4.0],
            "ki": [0.8, 0.4, 1.0, 1.0, 0.1, 0.8, 0.4, 1.0, 1.0, 0.1],
            "kd": [3.0, 2.0, 4.0, 4.0, 0.0, 3.0, 2.0, 4.0, 4.0, 0.0],
            "action_delay_steps": 0,
        },
        "domain_randomization": {
            "enabled": False,
        },
    }
    env = BalanceEnv(config=env_config)

    print(f"Environment obs_size: {env.obs_size}")
    print(f"Environment num_actions: {env.num_actions}")
    print(f"Environment CONTROL_DT: {env.CONTROL_DT}")

    # Create prior controller
    base_config_path = Path("configs/controllers/gain_scheduled_lqr.yaml")
    config = LQRIKConfig.from_yaml(base_config_path)
    prior = LQRIKPrior(config, mj_model)

    print(f"\nPrior config loaded: {config.variant_name}")

    # Reset environment
    rng = jax.random.PRNGKey(42)
    state = env.reset(rng)

    print(f"\nInitial state:")
    print(f"  obs shape: {state.obs.shape}")
    print(f"  obs dtype: {state.obs.dtype}")
    print(f"  done: {state.done}")

    # Check initial observation values
    obs_np = np.array(state.obs)
    print(f"\nInitial observation values:")
    print(f"  g_body (0:3): {obs_np[0:3]}")
    print(f"  body_lin_vel (3:6): {obs_np[3:6]}")
    print(f"  body_ang_vel (6:9): {obs_np[6:9]}")
    print(f"  qpos (9:19): {obs_np[9:19]}")
    print(f"  qvel (19:29): {obs_np[19:29]}")
    print(f"  prev_action (29:39): {obs_np[29:39]}")
    print(f"  height_cmd (39): {obs_np[39]}")
    print(f"  current_height (40): {obs_np[40]}")
    print(f"  yaw_error (41): {obs_np[41]}")

    # Override height command (normalize to [0, 1])
    height = 0.70
    height_norm = (height - env.MIN_HEIGHT_CMD) / (env.MAX_HEIGHT_CMD - env.MIN_HEIGHT_CMD)
    obs = state.obs.at[39].set(height_norm)
    state = state._replace(obs=obs)

    print(f"\nAfter height override:")
    print(f"  height_cmd (39): {np.array(state.obs)[39]} (normalized)")
    print(f"  height_cmd (meters): {height}")

    # Compute action from prior
    obs_np = np.array(state.obs)
    print(f"\nComputing action from prior...")
    print(f"  obs_np shape: {obs_np.shape}")
    print(f"  obs_np dtype: {obs_np.dtype}")

    # Print LQR state components
    g_body = obs_np[0:3]
    body_ang_vel = obs_np[3:6]
    body_lin_vel = obs_np[6:9]
    pitch = -np.arcsin(np.clip(g_body[1], -1.0, 1.0))
    pitch_rate = body_ang_vel[1]
    fwd_vel = body_lin_vel[1]
    print(f"\n  LQR state components:")
    print(f"    pitch: {np.rad2deg(pitch):.2f}°")
    print(f"    pitch_rate: {pitch_rate:.3f} rad/s")
    print(f"    fwd_vel: {fwd_vel:.3f} m/s")
    print(f"    LQR gains: {prior.lqr_gains}")

    try:
        action_np = prior.compute_action(obs_np)
        print(f"\n  action_np shape: {action_np.shape}")
        print(f"  action_np dtype: {action_np.dtype}")
        print(f"  action_np: {action_np}")
        print(f"  action_np min/max: [{action_np.min():.3f}, {action_np.max():.3f}]")
        print(f"  action_np has NaN: {np.any(np.isnan(action_np))}")
        print(f"  action_np has Inf: {np.any(np.isinf(action_np))}")
    except Exception as e:
        print(f"  ERROR computing action: {e}")
        import traceback
        traceback.print_exc()
        return

    # Convert to JAX array
    action = jax.numpy.array(action_np)

    # Step environment
    print(f"\nStepping environment...")
    try:
        state = env.step(state, action)
        print(f"  Step successful")
        print(f"  done: {state.done}")
        print(f"  reward: {state.reward}")
    except Exception as e:
        print(f"  ERROR stepping environment: {e}")
        import traceback
        traceback.print_exc()
        return

    # Check post-step state
    obs_np = np.array(state.obs)
    print(f"\nPost-step observation:")
    print(f"  g_body (0:3): {obs_np[0:3]}")
    print(f"  pitch: {-np.arcsin(np.clip(obs_np[1], -1.0, 1.0)) * 180/np.pi:.1f}°")
    print(f"  roll: {np.arcsin(np.clip(obs_np[0], -1.0, 1.0)) * 180/np.pi:.1f}°")
    print(f"  current_height (37): {obs_np[37]:.3f}")

    # Run full episode (up to 1000 steps)
    print(f"\nRunning full episode (max 1000 steps)...")
    for step in range(1000):
        obs_np = np.array(state.obs)

        # Log every 10 steps
        if (step + 1) % 10 == 0 or step < 5:
            g_body = obs_np[0:3]
            body_lin_vel = obs_np[3:6]
            body_ang_vel = obs_np[6:9]
            pitch = -np.arcsin(np.clip(g_body[1], -1.0, 1.0)) * 180/np.pi
            pitch_rate = body_ang_vel[1]
            fwd_vel = body_lin_vel[1]
            print(f"  Step {step+1}: pitch={pitch:.1f}°, pitch_rate={pitch_rate:.3f}, fwd_vel={fwd_vel:.3f}")

        action_np = prior.compute_action(obs_np)

        # Log action every 10 steps
        if (step + 1) % 10 == 0 or step < 5:
            print(f"    action: hip_pitch={action_np[2]:.3f}, knee={action_np[3]:.3f}, wheel={action_np[4]:.3f}")

        action = jax.numpy.array(action_np)
        state = env.step(state, action)

        if state.done:
            print(f"  Episode terminated at step {step+1}")
            break

    survival_time = (step + 2) * env.CONTROL_DT
    fall = survival_time < 10.0
    print(f"\nSurvival time: {survival_time:.2f}s")
    print(f"Fall detected: {fall} (threshold: 10.0s)")


def compare_obs_structures():
    """Compare observation structure between unit test and evaluation."""
    print("\n=== Observation Structure Comparison ===\n")

    # Unit test observation (from test_lqr_ik_prior_variants.py)
    print("Unit test observation structure (42 dims):")
    print("  0:3   - g_body (gravity in body frame)")
    print("  3:6   - body_ang_vel")
    print("  6:16  - qpos (10 joints)")
    print("  16:26 - qvel (10 joints)")
    print("  26:36 - prev_action (10)")
    print("  36    - height_cmd")
    print("  37    - current_height")
    print("  38    - yaw_error")
    print("  Total: 39 dims")

    # Check BalanceEnv observation
    env = BalanceEnv(config={"task": {"name": "balance"}})
    print(f"\nBalanceEnv obs_size: {env.obs_size}")

    # Reset and check actual observation
    rng = jax.random.PRNGKey(42)
    state = env.reset(rng)
    obs_np = np.array(state.obs)

    print(f"Actual observation shape: {obs_np.shape}")
    print(f"Actual observation length: {len(obs_np)}")

    if len(obs_np) != 42:
        print(f"\nWARNING: Expected 42 dims, got {len(obs_np)}")
        print("This could be the root cause of the evaluation failure!")


if __name__ == "__main__":
    compare_obs_structures()
    diagnose_single_rollout()
