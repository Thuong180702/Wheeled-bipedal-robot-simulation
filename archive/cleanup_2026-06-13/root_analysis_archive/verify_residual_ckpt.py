import pickle
import sys

ckpt_path = sys.argv[1]

with open(ckpt_path, 'rb') as f:
    ckpt = pickle.load(f)

print('=== Checkpoint Metadata ===')
print(f"policy_type: {ckpt.get('policy_type', 'NOT SET')}")
print(f"action_mode: {ckpt.get('action_mode', 'NOT SET')}")
print(f"obs_dim: {ckpt.get('obs_dim', 'NOT SET')}")
print(f"action_dim: {ckpt.get('action_dim', 'NOT SET')}")
print(f"base_action_in_obs: {ckpt.get('base_action_in_obs', 'NOT SET')}")
print(f"residual_scale: {ckpt.get('residual_scale', 'NOT SET')}")
print(f"base_controller_config: {ckpt.get('base_controller_config', 'NOT SET')}")
print(f"smoothing_alpha: {ckpt.get('smoothing_alpha', 'NOT SET')}")
print(f"action_delay_steps: {ckpt.get('action_delay_steps', 'NOT SET')}")
print()
print(f"obs_rms shape: {ckpt['obs_rms']['mean'].shape}")
print(f"global_step: {ckpt['global_step']}")
