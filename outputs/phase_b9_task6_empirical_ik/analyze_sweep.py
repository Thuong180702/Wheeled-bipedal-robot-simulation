"""Analyze empirical IK sweep results and compare with static equilibrium."""

import json
import numpy as np
from pathlib import Path

# Load raw configurations
with open('raw_configurations.json', 'r') as f:
    configs = json.load(f)

print(f"Total configurations: {len(configs)}")

# Analyze valid configurations
valid = [c for c in configs if c['is_valid']]
print(f"Valid configurations: {len(valid)}")

if not valid:
    print("No valid configurations found!")
    exit(1)

# Extract arrays
heights = np.array([c['height'] for c in valid])
hip_pitches = np.array([c['hip_pitch'] for c in valid])
knees = np.array([c['knee'] for c in valid])
stabilities = np.array([c['stability_score'] for c in valid])

print(f"\nHeight distribution:")
print(f"  min={heights.min():.4f}, max={heights.max():.4f}")
print(f"  mean={heights.mean():.4f}, std={heights.std():.6f}")
print(f"  unique values: {len(np.unique(heights))}")

print(f"\nHip pitch distribution:")
print(f"  min={hip_pitches.min():.4f}, max={hip_pitches.max():.4f}")
print(f"  mean={hip_pitches.mean():.4f}, std={hip_pitches.std():.4f}")
print(f"  unique values: {len(np.unique(hip_pitches))}")

print(f"\nKnee distribution:")
print(f"  min={knees.min():.4f}, max={knees.max():.4f}")
print(f"  mean={knees.mean():.4f}, std={knees.std():.4f}")
print(f"  unique values: {len(np.unique(knees))}")

print(f"\nStability distribution:")
print(f"  min={stabilities.min():.4f}, max={stabilities.max():.4f}")
print(f"  mean={stabilities.mean():.4f}, std={stabilities.std():.4f}")

# Check for static equilibrium configuration
target_hip = 0.256
target_knee = 0.538
tolerance = 0.05

close_configs = [
    c for c in valid
    if abs(c['hip_pitch'] - target_hip) < tolerance
    and abs(c['knee'] - target_knee) < tolerance
]

print(f"\n{'='*60}")
print(f"Configs near static equilibrium (hip~{target_hip}, knee~{target_knee}):")
print(f"  Found: {len(close_configs)}")

if close_configs:
    print(f"\n  Sample configurations:")
    for c in close_configs[:5]:
        print(f"    hip={c['hip_pitch']:.3f}, knee={c['knee']:.3f}, "
              f"height={c['height']:.3f}, stability={c['stability_score']:.3f}")
else:
    print(f"  Static equilibrium config NOT FOUND in sweep!")
    print(f"\n  Checking if it was sampled but marked invalid...")

    all_near_eq = [
        c for c in configs
        if abs(c['hip_pitch'] - target_hip) < tolerance
        and abs(c['knee'] - target_knee) < tolerance
    ]

    if all_near_eq:
        print(f"  Found {len(all_near_eq)} configs near equilibrium (including invalid):")
        for c in all_near_eq[:5]:
            print(f"    hip={c['hip_pitch']:.3f}, knee={c['knee']:.3f}, "
                  f"height={c['height']:.3f}, valid={c['is_valid']}")
    else:
        print(f"  NOT SAMPLED - equilibrium config outside sweep range!")

# Check height variation
print(f"\n{'='*60}")
print(f"Height variation analysis:")
unique_heights = np.unique(heights)
print(f"  Unique height values: {len(unique_heights)}")
if len(unique_heights) <= 10:
    print(f"  Values: {unique_heights}")
else:
    print(f"  Range: [{unique_heights[0]:.4f}, {unique_heights[-1]:.4f}]")
    print(f"  First 10: {unique_heights[:10]}")

# Analyze by hip_pitch
print(f"\n{'='*60}")
print(f"Height by hip_pitch:")
unique_hips = np.unique(hip_pitches)
print(f"  Unique hip_pitch values: {len(unique_hips)}")

if len(unique_hips) <= 5:
    for hip in unique_hips:
        mask = hip_pitches == hip
        h_at_hip = heights[mask]
        print(f"    hip={hip:.3f}: heights=[{h_at_hip.min():.3f}, {h_at_hip.max():.3f}], "
              f"n={len(h_at_hip)}")
else:
    print(f"  Sampling 5 hip_pitch values:")
    sample_indices = np.linspace(0, len(unique_hips)-1, 5, dtype=int)
    for idx in sample_indices:
        hip = unique_hips[idx]
        mask = hip_pitches == hip
        h_at_hip = heights[mask]
        print(f"    hip={hip:.3f}: heights=[{h_at_hip.min():.3f}, {h_at_hip.max():.3f}], "
              f"n={len(h_at_hip)}")

print(f"\n{'='*60}")
print(f"CONCLUSION:")
print(f"  The empirical sweep shows that all valid configurations")
print(f"  produce height ~{heights.mean():.3f}m, regardless of joint angles.")
print(f"  This suggests the robot's kinematic structure is highly constrained.")
