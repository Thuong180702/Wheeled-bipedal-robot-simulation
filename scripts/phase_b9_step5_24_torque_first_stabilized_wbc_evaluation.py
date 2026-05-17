#!/usr/bin/env python3
"""
Phase B.9 Step 5.24 — Torque-First Stabilized WBC

Add low-gain stabilization to torque-first WBC while preserving WBC dominance.

Goal: Improve stability without reintroducing PID authority suppression.

Stabilization components tested:
- Velocity damping (damping_gain)
- Temporal smoothing (smoothing_alpha)
- Weak impedance (impedance_kp)

WBC authority must remain >70%.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import yaml
from mujoco import mjx

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from wheeled_biped.controllers.wbc_balance_controller import WBCBalanceController
from wheeled_biped.envs.balance_env import BalanceEnv

OUTPUT_DIR = project_root / "outputs" / "phase_b9_step5_24_torque_first_stabilized_wbc"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class StabilizationCandidate:
    """Stabilization configuration candidate."""
    name: str
    k_roll: float
    k_roll_rate: float
    k_pitch: float
    k_pitch_rate: float
    wbc_authority_fraction: float
    damping_gain: float
    smoothing_alpha: float
    impedance_kp: float
    use_impedance_target: bool


def compute_impedance_target(obs: np.ndarray) -> np.ndarray:
    """Compute weak impedance target (nominal standing pose)."""
    # Nominal standing pose: slight knee bend, neutral hips
    target = np.zeros(10, dtype=np.float32)
    # L_HIP_ROLL, L_HIP_YAW, L_HIP_PITCH, L_KNEE, L_WHEEL
    target[0] = 0.0  # L_HIP_ROLL
    target[1] = 0.0  # L_HIP_YAW
    target[2] = 0.0  # L_HIP_PITCH
    target[3] = 0.15  # L_KNEE (slight bend)
    target[4] = 0.0  # L_WHEEL
    # R_HIP_ROLL, R_HIP_YAW, R_HIP_PITCH, R_KNEE, R_WHEEL
    target[5] = 0.0  # R_HIP_ROLL
    target[6] = 0.0  # R_HIP_YAW
    target[7] = 0.0  # R_HIP_PITCH
    target[8] = 0.15  # R_KNEE (slight bend)
    target[9] = 0.0  # R_WHEEL
    return target


def run_stabilization_ablation():
    """Test stabilization components separately and in combinations."""
    print("\n=== Phase B.9 Step 5.24: Stabilization Ablation ===\n")

    # Define ablation candidates
    candidates = [
        # Baseline: pure WBC (Step 5.22 reproduction)
        StabilizationCandidate(
            name="baseline_pure_wbc",
            k_roll=20.0,
            k_roll_rate=2.0,
            k_pitch=5.0,
            k_pitch_rate=0.5,
            wbc_authority_fraction=1.0,
            damping_gain=0.0,
            smoothing_alpha=0.0,
            impedance_kp=0.0,
            use_impedance_target=False,
        ),
        # Damping only
        StabilizationCandidate(
            name="damping_light",
            k_roll=20.0,
            k_roll_rate=2.0,
            k_pitch=5.0,
            k_pitch_rate=0.5,
            wbc_authority_fraction=1.0,
            damping_gain=0.5,
            smoothing_alpha=0.0,
            impedance_kp=0.0,
            use_impedance_target=False,
        ),
        StabilizationCandidate(
            name="damping_moderate",
            k_roll=20.0,
            k_roll_rate=2.0,
            k_pitch=5.0,
            k_pitch_rate=0.5,
            wbc_authority_fraction=1.0,
            damping_gain=1.0,
            smoothing_alpha=0.0,
            impedance_kp=0.0,
            use_impedance_target=False,
        ),
        # Smoothing only
        StabilizationCandidate(
            name="smoothing_light",
            k_roll=20.0,
            k_roll_rate=2.0,
            k_pitch=5.0,
            k_pitch_rate=0.5,
            wbc_authority_fraction=1.0,
            damping_gain=0.0,
            smoothing_alpha=0.3,
            impedance_kp=0.0,
            use_impedance_target=False,
        ),
        StabilizationCandidate(
            name="smoothing_moderate",
            k_roll=20.0,
            k_roll_rate=2.0,
            k_pitch=5.0,
            k_pitch_rate=0.5,
            wbc_authority_fraction=1.0,
            damping_gain=0.0,
            smoothing_alpha=0.5,
            impedance_kp=0.0,
            use_impedance_target=False,
        ),
        # Impedance only
        StabilizationCandidate(
            name="impedance_weak",
            k_roll=20.0,
            k_roll_rate=2.0,
            k_pitch=5.0,
            k_pitch_rate=0.5,
            wbc_authority_fraction=1.0,
            damping_gain=0.0,
            smoothing_alpha=0.0,
            impedance_kp=2.0,
            use_impedance_target=True,
        ),
        # Damping + Smoothing
        StabilizationCandidate(
            name="damping_smoothing",
            k_roll=20.0,
            k_roll_rate=2.0,
            k_pitch=5.0,
            k_pitch_rate=0.5,
            wbc_authority_fraction=1.0,
            damping_gain=1.0,
            smoothing_alpha=0.3,
            impedance_kp=0.0,
            use_impedance_target=False,
        ),
        # Damping + Impedance
        StabilizationCandidate(
            name="damping_impedance",
            k_roll=20.0,
            k_roll_rate=2.0,
            k_pitch=5.0,
            k_pitch_rate=0.5,
            wbc_authority_fraction=1.0,
            damping_gain=1.0,
            smoothing_alpha=0.0,
            impedance_kp=2.0,
            use_impedance_target=True,
        ),
        # Full stabilization
        StabilizationCandidate(
            name="full_stabilized",
            k_roll=20.0,
            k_roll_rate=2.0,
            k_pitch=5.0,
            k_pitch_rate=0.5,
            wbc_authority_fraction=1.0,
            damping_gain=1.0,
            smoothing_alpha=0.3,
            impedance_kp=2.0,
            use_impedance_target=True,
        ),
    ]

    # Load environment config
    config_path = project_root / "configs" / "training" / "balance.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Configure for h=0.60 fixed height with torque_first_wbc mode
    config["task"]["height_command_mode"] = "fixed"
    config["task"]["fixed_height"] = 0.60

    results = []

    for candidate in candidates:
        print(f"\nEvaluating: {candidate.name}")
        print(f"  WBC: k_roll={candidate.k_roll}, k_pitch={candidate.k_pitch}")
        print(f"  Stabilization: damping={candidate.damping_gain}, smoothing={candidate.smoothing_alpha}, impedance={candidate.impedance_kp}")

        # Update config for this candidate
        config["low_level_control"] = {
            "mode": "torque_first_wbc",
            "torque_control": {
                "enabled": True,
                "max_ctrl_fraction": candidate.wbc_authority_fraction,
            },
        }

        # Create environment
        env = BalanceEnv(config)

        # Create WBC controller
        controller = WBCBalanceController(
            k_roll=candidate.k_roll,
            k_roll_rate=candidate.k_roll_rate,
            k_pitch=candidate.k_pitch,
            k_pitch_rate=candidate.k_pitch_rate,
            allow_wheel_torque=False,
            wheel_roll_gain=0.0,
        )

        # Run 5 episodes
        NUM_EPISODES = 5
        MAX_STEPS = 60

        episode_survivals = []
        episode_falls = []
        episode_pitch_rms = []
        episode_roll_rms = []
        episode_sat_rates = []
        episode_torque_rms = []
        episode_wbc_authority = []
        episode_damping_authority = []
        episode_impedance_authority = []

        for ep in range(NUM_EPISODES):
            rng = jax.random.PRNGKey(42 + ep)
            state = env.reset(rng)
            obs = state.obs

            controller.reset()

            survival_steps = 0
            fell = False
            pitch_history = []
            roll_history = []
            torque_history = []
            sat_count = 0
            wbc_torque_sum = 0.0
            damping_torque_sum = 0.0
            impedance_torque_sum = 0.0

            # Get impedance target if needed
            impedance_target = compute_impedance_target(np.array(obs)) if candidate.use_impedance_target else None

            prev_ctrl = None

            for step in range(MAX_STEPS):
                obs_np = np.array(obs)

                # Compute WBC torque
                wbc_action = controller.compute_torque(obs_np)

                # Compute stabilization components for authority tracking
                joint_vel = np.array(state.mjx_data.qvel[6:16])
                damping_torque = -candidate.damping_gain * joint_vel if candidate.damping_gain > 0.0 else np.zeros(10)

                impedance_torque = np.zeros(10)
                if candidate.impedance_kp > 0.0 and impedance_target is not None:
                    joint_pos = np.array(state.mjx_data.qpos[7:17])
                    pos_error = impedance_target - joint_pos
                    impedance_torque = candidate.impedance_kp * pos_error

                # Track authority (RMS torque magnitude)
                wbc_torque_sum += np.sqrt(np.mean(wbc_action**2))
                damping_torque_sum += np.sqrt(np.mean(damping_torque**2))
                impedance_torque_sum += np.sqrt(np.mean(impedance_torque**2))

                # Set torque_residual_action and stabilization params in state.info
                state = state._replace(
                    info={
                        **state.info,
                        "torque_residual_action": jnp.array(wbc_action),
                        "stabilization_damping_gain": jnp.float32(candidate.damping_gain),
                        "stabilization_smoothing_alpha": jnp.float32(candidate.smoothing_alpha),
                        "stabilization_impedance_kp": jnp.float32(candidate.impedance_kp),
                        "stabilization_impedance_target": jnp.array(impedance_target) if impedance_target is not None else jnp.zeros(10),
                        "stabilization_prev_ctrl": prev_ctrl if prev_ctrl is not None else jnp.zeros(10),
                    }
                )

                # Step environment
                state = env.step(state, jnp.array(wbc_action))
                obs = state.obs
                done = state.done

                # Store prev_ctrl for next iteration
                prev_ctrl = state.info["last_actuator_ctrl"]

                survival_steps += 1

                # Extract state
                gravity_body = np.array(obs[0:3])
                roll = float(np.arcsin(np.clip(gravity_body[1], -1.0, 1.0)))
                pitch = float(np.arcsin(np.clip(-gravity_body[0], -1.0, 1.0)))
                pitch_history.append(np.rad2deg(pitch))
                roll_history.append(np.rad2deg(roll))

                # Track torque and saturation
                ctrl = np.array(state.mjx_data.ctrl)
                torque_history.append(np.abs(ctrl))
                if np.any(np.abs(wbc_action) > 0.99):
                    sat_count += 1

                if bool(done):
                    fell = True
                    break

            # Compute authority percentages
            total_authority = wbc_torque_sum + damping_torque_sum + impedance_torque_sum
            wbc_authority_pct = (wbc_torque_sum / total_authority * 100) if total_authority > 0 else 100.0
            damping_authority_pct = (damping_torque_sum / total_authority * 100) if total_authority > 0 else 0.0
            impedance_authority_pct = (impedance_torque_sum / total_authority * 100) if total_authority > 0 else 0.0

            episode_survivals.append(survival_steps * env.CONTROL_DT)
            episode_falls.append(1.0 if fell else 0.0)
            episode_pitch_rms.append(float(np.sqrt(np.mean(np.array(pitch_history)**2))))
            episode_roll_rms.append(float(np.sqrt(np.mean(np.array(roll_history)**2))))
            episode_sat_rates.append(sat_count / survival_steps if survival_steps > 0 else 0.0)
            episode_torque_rms.append(float(np.sqrt(np.mean(np.array(torque_history)**2))))
            episode_wbc_authority.append(wbc_authority_pct)
            episode_damping_authority.append(damping_authority_pct)
            episode_impedance_authority.append(impedance_authority_pct)

        # Aggregate results
        mean_survival = float(np.mean(episode_survivals))
        mean_fall_rate = float(np.mean(episode_falls))
        mean_pitch_rms = float(np.mean(episode_pitch_rms))
        mean_roll_rms = float(np.mean(episode_roll_rms))
        mean_sat_rate = float(np.mean(episode_sat_rates))
        mean_torque_rms = float(np.mean(episode_torque_rms))
        mean_wbc_authority = float(np.mean(episode_wbc_authority))
        mean_damping_authority = float(np.mean(episode_damping_authority))
        mean_impedance_authority = float(np.mean(episode_impedance_authority))

        result = {
            "candidate": candidate.name,
            "damping_gain": candidate.damping_gain,
            "smoothing_alpha": candidate.smoothing_alpha,
            "impedance_kp": candidate.impedance_kp,
            "mean_survival_s": mean_survival,
            "mean_fall_rate": mean_fall_rate,
            "mean_pitch_rms_deg": mean_pitch_rms,
            "mean_roll_rms_deg": mean_roll_rms,
            "mean_saturation_rate": mean_sat_rate,
            "mean_torque_rms_Nm": mean_torque_rms,
            "wbc_authority_pct": mean_wbc_authority,
            "damping_authority_pct": mean_damping_authority,
            "impedance_authority_pct": mean_impedance_authority,
        }
        results.append(result)

        print(f"  Results: survival={mean_survival:.2f}s, fall_rate={mean_fall_rate:.2f}, sat_rate={mean_sat_rate:.2%}")
        print(f"  Authority: WBC={mean_wbc_authority:.1f}%, damping={mean_damping_authority:.1f}%, impedance={mean_impedance_authority:.1f}%")

    # Save results
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "stabilization_ablation_results.csv", index=False)

    # Find best candidate
    best_idx = df["mean_survival_s"].idxmax()
    best = results[best_idx]

    # Compare against baselines
    baseline_step518c = 0.86
    baseline_step522 = 0.68

    summary = {
        "evaluation_complete": True,
        "architecture": "torque_first_stabilized_wbc",
        "best_candidate": best["candidate"],
        "best_survival_s": best["mean_survival_s"],
        "best_wbc_authority_pct": best["wbc_authority_pct"],
        "best_saturation_rate": best["mean_saturation_rate"],
        "step5_18c_baseline_survival_s": baseline_step518c,
        "step5_22_baseline_survival_s": baseline_step522,
        "improvement_vs_step522_pct": (best["mean_survival_s"] - baseline_step522) / baseline_step522 * 100,
        "improvement_vs_step518c_pct": (best["mean_survival_s"] - baseline_step518c) / baseline_step518c * 100,
        "candidates_tested": len(candidates),
        "wbc_dominance_maintained": best["wbc_authority_pct"] > 70.0,
    }

    with open(OUTPUT_DIR / "candidate_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Evaluation Complete ===")
    print(f"Best candidate: {best['candidate']}")
    print(f"  Survival: {best['mean_survival_s']:.2f}s")
    print(f"  vs Step 5.22 baseline: {summary['improvement_vs_step522_pct']:+.1f}%")
    print(f"  vs Step 5.18c baseline: {summary['improvement_vs_step518c_pct']:+.1f}%")
    print(f"  WBC authority: {best['wbc_authority_pct']:.1f}%")
    print(f"  WBC dominance maintained: {summary['wbc_dominance_maintained']}")
    print(f"\nResults saved to {OUTPUT_DIR}")

    return results, summary


def main():
    """Run complete Step 5.24 evaluation."""
    print("="*80)
    print("Phase B.9 Step 5.24 — Torque-First Stabilized WBC")
    print("="*80)

    # Run stabilization ablation
    results, summary = run_stabilization_ablation()

    print("\n" + "="*80)
    print("PHASE B.9 STEP 5.24 EVALUATION COMPLETE")
    print("="*80)
    print(f"\nBest result: {summary['best_candidate']}")
    print(f"  Survival: {summary['best_survival_s']:.2f}s")
    print(f"  WBC authority: {summary['best_wbc_authority_pct']:.1f}%")
    print(f"  vs Step 5.22: {summary['improvement_vs_step522_pct']:+.1f}%")
    print(f"  vs Step 5.18c: {summary['improvement_vs_step518c_pct']:+.1f}%")
    print(f"\nWBC dominance maintained: {summary['wbc_dominance_maintained']}")
    print("="*80)


if __name__ == "__main__":
    main()
