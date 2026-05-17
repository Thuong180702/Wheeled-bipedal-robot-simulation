#!/usr/bin/env python3
"""
Phase B.9 Step 5.18c — Motor-Torque Gain Scaling and Saturation Calibration

Diagnose Step 5.18b saturation (0.9375 rate) and calibrate torque gains.

Root cause from Phase 1/2:
- PID controller saturates at ctrlrange limits (15-30 Nm)
- Torque residuals remain small (~1 Nm) due to weak gains
- Magnitude ratio: PID is 30x larger than torque residual

Phase 3: Conservative response validation with larger torque gains.
Phase 4: Small h=0.60 survival evaluation for non-saturating candidates.
Phase 5: Full validation if candidates pass gate.
"""

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from wheeled_biped.envs.balance_env import BalanceEnv
from wheeled_biped.sim.low_level_control import hybrid_pid_plus_torque_control

OUTPUT_DIR = project_root / "outputs" / "phase_b9_step5_18c_torque_gain_saturation_calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Action indices
L_HIP_ROLL = 0
R_HIP_ROLL = 5
L_HIP_PITCH = 2
R_HIP_PITCH = 7
L_KNEE = 3
R_KNEE = 8
L_WHEEL = 4
R_WHEEL = 9
ACTION_DIM = 10

# Baseline reset-fixed Step 5 metrics (post-Step 5.13)
BASELINE_H060 = {
    "survival_s": 0.52,
    "fall_rate": 1.0,
    "pitch_rms_deg": 0.8745,
    "roll_rms_deg": 16.493,
    "action_saturation": 0.0,
}


class TorqueGainCandidate:
    """Torque gain candidate for calibration."""

    def __init__(
        self,
        name: str,
        k_roll: float,
        k_roll_rate: float,
        k_pitch: float,
        k_pitch_rate: float,
        max_ctrl_fraction: float,
        allow_wheel_torque: bool = False,
        wheel_roll_gain: float = 0.0,
    ):
        self.name = name
        self.k_roll = k_roll
        self.k_roll_rate = k_roll_rate
        self.k_pitch = k_pitch
        self.k_pitch_rate = k_pitch_rate
        self.max_ctrl_fraction = max_ctrl_fraction
        self.allow_wheel_torque = allow_wheel_torque
        self.wheel_roll_gain = wheel_roll_gain


def compute_torque_residual_action(obs: np.ndarray, candidate: TorqueGainCandidate) -> np.ndarray:
    """Compute torque residual action from observation."""
    gravity_body = obs[0:3]
    roll = float(np.arcsin(np.clip(gravity_body[1], -1.0, 1.0)))
    pitch = float(np.arcsin(np.clip(-gravity_body[0], -1.0, 1.0)))

    angular_vel = obs[6:9]
    pitch_rate = float(angular_vel[0])
    roll_rate = float(angular_vel[1])

    roll_cmd = -candidate.k_roll * roll - candidate.k_roll_rate * roll_rate
    pitch_cmd = -candidate.k_pitch * pitch - candidate.k_pitch_rate * pitch_rate
    wheel_cmd = -candidate.wheel_roll_gain * roll_rate if candidate.allow_wheel_torque else 0.0

    residual = np.zeros(ACTION_DIM, dtype=np.float32)
    residual[L_HIP_ROLL] = np.clip(roll_cmd, -1.0, 1.0)
    residual[R_HIP_ROLL] = np.clip(-roll_cmd, -1.0, 1.0)
    residual[L_HIP_PITCH] = np.clip(pitch_cmd, -1.0, 1.0)
    residual[R_HIP_PITCH] = np.clip(pitch_cmd, -1.0, 1.0)
    residual[L_KNEE] = np.clip(-0.5 * pitch_cmd, -1.0, 1.0)
    residual[R_KNEE] = np.clip(-0.5 * pitch_cmd, -1.0, 1.0)

    if candidate.allow_wheel_torque:
        residual[L_WHEEL] = np.clip(wheel_cmd, -1.0, 1.0)
        residual[R_WHEEL] = np.clip(wheel_cmd, -1.0, 1.0)

    return residual


def run_response_validation():
    """Phase 3: Test torque gains with ±2 deg roll perturbations."""
    print("\n=== Phase 3: Conservative Response Validation ===\n")

    # Test larger torque gains based on Phase 1/2 analysis
    # PID outputs 15-30 Nm, so torque residuals need to be 5-15 Nm to compete
    # For hip_roll: physical_torque = normalized * max_ctrl_fraction * 15
    # To get 5 Nm: normalized = 5 / (0.3 * 15) = 1.11 (saturates at 1.0)
    # To get 10 Nm: normalized = 10 / (0.5 * 15) = 1.33 (saturates at 1.0)
    # So we need k_roll large enough that small roll errors produce normalized ~0.5-1.0
    # For 2 deg (0.035 rad) roll: k_roll * 0.035 = 0.5 → k_roll = 14.3

    candidates = [
        TorqueGainCandidate("conservative_k10", k_roll=10.0, k_roll_rate=1.0, k_pitch=3.0, k_pitch_rate=0.3, max_ctrl_fraction=0.30),
        TorqueGainCandidate("moderate_k15", k_roll=15.0, k_roll_rate=1.5, k_pitch=4.0, k_pitch_rate=0.4, max_ctrl_fraction=0.40),
        TorqueGainCandidate("strong_k20", k_roll=20.0, k_roll_rate=2.0, k_pitch=5.0, k_pitch_rate=0.5, max_ctrl_fraction=0.50),
        TorqueGainCandidate("aggressive_k30", k_roll=30.0, k_roll_rate=3.0, k_pitch=7.0, k_pitch_rate=0.7, max_ctrl_fraction=0.60),
        TorqueGainCandidate("very_strong_k40", k_roll=40.0, k_roll_rate=4.0, k_pitch=10.0, k_pitch_rate=1.0, max_ctrl_fraction=0.70),
    ]

    # Load environment to get ctrl limits
    import yaml
    config_path = project_root / "configs" / "training" / "balance.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    env = BalanceEnv(config)

    ctrl_min = env._ctrl_min
    ctrl_max = env._ctrl_max

    results = []

    for candidate in candidates:
        print(f"\nTesting {candidate.name}:")
        print(f"  k_roll={candidate.k_roll}, k_pitch={candidate.k_pitch}, max_ctrl_fraction={candidate.max_ctrl_fraction}")

        # Test ±2 deg roll perturbations
        for roll_deg in [-2.0, 2.0]:
            roll_rad = np.deg2rad(roll_deg)

            # Create synthetic observation with roll perturbation
            obs = np.zeros(42, dtype=np.float32)
            obs[0] = 0.0  # gravity_x (pitch)
            obs[1] = np.sin(roll_rad)  # gravity_y (roll)
            obs[2] = np.cos(roll_rad)  # gravity_z
            obs[6:9] = 0.0  # angular velocity

            # Compute torque residual
            residual_normalized = compute_torque_residual_action(obs, candidate)

            # Convert to physical torque using hybrid control
            pid_ctrl = np.zeros(ACTION_DIM, dtype=np.float32)  # assume zero PID for response test

            final_ctrl, residual_ctrl = hybrid_pid_plus_torque_control(
                jnp.array(pid_ctrl),
                jnp.array(residual_normalized),
                jnp.array(ctrl_min),
                jnp.array(ctrl_max),
                candidate.max_ctrl_fraction,
                None,
            )

            residual_ctrl_np = np.array(residual_ctrl)
            final_ctrl_np = np.array(final_ctrl)

            # Check saturation
            saturated = np.any(np.abs(final_ctrl_np - ctrl_min) < 1e-6) or np.any(np.abs(final_ctrl_np - ctrl_max) < 1e-6)

            hip_roll_torque = residual_ctrl_np[L_HIP_ROLL]
            hip_pitch_torque = residual_ctrl_np[L_HIP_PITCH]

            results.append({
                "candidate": candidate.name,
                "k_roll": candidate.k_roll,
                "k_pitch": candidate.k_pitch,
                "max_ctrl_fraction": candidate.max_ctrl_fraction,
                "roll_perturbation_deg": roll_deg,
                "residual_normalized_hip_roll": float(residual_normalized[L_HIP_ROLL]),
                "residual_ctrl_hip_roll_Nm": float(hip_roll_torque),
                "residual_ctrl_hip_pitch_Nm": float(hip_pitch_torque),
                "saturated": bool(saturated),
            })

            print(f"    roll={roll_deg:+.1f}°: hip_roll_torque={hip_roll_torque:+.2f} Nm, saturated={saturated}")

    # Save results
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "response_validation.csv", index=False)

    # Summary
    summary = {
        "phase": "response_validation",
        "candidates_tested": len(candidates),
        "perturbations_per_candidate": 2,
        "total_tests": len(results),
        "non_saturating_candidates": df[~df["saturated"]]["candidate"].unique().tolist(),
        "mean_hip_roll_torque_by_candidate": df.groupby("candidate")["residual_ctrl_hip_roll_Nm"].mean().abs().to_dict(),
    }

    with open(OUTPUT_DIR / "response_validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResponse validation complete: {len(results)} tests")
    print(f"  Non-saturating candidates: {summary['non_saturating_candidates']}")

    return candidates, summary


def run_h060_survival_evaluation(candidates: list[TorqueGainCandidate]):
    """Phase 4: Small h=0.60 survival evaluation."""
    print("\n=== Phase 4: Small Survival Evaluation (h=0.60) ===\n")

    import yaml
    config_path = project_root / "configs" / "training" / "balance.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Override for h=0.60 fixed height
    config["task"]["height_command_mode"] = "fixed"
    config["task"]["fixed_height"] = 0.60

    env = BalanceEnv(config)

    NUM_EPISODES = 5
    MAX_STEPS = 60

    results = []

    for candidate in candidates:
        print(f"\nEvaluating {candidate.name} at h=0.60:")
        print(f"  k_roll={candidate.k_roll}, k_pitch={candidate.k_pitch}, max_ctrl_fraction={candidate.max_ctrl_fraction}")

        episode_survivals = []
        episode_falls = []
        episode_pitch_rms = []
        episode_roll_rms = []
        episode_sat_rates = []
        episode_torque_residual_abs = []

        for ep in range(NUM_EPISODES):
            rng = jax.random.PRNGKey(42 + ep)
            state = env.reset(rng)
            obs = state.obs

            survival_steps = 0
            fell = False
            pitch_history = []
            roll_history = []
            sat_count = 0
            torque_residual_sum = 0.0

            for step in range(MAX_STEPS):
                # Compute torque residual action
                obs_np = np.array(obs)
                residual_action = compute_torque_residual_action(obs_np, candidate)

                # Apply action (zero base action for pure torque control test)
                action = jnp.array(residual_action)

                state = env.step(state, action)
                obs = state.obs
                done = state.done

                survival_steps += 1

                # Track metrics
                gravity_body = np.array(obs[0:3])
                roll = float(np.arcsin(np.clip(gravity_body[1], -1.0, 1.0)))
                pitch = float(np.arcsin(np.clip(-gravity_body[0], -1.0, 1.0)))
                pitch_history.append(np.rad2deg(pitch))
                roll_history.append(np.rad2deg(roll))

                # Check saturation (simplified - just check if action is at limits)
                if np.any(np.abs(residual_action) > 0.99):
                    sat_count += 1

                torque_residual_sum += np.abs(residual_action).mean()

                if bool(done):
                    fell = True
                    break

            episode_survivals.append(survival_steps * env.CONTROL_DT)
            episode_falls.append(1.0 if fell else 0.0)
            episode_pitch_rms.append(float(np.sqrt(np.mean(np.array(pitch_history)**2))))
            episode_roll_rms.append(float(np.sqrt(np.mean(np.array(roll_history)**2))))
            episode_sat_rates.append(sat_count / survival_steps if survival_steps > 0 else 0.0)
            episode_torque_residual_abs.append(torque_residual_sum / survival_steps if survival_steps > 0 else 0.0)

        mean_survival = float(np.mean(episode_survivals))
        mean_fall_rate = float(np.mean(episode_falls))
        mean_pitch_rms = float(np.mean(episode_pitch_rms))
        mean_roll_rms = float(np.mean(episode_roll_rms))
        mean_sat_rate = float(np.mean(episode_sat_rates))
        mean_torque_residual = float(np.mean(episode_torque_residual_abs))

        results.append({
            "candidate": candidate.name,
            "k_roll": candidate.k_roll,
            "k_pitch": candidate.k_pitch,
            "max_ctrl_fraction": candidate.max_ctrl_fraction,
            "mean_survival_s": mean_survival,
            "fall_rate": mean_fall_rate,
            "pitch_rms_deg": mean_pitch_rms,
            "roll_rms_deg": mean_roll_rms,
            "action_saturation_rate": mean_sat_rate,
            "mean_torque_residual_abs": mean_torque_residual,
        })

        print(f"    survival={mean_survival:.2f}s, fall_rate={mean_fall_rate:.2f}, roll_rms={mean_roll_rms:.1f}°")

    # Save results
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "h060_survival_results.csv", index=False)

    # Determine which candidates beat baseline
    baseline_survival = BASELINE_H060["survival_s"]
    kept_candidates = df[df["mean_survival_s"] > baseline_survival]["candidate"].tolist()

    summary = {
        "phase": "h060_survival_evaluation",
        "baseline_h060_survival_s": baseline_survival,
        "candidates_tested": len(candidates),
        "episodes_per_candidate": NUM_EPISODES,
        "max_steps": MAX_STEPS,
        "kept_candidates": kept_candidates,
        "best_candidate": df.loc[df["mean_survival_s"].idxmax()].to_dict() if len(df) > 0 else None,
    }

    with open(OUTPUT_DIR / "h060_survival_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPhase 4 complete: {len(results)} candidates tested")
    print(f"  Baseline h=0.60 survival: {baseline_survival:.2f}s")
    print(f"  Candidates beating baseline: {kept_candidates}")

    return results, summary


def run_full_validation(kept_candidates: list[TorqueGainCandidate]):
    """Phase 5: Full validation across heights 0.65-0.40."""
    print("\n=== Phase 5: Full Validation (heights 0.65-0.40) ===\n")

    import yaml
    import pandas as pd

    config_path = project_root / "configs" / "training" / "balance.yaml"
    full_results_path = OUTPUT_DIR / "full_validation_results.csv"

    HEIGHTS = [0.65, 0.60, 0.55, 0.50, 0.45, 0.40]
    NUM_EPISODES = 5
    MAX_STEPS = 60

    if full_results_path.exists():
        existing_df = pd.read_csv(full_results_path)
        results = existing_df.to_dict("records")
    else:
        results = []

    completed_pairs = {(row["candidate"], float(row["height"])) for row in results}

    for candidate in kept_candidates:
        print(f"\nValidating {candidate.name} across all heights:")

        for height in HEIGHTS:
            if (candidate.name, float(height)) in completed_pairs:
                print(f"  h={height:.2f}: already completed, skipping")
                continue

            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            config["task"]["height_command_mode"] = "fixed"
            config["task"]["fixed_height"] = height

            env = BalanceEnv(config)

            episode_survivals = []
            episode_falls = []
            episode_pitch_rms = []
            episode_roll_rms = []
            episode_sat_rates = []

            for ep in range(NUM_EPISODES):
                rng = jax.random.PRNGKey(42 + ep)
                state = env.reset(rng)
                obs = state.obs

                survival_steps = 0
                fell = False
                pitch_history = []
                roll_history = []
                sat_count = 0

                for step in range(MAX_STEPS):
                    obs_np = np.array(obs)
                    residual_action = compute_torque_residual_action(obs_np, candidate)
                    action = jnp.array(residual_action)

                    state = env.step(state, action)
                    obs = state.obs
                    done = state.done

                    survival_steps += 1

                    gravity_body = np.array(obs[0:3])
                    roll = float(np.arcsin(np.clip(gravity_body[1], -1.0, 1.0)))
                    pitch = float(np.arcsin(np.clip(-gravity_body[0], -1.0, 1.0)))
                    pitch_history.append(np.rad2deg(pitch))
                    roll_history.append(np.rad2deg(roll))

                    if np.any(np.abs(residual_action) > 0.99):
                        sat_count += 1

                    if bool(done):
                        fell = True
                        break

                episode_survivals.append(survival_steps * env.CONTROL_DT)
                episode_falls.append(1.0 if fell else 0.0)
                episode_pitch_rms.append(float(np.sqrt(np.mean(np.array(pitch_history)**2))))
                episode_roll_rms.append(float(np.sqrt(np.mean(np.array(roll_history)**2))))
                episode_sat_rates.append(sat_count / survival_steps if survival_steps > 0 else 0.0)

            mean_survival = float(np.mean(episode_survivals))
            mean_fall_rate = float(np.mean(episode_falls))
            mean_pitch_rms = float(np.mean(episode_pitch_rms))
            mean_roll_rms = float(np.mean(episode_roll_rms))
            mean_sat_rate = float(np.mean(episode_sat_rates))

            row = {
                "candidate": candidate.name,
                "height": height,
                "k_roll": candidate.k_roll,
                "k_pitch": candidate.k_pitch,
                "max_ctrl_fraction": candidate.max_ctrl_fraction,
                "mean_survival_s": mean_survival,
                "fall_rate": mean_fall_rate,
                "pitch_rms_deg": mean_pitch_rms,
                "roll_rms_deg": mean_roll_rms,
                "action_saturation_rate": mean_sat_rate,
            }
            results.append(row)

            pd.DataFrame(results).to_csv(full_results_path, index=False)
            completed_pairs.add((candidate.name, float(height)))

            print(f"  h={height:.2f}: survival={mean_survival:.2f}s, fall_rate={mean_fall_rate:.2f}")

    df = pd.DataFrame(results)

    expected_rows = len(kept_candidates) * len(HEIGHTS)
    if len(df) < expected_rows:
        summary = {
            "phase": "full_validation",
            "heights": HEIGHTS,
            "episodes_per_height": NUM_EPISODES,
            "candidates_tested": len(kept_candidates),
            "full_validation_run": False,
            "reason": "incomplete rows due to timeout/resume; rerun script to continue",
            "current_rows": int(len(df)),
            "expected_rows": int(expected_rows),
        }
        with open(OUTPUT_DIR / "full_validation_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return results, summary

    agg = df.groupby("candidate").agg({
        "mean_survival_s": "mean",
        "fall_rate": "mean",
        "pitch_rms_deg": "mean",
        "roll_rms_deg": "mean",
        "action_saturation_rate": "mean",
    }).reset_index()

    best_candidate = agg.loc[agg["mean_survival_s"].idxmax()]

    summary = {
        "phase": "full_validation",
        "heights": HEIGHTS,
        "episodes_per_height": NUM_EPISODES,
        "candidates_tested": len(kept_candidates),
        "full_validation_run": True,
        "best_candidate": {
            "name": best_candidate["candidate"],
            "mean_survival_s": float(best_candidate["mean_survival_s"]),
            "fall_rate": float(best_candidate["fall_rate"]),
            "pitch_rms_deg": float(best_candidate["pitch_rms_deg"]),
            "roll_rms_deg": float(best_candidate["roll_rms_deg"]),
            "action_saturation_rate": float(best_candidate["action_saturation_rate"]),
        },
    }

    with open(OUTPUT_DIR / "full_validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPhase 5 complete: {len(kept_candidates)} candidates validated")
    print(f"  Best candidate: {best_candidate['candidate']}")
    print(f"    All-height mean survival: {best_candidate['mean_survival_s']:.2f}s")
    print(f"    All-height fall rate: {best_candidate['fall_rate']:.2f}")

    return results, summary


def main():
    """Run Step 5.18c calibration."""
    print("Phase B.9 Step 5.18c — Motor-Torque Gain Scaling and Saturation Calibration")
    print("=" * 80)

    # Phase 3: Response validation
    candidates, response_summary = run_response_validation()

    # Phase 4: Small survival evaluation (h=0.60)
    h060_results, h060_summary = run_h060_survival_evaluation(candidates)

    # Phase 5: Full validation if candidates passed gate
    if h060_summary["kept_candidates"]:
        ranked_h060 = sorted(
            h060_results,
            key=lambda x: (-x["mean_survival_s"], x["fall_rate"], x["roll_rms_deg"]),
        )
        top2_names = [row["candidate"] for row in ranked_h060[:2]]
        kept_candidate_objs = [c for c in candidates if c.name in top2_names]
        full_results, full_summary = run_full_validation(kept_candidate_objs)
    else:
        print("\nNo candidates passed h=0.60 gate. Skipping Phase 5.")
        full_summary = None

    # Final summary
    print("\n" + "=" * 80)
    print("Step 5.18c Phases 3-5 complete.")
    print(f"Output directory: {OUTPUT_DIR}")

    if full_summary:
        print(f"\nBest candidate: {full_summary['best_candidate']['name']}")
        print(f"  All-height mean survival: {full_summary['best_candidate']['mean_survival_s']:.2f}s")
        print(f"  All-height fall rate: {full_summary['best_candidate']['fall_rate']:.2f}")
    else:
        print("\nNo candidates beat baseline. Torque gain calibration insufficient.")


if __name__ == "__main__":
    main()
