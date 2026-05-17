#!/usr/bin/env python3
"""
Phase B.9 Step 5.20b — Canonical Evaluation Rebuild

CLEAN REBUILD of Step 5.20 evaluation using Step 5.18c's proven infrastructure.

Core hypothesis:
The current controller is over-stiff and fighting natural balancing dynamics.

Approach:
- Copy Step 5.18c's exact rollout/evaluation flow
- Replace TorqueGainCandidate with SoftModeCandidate
- Keep WBC gains fixed at strong_k20 values
- Vary only soft mode parameters (stiffness, deadband)
- Add instrumentation to verify soft parameters propagate
- MANDATORY: Verify baseline reproduces Step 5.18c (~0.86s) before testing variants

Previous attempts failed because:
- Baseline didn't reproduce Step 5.18c (0.38s vs 0.86s)
- All candidates produced identical results
- Soft mode parameters had zero observable effect

This rebuild uses Step 5.18c's proven control pipeline integration to ensure:
- Baseline reproduction within 10-15% tolerance
- Soft parameters actually modify runtime behavior
- Different candidates show different behavior
"""

import json
import sys
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

from scripts.phase_b9_step5_lqr_gain_strengthening import (
    apply_balanced_root_init,
    create_tuned_controller,
    load_balanced_init_table,
    rpy_to_quat,
)
from wheeled_biped.controllers.dual_rate_balance_controller import DualRateConfig
from wheeled_biped.envs.balance_env import BalanceEnv
from wheeled_biped.sim.low_level_control import hybrid_pid_plus_torque_control
from wheeled_biped.utils.config import get_model_path

OUTPUT_DIR = project_root / "outputs" / "phase_b9_step5_20b_canonical_eval_rebuild"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_LQR_PATH = project_root / "outputs" / "phase_b9_lqr_gain_strengthening" / "best_lqr_config.yaml"
CONTROLLER_CONFIG_PATH = project_root / "configs" / "controllers" / "dual_rate_balance_controller_b9.yaml"

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

# Step 5.18c baseline (strong_k20 at h=0.60)
STEP_5_18C_BASELINE = {
    "survival_s": 0.86,
    "fall_rate": 0.80,
    "pitch_rms_deg": 1.0,
    "roll_rms_deg": 15.9,
    "saturation_rate": 0.9375,
}

# Baseline reproduction tolerance
BASELINE_TOLERANCE = 0.15  # 15% tolerance


class SoftModeCandidate:
    """Soft mode candidate for stiffness reduction evaluation.

    Keeps WBC torque gains fixed at strong_k20 values.
    Varies only soft mode parameters (stiffness, deadband).
    """

    def __init__(
        self,
        name: str,
        # WBC torque gains (fixed at strong_k20 values)
        k_roll: float = 20.0,
        k_roll_rate: float = 2.0,
        k_pitch: float = 5.0,
        k_pitch_rate: float = 0.5,
        max_ctrl_fraction: float = 0.5,
        allow_wheel_torque: bool = False,
        wheel_roll_gain: float = 0.0,
        # Soft mode parameters (varied across candidates)
        stiffness_reduction: float = 1.0,
        deadband_deg: float = 0.0,
        posture_restore_delay_s: float = 0.0,
        balance_authority_boost: float = 1.0,
        allow_torso_lean: bool = False,
        allow_temporary_asymmetry: bool = False,
        max_torso_lean_deg: float = 5.0,
        max_wheel_offset_m: float = 0.05,
    ):
        self.name = name
        # WBC gains
        self.k_roll = k_roll
        self.k_roll_rate = k_roll_rate
        self.k_pitch = k_pitch
        self.k_pitch_rate = k_pitch_rate
        self.max_ctrl_fraction = max_ctrl_fraction
        self.allow_wheel_torque = allow_wheel_torque
        self.wheel_roll_gain = wheel_roll_gain
        # Soft mode parameters
        self.stiffness_reduction = stiffness_reduction
        self.deadband_deg = deadband_deg
        self.posture_restore_delay_s = posture_restore_delay_s
        self.balance_authority_boost = balance_authority_boost
        self.allow_torso_lean = allow_torso_lean
        self.allow_temporary_asymmetry = allow_temporary_asymmetry
        self.max_torso_lean_deg = max_torso_lean_deg
        self.max_wheel_offset_m = max_wheel_offset_m


def compute_torque_residual_action(obs: np.ndarray, candidate: SoftModeCandidate) -> np.ndarray:
    """Compute torque residual action from observation.

    Uses WBC gains from candidate (fixed at strong_k20 values).
    """
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


def load_best_lqr_params() -> dict[str, float]:
    """Load best LQR parameters from Step 5."""
    with BEST_LQR_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env_config(height: float = 0.60, episode_length: int = 60) -> dict:
    """Create base environment config (from Step 5.18c)."""
    return {
        "low_level_pid": {"enabled": True, "disable_pid_action_bias": True},
        "domain_randomization": {"enabled": False},
        "sensor_noise": {"enabled": False},
        "task": {"initial_min_height": height, "episode_length": episode_length},
        "termination": {"max_tilt_rad": 0.8, "min_height": 0.3},
    }


def activation_config(base_config: dict, candidate: SoftModeCandidate) -> dict:
    """Add low-level control config to enable hybrid_pid_plus_torque mode."""
    cfg = {
        key: (value.copy() if isinstance(value, dict) else value)
        for key, value in base_config.items()
    }
    cfg["low_level_control"] = {
        "mode": "hybrid_pid_plus_torque",
        "torque_control": {
            "enabled": True,
            "max_ctrl_fraction": candidate.max_ctrl_fraction,
            "allow_leg_torque": True,
            "allow_wheel_torque": candidate.allow_wheel_torque,
            "allow_hip_yaw_torque": False,
        },
    }
    return cfg


def make_controller(model: mujoco.MjModel, candidate: SoftModeCandidate):
    """Create controller with soft mode config merged in."""
    # Load base controller config
    with open(CONTROLLER_CONFIG_PATH) as f:
        full_cfg = yaml.safe_load(f)

    # Merge soft mode config
    full_cfg["soft_dynamic_balance"] = {
        "enabled": candidate.stiffness_reduction < 1.0 or candidate.deadband_deg > 0.0,
        "posture_stiffness_reduction": candidate.stiffness_reduction,
        "posture_deadband_deg": candidate.deadband_deg,
        "posture_restore_delay_s": candidate.posture_restore_delay_s,
        "balance_authority_boost": candidate.balance_authority_boost,
        "allow_torso_lean": candidate.allow_torso_lean,
        "allow_temporary_asymmetry": candidate.allow_temporary_asymmetry,
        "max_torso_lean_deg": candidate.max_torso_lean_deg,
        "max_wheel_offset_m": candidate.max_wheel_offset_m,
    }

    # Save temporary merged config
    temp_config_path = OUTPUT_DIR / f"temp_controller_{candidate.name}.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(full_cfg, f)

    # Create controller with merged config
    merged_config = DualRateConfig.from_yaml(temp_config_path)
    controller = create_tuned_controller(merged_config, load_best_lqr_params(), model)

    # Clean up temp file
    temp_config_path.unlink()

    return controller


def set_height_and_roll(state, env: BalanceEnv, height: float, roll_rad: float, init_table: dict):
    """Initialize state at target height with balanced root (from Step 5.18b)."""
    mjx_data = apply_balanced_root_init(state.mjx_data, height, init_table)
    if abs(roll_rad) > 0.0:
        qpos = mjx_data.qpos.at[3:7].set(jnp.array(rpy_to_quat(roll_rad, 0.0, 0.0), dtype=mjx_data.qpos.dtype))
        mjx_data = mjx_data.replace(qpos=qpos, qvel=jnp.zeros_like(mjx_data.qvel))
    base_obs = env._extract_obs(mjx_data, jnp.zeros(env.num_actions), state.info["noise_rng"])
    height_norm = (height - env.MIN_HEIGHT_CMD) / (env.MAX_HEIGHT_CMD - env.MIN_HEIGHT_CMD)
    current_height_norm = (mjx_data.qpos[2] - env.MIN_HEIGHT_CMD) / (env.MAX_HEIGHT_CMD - env.MIN_HEIGHT_CMD)
    obs = jnp.concatenate([base_obs, jnp.array([height_norm, current_height_norm, 0.0], dtype=base_obs.dtype)])
    info = {
        **state.info,
        "height_command": jnp.array(height, dtype=mjx_data.qpos.dtype),
        "initial_yaw": jnp.array(0.0, dtype=mjx_data.qpos.dtype),
    }
    return state._replace(mjx_data=mjx_data, obs=obs, info=info, prev_action=jnp.zeros(env.num_actions))

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


def run_episode(
    candidate: SoftModeCandidate,
    seed: int,
    height: float = 0.60,
    max_steps: int = 60,
    model: mujoco.MjModel | None = None,
    env: BalanceEnv | None = None,
    controller = None,
    init_table: dict[float, dict[str, float]] | None = None,
) -> dict:
    """Run single episode with soft mode candidate (adapted from Step 5.18b)."""
    if model is None:
        model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    if env is None:
        env = BalanceEnv(activation_config(make_env_config(height, max_steps), candidate))
    if controller is None:
        controller = make_controller(model, candidate)
    if init_table is None:
        init_table = load_balanced_init_table()

    # Reset and initialize at target height
    rng = jax.random.PRNGKey(seed)
    state = env.reset(rng)
    state = set_height_and_roll(state, env, height, 0.0, init_table)
    controller.reset()

    # Episode tracking
    survival_steps = 0
    pitch_history = []
    roll_history = []
    sat_count = 0
    torque_residual_sum = 0.0

    # Soft parameter propagation tracking
    soft_params_active = []

    for step in range(max_steps):
        obs_np = np.array(state.obs)

        # Controller action (with soft mode parameters)
        action = controller.compute_action(obs_np)

        # WBC torque residual
        residual = compute_torque_residual_action(obs_np, candidate)
        state = state._replace(info={**state.info, "torque_residual_action": jnp.array(residual)})

        # Step environment
        state = env.step(state, jnp.array(action))

        survival_steps += 1

        # Extract metrics
        gravity = obs_np[0:3]
        pitch = float(np.arcsin(np.clip(-gravity[0], -1.0, 1.0)))
        roll = float(np.arcsin(np.clip(gravity[1], -1.0, 1.0)))
        pitch_history.append(np.rad2deg(pitch))
        roll_history.append(np.rad2deg(roll))

        # Track saturation
        if "actuator_saturation_flags" in state.info:
            sat_flags = np.array(state.info["actuator_saturation_flags"])
            if np.any(sat_flags):
                sat_count += 1

        # Track torque residual magnitude
        torque_residual_sum += np.abs(residual).mean()

        # INSTRUMENTATION: Verify soft parameters are active
        # Check if controller config has soft mode enabled
        if hasattr(controller, 'config'):
            soft_enabled = getattr(controller.config, 'soft_dynamic_balance_enabled', False)
            soft_params_active.append(soft_enabled)

        if bool(state.done):
            break

    survival_time = survival_steps * env.CONTROL_DT
    fell = bool(state.done)

    # Compute metrics
    pitch_rms = float(np.sqrt(np.mean(np.array(pitch_history)**2)))
    roll_rms = float(np.sqrt(np.mean(np.array(roll_history)**2)))
    saturation_rate = float(sat_count / survival_steps if survival_steps > 0 else 0.0)
    mean_torque_residual = float(torque_residual_sum / survival_steps if survival_steps > 0 else 0.0)

    # INSTRUMENTATION: Check if soft parameters were consistently active
    soft_params_consistent = bool(all(soft_params_active) if soft_params_active else False)

    return {
        "candidate": candidate.name,
        "seed": int(seed),
        "height": float(height),
        "survival_time_s": float(survival_time),
        "fell": bool(fell),
        "pitch_rms_deg": float(pitch_rms),
        "roll_rms_deg": float(roll_rms),
        "saturation_rate": float(saturation_rate),
        "mean_torque_residual_abs": float(mean_torque_residual),
        # Soft mode instrumentation
        "stiffness_reduction": float(candidate.stiffness_reduction),
        "deadband_deg": float(candidate.deadband_deg),
        "soft_params_active": bool(soft_params_consistent),
        "soft_params_checks": int(len(soft_params_active)),
    }


def run_baseline_reproduction_check():
    """Phase 1: Verify baseline reproduces Step 5.18c (~0.86s survival).

    MANDATORY GATE: Must pass before testing soft mode variants.
    """
    print("\n" + "="*80)
    print("Phase 1: Baseline Reproduction Check")
    print("="*80)
    print("\nMandatory gate: Baseline must reproduce Step 5.18c within 15% tolerance")
    print(f"Expected: ~{STEP_5_18C_BASELINE['survival_s']:.2f}s survival")
    print(f"Tolerance: ±{BASELINE_TOLERANCE*100:.0f}%")

    # Baseline candidate (no stiffness reduction, same WBC as strong_k20)
    baseline = SoftModeCandidate(
        name="baseline_strong_k20",
        stiffness_reduction=1.0,
        deadband_deg=0.0,
    )

    model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    env = BalanceEnv(activation_config(make_env_config(0.60, 60), baseline))
    controller = make_controller(model, baseline)
    init_table = load_balanced_init_table()

    NUM_EPISODES = 5
    results = []

    print(f"\nRunning {NUM_EPISODES} episodes at h=0.60...")
    for ep in range(NUM_EPISODES):
        result = run_episode(
            baseline,
            seed=42 + ep,
            height=0.60,
            max_steps=60,
            model=model,
            env=env,
            controller=controller,
            init_table=init_table,
        )
        results.append(result)
        print(f"  Episode {ep+1}: survival={result['survival_time_s']:.2f}s, "
              f"fell={result['fell']}, roll_rms={result['roll_rms_deg']:.1f}°, "
              f"sat_rate={result['saturation_rate']:.2%}")

    # Aggregate results
    mean_survival = float(np.mean([r["survival_time_s"] for r in results]))
    mean_fall_rate = float(np.mean([r["fell"] for r in results]))
    mean_roll_rms = float(np.mean([r["roll_rms_deg"] for r in results]))
    mean_sat_rate = float(np.mean([r["saturation_rate"] for r in results]))

    # Check baseline reproduction
    expected_survival = STEP_5_18C_BASELINE["survival_s"]
    survival_error = abs(mean_survival - expected_survival) / expected_survival
    baseline_passed = survival_error <= BASELINE_TOLERANCE

    summary = {
        "phase": "baseline_reproduction_check",
        "expected_survival_s": expected_survival,
        "actual_survival_s": mean_survival,
        "survival_error_pct": survival_error * 100,
        "tolerance_pct": BASELINE_TOLERANCE * 100,
        "baseline_passed": baseline_passed,
        "mean_fall_rate": mean_fall_rate,
        "mean_roll_rms_deg": mean_roll_rms,
        "mean_saturation_rate": mean_sat_rate,
        "episodes": results,
    }

    # Save results
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "baseline_reproduction_results.csv", index=False)

    with open(OUTPUT_DIR / "baseline_reproduction_validation.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Report
    print(f"\n{'='*80}")
    print("Baseline Reproduction Results")
    print(f"{'='*80}")
    print(f"Expected survival: {expected_survival:.2f}s")
    print(f"Actual survival:   {mean_survival:.2f}s")
    print(f"Error:             {survival_error*100:.1f}% (tolerance: {BASELINE_TOLERANCE*100:.0f}%)")
    print(f"Fall rate:         {mean_fall_rate:.2f}")
    print(f"Roll RMS:          {mean_roll_rms:.1f}°")
    print(f"Saturation rate:   {mean_sat_rate:.2%}")

    if baseline_passed:
        print(f"\n[+] BASELINE PASSED - Proceeding to soft mode evaluation")
    else:
        print(f"\n[-] BASELINE FAILED - STOP")
        print(f"  Baseline must reproduce Step 5.18c within {BASELINE_TOLERANCE*100:.0f}% tolerance")
        print(f"  Current error: {survival_error*100:.1f}%")
        print(f"\n  DO NOT evaluate soft mode variants until baseline passes.")

    return baseline_passed, summary



def run_soft_mode_evaluation():
    """Phase 2: Evaluate soft mode candidates (only after baseline passes).

    Tests stiffness reduction hypothesis with proper instrumentation.
    """
    print("\n" + "="*80)
    print("Phase 2: Soft Mode Evaluation")
    print("="*80)
    print("\nTesting stiffness reduction hypothesis with 4 candidates")

    # Define soft mode candidates (WBC gains fixed at strong_k20)
    candidates = [
        SoftModeCandidate(
            name="baseline",
            stiffness_reduction=1.0,
            deadband_deg=0.0,
        ),
        SoftModeCandidate(
            name="conservative",
            stiffness_reduction=0.7,
            deadband_deg=1.0,
        ),
        SoftModeCandidate(
            name="moderate",
            stiffness_reduction=0.5,
            deadband_deg=2.0,
        ),
        SoftModeCandidate(
            name="aggressive",
            stiffness_reduction=0.3,
            deadband_deg=3.0,
        ),
    ]

    model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    init_table = load_balanced_init_table()

    NUM_EPISODES = 5
    all_results = []

    for candidate in candidates:
        print(f"\n{'='*80}")
        print(f"Evaluating: {candidate.name}")
        print(f"{'='*80}")
        print(f"  Stiffness reduction: {candidate.stiffness_reduction}")
        print(f"  Deadband: {candidate.deadband_deg}°")

        env = BalanceEnv(activation_config(make_env_config(0.60, 60), candidate))
        controller = make_controller(model, candidate)

        candidate_results = []
        for ep in range(NUM_EPISODES):
            result = run_episode(
                candidate,
                seed=42 + ep,
                height=0.60,
                max_steps=60,
                model=model,
                env=env,
                controller=controller,
                init_table=init_table,
            )
            candidate_results.append(result)
            all_results.append(result)

        # Aggregate candidate results
        mean_survival = float(np.mean([r["survival_time_s"] for r in candidate_results]))
        mean_fall_rate = float(np.mean([r["fell"] for r in candidate_results]))
        mean_roll_rms = float(np.mean([r["roll_rms_deg"] for r in candidate_results]))
        mean_sat_rate = float(np.mean([r["saturation_rate"] for r in candidate_results]))
        soft_params_active = all([r["soft_params_active"] for r in candidate_results])

        print(f"\nResults:")
        print(f"  Survival: {mean_survival:.2f}s")
        print(f"  Fall rate: {mean_fall_rate:.2f}")
        print(f"  Roll RMS: {mean_roll_rms:.1f}°")
        print(f"  Saturation rate: {mean_sat_rate:.2%}")
        print(f"  Soft params active: {soft_params_active}")

        # Compare to Step 5.18c baseline
        baseline_survival = STEP_5_18C_BASELINE["survival_s"]
        if mean_survival > baseline_survival:
            improvement = (mean_survival / baseline_survival - 1) * 100
            print(f"  [+] IMPROVEMENT: +{improvement:.1f}% vs Step 5.18c")
        else:
            degradation = (1 - mean_survival / baseline_survival) * 100
            print(f"  [-] DEGRADATION: -{degradation:.1f}% vs Step 5.18c")

    # Save all results
    import pandas as pd
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_DIR / "candidate_results.csv", index=False)

    # Aggregate by candidate
    agg = df.groupby("candidate").agg({
        "survival_time_s": "mean",
        "fell": "mean",
        "roll_rms_deg": "mean",
        "saturation_rate": "mean",
        "stiffness_reduction": "first",
        "deadband_deg": "first",
    }).reset_index()

    # Find best candidate
    best = agg.loc[agg["survival_time_s"].idxmax()]

    # Check if candidates show different behavior
    survival_variance = float(agg["survival_time_s"].std())
    candidates_differ = survival_variance > 0.01  # More than 10ms variance

    summary = {
        "phase": "soft_mode_evaluation",
        "candidates_tested": len(candidates),
        "episodes_per_candidate": NUM_EPISODES,
        "best_candidate": {
            "name": best["candidate"],
            "stiffness_reduction": float(best["stiffness_reduction"]),
            "deadband_deg": float(best["deadband_deg"]),
            "survival_time_s": float(best["survival_time_s"]),
            "fall_rate": float(best["fell"]),
            "roll_rms_deg": float(best["roll_rms_deg"]),
            "saturation_rate": float(best["saturation_rate"]),
        },
        "candidates_show_different_behavior": candidates_differ,
        "survival_time_variance": survival_variance,
        "all_candidates": agg.to_dict(orient="records"),
    }

    with open(OUTPUT_DIR / "candidate_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Report
    print(f"\n{'='*80}")
    print("Soft Mode Evaluation Summary")
    print(f"{'='*80}")
    print(f"\nBest candidate: {best['candidate']}")
    print(f"  Stiffness reduction: {best['stiffness_reduction']}")
    print(f"  Deadband: {best['deadband_deg']}°")
    print(f"  Survival: {best['survival_time_s']:.2f}s")
    print(f"  Fall rate: {best['fell']:.2f}")
    print(f"  Roll RMS: {best['roll_rms_deg']:.1f}°")
    print(f"  Saturation rate: {best['saturation_rate']:.2%}")

    print(f"\nCandidates show different behavior: {candidates_differ}")
    if not candidates_differ:
        print("  WARNING: All candidates produced similar results!")
        print("  This suggests soft mode parameters are NOT affecting behavior.")

    return summary


def main():
    """Main evaluation orchestration."""
    print("="*80)
    print("Phase B.9 Step 5.20b — Canonical Evaluation Rebuild")
    print("="*80)
    print("\nClean rebuild using Step 5.18c's proven infrastructure")
    print("\nHypothesis: Controller is over-stiff and fighting natural dynamics")

    # Phase 1: Baseline reproduction check (MANDATORY GATE)
    baseline_passed, baseline_summary = run_baseline_reproduction_check()

    if not baseline_passed:
        print(f"\n{'='*80}")
        print("EVALUATION STOPPED")
        print(f"{'='*80}")
        print("\nBaseline reproduction failed.")
        print("Soft mode evaluation cannot proceed until baseline passes.")
        print("\nNext steps:")
        print("  1. Debug why baseline doesn't reproduce Step 5.18c")
        print("  2. Fix control pipeline integration")
        print("  3. Re-run evaluation")
        return 1

    # Phase 2: Soft mode evaluation (only if baseline passed)
    soft_summary = run_soft_mode_evaluation()

    # Final summary
    print(f"\n{'='*80}")
    print("Step 5.20b Evaluation Complete")
    print(f"{'='*80}")

    print(f"\nBaseline reproduction: PASSED")
    print(f"  Actual survival: {baseline_summary['actual_survival_s']:.2f}s")
    print(f"  Expected survival: {baseline_summary['expected_survival_s']:.2f}s")
    print(f"  Error: {baseline_summary['survival_error_pct']:.1f}%")

    print(f"\nSoft mode evaluation: COMPLETE")
    print(f"  Best candidate: {soft_summary['best_candidate']['name']}")
    print(f"  Survival: {soft_summary['best_candidate']['survival_time_s']:.2f}s")
    print(f"  Candidates differ: {soft_summary['candidates_show_different_behavior']}")

    # Answer required questions
    print(f"\n{'='*80}")
    print("Required Answers")
    print(f"{'='*80}")

    baseline_survival = baseline_summary['actual_survival_s']
    best_survival = soft_summary['best_candidate']['survival_time_s']
    best_sat_rate = soft_summary['best_candidate']['saturation_rate']

    print(f"\n1. Can Step 5.18c now be faithfully reproduced?")
    print(f"   YES - Baseline: {baseline_survival:.2f}s (expected: {baseline_summary['expected_survival_s']:.2f}s)")

    print(f"\n2. Are soft parameters actually changing runtime behavior?")
    if soft_summary['candidates_show_different_behavior']:
        print(f"   YES - Survival variance: {soft_summary['survival_time_variance']:.3f}s")
    else:
        print(f"   NO - All candidates produced similar results")

    print(f"\n3. Does lower stiffness reduce saturation?")
    baseline_sat = baseline_summary['mean_saturation_rate']
    if best_sat_rate < baseline_sat:
        reduction = (1 - best_sat_rate / baseline_sat) * 100
        print(f"   YES - Best: {best_sat_rate:.2%} vs baseline: {baseline_sat:.2%} ({reduction:.1f}% reduction)")
    else:
        print(f"   NO - Best: {best_sat_rate:.2%} vs baseline: {baseline_sat:.2%}")

    print(f"\n4. Does lower stiffness improve survival?")
    if best_survival > baseline_survival:
        improvement = (best_survival / baseline_survival - 1) * 100
        print(f"   YES - Best: {best_survival:.2f}s vs baseline: {baseline_survival:.2f}s (+{improvement:.1f}%)")
    else:
        print(f"   NO - Best: {best_survival:.2f}s vs baseline: {baseline_survival:.2f}s")

    print(f"\n5. Is the controller still over-constrained?")
    if best_sat_rate > 0.8:
        print(f"   YES - Saturation rate: {best_sat_rate:.2%} (>80%)")
    else:
        print(f"   NO - Saturation rate: {best_sat_rate:.2%} (<80%)")

    print(f"\n6. Is behavior becoming closer to previous successful pure RL?")
    print(f"   INSUFFICIENT DATA - Would need pure RL baseline comparison")

    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"\nStep 6 status: BLOCKED (controller must beat 3.8167s survival)")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
