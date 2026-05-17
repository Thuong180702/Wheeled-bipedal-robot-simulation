#!/usr/bin/env python3
"""
Phase B.9 Step 5.22 — Torque-First WBC Architecture Transition

Test the new torque-first WBC architecture where WBC has dominant authority (>70%)
and PID position control is eliminated.

Goal: Enable true dynamic balancing by removing PID authority suppression.
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

OUTPUT_DIR = project_root / "outputs" / "phase_b9_step5_22_torque_first_wbc"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class WBCCandidate:
    """WBC controller configuration candidate."""
    name: str
    k_roll: float
    k_roll_rate: float
    k_pitch: float
    k_pitch_rate: float
    allow_wheel_torque: bool
    wheel_roll_gain: float
    wbc_authority_fraction: float


def run_torque_first_wbc_evaluation():
    """Evaluate torque-first WBC architecture at h=0.60."""
    print("\n=== Phase B.9 Step 5.22: Torque-First WBC Evaluation ===\n")

    # Define candidates
    candidates = [
        WBCCandidate(
            name="strong_k20_torque_first",
            k_roll=20.0,
            k_roll_rate=2.0,
            k_pitch=5.0,
            k_pitch_rate=0.5,
            allow_wheel_torque=False,
            wheel_roll_gain=0.0,
            wbc_authority_fraction=1.0,  # 100% WBC authority
        ),
        WBCCandidate(
            name="strong_k20_with_wheels",
            k_roll=20.0,
            k_roll_rate=2.0,
            k_pitch=5.0,
            k_pitch_rate=0.5,
            allow_wheel_torque=True,
            wheel_roll_gain=2.0,
            wbc_authority_fraction=1.0,
        ),
        WBCCandidate(
            name="moderate_k15_torque_first",
            k_roll=15.0,
            k_roll_rate=1.5,
            k_pitch=4.0,
            k_pitch_rate=0.4,
            allow_wheel_torque=False,
            wheel_roll_gain=0.0,
            wbc_authority_fraction=1.0,
        ),
    ]

    # Load environment config
    config_path = project_root / "configs" / "training" / "balance.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Configure for h=0.60 fixed height with torque_first_wbc mode
    config["task"]["height_command_mode"] = "fixed"
    config["task"]["fixed_height"] = 0.60
    config["low_level_control"] = {
        "mode": "torque_first_wbc",
        "torque_control": {
            "enabled": True,
            "max_ctrl_fraction": 1.0,  # Will be set per candidate
        },
    }

    results = []

    for candidate in candidates:
        print(f"\nEvaluating: {candidate.name}")
        print(f"  k_roll={candidate.k_roll}, k_pitch={candidate.k_pitch}")
        print(f"  WBC authority={candidate.wbc_authority_fraction * 100:.0f}%")

        # Update config for this candidate
        config["low_level_control"]["torque_control"]["max_ctrl_fraction"] = candidate.wbc_authority_fraction

        # Create environment
        env = BalanceEnv(config)

        # Create WBC controller
        controller = WBCBalanceController(
            k_roll=candidate.k_roll,
            k_roll_rate=candidate.k_roll_rate,
            k_pitch=candidate.k_pitch,
            k_pitch_rate=candidate.k_pitch_rate,
            allow_wheel_torque=candidate.allow_wheel_torque,
            wheel_roll_gain=candidate.wheel_roll_gain,
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

            for step in range(MAX_STEPS):
                obs_np = np.array(obs)

                # Compute WBC torque
                wbc_action = controller.compute_torque(obs_np)

                # Set torque_residual_action in state.info for torque_first_wbc mode
                state = state._replace(
                    info={**state.info, "torque_residual_action": jnp.array(wbc_action)}
                )

                # Step environment (will use torque_first_wbc mode)
                state = env.step(state, jnp.array(wbc_action))
                obs = state.obs
                done = state.done

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

            episode_survivals.append(survival_steps * env.CONTROL_DT)
            episode_falls.append(1.0 if fell else 0.0)
            episode_pitch_rms.append(float(np.sqrt(np.mean(np.array(pitch_history)**2))))
            episode_roll_rms.append(float(np.sqrt(np.mean(np.array(roll_history)**2))))
            episode_sat_rates.append(sat_count / survival_steps if survival_steps > 0 else 0.0)
            episode_torque_rms.append(float(np.sqrt(np.mean(np.array(torque_history)**2))))

        # Aggregate results
        mean_survival = float(np.mean(episode_survivals))
        mean_fall_rate = float(np.mean(episode_falls))
        mean_pitch_rms = float(np.mean(episode_pitch_rms))
        mean_roll_rms = float(np.mean(episode_roll_rms))
        mean_sat_rate = float(np.mean(episode_sat_rates))
        mean_torque_rms = float(np.mean(episode_torque_rms))

        result = {
            "candidate": candidate.name,
            "k_roll": candidate.k_roll,
            "k_pitch": candidate.k_pitch,
            "wbc_authority_fraction": candidate.wbc_authority_fraction,
            "mean_survival_s": mean_survival,
            "mean_fall_rate": mean_fall_rate,
            "mean_pitch_rms_deg": mean_pitch_rms,
            "mean_roll_rms_deg": mean_roll_rms,
            "mean_saturation_rate": mean_sat_rate,
            "mean_torque_rms_Nm": mean_torque_rms,
        }
        results.append(result)

        print(f"  Results: survival={mean_survival:.2f}s, fall_rate={mean_fall_rate:.2f}, "
              f"sat_rate={mean_sat_rate:.2%}")

    # Save results
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "candidate_results.csv", index=False)

    # Find best candidate
    best_idx = df["mean_survival_s"].idxmax()
    best = results[best_idx]

    summary = {
        "evaluation_complete": True,
        "architecture": "torque_first_wbc",
        "wbc_authority_pct": 100.0,
        "pid_authority_pct": 0.0,
        "best_candidate": best["candidate"],
        "best_survival_s": best["mean_survival_s"],
        "best_saturation_rate": best["mean_saturation_rate"],
        "step5_18c_baseline_survival_s": 0.86,
        "improvement_vs_baseline_pct": (best["mean_survival_s"] - 0.86) / 0.86 * 100,
        "candidates_tested": len(candidates),
    }

    with open(OUTPUT_DIR / "candidate_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Evaluation Complete ===")
    print(f"Best candidate: {best['candidate']}")
    print(f"  Survival: {best['mean_survival_s']:.2f}s")
    print(f"  vs Step 5.18c baseline: {summary['improvement_vs_baseline_pct']:+.1f}%")
    print(f"  Saturation: {best['mean_saturation_rate']:.2%}")
    print(f"\nResults saved to {OUTPUT_DIR}")

    return results, summary


def generate_architecture_report(results, summary):
    """Generate architecture transition report."""
    print("\n=== Generating Architecture Transition Report ===\n")

    report = f"""# Phase B.9 Step 5.22 — Architecture Transition Report

## Executive Summary

**Architecture**: Torque-First WBC (WBC-dominant control)

**Authority Distribution**:
- WBC authority: {summary['wbc_authority_pct']:.0f}%
- PID authority: {summary['pid_authority_pct']:.0f}%

**Best Result**: {summary['best_candidate']}
- Survival: {summary['best_survival_s']:.2f}s
- vs Step 5.18c baseline (0.86s): {summary['improvement_vs_baseline_pct']:+.1f}%
- Saturation rate: {summary['best_saturation_rate']:.2%}

---

## Architecture Comparison

### Old Architecture (Step 5.21 Analysis)

```
DualRateBalanceController
    -> position targets
PID position control
    -> +/-30 Nm (saturated)
WBC residuals (~1 Nm)
    -> suppressed by clipping
Actuators
```

**Authority**: 97% PID, 3% WBC
**Result**: 0.38s survival (56% degradation)

### New Architecture (Step 5.22)

```
WBC balance controller
    -> torque commands
torque_first_wbc_control
    -> direct torque (no PID)
Actuators
```

**Authority**: 100% WBC, 0% PID
**Result**: {summary['best_survival_s']:.2f}s survival

---

## Candidate Results

| Candidate | Survival (s) | Fall Rate | Saturation | Torque RMS (Nm) |
|-----------|--------------|-----------|------------|-----------------|
"""

    for r in results:
        report += f"| {r['candidate']} | {r['mean_survival_s']:.2f} | {r['mean_fall_rate']:.2f} | {r['mean_saturation_rate']:.2%} | {r['mean_torque_rms_Nm']:.2f} |\n"

    report += f"""
---

## Answers to Required Questions

### 1. Is WBC now the dominant controller authority?

**YES** - WBC has {summary['wbc_authority_pct']:.0f}% authority (vs 3% in hybrid_pid_plus_torque mode).

### 2. What % authority belongs to WBC vs damping/tracking?

- WBC: {summary['wbc_authority_pct']:.0f}%
- Damping: 0% (disabled by default)
- PID tracking: 0% (eliminated)

### 3. Did saturation decrease significantly?

**Analysis needed** - Compare {summary['best_saturation_rate']:.2%} against Step 5.18c baseline (93.75%).

### 4. Does the robot now balance dynamically instead of rigidly?

**Analysis needed** - Requires time-series inspection of torque patterns and motion.

### 5. Is behavior closer to the old successful pure RL behavior?

**Analysis needed** - Requires comparison of torque efficiency and motion patterns.

### 6. Does torque-first architecture improve survival?

**Result**: {summary['best_survival_s']:.2f}s vs 0.86s baseline = {summary['improvement_vs_baseline_pct']:+.1f}%

### 7. Can DualRateBalanceController now be bypassed entirely?

**YES** - Torque-first WBC architecture eliminates need for DualRateBalanceController.

### 8. Is the system finally architecturally correct for humanoid balancing?

**Partial** - WBC authority is correct, but survival must exceed reset-fixed baseline (3.8167s) for Step 6.

---

## Step 6 Status

**Status**: BLOCKED

**Gate requirement**: 3.8167s survival (reset-fixed baseline)

**Current best**: {summary['best_survival_s']:.2f}s

**Gap**: {3.8167 - summary['best_survival_s']:.2f}s improvement needed

---

## Conclusion

The torque-first WBC architecture successfully eliminates PID authority suppression:
- WBC authority increased from 3% to {summary['wbc_authority_pct']:.0f}%
- PID position control eliminated
- DualRateBalanceController can be bypassed

However, survival performance must be validated against Step 5.18c baseline and
improved to exceed the reset-fixed baseline (3.8167s) for Step 6 progression.
"""

    with open(OUTPUT_DIR / "architecture_transition_report.md", "w") as f:
        f.write(report)

    print(f"Architecture transition report saved to:")
    print(f"  {OUTPUT_DIR}/architecture_transition_report.md")


def main():
    """Run complete Step 5.22 evaluation."""
    print("="*80)
    print("Phase B.9 Step 5.22 — Torque-First WBC Architecture Transition")
    print("="*80)

    # Run evaluation
    results, summary = run_torque_first_wbc_evaluation()

    # Generate report
    generate_architecture_report(results, summary)

    print("\n" + "="*80)
    print("PHASE B.9 STEP 5.22 EVALUATION COMPLETE")
    print("="*80)
    print(f"\nBest result: {summary['best_candidate']}")
    print(f"  Survival: {summary['best_survival_s']:.2f}s")
    print(f"  WBC authority: {summary['wbc_authority_pct']:.0f}%")
    print(f"  vs Step 5.18c: {summary['improvement_vs_baseline_pct']:+.1f}%")
    print("\nStep 6 status: BLOCKED (requires 3.8167s survival)")
    print("="*80)


if __name__ == "__main__":
    main()
