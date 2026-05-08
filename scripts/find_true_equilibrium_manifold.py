"""Phase B.9 Task 6: Find true equilibrium manifold through numerical simulation.

Instead of analytical IK, use the actual MuJoCo simulator with PID to find
achievable equilibrium configurations at each height. This respects:
1. Kinematic constraints (joint limits)
2. Dynamic constraints (PID tracking limits)
3. Physical constraints (gravity, inertia)

Approach:
- For each target height, run closed-loop simulation with PID
- Let the system settle to equilibrium
- Record the resulting joint configuration
- Build empirical height → (hip_pitch, knee) mapping
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from rich.console import Console
from rich.progress import track
from rich.table import Table

from wheeled_biped.envs.balance_env import BalanceEnv
from wheeled_biped.utils.config import get_model_path

console = Console()


def find_equilibrium_via_simulation(
    env: BalanceEnv,
    height_cmd: float,
    max_settle_time: float = 20.0,
    settle_threshold: float = 0.01,
    rng_seed: int = 42,
) -> Dict[str, float]:
    """Find equilibrium configuration by running closed-loop simulation.

    Args:
        env: Balance environment with PID enabled
        height_cmd: Target height [m]
        max_settle_time: Maximum time to wait for settling [s]
        settle_threshold: Height error threshold for settling [m]
        rng_seed: Random seed

    Returns:
        Dictionary with equilibrium configuration and metrics
    """
    rng = jax.random.PRNGKey(rng_seed)
    state = env.reset(rng)

    # Override height command
    obs = state.obs.at[39].set(height_cmd)
    state = state._replace(obs=obs)

    max_steps = int(max_settle_time / env.CONTROL_DT)
    settle_window = int(2.0 / env.CONTROL_DT)  # 2 second window

    height_history = []
    joint_history = []

    for step in range(max_steps):
        # Zero action (let PID drive to neutral/IK targets)
        action = jnp.zeros(10)
        state = env.step(state, action)

        # Record state
        obs_np = np.array(state.obs)
        current_height = float(obs_np[40])
        qpos = obs_np[9:19]

        height_history.append(current_height)
        joint_history.append(qpos.copy())

        # Check settling
        if len(height_history) >= settle_window:
            recent_heights = height_history[-settle_window:]
            height_std = np.std(recent_heights)
            height_mean = np.mean(recent_heights)
            height_error = abs(height_mean - height_cmd)

            if height_std < 0.005 and height_error < settle_threshold:
                # Settled
                settled_qpos = np.mean(joint_history[-settle_window:], axis=0)
                settled_height = height_mean

                return {
                    "target_height": height_cmd,
                    "settled_height": settled_height,
                    "height_error": height_error,
                    "hip_pitch": float(settled_qpos[2]),  # l_hip_pitch
                    "knee": float(settled_qpos[3]),  # l_knee
                    "settle_time": step * env.CONTROL_DT,
                    "settled": True,
                }

        if state.done:
            # Failed to settle (fell)
            return {
                "target_height": height_cmd,
                "settled_height": float(height_history[-1]) if height_history else 0.0,
                "height_error": 999.0,
                "hip_pitch": 0.0,
                "knee": 0.0,
                "settle_time": step * env.CONTROL_DT,
                "settled": False,
            }

    # Timeout without settling
    settled_qpos = np.mean(joint_history[-settle_window:], axis=0)
    settled_height = np.mean(height_history[-settle_window:])

    return {
        "target_height": height_cmd,
        "settled_height": settled_height,
        "height_error": abs(settled_height - height_cmd),
        "hip_pitch": float(settled_qpos[2]),
        "knee": float(settled_qpos[3]),
        "settle_time": max_settle_time,
        "settled": False,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Find true equilibrium manifold via numerical simulation (Phase B.9 Task 6)"
    )
    parser.add_argument(
        "--heights",
        type=float,
        nargs="+",
        default=[0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40],
        help="Heights to evaluate [m]",
    )
    parser.add_argument(
        "--max-settle-time",
        type=float,
        default=20.0,
        help="Maximum time to wait for settling [s]",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase_b9_task6_true_manifold"),
        help="Output directory",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold cyan]Phase B.9 Task 6: Find True Equilibrium Manifold[/bold cyan]\n")
    console.print("Using closed-loop simulation with PID to find achievable configurations.\n")

    # Create environment with PID enabled
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
    }
    env = BalanceEnv(config=env_config)

    # Find equilibrium for each height
    results = []

    for height in track(args.heights, description="Finding equilibria"):
        eq = find_equilibrium_via_simulation(
            env, height, args.max_settle_time
        )
        results.append(eq)

        status = "[green]settled[/green]" if eq["settled"] else "[red]failed[/red]"
        console.print(
            f"  h={height:.2f}m: {status}, "
            f"actual={eq['settled_height']:.3f}m, "
            f"hip={eq['hip_pitch']:.3f}rad, knee={eq['knee']:.3f}rad, "
            f"time={eq['settle_time']:.1f}s"
        )

    # Display results table
    table = Table(title="True Equilibrium Manifold (via Simulation)")
    table.add_column("Target Height [m]", justify="right")
    table.add_column("Settled Height [m]", justify="right")
    table.add_column("Height Error [m]", justify="right")
    table.add_column("Hip Pitch [rad]", justify="right")
    table.add_column("Knee [rad]", justify="right")
    table.add_column("Settle Time [s]", justify="right")
    table.add_column("Settled", justify="center")

    for r in results:
        table.add_row(
            f"{r['target_height']:.2f}",
            f"{r['settled_height']:.3f}",
            f"{r['height_error']:.4f}",
            f"{r['hip_pitch']:.3f}",
            f"{r['knee']:.3f}",
            f"{r['settle_time']:.1f}",
            "Y" if r["settled"] else "N",
        )

    console.print(table)

    # Save results
    output_file = args.output_dir / "true_equilibrium_manifold.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"\n[green]Results saved to: {output_file}[/green]")

    # Analyze results
    console.print("\n[bold cyan]Analysis[/bold cyan]\n")

    settled_results = [r for r in results if r["settled"]]
    if settled_results:
        console.print(f"[green]Settled configurations: {len(settled_results)}/{len(results)}[/green]")

        # Check if height is achievable
        height_errors = [r["height_error"] for r in settled_results]
        mean_error = np.mean(height_errors)
        max_error = np.max(height_errors)

        console.print(f"Height tracking error: mean={mean_error:.4f}m, max={max_error:.4f}m")

        # Check joint configuration variation
        hip_pitches = [r["hip_pitch"] for r in settled_results]
        knees = [r["knee"] for r in settled_results]

        console.print(f"Hip pitch range: [{min(hip_pitches):.3f}, {max(hip_pitches):.3f}] rad")
        console.print(f"Knee range: [{min(knees):.3f}, {max(knees):.3f}] rad")

        # Compare with static equilibrium
        console.print("\n[yellow]Comparison with Static Equilibrium (Task 5):[/yellow]")
        console.print("Static equilibrium: hip~0.256rad, knee~0.538rad, height~0.71m")
        console.print(f"Simulation equilibrium: hip~{np.mean(hip_pitches):.3f}rad, "
                      f"knee~{np.mean(knees):.3f}rad, height~{np.mean([r['settled_height'] for r in settled_results]):.3f}m")
    else:
        console.print("[red]No configurations settled successfully![/red]")

    console.print("\n[bold green]Phase B.9 Task 6 complete![/bold green]")


if __name__ == "__main__":
    main()
