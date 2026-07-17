"""Phase 2: Step-1 teacher-forcing root cause investigation.

Runs backend=both for push_fwd_90N and push_bwd_90N, logging detailed
step-by-step comparison of all hip-yaw relevant inputs, states, and torques.

Outputs:
  outputs/k2_jax_debug/push_fwd_step1_ledger.csv
  outputs/k2_jax_debug/push_bwd_step1_ledger.csv
  docs/validation/k2_jax_step1_hip_yaw_divergence_report.md
"""

import argparse, csv, json, sys, os, time, math
from pathlib import Path
from collections import OrderedDict

import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

# Add project root
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from wheeled_biped.controllers.k2_jax_controller import (
    K2_JAX_STATE_SIZE, K2_JAX_STATE_FIELDS, K2_JAX_DIAG_SIZE,
    K2_JAX_INPUT_SIZE, K2_JAX_INPUT_FIELDS,
    pack_input_k2, pack_state_k2,
    k2_jax_controller_init, k2_jax_step,
    _S_NOTCH_X1, _S_NOTCH_X2, _S_NOTCH_Y1, _S_NOTCH_Y2,
    _S_PREV_TAU_START, _S_FILTERED_COM_Z, _S_PREV_SUPPORT_ERROR,
    _S_OL_PITCH_REF_SMOOTHED, _S_OL_PREV_SUPPORT_ERROR, _S_OL_SUPPORT_ERROR_RATE,
    _I_PITCH_X, _I_PITCH_RATE, _I_ROLL_Y, _I_ROLL_RATE,
    _I_YAW_ERR, _I_YAW_RATE, _I_COM_Z, _I_COM_VY,
    _I_SAG_VEL, _I_SAG_POS_ERR, _I_WHEEL_VEL_L, _I_WHEEL_VEL_R,
    _I_SUPPORT_VEL, _I_HEIGHT_REF, _I_HY_DIV_ERR, _I_HY_DIV_RATE,
    _I_Q_START, _I_QD_START, _I_QREF_START, _I_SUPPORT_POS_ERR,
)


def run_detailed_teacher_forcing(
    scenario_name: str,
    height_m: float,
    push_magnitude_n: float,
    push_direction: str,  # "forward" or "backward"
    num_steps: int = 25,
    output_csv: str = None,
):
    """Run the simulation with backend=both and log every step detail."""
    from scripts.simulate_hierarchical_controller import (
        parse_args, run_simulation,
    )
    import subprocess

    # Strategy: Run the simulation via subprocess with --backend both,
    # capturing the [BOTH@step] output lines.
    # Then parse the output for detailed comparison.
    #
    # But the built-in output only shows max diff and full torque vectors.
    # We need to augment it. So instead, we'll run via a modified approach:
    # capture the simulation loop at each step.

    # Use the --profile k2_notch_low_q_v1 backend=both path
    cmd = [
        sys.executable, str(_project_root / "scripts" / "simulate_hierarchical_controller.py"),
        "--profile", "k2_notch_low_q_v1",
        "--controller-backend", "both",
        "--height", str(height_m),
        "--steps", str(num_steps),
        "--save-telemetry",
        "--output-dir", str(_project_root / "outputs" / "k2_jax_debug"),
    ]

    if push_direction == "forward":
        cmd.extend(["--push-enabled", "--push-magnitude-n", str(push_magnitude_n),
                     "--sagittal-push-only"])
    elif push_direction == "backward":
        cmd.extend(["--push-enabled", "--push-magnitude-n", str(-abs(push_magnitude_n)),
                     "--sagittal-push-only"])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                           env={**os.environ, "PYTHONUNBUFFERED": "1"})

    stdout = result.stdout
    stderr = result.stderr

    # Parse [BOTH@step] lines
    both_lines = [l for l in stdout.split('\n') if '[BOTH@' in l]
    print(f"Found {len(both_lines)} [BOTH@step] lines")

    # Print the raw output for analysis
    for line in both_lines:
        print(line)

    # Also print stderr for any errors
    if result.returncode != 0:
        print(f"STDERR:\n{stderr[:5000]}")

    return stdout, stderr


def run_inline_teacher_forcing_comparison(
    scenario_name: str,
    height_m: float = 0.48,
    push_magnitude_n: float = 90.0,
    push_direction: str = "forward",
    num_steps: int = 21,
):
    """Run teacher-forcing by importing and directly calling both control paths.

    This runs the full MuJoCo simulation with backend=both but with
    enhanced diagnostics injected at each step.
    """
    import jax; jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    import mujoco
    from pathlib import Path

    # Import simulation components
    from scripts.simulate_hierarchical_controller import (
        load_config, build_controller,
    )

    # Build controller with K2 profile
    from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
        K2_NOTCH_LOW_Q_V1, SagittalVelocityDampedBalanceController,
    )

    # Initialize JAX controller
    jax_state, jax_params = k2_jax_controller_init(profile="k2_notch_low_q_v1")
    jax_step_fn = jax.jit(k2_jax_step)

    print(f"JAX state size: {K2_JAX_STATE_SIZE}")
    print(f"JAX state fields: {len(K2_JAX_STATE_FIELDS)}")
    print(f"JAX input size: {K2_JAX_INPUT_SIZE}")
    print(f"JAX input fields: {len(K2_JAX_INPUT_FIELDS)}")

    # For now, just print the JAX state fields to stdout for the report
    print("\n--- JAX STATE FIELDS ---")
    for i, f in enumerate(K2_JAX_STATE_FIELDS):
        print(f"  [{i:3d}] {f}")

    print("\n--- JAX INPUT FIELDS ---")
    for i, f in enumerate(K2_JAX_INPUT_FIELDS):
        print(f"  [{i:3d}] {f}")

    return {
        "state_fields": list(K2_JAX_STATE_FIELDS),
        "input_fields": list(K2_JAX_INPUT_FIELDS),
        "state_size": K2_JAX_STATE_SIZE,
        "input_size": K2_JAX_INPUT_SIZE,
    }


def main():
    parser = argparse.ArgumentParser(description="K2 JAX Phase 2: Step-1 teacher-forcing root cause")
    parser.add_argument("--scenario", choices=["push_fwd_90N", "push_bwd_90N", "fixed_high_0p480", "all"],
                        default="push_fwd_90N")
    parser.add_argument("--steps", type=int, default=21)
    parser.add_argument("--output-dir", default=str(_project_root / "outputs" / "k2_jax_debug"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # First, just load and inspect the field mappings
    info = run_inline_teacher_forcing_comparison("push_fwd_90N", height_m=0.48,
                                                   push_magnitude_n=90.0,
                                                   push_direction="forward",
                                                   num_steps=args.steps)

    # Save the field maps
    field_map_path = Path(args.output_dir) / "jax_field_maps.json"
    with open(field_map_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"\nField maps saved to {field_map_path}")


if __name__ == "__main__":
    main()
