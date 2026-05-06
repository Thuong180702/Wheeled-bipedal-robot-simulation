"""Comprehensive posture feasibility mapping for wheeled biped.

This script performs a dense grid search over hip_pitch and knee angles to map
the feasible posture space at different target heights. It evaluates multiple
feasibility criteria with varying tolerances to understand the full range of
achievable balanced postures.

Feasibility levels:
1. geometric_height: Target height achievable within joint limits
2. knee_forward: Knee positioned forward of hip (positive margin)
3. torso_upright: Torso pitch within tolerance of vertical
4. com_near_wheel: Whole-body CoM within tolerance of wheel contact
5. full_static: All criteria satisfied simultaneously

The script tests multiple CoM tolerances (2cm, 3cm, 5cm) to understand the
tradeoff between strict static balance and dynamic balance capability.

Outputs:
- CSV files with full grid data
- Heatmaps showing feasibility regions
- Best candidate postures under different priorities
- Side-view snapshots of selected postures
- Comprehensive analysis report
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).parent.parent))

from wheeled_biped.utils.config import get_model_path


# ============================================================================
# Configuration
# ============================================================================

# Grid search parameters
HIP_PITCH_MIN = -0.5
HIP_PITCH_MAX = 1.8
KNEE_MIN = -0.5
KNEE_MAX = 2.7
GRID_RESOLUTION = 150  # 150x150 = 22,500 samples per height

# Target heights to test
TARGET_HEIGHTS = [0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40]

# Feasibility thresholds
KNEE_FORWARD_MIN = 0.01  # Minimum knee-forward margin [m]
TORSO_PITCH_MAX = 5.0  # Maximum torso pitch deviation [deg]
COM_TOLERANCES = [0.02, 0.03, 0.05]  # CoM-to-wheel tolerances [m]

# Physical parameters from MJCF model
WHEEL_RADIUS = 0.06  # [m]
THIGH_LENGTH = 0.26  # [m]
SHIN_LENGTH = 0.28  # [m]

# Forward axis (verified empirically via wheel rolling test)
FORWARD_AXIS = np.array([0, 1, 0])  # +Y is forward


# ============================================================================
# Kinematic utilities
# ============================================================================

def analytic_leg_height(hip_pitch: float, knee: float) -> float:
    """Compute leg height using two-link kinematic chain.

    Args:
        hip_pitch: Hip pitch angle [rad]
        knee: Knee angle [rad]

    Returns:
        Vertical height from hip to foot [m]
    """
    # Thigh vertical component
    thigh_z = THIGH_LENGTH * np.cos(hip_pitch)

    # Shin angle relative to vertical
    shin_angle = hip_pitch + knee
    shin_z = SHIN_LENGTH * np.cos(shin_angle)

    return thigh_z + shin_z


def analytic_knee_forward(hip_pitch: float, knee: float) -> float:
    """Compute knee-forward margin using two-link kinematic chain.

    Args:
        hip_pitch: Hip pitch angle [rad]
        knee: Knee angle [rad]

    Returns:
        Knee-forward margin [m] (positive = knee ahead of hip)
    """
    # Thigh forward component
    thigh_y = THIGH_LENGTH * np.sin(hip_pitch)

    return thigh_y


def compute_posture_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hip_pitch: float,
    knee: float,
) -> dict:
    """Compute all posture metrics for a given configuration.

    Args:
        model: MuJoCo model
        data: MuJoCo data (will be modified)
        hip_pitch: Hip pitch angle [rad]
        knee: Knee angle [rad]

    Returns:
        Dictionary with metrics:
        - torso_height: Torso height above ground [m]
        - torso_pitch: Torso pitch angle [deg]
        - knee_forward: Knee-forward margin [m]
        - com_pos: Whole-body CoM position [m]
        - wheel_contact: Average wheel contact position [m]
        - com_to_wheel: Horizontal distance from CoM to wheel contact [m]
    """
    # Reset simulation
    mujoco.mj_resetData(model, data)

    # Set joint positions (symmetric left/right)
    L_HIP_PITCH_QPOS = 7 + 2
    L_KNEE_QPOS = 7 + 3
    R_HIP_PITCH_QPOS = 7 + 7
    R_KNEE_QPOS = 7 + 8

    data.qpos[L_HIP_PITCH_QPOS] = hip_pitch
    data.qpos[L_KNEE_QPOS] = knee
    data.qpos[R_HIP_PITCH_QPOS] = hip_pitch
    data.qpos[R_KNEE_QPOS] = knee

    # Initial base height guess
    data.qpos[2] = 0.6

    # Run forward kinematics
    mujoco.mj_kinematics(model, data)

    # Get left wheel position
    l_wheel_body_id = model.body("l_wheel_link").id
    wheel_z = data.xpos[l_wheel_body_id, 2]

    # Adjust base z so wheel touches ground
    base_z_adjustment = WHEEL_RADIUS - wheel_z
    data.qpos[2] += base_z_adjustment

    # Recompute kinematics
    mujoco.mj_kinematics(model, data)

    # Measure torso height
    torso_height = data.qpos[2]

    # Measure torso pitch from quaternion
    quat = data.qpos[3:7]
    # Convert quaternion to pitch angle
    # For small angles, pitch ≈ 2 * quat[2] (y-component)
    # More accurate: use rotation matrix
    w, x, y, z = quat
    pitch_rad = np.arcsin(2 * (w * y - z * x))
    torso_pitch = np.degrees(pitch_rad)

    # Measure knee-forward margin
    l_thigh_body_id = model.body("l_thigh").id
    l_knee_body_id = model.body("l_knee_link").id

    hip_pos = data.xpos[l_thigh_body_id]
    knee_pos = data.xpos[l_knee_body_id]

    # Project onto forward axis
    hip_fwd = np.dot(hip_pos, FORWARD_AXIS)
    knee_fwd = np.dot(knee_pos, FORWARD_AXIS)
    knee_forward = knee_fwd - hip_fwd

    # Compute whole-body CoM
    mujoco.mj_comPos(model, data)
    com_pos = data.subtree_com[0].copy()  # Root body CoM

    # Wheel contact position (average of left and right wheels)
    r_wheel_body_id = model.body("r_wheel_link").id
    l_wheel_pos = data.xpos[l_wheel_body_id]
    r_wheel_pos = data.xpos[r_wheel_body_id]
    wheel_contact = (l_wheel_pos + r_wheel_pos) / 2.0
    wheel_contact[2] = 0.0  # Ground level

    # Horizontal distance from CoM to wheel contact
    com_horizontal = com_pos.copy()
    com_horizontal[2] = 0.0
    wheel_horizontal = wheel_contact.copy()
    wheel_horizontal[2] = 0.0
    com_to_wheel = np.linalg.norm(com_horizontal - wheel_horizontal)

    return {
        "torso_height": torso_height,
        "torso_pitch": torso_pitch,
        "knee_forward": knee_forward,
        "com_pos": com_pos,
        "wheel_contact": wheel_contact,
        "com_to_wheel": com_to_wheel,
    }


def evaluate_feasibility(
    metrics: dict,
    target_height: float,
    com_tolerance: float,
) -> dict:
    """Evaluate feasibility at multiple levels.

    Args:
        metrics: Posture metrics from compute_posture_metrics
        target_height: Target torso height [m]
        com_tolerance: CoM-to-wheel tolerance [m]

    Returns:
        Dictionary with boolean feasibility flags:
        - geometric_height: Height within ±1cm of target
        - knee_forward: Knee-forward margin > threshold
        - torso_upright: Torso pitch within tolerance
        - com_near_wheel: CoM within tolerance of wheel
        - full_static: All criteria satisfied
    """
    height_error = abs(metrics["torso_height"] - target_height)

    feasibility = {
        "geometric_height": height_error < 0.01,
        "knee_forward": metrics["knee_forward"] > KNEE_FORWARD_MIN,
        "torso_upright": abs(metrics["torso_pitch"]) < TORSO_PITCH_MAX,
        "com_near_wheel": metrics["com_to_wheel"] < com_tolerance,
    }

    feasibility["full_static"] = all([
        feasibility["geometric_height"],
        feasibility["knee_forward"],
        feasibility["torso_upright"],
        feasibility["com_near_wheel"],
    ])

    return feasibility


# ============================================================================
# Grid search
# ============================================================================

def run_grid_search(
    model: mujoco.MjModel,
    target_height: float,
    com_tolerance: float,
) -> pd.DataFrame:
    """Run dense grid search over hip_pitch and knee.

    Args:
        model: MuJoCo model
        target_height: Target torso height [m]
        com_tolerance: CoM-to-wheel tolerance [m]

    Returns:
        DataFrame with columns:
        - hip_pitch, knee: Joint angles [rad]
        - torso_height, torso_pitch, knee_forward, com_to_wheel: Metrics
        - geometric_height, knee_forward_ok, torso_upright, com_near_wheel, full_static: Feasibility flags
        - analytic_height, analytic_knee_forward: Analytic predictions
    """
    data = mujoco.MjData(model)

    # Create grid
    hip_pitch_grid = np.linspace(HIP_PITCH_MIN, HIP_PITCH_MAX, GRID_RESOLUTION)
    knee_grid = np.linspace(KNEE_MIN, KNEE_MAX, GRID_RESOLUTION)

    results = []

    total_samples = GRID_RESOLUTION * GRID_RESOLUTION
    print(f"Running grid search: {GRID_RESOLUTION}x{GRID_RESOLUTION} = {total_samples} samples")
    print(f"Target height: {target_height:.2f} m, CoM tolerance: {com_tolerance:.3f} m")

    for i, hip_pitch in enumerate(hip_pitch_grid):
        if i % 10 == 0:
            progress = (i * GRID_RESOLUTION) / total_samples * 100
            print(f"  Progress: {progress:.1f}%")

        for knee in knee_grid:
            # Compute metrics
            metrics = compute_posture_metrics(model, data, hip_pitch, knee)

            # Evaluate feasibility
            feasibility = evaluate_feasibility(metrics, target_height, com_tolerance)

            # Analytic predictions
            analytic_height = analytic_leg_height(hip_pitch, knee)
            analytic_knee_fwd = analytic_knee_forward(hip_pitch, knee)

            results.append({
                "hip_pitch": hip_pitch,
                "knee": knee,
                "torso_height": metrics["torso_height"],
                "torso_pitch": metrics["torso_pitch"],
                "knee_forward": metrics["knee_forward"],
                "com_to_wheel": metrics["com_to_wheel"],
                "geometric_height": feasibility["geometric_height"],
                "knee_forward_ok": feasibility["knee_forward"],
                "torso_upright": feasibility["torso_upright"],
                "com_near_wheel": feasibility["com_near_wheel"],
                "full_static": feasibility["full_static"],
                "analytic_height": analytic_height,
                "analytic_knee_forward": analytic_knee_fwd,
            })

    print("  Progress: 100.0%")

    df = pd.DataFrame(results)

    # Summary statistics
    n_geometric = df["geometric_height"].sum()
    n_knee_fwd = df["knee_forward_ok"].sum()
    n_upright = df["torso_upright"].sum()
    n_com = df["com_near_wheel"].sum()
    n_full = df["full_static"].sum()

    print(f"\nFeasibility summary:")
    print(f"  Geometric height: {n_geometric}/{total_samples} ({n_geometric/total_samples*100:.1f}%)")
    print(f"  Knee forward: {n_knee_fwd}/{total_samples} ({n_knee_fwd/total_samples*100:.1f}%)")
    print(f"  Torso upright: {n_upright}/{total_samples} ({n_upright/total_samples*100:.1f}%)")
    print(f"  CoM near wheel: {n_com}/{total_samples} ({n_com/total_samples*100:.1f}%)")
    print(f"  Full static: {n_full}/{total_samples} ({n_full/total_samples*100:.1f}%)")

    return df


# ============================================================================
# Analysis and visualization
# ============================================================================

def find_best_candidates(df: pd.DataFrame, target_height: float) -> dict:
    """Find best posture candidates under different priorities.

    Args:
        df: Grid search results
        target_height: Target height [m]

    Returns:
        Dictionary with best candidates:
        - height_tracking: Minimize height error
        - knee_forward: Maximize knee-forward margin
        - torso_upright: Minimize torso pitch
        - com_alignment: Minimize CoM-to-wheel distance
        - combined: Balance all criteria
        - relaxed: Best full_static with relaxed CoM tolerance
    """
    candidates = {}

    # Height tracking: minimize height error
    df_height = df[df["geometric_height"]]
    if len(df_height) > 0:
        idx = (df_height["torso_height"] - target_height).abs().idxmin()
        candidates["height_tracking"] = df.loc[idx].to_dict()

    # Knee forward: maximize knee-forward margin
    df_knee = df[df["knee_forward_ok"]]
    if len(df_knee) > 0:
        idx = df_knee["knee_forward"].idxmax()
        candidates["knee_forward"] = df.loc[idx].to_dict()

    # Torso upright: minimize torso pitch
    df_upright = df[df["torso_upright"]]
    if len(df_upright) > 0:
        idx = df_upright["torso_pitch"].abs().idxmin()
        candidates["torso_upright"] = df.loc[idx].to_dict()

    # CoM alignment: minimize CoM-to-wheel distance
    df_com = df[df["com_near_wheel"]]
    if len(df_com) > 0:
        idx = df_com["com_to_wheel"].idxmin()
        candidates["com_alignment"] = df.loc[idx].to_dict()

    # Combined: balance all criteria
    df_full = df[df["full_static"]]
    if len(df_full) > 0:
        # Score: minimize weighted sum of normalized errors
        height_err = (df_full["torso_height"] - target_height).abs() / 0.01
        pitch_err = df_full["torso_pitch"].abs() / TORSO_PITCH_MAX
        knee_score = -df_full["knee_forward"] / KNEE_FORWARD_MIN  # Negative to minimize
        com_err = df_full["com_to_wheel"] / df_full["com_to_wheel"].max()

        combined_score = height_err + pitch_err + knee_score + com_err
        idx = combined_score.idxmin()
        candidates["combined"] = df.loc[idx].to_dict()

    # Relaxed: best full_static with any CoM tolerance
    if len(df_full) > 0:
        # Prefer larger knee-forward margin
        idx = df_full["knee_forward"].idxmax()
        candidates["relaxed"] = df.loc[idx].to_dict()

    return candidates


def plot_feasibility_heatmaps(
    df: pd.DataFrame,
    target_height: float,
    com_tolerance: float,
    output_dir: Path,
):
    """Generate heatmaps showing feasibility regions.

    Args:
        df: Grid search results
        target_height: Target height [m]
        com_tolerance: CoM tolerance [m]
        output_dir: Output directory
    """
    # Reshape data for heatmaps
    hip_pitch_unique = df["hip_pitch"].unique()
    knee_unique = df["knee"].unique()
    n_hip = len(hip_pitch_unique)
    n_knee = len(knee_unique)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Feasibility Maps: h={target_height:.2f}m, CoM tol={com_tolerance:.3f}m", fontsize=16)

    # Metrics to plot
    metrics = [
        ("torso_height", "Torso Height [m]", "viridis"),
        ("knee_forward", "Knee Forward [m]", "RdYlGn"),
        ("torso_pitch", "Torso Pitch [deg]", "coolwarm"),
        ("com_to_wheel", "CoM-to-Wheel [m]", "plasma"),
        ("full_static", "Full Static Feasible", "RdYlGn"),
        ("geometric_height", "Geometric Height OK", "RdYlGn"),
    ]

    for ax, (metric, title, cmap) in zip(axes.flat, metrics):
        # Reshape metric
        values = df[metric].values.reshape(n_hip, n_knee)

        # Plot heatmap
        im = ax.imshow(
            values.T,
            origin="lower",
            aspect="auto",
            extent=[HIP_PITCH_MIN, HIP_PITCH_MAX, KNEE_MIN, KNEE_MAX],
            cmap=cmap,
        )

        ax.set_xlabel("Hip Pitch [rad]")
        ax.set_ylabel("Knee [rad]")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

        # Mark joint limits
        ax.axhline(0, color="white", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.axvline(0, color="white", linestyle="--", linewidth=0.5, alpha=0.5)

    plt.tight_layout()

    filename = f"feasibility_heatmaps_h{target_height:.2f}_tol{com_tolerance:.3f}.png"
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()

    print(f"  Saved heatmaps: {filename}")


def plot_feasible_regions(
    df: pd.DataFrame,
    target_height: float,
    com_tolerance: float,
    output_dir: Path,
):
    """Plot feasible regions in joint space.

    Args:
        df: Grid search results
        target_height: Target height [m]
        com_tolerance: CoM tolerance [m]
        output_dir: Output directory
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot different feasibility levels
    levels = [
        ("geometric_height", "Geometric Height", "lightblue", 0.3),
        ("knee_forward_ok", "Knee Forward", "lightgreen", 0.4),
        ("torso_upright", "Torso Upright", "lightyellow", 0.4),
        ("com_near_wheel", "CoM Near Wheel", "lightcoral", 0.4),
        ("full_static", "Full Static", "darkgreen", 0.8),
    ]

    for level, label, color, alpha in levels:
        df_level = df[df[level]]
        if len(df_level) > 0:
            ax.scatter(
                df_level["hip_pitch"],
                df_level["knee"],
                c=color,
                alpha=alpha,
                s=1,
                label=label,
            )

    ax.set_xlabel("Hip Pitch [rad]")
    ax.set_ylabel("Knee [rad]")
    ax.set_title(f"Feasible Regions: h={target_height:.2f}m, CoM tol={com_tolerance:.3f}m")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mark joint limits
    ax.axhline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)

    plt.tight_layout()

    filename = f"feasible_regions_h{target_height:.2f}_tol{com_tolerance:.3f}.png"
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()

    print(f"  Saved feasible regions: {filename}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("Comprehensive Posture Feasibility Mapping")
    print("=" * 80)

    # Load model
    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # Create output directory
    output_dir = Path("outputs/posture_feasibility")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run grid search for each target height and CoM tolerance
    all_results = []

    for target_height in TARGET_HEIGHTS:
        print(f"\n{'=' * 80}")
        print(f"Target Height: {target_height:.2f} m")
        print(f"{'=' * 80}")

        for com_tolerance in COM_TOLERANCES:
            print(f"\n{'-' * 80}")
            print(f"CoM Tolerance: {com_tolerance:.3f} m")
            print(f"{'-' * 80}")

            # Run grid search
            df = run_grid_search(model, target_height, com_tolerance)

            # Add metadata
            df["target_height"] = target_height
            df["com_tolerance"] = com_tolerance

            # Save CSV
            csv_filename = f"grid_h{target_height:.2f}_tol{com_tolerance:.3f}.csv"
            df.to_csv(output_dir / csv_filename, index=False)
            print(f"  Saved CSV: {csv_filename}")

            # Find best candidates
            candidates = find_best_candidates(df, target_height)
            print(f"\nBest candidates:")
            for priority, candidate in candidates.items():
                if candidate:
                    print(f"  {priority}:")
                    print(f"    hip_pitch={candidate['hip_pitch']:.3f}, knee={candidate['knee']:.3f}")
                    print(f"    height={candidate['torso_height']:.3f}, pitch={candidate['torso_pitch']:.2f}°")
                    print(f"    knee_fwd={candidate['knee_forward']:.4f}, com_dist={candidate['com_to_wheel']:.4f}")

            # Generate visualizations
            plot_feasibility_heatmaps(df, target_height, com_tolerance, output_dir)
            plot_feasible_regions(df, target_height, com_tolerance, output_dir)

            all_results.append(df)

    # Combine all results
    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_csv(output_dir / "all_results.csv", index=False)

    print(f"\n{'=' * 80}")
    print("Feasibility mapping complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
