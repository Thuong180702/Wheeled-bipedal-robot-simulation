"""Practical evaluation of the three-level hierarchical controller.

Tests realistic scenarios:
1. Standing balance at nominal height
2. Height command tracking (squat/stand transitions)
3. Small lateral disturbances
4. Authority budget compliance

Reports clear metrics for each scenario.
"""

import jax
import jax.numpy as jnp
import numpy as np

from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
    CentroidalState,
)
from wheeled_biped.controllers.capture_point_estimator import (
    CapturePointEstimator,
    CapturePointEstimatorConfig,
)
from wheeled_biped.controllers.centroidal_balance_controller import (
    CentroidalBalanceController,
    CentroidalBalanceConfig,
)
from wheeled_biped.controllers.momentum_coordinator import (
    MomentumCoordinator,
    MomentumCoordinatorConfig,
)
from wheeled_biped.controllers.posture_regularizer import (
    PostureRegularizer,
    PostureRegularizerConfig,
)


def create_mock_state(com_height: float, com_vel: jnp.ndarray = None) -> CentroidalState:
    """Create a mock centroidal state for testing."""
    if com_vel is None:
        com_vel = jnp.zeros(3)

    return CentroidalState(
        com_pos=jnp.array([0.0, 0.0, com_height]),
        com_vel=com_vel,
        capture_point=jnp.array([0.0, 0.0]),
        divergence=jnp.array([0.0, 0.0]),
        linear_momentum=15.0 * com_vel,  # 15 kg robot mass
        angular_momentum=jnp.zeros(3),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=75.0,
        right_wheel_force=75.0,
    )


def create_mock_obs(height_cmd: float, current_height: float, joint_pos: jnp.ndarray) -> jnp.ndarray:
    """Create a mock observation array."""
    obs = jnp.zeros(42)
    obs = obs.at[0:3].set(jnp.array([0.0, 0.0, -9.81]))  # Gravity (upright)
    obs = obs.at[6:16].set(joint_pos)  # Joint positions
    obs = obs.at[36].set(height_cmd)  # Height command
    obs = obs.at[37].set(current_height)  # Current height
    return obs


def initialize_controllers():
    """Initialize all three controller levels."""
    # Phase 1: State Estimation
    capture_estimator = CapturePointEstimator(
        CapturePointEstimatorConfig(gravity=9.81, min_height=0.35)
    )

    # Phase 2: Centroidal WBC (60% authority)
    wbc_controller = CentroidalBalanceController(
        CentroidalBalanceConfig(
            k_roll=20.0,
            k_roll_rate=4.0,
            k_com_lateral=15.0,
            k_com_lateral_damping=3.0,
            k_com_sagittal=10.0,
            k_com_sagittal_damping=2.0,
            k_cp_lateral=25.0,
            k_cp_sagittal=20.0,
            k_cp_wheel_diff=8.0,
            k_height=5.0,
            wbc_authority_budget=0.6,
        )
    )

    # Phase 3: Momentum Coordinator (20% authority)
    momentum_coordinator = MomentumCoordinator(
        MomentumCoordinatorConfig(
            k_momentum_lateral=0.8,
            k_momentum_sagittal=1.2,
            k_angular_roll=1.5,
            k_feedforward=5.0,
            k_feedforward_hip=2.0,
            momentum_authority_budget=0.2,
        )
    )

    # Phase 4: Posture Regularizer (20% authority)
    posture_regularizer = PostureRegularizer(
        PostureRegularizerConfig(
            k_posture=2.0,
            hip_roll_deadband=0.05,
            hip_yaw_deadband=0.03,
            hip_pitch_deadband=0.08,
            knee_deadband=0.10,
            wbc_error_threshold=0.3,
            momentum_activity_threshold=0.1,
            momentum_active_scale=0.5,
            posture_authority_budget=0.2,
        )
    )

    return capture_estimator, wbc_controller, momentum_coordinator, posture_regularizer


def evaluate_scenario(name: str, states: list, joint_positions: list, height_cmds: list,
                      capture_estimator, wbc_controller, momentum_coordinator, posture_regularizer):
    """Evaluate controller on a scenario."""
    print(f"\n{'='*80}")
    print(f"Scenario: {name}")
    print(f"{'='*80}")

    max_wbc = 0.0
    max_momentum = 0.0
    max_posture = 0.0
    max_total = 0.0

    wbc_torques = []
    momentum_torques = []
    posture_torques = []

    for i, (state, joint_pos, height_cmd) in enumerate(zip(states, joint_positions, height_cmds)):
        # Update capture point
        state = capture_estimator.update(state)

        # Create observation
        obs = create_mock_obs(height_cmd, state.com_pos[2], joint_pos)

        # Compute torques from each level
        tau_wbc = wbc_controller.compute_centroidal_wbc_torque(obs, state)
        tau_momentum = momentum_coordinator.compute_momentum_coordinator_torque(obs, state)

        wbc_error_mag = float(jnp.max(jnp.abs(tau_wbc))) / 30.0
        momentum_mag = float(jnp.max(jnp.abs(tau_momentum))) / 30.0

        tau_posture = posture_regularizer.compute_posture_regularizer_torque(
            joint_pos, wbc_error_mag, momentum_mag
        )

        tau_total = tau_wbc + tau_momentum + tau_posture

        # Track max torques
        max_wbc = max(max_wbc, float(jnp.max(jnp.abs(tau_wbc))))
        max_momentum = max(max_momentum, float(jnp.max(jnp.abs(tau_momentum))))
        max_posture = max(max_posture, float(jnp.max(jnp.abs(tau_posture))))
        max_total = max(max_total, float(jnp.max(jnp.abs(tau_total))))

        wbc_torques.append(float(jnp.max(jnp.abs(tau_wbc))))
        momentum_torques.append(float(jnp.max(jnp.abs(tau_momentum))))
        posture_torques.append(float(jnp.max(jnp.abs(tau_posture))))

    # Report metrics
    print(f"Steps: {len(states)}")
    print(f"\nMaximum Torques:")
    print(f"  WBC (60% budget):       {max_wbc:.2f} Nm  (limit: 18.0 Nm)")
    print(f"  Momentum (20% budget):  {max_momentum:.2f} Nm  (limit: 6.0 Nm)")
    print(f"  Posture (20% budget):   {max_posture:.2f} Nm  (limit: 6.0 Nm)")
    print(f"  Total:                  {max_total:.2f} Nm  (limit: 30.0 Nm)")

    print(f"\nAverage Torques:")
    print(f"  WBC:       {np.mean(wbc_torques):.2f} Nm")
    print(f"  Momentum:  {np.mean(momentum_torques):.2f} Nm")
    print(f"  Posture:   {np.mean(posture_torques):.2f} Nm")

    print(f"\nBudget Utilization:")
    print(f"  WBC:       {max_wbc/18.0*100:.1f}% of 60% budget")
    print(f"  Momentum:  {max_momentum/6.0*100:.1f}% of 20% budget")
    print(f"  Posture:   {max_posture/6.0*100:.1f}% of 20% budget")
    print(f"  Total:     {max_total/30.0*100:.1f}% of 100% budget")

    # Check compliance
    budget_ok = max_wbc <= 18.0 and max_momentum <= 6.0 and max_posture <= 6.0 and max_total <= 30.0
    status = "[PASS]" if budget_ok else "[FAIL]"
    print(f"\nBudget Compliance: {status}")

    return budget_ok


def main():
    print("="*80)
    print("Hierarchical Controller Practical Evaluation")
    print("="*80)

    # Initialize controllers
    print("\nInitializing controllers...")
    capture_estimator, wbc_controller, momentum_coordinator, posture_regularizer = initialize_controllers()
    print("[OK] All controllers initialized")

    # Scenario 1: Standing balance at nominal height
    print("\n" + "="*80)
    print("Preparing Scenario 1: Standing Balance (h=0.60m)")
    print("="*80)

    states_1 = [create_mock_state(0.60) for _ in range(20)]
    joint_pos_1 = [jnp.zeros(10) for _ in range(20)]
    height_cmds_1 = [0.60 for _ in range(20)]

    result_1 = evaluate_scenario(
        "Standing Balance at h=0.60m",
        states_1, joint_pos_1, height_cmds_1,
        capture_estimator, wbc_controller, momentum_coordinator, posture_regularizer
    )

    # Scenario 2: Height tracking (squat down)
    print("\n" + "="*80)
    print("Preparing Scenario 2: Height Tracking (0.60m -> 0.45m)")
    print("="*80)

    heights = jnp.linspace(0.60, 0.45, 20)
    states_2 = [create_mock_state(float(h), jnp.array([0.0, 0.0, -0.01])) for h in heights]
    joint_pos_2 = [jnp.zeros(10) for _ in range(20)]
    height_cmds_2 = [0.45 for _ in range(20)]

    result_2 = evaluate_scenario(
        "Height Tracking (Squat Down)",
        states_2, joint_pos_2, height_cmds_2,
        capture_estimator, wbc_controller, momentum_coordinator, posture_regularizer
    )

    # Scenario 3: Lateral disturbance
    print("\n" + "="*80)
    print("Preparing Scenario 3: Lateral Disturbance Recovery")
    print("="*80)

    states_3 = []
    for i in range(20):
        # Simulate lateral velocity disturbance at step 5
        if i < 5:
            vel = jnp.zeros(3)
        elif i < 10:
            vel = jnp.array([0.0, 0.2, 0.0])  # Lateral velocity
        else:
            vel = jnp.array([0.0, 0.1, 0.0])  # Decaying
        states_3.append(create_mock_state(0.60, vel))

    joint_pos_3 = [jnp.zeros(10) for _ in range(20)]
    height_cmds_3 = [0.60 for _ in range(20)]

    result_3 = evaluate_scenario(
        "Lateral Disturbance Recovery",
        states_3, joint_pos_3, height_cmds_3,
        capture_estimator, wbc_controller, momentum_coordinator, posture_regularizer
    )

    # Scenario 4: Posture deviation
    print("\n" + "="*80)
    print("Preparing Scenario 4: Posture Deviation Correction")
    print("="*80)

    states_4 = [create_mock_state(0.60) for _ in range(20)]
    # Simulate joint position deviations
    joint_pos_4 = [jnp.array([0.1, 0.05, 0.15, 0.2, 0.0, 0.1, 0.05, 0.15, 0.2, 0.0]) for _ in range(20)]
    height_cmds_4 = [0.60 for _ in range(20)]

    result_4 = evaluate_scenario(
        "Posture Deviation Correction",
        states_4, joint_pos_4, height_cmds_4,
        capture_estimator, wbc_controller, momentum_coordinator, posture_regularizer
    )

    # Summary
    print("\n" + "="*80)
    print("Evaluation Summary")
    print("="*80)

    all_passed = result_1 and result_2 and result_3 and result_4

    print(f"\nScenario Results:")
    print(f"  1. Standing Balance:           {'[PASS]' if result_1 else '[FAIL]'}")
    print(f"  2. Height Tracking:            {'[PASS]' if result_2 else '[FAIL]'}")
    print(f"  3. Lateral Disturbance:        {'[PASS]' if result_3 else '[FAIL]'}")
    print(f"  4. Posture Deviation:          {'[PASS]' if result_4 else '[FAIL]'}")

    print(f"\nOverall Status: {'[ALL SCENARIOS PASSED]' if all_passed else '[SOME SCENARIOS FAILED]'}")

    print("\n" + "="*80)
    print("Controller Architecture:")
    print("  Phase 1: State Estimation")
    print("  Phase 2: Centroidal WBC (60% authority = 18 Nm)")
    print("  Phase 3: Momentum Coordinator (20% authority = 6 Nm)")
    print("  Phase 4: Posture Regularizer (20% authority = 6 Nm)")
    print("  Total: 100% authority = 30 Nm")
    print("="*80)


if __name__ == "__main__":
    main()
