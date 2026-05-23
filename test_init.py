"""Manual test for StaticBalanceController initialization.

Tests that equilibrium reference computation completes without errors
and produces reasonable torque values.
"""

import mujoco
import numpy as np

from wheeled_biped.controllers.static_balance_controller import StaticBalanceController


class MockWBC:
    """Mock WBC pipeline for testing."""

    def compute_wbc_torque(self, mj_data, obs, state, height_cmd, hip_roll_authority_scale):
        """Return zero torques."""
        return np.zeros(10)


def test_static_balance_controller_init():
    """Test StaticBalanceController initialization."""
    print("\n" + "="*80)
    print("TESTING STATIC BALANCE CONTROLLER INITIALIZATION")
    print("="*80 + "\n")

    # Load model
    model_path = "assets/robot/wheeled_biped_real.xml"
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

    print(f"[SETUP] Model loaded: {model_path}")
    print(f"  Robot mass: {np.sum(mj_model.body_mass):.2f} kg")
    print(f"  Gravity: {mj_model.opt.gravity[2]:.2f} m/s²")

    # Reset to keyframe 0 (standing pose)
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)

    print(f"\n[SETUP] Initial configuration set from keyframe 0")
    print(f"  Root height: {mj_data.qpos[2]:.6f} m")
    print(f"  Joint positions: {mj_data.qpos[7:17]}")

    # Create mock WBC pipeline
    wbc_pipeline = MockWBC()

    print(f"\n[SETUP] Mock WBC pipeline created")

    # Create StaticBalanceController (this triggers initialization)
    print(f"\n[TEST] Creating StaticBalanceController...")
    try:
        controller = StaticBalanceController(
            mj_model=mj_model,
            mj_data=mj_data,
            wbc_pipeline=wbc_pipeline,
            calibration_config={},
        )
        print(f"\n[TEST] [PASS] StaticBalanceController created successfully")
    except Exception as e:
        print(f"\n[TEST] [FAIL] StaticBalanceController creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Verify references were computed
    print(f"\n[VERIFY] Checking computed references...")

    if controller.tau_static_ref is None:
        print(f"  [FAIL] tau_static_ref is None")
        return False
    print(f"  [PASS] tau_static_ref computed: shape={controller.tau_static_ref.shape}")

    if controller.tau_wbc_equilibrium is None:
        print(f"  [FAIL] tau_wbc_equilibrium is None")
        return False
    print(f"  [PASS] tau_wbc_equilibrium computed: shape={controller.tau_wbc_equilibrium.shape}")

    if controller.equilibrium_state is None:
        print(f"  [FAIL] equilibrium_state is None")
        return False
    print(f"  [PASS] equilibrium_state captured")

    # Check torque magnitudes are reasonable
    print(f"\n[VERIFY] Checking torque magnitudes...")

    max_static = np.max(np.abs(controller.tau_static_ref))
    print(f"  Max |tau_static_ref|: {max_static:.4f} Nm")
    if max_static > 100.0:
        print(f"    Warning: Very large static torque (>{100.0} Nm)")

    max_wbc = np.max(np.abs(controller.tau_wbc_equilibrium))
    print(f"  Max |tau_wbc_equilibrium|: {max_wbc:.4f} Nm")
    if max_wbc > 100.0:
        print(f"    Warning: Very large WBC torque (>{100.0} Nm)")

    bias = controller.tau_wbc_equilibrium - controller.tau_static_ref
    max_bias = np.max(np.abs(bias))
    print(f"  Max |equilibrium_bias|: {max_bias:.4f} Nm")

    # Check support joints specifically (indices [2, 3, 7, 8])
    support_indices = [2, 3, 7, 8]
    print(f"\n[VERIFY] Support joint torques (indices {support_indices}):")
    print(f"  tau_static_ref: {controller.tau_static_ref[support_indices]}")
    print(f"  tau_wbc_equilibrium: {controller.tau_wbc_equilibrium[support_indices]}")
    print(f"  bias: {bias[support_indices]}")

    print(f"\n" + "="*80)
    print("TEST PASSED: Initialization completed successfully")
    print("="*80 + "\n")

    return True


if __name__ == "__main__":
    success = test_static_balance_controller_init()
    exit(0 if success else 1)
