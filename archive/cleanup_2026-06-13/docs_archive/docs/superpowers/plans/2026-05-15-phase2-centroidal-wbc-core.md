# Phase 2: Centroidal WBC Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Centroidal WBC with CoM regulation, capture point tracking, and 60% authority budget to achieve >20s survival time.

**Architecture:** Three-level hierarchy (60% WBC, 20% momentum, 20% posture) with integrated centroidal dynamics. Phase 2 focuses on Level 1 (Centroidal WBC) with CoM deadband control, height-dependent capture point tracking, and existing roll/height stabilization.

**Tech Stack:** Python, JAX, MuJoCo MJX, pytest

---

## File Structure

This phase creates the following new files:

- `wheeled_biped/controllers/centroidal_balance_controller.py` - Main controller with WBC torque computation
- `configs/controllers/step5_26_candidate_1_com.yaml` - Config for Candidate 1 (CoM only)
- `configs/controllers/step5_26_candidate_2_com_cp.yaml` - Config for Candidate 2 (CoM + CP)
- `tests/test_centroidal_balance_controller.py` - Unit tests for controller

This phase modifies:

- None (all new files)

---

## Task 1: Create CentroidalBalanceController Skeleton

**Files:**
- Create: `wheeled_biped/controllers/centroidal_balance_controller.py`
- Test: `tests/test_centroidal_balance_controller.py`

- [ ] **Step 1: Write the failing test for controller creation**

```python
# tests/test_centroidal_balance_controller.py
import jax.numpy as jnp
import pytest
from wheeled_biped.controllers.centroidal_balance_controller import (
    CentroidalBalanceController,
    CentroidalBalanceConfig,
)


def test_centroidal_balance_controller_creation():
    """Test CentroidalBalanceController can be created with config."""
    config = CentroidalBalanceConfig(
        # Roll stabilization
        k_roll=20.0,
        k_roll_rate=4.0,
        
        # CoM regulation
        k_com_lateral=15.0,
        k_com_lateral_damping=3.0,
        k_com_sagittal=10.0,
        k_com_sagittal_damping=2.0,
        
        # Deadbands
        com_deadband_lateral=0.02,
        com_deadband_sagittal=0.03,
        
        # Authority budget
        wbc_authority_budget=0.6,
    )
    controller = CentroidalBalanceController(config)
    
    assert controller.config.k_roll == 20.0
    assert controller.config.wbc_authority_budget == 0.6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_centroidal_balance_controller.py::test_centroidal_balance_controller_creation -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'wheeled_biped.controllers.centroidal_balance_controller'"

- [ ] **Step 3: Write minimal controller skeleton**

```python
# wheeled_biped/controllers/centroidal_balance_controller.py
"""Centroidal balance controller with integrated CoM and capture point tracking."""

import chex
import jax.numpy as jnp
from jax import Array


@chex.dataclass
class CentroidalBalanceConfig:
    """Configuration for centroidal balance controller."""
    # Roll stabilization (from Step 5.25)
    k_roll: float = 20.0
    k_roll_rate: float = 4.0
    
    # CoM regulation
    k_com_lateral: float = 15.0
    k_com_lateral_damping: float = 3.0
    k_com_sagittal: float = 10.0
    k_com_sagittal_damping: float = 2.0
    
    # Deadbands
    com_deadband_lateral: float = 0.02  # meters
    com_deadband_sagittal: float = 0.03  # meters
    
    # Authority budget
    wbc_authority_budget: float = 0.6  # 60% of actuator range


class CentroidalBalanceController:
    """Centroidal WBC with CoM regulation and capture point tracking."""
    
    def __init__(self, config: CentroidalBalanceConfig):
        self.config = config
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_centroidal_balance_controller.py::test_centroidal_balance_controller_creation -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add wheeled_biped/controllers/centroidal_balance_controller.py tests/test_centroidal_balance_controller.py
git commit -m "feat: add CentroidalBalanceController skeleton"
```

---

## Task 2: Implement Roll Stabilization Torque

**Files:**
- Modify: `wheeled_biped/controllers/centroidal_balance_controller.py`
- Modify: `tests/test_centroidal_balance_controller.py`

- [ ] **Step 1: Write the failing test for roll stabilization**

```python
# tests/test_centroidal_balance_controller.py (add to existing file)

def test_roll_stabilization_torque():
    """Test roll stabilization torque computation."""
    config = CentroidalBalanceConfig(
        k_roll=20.0,
        k_roll_rate=4.0,
    )
    controller = CentroidalBalanceController(config)
    
    # Mock observation with roll error and roll rate
    obs = jnp.zeros(42)
    obs = obs.at[3].set(0.1)  # roll = 0.1 rad
    obs = obs.at[10].set(0.05)  # roll_rate = 0.05 rad/s
    
    tau_roll = controller.compute_roll_stabilization_torque(obs)
    
    # Expected: tau = -k_roll * roll - k_roll_rate * roll_rate
    # tau = -20.0 * 0.1 - 4.0 * 0.05 = -2.0 - 0.2 = -2.2
    expected_hip_roll_torque = -2.2
    
    # Roll torque should be applied to hip roll joints (indices 0 and 5)
    assert jnp.allclose(tau_roll[0], expected_hip_roll_torque, atol=1e-6)
    assert jnp.allclose(tau_roll[5], expected_hip_roll_torque, atol=1e-6)
    
    # Other joints should be zero
    assert jnp.allclose(tau_roll[1:5], 0.0, atol=1e-6)
    assert jnp.allclose(tau_roll[6:10], 0.0, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_centroidal_balance_controller.py::test_roll_stabilization_torque -v`
Expected: FAIL with "AttributeError: 'CentroidalBalanceController' object has no attribute 'compute_roll_stabilization_torque'"

- [ ] **Step 3: Implement roll stabilization torque computation**

```python
# wheeled_biped/controllers/centroidal_balance_controller.py
# Add this method to CentroidalBalanceController class

def compute_roll_stabilization_torque(self, obs: Array) -> Array:
    """Compute roll stabilization torque for hip roll joints.
    
    Args:
        obs: Observation array with roll at index 3, roll_rate at index 10
        
    Returns:
        Torque array (10,) with roll correction on hip roll joints
    """
    # Extract roll state from observation
    roll = obs[3]
    roll_rate = obs[10]
    
    # PD control: tau = -k_p * error - k_d * error_rate
    tau_hip_roll = -self.config.k_roll * roll - self.config.k_roll_rate * roll_rate
    
    # Apply to both hip roll joints (symmetric)
    tau = jnp.zeros(10)
    tau = tau.at[0].set(tau_hip_roll)  # left hip roll
    tau = tau.at[5].set(tau_hip_roll)  # right hip roll
    
    return tau
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_centroidal_balance_controller.py::test_roll_stabilization_torque -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add wheeled_biped/controllers/centroidal_balance_controller.py tests/test_centroidal_balance_controller.py
git commit -m "feat: add roll stabilization torque computation"
```

---

## Task 3: Implement CoM Regulation with Deadband Control

**Files:**
- Modify: `wheeled_biped/controllers/centroidal_balance_controller.py`
- Modify: `tests/test_centroidal_balance_controller.py`

- [ ] **Step 1: Write the failing test for CoM regulation**

```python
# tests/test_centroidal_balance_controller.py (add to existing file)
from wheeled_biped.controllers.centroidal_state_estimator import CentroidalState


def test_com_regulation_torque_outside_deadband():
    """Test CoM regulation torque when error exceeds deadband."""
    config = CentroidalBalanceConfig(
        k_com_lateral=15.0,
        k_com_lateral_damping=3.0,
        k_com_sagittal=10.0,
        k_com_sagittal_damping=2.0,
        com_deadband_lateral=0.02,
        com_deadband_sagittal=0.03,
    )
    controller = CentroidalBalanceController(config)
    
    # CoM error outside deadband
    state = CentroidalState(
        com_pos=jnp.array([0.05, 0.04, 0.6]),  # x=5cm, y=4cm (both outside deadband)
        com_vel=jnp.array([0.1, 0.05, 0.0]),
        capture_point=jnp.zeros(2),
        divergence=jnp.zeros(2),
        linear_momentum=jnp.zeros(3),
        angular_momentum=jnp.zeros(3),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=75.0,
        right_wheel_force=75.0,
    )
    
    tau_com = controller.compute_com_regulation_torque(state)
    
    # Lateral error (y=0.04m) should produce hip roll torque
    # tau_lateral = -k_com_lateral * error - k_com_lateral_damping * vel
    # tau_lateral = -15.0 * 0.04 - 3.0 * 0.05 = -0.6 - 0.15 = -0.75
    assert jnp.abs(tau_com[0]) > 0.5  # left hip roll should have significant torque
    assert jnp.abs(tau_com[5]) > 0.5  # right hip roll should have significant torque
    
    # Sagittal error (x=0.05m) should produce wheel torque
    # tau_sagittal = -k_com_sagittal * error - k_com_sagittal_damping * vel
    # tau_sagittal = -10.0 * 0.05 - 2.0 * 0.1 = -0.5 - 0.2 = -0.7
    assert jnp.abs(tau_com[4]) > 0.5  # left wheel should have significant torque
    assert jnp.abs(tau_com[9]) > 0.5  # right wheel should have significant torque


def test_com_regulation_torque_inside_deadband():
    """Test CoM regulation torque is zero when error inside deadband."""
    config = CentroidalBalanceConfig(
        k_com_lateral=15.0,
        k_com_lateral_damping=3.0,
        k_com_sagittal=10.0,
        k_com_sagittal_damping=2.0,
        com_deadband_lateral=0.02,
        com_deadband_sagittal=0.03,
    )
    controller = CentroidalBalanceController(config)
    
    # CoM error inside deadband
    state = CentroidalState(
        com_pos=jnp.array([0.01, 0.01, 0.6]),  # x=1cm, y=1cm (both inside deadband)
        com_vel=jnp.array([0.0, 0.0, 0.0]),
        capture_point=jnp.zeros(2),
        divergence=jnp.zeros(2),
        linear_momentum=jnp.zeros(3),
        angular_momentum=jnp.zeros(3),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=75.0,
        right_wheel_force=75.0,
    )
    
    tau_com = controller.compute_com_regulation_torque(state)
    
    # All torques should be zero (within deadband)
    assert jnp.allclose(tau_com, 0.0, atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_centroidal_balance_controller.py::test_com_regulation_torque_outside_deadband tests/test_centroidal_balance_controller.py::test_com_regulation_torque_inside_deadband -v`
Expected: FAIL with "AttributeError: 'CentroidalBalanceController' object has no attribute 'compute_com_regulation_torque'"

- [ ] **Step 3: Implement CoM regulation torque computation**

```python
# wheeled_biped/controllers/centroidal_balance_controller.py
# Add these imports at the top
from wheeled_biped.controllers.centroidal_state_estimator import CentroidalState

# Add this method to CentroidalBalanceController class

def compute_com_regulation_torque(self, state: CentroidalState) -> Array:
    """Compute CoM regulation torque with deadband control.
    
    Args:
        state: CentroidalState with com_pos and com_vel
        
    Returns:
        Torque array (10,) with CoM correction on hip roll and wheels
    """
    # Extract CoM position and velocity
    com_x = state.com_pos[0]  # sagittal (forward)
    com_y = state.com_pos[1]  # lateral (sideways)
    com_vx = state.com_vel[0]
    com_vy = state.com_vel[1]
    
    # Apply deadband to lateral error
    if jnp.abs(com_y) < self.config.com_deadband_lateral:
        com_y_error = 0.0
    else:
        com_y_error = com_y
    
    # Apply deadband to sagittal error
    if jnp.abs(com_x) < self.config.com_deadband_sagittal:
        com_x_error = 0.0
    else:
        com_x_error = com_x
    
    # Lateral CoM error → hip roll torques (symmetric)
    tau_lateral = -self.config.k_com_lateral * com_y_error - self.config.k_com_lateral_damping * com_vy
    
    # Sagittal CoM error → wheel torques (common mode)
    tau_sagittal = -self.config.k_com_sagittal * com_x_error - self.config.k_com_sagittal_damping * com_vx
    
    # Build torque vector
    tau = jnp.zeros(10)
    tau = tau.at[0].set(tau_lateral)  # left hip roll
    tau = tau.at[5].set(tau_lateral)  # right hip roll
    tau = tau.at[4].set(tau_sagittal)  # left wheel
    tau = tau.at[9].set(tau_sagittal)  # right wheel
    
    return tau
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_centroidal_balance_controller.py::test_com_regulation_torque_outside_deadband tests/test_centroidal_balance_controller.py::test_com_regulation_torque_inside_deadband -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add wheeled_biped/controllers/centroidal_balance_controller.py tests/test_centroidal_balance_controller.py
git commit -m "feat: add CoM regulation with deadband control"
```

---

## Task 4: Implement Capture Point Tracking Torque

**Files:**
- Modify: `wheeled_biped/controllers/centroidal_balance_controller.py`
- Modify: `tests/test_centroidal_balance_controller.py`

- [ ] **Step 1: Add capture point config to CentroidalBalanceConfig**

```python
# wheeled_biped/controllers/centroidal_balance_controller.py
# Update CentroidalBalanceConfig dataclass

@chex.dataclass
class CentroidalBalanceConfig:
    """Configuration for centroidal balance controller."""
    # Roll stabilization (from Step 5.25)
    k_roll: float = 20.0
    k_roll_rate: float = 4.0
    
    # CoM regulation
    k_com_lateral: float = 15.0
    k_com_lateral_damping: float = 3.0
    k_com_sagittal: float = 10.0
    k_com_sagittal_damping: float = 2.0
    
    # Capture point tracking (NEW)
    k_cp_lateral: float = 25.0
    k_cp_sagittal: float = 20.0
    k_cp_wheel_diff: float = 8.0
    
    # Deadbands
    com_deadband_lateral: float = 0.02  # meters
    com_deadband_sagittal: float = 0.03  # meters
    cp_deadband: float = 0.05  # meters
    
    # Authority budget
    wbc_authority_budget: float = 0.6  # 60% of actuator range
```

- [ ] **Step 2: Write the failing test for capture point tracking**

```python
# tests/test_centroidal_balance_controller.py (add to existing file)
from wheeled_biped.controllers.capture_point_estimator import (
    CapturePointEstimator,
    CapturePointEstimatorConfig,
)


def test_capture_point_tracking_torque_outside_deadband():
    """Test capture point tracking torque when error exceeds deadband."""
    config = CentroidalBalanceConfig(
        k_cp_lateral=25.0,
        k_cp_sagittal=20.0,
        k_cp_wheel_diff=8.0,
        cp_deadband=0.05,
    )
    controller = CentroidalBalanceController(config)
    
    # Capture point error outside deadband
    state = CentroidalState(
        com_pos=jnp.array([0.0, 0.0, 0.6]),
        com_vel=jnp.array([0.0, 0.0, 0.0]),
        capture_point=jnp.array([0.10, 0.08]),  # 10cm forward, 8cm lateral
        divergence=jnp.array([0.10, 0.08]),
        linear_momentum=jnp.zeros(3),
        angular_momentum=jnp.zeros(3),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=75.0,
        right_wheel_force=75.0,
    )
    
    tau_cp = controller.compute_capture_point_tracking_torque(state)
    
    # Lateral divergence should produce hip roll torque
    assert jnp.abs(tau_cp[0]) > 1.0  # left hip roll
    assert jnp.abs(tau_cp[5]) > 1.0  # right hip roll
    
    # Sagittal divergence should produce wheel torque
    assert jnp.abs(tau_cp[4]) > 1.0  # left wheel
    assert jnp.abs(tau_cp[9]) > 1.0  # right wheel


def test_capture_point_tracking_torque_inside_deadband():
    """Test capture point tracking torque is zero when error inside deadband."""
    config = CentroidalBalanceConfig(
        k_cp_lateral=25.0,
        k_cp_sagittal=20.0,
        cp_deadband=0.05,
    )
    controller = CentroidalBalanceController(config)
    
    # Capture point error inside deadband
    state = CentroidalState(
        com_pos=jnp.array([0.0, 0.0, 0.6]),
        com_vel=jnp.array([0.0, 0.0, 0.0]),
        capture_point=jnp.array([0.02, 0.03]),  # 2cm forward, 3cm lateral (inside 5cm deadband)
        divergence=jnp.array([0.02, 0.03]),
        linear_momentum=jnp.zeros(3),
        angular_momentum=jnp.zeros(3),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=75.0,
        right_wheel_force=75.0,
    )
    
    tau_cp = controller.compute_capture_point_tracking_torque(state)
    
    # All torques should be zero (within deadband)
    assert jnp.allclose(tau_cp, 0.0, atol=1e-6)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_centroidal_balance_controller.py::test_capture_point_tracking_torque_outside_deadband tests/test_centroidal_balance_controller.py::test_capture_point_tracking_torque_inside_deadband -v`
Expected: FAIL with "AttributeError: 'CentroidalBalanceController' object has no attribute 'compute_capture_point_tracking_torque'"

- [ ] **Step 4: Implement capture point tracking torque computation**

```python
# wheeled_biped/controllers/centroidal_balance_controller.py
# Add this method to CentroidalBalanceController class

def compute_capture_point_tracking_torque(self, state: CentroidalState) -> Array:
    """Compute capture point tracking torque with deadband control.
    
    Args:
        state: CentroidalState with capture_point and divergence
        
    Returns:
        Torque array (10,) with CP correction on hip roll and wheels
    """
    # Extract capture point error (divergence from support center)
    cp_x = state.capture_point[0]  # sagittal
    cp_y = state.capture_point[1]  # lateral
    
    # Compute magnitude for deadband check
    cp_error_mag = jnp.sqrt(cp_x**2 + cp_y**2)
    
    # Apply deadband
    if cp_error_mag < self.config.cp_deadband:
        return jnp.zeros(10)
    
    # Lateral divergence → hip roll torques (asymmetric for differential correction)
    tau_lateral = -self.config.k_cp_lateral * cp_y
    
    # Sagittal divergence → wheel torques (common mode)
    tau_sagittal = -self.config.k_cp_sagittal * cp_x
    
    # Additional wheel differential for lateral correction
    tau_wheel_diff = -self.config.k_cp_wheel_diff * cp_y
    
    # Build torque vector
    tau = jnp.zeros(10)
    tau = tau.at[0].set(tau_lateral)  # left hip roll
    tau = tau.at[5].set(tau_lateral)  # right hip roll
    tau = tau.at[4].set(tau_sagittal + tau_wheel_diff)  # left wheel
    tau = tau.at[9].set(tau_sagittal - tau_wheel_diff)  # right wheel (opposite for differential)
    
    return tau
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_centroidal_balance_controller.py::test_capture_point_tracking_torque_outside_deadband tests/test_centroidal_balance_controller.py::test_capture_point_tracking_torque_inside_deadband -v`
Expected: PASS (both tests)

- [ ] **Step 6: Commit**

```bash
git add wheeled_biped/controllers/centroidal_balance_controller.py tests/test_centroidal_balance_controller.py
git commit -m "feat: add capture point tracking with deadband control"
```

---

## Task 5: Implement Height Tracking Torque

**Files:**
- Modify: `wheeled_biped/controllers/centroidal_balance_controller.py`
- Modify: `tests/test_centroidal_balance_controller.py`

- [ ] **Step 1: Add height tracking config to CentroidalBalanceConfig**

```python
# wheeled_biped/controllers/centroidal_balance_controller.py
# Update CentroidalBalanceConfig dataclass

@chex.dataclass
class CentroidalBalanceConfig:
    """Configuration for centroidal balance controller."""
    # Roll stabilization (from Step 5.25)
    k_roll: float = 20.0
    k_roll_rate: float = 4.0
    
    # CoM regulation
    k_com_lateral: float = 15.0
    k_com_lateral_damping: float = 3.0
    k_com_sagittal: float = 10.0
    k_com_sagittal_damping: float = 2.0
    
    # Capture point tracking
    k_cp_lateral: float = 25.0
    k_cp_sagittal: float = 20.0
    k_cp_wheel_diff: float = 8.0
    
    # Height tracking (NEW)
    k_height: float = 5.0
    
    # Deadbands
    com_deadband_lateral: float = 0.02  # meters
    com_deadband_sagittal: float = 0.03  # meters
    cp_deadband: float = 0.05  # meters
    
    # Authority budget
    wbc_authority_budget: float = 0.6  # 60% of actuator range
```

- [ ] **Step 2: Write the failing test for height tracking**

```python
# tests/test_centroidal_balance_controller.py (add to existing file)

def test_height_tracking_torque():
    """Test height tracking torque computation."""
    config = CentroidalBalanceConfig(
        k_height=5.0,
    )
    controller = CentroidalBalanceController(config)
    
    # Mock observation with height command and current height
    obs = jnp.zeros(42)
    obs = obs.at[39].set(0.65)  # height_cmd = 0.65m
    
    # Mock state with current height
    state = CentroidalState(
        com_pos=jnp.array([0.0, 0.0, 0.60]),  # current height = 0.60m
        com_vel=jnp.zeros(3),
        capture_point=jnp.zeros(2),
        divergence=jnp.zeros(2),
        linear_momentum=jnp.zeros(3),
        angular_momentum=jnp.zeros(3),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=75.0,
        right_wheel_force=75.0,
    )
    
    tau_height = controller.compute_height_tracking_torque(obs, state)
    
    # Height error = 0.65 - 0.60 = 0.05m (need to extend legs)
    # Should produce hip pitch and knee torques
    assert jnp.abs(tau_height[2]) > 0.1  # left hip pitch
    assert jnp.abs(tau_height[3]) > 0.1  # left knee
    assert jnp.abs(tau_height[7]) > 0.1  # right hip pitch
    assert jnp.abs(tau_height[8]) > 0.1  # right knee
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_centroidal_balance_controller.py::test_height_tracking_torque -v`
Expected: FAIL with "AttributeError: 'CentroidalBalanceController' object has no attribute 'compute_height_tracking_torque'"

- [ ] **Step 4: Implement height tracking torque computation**

```python
# wheeled_biped/controllers/centroidal_balance_controller.py
# Add this method to CentroidalBalanceController class

def compute_height_tracking_torque(self, obs: Array, state: CentroidalState) -> Array:
    """Compute height tracking torque using simplified IK.
    
    Args:
        obs: Observation array with height_cmd at index 39
        state: CentroidalState with current CoM height
        
    Returns:
        Torque array (10,) with height correction on hip pitch and knee
    """
    # Extract height command and current height
    height_cmd = obs[39]
    height_current = state.com_pos[2]
    
    # Height error
    height_error = height_cmd - height_current
    
    # Simplified IK: distribute height correction to hip pitch and knee
    # Positive error (need to extend) → negative hip pitch, positive knee
    # This is a simplified mapping; real IK would be more sophisticated
    tau_hip_pitch = -self.config.k_height * height_error
    tau_knee = self.config.k_height * height_error * 1.5  # knee has more leverage
    
    # Build torque vector
    tau = jnp.zeros(10)
    tau = tau.at[2].set(tau_hip_pitch)  # left hip pitch
    tau = tau.at[3].set(tau_knee)  # left knee
    tau = tau.at[7].set(tau_hip_pitch)  # right hip pitch
    tau = tau.at[8].set(tau_knee)  # right knee
    
    return tau
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_centroidal_balance_controller.py::test_height_tracking_torque -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add wheeled_biped/controllers/centroidal_balance_controller.py tests/test_centroidal_balance_controller.py
git commit -m "feat: add height tracking torque computation"
```

---

## Task 6: Implement WBC Authority Budget Clipping

**Files:**
- Modify: `wheeled_biped/controllers/centroidal_balance_controller.py`
- Modify: `tests/test_centroidal_balance_controller.py`

- [ ] **Step 1: Write the failing test for authority budget clipping**

```python
# tests/test_centroidal_balance_controller.py (add to existing file)

def test_wbc_authority_budget_clipping():
    """Test WBC torque is clipped to 60% authority budget."""
    config = CentroidalBalanceConfig(
        wbc_authority_budget=0.6,
    )
    controller = CentroidalBalanceController(config)
    
    # Create large torque vector that exceeds budget
    tau_desired = jnp.array([10.0, 8.0, 12.0, 15.0, 20.0, 10.0, 8.0, 12.0, 15.0, 20.0])
    
    tau_clipped = controller.clip_to_authority_budget(tau_desired, budget=0.6)
    
    # Maximum torque should be scaled to 60% of max actuator range
    # Assuming max actuator torque is ~30 Nm, 60% budget = 18 Nm
    # The largest torque (20.0) should be scaled down
    assert jnp.max(jnp.abs(tau_clipped)) <= 18.0
    
    # Relative proportions should be preserved
    ratio_original = tau_desired[0] / tau_desired[4]
    ratio_clipped = tau_clipped[0] / tau_clipped[4]
    assert jnp.allclose(ratio_original, ratio_clipped, rtol=0.01)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_centroidal_balance_controller.py::test_wbc_authority_budget_clipping -v`
Expected: FAIL with "AttributeError: 'CentroidalBalanceController' object has no attribute 'clip_to_authority_budget'"

- [ ] **Step 3: Implement authority budget clipping**

```python
# wheeled_biped/controllers/centroidal_balance_controller.py
# Add this method to CentroidalBalanceController class

def clip_to_authority_budget(self, tau: Array, budget: float) -> Array:
    """Clip torque vector to authority budget while preserving proportions.
    
    Args:
        tau: Desired torque array (10,)
        budget: Authority budget as fraction of max actuator range (0.0-1.0)
        
    Returns:
        Clipped torque array (10,) scaled to fit within budget
    """
    # Assume max actuator torque is 30 Nm (typical for this robot)
    max_actuator_torque = 30.0
    
    # Compute budget limit
    budget_limit = budget * max_actuator_torque
    
    # Find maximum absolute torque
    max_tau = jnp.max(jnp.abs(tau))
    
    # If within budget, return as-is
    if max_tau <= budget_limit:
        return tau
    
    # Scale down to fit within budget while preserving proportions
    scale_factor = budget_limit / max_tau
    tau_clipped = tau * scale_factor
    
    return tau_clipped
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_centroidal_balance_controller.py::test_wbc_authority_budget_clipping -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add wheeled_biped/controllers/centroidal_balance_controller.py tests/test_centroidal_balance_controller.py
git commit -m "feat: add WBC authority budget clipping"
```

---

## Task 7: Implement Hierarchical WBC Fusion

**Files:**
- Modify: `wheeled_biped/controllers/centroidal_balance_controller.py`
- Modify: `tests/test_centroidal_balance_controller.py`

- [ ] **Step 1: Add task weights to config**

```python
# wheeled_biped/controllers/centroidal_balance_controller.py
# Update CentroidalBalanceConfig dataclass

@chex.dataclass
class CentroidalBalanceConfig:
    """Configuration for centroidal balance controller."""
    # Roll stabilization (from Step 5.25)
    k_roll: float = 20.0
    k_roll_rate: float = 4.0
    
    # CoM regulation
    k_com_lateral: float = 15.0
    k_com_lateral_damping: float = 3.0
    k_com_sagittal: float = 10.0
    k_com_sagittal_damping: float = 2.0
    
    # Capture point tracking
    k_cp_lateral: float = 25.0
    k_cp_sagittal: float = 20.0
    k_cp_wheel_diff: float = 8.0
    
    # Height tracking
    k_height: float = 5.0
    
    # Task weights (NEW)
    w_roll: float = 1.0
    w_com: float = 0.8
    w_cp: float = 1.2  # Highest priority - divergence is critical
    w_height: float = 0.6
    
    # Deadbands
    com_deadband_lateral: float = 0.02  # meters
    com_deadband_sagittal: float = 0.03  # meters
    cp_deadband: float = 0.05  # meters
    
    # Authority budget
    wbc_authority_budget: float = 0.6  # 60% of actuator range
```

- [ ] **Step 2: Write the failing test for hierarchical fusion**

```python
# tests/test_centroidal_balance_controller.py (add to existing file)

def test_hierarchical_wbc_fusion():
    """Test hierarchical fusion of all WBC components."""
    config = CentroidalBalanceConfig(
        k_roll=20.0,
        k_roll_rate=4.0,
        k_com_lateral=15.0,
        k_com_sagittal=10.0,
        k_cp_lateral=25.0,
        k_cp_sagittal=20.0,
        k_height=5.0,
        w_roll=1.0,
        w_com=0.8,
        w_cp=1.2,
        w_height=0.6,
        wbc_authority_budget=0.6,
    )
    controller = CentroidalBalanceController(config)
    
    # Mock observation with roll, height command
    obs = jnp.zeros(42)
    obs = obs.at[3].set(0.05)  # roll = 0.05 rad
    obs = obs.at[10].set(0.02)  # roll_rate = 0.02 rad/s
    obs = obs.at[39].set(0.65)  # height_cmd = 0.65m
    
    # Mock state with CoM error and capture point error
    state = CentroidalState(
        com_pos=jnp.array([0.04, 0.03, 0.60]),  # x=4cm, y=3cm, h=0.60m
        com_vel=jnp.array([0.05, 0.02, 0.0]),
        capture_point=jnp.array([0.08, 0.06]),  # 8cm forward, 6cm lateral
        divergence=jnp.array([0.08, 0.06]),
        linear_momentum=jnp.zeros(3),
        angular_momentum=jnp.zeros(3),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=75.0,
        right_wheel_force=75.0,
    )
    
    tau_wbc = controller.compute_centroidal_wbc_torque(obs, state)
    
    # Should produce non-zero torques on multiple joints
    assert jnp.any(jnp.abs(tau_wbc) > 0.1)
    
    # Should respect 60% authority budget
    assert jnp.max(jnp.abs(tau_wbc)) <= 18.0  # 60% of 30 Nm
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_centroidal_balance_controller.py::test_hierarchical_wbc_fusion -v`
Expected: FAIL with "AttributeError: 'CentroidalBalanceController' object has no attribute 'compute_centroidal_wbc_torque'"

- [ ] **Step 4: Implement hierarchical WBC fusion**

```python
# wheeled_biped/controllers/centroidal_balance_controller.py
# Add this method to CentroidalBalanceController class

def compute_centroidal_wbc_torque(self, obs: Array, state: CentroidalState) -> Array:
    """Compute integrated centroidal WBC torque with all objectives.
    
    Args:
        obs: Observation array
        state: CentroidalState with all centroidal quantities
        
    Returns:
        WBC torque array (10,) clipped to 60% authority budget
    """
    # Compute individual objective torques
    tau_roll = self.compute_roll_stabilization_torque(obs)
    tau_com = self.compute_com_regulation_torque(state)
    tau_cp = self.compute_capture_point_tracking_torque(state)
    tau_height = self.compute_height_tracking_torque(obs, state)
    
    # Weighted fusion
    tau_wbc_desired = (
        self.config.w_roll * tau_roll +
        self.config.w_com * tau_com +
        self.config.w_cp * tau_cp +
        self.config.w_height * tau_height
    )
    
    # Clip to 60% authority budget
    tau_wbc = self.clip_to_authority_budget(tau_wbc_desired, self.config.wbc_authority_budget)
    
    return tau_wbc
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_centroidal_balance_controller.py::test_hierarchical_wbc_fusion -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add wheeled_biped/controllers/centroidal_balance_controller.py tests/test_centroidal_balance_controller.py
git commit -m "feat: add hierarchical WBC fusion with task weights"
```

---

## Task 8: Add Integration Test for Controller

**Files:**
- Create: `tests/test_centroidal_integration.py` (if not exists, or add to existing)
- Modify: `tests/test_centroidal_balance_controller.py`

- [ ] **Step 1: Write the failing integration test**

```python
# tests/test_centroidal_balance_controller.py (add to existing file)
import mujoco
import mujoco.mjx as mjx
from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)
from wheeled_biped.controllers.capture_point_estimator import (
    CapturePointEstimator,
    CapturePointEstimatorConfig,
)


def test_controller_integration_no_nan_100_steps():
    """Integration test: 100-step rollout with full controller produces no NaN."""
    # Load robot model
    model_path = "assets/robot/wheeled_biped_real.xml"
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mjx_model = mjx.put_model(mj_model)
    
    data = mjx.make_data(mjx_model)
    data = mjx.forward(mjx_model, data)
    
    # Create controller and estimators
    controller_config = CentroidalBalanceConfig(
        k_roll=20.0,
        k_com_lateral=15.0,
        k_cp_lateral=25.0,
        k_height=5.0,
        wbc_authority_budget=0.6,
    )
    controller = CentroidalBalanceController(controller_config)
    
    estimator_config = CentroidalStateEstimatorConfig(
        robot_mass=15.0,
        torso_inertia=jnp.array([0.5, 0.5, 0.3]),
    )
    state_estimator = CentroidalStateEstimator(estimator_config)
    
    cp_config = CapturePointEstimatorConfig(gravity=9.81)
    cp_estimator = CapturePointEstimator(cp_config)
    
    # Run 100-step rollout
    prev_com_pos = None
    for step in range(100):
        # Mock observation
        obs = jnp.zeros(42)
        obs = obs.at[39].set(0.60)  # height_cmd
        
        # Estimate centroidal state
        centroidal_state, prev_com_pos = state_estimator.estimate(obs, data, prev_com_pos)
        centroidal_state = cp_estimator.update(centroidal_state)
        
        # Compute WBC torque
        tau_wbc = controller.compute_centroidal_wbc_torque(obs, centroidal_state)
        
        # Verify no NaN
        assert not jnp.any(jnp.isnan(tau_wbc)), f"NaN in tau_wbc at step {step}"
        
        # Step simulation (simplified - just forward)
        data = mjx.forward(mjx_model, data)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_centroidal_balance_controller.py::test_controller_integration_no_nan_100_steps -v`
Expected: FAIL (may fail due to missing imports or integration issues)

- [ ] **Step 3: Fix any integration issues**

If test fails due to missing imports or integration issues, fix them in the controller or test file.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_centroidal_balance_controller.py::test_controller_integration_no_nan_100_steps -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_centroidal_balance_controller.py
git commit -m "test: add 100-step integration test for centroidal controller"
```

---

## Task 9: Create Configuration Files for Ablation Candidates

**Files:**
- Create: `configs/controllers/step5_26_candidate_1_com.yaml`
- Create: `configs/controllers/step5_26_candidate_2_com_cp.yaml`

- [ ] **Step 1: Create Candidate 1 config (CoM only)**

```yaml
# configs/controllers/step5_26_candidate_1_com.yaml
# Candidate 1: CoM regulation only (no capture point tracking)

controller_type: centroidal_balance

# Roll stabilization (from Step 5.25)
k_roll: 20.0
k_roll_rate: 4.0

# CoM regulation (ENABLED)
k_com_lateral: 15.0
k_com_lateral_damping: 3.0
k_com_sagittal: 10.0
k_com_sagittal_damping: 2.0

# Capture point tracking (DISABLED for Candidate 1)
k_cp_lateral: 0.0
k_cp_sagittal: 0.0
k_cp_wheel_diff: 0.0

# Height tracking
k_height: 5.0

# Task weights
w_roll: 1.0
w_com: 0.8
w_cp: 0.0  # Disabled
w_height: 0.6

# Deadbands
com_deadband_lateral: 0.02  # meters
com_deadband_sagittal: 0.03  # meters
cp_deadband: 0.05  # meters

# Authority budget
wbc_authority_budget: 0.6  # 60% of actuator range

# Evaluation metadata
candidate_name: "Candidate 1: CoM Only"
candidate_description: "Roll + CoM regulation, no capture point tracking"
baseline_comparison: "Step 5.25 (0.87s survival)"
```

- [ ] **Step 2: Create Candidate 2 config (CoM + CP)**

```yaml
# configs/controllers/step5_26_candidate_2_com_cp.yaml
# Candidate 2: CoM regulation + Capture point tracking

controller_type: centroidal_balance

# Roll stabilization (from Step 5.25)
k_roll: 20.0
k_roll_rate: 4.0

# CoM regulation (ENABLED)
k_com_lateral: 15.0
k_com_lateral_damping: 3.0
k_com_sagittal: 10.0
k_com_sagittal_damping: 2.0

# Capture point tracking (ENABLED)
k_cp_lateral: 25.0
k_cp_sagittal: 20.0
k_cp_wheel_diff: 8.0

# Height tracking
k_height: 5.0

# Task weights
w_roll: 1.0
w_com: 0.8
w_cp: 1.2  # Highest priority
w_height: 0.6

# Deadbands
com_deadband_lateral: 0.02  # meters
com_deadband_sagittal: 0.03  # meters
cp_deadband: 0.05  # meters

# Authority budget
wbc_authority_budget: 0.6  # 60% of actuator range

# Evaluation metadata
candidate_name: "Candidate 2: CoM + Capture Point"
candidate_description: "Roll + CoM regulation + capture point tracking"
baseline_comparison: "Candidate 1 (CoM only)"
```

- [ ] **Step 3: Commit configuration files**

```bash
git add configs/controllers/step5_26_candidate_1_com.yaml configs/controllers/step5_26_candidate_2_com_cp.yaml
git commit -m "config: add Candidate 1 and 2 configs for Phase 2 ablation"
```

---

## Phase 2 Completion Checklist

Before proceeding to Phase 3, verify:

- [ ] All unit tests pass: `pytest tests/test_centroidal_balance_controller.py -v`
- [ ] Integration test passes: 100-step rollout produces no NaN
- [ ] Controller computes roll stabilization torque correctly
- [ ] CoM regulation respects deadband (zero torque inside, non-zero outside)
- [ ] Capture point tracking respects deadband
- [ ] Height tracking produces reasonable torques
- [ ] WBC authority budget clipping works (max 60% of actuator range)
- [ ] Hierarchical fusion combines all objectives with task weights
- [ ] Configuration files created for Candidate 1 and Candidate 2
- [ ] All files committed to git

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-15-phase2-centroidal-wbc-core.md`.

**Two execution options:**

**1. Subagent-Driven (recommended)** - Dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**

