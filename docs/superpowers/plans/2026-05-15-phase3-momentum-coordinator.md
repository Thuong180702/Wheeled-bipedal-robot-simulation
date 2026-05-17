# Phase 3: Momentum Coordinator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Momentum Coordinator with momentum damping, feedforward compensation, and contact-aware recovery to achieve Level 2 stabilization with 20% authority budget.

**Architecture:** Three-component momentum coordinator (damping, feedforward, contact recovery) integrated with Phase 2 Centroidal WBC. Provides reactive momentum regulation, proactive height-transition compensation, and asymmetric contact-based recovery.

**Tech Stack:** Python, JAX, MuJoCo MJX, pytest

---

## File Structure

This phase creates the following new files:

- `wheeled_biped/controllers/momentum_coordinator.py` - Momentum coordinator with damping, feedforward, and contact recovery
- `configs/controllers/momentum_coordinator.yaml` - Default momentum coordinator config
- `tests/test_momentum_coordinator.py` - Unit tests for momentum coordinator

This phase modifies:

- None (all new files)

---

## Task 1: Create MomentumCoordinator Skeleton

**Files:**
- Create: `wheeled_biped/controllers/momentum_coordinator.py`
- Create: `tests/test_momentum_coordinator.py`

- [ ] **Step 1: Write the failing test for coordinator creation**

```python
# tests/test_momentum_coordinator.py
import jax.numpy as jnp
import pytest
from wheeled_biped.controllers.momentum_coordinator import (
    MomentumCoordinator,
    MomentumCoordinatorConfig,
)
from wheeled_biped.controllers.centroidal_state_estimator import CentroidalState


def test_momentum_coordinator_creation():
    """Test MomentumCoordinator can be created with config."""
    config = MomentumCoordinatorConfig(
        k_momentum_lateral=0.8,
        k_momentum_sagittal=1.2,
        k_angular_roll=1.5,
        momentum_authority_budget=0.2,
    )
    coordinator = MomentumCoordinator(config)
    
    assert coordinator.config.k_momentum_lateral == 0.8
    assert coordinator.config.momentum_authority_budget == 0.2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_momentum_coordinator.py::test_momentum_coordinator_creation -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'wheeled_biped.controllers.momentum_coordinator'"

- [ ] **Step 3: Create MomentumCoordinator skeleton**

```python
# wheeled_biped/controllers/momentum_coordinator.py
"""Momentum coordinator for Level 2 stabilization.

Provides reactive momentum damping, proactive feedforward compensation,
and contact-aware recovery with 20% authority budget.
"""

import chex
import jax.numpy as jnp
from jax import Array

from wheeled_biped.controllers.centroidal_state_estimator import CentroidalState


@chex.dataclass
class MomentumCoordinatorConfig:
    """Configuration for momentum coordinator."""
    # Momentum damping
    k_momentum_lateral: float = 0.8
    k_momentum_sagittal: float = 1.2
    k_angular_roll: float = 1.5
    
    # Feedforward compensation
    k_feedforward: float = 5.0
    k_feedforward_hip: float = 2.0
    height_transition_threshold: float = 0.05  # m/s
    
    # Contact-aware recovery
    k_contact_recovery: float = 10.0
    k_contact_wheel_diff: float = 4.0
    unloading_threshold: float = 0.3  # 30% force asymmetry
    
    # Deadbands
    momentum_deadband_linear: float = 0.5  # kg*m/s
    momentum_deadband_angular: float = 0.2  # kg*m^2/s
    
    # Authority budget
    momentum_authority_budget: float = 0.2  # 20% of actuator range


class MomentumCoordinator:
    """Momentum coordinator for Level 2 stabilization."""
    
    def __init__(self, config: MomentumCoordinatorConfig):
        """Initialize momentum coordinator.
        
        Args:
            config: MomentumCoordinatorConfig with gains and thresholds
        """
        self.config = config
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_momentum_coordinator.py::test_momentum_coordinator_creation -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add wheeled_biped/controllers/momentum_coordinator.py tests/test_momentum_coordinator.py
git commit -m "feat: create MomentumCoordinator skeleton"
```

---

## Task 2: Implement Momentum Damping

**Files:**
- Modify: `wheeled_biped/controllers/momentum_coordinator.py`
- Modify: `tests/test_momentum_coordinator.py`

- [ ] **Step 1: Write the failing test for momentum damping**

```python
# tests/test_momentum_coordinator.py (add to existing file)

def test_momentum_damping_outside_deadband():
    """Test momentum damping when momentum exceeds deadband."""
    config = MomentumCoordinatorConfig(
        k_momentum_lateral=0.8,
        k_momentum_sagittal=1.2,
        k_angular_roll=1.5,
        momentum_deadband_linear=0.5,
        momentum_deadband_angular=0.2,
    )
    coordinator = MomentumCoordinator(config)
    
    # State with significant momentum (outside deadband)
    state = CentroidalState(
        com_pos=jnp.array([0.0, 0.0, 0.6]),
        com_vel=jnp.array([0.0, 0.0, 0.0]),
        capture_point=jnp.zeros(2),
        divergence=jnp.zeros(2),
        linear_momentum=jnp.array([2.0, 1.5, 0.0]),  # 2.5 kg*m/s magnitude
        angular_momentum=jnp.array([0.5, 0.0, 0.0]),  # 0.5 kg*m^2/s roll
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=75.0,
        right_wheel_force=75.0,
    )
    
    tau_damping = coordinator.compute_momentum_damping_torque(state)
    
    # Should produce damping torques on hip roll and wheels
    assert jnp.abs(tau_damping[0]) > 0.5  # left hip roll
    assert jnp.abs(tau_damping[5]) > 0.5  # right hip roll
    assert jnp.abs(tau_damping[4]) > 0.5  # left wheel
    assert jnp.abs(tau_damping[9]) > 0.5  # right wheel


def test_momentum_damping_inside_deadband():
    """Test momentum damping is zero when momentum inside deadband."""
    config = MomentumCoordinatorConfig(
        k_momentum_lateral=0.8,
        k_momentum_sagittal=1.2,
        k_angular_roll=1.5,
        momentum_deadband_linear=0.5,
        momentum_deadband_angular=0.2,
    )
    coordinator = MomentumCoordinator(config)
    
    # State with small momentum (inside deadband)
    state = CentroidalState(
        com_pos=jnp.array([0.0, 0.0, 0.6]),
        com_vel=jnp.array([0.0, 0.0, 0.0]),
        capture_point=jnp.zeros(2),
        divergence=jnp.zeros(2),
        linear_momentum=jnp.array([0.1, 0.1, 0.0]),  # 0.14 kg*m/s magnitude
        angular_momentum=jnp.array([0.05, 0.0, 0.0]),  # 0.05 kg*m^2/s roll
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=75.0,
        right_wheel_force=75.0,
    )
    
    tau_damping = coordinator.compute_momentum_damping_torque(state)
    
    # Should produce near-zero torques
    assert jnp.max(jnp.abs(tau_damping)) < 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_momentum_coordinator.py::test_momentum_damping_outside_deadband -v`
Expected: FAIL with "AttributeError: 'MomentumCoordinator' object has no attribute 'compute_momentum_damping_torque'"

- [ ] **Step 3: Implement momentum damping**

```python
# wheeled_biped/controllers/momentum_coordinator.py
# Add this method to MomentumCoordinator class

def compute_momentum_damping_torque(self, state: CentroidalState) -> Array:
    """Compute momentum damping torque to prevent oscillation buildup.
    
    Args:
        state: CentroidalState with linear and angular momentum
        
    Returns:
        Damping torque array (10,) opposing unwanted momentum
    """
    tau = jnp.zeros(10)
    
    # Linear momentum damping (lateral and sagittal)
    linear_momentum_mag = jnp.sqrt(
        state.linear_momentum[0]**2 + state.linear_momentum[1]**2
    )
    
    # JAX-compatible deadband using jnp.where
    linear_active = jnp.where(
        linear_momentum_mag > self.config.momentum_deadband_linear,
        1.0,
        0.0,
    )
    
    # Lateral momentum → hip roll damping
    lateral_momentum = state.linear_momentum[1]
    tau_lateral = -self.config.k_momentum_lateral * lateral_momentum * linear_active
    tau = tau.at[0].set(tau_lateral)  # left hip roll
    tau = tau.at[5].set(tau_lateral)  # right hip roll
    
    # Sagittal momentum → wheel damping
    sagittal_momentum = state.linear_momentum[0]
    tau_sagittal = -self.config.k_momentum_sagittal * sagittal_momentum * linear_active
    tau = tau.at[4].set(tau_sagittal)  # left wheel
    tau = tau.at[9].set(tau_sagittal)  # right wheel
    
    # Angular momentum damping (roll axis most critical)
    angular_momentum_mag = jnp.abs(state.angular_momentum[0])
    
    angular_active = jnp.where(
        angular_momentum_mag > self.config.momentum_deadband_angular,
        1.0,
        0.0,
    )
    
    # Roll momentum → differential hip roll
    roll_momentum = state.angular_momentum[0]
    tau_angular_left = -self.config.k_angular_roll * roll_momentum * angular_active
    tau_angular_right = self.config.k_angular_roll * roll_momentum * angular_active
    
    tau = tau.at[0].add(tau_angular_left)  # left hip roll (add to existing)
    tau = tau.at[5].add(tau_angular_right)  # right hip roll (opposite sign)
    
    return tau
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_momentum_coordinator.py::test_momentum_damping_outside_deadband tests/test_momentum_coordinator.py::test_momentum_damping_inside_deadband -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add wheeled_biped/controllers/momentum_coordinator.py tests/test_momentum_coordinator.py
git commit -m "feat: add momentum damping with deadband control"
```

---

## Task 3: Implement Feedforward Compensation

**Files:**
- Modify: `wheeled_biped/controllers/momentum_coordinator.py`
- Modify: `tests/test_momentum_coordinator.py`

- [ ] **Step 1: Write the failing test for feedforward compensation**

```python
# tests/test_momentum_coordinator.py (add to existing file)

def test_feedforward_compensation_height_transition():
    """Test feedforward compensation during height transitions."""
    config = MomentumCoordinatorConfig(
        k_feedforward=5.0,
        k_feedforward_hip=2.0,
        height_transition_threshold=0.05,
    )
    coordinator = MomentumCoordinator(config)
    
    # Mock observation with height command and velocity
    obs = jnp.zeros(42)
    obs = obs.at[39].set(0.65)  # height_cmd = 0.65m
    
    # State with current height and velocity indicating transition
    state = CentroidalState(
        com_pos=jnp.array([0.0, 0.0, 0.60]),  # current height = 0.60m
        com_vel=jnp.array([0.0, 0.0, 0.08]),  # rising at 0.08 m/s
        capture_point=jnp.zeros(2),
        divergence=jnp.zeros(2),
        linear_momentum=jnp.zeros(3),
        angular_momentum=jnp.zeros(3),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=75.0,
        right_wheel_force=75.0,
    )
    
    tau_ff = coordinator.compute_feedforward_compensation_torque(obs, state)
    
    # Should produce feedforward torques on wheels and hip pitch
    assert jnp.abs(tau_ff[4]) > 0.1  # left wheel
    assert jnp.abs(tau_ff[9]) > 0.1  # right wheel
    assert jnp.abs(tau_ff[2]) > 0.05  # left hip pitch
    assert jnp.abs(tau_ff[7]) > 0.05  # right hip pitch
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_momentum_coordinator.py::test_feedforward_compensation_height_transition -v`
Expected: FAIL with "AttributeError: 'MomentumCoordinator' object has no attribute 'compute_feedforward_compensation_torque'"

- [ ] **Step 3: Implement feedforward compensation**

```python
# wheeled_biped/controllers/momentum_coordinator.py
# Add this method to MomentumCoordinator class

def compute_feedforward_compensation_torque(self, obs: Array, state: CentroidalState) -> Array:
    """Compute feedforward compensation for height transitions.
    
    Args:
        obs: Observation array with height_cmd at index 39
        state: CentroidalState with current height and velocity
        
    Returns:
        Feedforward torque array (10,) for proactive compensation
    """
    tau = jnp.zeros(10)
    
    # Extract height command and current state
    height_cmd = obs[39]
    height_current = state.com_pos[2]
    height_vel = state.com_vel[2]
    
    # Detect height transition
    height_error = height_cmd - height_current
    transition_active = jnp.where(
        jnp.abs(height_vel) > self.config.height_transition_threshold,
        1.0,
        0.0,
    )
    
    # Feedforward compensation based on height velocity direction
    # Rising (positive vel) → anticipate backward pitch, apply forward wheel torque
    # Squatting (negative vel) → anticipate forward pitch, apply backward wheel torque
    tau_wheel_ff = self.config.k_feedforward * height_vel * transition_active
    tau_hip_ff = -self.config.k_feedforward_hip * height_vel * transition_active
    
    # Apply to wheels (common mode)
    tau = tau.at[4].set(tau_wheel_ff)  # left wheel
    tau = tau.at[9].set(tau_wheel_ff)  # right wheel
    
    # Apply to hip pitch (both legs)
    tau = tau.at[2].set(tau_hip_ff)  # left hip pitch
    tau = tau.at[7].set(tau_hip_ff)  # right hip pitch
    
    return tau
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_momentum_coordinator.py::test_feedforward_compensation_height_transition -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add wheeled_biped/controllers/momentum_coordinator.py tests/test_momentum_coordinator.py
git commit -m "feat: add feedforward compensation for height transitions"
```

---

## Task 4: Implement Contact-Aware Recovery

**Files:**
- Modify: `wheeled_biped/controllers/momentum_coordinator.py`
- Modify: `tests/test_momentum_coordinator.py`

- [ ] **Step 1: Write the failing test for contact-aware recovery**

```python
# tests/test_momentum_coordinator.py (add to existing file)

def test_contact_aware_recovery_unloading():
    """Test contact-aware recovery when one wheel is unloading."""
    config = MomentumCoordinatorConfig(
        k_contact_recovery=10.0,
        k_contact_wheel_diff=4.0,
        unloading_threshold=0.3,
    )
    coordinator = MomentumCoordinator(config)
    
    # State with asymmetric contact forces (left wheel unloading)
    state = CentroidalState(
        com_pos=jnp.array([0.0, 0.0, 0.6]),
        com_vel=jnp.array([0.0, 0.0, 0.0]),
        capture_point=jnp.zeros(2),
        divergence=jnp.zeros(2),
        linear_momentum=jnp.zeros(3),
        angular_momentum=jnp.zeros(3),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=40.0,  # 40% of total
        right_wheel_force=60.0,  # 60% of total
    )
    
    tau_recovery = coordinator.compute_contact_aware_recovery_torque(state)
    
    # Should produce recovery torques to shift support toward loaded wheel
    assert jnp.abs(tau_recovery[0]) > 0.5  # left hip roll
    assert jnp.abs(tau_recovery[5]) > 0.5  # right hip roll
    assert jnp.abs(tau_recovery[4] - tau_recovery[9]) > 0.5  # wheel differential
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_momentum_coordinator.py::test_contact_aware_recovery_unloading -v`
Expected: FAIL with "AttributeError: 'MomentumCoordinator' object has no attribute 'compute_contact_aware_recovery_torque'"

- [ ] **Step 3: Implement contact-aware recovery**

```python
# wheeled_biped/controllers/momentum_coordinator.py
# Add this method to MomentumCoordinator class

def compute_contact_aware_recovery_torque(self, state: CentroidalState) -> Array:
    """Compute contact-aware recovery torque for asymmetric support.
    
    Args:
        state: CentroidalState with wheel contact forces
        
    Returns:
        Recovery torque array (10,) for contact-based redistribution
    """
    tau = jnp.zeros(10)
    
    # Compute force imbalance
    total_force = state.left_wheel_force + state.right_wheel_force
    
    # Avoid division by zero
    total_force_safe = jnp.where(total_force > 1.0, total_force, 1.0)
    
    force_ratio_left = state.left_wheel_force / total_force_safe
    force_ratio_right = state.right_wheel_force / total_force_safe
    
    # Detect unloading (force asymmetry exceeds threshold)
    force_imbalance = jnp.abs(force_ratio_left - force_ratio_right)
    unloading_active = jnp.where(
        force_imbalance > self.config.unloading_threshold,
        1.0,
        0.0,
    )
    
    # Recovery direction: shift toward loaded wheel
    # If left wheel has less force, apply positive hip roll (shift right)
    # If right wheel has less force, apply negative hip roll (shift left)
    recovery_direction = force_ratio_right - force_ratio_left
    
    # Hip roll recovery (symmetric - both legs same direction)
    tau_hip_roll = self.config.k_contact_recovery * recovery_direction * unloading_active
    tau = tau.at[0].set(tau_hip_roll)  # left hip roll
    tau = tau.at[5].set(tau_hip_roll)  # right hip roll
    
    # Wheel differential recovery
    tau_wheel_diff = self.config.k_contact_wheel_diff * recovery_direction * unloading_active
    tau = tau.at[4].set(tau_wheel_diff)  # left wheel
    tau = tau.at[9].set(-tau_wheel_diff)  # right wheel (opposite)
    
    return tau
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_momentum_coordinator.py::test_contact_aware_recovery_unloading -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add wheeled_biped/controllers/momentum_coordinator.py tests/test_momentum_coordinator.py
git commit -m "feat: add contact-aware recovery for asymmetric support"
```

---

## Task 5: Implement Authority Budget Clipping

**Files:**
- Modify: `wheeled_biped/controllers/momentum_coordinator.py`
- Modify: `tests/test_momentum_coordinator.py`

- [ ] **Step 1: Write the failing test for authority budget clipping**

```python
# tests/test_momentum_coordinator.py (add to existing file)

def test_momentum_authority_budget_clipping():
    """Test authority budget clipping scales torques proportionally."""
    config = MomentumCoordinatorConfig(
        momentum_authority_budget=0.2,
    )
    coordinator = MomentumCoordinator(config)
    
    # Create torque vector that exceeds 20% budget
    tau_desired = jnp.array([10.0, 0.0, 0.0, 0.0, 15.0, 10.0, 0.0, 0.0, 0.0, 15.0])
    
    tau_clipped = coordinator.clip_to_authority_budget(tau_desired)
    
    # Should respect 20% authority budget (6 Nm with max_actuator_torque=30)
    assert jnp.max(jnp.abs(tau_clipped)) <= 6.0
    
    # Should preserve proportions
    ratio = tau_clipped[0] / tau_clipped[4]
    expected_ratio = tau_desired[0] / tau_desired[4]
    assert jnp.abs(ratio - expected_ratio) < 0.01
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_momentum_coordinator.py::test_momentum_authority_budget_clipping -v`
Expected: FAIL with "AttributeError: 'MomentumCoordinator' object has no attribute 'clip_to_authority_budget'"

- [ ] **Step 3: Implement authority budget clipping**

```python
# wheeled_biped/controllers/momentum_coordinator.py
# Add this method to MomentumCoordinator class

def clip_to_authority_budget(self, tau: Array) -> Array:
    """Clip torque to momentum coordinator authority budget.
    
    Args:
        tau: Desired torque array (10,)
        
    Returns:
        Clipped torque array (10,) within 20% authority budget
    """
    # Maximum actuator torque (hardcoded as per Phase 2)
    max_actuator_torque = 30.0
    
    # Compute budget limit
    budget_limit = self.config.momentum_authority_budget * max_actuator_torque
    
    # Find maximum absolute torque
    max_tau = jnp.max(jnp.abs(tau))
    
    # JAX-compatible conditional scaling
    scale_factor = jnp.where(max_tau <= budget_limit, 1.0, budget_limit / max_tau)
    tau_clipped = tau * scale_factor
    
    return tau_clipped
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_momentum_coordinator.py::test_momentum_authority_budget_clipping -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add wheeled_biped/controllers/momentum_coordinator.py tests/test_momentum_coordinator.py
git commit -m "feat: add momentum authority budget clipping"
```

---

## Task 6: Implement Integrated Momentum Coordinator

**Files:**
- Modify: `wheeled_biped/controllers/momentum_coordinator.py`
- Modify: `tests/test_momentum_coordinator.py`

- [ ] **Step 1: Write the failing test for integrated coordinator**

```python
# tests/test_momentum_coordinator.py (add to existing file)

def test_integrated_momentum_coordinator():
    """Test integrated momentum coordinator combines all components."""
    config = MomentumCoordinatorConfig(
        k_momentum_lateral=0.8,
        k_momentum_sagittal=1.2,
        k_angular_roll=1.5,
        k_feedforward=5.0,
        k_contact_recovery=10.0,
        momentum_authority_budget=0.2,
    )
    coordinator = MomentumCoordinator(config)
    
    # Mock observation with height command
    obs = jnp.zeros(42)
    obs = obs.at[39].set(0.65)  # height_cmd = 0.65m
    
    # State with momentum, height transition, and contact asymmetry
    state = CentroidalState(
        com_pos=jnp.array([0.0, 0.0, 0.60]),
        com_vel=jnp.array([0.0, 0.0, 0.08]),  # rising
        capture_point=jnp.zeros(2),
        divergence=jnp.zeros(2),
        linear_momentum=jnp.array([1.0, 0.8, 0.0]),
        angular_momentum=jnp.array([0.3, 0.0, 0.0]),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=40.0,
        right_wheel_force=60.0,
    )
    
    tau_momentum = coordinator.compute_momentum_coordinator_torque(obs, state)
    
    # Should produce non-zero torques
    assert jnp.any(jnp.abs(tau_momentum) > 0.1)
    
    # Should respect 20% authority budget
    assert jnp.max(jnp.abs(tau_momentum)) <= 6.0  # 20% of 30 Nm
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_momentum_coordinator.py::test_integrated_momentum_coordinator -v`
Expected: FAIL with "AttributeError: 'MomentumCoordinator' object has no attribute 'compute_momentum_coordinator_torque'"

- [ ] **Step 3: Implement integrated momentum coordinator**

```python
# wheeled_biped/controllers/momentum_coordinator.py
# Add this method to MomentumCoordinator class

def compute_momentum_coordinator_torque(self, obs: Array, state: CentroidalState) -> Array:
    """Compute integrated momentum coordinator torque.
    
    Combines momentum damping, feedforward compensation, and contact-aware
    recovery with 20% authority budget.
    
    Args:
        obs: Observation array
        state: CentroidalState with momentum and contact information
        
    Returns:
        Momentum coordinator torque array (10,) clipped to 20% authority
    """
    # Compute individual components
    tau_damping = self.compute_momentum_damping_torque(state)
    tau_feedforward = self.compute_feedforward_compensation_torque(obs, state)
    tau_recovery = self.compute_contact_aware_recovery_torque(state)
    
    # Sum all components
    tau_momentum_desired = tau_damping + tau_feedforward + tau_recovery
    
    # Clip to 20% authority budget
    tau_momentum = self.clip_to_authority_budget(tau_momentum_desired)
    
    return tau_momentum
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_momentum_coordinator.py::test_integrated_momentum_coordinator -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add wheeled_biped/controllers/momentum_coordinator.py tests/test_momentum_coordinator.py
git commit -m "feat: implement integrated momentum coordinator"
```

---

## Task 7: Add Integration Test

**Files:**
- Modify: `tests/test_momentum_coordinator.py`

- [ ] **Step 1: Write the integration test**

```python
# tests/test_momentum_coordinator.py (add to existing file)

def test_momentum_coordinator_integration_no_nan():
    """Integration test: 100-step rollout produces no NaN."""
    config = MomentumCoordinatorConfig(
        k_momentum_lateral=0.8,
        k_momentum_sagittal=1.2,
        k_angular_roll=1.5,
        k_feedforward=5.0,
        k_contact_recovery=10.0,
        momentum_authority_budget=0.2,
    )
    coordinator = MomentumCoordinator(config)
    
    # Run 100-step rollout with varying conditions
    for step in range(100):
        # Mock observation
        obs = jnp.zeros(42)
        obs = obs.at[39].set(0.60 + 0.05 * jnp.sin(step * 0.1))  # varying height cmd
        
        # Mock state with time-varying momentum and contact
        state = CentroidalState(
            com_pos=jnp.array([0.0, 0.0, 0.60]),
            com_vel=jnp.array([0.0, 0.0, 0.05 * jnp.cos(step * 0.1)]),
            capture_point=jnp.zeros(2),
            divergence=jnp.zeros(2),
            linear_momentum=jnp.array([
                0.5 * jnp.sin(step * 0.05),
                0.3 * jnp.cos(step * 0.05),
                0.0
            ]),
            angular_momentum=jnp.array([0.1 * jnp.sin(step * 0.08), 0.0, 0.0]),
            left_wheel_contact=True,
            right_wheel_contact=True,
            left_wheel_force=50.0 + 10.0 * jnp.sin(step * 0.06),
            right_wheel_force=50.0 - 10.0 * jnp.sin(step * 0.06),
        )
        
        # Compute momentum coordinator torque
        tau_momentum = coordinator.compute_momentum_coordinator_torque(obs, state)
        
        # Verify no NaN
        assert not jnp.any(jnp.isnan(tau_momentum)), f"NaN at step {step}"
        
        # Verify within authority budget
        assert jnp.max(jnp.abs(tau_momentum)) <= 6.0, f"Budget exceeded at step {step}"
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_momentum_coordinator.py::test_momentum_coordinator_integration_no_nan -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_momentum_coordinator.py
git commit -m "test: add 100-step integration test for momentum coordinator"
```

---

## Task 8: Create Configuration File

**Files:**
- Create: `configs/controllers/momentum_coordinator.yaml`

- [ ] **Step 1: Create momentum coordinator config**

```yaml
# configs/controllers/momentum_coordinator.yaml
# Momentum coordinator configuration for Level 2 stabilization

controller_type: momentum_coordinator

# Momentum damping
k_momentum_lateral: 0.8
k_momentum_sagittal: 1.2
k_angular_roll: 1.5

# Feedforward compensation
k_feedforward: 5.0
k_feedforward_hip: 2.0
height_transition_threshold: 0.05  # m/s

# Contact-aware recovery
k_contact_recovery: 10.0
k_contact_wheel_diff: 4.0
unloading_threshold: 0.3  # 30% force asymmetry

# Deadbands
momentum_deadband_linear: 0.5  # kg*m/s
momentum_deadband_angular: 0.2  # kg*m^2/s

# Authority budget
momentum_authority_budget: 0.2  # 20% of actuator range

# Evaluation metadata
level: "Level 2 Stabilization"
description: "Momentum damping, feedforward compensation, and contact-aware recovery"
baseline_comparison: "Phase 2 Centroidal WBC (Level 1 only)"
```

- [ ] **Step 2: Commit configuration file**

```bash
git add configs/controllers/momentum_coordinator.yaml
git commit -m "config: add momentum coordinator configuration"
```

---

## Phase 3 Completion Checklist

Before proceeding to Phase 4 (Posture Regularization), verify:

- [ ] All unit tests pass: `pytest tests/test_momentum_coordinator.py -v`
- [ ] Integration test passes: 100-step rollout produces no NaN
- [ ] Momentum damping respects deadband (zero torque inside, non-zero outside)
- [ ] Feedforward compensation activates during height transitions
- [ ] Contact-aware recovery detects and responds to wheel unloading
- [ ] Authority budget clipping works (max 20% of actuator range)
- [ ] Integrated coordinator combines all three components
- [ ] Configuration file created
- [ ] All files committed to git

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-15-phase3-momentum-coordinator.md`.

**Two execution options:**

**1. Subagent-Driven (recommended)** - Dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**

