# Phase 4: Posture Regularization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Posture Regularization with two-level gating (WBC error gate, momentum coordinator gate) and per-joint deadbands to achieve Level 3 stabilization with 20% authority budget.

**Architecture:** Permissive posture correction that guides toward good posture without fighting dynamic balance requirements. Uses WBC error gating (disable if >30% capacity) and momentum coordinator gating (50% authority reduction when active). Per-joint deadbands allow natural sway.

**Tech Stack:** Python, JAX, MuJoCo MJX, pytest

---

## File Structure

This phase creates the following new files:

- `wheeled_biped/controllers/posture_regularizer.py` - Posture regularization with two-level gating and per-joint deadbands
- `configs/controllers/posture_regularizer.yaml` - Default posture regularization config
- `tests/test_posture_regularizer.py` - Unit tests for posture regularizer

This phase modifies:

- None (all new files)

---

## Task 1: Create PostureRegularizer Skeleton

**Files:**
- Create: `wheeled_biped/controllers/posture_regularizer.py`
- Create: `tests/test_posture_regularizer.py`

- [ ] **Step 1: Write the failing test for regularizer creation**

```python
# tests/test_posture_regularizer.py
import jax.numpy as jnp
import pytest
from wheeled_biped.controllers.posture_regularizer import (
    PostureRegularizer,
    PostureRegularizerConfig,
)


def test_posture_regularizer_creation():
    """Test PostureRegularizer can be created with config."""
    config = PostureRegularizerConfig(
        k_posture=2.0,
        posture_authority_budget=0.2,
    )
    regularizer = PostureRegularizer(config)
    
    assert regularizer.config.k_posture == 2.0
    assert regularizer.config.posture_authority_budget == 0.2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_posture_regularizer.py::test_posture_regularizer_creation -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'wheeled_biped.controllers.posture_regularizer'"

- [ ] **Step 3: Create PostureRegularizer skeleton**

```python
# wheeled_biped/controllers/posture_regularizer.py
"""Posture regularization for Level 3 stabilization.

Provides weak posture restoration with two-level gating (WBC error gate,
momentum coordinator gate) and per-joint deadbands with 20% authority budget.
"""

import chex
import jax.numpy as jnp
from jax import Array


@chex.dataclass(frozen=True)
class PostureRegularizerConfig:
    """Configuration for posture regularizer."""
    # Proportional gain
    k_posture: float = 2.0  # Weak compared to WBC gains
    
    # Per-joint deadbands (radians)
    hip_roll_deadband: float = 0.05  # ±2.9° - allow lateral sway
    hip_yaw_deadband: float = 0.03  # ±1.7° - tighter, yaw drift is bad
    hip_pitch_deadband: float = 0.08  # ±4.6° - allow squat variation
    knee_deadband: float = 0.10  # ±5.7° - allow knee bend variation
    wheel_deadband: float = 0.0  # wheels don't have posture target
    
    # Gating thresholds
    wbc_error_threshold: float = 0.3  # 30% of WBC capacity
    momentum_active_scale: float = 0.5  # 50% authority when momentum active
    
    # Authority budget
    posture_authority_budget: float = 0.2  # 20% of actuator range


class PostureRegularizer:
    """Posture regularizer for Level 3 stabilization."""
    
    def __init__(self, config: PostureRegularizerConfig):
        """Initialize posture regularizer.
        
        Args:
            config: PostureRegularizerConfig with gains and thresholds
        """
        self.config = config
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_posture_regularizer.py::test_posture_regularizer_creation -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add wheeled_biped/controllers/posture_regularizer.py tests/test_posture_regularizer.py
git commit -m "feat: create PostureRegularizer skeleton"
```

---

## Task 2: Implement Posture Restoration Torque

**Files:**
- Modify: `wheeled_biped/controllers/posture_regularizer.py`
- Modify: `tests/test_posture_regularizer.py`

- [ ] **Step 1: Write the failing test for posture restoration**

```python
# tests/test_posture_regularizer.py (add to existing file)

def test_posture_restoration_outside_deadband():
    """Test posture restoration when joint error exceeds deadband."""
    config = PostureRegularizerConfig(
        k_posture=2.0,
        hip_roll_deadband=0.05,
        hip_yaw_deadband=0.03,
        hip_pitch_deadband=0.08,
        knee_deadband=0.10,
    )
    regularizer = PostureRegularizer(config)
    
    # Joint positions with errors outside deadband
    # Target posture is zero for all joints
    joint_pos = jnp.array([
        0.08,  # left hip roll - outside 0.05 deadband
        0.05,  # left hip yaw - outside 0.03 deadband
        0.12,  # left hip pitch - outside 0.08 deadband
        0.15,  # left knee - outside 0.10 deadband
        0.0,   # left wheel - no posture target
        0.08,  # right hip roll
        0.05,  # right hip yaw
        0.12,  # right hip pitch
        0.15,  # right knee
        0.0,   # right wheel
    ])
    
    tau_posture = regularizer.compute_posture_restoration_torque(joint_pos)
    
    # Should produce restoration torques on leg joints
    assert jnp.abs(tau_posture[0]) > 0.05  # left hip roll
    assert jnp.abs(tau_posture[1]) > 0.05  # left hip yaw
    assert jnp.abs(tau_posture[2]) > 0.05  # left hip pitch
    assert jnp.abs(tau_posture[3]) > 0.05  # left knee
    assert jnp.abs(tau_posture[4]) < 0.01  # left wheel (no target)
    assert jnp.abs(tau_posture[5]) > 0.05  # right hip roll
    assert jnp.abs(tau_posture[6]) > 0.05  # right hip yaw
    assert jnp.abs(tau_posture[7]) > 0.05  # right hip pitch
    assert jnp.abs(tau_posture[8]) > 0.05  # right knee
    assert jnp.abs(tau_posture[9]) < 0.01  # right wheel (no target)


def test_posture_restoration_inside_deadband():
    """Test posture restoration is zero when joint error inside deadband."""
    config = PostureRegularizerConfig(
        k_posture=2.0,
        hip_roll_deadband=0.05,
        hip_yaw_deadband=0.03,
        hip_pitch_deadband=0.08,
        knee_deadband=0.10,
    )
    regularizer = PostureRegularizer(config)
    
    # Joint positions with errors inside deadband
    joint_pos = jnp.array([
        0.02,  # left hip roll - inside 0.05 deadband
        0.01,  # left hip yaw - inside 0.03 deadband
        0.04,  # left hip pitch - inside 0.08 deadband
        0.05,  # left knee - inside 0.10 deadband
        0.0,   # left wheel
        0.02,  # right hip roll
        0.01,  # right hip yaw
        0.04,  # right hip pitch
        0.05,  # right knee
        0.0,   # right wheel
    ])
    
    tau_posture = regularizer.compute_posture_restoration_torque(joint_pos)
    
    # Should produce near-zero torques
    assert jnp.max(jnp.abs(tau_posture)) < 0.01
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_posture_regularizer.py::test_posture_restoration_outside_deadband -v`
Expected: FAIL with "AttributeError: 'PostureRegularizer' object has no attribute 'compute_posture_restoration_torque'"

- [ ] **Step 3: Implement posture restoration torque**

```python
# wheeled_biped/controllers/posture_regularizer.py
# Add this method to PostureRegularizer class

def compute_posture_restoration_torque(self, joint_pos: Array) -> Array:
    """Compute posture restoration torque with per-joint deadbands.
    
    Args:
        joint_pos: Joint position array (10,) - current joint angles
        
    Returns:
        Posture restoration torque array (10,) opposing posture errors
    """
    # Target posture is zero for all joints
    target_pos = jnp.zeros(10)
    
    # Compute posture errors
    posture_error = joint_pos - target_pos
    
    # Per-joint deadbands
    deadbands = jnp.array([
        self.config.hip_roll_deadband,   # 0: left hip roll
        self.config.hip_yaw_deadband,    # 1: left hip yaw
        self.config.hip_pitch_deadband,  # 2: left hip pitch
        self.config.knee_deadband,       # 3: left knee
        self.config.wheel_deadband,      # 4: left wheel
        self.config.hip_roll_deadband,   # 5: right hip roll
        self.config.hip_yaw_deadband,    # 6: right hip yaw
        self.config.hip_pitch_deadband,  # 7: right hip pitch
        self.config.knee_deadband,       # 8: right knee
        self.config.wheel_deadband,      # 9: right wheel
    ])
    
    # JAX-compatible deadband using jnp.where
    # Only apply torque if error exceeds deadband
    active = jnp.where(jnp.abs(posture_error) > deadbands, 1.0, 0.0)
    
    # Proportional control with deadband gating
    tau = -self.config.k_posture * posture_error * active
    
    return tau
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_posture_regularizer.py::test_posture_restoration_outside_deadband tests/test_posture_regularizer.py::test_posture_restoration_inside_deadband -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add wheeled_biped/controllers/posture_regularizer.py tests/test_posture_regularizer.py
git commit -m "feat: add posture restoration with per-joint deadbands"
```

---

## Task 3: Implement WBC Error Gating

**Files:**
- Modify: `wheeled_biped/controllers/posture_regularizer.py`
- Modify: `tests/test_posture_regularizer.py`

- [ ] **Step 1: Write the failing test for WBC error gating**

```python
# tests/test_posture_regularizer.py (add to existing file)

def test_wbc_error_gating_disabled():
    """Test posture regularization is disabled when WBC error exceeds threshold."""
    config = PostureRegularizerConfig(
        k_posture=2.0,
        wbc_error_threshold=0.3,
    )
    regularizer = PostureRegularizer(config)
    
    # Joint positions with errors outside deadband
    joint_pos = jnp.array([0.1, 0.1, 0.1, 0.15, 0.0, 0.1, 0.1, 0.1, 0.15, 0.0])
    
    # WBC error exceeds 30% threshold (0.4 > 0.3)
    wbc_error_magnitude = 0.4
    
    tau_posture = regularizer.apply_wbc_error_gate(
        joint_pos, wbc_error_magnitude
    )
    
    # Should produce zero torques when WBC error is high
    assert jnp.max(jnp.abs(tau_posture)) < 0.01


def test_wbc_error_gating_enabled():
    """Test posture regularization is enabled when WBC error is low."""
    config = PostureRegularizerConfig(
        k_posture=2.0,
        hip_roll_deadband=0.05,
        wbc_error_threshold=0.3,
    )
    regularizer = PostureRegularizer(config)
    
    # Joint positions with errors outside deadband
    joint_pos = jnp.array([0.1, 0.1, 0.1, 0.15, 0.0, 0.1, 0.1, 0.1, 0.15, 0.0])
    
    # WBC error below 30% threshold (0.2 < 0.3)
    wbc_error_magnitude = 0.2
    
    tau_posture = regularizer.apply_wbc_error_gate(
        joint_pos, wbc_error_magnitude
    )
    
    # Should produce non-zero torques when WBC error is low
    assert jnp.any(jnp.abs(tau_posture) > 0.05)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_posture_regularizer.py::test_wbc_error_gating_disabled -v`
Expected: FAIL with "AttributeError: 'PostureRegularizer' object has no attribute 'apply_wbc_error_gate'"

- [ ] **Step 3: Implement WBC error gating**

```python
# wheeled_biped/controllers/posture_regularizer.py
# Add this method to PostureRegularizer class

def apply_wbc_error_gate(self, joint_pos: Array, wbc_error_magnitude: float) -> Array:
    """Apply WBC error gate to posture restoration.
    
    If WBC error exceeds threshold, completely disable posture regularization.
    
    Args:
        joint_pos: Joint position array (10,)
        wbc_error_magnitude: WBC error magnitude (normalized 0-1)
        
    Returns:
        Gated posture torque array (10,)
    """
    # Compute base posture restoration torque
    tau_posture = self.compute_posture_restoration_torque(joint_pos)
    
    # JAX-compatible gating using jnp.where
    # Disable completely if WBC error exceeds threshold
    gate = jnp.where(
        wbc_error_magnitude > self.config.wbc_error_threshold,
        0.0,
        1.0,
    )
    
    # Apply gate
    tau_gated = tau_posture * gate
    
    return tau_gated
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_posture_regularizer.py::test_wbc_error_gating_disabled tests/test_posture_regularizer.py::test_wbc_error_gating_enabled -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add wheeled_biped/controllers/posture_regularizer.py tests/test_posture_regularizer.py
git commit -m "feat: add WBC error gating for posture regularization"
```

---

## Task 4: Implement Momentum Coordinator Gating

**Files:**
- Modify: `wheeled_biped/controllers/posture_regularizer.py`
- Modify: `tests/test_posture_regularizer.py`

- [ ] **Step 1: Write the failing test for momentum coordinator gating**

```python
# tests/test_posture_regularizer.py (add to existing file)

def test_momentum_coordinator_gating_reduced():
    """Test posture authority is reduced when momentum coordinator is active."""
    config = PostureRegularizerConfig(
        k_posture=2.0,
        hip_roll_deadband=0.05,
        momentum_active_scale=0.5,
    )
    regularizer = PostureRegularizer(config)
    
    # Joint positions with errors outside deadband
    joint_pos = jnp.array([0.1, 0.1, 0.1, 0.15, 0.0, 0.1, 0.1, 0.1, 0.15, 0.0])
    
    # Momentum coordinator is active (magnitude > threshold)
    momentum_magnitude = 0.8
    
    tau_posture = regularizer.apply_momentum_gate(
        joint_pos, momentum_magnitude
    )
    
    # Compute expected torque with 50% reduction
    tau_full = regularizer.compute_posture_restoration_torque(joint_pos)
    expected_magnitude = jnp.max(jnp.abs(tau_full)) * 0.5
    
    # Should be reduced to 50% when momentum is active
    actual_magnitude = jnp.max(jnp.abs(tau_posture))
    assert jnp.abs(actual_magnitude - expected_magnitude) < 0.01


def test_momentum_coordinator_gating_full():
    """Test posture authority is full when momentum coordinator is inactive."""
    config = PostureRegularizerConfig(
        k_posture=2.0,
        hip_roll_deadband=0.05,
        momentum_active_scale=0.5,
    )
    regularizer = PostureRegularizer(config)
    
    # Joint positions with errors outside deadband
    joint_pos = jnp.array([0.1, 0.1, 0.1, 0.15, 0.0, 0.1, 0.1, 0.1, 0.15, 0.0])
    
    # Momentum coordinator is inactive (magnitude = 0)
    momentum_magnitude = 0.0
    
    tau_posture = regularizer.apply_momentum_gate(
        joint_pos, momentum_magnitude
    )
    
    # Compute expected full torque
    tau_full = regularizer.compute_posture_restoration_torque(joint_pos)
    
    # Should be at full authority when momentum is inactive
    assert jnp.allclose(tau_posture, tau_full, atol=0.01)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_posture_regularizer.py::test_momentum_coordinator_gating_reduced -v`
Expected: FAIL with "AttributeError: 'PostureRegularizer' object has no attribute 'apply_momentum_gate'"

- [ ] **Step 3: Implement momentum coordinator gating**

```python
# wheeled_biped/controllers/posture_regularizer.py
# Add this method to PostureRegularizer class

def apply_momentum_gate(self, joint_pos: Array, momentum_magnitude: float) -> Array:
    """Apply momentum coordinator gate to posture restoration.
    
    Reduces posture authority by 50% when momentum coordinator is active.
    
    Args:
        joint_pos: Joint position array (10,)
        momentum_magnitude: Momentum coordinator activity magnitude (0-1)
        
    Returns:
        Gated posture torque array (10,)
    """
    # Compute base posture restoration torque
    tau_posture = self.compute_posture_restoration_torque(joint_pos)
    
    # JAX-compatible gating using jnp.where
    # Reduce to 50% if momentum coordinator is active (magnitude > small threshold)
    gate = jnp.where(
        momentum_magnitude > 0.1,  # Small threshold to detect activity
        self.config.momentum_active_scale,
        1.0,
    )
    
    # Apply gate
    tau_gated = tau_posture * gate
    
    return tau_gated
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_posture_regularizer.py::test_momentum_coordinator_gating_reduced tests/test_posture_regularizer.py::test_momentum_coordinator_gating_full -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add wheeled_biped/controllers/posture_regularizer.py tests/test_posture_regularizer.py
git commit -m "feat: add momentum coordinator gating for posture regularization"
```

---

## Task 5: Implement Authority Budget Clipping

**Files:**
- Modify: `wheeled_biped/controllers/posture_regularizer.py`
- Modify: `tests/test_posture_regularizer.py`

- [ ] **Step 1: Write the failing test for authority budget clipping**

```python
# tests/test_posture_regularizer.py (add to existing file)

def test_posture_authority_budget_clipping():
    """Test authority budget clipping scales torques proportionally."""
    config = PostureRegularizerConfig(
        posture_authority_budget=0.2,
    )
    regularizer = PostureRegularizer(config)
    
    # Create torque vector that exceeds 20% budget
    tau_desired = jnp.array([10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0])
    
    tau_clipped = regularizer.clip_to_authority_budget(tau_desired)
    
    # Should respect 20% authority budget (6 Nm with max_actuator_torque=30)
    assert jnp.max(jnp.abs(tau_clipped)) <= 6.0
    
    # Should preserve proportions
    ratio = tau_clipped[0] / tau_clipped[5]
    expected_ratio = tau_desired[0] / tau_desired[5]
    assert jnp.abs(ratio - expected_ratio) < 0.01
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_posture_regularizer.py::test_posture_authority_budget_clipping -v`
Expected: FAIL with "AttributeError: 'PostureRegularizer' object has no attribute 'clip_to_authority_budget'"

- [ ] **Step 3: Implement authority budget clipping**

```python
# wheeled_biped/controllers/posture_regularizer.py
# Add this method to PostureRegularizer class

def clip_to_authority_budget(self, tau: Array) -> Array:
    """Clip torque to posture regularizer authority budget.
    
    Args:
        tau: Desired torque array (10,)
        
    Returns:
        Clipped torque array (10,) within 20% authority budget
    """
    # Maximum actuator torque (hardcoded as per Phase 2)
    max_actuator_torque = 30.0
    
    # Compute budget limit
    budget_limit = self.config.posture_authority_budget * max_actuator_torque
    
    # Find maximum absolute torque
    max_tau = jnp.max(jnp.abs(tau))
    
    # JAX-compatible conditional scaling
    scale_factor = jnp.where(max_tau <= budget_limit, 1.0, budget_limit / max_tau)
    tau_clipped = tau * scale_factor
    
    return tau_clipped
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_posture_regularizer.py::test_posture_authority_budget_clipping -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add wheeled_biped/controllers/posture_regularizer.py tests/test_posture_regularizer.py
git commit -m "feat: add posture authority budget clipping"
```

---

## Task 6: Implement Integrated Posture Regularizer

**Files:**
- Modify: `wheeled_biped/controllers/posture_regularizer.py`
- Modify: `tests/test_posture_regularizer.py`

- [ ] **Step 1: Write the failing test for integrated regularizer**

```python
# tests/test_posture_regularizer.py (add to existing file)

def test_integrated_posture_regularizer():
    """Test integrated posture regularizer with two-level gating."""
    config = PostureRegularizerConfig(
        k_posture=2.0,
        hip_roll_deadband=0.05,
        wbc_error_threshold=0.3,
        momentum_active_scale=0.5,
        posture_authority_budget=0.2,
    )
    regularizer = PostureRegularizer(config)
    
    # Joint positions with errors outside deadband
    joint_pos = jnp.array([0.1, 0.1, 0.1, 0.15, 0.0, 0.1, 0.1, 0.1, 0.15, 0.0])
    
    # WBC error below threshold, momentum coordinator active
    wbc_error_magnitude = 0.2
    momentum_magnitude = 0.8
    
    tau_posture = regularizer.compute_posture_regularizer_torque(
        joint_pos, wbc_error_magnitude, momentum_magnitude
    )
    
    # Should produce non-zero torques
    assert jnp.any(jnp.abs(tau_posture) > 0.05)
    
    # Should respect 20% authority budget
    assert jnp.max(jnp.abs(tau_posture)) <= 6.0  # 20% of 30 Nm
    
    # Should be reduced due to momentum coordinator activity
    tau_full = regularizer.compute_posture_restoration_torque(joint_pos)
    tau_full_clipped = regularizer.clip_to_authority_budget(tau_full)
    expected_magnitude = jnp.max(jnp.abs(tau_full_clipped)) * 0.5
    actual_magnitude = jnp.max(jnp.abs(tau_posture))
    assert jnp.abs(actual_magnitude - expected_magnitude) < 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_posture_regularizer.py::test_integrated_posture_regularizer -v`
Expected: FAIL with "AttributeError: 'PostureRegularizer' object has no attribute 'compute_posture_regularizer_torque'"

- [ ] **Step 3: Implement integrated posture regularizer**

```python
# wheeled_biped/controllers/posture_regularizer.py
# Add this method to PostureRegularizer class

def compute_posture_regularizer_torque(
    self,
    joint_pos: Array,
    wbc_error_magnitude: float,
    momentum_magnitude: float,
) -> Array:
    """Compute integrated posture regularizer torque with two-level gating.
    
    Combines posture restoration with WBC error gating and momentum coordinator
    gating, then applies 20% authority budget.
    
    Args:
        joint_pos: Joint position array (10,)
        wbc_error_magnitude: WBC error magnitude (normalized 0-1)
        momentum_magnitude: Momentum coordinator activity magnitude (0-1)
        
    Returns:
        Posture regularizer torque array (10,) with gating and budget clipping
    """
    # Compute base posture restoration torque
    tau_posture = self.compute_posture_restoration_torque(joint_pos)
    
    # Apply WBC error gate (disable if WBC error > 30%)
    wbc_gate = jnp.where(
        wbc_error_magnitude > self.config.wbc_error_threshold,
        0.0,
        1.0,
    )
    tau_posture = tau_posture * wbc_gate
    
    # Apply momentum coordinator gate (reduce to 50% if active)
    momentum_gate = jnp.where(
        momentum_magnitude > 0.1,
        self.config.momentum_active_scale,
        1.0,
    )
    tau_posture = tau_posture * momentum_gate
    
    # Clip to 20% authority budget
    tau_posture = self.clip_to_authority_budget(tau_posture)
    
    return tau_posture
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_posture_regularizer.py::test_integrated_posture_regularizer -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add wheeled_biped/controllers/posture_regularizer.py tests/test_posture_regularizer.py
git commit -m "feat: implement integrated posture regularizer with two-level gating"
```

---

## Task 7: Add Integration Test

**Files:**
- Modify: `tests/test_posture_regularizer.py`

- [ ] **Step 1: Write the integration test**

```python
# tests/test_posture_regularizer.py (add to existing file)

def test_posture_regularizer_integration_no_nan():
    """Integration test: 100-step rollout produces no NaN."""
    config = PostureRegularizerConfig(
        k_posture=2.0,
        hip_roll_deadband=0.05,
        hip_yaw_deadband=0.03,
        hip_pitch_deadband=0.08,
        knee_deadband=0.10,
        wbc_error_threshold=0.3,
        momentum_active_scale=0.5,
        posture_authority_budget=0.2,
    )
    regularizer = PostureRegularizer(config)
    
    # Run 100-step rollout with varying conditions
    for step in range(100):
        # Mock joint positions with time-varying errors
        joint_pos = jnp.array([
            0.05 * jnp.sin(step * 0.05),  # left hip roll
            0.03 * jnp.cos(step * 0.06),  # left hip yaw
            0.08 * jnp.sin(step * 0.04),  # left hip pitch
            0.10 * jnp.cos(step * 0.07),  # left knee
            0.0,  # left wheel
            0.05 * jnp.sin(step * 0.05),  # right hip roll
            0.03 * jnp.cos(step * 0.06),  # right hip yaw
            0.08 * jnp.sin(step * 0.04),  # right hip pitch
            0.10 * jnp.cos(step * 0.07),  # right knee
            0.0,  # right wheel
        ])
        
        # Time-varying WBC error and momentum magnitude
        wbc_error_magnitude = 0.15 + 0.1 * jnp.sin(step * 0.03)
        momentum_magnitude = 0.5 + 0.3 * jnp.cos(step * 0.08)
        
        # Compute posture regularizer torque
        tau_posture = regularizer.compute_posture_regularizer_torque(
            joint_pos, wbc_error_magnitude, momentum_magnitude
        )
        
        # Verify no NaN
        assert not jnp.any(jnp.isnan(tau_posture)), f"NaN at step {step}"
        
        # Verify within authority budget
        assert jnp.max(jnp.abs(tau_posture)) <= 6.0, f"Budget exceeded at step {step}"
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_posture_regularizer.py::test_posture_regularizer_integration_no_nan -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_posture_regularizer.py
git commit -m "test: add 100-step integration test for posture regularizer"
```

---

## Task 8: Create Configuration File

**Files:**
- Create: `configs/controllers/posture_regularizer.yaml`

- [ ] **Step 1: Create posture regularizer config**

```yaml
# configs/controllers/posture_regularizer.yaml
# Posture regularization configuration for Level 3 stabilization

controller_type: posture_regularizer

# Proportional gain
k_posture: 2.0  # Weak compared to WBC gains

# Per-joint deadbands (radians)
hip_roll_deadband: 0.05  # ±2.9° - allow lateral sway
hip_yaw_deadband: 0.03  # ±1.7° - tighter, yaw drift is bad
hip_pitch_deadband: 0.08  # ±4.6° - allow squat variation
knee_deadband: 0.10  # ±5.7° - allow knee bend variation
wheel_deadband: 0.0  # wheels don't have posture target

# Gating thresholds
wbc_error_threshold: 0.3  # 30% of WBC capacity
momentum_active_scale: 0.5  # 50% authority when momentum active

# Authority budget
posture_authority_budget: 0.2  # 20% of actuator range

# Evaluation metadata
level: "Level 3 Stabilization"
description: "Permissive posture restoration with two-level gating"
baseline_comparison: "Phase 2 Centroidal WBC + Phase 3 Momentum Coordinator"
```

- [ ] **Step 2: Commit configuration file**

```bash
git add configs/controllers/posture_regularizer.yaml
git commit -m "config: add posture regularizer configuration"
```

---

## Phase 4 Completion Checklist

Before proceeding to integration, verify:

- [ ] All unit tests pass: `pytest tests/test_posture_regularizer.py -v`
- [ ] Integration test passes: 100-step rollout produces no NaN
- [ ] Posture restoration respects per-joint deadbands
- [ ] WBC error gate disables posture when error > 30%
- [ ] Momentum coordinator gate reduces authority to 50% when active
- [ ] Authority budget clipping works (max 20% of actuator range)
- [ ] Integrated regularizer combines all components with two-level gating
- [ ] Configuration file created
- [ ] All files committed to git

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-15-phase4-posture-regularization.md`.

**Two execution options:**

**1. Subagent-Driven (recommended)** - Dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**

