# Unified QP Force Distribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace broken force distributor with OSQP-JAX based unified optimization that properly handles wheeled biped morphology (zero lateral wheel separation).

**Architecture:** Single QP simultaneously optimizes wheel contact forces and hip roll torques to achieve desired centroidal wrench. Uses contact Jacobians to map forces/torques to joint space, with hard constraints on contact compression and torque limits.

**Tech Stack:** JAX, OSQP-JAX, MuJoCo, NumPy

---

## File Structure

**New files to create:**
- `wheeled_biped/controllers/unified_force_distributor.py` - OSQP-JAX based QP solver
- `tests/test_unified_force_distributor.py` - Unit tests for force distributor
- `tests/test_hierarchical_controller_integration.py` - Integration tests

**Files to modify:**
- `wheeled_biped/controllers/contact_jacobian.py` - Extend for hip roll mapping and wrench matrix
- `wheeled_biped/controllers/centroidal_balance_controller.py` - Add wrench output method
- `scripts/simulate_hierarchical_controller.py` - Wire in new force distributor

**Dependencies to add:**
- `osqp-jax` - QP solver for JAX

---

## Phase 1: Contact Jacobian Foundation (Day 1)

### Task 1: Install OSQP-JAX Dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add osqp-jax to requirements**

Add to `requirements.txt`:
```
osqp-jax>=0.1.0
```

- [ ] **Step 2: Install dependency**

Run: `pip install osqp-jax`
Expected: Package installs successfully

- [ ] **Step 3: Verify import**

Run: `python -c "import osqp_jax; print('OSQP-JAX version:', osqp_jax.__version__)"`
Expected: Version prints without error

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "deps: add osqp-jax for QP-based force distribution"
```

---

### Task 2: Extend ContactJacobian for Hip Roll Mapping

**Files:**
- Modify: `wheeled_biped/controllers/contact_jacobian.py`
- Test: `tests/test_contact_jacobian.py`

- [ ] **Step 1: Write failing test for hip roll Jacobian**

Create `tests/test_contact_jacobian.py`:
```python
"""Tests for contact Jacobian computation."""
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from wheeled_biped.controllers.contact_jacobian import ContactJacobian


def test_compute_hip_roll_jacobian():
    """Test hip roll torque to joint torque mapping."""
    # Load robot model
    mj_model = mujoco.MjModel.from_xml_path("assets/robot/wheeled_biped_real.xml")
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mujoco.mj_forward(mj_model, mj_data)
    
    # Create Jacobian computer
    jac_computer = ContactJacobian(mj_model)
    
    # Compute hip roll Jacobian
    J_hip_roll = jac_computer.compute_hip_roll_jacobian(mj_data)
    
    # Should be (2, 10) - maps 2 hip roll torques to 10 joint torques
    assert J_hip_roll.shape == (2, 10)
    
    # Hip roll joints are indices 0 and 5
    # Left hip roll (index 0) should map to joint 0
    assert J_hip_roll[0, 0] == 1.0
    assert jnp.sum(jnp.abs(J_hip_roll[0, 1:])) < 1e-6  # Other joints zero
    
    # Right hip roll (index 5) should map to joint 5
    assert J_hip_roll[1, 5] == 1.0
    assert jnp.sum(jnp.abs(J_hip_roll[1, :5])) < 1e-6  # Joints 0-4 zero
    assert jnp.sum(jnp.abs(J_hip_roll[1, 6:])) < 1e-6  # Joints 6-9 zero
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_contact_jacobian.py::test_compute_hip_roll_jacobian -v`
Expected: FAIL with "AttributeError: 'ContactJacobian' object has no attribute 'compute_hip_roll_jacobian'"

- [ ] **Step 3: Implement compute_hip_roll_jacobian**

Add to `wheeled_biped/controllers/contact_jacobian.py`:
```python
def compute_hip_roll_jacobian(self, mj_data: mujoco.MjData) -> Array:
    """Compute hip roll torque to joint torque mapping.
    
    Hip roll torques directly map to their respective joint torques.
    
    Args:
        mj_data: MuJoCo data with current robot state
        
    Returns:
        J_hip_roll (2, 10): maps [tau_hip_roll_L, tau_hip_roll_R] to joint torques
    """
    # Hip roll joints are indices 0 (left) and 5 (right)
    J = jnp.zeros((2, 10))
    J = J.at[0, 0].set(1.0)  # Left hip roll
    J = J.at[1, 5].set(1.0)  # Right hip roll
    return J
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_contact_jacobian.py::test_compute_hip_roll_jacobian -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add wheeled_biped/controllers/contact_jacobian.py tests/test_contact_jacobian.py
git commit -m "feat: add hip roll Jacobian computation

Maps hip roll torques directly to joint torques for unified QP."
```

---

### Task 3: Implement Wrench Matrix Builder

**Files:**
- Modify: `wheeled_biped/controllers/contact_jacobian.py`
- Test: `tests/test_contact_jacobian.py`

- [ ] **Step 1: Write failing test for wrench matrix**

Add to `tests/test_contact_jacobian.py`:
```python
def test_build_wrench_matrix():
    """Test A_wrench matrix construction from Jacobians."""
    # Load robot model
    mj_model = mujoco.MjModel.from_xml_path("assets/robot/wheeled_biped_real.xml")
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mujoco.mj_forward(mj_model, mj_data)
    
    # Create Jacobian computer
    jac_computer = ContactJacobian(mj_model)
    
    # Get CoM position
    com_pos = jnp.array([0.0, 0.0, 0.6])
    
    # Build wrench matrix
    A_wrench = jac_computer.build_wrench_matrix(mj_data, com_pos)
    
    # Should be (6, 8) - maps 8 decision vars to 6D wrench
    assert A_wrench.shape == (6, 8)
    
    # Verify structure: forces map through Jacobians, hip rolls contribute to Mx
    # Row 3 (Mx - roll moment) should have non-zero entries for hip roll torques
    assert jnp.abs(A_wrench[3, 6]) > 0.5  # Left hip roll contributes to Mx
    assert jnp.abs(A_wrench[3, 7]) > 0.5  # Right hip roll contributes to Mx
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_contact_jacobian.py::test_build_wrench_matrix -v`
Expected: FAIL with "AttributeError: 'ContactJacobian' object has no attribute 'build_wrench_matrix'"

- [ ] **Step 3: Implement build_wrench_matrix**

Add to `wheeled_biped/controllers/contact_jacobian.py`:
```python
def build_wrench_matrix(
    self,
    mj_data: mujoco.MjData,
    com_pos: Array,
) -> Array:
    """Build A_wrench matrix mapping decision variables to centroidal wrench.
    
    Maps [f_left (3), f_right (3), tau_hip_roll (2)] to [Fx, Fy, Fz, Mx, My, Mz].
    
    Args:
        mj_data: MuJoCo data with current robot state
        com_pos: Center of mass position (3,) in world frame
        
    Returns:
        A_wrench (6, 8): wrench mapping matrix
    """
    # Get wheel Jacobians
    J_left, J_right = self.compute_wheel_jacobians(mj_data)
    
    # Get wheel positions in world frame
    l_wheel_pos = jnp.array(mj_data.xpos[self.l_wheel_id])
    r_wheel_pos = jnp.array(mj_data.xpos[self.r_wheel_id])
    
    # Compute moment arms relative to CoM
    r_left = l_wheel_pos - com_pos
    r_right = r_wheel_pos - com_pos
    
    # Build wrench matrix (6, 8)
    A = jnp.zeros((6, 8))
    
    # Forces (rows 0-2): sum of wheel forces
    # Fx = f_left_x + f_right_x
    A = A.at[0, 0].set(1.0)  # f_left_x
    A = A.at[0, 3].set(1.0)  # f_right_x
    
    # Fy = f_left_y + f_right_y
    A = A.at[1, 1].set(1.0)  # f_left_y
    A = A.at[1, 4].set(1.0)  # f_right_y
    
    # Fz = f_left_z + f_right_z
    A = A.at[2, 2].set(1.0)  # f_left_z
    A = A.at[2, 5].set(1.0)  # f_right_z
    
    # Moments (rows 3-5): r × f for each wheel + hip roll contribution
    # Mx (roll) = r_y * Fz - r_z * Fy for each wheel + hip roll torques
    A = A.at[3, 1].set(-r_left[2])  # -r_left_z * f_left_y
    A = A.at[3, 2].set(r_left[1])   # r_left_y * f_left_z
    A = A.at[3, 4].set(-r_right[2]) # -r_right_z * f_right_y
    A = A.at[3, 5].set(r_right[1])  # r_right_y * f_right_z
    A = A.at[3, 6].set(1.0)         # tau_hip_roll_L
    A = A.at[3, 7].set(1.0)         # tau_hip_roll_R
    
    # My (pitch) = r_z * Fx - r_x * Fz for each wheel
    A = A.at[4, 0].set(r_left[2])   # r_left_z * f_left_x
    A = A.at[4, 2].set(-r_left[0])  # -r_left_x * f_left_z
    A = A.at[4, 3].set(r_right[2])  # r_right_z * f_right_x
    A = A.at[4, 5].set(-r_right[0]) # -r_right_x * f_right_z
    
    # Mz (yaw) = r_x * Fy - r_y * Fx for each wheel
    A = A.at[5, 0].set(-r_left[1])  # -r_left_y * f_left_x
    A = A.at[5, 1].set(r_left[0])   # r_left_x * f_left_y
    A = A.at[5, 3].set(-r_right[1]) # -r_right_y * f_right_x
    A = A.at[5, 4].set(r_right[0])  # r_right_x * f_right_y
    
    return A
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_contact_jacobian.py::test_build_wrench_matrix -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add wheeled_biped/controllers/contact_jacobian.py tests/test_contact_jacobian.py
git commit -m "feat: implement wrench matrix builder

Constructs A_wrench (6x8) mapping decision variables to centroidal wrench.
Includes moment arm computation and hip roll contribution to roll moment."
```

---

## Phase 2: OSQP-JAX Integration (Day 2)

### Task 4: Create UnifiedForceDistributor Skeleton

**Files:**
- Create: `wheeled_biped/controllers/unified_force_distributor.py`
- Create: `tests/test_unified_force_distributor.py`

- [ ] **Step 1: Write failing test for distributor creation**

Create `tests/test_unified_force_distributor.py`:
```python
"""Tests for unified QP force distributor."""
import jax.numpy as jnp
import pytest

from wheeled_biped.controllers.unified_force_distributor import (
    UnifiedForceDistributor,
    UnifiedForceDistributorConfig,
)


def test_create_distributor():
    """Test force distributor creation with default config."""
    config = UnifiedForceDistributorConfig(
        w_force=0.01,
        w_torque=0.1,
        w_smoothness=0.5,
        tau_max=30.0,
        max_iter=10,
        eps_abs=1e-3,
        eps_rel=1e-3,
    )
    
    distributor = UnifiedForceDistributor(config)
    
    assert distributor.config.w_force == 0.01
    assert distributor.config.w_torque == 0.1
    assert distributor.config.max_iter == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_unified_force_distributor.py::test_create_distributor -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'wheeled_biped.controllers.unified_force_distributor'"

- [ ] **Step 3: Create distributor skeleton**

Create `wheeled_biped/controllers/unified_force_distributor.py`:
```python
"""Unified QP force distributor for wheeled biped.

Maps desired centroidal wrench to wheel contact forces and hip roll torques
using OSQP-JAX optimization with hard contact and torque constraints.
"""

import chex
import jax.numpy as jnp
from jax import Array


@chex.dataclass(frozen=True)
class UnifiedForceDistributorConfig:
    """Configuration for unified force distributor."""
    # Cost function weights
    w_force: float = 0.01  # Penalty on contact forces
    w_torque: float = 0.1  # Penalty on hip roll torques
    w_smoothness: float = 0.5  # Penalty on deviation from previous solution
    
    # Constraints
    tau_max: float = 30.0  # Maximum hip roll torque (Nm)
    
    # OSQP solver settings (100Hz optimized)
    max_iter: int = 10
    eps_abs: float = 1e-3
    eps_rel: float = 1e-3
    polish: bool = False


class UnifiedForceDistributor:
    """Unified QP force distributor."""
    
    def __init__(self, config: UnifiedForceDistributorConfig):
        """Initialize force distributor.
        
        Args:
            config: UnifiedForceDistributorConfig with weights and solver settings
        """
        self.config = config
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_unified_force_distributor.py::test_create_distributor -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add wheeled_biped/controllers/unified_force_distributor.py tests/test_unified_force_distributor.py
git commit -m "feat: create unified force distributor skeleton

Adds config dataclass and empty distributor class for OSQP-JAX integration."
```

---

### Task 5: Implement QP Matrix Builders

**Files:**
- Modify: `wheeled_biped/controllers/unified_force_distributor.py`
- Test: `tests/test_unified_force_distributor.py`

- [ ] **Step 1: Write failing test for cost matrix**

Add to `tests/test_unified_force_distributor.py`:
```python
def test_build_cost_matrix():
    """Test quadratic cost matrix construction."""
    config = UnifiedForceDistributorConfig(
        w_force=0.01,
        w_torque=0.1,
        w_smoothness=0.5,
    )
    distributor = UnifiedForceDistributor(config)
    
    P = distributor.build_cost_matrix()
    
    # Should be (8, 8) diagonal
    assert P.shape == (8, 8)
    
    # Check diagonal values
    # Forces (indices 0-5): w_force
    assert jnp.allclose(jnp.diag(P)[:6], 0.01)
    
    # Torques (indices 6-7): w_torque
    assert jnp.allclose(jnp.diag(P)[6:], 0.1)
    
    # Off-diagonal should be zero
    assert jnp.allclose(P - jnp.diag(jnp.diag(P)), 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_unified_force_distributor.py::test_build_cost_matrix -v`
Expected: FAIL with "AttributeError: 'UnifiedForceDistributor' object has no attribute 'build_cost_matrix'"

- [ ] **Step 3: Implement build_cost_matrix**

Add to `wheeled_biped/controllers/unified_force_distributor.py`:
```python
def build_cost_matrix(self) -> Array:
    """Build quadratic cost matrix P for QP.
    
    Returns:
        P (8, 8): diagonal cost matrix
    """
    # Diagonal weights: [f_left (3), f_right (3), tau_hip_roll (2)]
    weights = jnp.array([
        self.config.w_force, self.config.w_force, self.config.w_force,  # f_left
        self.config.w_force, self.config.w_force, self.config.w_force,  # f_right
        self.config.w_torque, self.config.w_torque,  # tau_hip_roll
    ])
    
    P = jnp.diag(weights)
    return P
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_unified_force_distributor.py::test_build_cost_matrix -v`
Expected: PASS

- [ ] **Step 5: Write failing test for linear cost**

Add to `tests/test_unified_force_distributor.py`:
```python
def test_build_linear_cost():
    """Test linear cost vector for smoothness."""
    config = UnifiedForceDistributorConfig(
        w_force=0.01,
        w_torque=0.1,
        w_smoothness=0.5,
    )
    distributor = UnifiedForceDistributor(config)
    
    # Previous solution
    x_prev = jnp.array([1.0, 2.0, 75.0, 1.0, 2.0, 75.0, 5.0, 5.0])
    
    P = distributor.build_cost_matrix()
    q = distributor.build_linear_cost(P, x_prev)
    
    # Should be (8,)
    assert q.shape == (8,)
    
    # q = -2 * w_smoothness * P @ x_prev
    expected = -2.0 * 0.5 * P @ x_prev
    assert jnp.allclose(q, expected)
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_unified_force_distributor.py::test_build_linear_cost -v`
Expected: FAIL with "AttributeError: 'UnifiedForceDistributor' object has no attribute 'build_linear_cost'"

- [ ] **Step 7: Implement build_linear_cost**

Add to `wheeled_biped/controllers/unified_force_distributor.py`:
```python
def build_linear_cost(self, P: Array, x_prev: Array) -> Array:
    """Build linear cost vector q for smoothness penalty.
    
    Args:
        P: Quadratic cost matrix (8, 8)
        x_prev: Previous solution (8,)
        
    Returns:
        q (8,): linear cost vector
    """
    # Penalize deviation from previous solution
    q = -2.0 * self.config.w_smoothness * P @ x_prev
    return q
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/test_unified_force_distributor.py::test_build_linear_cost -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add wheeled_biped/controllers/unified_force_distributor.py tests/test_unified_force_distributor.py
git commit -m "feat: implement QP cost matrix builders

Adds build_cost_matrix() for quadratic cost and build_linear_cost() 
for smoothness penalty."
```

---

### Task 6: Implement Constraint Matrix Builders

**Files:**
- Modify: `wheeled_biped/controllers/unified_force_distributor.py`
- Test: `tests/test_unified_force_distributor.py`

- [ ] **Step 1: Write failing test for constraint matrix**

Add to `tests/test_unified_force_distributor.py`:
```python
def test_build_constraint_matrix():
    """Test constraint matrix construction."""
    config = UnifiedForceDistributorConfig(tau_max=30.0)
    distributor = UnifiedForceDistributor(config)
    
    # Mock wrench matrix (6, 8)
    A_wrench = jnp.eye(6, 8)
    
    # Desired wrench
    desired_wrench = jnp.array([0.0, 0.0, 150.0, 0.0, 0.0, 0.0])
    
    A, l, u = distributor.build_constraint_matrix(A_wrench, desired_wrench)
    
    # A should be (10, 8): 6 equality + 4 inequality
    assert A.shape == (10, 8)
    
    # First 6 rows are wrench equality constraints
    assert jnp.allclose(A[:6, :], A_wrench)
    
    # Lower bounds: wrench equality + contact/torque limits
    assert l.shape == (10,)
    assert jnp.allclose(l[:6], desired_wrench)  # Equality constraints
    assert l[6] == 0.0  # f_left_z >= 0
    assert l[7] == 0.0  # f_right_z >= 0
    assert l[8] == -30.0  # tau_hip_roll_L >= -30
    assert l[9] == -30.0  # tau_hip_roll_R >= -30
    
    # Upper bounds: wrench equality + torque limits
    assert u.shape == (10,)
    assert jnp.allclose(u[:6], desired_wrench)  # Equality constraints
    assert u[6] == jnp.inf  # f_left_z <= inf
    assert u[7] == jnp.inf  # f_right_z <= inf
    assert u[8] == 30.0  # tau_hip_roll_L <= 30
    assert u[9] == 30.0  # tau_hip_roll_R <= 30
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_unified_force_distributor.py::test_build_constraint_matrix -v`
Expected: FAIL with "AttributeError: 'UnifiedForceDistributor' object has no attribute 'build_constraint_matrix'"

- [ ] **Step 3: Implement build_constraint_matrix**

Add to `wheeled_biped/controllers/unified_force_distributor.py`:
```python
def build_constraint_matrix(
    self,
    A_wrench: Array,
    desired_wrench: Array,
) -> tuple[Array, Array, Array]:
    """Build constraint matrices for QP.
    
    Constructs:
    - Equality constraints: A_wrench @ x = desired_wrench
    - Inequality constraints: contact forces compressive, torque limits
    
    Args:
        A_wrench: Wrench mapping matrix (6, 8)
        desired_wrench: Desired centroidal wrench (6,)
        
    Returns:
        Tuple of (A, l, u) where:
        - A (10, 8): constraint matrix
        - l (10,): lower bounds
        - u (10,): upper bounds
    """
    # Constraint matrix: [equality; inequality]
    # Equality (6 rows): A_wrench @ x = desired_wrench
    # Inequality (4 rows): [0 0 1 0 0 0 0 0] @ x >= 0 (f_left_z)
    #                      [0 0 0 0 0 1 0 0] @ x >= 0 (f_right_z)
    #                      [0 0 0 0 0 0 1 0] @ x in [-tau_max, tau_max]
    #                      [0 0 0 0 0 0 0 1] @ x in [-tau_max, tau_max]
    
    # Build inequality constraint rows
    ineq_rows = jnp.array([
        [0, 0, 1, 0, 0, 0, 0, 0],  # f_left_z
        [0, 0, 0, 0, 0, 1, 0, 0],  # f_right_z
        [0, 0, 0, 0, 0, 0, 1, 0],  # tau_hip_roll_L
        [0, 0, 0, 0, 0, 0, 0, 1],  # tau_hip_roll_R
    ])
    
    # Stack equality and inequality
    A = jnp.vstack([A_wrench, ineq_rows])  # (10, 8)
    
    # Lower bounds
    l_eq = desired_wrench  # Equality: A @ x = b means l = u = b
    l_ineq = jnp.array([
        0.0,  # f_left_z >= 0
        0.0,  # f_right_z >= 0
        -self.config.tau_max,  # tau_hip_roll_L >= -tau_max
        -self.config.tau_max,  # tau_hip_roll_R >= -tau_max
    ])
    l = jnp.concatenate([l_eq, l_ineq])  # (10,)
    
    # Upper bounds
    u_eq = desired_wrench  # Equality: A @ x = b means l = u = b
    u_ineq = jnp.array([
        jnp.inf,  # f_left_z <= inf (no upper limit)
        jnp.inf,  # f_right_z <= inf
        self.config.tau_max,  # tau_hip_roll_L <= tau_max
        self.config.tau_max,  # tau_hip_roll_R <= tau_max
    ])
    u = jnp.concatenate([u_eq, u_ineq])  # (10,)
    
    return A, l, u
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_unified_force_distributor.py::test_build_constraint_matrix -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add wheeled_biped/controllers/unified_force_distributor.py tests/test_unified_force_distributor.py
git commit -m "feat: implement constraint matrix builder

Constructs QP constraint matrices for wrench equality and 
contact/torque inequality constraints."
```

---

### Task 7: Implement OSQP Solver Integration

**Files:**
- Modify: `wheeled_biped/controllers/unified_force_distributor.py`
- Test: `tests/test_unified_force_distributor.py`

- [ ] **Step 1: Write failing test for solve method**

Add to `tests/test_unified_force_distributor.py`:
```python
def test_solve_simple_wrench():
    """Test QP solve for simple vertical force distribution."""
    import mujoco
    from wheeled_biped.controllers.contact_jacobian import ContactJacobian
    
    config = UnifiedForceDistributorConfig(
        w_force=0.01,
        w_torque=0.1,
        w_smoothness=0.5,
        tau_max=30.0,
    )
    distributor = UnifiedForceDistributor(config)
    
    # Load robot model
    mj_model = mujoco.MjModel.from_xml_path("assets/robot/wheeled_biped_real.xml")
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mujoco.mj_forward(mj_model, mj_data)
    
    # Create Jacobian computer
    jac_computer = ContactJacobian(mj_model)
    
    # Simple wrench: 150N vertical force (gravity compensation)
    desired_wrench = jnp.array([0.0, 0.0, 150.0, 0.0, 0.0, 0.0])
    com_pos = jnp.array([0.0, 0.0, 0.6])
    
    # Build wrench matrix
    A_wrench = jac_computer.build_wrench_matrix(mj_data, com_pos)
    
    # Solve
    solution = distributor.solve(A_wrench, desired_wrench, x_prev=None)
    
    # Should return 8D solution
    assert solution.shape == (8,)
    
    # Vertical forces should sum to ~150N
    fz_total = solution[2] + solution[5]
    assert jnp.abs(fz_total - 150.0) < 1.0
    
    # Contact forces should be compressive
    assert solution[2] >= 0.0  # f_left_z
    assert solution[5] >= 0.0  # f_right_z
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_unified_force_distributor.py::test_solve_simple_wrench -v`
Expected: FAIL with "AttributeError: 'UnifiedForceDistributor' object has no attribute 'solve'"

- [ ] **Step 3: Implement solve method**

Add to `wheeled_biped/controllers/unified_force_distributor.py`:
```python
import osqp_jax

def solve(
    self,
    A_wrench: Array,
    desired_wrench: Array,
    x_prev: Array | None = None,
) -> Array:
    """Solve QP to distribute desired wrench to forces/torques.
    
    Args:
        A_wrench: Wrench mapping matrix (6, 8)
        desired_wrench: Desired centroidal wrench (6,)
        x_prev: Previous solution for warm start (8,), or None for cold start
        
    Returns:
        solution (8,): [f_left (3), f_right (3), tau_hip_roll (2)]
    """
    # Build cost matrices
    P = self.build_cost_matrix()
    
    if x_prev is None:
        # Cold start: zero previous solution
        x_prev = jnp.zeros(8)
        # Initialize with equal weight distribution
        x_prev = x_prev.at[2].set(desired_wrench[2] / 2.0)  # f_left_z
        x_prev = x_prev.at[5].set(desired_wrench[2] / 2.0)  # f_right_z
    
    q = self.build_linear_cost(P, x_prev)
    
    # Build constraint matrices
    A, l, u = self.build_constraint_matrix(A_wrench, desired_wrench)
    
    # Solve QP using OSQP-JAX
    # Convert to float64 for numerical stability
    P_np = np.array(P, dtype=np.float64)
    q_np = np.array(q, dtype=np.float64)
    A_np = np.array(A, dtype=np.float64)
    l_np = np.array(l, dtype=np.float64)
    u_np = np.array(u, dtype=np.float64)
    x_init_np = np.array(x_prev, dtype=np.float64)
    
    # Create OSQP solver
    solver = osqp_jax.OSQP()
    
    # Solve
    result = solver.solve(
        P=P_np,
        q=q_np,
        A=A_np,
        l=l_np,
        u=u_np,
        x_init=x_init_np,
        max_iter=self.config.max_iter,
        eps_abs=self.config.eps_abs,
        eps_rel=self.config.eps_rel,
        polish=self.config.polish,
    )
    
    # Convert back to JAX array
    solution = jnp.array(result.x)
    
    return solution
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_unified_force_distributor.py::test_solve_simple_wrench -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add wheeled_biped/controllers/unified_force_distributor.py tests/test_unified_force_distributor.py
git commit -m "feat: implement OSQP-JAX solver integration

Adds solve() method that formulates and solves QP for force distribution.
Includes warm-starting and 100Hz-optimized solver settings."
```

---

## Phase 3: Centroidal Controller Refactor (Day 3)

### Task 8: Add Wrench Output Method to CentroidalBalanceController

**Files:**
- Modify: `wheeled_biped/controllers/centroidal_balance_controller.py`
- Test: `tests/test_centroidal_balance_controller.py`

- [ ] **Step 1: Write failing test for wrench computation**

Add to `tests/test_centroidal_balance_controller.py`:
```python
def test_compute_desired_wrench():
    """Test centroidal wrench computation from control objectives."""
    from wheeled_biped.controllers.centroidal_state_estimator import CentroidalState
    
    config = CentroidalBalanceConfig(
        k_roll=20.0,
        k_roll_rate=4.0,
        k_com_lateral=15.0,
        k_height=5.0,
    )
    controller = CentroidalBalanceController(config)
    
    # Mock observation with roll error
    obs = jnp.zeros(42)
    obs = obs.at[0].set(0.0)  # gx
    obs = obs.at[1].set(0.1)  # gy (roll = atan2(0.1, 1.0) ≈ 0.1 rad)
    obs = obs.at[2].set(1.0)  # gz
    obs = obs.at[6].set(0.05)  # roll_rate
    obs = obs.at[36].set(0.6)  # height_cmd
    obs = obs.at[37].set(0.55)  # current height (0.05m error)
    
    # Mock state
    state = CentroidalState(
        com_pos=jnp.array([0.0, 0.02, 0.55]),  # 2cm lateral error
        com_vel=jnp.array([0.0, 0.1, 0.0]),
        capture_point=jnp.array([0.0, 0.05]),  # 5cm lateral divergence
        divergence=jnp.array([0.0, 0.05]),
        linear_momentum=jnp.zeros(3),
        angular_momentum=jnp.zeros(3),
        left_wheel_contact=True,
        right_wheel_contact=True,
        left_wheel_force=75.0,
        right_wheel_force=75.0,
    )
    
    # Compute wrench
    wrench = controller.compute_desired_wrench(obs, state)
    
    # Should be (6,): [Fx, Fy, Fz, Mx, My, Mz]
    assert wrench.shape == (6,)
    
    # Roll moment (Mx) should be non-zero due to roll error
    assert jnp.abs(wrench[3]) > 0.1
    
    # Lateral force (Fy) should be non-zero due to CoM/CP error
    assert jnp.abs(wrench[1]) > 0.1
    
    # Vertical force (Fz) should be non-zero due to height error
    assert jnp.abs(wrench[2]) > 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_centroidal_balance_controller.py::test_compute_desired_wrench -v`
Expected: FAIL with "AttributeError: 'CentroidalBalanceController' object has no attribute 'compute_desired_wrench'"

- [ ] **Step 3: Implement compute_desired_wrench**

Add to `wheeled_biped/controllers/centroidal_balance_controller.py`:
```python
def compute_desired_wrench(self, obs: Array, state: CentroidalState) -> Array:
    """Compute desired centroidal wrench from control objectives.
    
    Combines roll stabilization, CoM regulation, CP tracking, and height control
    into a single 6D wrench command.
    
    Args:
        obs: Observation array (42,)
        state: CentroidalState with CoM, CP, contact info
        
    Returns:
        wrench (6,): [Fx, Fy, Fz, Mx, My, Mz] desired centroidal wrench
    """
    # Extract robot mass for force computation
    robot_mass = 15.0  # kg
    g = 9.81  # m/s^2
    
    # Roll stabilization
    roll = jnp.arctan2(obs[1], obs[2])
    roll_rate = obs[6]
    Mx = -self.config.k_roll * roll - self.config.k_roll_rate * roll_rate
    
    # CoM lateral regulation
    com_y = state.com_pos[1]
    com_vy = state.com_vel[1]
    com_y_error = jnp.where(
        jnp.abs(com_y) < self.config.com_deadband_lateral,
        0.0,
        com_y
    )
    Fy_com = -self.config.k_com_lateral * com_y_error - self.config.k_com_lateral_damping * com_vy
    
    # Capture point lateral tracking
    cp_y = state.capture_point[1]
    cp_y_error = jnp.where(
        jnp.abs(cp_y) < self.config.cp_deadband,
        0.0,
        cp_y
    )
    Fy_cp = -self.config.k_cp_lateral * cp_y_error
    
    # Total lateral force
    Fy = self.config.w_com * Fy_com + self.config.w_cp * Fy_cp
    
    # Height control
    height_cmd = obs[36]
    height_current = obs[37]
    height_error = height_cmd - height_current
    Fz = robot_mass * g + self.config.k_height * height_error
    
    # Sagittal control (similar to lateral, but for x-axis)
    com_x = state.com_pos[0]
    com_vx = state.com_vel[0]
    com_x_error = jnp.where(
        jnp.abs(com_x) < self.config.com_deadband_sagittal,
        0.0,
        com_x
    )
    Fx_com = -self.config.k_com_sagittal * com_x_error - self.config.k_com_sagittal_damping * com_vx
    
    cp_x = state.capture_point[0]
    cp_x_error = jnp.where(
        jnp.abs(cp_x) < self.config.cp_deadband,
        0.0,
        cp_x
    )
    Fx_cp = -self.config.k_cp_sagittal * cp_x_error
    
    Fx = self.config.w_com * Fx_com + self.config.w_cp * Fx_cp
    
    # Pitch moment (My) from CP sagittal
    My = -self.config.k_cp_sagittal * cp_x_error
    
    # Yaw moment (Mz) - minimal for now
    Mz = 0.0
    
    # Assemble wrench
    wrench = jnp.array([Fx, Fy, Fz, Mx, My, Mz])
    
    return wrench
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_centroidal_balance_controller.py::test_compute_desired_wrench -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add wheeled_biped/controllers/centroidal_balance_controller.py tests/test_centroidal_balance_controller.py
git commit -m "feat: add wrench output method to centroidal controller

Implements compute_desired_wrench() that combines roll, CoM, CP, and height
objectives into 6D centroidal wrench for QP-based force distribution."
```

---

## Phase 4: Integration and Validation (Day 4)

### Task 9: Wire Unified Force Distributor into Simulation

**Files:**
- Modify: `scripts/simulate_hierarchical_controller.py`

- [ ] **Step 1: Add imports for new components**

Add to imports section of `scripts/simulate_hierarchical_controller.py`:
```python
from wheeled_biped.controllers.unified_force_distributor import (
    UnifiedForceDistributor,
    UnifiedForceDistributorConfig,
)
from wheeled_biped.controllers.contact_jacobian import ContactJacobian
```

- [ ] **Step 2: Initialize force distributor**

Replace WBC controller initialization with:
```python
# Initialize unified force distributor
print("\nInitializing unified force distributor...")
force_distributor = UnifiedForceDistributor(
    UnifiedForceDistributorConfig(
        w_force=0.01,
        w_torque=0.1,
        w_smoothness=0.5,
        tau_max=30.0,
        max_iter=10,
        eps_abs=1e-3,
        eps_rel=1e-3,
    )
)

# Initialize contact Jacobian computer
contact_jacobian = ContactJacobian(mj_model)

print("[OK] Force distributor initialized")
```

- [ ] **Step 3: Update control loop to use new path**

Replace WBC torque computation with:
```python
# Compute desired wrench from centroidal objectives
desired_wrench = wbc_controller.compute_desired_wrench(obs, centroidal_state)

# Get CoM position for wrench matrix
com_pos = centroidal_state.com_pos

# Build wrench matrix
A_wrench = contact_jacobian.build_wrench_matrix(mj_data, com_pos)

# Solve QP for force distribution
solution = force_distributor.solve(A_wrench, desired_wrench, x_prev=prev_solution)

# Extract forces and torques
f_left = solution[0:3]
f_right = solution[3:6]
tau_hip_roll = solution[6:8]

# Map to joint torques via Jacobian
tau_wbc = contact_jacobian.map_contact_forces_to_torques(mj_data, f_left, f_right)

# Add hip roll torques directly
tau_wbc = tau_wbc.at[0].add(tau_hip_roll[0])  # Left hip roll
tau_wbc = tau_wbc.at[5].add(tau_hip_roll[1])  # Right hip roll

# Store for next iteration
prev_solution = solution
```

- [ ] **Step 4: Add telemetry logging**

Add to telemetry dict:
```python
telemetry["qp_solve_time"] = []
telemetry["qp_iterations"] = []
telemetry["wrench_error"] = []
telemetry["f_left_z"] = []
telemetry["f_right_z"] = []
```

Add to telemetry recording:
```python
telemetry["qp_solve_time"].append(solve_time)
telemetry["qp_iterations"].append(iterations)
telemetry["wrench_error"].append(float(jnp.linalg.norm(A_wrench @ solution - desired_wrench)))
telemetry["f_left_z"].append(float(solution[2]))
telemetry["f_right_z"].append(float(solution[5]))
```

- [ ] **Step 5: Test simulation runs**

Run: `python scripts/simulate_hierarchical_controller.py`
Expected: Robot balances for >1 second, QP solves successfully

- [ ] **Step 6: Commit**

```bash
git add scripts/simulate_hierarchical_controller.py
git commit -m "feat: integrate unified force distributor into simulation

Replaces direct torque output with QP-based force distribution.
Adds telemetry for QP solve time, iterations, and wrench error."
```

---

### Task 10: Add Integration Tests

**Files:**
- Create: `tests/test_hierarchical_controller_integration.py`

- [ ] **Step 1: Write test for 100Hz timing**

Create `tests/test_hierarchical_controller_integration.py`:
```python
"""Integration tests for hierarchical controller with unified force distribution."""
import time

import jax.numpy as jnp
import mujoco
import pytest

from wheeled_biped.controllers.centroidal_balance_controller import (
    CentroidalBalanceConfig,
    CentroidalBalanceController,
)
from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)
from wheeled_biped.controllers.contact_jacobian import ContactJacobian
from wheeled_biped.controllers.unified_force_distributor import (
    UnifiedForceDistributor,
    UnifiedForceDistributorConfig,
)


@pytest.mark.slow
def test_100hz_timing():
    """Test that QP solve completes within 10ms budget for 100Hz control."""
    # Load robot model
    mj_model = mujoco.MjModel.from_xml_path("assets/robot/wheeled_biped_real.xml")
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mujoco.mj_forward(mj_model, mj_data)
    
    # Initialize components
    wbc_controller = CentroidalBalanceController(CentroidalBalanceConfig())
    estimator = CentroidalStateEstimator(CentroidalStateEstimatorConfig(robot_mass=15.0))
    contact_jacobian = ContactJacobian(mj_model)
    force_distributor = UnifiedForceDistributor(UnifiedForceDistributorConfig())
    
    # Run 100 iterations and measure timing
    solve_times = []
    
    for _ in range(100):
        # Estimate state
        obs = jnp.zeros(42)
        obs = obs.at[2].set(1.0)  # gz
        obs = obs.at[36].set(0.6)  # height_cmd
        obs = obs.at[37].set(0.6)  # current_height
        
        state, _ = estimator.estimate(obs, mj_data, None)
        
        # Compute desired wrench
        desired_wrench = wbc_controller.compute_desired_wrench(obs, state)
        
        # Build wrench matrix
        A_wrench = contact_jacobian.build_wrench_matrix(mj_data, state.com_pos)
        
        # Time the solve
        start = time.perf_counter()
        solution = force_distributor.solve(A_wrench, desired_wrench, x_prev=None)
        end = time.perf_counter()
        
        solve_times.append((end - start) * 1000)  # Convert to ms
    
    # Check timing statistics
    mean_time = sum(solve_times) / len(solve_times)
    p95_time = sorted(solve_times)[int(0.95 * len(solve_times))]
    p99_time = sorted(solve_times)[int(0.99 * len(solve_times))]
    
    print(f"\nQP solve timing (100 iterations):")
    print(f"  Mean: {mean_time:.2f} ms")
    print(f"  P95: {p95_time:.2f} ms")
    print(f"  P99: {p99_time:.2f} ms")
    
    # Target: P95 < 10ms for 100Hz control
    assert p95_time < 10.0, f"P95 solve time {p95_time:.2f}ms exceeds 10ms budget"
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_hierarchical_controller_integration.py::test_100hz_timing -v -s`
Expected: PASS with timing statistics printed

- [ ] **Step 3: Commit**

```bash
git add tests/test_hierarchical_controller_integration.py
git commit -m "test: add 100Hz timing integration test

Verifies QP solve completes within 10ms budget at P95."
```

---

### Task 11: Validation and Tuning

**Files:**
- Modify: `scripts/simulate_hierarchical_controller.py`

- [ ] **Step 1: Run extended simulation**

Run: `python scripts/simulate_hierarchical_controller.py`
Observe: survival time, QP convergence rate, contact forces

- [ ] **Step 2: Analyze telemetry**

Check telemetry CSV for:
- QP solve time (target: <10ms at P95)
- Convergence rate (target: >95%)
- Wrench error (target: <0.01 N/Nm)
- Contact forces compressive (fz >= 0)

- [ ] **Step 3: Tune weights if needed**

If robot behavior is suboptimal, adjust weights in config:
- Too aggressive → increase w_force/w_torque
- Too sluggish → decrease w_smoothness
- Hip rolls saturate → decrease w_torque
- Wheels slip → increase w_force

- [ ] **Step 4: Document results**

Create summary of:
- Survival time comparison (old vs new controller)
- QP performance metrics
- Any weight tuning performed

- [ ] **Step 5: Final commit**

```bash
git add scripts/simulate_hierarchical_controller.py
git commit -m "tune: optimize QP weights for 100Hz control

Final tuning based on telemetry analysis. Robot balances for X seconds
with Y% QP convergence rate and Z ms mean solve time."
```

---

## Success Criteria Checklist

- [ ] Robot balances for >5 seconds without falling
- [ ] QP solves in <10ms at P95
- [ ] Wrench error <0.01 N/Nm
- [ ] Contact forces compressive (fz >= 0)
- [ ] Fallback rate <5%
- [ ] New controller >= old controller survival time

---

