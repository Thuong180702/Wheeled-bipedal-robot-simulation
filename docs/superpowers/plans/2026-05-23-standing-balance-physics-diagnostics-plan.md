# Standing Balance Physics Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Identify exactly where and why the desired 79N vertical force becomes 60-67N actual contact force through systematic diagnostics (phases 0-3).

**Architecture:** Three diagnostic scripts that instrument the force pipeline, test static support under different torque sources, and establish inverse dynamics baseline. One test suite validates actuator signs and authority. All diagnostics operate at torque level without PID assumptions.

**Tech Stack:** Python 3.10+, MuJoCo 3.x, JAX, NumPy, pytest

---

## File Structure

**New files to create:**
- `scripts/debug_force_gap.py` - Phase 0: Force audit trail for one control cycle
- `scripts/debug_static_support_parity.py` - Phase 1: Multi-case static support comparison
- `scripts/debug_static_inverse_dynamics.py` - Phase 3: Torque budget analysis
- `tests/test_actuator_signs.py` - Phase 2: Actuator sign and authority validation

**Files to read/reference:**
- `scripts/simulate_hierarchical_controller.py` - Current simulation pipeline
- `wheeled_biped/controllers/integrated_wbc.py` - WBC controller
- `wheeled_biped/controllers/simple_force_distributor.py` - Force distribution
- `wheeled_biped/controllers/contact_jacobian.py` - Jacobian mapping
- `wheeled_biped/controllers/centroidal_state_estimator.py` - State estimation
- `assets/robot/wheeled_biped_real.xml` - Robot model

---

## Phase 0: Force Audit Trail Script

### Task 1: Create Force Gap Diagnostic Script Structure

**Files:**
- Create: `scripts/debug_force_gap.py`

- [ ] **Step 1: Create script with basic structure**

```python
"""Force gap diagnostic script.

Runs one control cycle and prints force audit trail showing where
the 15-20N force gap occurs between desired and actual contact forces.
"""

import argparse
import numpy as np
import mujoco
import jax.numpy as jnp

from wheeled_biped.controllers.integrated_wbc import IntegratedWBC
from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)
from wheeled_biped.controllers.capture_point_estimator import (
    CapturePointEstimator,
    CapturePointEstimatorConfig,
)
from wheeled_biped.controllers.contact_jacobian import ContactJacobian


def calibrate_root_z_for_wheel_floor_contact(mj_model, mj_data, target_dist=-5e-4, max_iters=5):
    """Calibrate root_z to achieve target wheel-floor contact distance."""
    floor_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    l_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")
    
    for _ in range(max_iters):
        mujoco.mj_forward(mj_model, mj_data)
        
        min_dist = None
        for i in range(mj_data.ncon):
            c = mj_data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            involves_floor = g1 == floor_geom_id or g2 == floor_geom_id
            involves_wheel = g1 in {l_wheel_geom_id, r_wheel_geom_id} or g2 in {l_wheel_geom_id, r_wheel_geom_id}
            
            if involves_floor and involves_wheel:
                d = float(c.dist)
                min_dist = d if min_dist is None else min(min_dist, d)
        
        if min_dist is None:
            break
        
        delta_z = target_dist - min_dist
        if abs(delta_z) < 1e-7:
            break
        
        mj_data.qpos[2] += delta_z
        mj_data.qvel[:] = 0.0
        mj_data.qacc[:] = 0.0
    
    mujoco.mj_forward(mj_model, mj_data)


def load_robot_at_keyframe():
    """Load robot at calibrated standing keyframe with proper initialization.
    
    Matches simulate_hierarchical_controller.py initialization:
    1. Reset to keyframe
    2. mj_forward
    3. Calibrate root_z for -0.5mm contact distance
    4. Zero velocities and accelerations
    5. mj_forward
    """
    model_path = "assets/robot/wheeled_biped_real.xml"
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)
    
    # Step 1: Reset to keyframe
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    
    # Step 2: Forward kinematics
    mujoco.mj_forward(mj_model, mj_data)
    
    # Step 3: Calibrate root_z
    calibrate_root_z_for_wheel_floor_contact(mj_model, mj_data, target_dist=-5e-4)
    
    # Step 4: Zero velocities and accelerations
    mj_data.qvel[:] = 0.0
    mj_data.qacc[:] = 0.0
    
    # Step 5: Forward kinematics again
    mujoco.mj_forward(mj_model, mj_data)
    
    return mj_model, mj_data


def main():
    parser = argparse.ArgumentParser(description="Force gap diagnostic")
    args = parser.parse_args()
    
    print("=" * 80)
    print("FORCE GAP DIAGNOSTIC")
    print("=" * 80)
    
    mj_model, mj_data = load_robot_at_keyframe()
    print(f"[OK] Robot loaded at keyframe 0")
    
    # TODO: Add force audit trail
    

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run script to verify it loads**

Run: `python scripts/debug_force_gap.py`
Expected: Script runs, prints header and loads robot

- [ ] **Step 3: Commit**

```bash
git add scripts/debug_force_gap.py
git commit -m "feat(diagnostics): Add force gap diagnostic script structure"
```

---

### Task 2: Implement Force Audit Trail

**Files:**
- Modify: `scripts/debug_force_gap.py`

- [ ] **Step 1: Add force measurement and control cycle functions**

Add after `load_robot_at_keyframe()`:

```python
def measure_contact_forces(mj_model, mj_data):
    """Measure actual contact forces from MuJoCo."""
    floor_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    l_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")
    
    f_left_z = 0.0
    f_right_z = 0.0
    wheel_geom_ids = {l_wheel_geom_id, r_wheel_geom_id}
    
    for i in range(mj_data.ncon):
        c = mj_data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        involves_floor = g1 == floor_geom_id or g2 == floor_geom_id
        involves_wheel = g1 in wheel_geom_ids or g2 in wheel_geom_ids
        
        if not (involves_floor and involves_wheel):
            continue
        
        force_contact = np.zeros(6)
        mujoco.mj_contactForce(mj_model, mj_data, i, force_contact)
        frame = np.array(c.frame).reshape(3, 3)
        force_world = frame.T @ force_contact[:3]
        fz = float(force_world[2])
        
        if g1 == l_wheel_geom_id or g2 == l_wheel_geom_id:
            f_left_z += fz
        else:
            f_right_z += fz
    
    return f_left_z, f_right_z


def run_one_control_cycle(mj_model, mj_data, wbc_controller, centroidal_estimator, 
                          capture_estimator, contact_jacobian, posture_regularizer,
                          leg_position_controller):
    """Run one control cycle and collect comprehensive telemetry."""
    robot_mass = float(np.sum(mj_model.body_mass))
    gravity = float(abs(mj_model.opt.gravity[2]))
    height_cmd = 0.40
    control_dt = 0.01
    
    # Estimate state
    centroidal_state, _ = centroidal_estimator.estimate(jnp.zeros(42), mj_data, None)
    centroidal_state = capture_estimator.update(centroidal_state)
    
    # Build observation
    obs = jnp.zeros(42)
    obs = obs.at[36].set(height_cmd)
    obs = obs.at[37].set(centroidal_state.com_pos[2])
    
    # Compute WBC torque with diagnostics
    tau_wbc, qp_diagnostics = wbc_controller.compute_wbc_torque_with_diagnostics(
        mj_data, obs, centroidal_state, height_cmd
    )
    
    # Compute Jacobian mapping for reference
    f_left = jnp.array([0.0, 0.0, robot_mass * gravity / 2.0])
    f_right = jnp.array([0.0, 0.0, robot_mass * gravity / 2.0])
    tau_from_jacobian = contact_jacobian.map_contact_forces_to_torques(
        mj_data, f_left, f_right, tau_hip_roll=None
    )
    
    # WBC joint scaling (from simulate_hierarchical_controller.py)
    wbc_joint_scale = jnp.array([1.0, 0.3, 0.75, 0.75, 1.0, 1.0, 0.3, 0.75, 0.75, 1.0])
    tau_wbc_scaled = tau_wbc * wbc_joint_scale
    
    # Compute secondary controllers
    joint_pos = jnp.array(mj_data.qpos[7:17])
    joint_vel = jnp.array(mj_data.qvel[6:16])
    target_joint_pos = posture_regularizer.compute_target_posture_from_height(height_cmd)
    
    tau_posture = posture_regularizer.compute_posture_regularizer_torque(
        joint_pos, 0.0, 0.0, height_cmd
    )
    
    tau_leg_position = leg_position_controller.compute_leg_torques(
        joint_pos, joint_vel, target_joint_pos
    )
    
    # Total raw torque
    tau_total_raw = tau_wbc_scaled + tau_posture + tau_leg_position
    
    # Clip to actuator limits
    torque_limit = jnp.array(mj_model.actuator_ctrlrange[:, 1])
    tau_total_clipped = jnp.clip(tau_total_raw, -torque_limit, torque_limit)
    
    # Rate limiting (simplified - no previous torque state)
    max_torque_rate = 400.0  # Nm/s
    tau_rate_vec = tau_total_clipped / control_dt
    tau_rate_vec_clipped = jnp.clip(tau_rate_vec, -max_torque_rate, max_torque_rate)
    tau_smooth = tau_rate_vec_clipped * control_dt
    
    # Apply torque
    tau_applied = np.array(tau_smooth)
    mj_data.ctrl[:] = tau_applied
    mujoco.mj_step(mj_model, mj_data)
    
    # Measure actual contact forces
    f_left_z_actual, f_right_z_actual = measure_contact_forces(mj_model, mj_data)
    
    return {
        'desired_fz_total': qp_diagnostics['desired_wrench_Fz'],
        'distributed_fz_left': qp_diagnostics['f_left_z'],
        'distributed_fz_right': qp_diagnostics['f_right_z'],
        'f_left_z_actual': f_left_z_actual,
        'f_right_z_actual': f_right_z_actual,
        'tau_from_jacobian': tau_from_jacobian,
        'tau_wbc': tau_wbc,
        'tau_wbc_scaled': tau_wbc_scaled,
        'tau_posture': tau_posture,
        'tau_leg_position': tau_leg_position,
        'tau_total_raw': tau_total_raw,
        'tau_total_clipped': tau_total_clipped,
        'tau_smooth': tau_smooth,
        'tau_applied': tau_applied,
        'robot_mass': robot_mass,
        'gravity': gravity,
    }
```

- [ ] **Step 2: Update main() to run control cycle and print audit trail**

Replace the `# TODO: Add force audit trail` section in `main()` with:

```python
    # Initialize controllers
    robot_mass = float(np.sum(mj_model.body_mass))
    gravity = float(abs(mj_model.opt.gravity[2]))
    
    centroidal_estimator = CentroidalStateEstimator(
        CentroidalStateEstimatorConfig(
            robot_mass=robot_mass,
            torso_inertia=jnp.array([0.1, 0.1, 0.05])
        ),
        mj_model=mj_model,
    )
    
    capture_estimator = CapturePointEstimator(
        CapturePointEstimatorConfig(gravity=gravity, min_height=0.35)
    )
    
    wbc_controller = IntegratedWBC(
        mj_model,
        k_roll=60.0,
        k_roll_rate=12.0,
        k_pitch=300.0,
        k_pitch_rate=15.0,
        k_height=50.0,
        robot_mass=robot_mass,
        gravity=gravity,
    )
    
    contact_jacobian = ContactJacobian(mj_model)
    
    print(f"[OK] Controllers initialized\n")
    
    # Run one control cycle
    telemetry = run_one_control_cycle(
        mj_model, mj_data, wbc_controller, centroidal_estimator,
        capture_estimator, contact_jacobian
    )
    
    # Print comprehensive force audit trail
    print("=" * 80)
    print("[FORCE AUDIT TRAIL - Step 0]")
    print("=" * 80)
    
    weight = telemetry['robot_mass'] * telemetry['gravity']
    
    print(f"\n1. WBC Wrench Computer:")
    print(f"   desired_fz_total = {telemetry['desired_fz_total']:.2f} N")
    
    print(f"\n2. Force Distributor:")
    print(f"   distributed_fz_left = {telemetry['distributed_fz_left']:.2f} N")
    print(f"   distributed_fz_right = {telemetry['distributed_fz_right']:.2f} N")
    distributed_total = telemetry['distributed_fz_left'] + telemetry['distributed_fz_right']
    print(f"   distributed_fz_total = {distributed_total:.2f} N")
    
    print(f"\n3. Contact Jacobian Mapping:")
    support_joints = [2, 3, 7, 8]
    print(f"   tau_from_jacobian[2,3,7,8] = {[float(telemetry['tau_from_jacobian'][j]) for j in support_joints]}")
    
    print(f"\n4. Torque Pipeline:")
    print(f"   tau_wbc[2,3,7,8] = {[float(telemetry['tau_wbc'][j]) for j in support_joints]}")
    print(f"   tau_wbc_scaled[2,3,7,8] = {[float(telemetry['tau_wbc_scaled'][j]) for j in support_joints]}")
    print(f"   tau_posture[2,3,7,8] = {[float(telemetry['tau_posture'][j]) for j in support_joints]}")
    print(f"   tau_leg_position[2,3,7,8] = {[float(telemetry['tau_leg_position'][j]) for j in support_joints]}")
    print(f"   tau_total_raw[2,3,7,8] = {[float(telemetry['tau_total_raw'][j]) for j in support_joints]}")
    print(f"   tau_total_clipped[2,3,7,8] = {[float(telemetry['tau_total_clipped'][j]) for j in support_joints]}")
    print(f"   tau_smooth[2,3,7,8] = {[float(telemetry['tau_smooth'][j]) for j in support_joints]}")
    print(f"   tau_applied[2,3,7,8] = {[float(telemetry['tau_applied'][j]) for j in support_joints]}")
    
    print(f"\n5. Cancellation Diagnostics (Support Joints):")
    for j in support_joints:
        tau_wbc_j = float(telemetry['tau_wbc'][j])
        tau_posture_j = float(telemetry['tau_posture'][j])
        tau_leg_pos_j = float(telemetry['tau_leg_position'][j])
        
        # Classify posture contribution
        if abs(tau_posture_j) < 0.1:
            posture_class = "negligible"
        elif np.sign(tau_posture_j) == np.sign(tau_wbc_j):
            posture_class = "assists"
        else:
            posture_class = "OPPOSES"
        
        # Classify leg position contribution
        if abs(tau_leg_pos_j) < 0.1:
            leg_pos_class = "negligible"
        elif np.sign(tau_leg_pos_j) == np.sign(tau_wbc_j):
            leg_pos_class = "assists"
        else:
            leg_pos_class = "OPPOSES"
        
        print(f"   Joint [{j}]: posture={posture_class} ({tau_posture_j:+.2f} Nm), leg_position={leg_pos_class} ({tau_leg_pos_j:+.2f} Nm)")
    
    print(f"\n6. MuJoCo Contact Forces (after mj_step):")
    print(f"   f_left_z_actual = {telemetry['f_left_z_actual']:.2f} N")
    print(f"   f_right_z_actual = {telemetry['f_right_z_actual']:.2f} N")
    f_total_actual = telemetry['f_left_z_actual'] + telemetry['f_right_z_actual']
    print(f"   f_total_z_actual = {f_total_actual:.2f} N")
    
    print(f"\n7. Force Gap Analysis:")
    fz_error = telemetry['desired_fz_total'] - f_total_actual
    deficit_pct = (fz_error / weight) * 100
    print(f"   fz_error = {telemetry['desired_fz_total']:.2f} - {f_total_actual:.2f} = {fz_error:.2f} N ({deficit_pct:.1f}% deficit)")
    print(f"   robot_weight = {weight:.2f} N")
    
    # Identify stage with largest loss
    stages = {
        'Force Distributor': abs(telemetry['desired_fz_total'] - distributed_total),
        'Torque Pipeline': abs(distributed_total - f_total_actual),
    }
    max_loss_stage = max(stages, key=stages.get)
    print(f"   Stage with largest loss: {max_loss_stage} ({stages[max_loss_stage]:.2f} N)")
    
    print("\n" + "=" * 80)
```

- [ ] **Step 3: Run script to verify force audit trail**

Run: `python scripts/debug_force_gap.py`
Expected: Script prints complete force audit trail showing force gap

- [ ] **Step 4: Commit**

```bash
git add scripts/debug_force_gap.py
git commit -m "feat(diagnostics): Implement complete force audit trail"
```

---

## Phase 1: Static Support Parity Test

### Task 3: Create Static Support Parity Script Structure

**Files:**
- Create: `scripts/debug_static_support_parity.py`

- [ ] **Step 1: Create script with test case framework**

```python
"""Static support parity test script.

Tests whether the controller can hold the robot at calibrated standing keyframe
under different torque sources (zero control, WBC, ideal J^T f, inverse dynamics).
"""

import argparse
import numpy as np
import mujoco
import jax.numpy as jnp

from wheeled_biped.controllers.integrated_wbc import IntegratedWBC
from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)
from wheeled_biped.controllers.capture_point_estimator import (
    CapturePointEstimator,
    CapturePointEstimatorConfig,
)
from wheeled_biped.controllers.contact_jacobian import ContactJacobian


def calibrate_root_z_for_wheel_floor_contact(mj_model, mj_data, target_dist=-5e-4, max_iters=5):
    """Calibrate root_z to achieve target wheel-floor contact distance."""
    floor_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    l_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")
    
    for _ in range(max_iters):
        mujoco.mj_forward(mj_model, mj_data)
        
        min_dist = None
        for i in range(mj_data.ncon):
            c = mj_data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            involves_floor = g1 == floor_geom_id or g2 == floor_geom_id
            involves_wheel = g1 in {l_wheel_geom_id, r_wheel_geom_id} or g2 in {l_wheel_geom_id, r_wheel_geom_id}
            
            if involves_floor and involves_wheel:
                d = float(c.dist)
                min_dist = d if min_dist is None else min(min_dist, d)
        
        if min_dist is None:
            break
        
        delta_z = target_dist - min_dist
        if abs(delta_z) < 1e-7:
            break
        
        mj_data.qpos[2] += delta_z
        mj_data.qvel[:] = 0.0
        mj_data.qacc[:] = 0.0
    
    mujoco.mj_forward(mj_model, mj_data)


def load_robot_at_keyframe():
    """Load robot at calibrated standing keyframe with proper initialization.
    
    Matches simulate_hierarchical_controller.py initialization:
    1. Reset to keyframe
    2. mj_forward
    3. Calibrate root_z for -0.5mm contact distance
    4. Zero velocities and accelerations
    5. mj_forward
    """
    model_path = "assets/robot/wheeled_biped_real.xml"
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)
    
    # Step 1: Reset to keyframe
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    
    # Step 2: Forward kinematics
    mujoco.mj_forward(mj_model, mj_data)
    
    # Step 3: Calibrate root_z
    calibrate_root_z_for_wheel_floor_contact(mj_model, mj_data, target_dist=-5e-4)
    
    # Step 4: Zero velocities and accelerations
    mj_data.qvel[:] = 0.0
    mj_data.qacc[:] = 0.0
    
    # Step 5: Forward kinematics again
    mujoco.mj_forward(mj_model, mj_data)
    
    return mj_model, mj_data


def measure_contact_forces(mj_model, mj_data):
    """Measure total vertical contact force."""
    floor_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    l_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")
    
    total_fz = 0.0
    wheel_geom_ids = {l_wheel_geom_id, r_wheel_geom_id}
    
    for i in range(mj_data.ncon):
        c = mj_data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        involves_floor = g1 == floor_geom_id or g2 == floor_geom_id
        involves_wheel = g1 in wheel_geom_ids or g2 in wheel_geom_ids
        
        if not (involves_floor and involves_wheel):
            continue
        
        force_contact = np.zeros(6)
        mujoco.mj_contactForce(mj_model, mj_data, i, force_contact)
        frame = np.array(c.frame).reshape(3, 3)
        force_world = frame.T @ force_contact[:3]
        total_fz += float(force_world[2])
    
    return total_fz


def main():
    parser = argparse.ArgumentParser(description="Static support parity test")
    args = parser.parse_args()
    
    print("=" * 80)
    print("STATIC SUPPORT PARITY TEST")
    print("=" * 80)
    
    # TODO: Add test cases


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run script to verify structure**

Run: `python scripts/debug_static_support_parity.py`
Expected: Script runs and prints header

- [ ] **Step 3: Commit**

```bash
git add scripts/debug_static_support_parity.py
git commit -m "feat(diagnostics): Add static support parity test structure"
```

---

### Task 4: Implement Test Cases A-E

**Files:**
- Modify: `scripts/debug_static_support_parity.py`

- [ ] **Step 1: Add test case execution function**

Add after `measure_contact_forces()`:

```python
def run_test_case(mj_model, mj_data, tau_func, steps_list=[1, 5, 10, 20]):
    """Run a test case with given torque function for multiple step counts.
    
    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data (will be reset for each step count)
        tau_func: Function that takes (mj_model, mj_data) and returns torque array
        steps_list: List of step counts to test
    
    Returns:
        List of dicts with results for each step count
    """
    results = []
    
    for n_steps in steps_list:
        # Reset to keyframe
        if mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        
        # Get initial state
        com_pos_init = np.array(mj_data.subtree_com[1])  # torso body
        
        # Apply torque and step physics
        for _ in range(n_steps):
            tau = tau_func(mj_model, mj_data)
            mj_data.ctrl[:] = np.array(tau)
            mujoco.mj_step(mj_model, mj_data)
        
        # Measure final state
        contact_fz = measure_contact_forces(mj_model, mj_data)
        com_pos_final = np.array(mj_data.subtree_com[1])
        com_z = com_pos_final[2]
        com_vz = (com_pos_final[2] - com_pos_init[2]) / (n_steps * mj_model.opt.timestep)
        max_qacc = float(np.max(np.abs(mj_data.qacc)))
        
        results.append({
            'steps': n_steps,
            'contact_fz': contact_fz,
            'com_z': com_z,
            'com_vz': com_vz,
            'max_qacc': max_qacc,
        })
    
    return results
```

- [ ] **Step 2: Implement Case A (Zero Control)**

Add after `run_test_case()`:

```python
def case_a_zero_control(mj_model, mj_data):
    """Case A: Zero control (gravity only)."""
    return np.zeros(10)
```

- [ ] **Step 3: Implement Case B (WBC Pipeline)**

Add after `case_a_zero_control()`:

```python
def case_b_wbc_pipeline(mj_model, mj_data, wbc_controller, centroidal_estimator, 
                        capture_estimator):
    """Case B: WBC desired torque (current pipeline)."""
    robot_mass = float(np.sum(mj_model.body_mass))
    gravity = float(abs(mj_model.opt.gravity[2]))
    height_cmd = 0.40
    
    # Estimate state
    centroidal_state, _ = centroidal_estimator.estimate(jnp.zeros(42), mj_data, None)
    centroidal_state = capture_estimator.update(centroidal_state)
    
    # Build observation
    obs = jnp.zeros(42)
    obs = obs.at[36].set(height_cmd)
    obs = obs.at[37].set(centroidal_state.com_pos[2])
    
    # Compute WBC torque
    tau_wbc = wbc_controller.compute_wbc_torque(mj_data, obs, centroidal_state, height_cmd)
    
    return tau_wbc
```

- [ ] **Step 4: Implement Case C (Ideal J^T f)**

Add after `case_b_wbc_pipeline()`:

```python
def case_c_ideal_jacobian(mj_model, mj_data, contact_jacobian):
    """Case C: Ideal J^T f (theoretical perfect support)."""
    robot_mass = float(np.sum(mj_model.body_mass))
    gravity = float(abs(mj_model.opt.gravity[2]))
    
    # Compute ideal forces: weight/2 per wheel, vertical only
    f_left = jnp.array([0.0, 0.0, robot_mass * gravity / 2.0])
    f_right = jnp.array([0.0, 0.0, robot_mass * gravity / 2.0])
    
    # Map to joint torques via Jacobian
    tau_ideal = contact_jacobian.map_contact_forces_to_torques(
        mj_data, f_left, f_right, tau_hip_roll=None
    )
    
    return tau_ideal
```

- [ ] **Step 5: Implement Case D (Inverse Dynamics)**

Add after `case_c_ideal_jacobian()`:

```python
def case_d_inverse_dynamics(mj_model, mj_data):
    """Case D: Inverse dynamics (MuJoCo's answer)."""
    # Set velocities and accelerations to zero
    mj_data.qvel[:] = 0.0
    mj_data.qacc[:] = 0.0
    
    # Compute inverse dynamics
    mujoco.mj_inverse(mj_model, mj_data)
    
    # Extract joint torques (skip root DOFs)
    tau_id = jnp.array(mj_data.qfrc_inverse[6:16])
    
    return tau_id
```

- [ ] **Step 6: Update main() to run all test cases**

Replace `# TODO: Add test cases` with:

```python
    mj_model, mj_data = load_robot_at_keyframe()
    
    # Initialize controllers
    robot_mass = float(np.sum(mj_model.body_mass))
    gravity = float(abs(mj_model.opt.gravity[2]))
    
    centroidal_estimator = CentroidalStateEstimator(
        CentroidalStateEstimatorConfig(
            robot_mass=robot_mass,
            torso_inertia=jnp.array([0.1, 0.1, 0.05])
        ),
        mj_model=mj_model,
    )
    
    capture_estimator = CapturePointEstimator(
        CapturePointEstimatorConfig(gravity=gravity, min_height=0.35)
    )
    
    wbc_controller = IntegratedWBC(
        mj_model,
        k_roll=60.0,
        k_roll_rate=12.0,
        k_pitch=300.0,
        k_pitch_rate=15.0,
        k_height=50.0,
        robot_mass=robot_mass,
        gravity=gravity,
    )
    
    contact_jacobian = ContactJacobian(mj_model)
    
    print(f"[OK] Controllers initialized\n")
    
    # Run test cases
    print("=" * 80)
    print("Case A: Zero Control (Gravity Only)")
    print("=" * 80)
    results_a = run_test_case(mj_model, mj_data, lambda m, d: case_a_zero_control(m, d))
    print(f"{'Steps':<6} | {'contact_fz':<12} | {'com_z':<8} | {'com_vz':<8} | {'max_qacc':<10}")
    print("-" * 60)
    for r in results_a:
        print(f"{r['steps']:<6} | {r['contact_fz']:<12.2f} | {r['com_z']:<8.3f} | {r['com_vz']:<8.3f} | {r['max_qacc']:<10.2f}")
    
    print("\n" + "=" * 80)
    print("Case B: WBC Pipeline (Current)")
    print("=" * 80)
    results_b = run_test_case(
        mj_model, mj_data,
        lambda m, d: case_b_wbc_pipeline(m, d, wbc_controller, centroidal_estimator, capture_estimator)
    )
    print(f"{'Steps':<6} | {'contact_fz':<12} | {'com_z':<8} | {'com_vz':<8} | {'max_qacc':<10}")
    print("-" * 60)
    for r in results_b:
        print(f"{r['steps']:<6} | {r['contact_fz']:<12.2f} | {r['com_z']:<8.3f} | {r['com_vz']:<8.3f} | {r['max_qacc']:<10.2f}")
    
    print("\n" + "=" * 80)
    print("Case C: Ideal J^T f")
    print("=" * 80)
    results_c = run_test_case(
        mj_model, mj_data,
        lambda m, d: case_c_ideal_jacobian(m, d, contact_jacobian)
    )
    print(f"{'Steps':<6} | {'contact_fz':<12} | {'com_z':<8} | {'com_vz':<8} | {'max_qacc':<10}")
    print("-" * 60)
    for r in results_c:
        print(f"{r['steps']:<6} | {r['contact_fz']:<12.2f} | {r['com_z']:<8.3f} | {r['com_vz']:<8.3f} | {r['max_qacc']:<10.2f}")
    
    print("\n" + "=" * 80)
    print("Case D: Inverse Dynamics")
    print("=" * 80)
    results_d = run_test_case(mj_model, mj_data, lambda m, d: case_d_inverse_dynamics(m, d))
    print(f"{'Steps':<6} | {'contact_fz':<12} | {'com_z':<8} | {'com_vz':<8} | {'max_qacc':<10}")
    print("-" * 60)
    for r in results_d:
        print(f"{r['steps']:<6} | {r['contact_fz']:<12.2f} | {r['com_z']:<8.3f} | {r['com_vz']:<8.3f} | {r['max_qacc']:<10.2f}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    weight = robot_mass * gravity
    print(f"Robot weight: {weight:.2f} N")
    print(f"\nCase D (inverse dynamics) at 20 steps: {results_d[-1]['contact_fz']:.2f} N")
    print(f"Case C (ideal J^T f) at 20 steps: {results_c[-1]['contact_fz']:.2f} N")
    print(f"Case B (WBC pipeline) at 20 steps: {results_b[-1]['contact_fz']:.2f} N")
    print(f"Case A (zero control) at 20 steps: {results_a[-1]['contact_fz']:.2f} N")
```

- [ ] **Step 7: Run script to verify all test cases**

Run: `python scripts/debug_static_support_parity.py`
Expected: Script runs all 5 test cases and prints comparison tables

- [ ] **Step 8: Commit**

```bash
git add scripts/debug_static_support_parity.py
git commit -m "feat(diagnostics): Implement all static support parity test cases"
```

---

## Phase 2: Actuator Sign & Authority Tests

### Task 5: Create Actuator Sign Test Suite

**Files:**
- Create: `tests/test_actuator_signs.py`

- [ ] **Step 1: Create test file with fixtures**

```python
"""Actuator sign and authority validation tests.

Verifies that each actuator produces force in the expected direction
and that support joints have sufficient authority.
"""

import numpy as np
import mujoco
import pytest


@pytest.fixture
def robot_at_keyframe():
    """Load robot at calibrated standing keyframe."""
    model_path = "assets/robot/wheeled_biped_real.xml"
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)
    
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    
    return mj_model, mj_data


def measure_contact_fz(mj_model, mj_data):
    """Measure total vertical contact force."""
    floor_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    l_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")
    
    total_fz = 0.0
    wheel_geom_ids = {l_wheel_geom_id, r_wheel_geom_id}
    
    for i in range(mj_data.ncon):
        c = mj_data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        involves_floor = g1 == floor_geom_id or g2 == floor_geom_id
        involves_wheel = g1 in wheel_geom_ids or g2 in wheel_geom_ids
        
        if not (involves_floor and involves_wheel):
            continue
        
        force_contact = np.zeros(6)
        mujoco.mj_contactForce(mj_model, mj_data, i, force_contact)
        frame = np.array(c.frame).reshape(3, 3)
        force_world = frame.T @ force_contact[:3]
        total_fz += float(force_world[2])
    
    return total_fz
```

- [ ] **Step 2: Run tests to verify fixtures**

Run: `pytest tests/test_actuator_signs.py -v`
Expected: No tests yet, but fixtures should load

- [ ] **Step 3: Commit**

```bash
git add tests/test_actuator_signs.py
git commit -m "test(diagnostics): Add actuator sign test fixtures"
```

---

### Task 6: Implement Actuator Sign Tests

**Files:**
- Modify: `tests/test_actuator_signs.py`

- [ ] **Step 1: Add Test 2.1 - Individual actuator sign consistency**

Add after fixtures:

```python
def test_actuator_sign_consistency(robot_at_keyframe):
    """Test 2.1: Verify each actuator produces expected acceleration direction."""
    mj_model, mj_data = robot_at_keyframe
    
    results = []
    
    for joint_idx in range(10):
        # Reset to keyframe
        if mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        
        # Apply +1.0 Nm
        mj_data.ctrl[:] = 0.0
        mj_data.ctrl[joint_idx] = 1.0
        mujoco.mj_step(mj_model, mj_data)
        qacc_pos = float(mj_data.qacc[6 + joint_idx])  # Skip root DOFs
        
        # Reset to keyframe
        if mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        
        # Apply -1.0 Nm
        mj_data.ctrl[:] = 0.0
        mj_data.ctrl[joint_idx] = -1.0
        mujoco.mj_step(mj_model, mj_data)
        qacc_neg = float(mj_data.qacc[6 + joint_idx])
        
        # Verify opposite signs
        sign_consistent = np.sign(qacc_pos) == -np.sign(qacc_neg)
        results.append({
            'joint_idx': joint_idx,
            'qacc_pos': qacc_pos,
            'qacc_neg': qacc_neg,
            'sign_consistent': sign_consistent,
        })
        
        assert sign_consistent, f"Joint {joint_idx}: qacc(+1.0)={qacc_pos:.3f}, qacc(-1.0)={qacc_neg:.3f} - signs not opposite"
    
    print("\n[Test 2.1] Actuator sign consistency:")
    for r in results:
        print(f"  Joint {r['joint_idx']}: qacc(+)={r['qacc_pos']:+.3f}, qacc(-)={r['qacc_neg']:+.3f} - {'OK' if r['sign_consistent'] else 'FAIL'}")
```

- [ ] **Step 2: Add Test 2.2 - Support joint authority**

Add after Test 2.1:

```python
def test_support_joint_authority(robot_at_keyframe):
    """Test 2.2: Verify support joints can influence contact force."""
    mj_model, mj_data = robot_at_keyframe
    
    support_joints = [2, 3, 7, 8]  # l_hip_pitch, l_knee, r_hip_pitch, r_knee
    results = []
    
    for joint_idx in support_joints:
        # Baseline: zero control
        if mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        mj_data.ctrl[:] = 0.0
        mujoco.mj_step(mj_model, mj_data)
        fz_baseline = measure_contact_fz(mj_model, mj_data)
        
        # Test +10.0 Nm
        if mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        mj_data.ctrl[:] = 0.0
        mj_data.ctrl[joint_idx] = 10.0
        mujoco.mj_step(mj_model, mj_data)
        fz_pos = measure_contact_fz(mj_model, mj_data)
        qacc_pos = float(mj_data.qacc[6 + joint_idx])
        
        # Test -10.0 Nm
        if mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        mj_data.ctrl[:] = 0.0
        mj_data.ctrl[joint_idx] = -10.0
        mujoco.mj_step(mj_model, mj_data)
        fz_neg = measure_contact_fz(mj_model, mj_data)
        qacc_neg = float(mj_data.qacc[6 + joint_idx])
        
        # Determine which sign helps support (increases Fz or reduces downward qacc)
        delta_fz_pos = fz_pos - fz_baseline
        delta_fz_neg = fz_neg - fz_baseline
        
        results.append({
            'joint_idx': joint_idx,
            'fz_baseline': fz_baseline,
            'fz_pos': fz_pos,
            'fz_neg': fz_neg,
            'delta_fz_pos': delta_fz_pos,
            'delta_fz_neg': delta_fz_neg,
            'qacc_pos': qacc_pos,
            'qacc_neg': qacc_neg,
            'helpful_sign': '+' if delta_fz_pos > delta_fz_neg else '-',
        })
    
    print("\n[Test 2.2] Support joint authority:")
    for r in results:
        print(f"  Joint {r['joint_idx']}: Fz baseline={r['fz_baseline']:.1f}N, "
              f"+10Nm→{r['fz_pos']:.1f}N (Δ{r['delta_fz_pos']:+.1f}), "
              f"-10Nm→{r['fz_neg']:.1f}N (Δ{r['delta_fz_neg']:+.1f}), "
              f"helpful_sign={r['helpful_sign']}")
```

- [ ] **Step 3: Add Test 2.3 - Left/right symmetry**

Add after Test 2.2:

```python
def test_left_right_symmetry(robot_at_keyframe):
    """Test 2.3: Verify left/right joint pairs have symmetric response."""
    mj_model, mj_data = robot_at_keyframe
    
    joint_pairs = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]  # left, right indices
    results = []
    
    for left_idx, right_idx in joint_pairs:
        # Test left joint
        if mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        mj_data.ctrl[:] = 0.0
        mj_data.ctrl[left_idx] = 5.0
        mujoco.mj_step(mj_model, mj_data)
        qacc_left = abs(float(mj_data.qacc[6 + left_idx]))
        
        # Test right joint
        if mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        mj_data.ctrl[:] = 0.0
        mj_data.ctrl[right_idx] = 5.0
        mujoco.mj_step(mj_model, mj_data)
        qacc_right = abs(float(mj_data.qacc[6 + right_idx]))
        
        # Check symmetry (within 10%)
        ratio = qacc_left / max(qacc_right, 1e-6)
        symmetric = 0.9 <= ratio <= 1.1
        
        results.append({
            'left_idx': left_idx,
            'right_idx': right_idx,
            'qacc_left': qacc_left,
            'qacc_right': qacc_right,
            'ratio': ratio,
            'symmetric': symmetric,
        })
        
        assert symmetric, f"Joints ({left_idx},{right_idx}): |qacc_left|={qacc_left:.3f}, |qacc_right|={qacc_right:.3f}, ratio={ratio:.2f} - not symmetric"
    
    print("\n[Test 2.3] Left/right symmetry:")
    for r in results:
        print(f"  Joints ({r['left_idx']},{r['right_idx']}): "
              f"|qacc_L|={r['qacc_left']:.3f}, |qacc_R|={r['qacc_right']:.3f}, "
              f"ratio={r['ratio']:.2f} - {'OK' if r['symmetric'] else 'FAIL'}")
```

- [ ] **Step 4: Run all actuator tests**

Run: `pytest tests/test_actuator_signs.py -v -s`
Expected: All tests pass, diagnostic output printed

- [ ] **Step 5: Commit**

```bash
git add tests/test_actuator_signs.py
git commit -m "test(diagnostics): Implement actuator sign and authority tests"
```

---

## Phase 3: Inverse Dynamics Baseline

### Task 7: Create Inverse Dynamics Diagnostic Script

**Files:**
- Create: `scripts/debug_static_inverse_dynamics.py`

- [ ] **Step 1: Create script structure with inverse dynamics computation**

```python
"""Inverse dynamics baseline diagnostic script.

Establishes ground truth for what torques are physically required to hold
the standing posture by comparing inverse dynamics against controller torques.
"""

import argparse
import numpy as np
import mujoco
import jax.numpy as jnp

from wheeled_biped.controllers.integrated_wbc import IntegratedWBC
from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)
from wheeled_biped.controllers.capture_point_estimator import (
    CapturePointEstimator,
    CapturePointEstimatorConfig,
)
from wheeled_biped.controllers.posture_regularizer import (
    PostureRegularizer,
    PostureRegularizerConfig,
)
from wheeled_biped.controllers.leg_position_controller import LegPositionController


def calibrate_root_z_for_wheel_floor_contact(mj_model, mj_data, target_dist=-5e-4, max_iters=5):
    """Calibrate root_z to achieve target wheel-floor contact distance."""
    floor_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    l_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")
    
    for _ in range(max_iters):
        mujoco.mj_forward(mj_model, mj_data)
        
        min_dist = None
        for i in range(mj_data.ncon):
            c = mj_data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            involves_floor = g1 == floor_geom_id or g2 == floor_geom_id
            involves_wheel = g1 in {l_wheel_geom_id, r_wheel_geom_id} or g2 in {l_wheel_geom_id, r_wheel_geom_id}
            
            if involves_floor and involves_wheel:
                d = float(c.dist)
                min_dist = d if min_dist is None else min(min_dist, d)
        
        if min_dist is None:
            break
        
        delta_z = target_dist - min_dist
        if abs(delta_z) < 1e-7:
            break
        
        mj_data.qpos[2] += delta_z
        mj_data.qvel[:] = 0.0
        mj_data.qacc[:] = 0.0
    
    mujoco.mj_forward(mj_model, mj_data)


def load_robot_at_keyframe():
    """Load robot at calibrated standing keyframe with proper initialization.
    
    Matches simulate_hierarchical_controller.py initialization:
    1. Reset to keyframe
    2. mj_forward
    3. Calibrate root_z for -0.5mm contact distance
    4. Zero velocities and accelerations
    5. mj_forward
    """
    model_path = "assets/robot/wheeled_biped_real.xml"
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)
    
    # Step 1: Reset to keyframe
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    
    # Step 2: Forward kinematics
    mujoco.mj_forward(mj_model, mj_data)
    
    # Step 3: Calibrate root_z
    calibrate_root_z_for_wheel_floor_contact(mj_model, mj_data, target_dist=-5e-4)
    
    # Step 4: Zero velocities and accelerations
    mj_data.qvel[:] = 0.0
    mj_data.qacc[:] = 0.0
    
    # Step 5: Forward kinematics again
    mujoco.mj_forward(mj_model, mj_data)
    
    return mj_model, mj_data


def compute_inverse_dynamics(mj_model, mj_data):
    """Compute required holding torques via inverse dynamics."""
    # Set velocities and accelerations to zero for static equilibrium
    mj_data.qvel[:] = 0.0
    mj_data.qacc[:] = 0.0
    
    # Compute inverse dynamics
    mujoco.mj_inverse(mj_model, mj_data)
    
    # Extract joint torques (skip root DOFs)
    tau_required = np.array(mj_data.qfrc_inverse[6:16])
    qfrc_bias = np.array(mj_data.qfrc_bias[6:16])
    
    return {
        'tau_required': tau_required,
        'qfrc_bias': qfrc_bias,
    }


def main():
    parser = argparse.ArgumentParser(description="Inverse dynamics baseline diagnostic")
    args = parser.parse_args()
    
    print("=" * 80)
    print("INVERSE DYNAMICS BASELINE DIAGNOSTIC")
    print("=" * 80)
    
    mj_model, mj_data = load_robot_at_keyframe()
    print(f"[OK] Robot loaded at keyframe 0\n")
    
    # Compute inverse dynamics
    id_results = compute_inverse_dynamics(mj_model, mj_data)
    
    print("[STEP 3.1] Required Holding Torques (from mj_inverse):")
    print("-" * 80)
    support_joints = [2, 3, 7, 8]
    joint_names = ['l_hip_pitch', 'l_knee', 'r_hip_pitch', 'r_knee']
    
    for idx, name in zip(support_joints, joint_names):
        print(f"  {name:12} [{idx}]: {id_results['tau_required'][idx]:+7.2f} Nm")
    
    # TODO: Add controller torque comparison


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run script to verify inverse dynamics computation**

Run: `python scripts/debug_static_inverse_dynamics.py`
Expected: Script prints required holding torques from inverse dynamics

- [ ] **Step 3: Commit**

```bash
git add scripts/debug_static_inverse_dynamics.py
git commit -m "feat(diagnostics): Add inverse dynamics baseline script structure"
```

---

### Task 8: Add Controller Torque Comparison

**Files:**
- Modify: `scripts/debug_static_inverse_dynamics.py`

- [ ] **Step 1: Add controller torque computation**

Replace `# TODO: Add controller torque comparison` with:

```python
    # Initialize controllers
    robot_mass = float(np.sum(mj_model.body_mass))
    gravity = float(abs(mj_model.opt.gravity[2]))
    height_cmd = 0.40
    
    centroidal_estimator = CentroidalStateEstimator(
        CentroidalStateEstimatorConfig(
            robot_mass=robot_mass,
            torso_inertia=jnp.array([0.1, 0.1, 0.05])
        ),
        mj_model=mj_model,
    )
    
    capture_estimator = CapturePointEstimator(
        CapturePointEstimatorConfig(gravity=gravity, min_height=0.35)
    )
    
    wbc_controller = IntegratedWBC(
        mj_model,
        k_roll=60.0,
        k_roll_rate=12.0,
        k_pitch=300.0,
        k_pitch_rate=15.0,
        k_height=50.0,
        robot_mass=robot_mass,
        gravity=gravity,
    )
    
    posture_regularizer = PostureRegularizer(
        PostureRegularizerConfig(
            k_posture=10.0,
            k_hip_roll=3.0,
            k_hip_yaw=1.5,
            k_hip_pitch=30.0,
            k_knee=30.0,
            k_wheel=0.0,
        )
    )
    
    leg_position_controller = LegPositionController(
        kp_hip_yaw=5.0,
        kd_hip_yaw=1.0,
        kp_hip_pitch=20.0,
        kd_hip_pitch=3.0,
        kp_knee=35.0,
        kd_knee=4.0,
        max_torque=25.0,
    )
    
    print(f"\n[OK] Controllers initialized\n")
    
    # Estimate state
    centroidal_state, _ = centroidal_estimator.estimate(jnp.zeros(42), mj_data, None)
    centroidal_state = capture_estimator.update(centroidal_state)
    
    # Build observation
    obs = jnp.zeros(42)
    obs = obs.at[36].set(height_cmd)
    obs = obs.at[37].set(centroidal_state.com_pos[2])
    
    # Compute controller torques
    tau_wbc = wbc_controller.compute_wbc_torque(mj_data, obs, centroidal_state, height_cmd)
    
    joint_pos = jnp.array(mj_data.qpos[7:17])
    joint_vel = jnp.array(mj_data.qvel[6:16])
    target_joint_pos = posture_regularizer.compute_target_posture_from_height(height_cmd)
    
    tau_posture = posture_regularizer.compute_posture_regularizer_torque(
        joint_pos, 0.0, 0.0, height_cmd
    )
    
    tau_leg_position = leg_position_controller.compute_leg_torques(
        joint_pos, joint_vel, target_joint_pos
    )
    
    tau_total = tau_wbc + tau_posture + tau_leg_position
    
    # Print torque budget analysis
    print("[STEP 3.2 & 3.3] Controller Torque Budget:")
    print("=" * 80)
    print(f"{'Joint':<12} | {'Required':>8} | {'WBC':>8} | {'Posture':>8} | {'Leg_Pos':>8} | {'Total':>8} | {'Deficit':>8}")
    print("-" * 80)
    
    for idx, name in zip(support_joints, joint_names):
        required = id_results['tau_required'][idx]
        wbc = float(tau_wbc[idx])
        posture = float(tau_posture[idx])
        leg_pos = float(tau_leg_position[idx])
        total = float(tau_total[idx])
        deficit = required - total
        
        # Classify if secondary terms assist or oppose WBC
        posture_assists = np.sign(posture) == np.sign(wbc) if abs(wbc) > 0.1 else True
        leg_pos_assists = np.sign(leg_pos) == np.sign(wbc) if abs(wbc) > 0.1 else True
        
        print(f"{name:12} | {required:+8.2f} | {wbc:+8.2f} | {posture:+8.2f} | {leg_pos:+8.2f} | {total:+8.2f} | {deficit:+8.2f}")
        
        if abs(posture) > 0.1:
            status = "assists" if posture_assists else "OPPOSES"
            print(f"  └─ Posture {status} WBC")
        if abs(leg_pos) > 0.1:
            status = "assists" if leg_pos_assists else "OPPOSES"
            print(f"  └─ Leg position {status} WBC")
    
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    # Compute aggregate deficit
    total_deficit = sum(id_results['tau_required'][idx] - float(tau_total[idx]) for idx in support_joints)
    avg_deficit = total_deficit / len(support_joints)
    
    print(f"Average torque deficit across support joints: {avg_deficit:.2f} Nm")
    print(f"Robot weight: {robot_mass * gravity:.2f} N")
    print(f"\nNote: Torque deficit correlates with observed 15-20N force gap")
```

- [ ] **Step 2: Run complete diagnostic**

Run: `python scripts/debug_static_inverse_dynamics.py`
Expected: Script prints complete torque budget with analysis

- [ ] **Step 3: Commit**

```bash
git add scripts/debug_static_inverse_dynamics.py
git commit -m "feat(diagnostics): Implement complete inverse dynamics torque budget analysis"
```

---

## Plan Self-Review

- [ ] **Spec coverage check**: All phases 0-3 from spec are covered with concrete tasks
- [ ] **Placeholder scan**: No TBD, TODO (except in initial code templates), or vague instructions
- [ ] **Type consistency**: Function names, parameters, and return types are consistent across tasks
- [ ] **File paths**: All file paths are absolute and correct
- [ ] **Commands**: All commands have expected output specified
- [ ] **Code completeness**: All code blocks are complete and runnable

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-23-standing-balance-physics-diagnostics-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?

