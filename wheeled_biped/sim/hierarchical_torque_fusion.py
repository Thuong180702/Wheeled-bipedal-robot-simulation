"""Hierarchical task-priority torque fusion for Phase B.9 Step 5.25.

Implements explicit authority allocation, state-dependent stabilization,
and contact-aware control to prevent authority suppression while maintaining
WBC dominance.
"""

from __future__ import annotations

import jax.numpy as jnp
from mujoco import mjx


def hierarchical_torque_fusion_control(
    mjx_data: mjx.Data,
    normalized_wbc_torque: jnp.ndarray,
    ctrl_min: jnp.ndarray,
    ctrl_max: jnp.ndarray,
    wbc_authority_min: float = 0.60,
    contact_stabilization_gain: float = 0.0,
    contact_asymmetry_threshold: float = 0.15,
    damping_gain: float = 0.0,
    oscillation_threshold: float = 0.5,
    impedance_kp: float = 0.0,
    impedance_target: jnp.ndarray | None = None,
    wbc_error_threshold: float = 0.3,
    left_foot_contact: float = 0.0,
    right_foot_contact: float = 0.0,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Hierarchical task-priority torque fusion with explicit authority allocation.

    Priority hierarchy:
        Level 1: Balance-critical WBC torque (guaranteed minimum authority)
        Level 2: Contact stabilization (state-dependent, contact-aware)
        Level 3: Velocity damping (oscillation-triggered, not continuous)
        Level 4: Posture regularization (weak, nullspace-only)

    Args:
        mjx_data: Current MJX simulation data.
        normalized_wbc_torque: WBC torque commands in [-1, 1], shape (num_joints,).
        ctrl_min: Actuator control range lower bound.
        ctrl_max: Actuator control range upper bound.
        wbc_authority_min: Guaranteed minimum WBC authority fraction (0.0-1.0).
            Default 0.60 = WBC guaranteed at least 60% of actuator range.
        contact_stabilization_gain: Contact-aware stabilization gain.
            Only activates when contact asymmetry > threshold.
        contact_asymmetry_threshold: Threshold for contact asymmetry activation.
            Default 0.15 = activate when load difference > 15%.
        damping_gain: Velocity damping gain.
            Only activates when oscillation detected (joint_vel_rms > threshold).
        oscillation_threshold: Joint velocity RMS threshold for damping activation.
            Default 0.5 rad/s.
        impedance_kp: Weak impedance gain for posture regularization.
            Only activates when WBC error is small (< wbc_error_threshold).
        impedance_target: Target joint positions for impedance (optional, radians).
        wbc_error_threshold: Normalized WBC error threshold for posture activation.
            Default 0.3 = activate posture only when WBC error < 30% of capacity.
        left_foot_contact: Left foot contact force (N).
        right_foot_contact: Right foot contact force (N).

    Returns:
        Tuple of (final_ctrl, telemetry_dict) where:
            final_ctrl: Final actuator control, shape (num_joints,).
            telemetry_dict: Dictionary of telemetry metrics for analysis.
    """
    num_joints = normalized_wbc_torque.shape[0]
    joint_pos = mjx_data.qpos[7:17]
    joint_vel = mjx_data.qvel[6:16]

    # ── Level 1: WBC Authority Budget ────────────────────────────────────────
    normalized_wbc_torque = jnp.clip(normalized_wbc_torque, -1.0, 1.0)
    wbc_authority_min = jnp.clip(jnp.asarray(wbc_authority_min, dtype=ctrl_min.dtype), 0.0, 1.0)
    ctrl_limit = jnp.minimum(jnp.abs(ctrl_min), jnp.abs(ctrl_max))

    # Allocate guaranteed WBC budget
    wbc_budget = ctrl_limit * wbc_authority_min
    tau_wbc_desired = normalized_wbc_torque * ctrl_limit
    tau_wbc = jnp.clip(tau_wbc_desired, -wbc_budget, wbc_budget)

    # Compute remaining budget for stabilization
    remaining_budget = ctrl_limit - jnp.abs(tau_wbc)
    stabilization_budget = remaining_budget * 0.8  # 80% for stabilization
    posture_budget = remaining_budget * 0.2        # 20% for posture

    # ── Level 2: Contact Stabilization (state-dependent) ─────────────────────
    # Only activate when contact asymmetry detected
    total_contact = left_foot_contact + right_foot_contact + 1e-6  # avoid division by zero
    left_load = left_foot_contact / total_contact
    right_load = right_foot_contact / total_contact
    contact_asymmetry = jnp.abs(left_load - 0.5)

    contact_active = contact_asymmetry > contact_asymmetry_threshold

    # Apply corrective hip roll torque based on load asymmetry
    # Left unloaded → positive left hip roll, negative right hip roll
    # Right unloaded → negative left hip roll, positive right hip roll
    load_error = left_load - right_load  # positive = left heavier
    tau_contact_raw = jnp.zeros(num_joints, dtype=ctrl_min.dtype)

    # Hip roll indices: 0 (left), 5 (right)
    contact_correction = contact_stabilization_gain * load_error
    tau_contact_raw = tau_contact_raw.at[0].set(contact_correction)   # l_hip_roll
    tau_contact_raw = tau_contact_raw.at[5].set(-contact_correction)  # r_hip_roll

    tau_contact_clipped = jnp.clip(tau_contact_raw, -stabilization_budget, stabilization_budget)
    tau_contact = jnp.where(contact_active, tau_contact_clipped, jnp.zeros_like(tau_contact_raw))

    # ── Level 3: Velocity Damping (oscillation-triggered) ────────────────────
    # Only activate during oscillation
    joint_vel_rms = jnp.sqrt(jnp.mean(joint_vel**2))
    oscillation_detected = joint_vel_rms > oscillation_threshold

    tau_damping_raw = -jnp.asarray(damping_gain, dtype=ctrl_min.dtype) * joint_vel
    tau_damping_clipped = jnp.clip(tau_damping_raw, -stabilization_budget, stabilization_budget)
    tau_damping = jnp.where(oscillation_detected, tau_damping_clipped, jnp.zeros_like(tau_damping_raw))

    # ── Level 4: Posture Regularization (weak, nullspace-only) ───────────────
    # Only activate when WBC error is small
    wbc_error_norm = jnp.sqrt(jnp.sum(tau_wbc_desired**2))
    wbc_error_normalized = wbc_error_norm / (num_joints * jnp.mean(ctrl_limit) + 1e-6)
    posture_active = wbc_error_normalized < wbc_error_threshold

    impedance_target_safe = jnp.zeros_like(tau_wbc) if impedance_target is None else impedance_target
    pos_error = impedance_target_safe - joint_pos
    tau_posture_raw = jnp.asarray(impedance_kp, dtype=ctrl_min.dtype) * pos_error
    tau_posture_clipped = jnp.clip(tau_posture_raw, -posture_budget, posture_budget)
    tau_posture = jnp.where(posture_active, tau_posture_clipped, jnp.zeros_like(tau_posture_raw))

    # ── Final Fusion ──────────────────────────────────────────────────────────
    tau_total = tau_wbc + tau_contact + tau_damping + tau_posture
    final_ctrl = jnp.clip(tau_total, ctrl_min, ctrl_max)

    # ── Telemetry ─────────────────────────────────────────────────────────────
    tau_wbc_rms = jnp.sqrt(jnp.mean(tau_wbc**2))
    tau_contact_rms = jnp.sqrt(jnp.mean(tau_contact**2))
    tau_damping_rms = jnp.sqrt(jnp.mean(tau_damping**2))
    tau_posture_rms = jnp.sqrt(jnp.mean(tau_posture**2))
    tau_total_rms = jnp.sqrt(jnp.mean(tau_total**2))

    total_authority = tau_wbc_rms + tau_contact_rms + tau_damping_rms + tau_posture_rms + 1e-6

    telemetry = {
        # Authority tracking
        "tau_wbc_rms": tau_wbc_rms,
        "tau_contact_rms": tau_contact_rms,
        "tau_damping_rms": tau_damping_rms,
        "tau_posture_rms": tau_posture_rms,
        "tau_total_rms": tau_total_rms,
        "wbc_authority_pct": (tau_wbc_rms / total_authority) * 100.0,
        "contact_authority_pct": (tau_contact_rms / total_authority) * 100.0,
        "damping_authority_pct": (tau_damping_rms / total_authority) * 100.0,
        "posture_authority_pct": (tau_posture_rms / total_authority) * 100.0,

        # Activation tracking
        "contact_active": jnp.asarray(contact_active, dtype=jnp.float32),
        "oscillation_detected": jnp.asarray(oscillation_detected, dtype=jnp.float32),
        "posture_active": jnp.asarray(posture_active, dtype=jnp.float32),
        "contact_asymmetry": contact_asymmetry,
        "joint_vel_rms": joint_vel_rms,
        "wbc_error_normalized": wbc_error_normalized,

        # Clipping tracking
        "wbc_clipped": jnp.any(jnp.abs(tau_wbc_desired) > wbc_budget),
        "contact_clipped": jnp.any(jnp.abs(tau_contact_raw) > stabilization_budget),
        "damping_clipped": jnp.any(jnp.abs(tau_damping_raw) > stabilization_budget),
        "posture_clipped": jnp.any(jnp.abs(tau_posture_raw) > posture_budget),
        "total_saturated": jnp.any((final_ctrl <= ctrl_min + 1e-6) | (final_ctrl >= ctrl_max - 1e-6)),

        # Component torques (for detailed analysis)
        "tau_wbc": tau_wbc,
        "tau_contact": tau_contact,
        "tau_damping": tau_damping,
        "tau_posture": tau_posture,
    }

    return final_ctrl, telemetry
