"""Diagnostic script for wheel-floor contact lifecycle and contact-force consistency.

Usage:
    python scripts/debug_contact_lifecycle.py
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import mujoco

from wheeled_biped.controllers.capture_point_estimator import (
    CapturePointEstimator,
    CapturePointEstimatorConfig,
)
from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)
from wheeled_biped.controllers.integrated_wbc import IntegratedWBC


MODEL_PATH = "assets/robot/wheeled_biped_real.xml"


def geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    return name if name is not None else f"<unnamed:{geom_id}>"


def estimate_state(
    estimator: CentroidalStateEstimator,
    capture_estimator: CapturePointEstimator,
    data: mujoco.MjData,
):
    state, _ = estimator.estimate(jnp.zeros(42), data, None)
    state = capture_estimator.update(state)
    return state


def dump_contacts(
    phase: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    l_wheel_geom_id: int,
    r_wheel_geom_id: int,
    floor_geom_id: int,
    estimator: CentroidalStateEstimator,
    capture_estimator: CapturePointEstimator,
):
    print("\n" + "=" * 120)
    print(f"PHASE {phase}")
    print("=" * 120)

    state = estimate_state(estimator, capture_estimator, data)

    wheel_floor_contact_count = 0
    wheel_floor_fz_total = 0.0
    pair_counts: dict[tuple[str, str], int] = {}

    for i in range(data.ncon):
        c = data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        g1_name = geom_name(model, g1)
        g2_name = geom_name(model, g2)

        force_contact = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force_contact)

        frame = np.array(c.frame).reshape(3, 3)
        force_world = frame.T @ force_contact[:3]

        involves_left_wheel = g1 == l_wheel_geom_id or g2 == l_wheel_geom_id
        involves_right_wheel = g1 == r_wheel_geom_id or g2 == r_wheel_geom_id
        involves_wheel = involves_left_wheel or involves_right_wheel
        involves_floor = g1 == floor_geom_id or g2 == floor_geom_id

        if involves_wheel and involves_floor:
            wheel_floor_contact_count += 1
            wheel_floor_fz_total += float(force_world[2])

        pair = tuple(sorted((g1_name, g2_name)))
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

        print(f"contact[{i}]")
        print(f"  geom1: id={g1}, name={g1_name}")
        print(f"  geom2: id={g2}, name={g2_name}")
        print(f"  dist: {float(c.dist):+.6f}")
        print(f"  pos: {np.array(c.pos)}")
        print(f"  force_contact(raw): {force_contact}")
        print(f"  frame: {frame}")
        print(f"  force_world(frame.T @ force_contact[:3]): {force_world}")
        print(f"  matches_left_or_right_wheel: {involves_wheel}")
        print(f"  matches_floor: {involves_floor}")

    print("\nTotals")
    print(f"  ncon: {data.ncon}")
    print(f"  wheel_floor_contact_count: {wheel_floor_contact_count}")
    print(f"  wheel_floor_fz_total_from_mj_contactForce: {wheel_floor_fz_total:+.6f}")
    print(f"  estimator.left_wheel_contact: {state.left_wheel_contact}")
    print(f"  estimator.right_wheel_contact: {state.right_wheel_contact}")
    print(f"  estimator.total_contact_force_z: {state.total_contact_force_z:+.6f}")
    print(f"  estimator.left_contact_force_world: {np.array(state.left_contact_force_world)}")
    print(f"  estimator.right_contact_force_world: {np.array(state.right_contact_force_world)}")

    if pair_counts:
        print("  contact_pairs:")
        for (a, b), cnt in sorted(pair_counts.items()):
            print(f"    ({a}, {b}) x{cnt}")
    else:
        print("  contact_pairs: <none>")

    return {
        "ncon": int(data.ncon),
        "wheel_floor_contact_count": int(wheel_floor_contact_count),
        "wheel_floor_fz_total": float(wheel_floor_fz_total),
        "est_left": bool(state.left_wheel_contact),
        "est_right": bool(state.right_wheel_contact),
        "est_fz": float(state.total_contact_force_z),
        "pairs": pair_counts,
    }


def main() -> None:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    robot_mass = float(np.sum(model.body_mass))
    gravity = float(abs(model.opt.gravity[2]))

    estimator = CentroidalStateEstimator(
        CentroidalStateEstimatorConfig(
            robot_mass=robot_mass,
            torso_inertia=jnp.array([0.1, 0.1, 0.05]),
        ),
        mj_model=model,
    )
    capture_estimator = CapturePointEstimator(CapturePointEstimatorConfig(gravity=gravity, min_height=0.35))

    l_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r_wheel_collision")
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    l_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "l_wheel_contact")
    r_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "r_wheel_contact")

    mujoco.mj_resetDataKeyframe(model, data, 0)

    print("=" * 120)
    print("Initial static info")
    print("=" * 120)
    print(f"model: {MODEL_PATH}")
    print(f"root qpos z: {float(data.qpos[2]):+.6f}")

    # subtree_com may not be valid before first mj_forward; print raw value then updated later.
    print(f"robot CoM z from data.subtree_com[1] (pre-forward): {float(data.subtree_com[1][2]):+.6f}")

    print(f"l_wheel_collision geom id/name: {l_wheel_geom_id} / {geom_name(model, l_wheel_geom_id)}")
    print(f"r_wheel_collision geom id/name: {r_wheel_geom_id} / {geom_name(model, r_wheel_geom_id)}")
    print(f"floor geom id/name: {floor_geom_id} / {geom_name(model, floor_geom_id)}")

    l_geom_pos = np.array(data.geom_xpos[l_wheel_geom_id])
    r_geom_pos = np.array(data.geom_xpos[r_wheel_geom_id])
    print(f"l_wheel_collision world position: {l_geom_pos}")
    print(f"r_wheel_collision world position: {r_geom_pos}")

    l_radius = float(model.geom_size[l_wheel_geom_id][0])
    r_radius = float(model.geom_size[r_wheel_geom_id][0])
    print(f"l_wheel_collision radius: {l_radius:.6f}, estimated wheel bottom z: {l_geom_pos[2] - l_radius:+.6f}")
    print(f"r_wheel_collision radius: {r_radius:.6f}, estimated wheel bottom z: {r_geom_pos[2] - r_radius:+.6f}")

    if l_site_id != -1:
        print(f"l_wheel_contact site world position: {np.array(data.site_xpos[l_site_id])}")
    else:
        print("l_wheel_contact site world position: <site not found>")

    if r_site_id != -1:
        print(f"r_wheel_contact site world position: {np.array(data.site_xpos[r_site_id])}")
    else:
        print("r_wheel_contact site world position: <site not found>")

    # A: after mj_resetDataKeyframe, before mj_forward
    A = dump_contacts("A: after mj_resetDataKeyframe (before mj_forward)", model, data, l_wheel_geom_id, r_wheel_geom_id, floor_geom_id, estimator, capture_estimator)

    # B: after mj_forward
    mujoco.mj_forward(model, data)
    print(f"robot CoM z from data.subtree_com[1] (after first forward): {float(data.subtree_com[1][2]):+.6f}")
    B = dump_contacts("B: after mj_forward", model, data, l_wheel_geom_id, r_wheel_geom_id, floor_geom_id, estimator, capture_estimator)

    # C: after zero qvel/qacc + mj_forward
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)
    C = dump_contacts("C: after zero qvel/qacc + mj_forward", model, data, l_wheel_geom_id, r_wheel_geom_id, floor_geom_id, estimator, capture_estimator)

    # D: after 1 mj_step with zero ctrl
    data.ctrl[:] = 0.0
    mujoco.mj_step(model, data)
    D = dump_contacts("D: after 1 mj_step with zero ctrl", model, data, l_wheel_geom_id, r_wheel_geom_id, floor_geom_id, estimator, capture_estimator)

    # E: after 5 mj_steps with zero ctrl
    for _ in range(5):
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
    E = dump_contacts("E: after 5 mj_steps with zero ctrl", model, data, l_wheel_geom_id, r_wheel_geom_id, floor_geom_id, estimator, capture_estimator)

    # F: after 1 mj_step with current WBC tau_smooth (optional if easy)
    # Implemented as optional best-effort path.
    F = None
    try:
        wbc = IntegratedWBC(
            model,
            k_roll=60.0,
            k_roll_rate=12.0,
            k_roll_integral=0.0,
            k_pitch=300.0,
            k_pitch_rate=15.0,
            k_com_lateral=15.0,
            k_com_lateral_damping=3.0,
            k_com_sagittal=50.0,
            k_com_sagittal_damping=6.0,
            k_cp_lateral=50.0,
            k_cp_sagittal=100.0,
            k_height=50.0,
            robot_mass=robot_mass,
            gravity=gravity,
            max_roll_moment=25.0,
            wbc_authority_budget=0.95,
            max_actuator_torque=60.0,
            force_feedback_gain=0.2,
            force_feedback_warmup_steps=5,
            tau_hip_roll_max=15.0,
            max_force_asymmetry=60.0,
            min_wheel_force=20.0,
            roll_integral_limit=0.52,
            dt=model.opt.timestep,
        )

        state = estimate_state(estimator, capture_estimator, data)
        obs = jnp.zeros(42)
        obs = obs.at[36].set(0.40)
        obs = obs.at[37].set(state.com_pos[2])
        tau_wbc, _ = wbc.compute_wbc_torque_with_diagnostics(
            data,
            obs,
            state,
            0.40,
            hip_roll_authority_scale=1.0,
        )

        control_dt = 0.01
        max_torque_rate = 400.0
        tau_prev = jnp.zeros(10)
        tau_rate_vec = (tau_wbc - tau_prev) / control_dt
        tau_rate_vec_clipped = jnp.clip(tau_rate_vec, -max_torque_rate, max_torque_rate)
        tau_smooth = tau_prev + tau_rate_vec_clipped * control_dt

        data.ctrl[:] = np.array(tau_smooth)
        mujoco.mj_step(model, data)
        F = dump_contacts(
            "F: after 1 mj_step with current WBC tau_smooth",
            model,
            data,
            l_wheel_geom_id,
            r_wheel_geom_id,
            floor_geom_id,
            estimator,
            capture_estimator,
        )
    except Exception as exc:
        print("\n[WARN] Phase F skipped due to exception:")
        print(exc)

    # First 10 physics steps contact pairs summary from a fresh reset
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)

    pair_10: dict[tuple[str, str], int] = {}
    wheel_floor_count_10 = 0
    for _ in range(10):
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
        for i in range(data.ncon):
            c = data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            p = tuple(sorted((geom_name(model, g1), geom_name(model, g2))))
            pair_10[p] = pair_10.get(p, 0) + 1
            involves_wheel = g1 in (l_wheel_geom_id, r_wheel_geom_id) or g2 in (l_wheel_geom_id, r_wheel_geom_id)
            involves_floor = g1 == floor_geom_id or g2 == floor_geom_id
            if involves_wheel and involves_floor:
                wheel_floor_count_10 += 1

    print("\n" + "=" * 120)
    print("First 10 physics steps contact-pair summary")
    print("=" * 120)
    if pair_10:
        for (a, b), cnt in sorted(pair_10.items()):
            print(f"({a}, {b}) x{cnt}")
    else:
        print("<none>")
    print(f"wheel_floor_contact_count_in_first_10_steps: {wheel_floor_count_10}")

    # CASE classification
    phases = [A, B, C, D, E] + ([F] if F is not None else [])
    has_wheel_floor_contact = any(p["wheel_floor_contact_count"] > 0 for p in phases)
    has_est_contact = any((p["est_left"] or p["est_right"]) for p in phases)
    has_est_force = any(abs(p["est_fz"]) > 1e-6 for p in phases)
    has_world_fz = any(abs(p["wheel_floor_fz_total"]) > 1e-6 for p in phases)

    print("\n" + "=" * 120)
    print("CASE classification")
    print("=" * 120)

    if not has_wheel_floor_contact:
        case = "CASE A"
        print("CASE A confirmed: No wheel-floor contact exists at keyframe/early steps.")
        print("Root-cause target: initialization/contact geometry mismatch.")
        print("Measured wheel heights for root_z correction:")
        print(f"  left wheel bottom z: {l_geom_pos[2] - l_radius:+.6f}")
        print(f"  right wheel bottom z: {r_geom_pos[2] - r_radius:+.6f}")
        if l_site_id != -1:
            print(f"  left wheel contact site z: {float(data.site_xpos[l_site_id][2]):+.6f}")
        if r_site_id != -1:
            print(f"  right wheel contact site z: {float(data.site_xpos[r_site_id][2]):+.6f}")
    elif has_wheel_floor_contact and not has_est_force:
        case = "CASE B"
        print("CASE B confirmed: Wheel-floor contact exists, but estimator total_contact_force_z remains zero.")
        print("Root-cause target: CentroidalStateEstimator contact-force extraction.")
    elif has_wheel_floor_contact and has_est_contact and not has_world_fz:
        case = "CASE C"
        print("CASE C confirmed: Contact exists and estimator sees contact, but world-force z is near zero.")
        print("Root-cause target: contact force generation/timing/solver state.")
    elif has_wheel_floor_contact and has_world_fz and has_est_force:
        case = "CASE D"
        print("CASE D confirmed: Contact and force exist, estimator sees force.")
        print("Root-cause target: telemetry/logging order staleness (outside estimator-force extraction).")
    else:
        case = "UNRESOLVED"
        print("UNRESOLVED: mixed signal, needs focused follow-up on specific phase discrepancies.")

    print(f"\nFinal classification: {case}")


if __name__ == "__main__":
    main()
