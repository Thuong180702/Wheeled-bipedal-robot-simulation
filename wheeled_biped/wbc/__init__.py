"""QP-WBC (Whole-Body Control) module for the K2 wheeled-biped robot.

Phase 3   — Offline QP-WBC Prototype.
Phase 3B  — Offline Task Stack Expansion.
Phase 3B.1 — Full Ablation Audit + Compilation Hardening.
Phase 3C  — Offline Rolling Constraints and Task Refinement.

Provides offline QP-based whole-body control using the validated dynamics
stack from Phases 2A–2D.1.  All functions are offline only — no realtime
torque injection, no controller integration.
"""

from wheeled_biped.wbc.offline_qp_wbc import (
    build_qp_wbc_constants,
    build_actuator_selection_matrix,
    build_contact_stack,
    build_qp_matrices,
    solve_offline_qp,
    validate_qp_solution,
    make_default_offline_task_spec,
    finite_difference_jdot_qdot,
    compute_contact_jdot_qdot,
    integrate_qpos,
    CONSTANTS_VERSION,
)

from wheeled_biped.wbc.offline_task_stack import (
    make_phase3b_task_spec,
    build_task_cost_matrices,
    evaluate_task_residuals,
    run_task_weight_ablation,
    compute_com_jacobian,
    compute_com_jdot_qdot,
    compute_torso_angular_velocity_jacobian,
    compute_torso_jdotw_qdot,
    compute_torso_orientation_error,
    build_qp_matrices_phase3b,
    check_solution_sanity,
    TASK_STACK_VERSION,
    TASK_WEIGHT_MODES,
)

from wheeled_biped.wbc.phase3b_cached_stack import (
    PaddedContactStack,
    build_padded_contact_stack,
    Phase3BSnapshot,
    prepare_phase3b_snapshot,
    build_phase3b_qp_from_snapshot,
    evaluate_task_residuals_from_snapshot,
    validate_solution_from_snapshot,
    MAX_CONTACTS,
)

# Phase 3C modules are NOT imported here to avoid circular imports
# when offline_rolling_constraints imports from offline_qp_wbc.
# Import them directly:
#   from wheeled_biped.wbc.offline_rolling_constants import ...
#   from wheeled_biped.wbc.phase3c_rolling_qp import ...

__all__ = [
    # Phase 3
    "build_qp_wbc_constants",
    "build_actuator_selection_matrix",
    "build_contact_stack",
    "build_qp_matrices",
    "solve_offline_qp",
    "validate_qp_solution",
    "make_default_offline_task_spec",
    "finite_difference_jdot_qdot",
    "compute_contact_jdot_qdot",
    "integrate_qpos",
    "CONSTANTS_VERSION",
    # Phase 3B
    "make_phase3b_task_spec",
    "build_task_cost_matrices",
    "evaluate_task_residuals",
    "run_task_weight_ablation",
    "compute_com_jacobian",
    "compute_com_jdot_qdot",
    "compute_torso_angular_velocity_jacobian",
    "compute_torso_jdotw_qdot",
    "compute_torso_orientation_error",
    "build_qp_matrices_phase3b",
    "check_solution_sanity",
    "TASK_STACK_VERSION",
    "TASK_WEIGHT_MODES",
    # Phase 3B.1 — Cached Stack
    "PaddedContactStack",
    "build_padded_contact_stack",
    "Phase3BSnapshot",
    "prepare_phase3b_snapshot",
    "build_phase3b_qp_from_snapshot",
    "evaluate_task_residuals_from_snapshot",
    "validate_solution_from_snapshot",
    "MAX_CONTACTS",
]
