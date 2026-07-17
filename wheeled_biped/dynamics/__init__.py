"""Dynamics truth layer for K2 wheeled-biped controller.

Read-only diagnostics and validation utilities for physical quantities
needed by future QP-WBC development. No controller logic, no tuning,
no profile changes.

Exports:
    build_model_index_report
    extract_state_snapshot
    compute_task_jacobian
    finite_difference_jacobian_check
    inspect_contacts
    torque_sign_probe
    build_kinematic_tree_constants
    jax_forward_kinematics
    jax_compute_com
    jax_compute_subtree_or_total_com
    jax_body_position_jacobian
    jax_compute_all_target_jacobians
    validate_jacobian_actuated_columns
    build_mass_matrix_constants
    jax_mass_matrix
    jax_mass_matrix_fk_arrays
    jax_actuated_mass_submatrix
    jax_body_spatial_velocities
    jax_compute_kinetic_energy
    compare_mass_matrix_to_mujoco
"""

from wheeled_biped.dynamics.model_inspector import (
    build_model_index_report,
    extract_state_snapshot,
)
from wheeled_biped.dynamics.jacobian_checks import (
    compute_task_jacobian,
    finite_difference_jacobian_check,
)
from wheeled_biped.dynamics.contact_inspector import (
    inspect_contacts,
)
from wheeled_biped.dynamics.torque_sign_checks import (
    torque_sign_probe,
)
from wheeled_biped.dynamics.jax_kinematics import (
    build_kinematic_tree_constants,
    jax_forward_kinematics,
)
from wheeled_biped.dynamics.jax_com import (
    jax_compute_com,
    jax_compute_subtree_or_total_com,
)
from wheeled_biped.dynamics.jax_jacobians import (
    jax_body_position_jacobian,
    jax_compute_all_target_jacobians,
    validate_jacobian_actuated_columns,
)
from wheeled_biped.dynamics.jax_mass_matrix import (
    build_mass_matrix_constants,
    jax_mass_matrix,
    jax_mass_matrix_fk_arrays,
    jax_actuated_mass_submatrix,
    jax_body_spatial_velocities,
    jax_compute_kinetic_energy,
    compare_mass_matrix_to_mujoco,
)
from wheeled_biped.dynamics.jax_bias_forces import (
    build_bias_force_constants,
    extract_jax_bias_arrays,
    jax_bias_forces,
    jax_bias_forces_fk_arrays,
    jax_gravity_forces,
    jax_velocity_bias_forces,
    compare_bias_forces_to_mujoco,
)
from wheeled_biped.dynamics.bias_force_diagnostics import (
    decompose_bias_errors,
    decompose_velocity_components,
    compute_cross_term_decomposition,
)

__all__ = [
    "build_model_index_report",
    "extract_state_snapshot",
    "compute_task_jacobian",
    "finite_difference_jacobian_check",
    "inspect_contacts",
    "torque_sign_probe",
    "build_kinematic_tree_constants",
    "jax_forward_kinematics",
    "jax_compute_com",
    "jax_compute_subtree_or_total_com",
    "jax_body_position_jacobian",
    "jax_compute_all_target_jacobians",
    "validate_jacobian_actuated_columns",
    "build_mass_matrix_constants",
    "jax_mass_matrix",
    "jax_mass_matrix_fk_arrays",
    "jax_actuated_mass_submatrix",
    "jax_body_spatial_velocities",
    "jax_compute_kinetic_energy",
    "compare_mass_matrix_to_mujoco",
    "build_bias_force_constants",
    "extract_jax_bias_arrays",
    "jax_bias_forces",
    "jax_bias_forces_fk_arrays",
    "jax_gravity_forces",
    "jax_velocity_bias_forces",
    "compare_bias_forces_to_mujoco",
    "decompose_bias_errors",
    "decompose_velocity_components",
    "compute_cross_term_decomposition",
]
