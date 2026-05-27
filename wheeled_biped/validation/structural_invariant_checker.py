"""Structural invariant checker for balance-core controller architecture.

This module implements Priority 0 architecture regression checks that must pass
before performance analysis. These checks verify that the balance-core controller
maintains its architectural invariants during execution.
"""

import ast
from typing import Dict, List, Any
import pandas as pd
import numpy as np


class ArchitectureRegressionError(Exception):
    """Raised when a structural invariant is violated."""
    pass


class StructuralInvariantChecker:
    """Checks structural invariants of balance-core controller telemetry.

    Priority 0 checks that must pass before performance analysis:
    1. Controller mode is "balance-core"
    2. No ownership violations
    3. All torque owners are valid balance-core components
    4. Hidden torque is negligible
    5. All torque vectors contain finite values
    6. Safety masks are valid boolean vectors
    7. Contact state is valid
    """

    # Tolerance for hidden torque norm (N·m)
    TOLERANCE = 1e-6

    # Valid contact supervisor states
    VALID_CONTACT_STATES = {
        "DOUBLE_CONTACT",
        "SINGLE_LEFT",
        "SINGLE_RIGHT",
        "NO_CONTACT",
        "UNKNOWN",
        "INIT",
    }

    # Valid balance-core torque owners
    VALID_BALANCE_CORE_OWNERS = {
        "shape_posture",
        "support_feedforward",
        "sagittal_wheel_balance",
        "lateral_roll_balance",
    }

    def check_all(self, df: pd.DataFrame) -> Dict[str, str]:
        """Run all structural invariant checks.

        Args:
            df: Telemetry dataframe with balance-core controller data

        Returns:
            Dictionary mapping check name to "PASS" or error message

        Raises:
            ArchitectureRegressionError: If any invariant is violated
        """
        results = {}

        # Run all checks
        try:
            self._check_controller_mode(df)
            results["controller_mode"] = "PASS"
        except ArchitectureRegressionError as e:
            results["controller_mode"] = str(e)
            raise

        try:
            self._check_ownership_violations(df)
            results["ownership_violations"] = "PASS"
        except ArchitectureRegressionError as e:
            results["ownership_violations"] = str(e)
            raise

        try:
            self._check_torque_owners(df)
            results["torque_owners"] = "PASS"
        except ArchitectureRegressionError as e:
            results["torque_owners"] = str(e)
            raise

        try:
            self._check_hidden_torque(df)
            results["hidden_torque"] = "PASS"
        except ArchitectureRegressionError as e:
            results["hidden_torque"] = str(e)
            raise

        try:
            self._check_finite_torques(df)
            results["finite_torques"] = "PASS"
        except ArchitectureRegressionError as e:
            results["finite_torques"] = str(e)
            raise

        try:
            self._check_safety_masks(df)
            results["safety_masks"] = "PASS"
        except ArchitectureRegressionError as e:
            results["safety_masks"] = str(e)
            raise

        try:
            self._check_contact_state(df)
            results["contact_state"] = "PASS"
        except ArchitectureRegressionError as e:
            results["contact_state"] = str(e)
            raise

        return results

    def _check_controller_mode(self, df: pd.DataFrame) -> None:
        """Verify controller_mode == "balance-core" for all rows.

        Args:
            df: Telemetry dataframe

        Raises:
            ArchitectureRegressionError: If any row has wrong controller mode
        """
        if "controller_mode" not in df.columns:
            raise ArchitectureRegressionError(
                "Missing required field: controller_mode"
            )

        wrong_modes = df[df["controller_mode"] != "balance-core"]
        if not wrong_modes.empty:
            unique_modes = wrong_modes["controller_mode"].unique()
            raise ArchitectureRegressionError(
                f"controller_mode must be 'balance-core', found: {unique_modes.tolist()}"
            )

    def _check_ownership_violations(self, df: pd.DataFrame) -> None:
        """Verify ownership_violation_count == 0 for all rows.

        Args:
            df: Telemetry dataframe

        Raises:
            ArchitectureRegressionError: If any ownership violations detected
        """
        if "ownership_violation_count" not in df.columns:
            raise ArchitectureRegressionError(
                "Missing required field: ownership_violation_count"
            )

        violations = df[df["ownership_violation_count"] != 0]
        if not violations.empty:
            total_violations = violations["ownership_violation_count"].sum()
            raise ArchitectureRegressionError(
                f"Found {total_violations} ownership violations across "
                f"{len(violations)} timesteps"
            )

    def _check_torque_owners(self, df: pd.DataFrame) -> None:
        """Verify all torque owners are valid balance-core components.

        Args:
            df: Telemetry dataframe

        Raises:
            ArchitectureRegressionError: If invalid torque owners found
        """
        if "active_torque_owner_per_joint" not in df.columns:
            raise ArchitectureRegressionError(
                "Missing required field: active_torque_owner_per_joint"
            )

        invalid_owners = set()
        for idx, row in df.iterrows():
            owners_str = row["active_torque_owner_per_joint"]
            try:
                owners = ast.literal_eval(owners_str)
                if not isinstance(owners, list):
                    raise ArchitectureRegressionError(
                        f"Row {idx}: active_torque_owner_per_joint must be a list, "
                        f"got {type(owners)}"
                    )

                for owner in owners:
                    if owner not in self.VALID_BALANCE_CORE_OWNERS:
                        invalid_owners.add(owner)
            except (ValueError, SyntaxError) as e:
                raise ArchitectureRegressionError(
                    f"Row {idx}: Failed to parse active_torque_owner_per_joint: {e}"
                )

        if invalid_owners:
            raise ArchitectureRegressionError(
                f"Invalid torque owners found: {invalid_owners}. "
                f"Valid owners: {self.VALID_BALANCE_CORE_OWNERS}"
            )

    def _check_hidden_torque(self, df: pd.DataFrame) -> None:
        """Verify hidden_torque_norm < TOLERANCE for all rows.

        Args:
            df: Telemetry dataframe

        Raises:
            ArchitectureRegressionError: If hidden torque exceeds tolerance
        """
        if "hidden_torque_norm" not in df.columns:
            raise ArchitectureRegressionError(
                "Missing required field: hidden_torque_norm"
            )

        excessive = df[df["hidden_torque_norm"] >= self.TOLERANCE]
        if not excessive.empty:
            max_hidden = excessive["hidden_torque_norm"].max()
            raise ArchitectureRegressionError(
                f"Hidden torque exceeds tolerance ({self.TOLERANCE} N·m). "
                f"Max found: {max_hidden:.6e} N·m across {len(excessive)} timesteps"
            )

    def _check_finite_torques(self, df: pd.DataFrame) -> None:
        """Verify all torque vectors contain finite values.

        Args:
            df: Telemetry dataframe

        Raises:
            ArchitectureRegressionError: If any torque vector contains NaN/inf
        """
        torque_fields = [
            "tau_shape_posture_per_joint",
            "tau_support_feedforward_per_joint",
            "tau_sagittal_wheel_balance_per_joint",
            "tau_lateral_roll_balance_per_joint",
            "tau_total_raw_per_joint",
            "tau_total_clipped_per_joint",
            "tau_final_per_joint",
            "actuator_ctrl_per_joint",
        ]

        for field in torque_fields:
            if field not in df.columns:
                raise ArchitectureRegressionError(
                    f"Missing required field: {field}"
                )

            for idx, row in df.iterrows():
                torque_str = row[field]
                try:
                    torques = ast.literal_eval(torque_str)
                    if not isinstance(torques, list):
                        raise ArchitectureRegressionError(
                            f"Row {idx}: {field} must be a list, got {type(torques)}"
                        )

                    torques_array = np.array(torques, dtype=float)
                    if not np.all(np.isfinite(torques_array)):
                        raise ArchitectureRegressionError(
                            f"Row {idx}: {field} contains non-finite values: {torques}"
                        )
                except (ValueError, SyntaxError) as e:
                    raise ArchitectureRegressionError(
                        f"Row {idx}: Failed to parse {field}: {e}"
                    )

    def _check_safety_masks(self, df: pd.DataFrame) -> None:
        """Verify safety masks are valid boolean vectors of length 10.

        Args:
            df: Telemetry dataframe

        Raises:
            ArchitectureRegressionError: If safety masks are invalid
        """
        mask_fields = [
            "torque_saturation_mask_per_joint",
            "torque_rate_saturation_mask_per_joint",
        ]

        for field in mask_fields:
            if field not in df.columns:
                raise ArchitectureRegressionError(
                    f"Missing required field: {field}"
                )

            for idx, row in df.iterrows():
                mask_str = row[field]
                try:
                    mask = ast.literal_eval(mask_str)
                    if not isinstance(mask, list):
                        raise ArchitectureRegressionError(
                            f"Row {idx}: {field} must be a list, got {type(mask)}"
                        )

                    if len(mask) != 10:
                        raise ArchitectureRegressionError(
                            f"Row {idx}: {field} must have length 10, got {len(mask)}"
                        )

                    if not all(isinstance(x, bool) for x in mask):
                        raise ArchitectureRegressionError(
                            f"Row {idx}: {field} must contain only booleans, got {mask}"
                        )
                except (ValueError, SyntaxError) as e:
                    raise ArchitectureRegressionError(
                        f"Row {idx}: Failed to parse {field}: {e}"
                    )

    def _check_contact_state(self, df: pd.DataFrame) -> None:
        """Verify contact states are valid and duration is non-negative.

        Args:
            df: Telemetry dataframe

        Raises:
            ArchitectureRegressionError: If contact state is invalid
        """
        if "contact_supervisor_state" not in df.columns:
            raise ArchitectureRegressionError(
                "Missing required field: contact_supervisor_state"
            )

        if "contact_duration_s" not in df.columns:
            raise ArchitectureRegressionError(
                "Missing required field: contact_duration_s"
            )

        # Check valid contact states
        invalid_states = df[
            ~df["contact_supervisor_state"].isin(self.VALID_CONTACT_STATES)
        ]
        if not invalid_states.empty:
            unique_invalid = invalid_states["contact_supervisor_state"].unique()
            raise ArchitectureRegressionError(
                f"Invalid contact states found: {unique_invalid.tolist()}. "
                f"Valid states: {self.VALID_CONTACT_STATES}"
            )

        # Check non-negative duration
        negative_duration = df[df["contact_duration_s"] < 0]
        if not negative_duration.empty:
            raise ArchitectureRegressionError(
                f"Found {len(negative_duration)} timesteps with negative contact_duration_s"
            )
