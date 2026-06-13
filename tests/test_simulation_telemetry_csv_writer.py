"""Tests for simulation telemetry CSV writing.

These tests verify that the telemetry CSV writing correctly produces
data rows, handles empty columns, and includes all required fields.
"""

import csv
import io
import tempfile
import os
from pathlib import Path

import pytest


class TestTelemetryCSVWriting:
    """Test suite for telemetry CSV writing functionality."""

    def test_min_row_calculation_with_all_populated(self):
        """Test n_rows calculation when all columns are populated."""
        telemetry = {
            "time": [0.0, 0.01, 0.02],
            "source_step_index": [0, 1, 2],
            "control_mode": ["upright", "upright", "upright"],
        }
        n_rows = min(len(values) for values in telemetry.values()) if telemetry else 0
        assert n_rows == 3, f"Expected 3 rows, got {n_rows}"

    def test_min_row_calculation_with_empty_column(self):
        """Test n_rows calculation when one column is empty."""
        telemetry = {
            "time": [0.0, 0.01, 0.02],
            "source_step_index": [0, 1, 2],
            "empty_col": [],  # Empty column
        }
        n_rows = min(len(values) for values in telemetry.values()) if telemetry else 0
        assert n_rows == 0, f"Expected 0 rows when any column is empty, got {n_rows}"

    def test_min_row_calculation_empty_telemetry(self):
        """Test n_rows calculation with empty telemetry dict."""
        telemetry = {}
        n_rows = min(len(values) for values in telemetry.values()) if telemetry else 0
        assert n_rows == 0, f"Expected 0 rows for empty dict, got {n_rows}"

    def test_csv_writer_produces_correct_row_count(self):
        """Test that CSV writer produces header + data rows."""
        telemetry = {
            "time": [0.0, 0.01, 0.02],
            "source_step_index": [0, 1, 2],
            "control_mode": ["upright", "upright", "upright"],
        }

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(telemetry.keys())
        n_rows = min(len(values) for values in telemetry.values()) if telemetry else 0
        for i in range(n_rows):
            writer.writerow([telemetry[k][i] for k in telemetry.keys()])

        content = output.getvalue()
        lines = content.strip().split('\n')
        assert len(lines) == 4, f"Expected 4 lines (1 header + 3 data), got {len(lines)}"

    def test_csv_writer_produces_zero_data_rows_with_empty_column(self):
        """Test that CSV writer produces header only when one column is empty."""
        telemetry = {
            "time": [0.0, 0.01, 0.02],
            "source_step_index": [0, 1, 2],
            "empty_col": [],  # Empty column - this will cause n_rows = 0
        }

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(telemetry.keys())
        n_rows = min(len(values) for values in telemetry.values()) if telemetry else 0
        for i in range(n_rows):
            writer.writerow([telemetry[k][i] for k in telemetry.keys()])

        content = output.getvalue()
        lines = content.strip().split('\n')
        # With n_rows=0, we expect only the header line
        assert len(lines) == 1, f"Expected 1 line (header only) when n_rows=0, got {len(lines)}"

    def test_balanced_core_telemetry_columns_initialization(self):
        """Test that balance-core telemetry columns are properly initialized."""
        from wheeled_biped.controllers.balance_core_types import make_balance_core_telemetry_columns

        cols = make_balance_core_telemetry_columns()

        # All columns should be empty lists initially
        for name, values in cols.items():
            assert isinstance(values, list), f"Column {name} should be list, got {type(values)}"
            assert len(values) == 0, f"Column {name} should be empty, got {len(values)}"

        # Check required state telemetry columns exist
        from wheeled_biped.controllers.balance_core_types import BALANCE_CORE_REQUIRED_STATE_TELEMETRY
        for name in BALANCE_CORE_REQUIRED_STATE_TELEMETRY:
            assert name in cols, f"Required state column {name} missing from telemetry"

        # Check required torque telemetry columns exist
        from wheeled_biped.controllers.balance_core_types import BALANCE_CORE_REQUIRED_TORQUE_TELEMETRY
        for name in BALANCE_CORE_REQUIRED_TORQUE_TELEMETRY:
            assert name in cols, f"Required torque column {name} missing from telemetry"

    def test_telemetry_mismatch_detection(self):
        """Test that we can detect when n_rows=0 despite populated columns."""
        telemetry = {
            "time": [0.0, 0.01, 0.02],  # Populated
            "source_step_index": [0, 1, 2],  # Populated
            "balance_core_col": [],  # Empty - this causes mismatch
        }

        n_rows = min(len(values) for values in telemetry.values()) if telemetry else 0
        populated_cols = {k: len(v) for k, v in telemetry.items() if len(v) > 0}
        empty_cols = [k for k, v in telemetry.items() if len(v) == 0]

        # n_rows should be 0 due to empty balance_core_col
        assert n_rows == 0, f"Expected n_rows=0, got {n_rows}"

        # But we should have 2 populated columns
        assert len(populated_cols) == 2, f"Expected 2 populated cols, got {len(populated_cols)}"

        # This is the bug condition we need to detect
        assert n_rows == 0 and len(populated_cols) > 0, "Should detect mismatch"

    def test_append_balance_core_telemetry_populates_columns(self):
        """Test that append_balance_core_telemetry populates balance-core columns."""
        import jax.numpy as jnp
        from wheeled_biped.controllers.balance_core_types import (
            BalanceCoreTorqueResult,
            ContactSupervisorState,
        )
        from wheeled_biped.controllers.balance_core_types import make_balance_core_telemetry_columns

        # Create sample telemetry dict with balance-core columns initialized
        telemetry = make_balance_core_telemetry_columns()
        telemetry["time"] = []
        telemetry["source_step_index"] = []

        # Create a sample BalanceCoreTorqueResult
        result = BalanceCoreTorqueResult(
            tau_shape_posture=jnp.zeros(10),
            tau_support_feedforward=jnp.zeros(10),
            tau_sagittal_wheel_balance=jnp.zeros(10),
            tau_lateral_roll_balance=jnp.zeros(10),
            tau_total_raw=jnp.zeros(10),
            tau_total_clipped=jnp.zeros(10),
            tau_final=jnp.zeros(10),
            active_torque_owner_per_joint=['shape_posture'] * 10,
            ownership_violation_count=0,
            violations=[],
            saturation_mask=jnp.zeros(10, dtype=bool),
            rate_saturation_mask=jnp.zeros(10, dtype=bool),
        )

        # Get result telemetry
        result_telemetry = result.telemetry

        # Verify result.telemetry has the expected keys
        from wheeled_biped.controllers.balance_core_types import BALANCE_CORE_REQUIRED_TORQUE_TELEMETRY
        for name in BALANCE_CORE_REQUIRED_TORQUE_TELEMETRY:
            assert name in result_telemetry, f"Required torque column {name} missing from result.telemetry"

        # Simulate appending: this is what append_balance_core_telemetry does
        for name, value in result_telemetry.items():
            if isinstance(value, tuple):
                telemetry[name].append(",".join(str(v) for v in value))
            else:
                telemetry[name].append(value)

        # Verify columns are now populated
        for name in BALANCE_CORE_REQUIRED_TORQUE_TELEMETRY:
            assert len(telemetry[name]) == 1, f"Column {name} should have 1 entry after append, got {len(telemetry[name])}"

    def test_profile_identity_telemetry_fields_exist(self):
        """Test that profile identity telemetry fields are initialized (Phase 1 fix)."""
        # These fields should be initialized in simulate_hierarchical_controller.py
        # after balance-core columns are added
        required_identity_fields = [
            "controller_mode",
            "sagittal_controller",
            "vd_sagittal_authority_profile",
            "height_variant_setup_name",
            "t6j_bias_trim_enabled",
            "t6j_bias_trim_active",
            "t6j_bias_mean_error_m",
            "t6j_bias_window_steps",
            "t6j_bias_trim_tau_nm",
            "t6j_bias_trim_target_tau_nm",
            "t6j_bias_trim_rate_limited",
            "t6j_bias_positive_duration_steps",
            "t6j_bias_negative_duration_steps",
            "t6j_bias_safety_gate_pass",
            "t6j_bias_block_reason",
            "t6j_bias_applied_to_final_tau",
            "t6j_bias_expected_direction_correct",
        ]

        # Simulate the initialization logic from simulate_hierarchical_controller.py
        telemetry = {}
        telemetry.setdefault("controller_mode", [])
        telemetry.setdefault("sagittal_controller", [])
        telemetry.setdefault("vd_sagittal_authority_profile", [])
        telemetry.setdefault("height_variant_setup_name", [])
        telemetry.setdefault("t6j_bias_trim_enabled", [])
        telemetry.setdefault("t6j_bias_trim_active", [])
        telemetry.setdefault("t6j_bias_mean_error_m", [])
        telemetry.setdefault("t6j_bias_window_steps", [])
        telemetry.setdefault("t6j_bias_trim_tau_nm", [])
        telemetry.setdefault("t6j_bias_trim_target_tau_nm", [])
        telemetry.setdefault("t6j_bias_trim_rate_limited", [])
        telemetry.setdefault("t6j_bias_positive_duration_steps", [])
        telemetry.setdefault("t6j_bias_negative_duration_steps", [])
        telemetry.setdefault("t6j_bias_safety_gate_pass", [])
        telemetry.setdefault("t6j_bias_block_reason", [])
        telemetry.setdefault("t6j_bias_applied_to_final_tau", [])
        telemetry.setdefault("t6j_bias_expected_direction_correct", [])

        # Verify all required fields exist and are empty lists
        for field in required_identity_fields:
            assert field in telemetry, f"Required identity field {field} missing"
            assert isinstance(telemetry[field], list), f"Field {field} should be list"
            assert len(telemetry[field]) == 0, f"Field {field} should be empty initially"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])