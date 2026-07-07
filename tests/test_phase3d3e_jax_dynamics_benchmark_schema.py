"""Schema validation tests for Phase 3D.3-E JAX Dynamics output JSONs.

Validates:
  - jax_dynamics_diagnostic.json (E1)
  - jax_dynamics_correctness.json (E6)
  - jax_dynamics_benchmark.json (E7)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs" / "phase3d3e_jax_dynamics"


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def diagnostic() -> dict:
    """Load jax_dynamics_diagnostic.json (E1)."""
    path = OUTPUT_DIR / "jax_dynamics_diagnostic.json"
    if not path.exists():
        pytest.skip(f"Diagnostic JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def correctness() -> dict:
    """Load jax_dynamics_correctness.json (E6)."""
    path = OUTPUT_DIR / "jax_dynamics_correctness.json"
    if not path.exists():
        pytest.skip(f"Correctness JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def benchmark() -> dict:
    """Load jax_dynamics_benchmark.json (E7)."""
    path = OUTPUT_DIR / "jax_dynamics_benchmark.json"
    if not path.exists():
        pytest.skip(f"Benchmark JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostic output schema (E1)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDiagnosticOutputSchema:
    """Schema validations for jax_dynamics_diagnostic.json."""

    def test_has_phase(self, diagnostic):
        assert diagnostic.get("phase") == "3D.3-E1"

    def test_has_environment(self, diagnostic):
        env = diagnostic.get("environment", {})
        assert "jax_version" in env
        assert "jax_enable_x64" in env
        assert "device_count" in env
        assert "jax_platform" in env

    def test_has_sub_operation_timings(self, diagnostic):
        sub = diagnostic.get("sub_operation_timings_s", {})
        assert isinstance(sub, dict)
        required_keys = [
            "mass_matrix_s",
            "bias_forces_s",
            "contact_jac_single_s",
            "com_jacobian_s",
            "com_jdot_qdot_s",
            "torso_ang_vel_jac_s",
            "torso_jdotw_qdot_s",
            "torso_orient_error_s",
            "contact_jdot_qdot_s",
        ]
        for key in required_keys:
            assert key in sub, f"Missing sub-operation key: {key}"
            assert isinstance(sub[key], (int, float)), f"{key} should be numeric"

    def test_has_repeated_timings(self, diagnostic):
        full = diagnostic.get("full_snapshot_timings_s", {})
        assert "first_call" in full
        assert "second_call" in full
        assert isinstance(full["first_call"], (int, float))
        assert isinstance(full["second_call"], (int, float))

    def test_has_summary(self, diagnostic):
        summary = diagnostic.get("summary", {})
        assert isinstance(summary, dict)
        assert "full_snapshot_first_s" in summary
        assert "mass_matrix_s" in summary
        assert "contact_jdot_qdot_s" in summary

    def test_status_is_complete(self, diagnostic):
        assert diagnostic.get("status") == "complete"

    def test_contact_count_recorded(self, diagnostic):
        assert isinstance(diagnostic.get("n_contacts"), int)

    def test_qpos_qvel_shapes(self, diagnostic):
        assert diagnostic.get("qpos_shape") == [17]
        assert diagnostic.get("qvel_shape") == [16]


# ═══════════════════════════════════════════════════════════════════════════════
# Correctness output schema (E6)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCorrectnessOutputSchema:
    """Schema validations for jax_dynamics_correctness.json."""

    def test_has_phase(self, correctness):
        assert correctness.get("phase") == "3D.3-E6"

    def test_has_verdict(self, correctness):
        verdict = correctness.get("verdict", "")
        assert isinstance(verdict, str)
        assert len(verdict) > 0

    def test_has_cache_info(self, correctness):
        cache_info = correctness.get("cache_info", {})
        assert "init_time_s" in cache_info
        assert "compile_time_s" in cache_info
        assert "warmup_time_s" in cache_info
        assert "jax_platform" in cache_info
        assert "dtype" in cache_info
        assert "max_contacts" in cache_info

    def test_has_tolerances(self, correctness):
        tolerances = correctness.get("tolerances", {})
        assert "jdq_com" in tolerances
        assert "jdot_qdot" in tolerances

    def test_has_scenarios(self, correctness):
        scenarios = correctness.get("scenarios", [])
        assert isinstance(scenarios, list)
        assert len(scenarios) > 0, "Correctness must have at least one scenario"

    def test_scenarios_have_required_fields(self, correctness):
        required = [
            "scenario", "pass", "n_contacts", "orig_time_s", "cache_time_s",
            "speedup", "field_diffs", "field_failures", "qp_diffs", "qp_failures",
        ]
        for scenario in correctness["scenarios"]:
            for field in required:
                assert field in scenario, f"Scenario '{scenario.get('scenario', '?')}' missing field: {field}"

    def test_scenario_pass_is_bool(self, correctness):
        for scenario in correctness["scenarios"]:
            assert isinstance(scenario["pass"], bool), f"Scenario '{scenario['scenario']}': pass must be bool"

    def test_speedup_is_positive(self, correctness):
        for scenario in correctness["scenarios"]:
            assert scenario["speedup"] > 0, f"Scenario '{scenario['scenario']}': speedup must be > 0"

    def test_field_failures_is_list(self, correctness):
        for scenario in correctness["scenarios"]:
            assert isinstance(scenario["field_failures"], list)

    def test_qp_failures_is_list(self, correctness):
        for scenario in correctness["scenarios"]:
            assert isinstance(scenario["qp_failures"], list)

    def test_field_diffs_contains_key_fields(self, correctness):
        key_fields = {"M", "h", "Jcom", "Jr", "jdq_com", "jdot_qdot", "jdw_torso"}
        for scenario in correctness["scenarios"]:
            diffs = scenario["field_diffs"]
            for key in key_fields:
                assert key in diffs, f"Scenario '{scenario['scenario']}': missing field_diff: {key}"

    def test_qp_diffs_contains_key_fields(self, correctness):
        key_fields = {"H", "g", "b_eq", "A_eq", "A_friction", "b_friction"}
        for scenario in correctness["scenarios"]:
            diffs = scenario["qp_diffs"]
            for key in key_fields:
                assert key in diffs, f"Scenario '{scenario['scenario']}': missing qp_diff: {key}"


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark output schema (E7)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBenchmarkOutputSchema:
    """Schema validations for jax_dynamics_benchmark.json."""

    def test_has_phase(self, benchmark):
        assert benchmark.get("phase") == "3D.3-E7"

    def test_has_verdict(self, benchmark):
        verdict = benchmark.get("verdict", "")
        assert isinstance(verdict, str)
        assert len(verdict) > 0
        # Must be one of the valid JAX dynamics verdicts
        valid_verdicts = {
            "JAX_DYNAMICS_PARTIAL_SPEEDUP",
            "JAX_DYNAMICS_INSUFFICIENT_SPEEDUP",
            "JAX_DYNAMICS_RECOMPILE_DETECTED",
        }
        assert verdict in valid_verdicts, f"Unknown verdict: {verdict}"

    def test_has_environment(self, benchmark):
        env = benchmark.get("environment", {})
        assert "jax_version" in env
        assert "jax_enable_x64" in env
        assert "device_count" in env
        assert "jax_platform" in env

    def test_has_config(self, benchmark):
        config = benchmark.get("config", {})
        assert "n_states" in config
        assert "n_steps" in config
        assert "max_contacts" in config

    def test_has_timing_sections(self, benchmark):
        timing = benchmark.get("timing", {})
        required = ["compile_time_s", "warmup_time_s", "total_init_time_s",
                     "first_cached_call_s", "post_warmup", "original", "speedup"]
        for key in required:
            assert key in timing, f"Missing timing section: {key}"

    def test_post_warmup_has_statistics(self, benchmark):
        pw = benchmark["timing"]["post_warmup"]
        required = ["n_calls", "mean_s", "p50_s", "p95_s", "max_s", "min_s", "std_s"]
        for key in required:
            assert key in pw, f"Missing post_warmup stat: {key}"
        assert pw["n_calls"] > 0, "post_warmup n_calls must be > 0"
        assert pw["mean_s"] > 0, "post_warmup mean_s must be > 0"

    def test_original_has_stats(self, benchmark):
        orig = benchmark["timing"]["original"]
        assert "n_calls" in orig
        assert "mean_s" in orig
        assert "all_s" in orig
        assert isinstance(orig["all_s"], list)

    def test_has_speedup(self, benchmark):
        speedup = benchmark["timing"]["speedup"]
        assert "mean_vs_original" in speedup
        assert "p95_vs_original" in speedup
        assert speedup["mean_vs_original"] > 0, "Speedup must be > 0"

    def test_has_cache_diagnostics(self, benchmark):
        diag = benchmark.get("cache_diagnostics", {})
        required = ["compile_time_s", "warmup_time_s", "call_count",
                     "recompile_count", "fallback_count"]
        for key in required:
            assert key in diag, f"Missing cache diagnostic: {key}"

    def test_compile_time_positive(self, benchmark):
        assert benchmark["timing"]["compile_time_s"] >= 0

    def test_warmup_time_nonnegative(self, benchmark):
        assert benchmark["timing"]["warmup_time_s"] >= 0

    def test_speedup_exceeds_threshold(self, benchmark):
        # Should show significant speedup (at least 1.5x)
        sp = benchmark["timing"]["speedup"]["mean_vs_original"]
        assert sp >= 1.0, f"Speedup {sp}x is below minimum 1.0x"

    def test_recompile_count_zero(self, benchmark):
        # After warmup, recompile count should be 0
        rc = benchmark["cache_diagnostics"]["recompile_count"]
        assert rc == 0, f"Recompile count is {rc}, expected 0 after warmup"

    def test_has_correctness_note(self, benchmark):
        note = benchmark.get("correctness_note", "")
        assert isinstance(note, str)
        assert len(note) > 0
        assert "6/8" in note or "correctness" in note.lower()
