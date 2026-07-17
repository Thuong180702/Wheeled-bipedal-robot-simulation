"""Slow tests for Phase 3C — Offline Rolling Constraints Audit Validation.

Validates the completed audit JSONL/JSON against Phase 3C readiness gates.
All tests are marked @pytest.mark.slow.

Requires `outputs/phase3c_rolling_audit_results.jsonl` to exist.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Preload Phase 3C modules (not registered in editable install finder)
import importlib.util as _iu
for _name, _relpath in [
    ("wheeled_biped.wbc.offline_rolling_constraints",
     "wheeled_biped/wbc/offline_rolling_constraints.py"),
    ("wheeled_biped.wbc.phase3c_rolling_qp",
     "wheeled_biped/wbc/phase3c_rolling_qp.py"),
]:
    if _name not in sys.modules:
        _spec = _iu.spec_from_file_location(_name, str(PROJECT_ROOT / _relpath))
        _mod = _iu.module_from_spec(_spec)
        sys.modules[_name] = _mod
        _spec.loader.exec_module(_mod)

pytestmark = pytest.mark.slow

JSONL_PATH = PROJECT_ROOT / "outputs" / "phase3c_rolling_audit_results.jsonl"
REPORT_JSON_PATH = PROJECT_ROOT / "docs" / "validation" / "k2_phase3c_offline_rolling_audit.json"


def _load_entries():
    """Load all audit entries from JSONL."""
    if not JSONL_PATH.exists():
        return []
    entries = []
    seen = set()
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                key = (entry.get("scenario"), entry.get("task_mode"), entry.get("rolling_mode"))
                if key not in seen:
                    seen.add(key)
                    entries.append(entry)
            except (json.JSONDecodeError, KeyError):
                continue
    return entries


def _load_report():
    """Load JSON report if it exists."""
    if not REPORT_JSON_PATH.exists():
        return None
    with open(REPORT_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: total_qp_solves_completed == 120
# ═══════════════════════════════════════════════════════════════════════════

def test_total_solves_120():
    """Test 1: total QP solves completed == 120."""
    entries = _load_entries()
    if not entries:
        pytest.skip("No audit JSONL found — run phase3c_offline_rolling_audit.py first")
    assert len(entries) == 120, f"Expected 120 entries, got {len(entries)}"


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: normal_only feasibility solves 12/12
# ═══════════════════════════════════════════════════════════════════════════

def test_normal_only_feasibility_12_of_12():
    """Test 2: normal_only feasibility solves 12/12."""
    entries = _load_entries()
    if not entries:
        pytest.skip("No audit JSONL found")
    matching = [e for e in entries
                if e.get("task_mode") == "feasibility_only"
                and e.get("rolling_mode") == "normal_only"]
    solved = [e for e in matching if e.get("solved", False)]
    assert len(matching) == 12, f"Expected 12 entries, got {len(matching)}"
    assert len(solved) == 12, f"Expected 12 solved, got {len(solved)}"


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: normal_only balanced solves 12/12
# ═══════════════════════════════════════════════════════════════════════════

def test_normal_only_balanced_12_of_12():
    """Test 3: normal_only balanced solves 12/12."""
    entries = _load_entries()
    if not entries:
        pytest.skip("No audit JSONL found")
    matching = [e for e in entries
                if e.get("task_mode") == "balanced_default"
                and e.get("rolling_mode") == "normal_only"]
    solved = [e for e in matching if e.get("solved", False)]
    assert len(matching) == 12, f"Expected 12 entries, got {len(matching)}"
    assert len(solved) == 12, f"Expected 12 solved, got {len(solved)}"


# ═══════════════════════════════════════════════════════════════════════════
# Test 4: lateral_soft feasibility solves 12/12
# ═══════════════════════════════════════════════════════════════════════════

def test_lateral_soft_feasibility_12_of_12():
    """Test 4: lateral_soft feasibility solves 12/12."""
    entries = _load_entries()
    if not entries:
        pytest.skip("No audit JSONL found")
    matching = [e for e in entries
                if e.get("task_mode") == "feasibility_only"
                and e.get("rolling_mode") == "lateral_soft"]
    solved = [e for e in matching if e.get("solved", False)]
    assert len(matching) == 12
    assert len(solved) == 12, f"Expected 12 solved, got {len(solved)}"


# ═══════════════════════════════════════════════════════════════════════════
# Test 5: lateral_soft balanced solves 12/12
# ═══════════════════════════════════════════════════════════════════════════

def test_lateral_soft_balanced_12_of_12():
    """Test 5: lateral_soft balanced solves 12/12."""
    entries = _load_entries()
    if not entries:
        pytest.skip("No audit JSONL found")
    matching = [e for e in entries
                if e.get("task_mode") == "balanced_default"
                and e.get("rolling_mode") == "lateral_soft"]
    solved = [e for e in matching if e.get("solved", False)]
    assert len(matching) == 12
    assert len(solved) == 12, f"Expected 12 solved, got {len(solved)}"


# ═══════════════════════════════════════════════════════════════════════════
# Test 6: full_rolling_soft feasibility solves 12/12
# ═══════════════════════════════════════════════════════════════════════════

def test_full_rolling_soft_feasibility_12_of_12():
    """Test 6: full_rolling_soft feasibility solves 12/12."""
    entries = _load_entries()
    if not entries:
        pytest.skip("No audit JSONL found")
    matching = [e for e in entries
                if e.get("task_mode") == "feasibility_only"
                and e.get("rolling_mode") == "full_rolling_soft"]
    solved = [e for e in matching if e.get("solved", False)]
    assert len(matching) == 12
    assert len(solved) == 12, f"Expected 12 solved, got {len(solved)}"


# ═══════════════════════════════════════════════════════════════════════════
# Test 7: full_rolling_soft balanced solves 12/12
# ═══════════════════════════════════════════════════════════════════════════

def test_full_rolling_soft_balanced_12_of_12():
    """Test 7: full_rolling_soft balanced solves 12/12."""
    entries = _load_entries()
    if not entries:
        pytest.skip("No audit JSONL found")
    matching = [e for e in entries
                if e.get("task_mode") == "balanced_default"
                and e.get("rolling_mode") == "full_rolling_soft"]
    solved = [e for e in matching if e.get("solved", False)]
    assert len(matching) == 12
    assert len(solved) == 12, f"Expected 12 solved, got {len(solved)}"


# ═══════════════════════════════════════════════════════════════════════════
# Test 8: lateral_hard balanced solves at least 10/12
# ═══════════════════════════════════════════════════════════════════════════

def test_lateral_hard_balanced_at_least_10():
    """Test 8: lateral_hard balanced solves >= 10/12."""
    entries = _load_entries()
    if not entries:
        pytest.skip("No audit JSONL found")
    matching = [e for e in entries
                if e.get("task_mode") == "balanced_default"
                and e.get("rolling_mode") == "lateral_hard"]
    solved = [e for e in matching if e.get("solved", False)]
    assert len(matching) == 12
    assert len(solved) >= 10, f"Expected >=10 solved, got {len(solved)}"


# ═══════════════════════════════════════════════════════════════════════════
# Test 9: full_rolling_hard attempted all 12 scenarios
# ═══════════════════════════════════════════════════════════════════════════

def test_full_rolling_hard_attempted_12():
    """Test 9: full_rolling_hard attempted all 12 scenarios."""
    entries = _load_entries()
    if not entries:
        pytest.skip("No audit JSONL found")
    # Check feasibility_only (should always attempt)
    matching = [e for e in entries
                if e.get("task_mode") == "feasibility_only"
                and e.get("rolling_mode") == "full_rolling_hard"]
    assert len(matching) == 12, f"Expected 12 full_rolling_hard attempts, got {len(matching)}"
    # At least some should have results (solved or not)
    attempted = len(matching)
    assert attempted == 12


# ═══════════════════════════════════════════════════════════════════════════
# Test 10: Hard constraints pass in all solved cases
# ═══════════════════════════════════════════════════════════════════════════

def test_hard_constraints_pass_all_solved():
    """Test 10: hard constraints pass in all solved cases."""
    entries = _load_entries()
    if not entries:
        pytest.skip("No audit JSONL found")
    solved = [e for e in entries if e.get("solved", False)]
    assert len(solved) > 0, "No solved entries to check"

    failures = []
    for e in solved:
        if e.get("max_dynamics_residual", float("inf")) >= 1e-5:
            failures.append(f"{e['scenario']}/{e['task_mode']}/{e['rolling_mode']}: dyn={e['max_dynamics_residual']:.2e}")
        if e.get("max_contact_accel_residual", float("inf")) >= 1e-4:
            failures.append(f"{e['scenario']}/{e['task_mode']}/{e['rolling_mode']}: contact_accel={e['max_contact_accel_residual']:.2e}")
        if e.get("max_friction_violation", float("inf")) > 1e-6:
            failures.append(f"{e['scenario']}/{e['task_mode']}/{e['rolling_mode']}: friction={e['max_friction_violation']:.2e}")
        if e.get("max_torque_violation", float("inf")) > 1e-6:
            failures.append(f"{e['scenario']}/{e['task_mode']}/{e['rolling_mode']}: torque={e['max_torque_violation']:.2e}")
        if e.get("max_abs_qdd", float("inf")) >= 100:
            failures.append(f"{e['scenario']}/{e['task_mode']}/{e['rolling_mode']}: qdd={e['max_abs_qdd']:.1f}")
        if e.get("max_abs_lambda", float("inf")) >= 500:
            failures.append(f"{e['scenario']}/{e['task_mode']}/{e['rolling_mode']}: lambda={e['max_abs_lambda']:.1f}")

    assert len(failures) == 0, f"Hard constraint failures in solved cases:\n" + "\n".join(failures)


# ═══════════════════════════════════════════════════════════════════════════
# Test 11: Controller unchanged
# ═══════════════════════════════════════════════════════════════════════════

def test_controller_unchanged():
    """Test 11: controller files were not modified by Phase 3C."""
    report = _load_report()
    if report is None:
        pytest.skip("No report JSON found")
    assert report.get("controller_modified", True) is False


# ═══════════════════════════════════════════════════════════════════════════
# Test 12: No realtime integration
# ═══════════════════════════════════════════════════════════════════════════

def test_no_realtime_integration():
    """Test 12: no realtime integration in Phase 3C."""
    report = _load_report()
    if report is None:
        pytest.skip("No report JSON found")
    assert report.get("realtime_integration", True) is False


# ═══════════════════════════════════════════════════════════════════════════
# Test 13: Rolling residuals finite in all solved cases
# ═══════════════════════════════════════════════════════════════════════════

def test_rolling_residuals_finite():
    """Test 13: rolling residuals are finite in all solved cases."""
    entries = _load_entries()
    if not entries:
        pytest.skip("No audit JSONL found")
    solved = [e for e in entries if e.get("solved", False)]
    for e in solved:
        pre_lat = e.get("pre_max_lat_slip", 0)
        pre_roll = e.get("pre_max_roll_residual", 0)
        assert np.isfinite(pre_lat), f"Non-finite pre_lat in {e['scenario']}/{e['task_mode']}/{e['rolling_mode']}"
        assert np.isfinite(pre_roll), f"Non-finite pre_roll in {e['scenario']}/{e['task_mode']}/{e['rolling_mode']}"


# ═══════════════════════════════════════════════════════════════════════════
# Test 14: Report JSON contains all required sections
# ═══════════════════════════════════════════════════════════════════════════

def test_report_json_sections():
    """Test 14: report JSON has all required sections."""
    report = _load_report()
    if report is None:
        pytest.skip("No report JSON found")

    required_sections = [
        "phase", "verdict", "constants_version", "phase3b1_cleanup",
        "wheel_geometry", "num_scenarios", "task_modes", "rolling_modes",
        "total_qp_solves_expected", "total_qp_solves_completed",
        "mode_results", "normal_only_regression", "lateral_soft",
        "lateral_hard", "full_rolling_soft", "full_rolling_hard",
        "hard_constraints_pass", "controller_modified",
        "qp_torque_injected", "realtime_integration", "limitations",
    ]
    for section in required_sections:
        assert section in report, f"Missing section: {section}"


# ═══════════════════════════════════════════════════════════════════════════
# Test 15: Wheel geometry populated
# ═══════════════════════════════════════════════════════════════════════════

def test_wheel_geometry_populated():
    """Test 15: wheel geometry section has real values."""
    report = _load_report()
    if report is None:
        pytest.skip("No report JSON found")

    wg = report.get("wheel_geometry", {})
    assert wg.get("wheel_radius_left") is not None
    assert wg.get("wheel_radius_right") is not None
    assert wg.get("wheel_qvel_index_left") is not None
    assert wg.get("wheel_qvel_index_right") is not None
    assert wg["wheel_radius_left"] > 0
    assert wg["wheel_radius_right"] > 0
