"""Phase 3D — Offline Counterfactual Evaluation Helpers.

Provides convenience functions for loading results, computing aggregate
metrics, and checking readiness gates for the three-arm counterfactual
evaluation.

All functions are offline only. No realtime integration.
"""

from __future__ import annotations

from typing import Any
import json
from pathlib import Path


def load_counterfactual_results(jsonl_path: str | Path) -> list[dict[str, Any]]:
    """Load all counterfactual results from JSONL.

    Args:
        jsonl_path: path to JSONL results file.

    Returns:
        list of result entries.
    """
    entries = []
    if not Path(jsonl_path).exists():
        return entries

    seen = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                key = (entry.get("scenario"), entry.get("arm"), entry.get("suite"))
                if key not in seen:
                    seen.add(key)
                    entries.append(entry)
            except (json.JSONDecodeError, KeyError):
                continue
    return entries


def check_readiness_gates(report: dict[str, Any]) -> dict[str, Any]:
    """Check all Phase 3D readiness gates against a report dict.

    Args:
        report: JSON report dict from generate_reports.

    Returns:
        dict with per-gate results and overall verdict.
    """
    gates = {}

    safety = report.get("safety_comparison", {})
    poc = report.get("physical_outcome_comparison", {})
    assist = poc.get("assist", {})
    wbc = poc.get("wbc_only", {})
    ca = report.get("counterfactual_audit", {})

    # Gate 1: Assist falls <= V3 falls
    gates["assist_falls_le_v3"] = safety.get("assist_falls", 0) <= safety.get("v3_falls", 0)

    # Gate 2: Assist safety fails <= V3 safety fails
    gates["assist_safety_le_v3"] = safety.get("assist_safety_fails", 0) <= safety.get("v3_safety_fails", 0)

    # Gate 3: NaN/Inf = 0
    gates["nan_inf_zero"] = safety.get("nan_inf_count", 0) == 0

    # Gate 4: Torque limit violations = 0
    gates["torque_limit_violations_zero"] = safety.get("torque_limit_violations", 0) == 0

    # Gate 5: WBC solve rate >= 99%
    wbc_rate = ca.get("wbc_solve_success_rate")
    gates["wbc_solve_rate_ok"] = wbc_rate is None or wbc_rate >= 0.99

    # Gate 6: Controller not modified
    gates["controller_not_modified"] = not report.get("controller_modified", True)

    # Gate 7: WBC torque only in offline clones
    gates["wbc_torque_offline_only"] = report.get("wbc_torque_applied_only_to_offline_clones", False)

    # Gate 8: Assist torque only in offline clones
    gates["assist_torque_offline_only"] = report.get("assist_torque_applied_only_to_offline_clones", False)

    # Gate 9: Assist improved or equivalent in >= 70% of scenarios
    total = (assist.get("improved", 0) + assist.get("equivalent", 0) +
             assist.get("mixed", 0) + assist.get("regressed", 0) +
             assist.get("safety_fail", 0))
    assist_good = assist.get("improved", 0) + assist.get("equivalent", 0)
    gates["assist_good_rate_ok"] = total == 0 or assist_good / total >= 0.70

    # Gate 10: Material regression in <= 20% of scenarios
    assist_bad = assist.get("regressed", 0) + assist.get("safety_fail", 0)
    gates["assist_bad_rate_ok"] = total == 0 or assist_bad / total <= 0.20

    all_ok = all(gates.values())

    return {
        "gates": gates,
        "all_passed": all_ok,
        "verdict": "READY_FOR_PHASE_3E_GUARDED_WBC_ASSIST_EXPERIMENT" if all_ok else "PARTIAL_READY",
    }


def compute_aggregate_metrics(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate metrics across all scenario entries.

    Args:
        entries: list of per-scenario result entries.

    Returns:
        dict with aggregate metrics.
    """
    import numpy as np

    comparisons = [e.get("comparison", {}) for e in entries if e.get("arm") == "comparison"]

    if not comparisons:
        return {"n_scenarios": 0}

    # Collect metrics
    wbc_improved = sum(1 for c in comparisons if "IMPROVED" in c.get("classification", {}).get("wbc_only", ""))
    wbc_equivalent = sum(1 for c in comparisons if "EQUIVALENT" in c.get("classification", {}).get("wbc_only", ""))
    wbc_mixed = sum(1 for c in comparisons if "MIXED" in c.get("classification", {}).get("wbc_only", ""))
    wbc_regressed = sum(1 for c in comparisons if "REGRESSED" in c.get("classification", {}).get("wbc_only", ""))
    wbc_safety = sum(1 for c in comparisons if "SAFETY_FAIL" in c.get("classification", {}).get("wbc_only", ""))

    assist_improved = sum(1 for c in comparisons if "IMPROVED" in c.get("classification", {}).get("assist", ""))
    assist_equivalent = sum(1 for c in comparisons if "EQUIVALENT" in c.get("classification", {}).get("assist", ""))
    assist_mixed = sum(1 for c in comparisons if "MIXED" in c.get("classification", {}).get("assist", ""))
    assist_regressed = sum(1 for c in comparisons if "REGRESSED" in c.get("classification", {}).get("assist", ""))
    assist_safety = sum(1 for c in comparisons if "SAFETY_FAIL" in c.get("classification", {}).get("assist", ""))

    best_arms = {"V3_BASELINE": 0, "WBC_ONLY": 0, "V3_PLUS_WBC_ASSIST": 0, "INCONCLUSIVE": 0}
    for c in comparisons:
        ba = c.get("best_arm", "INCONCLUSIVE")
        if ba in best_arms:
            best_arms[ba] += 1

    return {
        "n_scenarios": len(comparisons),
        "wbc_only": {
            "improved": wbc_improved,
            "equivalent": wbc_equivalent,
            "mixed": wbc_mixed,
            "regressed": wbc_regressed,
            "safety_fail": wbc_safety,
        },
        "assist": {
            "improved": assist_improved,
            "equivalent": assist_equivalent,
            "mixed": assist_mixed,
            "regressed": assist_regressed,
            "safety_fail": assist_safety,
        },
        "best_arms": best_arms,
    }
