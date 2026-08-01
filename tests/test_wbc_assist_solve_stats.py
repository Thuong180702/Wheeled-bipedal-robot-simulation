"""Regression test for F5: the assist arm's solve counters read the wrong key.

``compare_three_arm_rollout`` used to count assist entries containing
``"wbc_result"``, but the batch runner writes the assist arm's per-step record
under ``"assist_result"`` (scripts/phase3d_full_batch_execution.py). Both
``assist_total`` and ``assist_successes`` were therefore structurally zero no
matter what the QP did, which silently hides a fail-closed assist arm.
"""
from __future__ import annotations

from wheeled_biped.wbc.offline_three_arm_counterfactual import (
    compare_three_arm_rollout,
)

N = 8


def _entries(result_key: str) -> list[dict]:
    metrics = {
        "fall": False,
        "safety_fail": False,
        "base_height": 0.53,
        "roll_rad": 0.0,
        "pitch_rad": 0.0,
        "yaw_drift_rad": 0.0,
        "total_planar_drift_m": 0.0,
    }
    return [
        {"metrics": dict(metrics), result_key: {"solve_success": True}}
        for _ in range(N)
    ]


def test_assist_solve_stats_count_assist_result_entries():
    out = compare_three_arm_rollout(
        v3_entries=_entries("v3_result"),
        wbc_entries=_entries("wbc_result"),
        assist_entries=_entries("assist_result"),
        constants={},
    )
    stats = out["wbc_solve_stats"]

    assert stats["assist_total"] == N, (
        f"assist_total={stats['assist_total']}, expected {N} -- the counter is "
        "reading the wrong per-step key again (F5)"
    )
    assert stats["assist_successes"] == N
    assert stats["assist_success_rate"] == 1.0
    # The WBC arm was already correct; guard it against a copy-paste regression.
    assert stats["wbc_only_total"] == N
    assert stats["wbc_only_successes"] == N


if __name__ == "__main__":
    test_assist_solve_stats_count_assist_result_entries()
    print("ok")
