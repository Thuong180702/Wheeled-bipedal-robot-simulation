"""
Tests for CurriculumManager logic.

Focusses on _evaluate_promotion() (promote / hold / demote branches)
and the promote() / demote() state-machine without needing a real trainer or env.
All tests run fully on CPU with no JAX compilation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixtures — build a CurriculumManager without touching disk
# ---------------------------------------------------------------------------

def _make_manager(num_stages: int = 3, promotion_threshold: float = 0.8,
                  demotion_threshold: float = 0.3, window: int = 10,
                  success_value: float = 1.0,
                  max_retries: int = 5) -> "CurriculumManager":
    """
    Build a CurriculumManager whose stages list is fabricated in memory
    (no YAML file needed) and whose PPO trainer creation is never called.
    """
    from wheeled_biped.training.curriculum import CurriculumManager

    # Patch load_yaml so no file I/O happens
    fake_config = {
        "curriculum": {
            "stages": [
                {"name": f"stage_{i}", "config": "dummy.yaml", "success_value": success_value}
                for i in range(num_stages)
            ],
            "promotion_threshold": promotion_threshold,
            "promotion_window": window,
            "demotion_threshold": demotion_threshold,
            "max_stage_steps": 1_000,
            "max_retries_per_stage": max_retries,
        }
    }

    with patch("wheeled_biped.training.curriculum.load_yaml", return_value=fake_config):
        with patch("wheeled_biped.training.curriculum.Path.mkdir"):  # suppress mkdir
            mgr = CurriculumManager.__new__(CurriculumManager)
            # Manually replicate __init__ without disk I/O
            mgr.stages = fake_config["curriculum"]["stages"]
            mgr.promotion_threshold = promotion_threshold
            mgr.promotion_window = window
            mgr.demotion_threshold = demotion_threshold
            mgr.max_stage_steps = 1_000
            mgr.max_retries_per_stage = max_retries
            mgr.current_stage_idx = 0
            mgr.output_dir = Path("/tmp/fake_curriculum")
            mgr._performance_history = []

    return mgr


# ---------------------------------------------------------------------------
# Tests: _evaluate_promotion branches
# ---------------------------------------------------------------------------

class TestEvaluatePromotion:
    """Direct unit-tests of the 3-way promotion decision."""

    def test_hold_when_window_not_full(self):
        """Returns 'continue' when fewer entries than promotion_window."""
        mgr = _make_manager(window=10, success_value=1.0)
        # Only 5 entries (< window=10)
        for _ in range(5):
            result = mgr._evaluate_promotion(avg_reward=2.0)
        assert result == "continue", f"Expected 'continue', got '{result}'"

    def test_promote_when_above_threshold(self):
        """Returns 'promote' when ≥80% of window beats success_value."""
        mgr = _make_manager(window=10, promotion_threshold=0.8, success_value=1.0)
        # All 10 entries well above success_value=1.0
        result = None
        for _ in range(10):
            result = mgr._evaluate_promotion(avg_reward=2.0)
        assert result == "promote", f"Expected 'promote', got '{result}'"

    def test_hold_when_partial_success(self):
        """Returns 'continue' when success rate is between thresholds."""
        mgr = _make_manager(
            window=10, promotion_threshold=0.8, demotion_threshold=0.3, success_value=1.0
        )
        # 5/10 success (50%) — between demotion (30%) and promotion (80%)
        result = None
        for i in range(10):
            reward = 2.0 if i < 5 else 0.0    # 5 success, 5 fail
            result = mgr._evaluate_promotion(avg_reward=reward)
        assert result == "continue", f"Expected 'continue', got '{result}'"

    def test_demote_when_poor_performance_nonzero_stage(self):
        """Returns 'demote' when success rate < demotion_threshold and stage > 0."""
        mgr = _make_manager(
            window=10, promotion_threshold=0.8, demotion_threshold=0.3, success_value=1.0
        )
        mgr.current_stage_idx = 1   # must be > 0 to allow demotion

        # All rewards well below success_value → 0% success rate
        result = None
        for _ in range(10):
            result = mgr._evaluate_promotion(avg_reward=0.0)
        assert result == "demote", f"Expected 'demote', got '{result}'"

    def test_no_demote_at_stage_zero(self):
        """Returns 'continue' (not 'demote') when already at stage 0."""
        mgr = _make_manager(
            window=10, promotion_threshold=0.8, demotion_threshold=0.3, success_value=1.0
        )
        mgr.current_stage_idx = 0   # cannot demote below 0

        result = None
        for _ in range(10):
            result = mgr._evaluate_promotion(avg_reward=0.0)
        # Should be 'continue', not 'demote'
        assert result != "demote", f"Should not demote at stage 0, got '{result}'"

    def test_history_accumulates(self):
        """_performance_history grows with each call."""
        mgr = _make_manager(window=10, success_value=1.0)
        for i in range(7):
            mgr._evaluate_promotion(avg_reward=float(i))
        assert len(mgr._performance_history) == 7

    def test_promote_uses_most_recent_window(self):
        """Promotion uses only the last N entries, not the entire history."""
        mgr = _make_manager(window=5, promotion_threshold=0.8, success_value=1.0)
        # First 10 entries: bad
        for _ in range(10):
            mgr._evaluate_promotion(avg_reward=0.0)
        # Next 5 entries: all good — should trigger promote
        result = None
        for _ in range(5):
            result = mgr._evaluate_promotion(avg_reward=5.0)
        assert result == "promote", f"Expected 'promote' on recent window, got '{result}'"


# ---------------------------------------------------------------------------
# Tests: promote() / demote() state machine
# ---------------------------------------------------------------------------

class TestStateMachine:
    def test_promote_increments_index(self):
        """promote() advances current_stage_idx by 1."""
        mgr = _make_manager(num_stages=3)
        assert mgr.current_stage_idx == 0
        promoted = mgr.promote()
        assert promoted is True
        assert mgr.current_stage_idx == 1

    def test_promote_clears_history(self):
        """promote() resets _performance_history."""
        mgr = _make_manager(num_stages=3)
        mgr._performance_history = [1.0, 2.0, 3.0]
        mgr.promote()
        assert mgr._performance_history == []

    def test_promote_at_last_stage_returns_false(self):
        """promote() at the final stage returns False and does not advance."""
        mgr = _make_manager(num_stages=2)
        mgr.current_stage_idx = 1   # last stage (0-indexed)
        promoted = mgr.promote()
        assert promoted is False
        assert mgr.current_stage_idx == 1

    def test_demote_decrements_index(self):
        """demote() moves current_stage_idx back by 1."""
        mgr = _make_manager(num_stages=3)
        mgr.current_stage_idx = 2
        demoted = mgr.demote()
        assert demoted is True
        assert mgr.current_stage_idx == 1

    def test_demote_clears_history(self):
        """demote() resets _performance_history."""
        mgr = _make_manager(num_stages=3)
        mgr.current_stage_idx = 1
        mgr._performance_history = [0.0, 0.1]
        mgr.demote()
        assert mgr._performance_history == []

    def test_demote_noop_at_stage_zero(self):
        """demote() at stage 0 returns False and keeps index at 0."""
        mgr = _make_manager(num_stages=3)
        mgr.current_stage_idx = 0
        demoted = mgr.demote()
        assert demoted is False
        assert mgr.current_stage_idx == 0

    def test_is_complete_after_all_stages(self):
        """is_complete returns True after advancing past the last stage."""
        mgr = _make_manager(num_stages=2)
        assert not mgr.is_complete
        mgr.current_stage_idx = 2   # past all stages
        assert mgr.is_complete

    def test_num_stages_matches_config(self):
        """num_stages property matches number of stages in config."""
        mgr = _make_manager(num_stages=5)
        assert mgr.num_stages == 5

    def test_current_stage_returns_correct_dict(self):
        """current_stage property returns the dict for current_stage_idx."""
        mgr = _make_manager(num_stages=3)
        mgr.current_stage_idx = 1
        assert mgr.current_stage["name"] == "stage_1"

    def test_sequential_promote_demote_cycle(self):
        """Promote then demote ends back at original index."""
        mgr = _make_manager(num_stages=3)
        mgr.current_stage_idx = 1
        mgr.promote()
        assert mgr.current_stage_idx == 2
        mgr.demote()
        assert mgr.current_stage_idx == 1


# ---------------------------------------------------------------------------
# Tests: run() — performance-gated main loop
# ---------------------------------------------------------------------------

def _make_stub_trainer(best_reward: float):
    """Return a MagicMock trainer whose .train() returns the given best_reward."""
    stub = MagicMock()
    stub.train.return_value = {"best_reward": best_reward, "total_steps": 100}
    return stub, MagicMock()   # (trainer, logger)


class TestCurriculumRun:
    """Integration-style smoke tests for CurriculumManager.run().

    _create_trainer_for_stage is monkeypatched so no real env/network is created.
    """

    def _run_with_rewards(self, mgr, rewards_per_call: list[float]) -> dict:
        """Run mgr.run() where each _create_trainer_for_stage call returns a stub
        trainer that reports the next reward from rewards_per_call."""
        call_iter = iter(rewards_per_call)

        def _fake_create(stage_idx):
            try:
                reward = next(call_iter)
            except StopIteration:
                reward = float("inf")   # always promote if list is exhausted
            return _make_stub_trainer(reward)

        mgr._create_trainer_for_stage = _fake_create
        with patch("wheeled_biped.training.curriculum.Path.mkdir"):
            return mgr.run(total_steps_per_stage=100)

    # ---- promote path -------------------------------------------------------

    def test_run_promotes_on_good_reward(self):
        """run() advances through all stages when every call earns a high reward.

        With window=2, success_value=1.0, promotion_threshold=0.8:
        - Two rewards of 2.0 fill the window at 100% success → promote.
        - Repeated per remaining stages until all stages done.
        """
        mgr = _make_manager(
            num_stages=2,
            window=2,
            promotion_threshold=0.8,
            success_value=1.0,
            max_retries=10,
        )
        # Stage 0 needs 2 calls to fill window, then promotes and moves to stage 1.
        # Stage 1 needs 2 more calls, then promote returns False (last stage).
        rewards = [2.0, 2.0,   # stage 0: 2 calls → promote
                   2.0, 2.0]   # stage 1: 2 calls → promote (returns False)
        results = self._run_with_rewards(mgr, rewards)

        assert mgr.is_complete or mgr.current_stage_idx == mgr.num_stages - 1
        assert "stage_0" in results
        assert "stage_1" in results

    def test_run_final_stage_completes(self):
        """run() terminates after the final stage promotes (returns False)."""
        mgr = _make_manager(num_stages=1, window=1, promotion_threshold=0.5,
                            success_value=1.0, max_retries=5)
        rewards = [2.0]   # 1 call fills window, 100% success → promote → False → break
        results = self._run_with_rewards(mgr, rewards)
        assert "stage_0" in results

    # ---- hold / force-promote path ------------------------------------------

    def test_run_hold_then_force_promotes(self):
        """run() force-promotes after max_retries_per_stage hold decisions.

        With window=100 and only 1 call per attempt, the window never fills
        (returns 'continue' every time). After max_retries attempts, run()
        should force-promote and move to the next stage.
        """
        max_retries = 2
        mgr = _make_manager(
            num_stages=2,
            window=100,           # window never fills in 1 call
            promotion_threshold=0.8,
            success_value=1.0,
            max_retries=max_retries,
        )
        # Stage 0: 2 hold calls → force-promote to stage 1
        # Stage 1: 2 hold calls → force-promote → promote() returns False → break
        rewards = [0.0] * 10   # all below success_value, window never fills
        results = self._run_with_rewards(mgr, rewards)

        assert "stage_0" in results
        # After force-promote through stage 0 and 1, loop breaks
        assert "stage_1" in results

    def test_run_hold_increments_attempt_counter(self):
        """Each hold decision increments the attempt counter for the stage."""
        max_retries = 3
        mgr = _make_manager(
            num_stages=2,
            window=50,            # large window → always 'continue'
            promotion_threshold=0.9,
            success_value=10.0,
            max_retries=max_retries,
        )
        train_call_count = {"n": 0}

        def _fake_create(stage_idx):
            train_call_count["n"] += 1
            return _make_stub_trainer(best_reward=0.0)

        mgr._create_trainer_for_stage = _fake_create
        with patch("wheeled_biped.training.curriculum.Path.mkdir"):
            mgr.run(total_steps_per_stage=10)

        # stage 0: max_retries=3 hold calls, then force-promote
        # stage 1: max_retries=3 hold calls, then force-promote (returns False)
        # Total calls = max_retries * num_stages = 3 * 2 = 6
        assert train_call_count["n"] == max_retries * 2

    # ---- demote path --------------------------------------------------------

    def test_run_demotes_on_poor_reward(self):
        """run() demotes to stage 0 when performance is poor on stage 1.

        Sequence:
          - stage 0 → window fills with good rewards → promote to stage 1
          - stage 1 → window fills with very poor rewards → demote to stage 0
          - stage 0 → max_retries exhausted with no fill → force-promote stage 1
          - stage 1 → max_retries exhausted → force-promote → done
        """
        max_retries = 1
        mgr = _make_manager(
            num_stages=2,
            window=2,
            promotion_threshold=0.8,
            demotion_threshold=0.5,
            success_value=1.0,
            max_retries=max_retries,
        )
        # stage 0 attempt 1: 2 calls with reward=2.0 → promote (stage 1)
        # stage 1 attempt 1: 2 calls with reward=0.0 (0% success < 50%) → demote (stage 0)
        # stage 0 attempt 1 (post-demote): 1 call → max_retries=1 → force-promote (stage 1)
        # stage 1 attempt 1 (post-recover): 1 call → max_retries=1 → force-promote → False → break
        rewards = [2.0, 2.0,   # stage 0: full window → promote
                   0.0, 0.0,   # stage 1: full window → demote
                   0.0,        # stage 0: max_retries hit → force-promote
                   0.0]        # stage 1: max_retries hit → force-promote → done
        results = self._run_with_rewards(mgr, rewards)

        # Both stages should appear (stage_0 may appear twice in results dict
        # but dict update means we see the last run)
        assert "stage_0" in results
        assert "stage_1" in results
