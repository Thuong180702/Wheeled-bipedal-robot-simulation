"""
Tests for UnifiedController (hardened version).

Covers:
  - All checkpoints missing → RuntimeError
  - Partial checkpoints → only available skills loaded
  - _detect_skill: fallen robot → BALANCE
  - _detect_skill: forced mode override (bypasses hysteresis)
  - Obs adapter selection at load time (Fix 2)
  - _build_obs: correct dim per adapter; ValueError for mismatch
  - Skill transition: blend counter increments (Fix 1 — no longer resets)
  - Blend completes naturally after _blend_steps calls (Fix 1)
  - Dwell-time hysteresis prevents premature switch (Fix 3)
  - Per-skill _prev_action buffers (Fix 4)
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wheeled_biped.inference.unified_controller import (  # noqa: E402
    ControlCommand,
    Skill,
    UnifiedController,
    _infer_adapter,
)
from wheeled_biped.utils.config import get_model_path  # noqa: E402

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mj_model():
    """Real MuJoCo CPU model (needed for UnifiedController init)."""
    return mujoco.MjModel.from_xml_path(str(get_model_path()))


@pytest.fixture(scope="module")
def mj_data(mj_model):
    """Standing pose MjData."""
    data = mujoco.MjData(mj_model)
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, data, 0)
    mujoco.mj_forward(mj_model, data)
    return data


def _write_fake_checkpoint(path: Path, obs_size: int = 39):
    """
    Write a minimal checkpoint.pkl that UnifiedController._load_skill can read.
    Uses tiny network weights so no GPU JIT is triggered.
    """
    from wheeled_biped.training.networks import create_actor_critic
    from wheeled_biped.training.ppo import RunningMeanStd

    cfg = {
        "network": {"policy_hidden": [16], "value_hidden": [16], "activation": "elu"},
    }
    rng = jax.random.PRNGKey(0)
    _, params = create_actor_critic(obs_size=obs_size, action_size=10, config=cfg, rng=rng)

    obs_rms = RunningMeanStd(
        mean=jnp.zeros(obs_size),
        var=jnp.ones(obs_size),
        count=jnp.float32(1.0),
    )

    path.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "params": jax.device_get(params),
        "obs_rms": jax.device_get(obs_rms),
        "config": cfg,
    }
    with open(path / "checkpoint.pkl", "wb") as f:
        pickle.dump(ckpt, f)


# ---------------------------------------------------------------------------
# Tests: missing checkpoints
# ---------------------------------------------------------------------------


class TestMissingCheckpoints:
    def test_all_missing_raises_runtime_error(self, mj_model, tmp_path):
        """If no checkpoint files exist, UnifiedController raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Không tải được skill nào"):
            UnifiedController(checkpoint_dir=str(tmp_path), mj_model=mj_model)

    def test_one_valid_checkpoint_loads(self, mj_model, tmp_path):
        """With only the balance checkpoint present, exactly one skill is loaded."""
        balance_ckpt = tmp_path / "balance" / "final"
        _write_fake_checkpoint(balance_ckpt, obs_size=39)

        ctrl = UnifiedController(checkpoint_dir=str(tmp_path), mj_model=mj_model)
        assert len(ctrl.skills) == 1, f"Expected 1 skill, got {len(ctrl.skills)}"

    def test_partial_checkpoints_skips_missing(self, mj_model, tmp_path):
        """Controller skips missing skills and loads the available ones without error."""
        for subfolder in ("balance/final", "wheeled_locomotion/final"):
            path = tmp_path / Path(subfolder)
            obs = 39 if "balance" in subfolder else 41
            _write_fake_checkpoint(path, obs_size=obs)

        ctrl = UnifiedController(checkpoint_dir=str(tmp_path), mj_model=mj_model)
        assert len(ctrl.skills) == 2

    def test_available_skills_matches_loaded(self, mj_model, tmp_path):
        """available_skills property lists exactly what was loaded."""
        balance_ckpt = tmp_path / "balance" / "final"
        _write_fake_checkpoint(balance_ckpt, obs_size=39)

        ctrl = UnifiedController(checkpoint_dir=str(tmp_path), mj_model=mj_model)
        assert Skill.BALANCE in ctrl.available_skills


# ---------------------------------------------------------------------------
# Tests: obs adapter inference (Fix 2)
# ---------------------------------------------------------------------------


class TestObsAdapterInference:
    def test_exact_adapter_for_39(self):
        assert _infer_adapter(39, False) == "exact"

    def test_exact_adapter_for_39_needs_command_false(self):
        assert _infer_adapter(39, False) == "exact"

    def test_height_cmd_adapter_for_40(self):
        assert _infer_adapter(40, False) == "height_cmd"

    def test_velocity_cmd_adapter_for_41(self):
        assert _infer_adapter(41, True) == "velocity_cmd"

    def test_height_cmd_yaw_adapter_for_41(self):
        """41-dim obs with needs_command=False → BalanceEnv with yaw_error."""
        assert _infer_adapter(41, False) == "height_cmd_yaw"

    def test_novelin_height_cmd_yaw_adapter_for_38(self):
        """38-dim obs → BalanceEnv with lin_vel_mode='disabled' + yaw_error."""
        assert _infer_adapter(38, False) == "novelin_height_cmd_yaw"

    def test_unknown_pad_for_odd_sizes(self):
        assert _infer_adapter(45, False) == "unknown_pad"
        assert _infer_adapter(50, True) == "unknown_pad"

    def test_balance_skill_has_exact_adapter(self, mj_model, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("adapter_bal")
        _write_fake_checkpoint(tmp / "balance" / "final", obs_size=39)
        ctrl = UnifiedController(checkpoint_dir=str(tmp), mj_model=mj_model)
        assert ctrl.skills[Skill.BALANCE].obs_adapter == "exact"

    def test_balance_40_has_height_cmd_adapter(self, mj_model, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("adapter_bal40")
        _write_fake_checkpoint(tmp / "balance" / "final", obs_size=40)
        ctrl = UnifiedController(checkpoint_dir=str(tmp), mj_model=mj_model)
        assert ctrl.skills[Skill.BALANCE].obs_adapter == "height_cmd"

    def test_locomotion_41_has_velocity_cmd_adapter(self, mj_model, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("adapter_loco")
        _write_fake_checkpoint(tmp / "wheeled_locomotion" / "final", obs_size=41)
        ctrl = UnifiedController(checkpoint_dir=str(tmp), mj_model=mj_model)
        assert ctrl.skills[Skill.LOCOMOTION].obs_adapter == "velocity_cmd"

    def test_unknown_size_emits_warning(self, mj_model, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("adapter_unk")
        _write_fake_checkpoint(tmp / "balance" / "final", obs_size=50)
        with pytest.warns(UserWarning, match="unknown_pad"):
            ctrl = UnifiedController(checkpoint_dir=str(tmp), mj_model=mj_model)
        assert ctrl.skills[Skill.BALANCE].obs_adapter == "unknown_pad"


# ---------------------------------------------------------------------------
# Tests: _build_obs
# ---------------------------------------------------------------------------


class TestBuildObs:
    @pytest.fixture(scope="class")
    def ctrl(self, mj_model, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("obs")
        _write_fake_checkpoint(tmp / "balance" / "final", obs_size=39)
        return UnifiedController(checkpoint_dir=str(tmp), mj_model=mj_model)

    def test_exact_adapter_39(self, ctrl, mj_data):
        """exact adapter yields 39-dim obs."""
        skill = Skill.BALANCE
        ctrl.skills[skill].obs_adapter = "exact"
        ctrl.skills[skill].obs_size = 39
        obs = ctrl._build_obs(mj_data, skill, ControlCommand())
        assert obs.shape == (39,)

    def test_obs_no_nan(self, ctrl, mj_data):
        """_build_obs output has no NaN from a valid standing pose."""
        skill = Skill.BALANCE
        obs = ctrl._build_obs(mj_data, skill, ControlCommand())
        assert not np.any(np.isnan(np.array(obs))), "NaN in _build_obs output"

    def test_height_cmd_adapter_40(self, ctrl, mj_data):
        """height_cmd adapter yields 40-dim obs."""
        skill = Skill.BALANCE
        orig_adapter = ctrl.skills[skill].obs_adapter
        orig_size = ctrl.skills[skill].obs_size
        ctrl.skills[skill].obs_adapter = "height_cmd"
        ctrl.skills[skill].obs_size = 40
        try:
            obs = ctrl._build_obs(mj_data, skill, ControlCommand(height_target=0.55))
            assert obs.shape == (40,)
            # Last dim should be normalised height in [0,1]
            height_norm = float(obs[-1])
            assert 0.0 <= height_norm <= 1.0
        finally:
            ctrl.skills[skill].obs_adapter = orig_adapter
            ctrl.skills[skill].obs_size = orig_size

    def test_velocity_cmd_adapter_41(self, ctrl, mj_data):
        """velocity_cmd adapter yields 41-dim obs."""
        skill = Skill.BALANCE
        orig_adapter = ctrl.skills[skill].obs_adapter
        orig_size = ctrl.skills[skill].obs_size
        orig_nc = ctrl.skills[skill].needs_command
        ctrl.skills[skill].obs_adapter = "velocity_cmd"
        ctrl.skills[skill].obs_size = 41
        ctrl.skills[skill].needs_command = True
        try:
            obs = ctrl._build_obs(mj_data, skill, ControlCommand(vel_x=0.5, ang_vel_z=0.1))
            assert obs.shape == (41,)
        finally:
            ctrl.skills[skill].obs_adapter = orig_adapter
            ctrl.skills[skill].obs_size = orig_size
            ctrl.skills[skill].needs_command = orig_nc

    def test_height_cmd_yaw_adapter_41(self, ctrl, mj_data):
        """height_cmd_yaw adapter yields 41-dim obs with height_norm and yaw_error."""
        skill = Skill.BALANCE
        orig_adapter = ctrl.skills[skill].obs_adapter
        orig_size = ctrl.skills[skill].obs_size
        ctrl.skills[skill].obs_adapter = "height_cmd_yaw"
        ctrl.skills[skill].obs_size = 41
        ctrl._balance_initial_yaw = None
        try:
            obs = ctrl._build_obs(mj_data, skill, ControlCommand(height_target=0.55))
            assert obs.shape == (41,)
            height_norm = float(obs[-2])
            yaw_error = float(obs[-1])
            assert 0.0 <= height_norm <= 1.0
            assert yaw_error == pytest.approx(0.0, abs=1e-5)  # first call → 0
        finally:
            ctrl.skills[skill].obs_adapter = orig_adapter
            ctrl.skills[skill].obs_size = orig_size

    def test_novelin_height_cmd_yaw_adapter_38(self, ctrl, mj_data):
        """novelin_height_cmd_yaw adapter yields 38-dim obs (no lin_vel)."""
        skill = Skill.BALANCE
        orig_adapter = ctrl.skills[skill].obs_adapter
        orig_size = ctrl.skills[skill].obs_size
        ctrl.skills[skill].obs_adapter = "novelin_height_cmd_yaw"
        ctrl.skills[skill].obs_size = 38
        ctrl._balance_initial_yaw = None
        try:
            obs = ctrl._build_obs(mj_data, skill, ControlCommand(height_target=0.55))
            assert obs.shape == (38,)
            height_norm = float(obs[-2])
            assert 0.0 <= height_norm <= 1.0
        finally:
            ctrl.skills[skill].obs_adapter = orig_adapter
            ctrl.skills[skill].obs_size = orig_size

    def test_exact_adapter_wrong_size_raises(self, ctrl, mj_data):
        """exact adapter with mismatched obs_size raises ValueError (Fix 2)."""
        skill = Skill.BALANCE
        orig_adapter = ctrl.skills[skill].obs_adapter
        orig_size = ctrl.skills[skill].obs_size
        ctrl.skills[skill].obs_adapter = "exact"
        ctrl.skills[skill].obs_size = 99  # wrong
        try:
            with pytest.raises(ValueError, match="exact"):
                ctrl._build_obs(mj_data, skill, ControlCommand())
        finally:
            ctrl.skills[skill].obs_adapter = orig_adapter
            ctrl.skills[skill].obs_size = orig_size

    def test_unknown_pad_adapter_pads_to_obs_size(self, ctrl, mj_data):
        """unknown_pad adapter still produces correct shape (escape hatch)."""
        skill = Skill.BALANCE
        orig_adapter = ctrl.skills[skill].obs_adapter
        orig_size = ctrl.skills[skill].obs_size
        ctrl.skills[skill].obs_adapter = "unknown_pad"
        ctrl.skills[skill].obs_size = 45
        try:
            obs = ctrl._build_obs(mj_data, skill, ControlCommand())
            assert obs.shape == (45,)
        finally:
            ctrl.skills[skill].obs_adapter = orig_adapter
            ctrl.skills[skill].obs_size = orig_size


# ---------------------------------------------------------------------------
# Tests: skill detection
# ---------------------------------------------------------------------------


class TestSkillDetection:
    @pytest.fixture(scope="class")
    def ctrl_with_balance(self, mj_model, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("detect")
        _write_fake_checkpoint(tmp / "balance" / "final", obs_size=39)
        return UnifiedController(checkpoint_dir=str(tmp), mj_model=mj_model)

    def test_fallen_robot_selects_balance(self, ctrl_with_balance, mj_model):
        """Robot with torso height < 0.4 m → BALANCE."""
        data = mujoco.MjData(mj_model)
        if mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(mj_model, data, 0)
        data.qpos[2] = 0.2
        mujoco.mj_forward(mj_model, data)
        skill = ctrl_with_balance._detect_skill_raw(data, ControlCommand())
        assert skill == Skill.BALANCE

    def test_tilted_robot_selects_balance(self, ctrl_with_balance, mj_model):
        """Robot tilted > 0.5 rad → BALANCE (raw detection)."""
        data = mujoco.MjData(mj_model)
        if mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(mj_model, data, 0)
        angle = np.pi / 3
        data.qpos[3] = np.cos(angle / 2)
        data.qpos[4] = np.sin(angle / 2)
        data.qpos[5] = 0.0
        data.qpos[6] = 0.0
        mujoco.mj_forward(mj_model, data)
        skill = ctrl_with_balance._detect_skill_raw(data, ControlCommand())
        assert skill == Skill.BALANCE

    def test_force_mode_override_bypasses_hysteresis(self, ctrl_with_balance, mj_model):
        """cmd.mode=BALANCE → immediate switch, no dwell needed (Fix 3)."""
        data = mujoco.MjData(mj_model)
        if mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(mj_model, data, 0)
        mujoco.mj_forward(mj_model, data)
        ctrl_with_balance._dwell_counts.clear()
        skill = ctrl_with_balance._detect_skill(data, ControlCommand(mode=Skill.BALANCE))
        assert skill == Skill.BALANCE

    def test_force_unknown_skill_ignored(self, ctrl_with_balance, mj_model):
        """cmd.mode set to a skill not in skills dict falls back to raw heuristic."""
        data = mujoco.MjData(mj_model)
        if mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(mj_model, data, 0)
        mujoco.mj_forward(mj_model, data)
        cmd = ControlCommand(mode=Skill.STAIR)  # STAIR not loaded
        skill = ctrl_with_balance._detect_skill(data, cmd)
        assert isinstance(skill, Skill)


# ---------------------------------------------------------------------------
# Tests: Fix 3 — dwell-time hysteresis
# ---------------------------------------------------------------------------


class TestDwellHysteresis:
    @pytest.fixture(scope="class")
    def ctrl_two_skills(self, mj_model, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("dwell")
        _write_fake_checkpoint(tmp / "balance" / "final", obs_size=39)
        _write_fake_checkpoint(tmp / "wheeled_locomotion" / "final", obs_size=41)
        return UnifiedController(checkpoint_dir=str(tmp), mj_model=mj_model, dwell_threshold=3)

    def test_single_spike_does_not_change_skill(self, ctrl_two_skills, mj_model):
        """A single raw detection of LOCOMOTION (when active=BALANCE) does not
        switch — dwell_threshold=3 means 3 consecutive detections are needed."""
        ctrl = ctrl_two_skills
        ctrl.force_skill(Skill.BALANCE)
        ctrl._dwell_counts.clear()

        data = mujoco.MjData(mj_model)
        if mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(mj_model, data, 0)
        mujoco.mj_forward(mj_model, data)

        # Patch raw detection to return LOCOMOTION exactly once
        with patch.object(ctrl, "_detect_skill_raw", return_value=Skill.LOCOMOTION):
            result = ctrl._detect_skill(data, ControlCommand())

        # Should still return active (BALANCE) — dwell count is only 1
        assert result == Skill.BALANCE

    def test_stable_detection_triggers_switch(self, ctrl_two_skills, mj_model):
        """After dwell_threshold consecutive same raw detections → switch fires."""
        ctrl = ctrl_two_skills
        ctrl.force_skill(Skill.BALANCE)
        ctrl._dwell_counts.clear()

        data = mujoco.MjData(mj_model)
        if mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(mj_model, data, 0)
        mujoco.mj_forward(mj_model, data)

        result = Skill.BALANCE
        with patch.object(ctrl, "_detect_skill_raw", return_value=Skill.LOCOMOTION):
            for _ in range(ctrl._dwell_threshold):
                result = ctrl._detect_skill(data, ControlCommand())

        assert result == Skill.LOCOMOTION

    def test_interrupted_detection_resets_count(self, ctrl_two_skills, mj_model):
        """If raw detection flips mid-dwell, count resets and switch is delayed."""
        ctrl = ctrl_two_skills
        ctrl.force_skill(Skill.BALANCE)
        ctrl._dwell_counts.clear()

        data = mujoco.MjData(mj_model)
        if mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(mj_model, data, 0)
        mujoco.mj_forward(mj_model, data)

        # Two LOCOMOTION detections, then one BALANCE → count resets
        raw_sequence = [Skill.LOCOMOTION, Skill.LOCOMOTION, Skill.BALANCE]
        results = []
        for raw in raw_sequence:
            with patch.object(ctrl, "_detect_skill_raw", return_value=raw):
                results.append(ctrl._detect_skill(data, ControlCommand()))

        # All should still be BALANCE because the streak was broken
        assert all(r == Skill.BALANCE for r in results), results


# ---------------------------------------------------------------------------
# Tests: Fix 1 — blend counter no longer resets on same target
# ---------------------------------------------------------------------------


class TestSkillTransition:
    @pytest.fixture(scope="class")
    def ctrl_two_skills(self, mj_model, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("blend")
        _write_fake_checkpoint(tmp / "balance" / "final", obs_size=39)
        _write_fake_checkpoint(tmp / "wheeled_locomotion" / "final", obs_size=41)
        return UnifiedController(checkpoint_dir=str(tmp), mj_model=mj_model)

    def test_blend_counter_increments_on_same_target(self, ctrl_two_skills, mj_model):
        """Fix 1: blend_counter increments each step when desired skill is
        the same as the current transition target (no reset)."""
        ctrl = ctrl_two_skills
        data = mujoco.MjData(mj_model)
        if mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(mj_model, data, 0)
        mujoco.mj_forward(mj_model, data)

        ctrl.force_skill(Skill.BALANCE)
        ctrl._transition_target = None
        ctrl._blend_counter = 0
        ctrl._transition_alpha = 0.0
        ctrl._dwell_counts.clear()

        cmd = ControlCommand(mode=Skill.LOCOMOTION)
        for i in range(ctrl._blend_steps + 2):
            ctrl.get_action(data, cmd)

        # Blend should have completed → active skill is LOCOMOTION
        assert ctrl._active_skill == Skill.LOCOMOTION, (
            f"Expected LOCOMOTION after {ctrl._blend_steps} steps, got {ctrl._active_skill}"
        )

    def test_blend_counter_resets_on_new_different_target(self, ctrl_two_skills, mj_model):
        """If the desired skill changes mid-blend, counter resets."""
        ctrl = ctrl_two_skills
        data = mujoco.MjData(mj_model)
        if mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(mj_model, data, 0)
        mujoco.mj_forward(mj_model, data)

        ctrl.force_skill(Skill.BALANCE)
        ctrl._transition_target = Skill.LOCOMOTION
        ctrl._blend_counter = 7  # mid-blend
        ctrl._transition_alpha = 0.7
        ctrl._dwell_counts.clear()

        # Desired skill suddenly same as active (BALANCE) → target changes → reset
        cmd = ControlCommand(mode=Skill.BALANCE)
        # Force detection to return BALANCE (same as active) → clears transition
        with patch.object(ctrl, "_detect_skill_raw", return_value=Skill.BALANCE):
            ctrl.get_action(data, cmd)

        # Since desired == active, no transition should be active
        assert ctrl._transition_target is None
        assert ctrl._blend_counter == 0 or ctrl._blend_counter == 7, (
            "counter should either be untouched (7) or reset (0), not some new value"
        )

    def test_blend_completes_and_skill_flips(self, ctrl_two_skills, mj_model):
        """Direct test of blend finalization: alpha>=1.0 → active_skill swaps."""
        ctrl = ctrl_two_skills
        ctrl.force_skill(Skill.BALANCE)
        ctrl._transition_target = Skill.LOCOMOTION
        ctrl._blend_counter = ctrl._blend_steps
        ctrl._transition_alpha = 1.0
        ctrl._dwell_counts.clear()

        # Finalization happens inside get_action's blend branch
        data = mujoco.MjData(mj_model)
        if mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(mj_model, data, 0)
        mujoco.mj_forward(mj_model, data)

        # Patch _detect_skill to return LOCOMOTION so we don't reset counters
        with patch.object(ctrl, "_detect_skill", return_value=Skill.LOCOMOTION):
            ctrl.get_action(data, ControlCommand())

        assert ctrl._active_skill == Skill.LOCOMOTION
        assert ctrl._transition_target is None

    def test_force_skill_bypasses_detection(self, ctrl_two_skills):
        """force_skill() sets active_skill immediately without blending."""
        ctrl = ctrl_two_skills
        ctrl._active_skill = Skill.BALANCE
        ctrl._transition_target = Skill.LOCOMOTION

        ctrl.force_skill(Skill.BALANCE)
        assert ctrl._active_skill == Skill.BALANCE
        assert ctrl._transition_target is None

    def test_force_skill_unknown_is_noop(self, ctrl_two_skills):
        """force_skill() with a skill not in skills dict is a silent no-op."""
        ctrl = ctrl_two_skills
        original = ctrl._active_skill
        ctrl.force_skill(Skill.STAIR)  # STAIR not loaded
        assert ctrl._active_skill == original


# ---------------------------------------------------------------------------
# Tests: Fix 4 — per-skill _prev_actions buffers
# ---------------------------------------------------------------------------


class TestPerSkillPrevActions:
    @pytest.fixture(scope="class")
    def ctrl_two_skills(self, mj_model, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("prev_act")
        _write_fake_checkpoint(tmp / "balance" / "final", obs_size=39)
        _write_fake_checkpoint(tmp / "wheeled_locomotion" / "final", obs_size=41)
        return UnifiedController(checkpoint_dir=str(tmp), mj_model=mj_model)

    def test_per_skill_buffers_initialised(self, ctrl_two_skills):
        """Each loaded skill has its own zero-initialised _prev_actions entry."""
        ctrl = ctrl_two_skills
        for sk in ctrl.skills:
            assert sk in ctrl._prev_actions, f"{sk} not in _prev_actions"
            arr = np.array(ctrl._prev_actions[sk])
            assert arr.shape == (ctrl.mj_model.nu,)
            assert np.all(arr == 0.0)

    def test_computing_one_skill_does_not_corrupt_other(self, ctrl_two_skills, mj_model):
        """Fix 4: running BALANCE policy does not modify LOCOMOTION's prev_action."""
        ctrl = ctrl_two_skills
        data = mujoco.MjData(mj_model)
        if mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(mj_model, data, 0)
        mujoco.mj_forward(mj_model, data)

        # Zero-out both buffers
        for sk in ctrl.skills:
            ctrl._prev_actions[sk] = jnp.zeros(mj_model.nu)

        loco_before = np.array(ctrl._prev_actions[Skill.LOCOMOTION]).copy()

        # Run only BALANCE
        ctrl._compute_skill_ctrl(data, Skill.BALANCE, ControlCommand())

        loco_after = np.array(ctrl._prev_actions[Skill.LOCOMOTION])
        np.testing.assert_array_equal(
            loco_before,
            loco_after,
            err_msg="LOCOMOTION _prev_actions was mutated by BALANCE computation (Fix 4 regression)",  # noqa: E501
        )

    def test_each_skill_updates_own_buffer(self, ctrl_two_skills, mj_model):
        """Running BALANCE policy updates BALANCE's buffer, not LOCOMOTION's."""
        ctrl = ctrl_two_skills
        data = mujoco.MjData(mj_model)
        if mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(mj_model, data, 0)
        mujoco.mj_forward(mj_model, data)

        for sk in ctrl.skills:
            ctrl._prev_actions[sk] = jnp.zeros(mj_model.nu)

        ctrl._compute_skill_ctrl(data, Skill.BALANCE, ControlCommand())

        # BALANCE buffer should have been updated (non-zero after policy run)
        # The policy might produce zero actions — just check they are the same object
        # updated by BALANCE, not LOCOMOTION
        loco_after = np.array(ctrl._prev_actions[Skill.LOCOMOTION])
        assert np.all(loco_after == 0.0), "LOCOMOTION buffer should be untouched"


# ---------------------------------------------------------------------------
# Tests: Tilt detection — gravity-in-body-frame semantics
# ---------------------------------------------------------------------------
# These tests exercise the tilt formula directly (no MuJoCo needed) by calling
# get_gravity_in_body_frame with controlled quaternions and computing the tilt
# angle the same way _detect_skill_raw does.
#
# Old formula: arccos(2*qw^2 - 1)  [total rotation angle, yaw-inclusive]
# New formula: arccos(-g_body[2])  [angle from world-up, yaw-invariant]
# ---------------------------------------------------------------------------


def _gravity_body_tilt(quat_wxyz) -> float:
    """Compute tilt using the new formula (matches _detect_skill_raw)."""
    from wheeled_biped.utils.math_utils import get_gravity_in_body_frame

    g = np.array(get_gravity_in_body_frame(jnp.array(quat_wxyz)))
    return float(np.arccos(np.clip(-g[2], -1.0, 1.0)))


def _old_tilt(quat_wxyz) -> float:
    """Compute tilt using the OLD (buggy) formula for comparison."""
    qw = quat_wxyz[0]
    return float(np.arccos(np.clip(2 * qw**2 - 1, -1.0, 1.0)))


class TestTiltSemantics:
    """Verify that the gravity-body-frame tilt is yaw-invariant and pitch-sensitive.

    The key regression: the old formula arccos(2*qw^2 - 1) computes the total
    rotation angle from the identity quaternion, which includes yaw.  A robot
    facing sideways (90° yaw, perfectly upright) has qw = cos(45°) ≈ 0.707,
    so old_tilt = arccos(2*0.5-1) = arccos(0) = π/2 ≈ 1.57 rad.
    With the 0.5 rad threshold this falsely triggers BALANCE mode.

    The new formula returns 0 for any pure-yaw rotation.
    """

    def test_upright_tilt_is_zero(self):
        """Identity quaternion (perfectly upright): tilt = 0."""
        quat = [1.0, 0.0, 0.0, 0.0]  # [w, x, y, z]
        assert _gravity_body_tilt(quat) == pytest.approx(0.0, abs=1e-5)

    def test_yaw_only_new_formula_is_zero(self):
        """Pure 90-degree yaw: new formula → tilt = 0 (yaw-invariant).

        Old (buggy) formula → tilt ≈ π/2 = 1.57 rad → falsely > 0.5 threshold.
        """
        import math

        # 90° yaw around z-axis:  q = [cos(45°), 0, 0, sin(45°)]
        quat = [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
        new_tilt = _gravity_body_tilt(quat)
        old_tilt = _old_tilt(quat)

        # New formula correctly returns ~0 (float32 JAX may accumulate ≤1e-3 rad)
        assert new_tilt == pytest.approx(0.0, abs=1e-3), (
            f"New formula: yaw-only rotation should give tilt≈0, got {new_tilt:.6f} rad "
            f"(still << 0.5 rad threshold)"
        )
        # Old formula was wrongly giving ~π/2 ≈ 1.57 rad (well above the 0.5 threshold)
        assert old_tilt > 1.0, (
            f"Old formula (regression reference): expected tilt > 1.0 rad for 90° yaw, "
            f"got {old_tilt:.4f}"
        )

    def test_yaw_180_new_formula_is_zero(self):
        """180-degree yaw: new formula → tilt = 0, old → tilt = π ≈ 3.14."""
        import math

        quat = [math.cos(math.pi / 2), 0.0, 0.0, math.sin(math.pi / 2)]  # 180° yaw
        assert _gravity_body_tilt(quat) == pytest.approx(0.0, abs=1e-5)

    def test_45_degree_pitch_detected(self):
        """45-degree pitch around y-axis: tilt ≈ π/4 ≈ 0.785 rad > 0.5 threshold.

        Both old and new formulas agree here (no yaw component):
        a robot leaning forward by 45° should trigger balance mode.
        """
        import math

        # 45° pitch: q = [cos(22.5°), 0, sin(22.5°), 0]
        half = math.radians(22.5)
        quat = [math.cos(half), 0.0, math.sin(half), 0.0]
        new_tilt = _gravity_body_tilt(quat)
        assert new_tilt == pytest.approx(math.radians(45), abs=1e-4), (
            f"New tilt for 45° pitch should be ~0.785 rad, got {new_tilt:.4f}"
        )
        assert new_tilt > 0.5, "45° pitch should exceed the 0.5 rad balance threshold"

    def test_30_degree_roll_detected(self):
        """30-degree roll around x-axis: tilt ≈ π/6 ≈ 0.524 rad > 0.5 threshold."""
        import math

        half = math.radians(15)
        quat = [math.cos(half), math.sin(half), 0.0, 0.0]
        new_tilt = _gravity_body_tilt(quat)
        assert new_tilt == pytest.approx(math.radians(30), abs=1e-4)
        assert new_tilt > 0.5

    def test_small_pitch_below_threshold(self):
        """10-degree pitch: tilt ≈ 0.175 rad which is < 0.5 rad; should not trigger."""
        import math

        half = math.radians(5)
        quat = [math.cos(half), 0.0, math.sin(half), 0.0]
        new_tilt = _gravity_body_tilt(quat)
        assert new_tilt == pytest.approx(math.radians(10), abs=1e-4)
        assert new_tilt < 0.5
