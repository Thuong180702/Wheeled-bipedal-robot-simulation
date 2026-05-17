import numpy as np

from scripts.phase_b9_step4_slow_loop_gating import (
    apply_slow_loop_correction,
    freeze_controller_posture,
    to_jsonable_or_yamlable,
)


class DummyController:
    def __init__(self):
        self.target_hip_pitch = 0.3
        self.target_knee = 0.5
        self.last_stable_hip_pitch = 0.3
        self.last_stable_knee = 0.5
        self.slow_loop_interval = 10


def test_freeze_controller_posture_matches_step3_behavior():
    controller = DummyController()
    init_table = {
        "0.40": {
            "hip_pitch": 0.12,
            "knee": 0.84,
        }
    }

    freeze_controller_posture(controller, height=0.40, init_table=init_table, disable_slow_loop=True)

    assert controller.target_hip_pitch == 0.12
    assert controller.target_knee == 0.84
    assert controller.last_stable_hip_pitch == 0.12
    assert controller.last_stable_knee == 0.84
    assert controller.slow_loop_interval == 999999


def test_apply_slow_loop_correction_blocks_when_gate_active():
    raw = {"hip_pitch_delta": 0.03, "knee_delta": -0.02}

    blocked = apply_slow_loop_correction(raw, gate_active=True, slow_loop_scale=0.0)
    assert blocked["applied_slow_correction"]["hip_pitch_delta"] == 0.0
    assert blocked["applied_slow_correction"]["knee_delta"] == 0.0

    scaled = apply_slow_loop_correction(raw, gate_active=True, slow_loop_scale=0.25)
    assert scaled["applied_slow_correction"]["hip_pitch_delta"] == 0.03 * 0.25
    assert scaled["applied_slow_correction"]["knee_delta"] == -0.02 * 0.25


def test_apply_slow_loop_correction_keeps_raw_when_gate_inactive():
    raw = {"hip_pitch_delta": 0.03, "knee_delta": -0.02}
    result = apply_slow_loop_correction(raw, gate_active=False, slow_loop_scale=0.0)

    assert result["applied_slow_correction"] == raw


def test_to_jsonable_or_yamlable_removes_numpy_types():
    data = {
        "a": np.float64(1.5),
        "b": np.int64(2),
        "c": np.bool_(True),
        "d": [np.float32(3.0), {"e": np.int32(4)}],
    }

    converted = to_jsonable_or_yamlable(data)

    assert isinstance(converted["a"], float)
    assert isinstance(converted["b"], int)
    assert isinstance(converted["c"], bool)
    assert isinstance(converted["d"][0], float)
    assert isinstance(converted["d"][1]["e"], int)
