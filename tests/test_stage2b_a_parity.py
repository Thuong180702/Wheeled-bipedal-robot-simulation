import numpy as np

from scripts.debug_stage2b_a_parity import classify_parity_root_cause


def _row(tau_total_raw, tau_final, com_z, roll_y):
    return {
        "tau_total_raw_2_3_7_8": np.array(tau_total_raw, dtype=float),
        "tau_final_2_3_7_8": np.array(tau_final, dtype=float),
        "com_z": float(com_z),
        "roll_y": float(roll_y),
    }


def test_classify_controller_computation_mismatch_when_tau_total_raw_differs():
    standalone = [_row([1, 2, 3, 4], [1, 2, 3, 4], 0.40, 0.01)]
    main = [_row([1, 2, 3, 5], [1, 2, 3, 5], 0.40, 0.01)]

    result = classify_parity_root_cause(standalone, main)

    assert result == "controller_computation_mismatch"


def test_classify_clipping_or_rate_limit_mismatch_when_tau_raw_same_but_final_differs():
    standalone = [_row([1, 2, 3, 4], [1, 2, 3, 4], 0.40, 0.01)]
    main = [_row([1, 2, 3, 4], [1, 1, 3, 4], 0.40, 0.01)]

    result = classify_parity_root_cause(standalone, main)

    assert result == "actuator_pipeline_mismatch"


def test_classify_initialization_or_contact_mismatch_when_torque_same_but_state_diverges():
    standalone = [_row([1, 2, 3, 4], [1, 2, 3, 4], 0.404, 0.00)]
    main = [_row([1, 2, 3, 4], [1, 2, 3, 4], 0.392, 0.20)]

    result = classify_parity_root_cause(standalone, main)

    assert result == "initialization_contact_timestep_model_mismatch"


def test_classify_equivalent_when_torque_and_state_match_within_tolerance():
    standalone = [_row([1, 2, 3, 4], [1, 2, 3, 4], 0.4040, 0.0200)]
    main = [_row([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], 0.4041, 0.0201)]

    result = classify_parity_root_cause(standalone, main)

    assert result == "equivalent_paths"
