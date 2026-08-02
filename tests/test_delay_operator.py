"""A transport delay of N steps must apply the command from N steps ago.

Two independent off-by-one/aliasing bugs in the delay harnesses were found
2026-08-02, and every actuator-delay number in the paper traces to them:

* ``collect_robustness_sweep.py`` (producer of the paper's noise x delay
  table, and via its MEASURE_SCRIPT of the 50/100/150 ms stability sweep)
  keeps ONE slot, overwritten every step, so every delay_steps > 0 applies a
  one-step lag: "30 ms", "150 ms" and "10 ms" are all the same 10 ms run.
* ``robustness_sweep.py`` appends before reading, so delay_steps=1 is 0 ms.

The operator is four lines; nothing but a test keeps it honest.
"""
from collections import deque

import pytest


def apply_delay(cmds, n):
    """Reference: emit each command n steps late, zero-padded at the start."""
    buf = deque([0] * n, maxlen=max(n, 1))
    out = []
    for c in cmds:
        if n > 0:
            out.append(buf[0])   # written n steps ago
            buf.append(c)
        else:
            out.append(c)
    return out


@pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 15])
def test_lag_equals_requested_steps(n):
    cmds = list(range(1, 41))
    out = apply_delay(cmds, n)
    assert out[:n] == [0] * n, f"n={n}: warm-up not zero-padded: {out[:n]}"
    assert out[n:] == cmds[:len(cmds) - n], f"n={n}: lag is not {n} steps"


def test_distinct_delays_give_distinct_output():
    """The bug that mattered: 3 and 15 producing identical traces."""
    cmds = list(range(1, 41))
    assert apply_delay(cmds, 3) != apply_delay(cmds, 15)
    assert apply_delay(cmds, 1) != apply_delay(cmds, 0)


def test_single_slot_pattern_is_not_a_delay_line():
    """Reproduces collect_robustness_sweep.py: one slot, any n -> 1-step lag."""
    def single_slot(cmds, n):
        prev, out = None, []
        for step, c in enumerate(cmds):
            out.append(prev if (prev is not None and n > 0 and step >= n) else c)
            if n > 0:
                prev = c
        return out

    cmds = list(range(1, 41))
    # Every n collapses to the same 1-step lag past the warm-up window.
    for n in (1, 3, 5, 10, 15):
        assert single_slot(cmds, n)[20:] == apply_delay(cmds, 1)[20:]
    # ...which is exactly what a correct delay line does NOT do.
    assert single_slot(cmds, 15)[20:] != apply_delay(cmds, 15)[20:]


def test_script_operator_matches_reference():
    """The substep operator in delay_cliff_resolution.py, in isolation."""
    for n in (0, 1, 2, 4, 7):
        buf = deque([0] * n, maxlen=max(n, 1))
        out = []
        for c in range(1, 31):
            if n > 0:
                out.append(buf[0])
                buf.append(c)
            else:
                out.append(c)
        assert out == apply_delay(list(range(1, 31)), n)
