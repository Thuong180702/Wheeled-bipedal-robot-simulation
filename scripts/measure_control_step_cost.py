#!/usr/bin/env python3
"""Wall-clock cost of one ACC control step on the development host.

The hardware-budget paragraph of the Discussion assembles its ~9.5 ms
end-to-end estimate from assumed component figures, of which the ~3 ms
"ACC computation" term is the largest. This measures that term the same way
the WBC baseline's QP solve time is measured (Section: WBC baselines), so the
shipped controller and the discarded one are reported on the same footing.

Two quantities are separated:

  step_total  the full per-control-step call the simulation makes,
              compute_v3_torque_for_state(), which includes the CoM /
              support-polygon computation the controller does for itself
  jax_only    the compiled torque assembly alone, i.e. the part that would be
              ported to an embedded target

Caveats, stated in the paper alongside the number: this is a JAX/XLA CPU build
on a development laptop, not an embedded target, and JAX would not be the
deployment runtime. It bounds the assumed 3 ms rather than replacing it with an
embedded figure.

Writes outputs/paper_verification/control_step_cost.json
"""
import json
import platform
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.promote_v3_vs_assist as P  # noqa: E402
from wheeled_biped.controllers.k2_jax_controller import pack_state_k2  # noqa: E402
from wheeled_biped.wbc.offline_three_arm_counterfactual import (  # noqa: E402
    compute_v3_torque_for_state,
    init_v3_controller,
)

PROFILE = "K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR"
DV = "archive/cleanup_2026-06-13/output_summaries/balance_core_true_height_variants"
WARMUP = 200
N = 5000
SUBSTEPS = 5


def percentiles(samples_s):
    a = np.asarray(samples_s) * 1e3  # ms
    return {
        "mean_ms": float(a.mean()),
        "median_ms": float(np.median(a)),
        "p99_ms": float(np.percentile(a, 99)),
        "max_ms": float(a.max()),
        "min_ms": float(a.min()),
        "n": int(a.size),
    }


def main():
    nom = json.load(open(ROOT / DV / "variant_nominal__variant_setup.json"))
    posture = np.array([
        nom["hip_roll_left"], nom["hip_yaw_left"], nom["hip_pitch_ref"],
        nom["knee_ref"], 0.0, nom["hip_roll_right"], nom["hip_yaw_right"],
        nom["hip_pitch_ref"], nom["knee_ref"], 0.0,
    ])

    model = mujoco.MjModel.from_xml_path(str(P.get_model_path()))
    v3 = dict(init_v3_controller(profile_name=PROFILE, model=model))
    v3["jax_state"] = pack_state_k2()
    data = mujoco.MjData(model)
    data.qpos[7:17] = posture
    data.qpos[2] = float(nom["calibrated_root_z_m"])
    mujoco.mj_forward(model, data)
    ctx = P._build_v3_controller_context(
        model, data, v3, eq_joint=posture, height_ref=float(nom["target_com_z_m"])
    )

    def one_step(measure):
        """Advance one control step; return (total_s, jax_s) if measuring."""
        t0 = time.perf_counter() if measure else 0.0
        r = compute_v3_torque_for_state(
            data, model, v3["jax_step_fn"], v3["jax_state"], v3["jax_params"],
            ctx, teleop=None,
        )
        # force the async XLA dispatch to complete before stopping the clock
        tau = np.asarray(r["tau_v3"])
        nxt = np.asarray(r["next_jax_state"])
        t1 = time.perf_counter() if measure else 0.0
        v3["jax_state"] = r["next_jax_state"]
        data.ctrl[:] = tau
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)
        return (t1 - t0, nxt) if measure else (0.0, nxt)

    for _ in range(WARMUP):
        one_step(False)

    # settle so the timing runs at the quiet-stance operating point, i.e. the
    # branch profile the deployed loop actually executes
    for _ in range(int(20.0 / 0.01)):
        one_step(False)

    total = []
    for _ in range(N):
        dt, _ = one_step(True)
        total.append(dt)

    # jax-only: same compiled step, called directly on the packed args
    args = None

    def capture(*a, **k):
        nonlocal args
        args = (a, k)
        return orig(*a, **k)

    orig = v3["jax_step_fn"]
    v3["jax_step_fn"] = capture
    one_step(False)
    v3["jax_step_fn"] = orig

    jax_only = []
    kernel_cost = None
    if args is not None:
        a, k = args
        for _ in range(N):
            t0 = time.perf_counter()
            out = orig(*a, **k)
            np.asarray(out[0] if isinstance(out, tuple) else out)
            jax_only.append(time.perf_counter() - t0)

        # Host-independent cost of the same kernel. Wall-clock on a laptop does
        # not answer "will this run on an embedded target"; an operation count
        # does, because the step has no data-dependent control flow (every gate
        # is a branchless select), so worst case equals average case.
        ca = orig.lower(*a, **k).compile().cost_analysis()
        if isinstance(ca, list):
            ca = ca[0]
        kernel_cost = {
            "flops": float(ca.get("flops", 0.0)),
            "transcendentals": float(ca.get("transcendentals", 0.0)),
            "bytes_accessed": float(ca.get("bytes accessed", 0.0)),
            "working_set_floats": int(sum(x.size for x in a)),
        }

    out = {
        "profile": PROFILE,
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "python": platform.python_version(),
        },
        "control_period_ms": 10.0,
        "step_total": percentiles(total),
        "jax_only": percentiles(jax_only) if jax_only else None,
        "kernel_cost": kernel_cost,
        "note": (
            "JAX/XLA CPU build on a development host, not an embedded target. "
            "step_total is the full per-control-step call including the CoM and "
            "support-polygon computation; jax_only is the compiled torque "
            "assembly alone."
        ),
    }
    dest = ROOT / "outputs/paper_verification/control_step_cost.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, dest.open("w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nSaved {dest}")


if __name__ == "__main__":
    main()
