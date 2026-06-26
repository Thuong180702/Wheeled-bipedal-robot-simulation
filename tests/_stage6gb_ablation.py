"""Stage 6G-B: State ablation — compare Python vs JAX state after step 0."""
import subprocess, sys, ast, os, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SIM = str(ROOT / "scripts" / "simulate_hierarchical_controller.py")
SETUP = str(ROOT / "outputs" / "physical_target_height_setups_centered" / "low_0p320_setup.json")

with open(SIM) as f:
    content = f.read()

# Add comprehensive state dump after step 0
dump = '''
                if step == 0:
                    # --- JAX state (25 fields) ---
                    _js = [float(_jax_state[i]) for i in range(25)]
                    print("JAX_STATE=" + json.dumps(_js), flush=True)

                    # --- Python equivalent state ---
                    _nf = sagittal._wip_notch_pitch_rate
                    _py_notch = [_nf._x1, _nf._x2, _nf._y1, _nf._y2] if _nf else [0,0,0,0]

                    # Python tau_prev from sim loop
                    _py_tprev = [float(tau_prev[i]) for i in range(10)]

                    # Python outer loop state from sim loop
                    _py_ol_ref = float(outer_loop_pitch_ref_smoothed_deg)
                    _py_ol_pe = float(outer_loop_prev_support_error_m) if outer_loop_prev_support_error_m is not None else 0.0
                    _py_ol_rate = float(outer_loop_support_error_rate_smoothed)

                    # Python adaptive bias trim state
                    _py_abs = [
                        sagittal._adaptive_bias_trim_tau,
                        sagittal._adaptive_bias_hold_steps,
                        sagittal._adaptive_bias_prev_error_sign,
                        0.0,  # zc_count not directly available
                        0.0,  # slow_ema (not available as scalar)
                        0.0,  # fast_ema
                    ]

                    _py_fcz = float(sagittal._filtered_com_z)
                    _py_perr = float(prev_support_error)

                    _py_full = {
                        "notch": _py_notch,
                        "tau_prev": _py_tprev,
                        "filtered_com_z": _py_fcz,
                        "prev_support_error": _py_perr,
                        "ol_pitch_ref": _py_ol_ref,
                        "ol_prev_error": _py_ol_pe,
                        "ol_rate": _py_ol_rate,
                        "abs_trim_tau": _py_abs[0],
                        "abs_hold_steps": _py_abs[1],
                        "abs_prev_sign": _py_abs[2],
                    }
                    print("PY_FULL=" + json.dumps(_py_full), flush=True)

                    # Torque comparison
                    _pt = [float(balance_core_result.tau_final[i]) for i in range(10)]
                    _jt = [float(_jax_tau[i]) for i in range(10)]
                    print("TAU_PY=" + json.dumps(_pt), flush=True)
                    print("TAU_JX=" + json.dumps(_jt), flush=True)

                if step <= 2:
                    _pt = [float(balance_core_result.tau_final[i]) for i in range(10)]
                    _jt = [float(_jax_tau[i]) for i in range(10)]
                    _d = max(abs(_pt[i]-_jt[i]) for i in range(10))
                    print(f"TAU_DIFF_S{step}={_d:.6e}", flush=True)
'''

old = '_jax_tau, _jax_state, _jax_diag = _jax_step_fn(_jax_state, _jax_input, _jax_params)'
if old in content:
    content = content.replace(old, old + dump)
    tmp = str(ROOT / "scripts" / "_sim6gb.py")
    with open(tmp, "w") as f:
        f.write(content)

    cmd = [
        sys.executable, tmp,
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", "k2_notch_low_q_v1",
        "--controller-backend", "jax",
        "--height-variant-setup", SETUP,
        "--steps", "4",
        "--wbc-quiet",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    print("=== STATE COMPARISON AFTER STEP 0 ===")
    jax_state = None
    py_full = None
    tau_diffs = {}
    for line in r.stdout.split("\n"):
        if line.startswith("JAX_STATE="):
            jax_state = json.loads(line.split("=", 1)[1])
        elif line.startswith("PY_FULL="):
            py_full = json.loads(line.split("=", 1)[1])
        elif line.startswith("TAU_DIFF_S"):
            parts = line.split("=")
            tau_diffs[parts[0]] = float(parts[1])

    if jax_state and py_full:
        print(f"\nStep 0 tau diff: {tau_diffs.get('TAU_DIFF_S0', 'N/A')}")
        print(f"Step 1 tau diff: {tau_diffs.get('TAU_DIFF_S1', 'N/A')}")
        print(f"Step 2 tau diff: {tau_diffs.get('TAU_DIFF_S2', 'N/A')}")

        print(f"\nState field comparison:")

        # Notch
        jx_notch = jax_state[0:4]
        py_notch = py_full["notch"]
        print(f"  notch:         PY={[f'{v:.6f}' for v in py_notch]}  JX={[f'{v:.6f}' for v in jx_notch]}")

        # tau_prev
        jx_tprev = jax_state[4:14]
        py_tprev = py_full["tau_prev"]
        print(f"  tau_prev[4,9]: PY=[{py_tprev[4]:.6f},{py_tprev[9]:.6f}]  JX=[{jx_tprev[4]:.6f},{jx_tprev[9]:.6f}]")

        # filtered_com_z
        print(f"  filt_com_z:    PY={py_full['filtered_com_z']:.6f}  JX={jax_state[14]:.6f}")

        # Outer loop
        print(f"  ol_ref:        PY={py_full['ol_pitch_ref']:.6f}  JX={jax_state[16]:.6f}")
        print(f"  ol_prev_e:     PY={py_full['ol_prev_error']:.6f}  JX={jax_state[17]:.6f}")
        print(f"  ol_rate:       PY={py_full['ol_rate']:.6f}  JX={jax_state[18]:.6f}")

        # Adaptive bias
        print(f"  abs_trim:      PY={py_full['abs_trim_tau']:.6f}  JX={jax_state[21]:.6f}")
        print(f"  abs_hold:      PY={py_full['abs_hold_steps']:.6f}  JX={jax_state[22]:.6f}")
        print(f"  abs_sign:      PY={py_full['abs_prev_sign']:.6f}  JX={jax_state[23]:.6f}")

        # prev_support_error
        print(f"  prev_perr:     PY={py_full['prev_support_error']:.6f}  JX={jax_state[15]:.6f}")

    os.remove(tmp)
else:
    print("Pattern not found")
