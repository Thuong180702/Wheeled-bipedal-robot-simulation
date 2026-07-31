#!/usr/bin/env python3
"""Unified idle accuracy+repeatability across the FULL ablation ladder.

Same protocol for every row (100 Hz ctrl / 500 Hz phys, 25 s, 5 s settle,
20 s window, seeds 20260727+i) so the paper's 'Idle RMS' column stops mixing
three incompatible metrics.

Patch recipes copied verbatim from scripts/replicate_ablation_n10.py::CONFIGS, so
the idle table and the push table describe the same thirteen controllers.

Writes outputs/paper_verification/idle_ladder.json (Table III of the paper).
Usage:  .venv/bin/python scripts/collect_idle_ladder.py [N] [config ...]
"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CTRL = ROOT / "wheeled_biped/controllers/k2_jax_controller.py"
BACKUP = str(CTRL) + ".idle_backup"
WORKER = Path(__file__).with_name("_idle_ladder_worker.py")
PY = str(ROOT / ".venv/bin/python")
ANCHOR = "K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR"

P_PROX = ("_anchor_integ_applied = _anchor_integ * _anchor_prox",
          "_anchor_integ_applied = _anchor_integ  # no prox gate")
P_ENV = ('_anchor_quiet = 1.0 - _jax_smoothstep01((_act_ema - 0.25) / (0.50 - 0.25))',
         '_anchor_quiet = 1.0  # no envelope gate')

CONFIGS = [
    ("L0_P_only",  "K2_JAX_DEDICATED_DEFAULT_V3",        [], {}),
    ("L1_homing",  "K2_JAX_DEDICATED_DEFAULT_V3_HOMING", [], {}),
    ("L2_global_I", ANCHOR, [P_PROX, P_ENV], {32: 1.0, 54: 6.0, 91: 0.0, 94: 50.0}),
    ("L2_5_prox_only", ANCHOR, [P_ENV],      {32: 1.0, 54: 6.0, 91: 0.0, 94: 50.0}),
    ("S0_sched_kp", ANCHOR, [], {}),
    ("S1_L3_full",  ANCHOR, [], {94: 50.0}),
    ("S2_no_boost", ANCHOR, [], {91: 0.0}),
    ("S3_sym_ema",  ANCHOR,
     [("jnp.where(_act_dev > 0.0, 0.35, 0.0067) * _act_dev",
       "0.0067 * _act_dev  # symmetric EMA")], {}),
    ("S4_no_prox",  ANCHOR, [P_PROX], {}),
    ("S5_no_pitch_gate", ANCHOR,
     [("* _anchor_stab)", "* 1.0)  # pitch gate removed")], {}),
    # Mechanism isolation: M1 turns the anchor integral off (idx 88), M2 reverts
    # the two velocity-damping coefficients to their L2 values, M3 does both.
    ("M1_no_integral",   ANCHOR, [], {88: 0.0}),
    ("M2_L2_damping",    ANCHOR, [], {32: 1.0, 54: 6.0, 91: 0.0}),
    ("M3_no_I_no_boost", ANCHOR, [], {88: 0.0, 91: 0.0}),
]


def clear_cache():
    for root, dirs, _ in os.walk(str(ROOT / "wheeled_biped")):
        if "__pycache__" in dirs:
            d = os.path.join(root, "__pycache__")
            for f in os.listdir(d):
                if any(k in f for k in ("k2_jax_controller", "offline_three_arm",
                                        "sagittal_velocity", "centroidal")):
                    os.remove(os.path.join(d, f))


def apply(patches):
    shutil.copy2(str(CTRL), BACKUP)
    txt = CTRL.read_text()
    for old, new in patches:
        if old not in txt:
            restore()
            raise SystemExit(f"PATCH NOT FOUND: {old[:90]}")
        txt = txt.replace(old, new)
    CTRL.write_text(txt)
    clear_cache()


def restore():
    if os.path.exists(BACKUP):
        shutil.copy2(BACKUP, str(CTRL))
        os.remove(BACKUP)
        clear_cache()


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    only = sys.argv[2:] if len(sys.argv) > 2 else None
    dest = ROOT / "outputs/paper_verification/idle_ladder.json"
    out = json.load(dest.open()) if dest.exists() else {
        "protocol": {"control_hz": 100, "physics_hz": 500, "total_s": 25.0,
                     "settle_s": 5.0, "window_s": 20.0, "base_seed": 20260727,
                     "ic_joint_sigma_rad": 0.005, "ic_root_z_sigma_m": 0.001,
                     "accuracy": "RMS of horizontal displacement from commanded home",
                     "repeatability": "RMS about the per-trial mean (DC offset removed)"},
        "results": {}}
    for name, prof, patches, ov in CONFIGS:
        if only and name not in only:
            continue
        print(f"\n=== {name} ({prof}) patches={len(patches)} ov={ov} ===", flush=True)
        t0 = time.time()
        try:
            if patches:
                apply(patches)
            p = subprocess.run([PY, str(WORKER), prof, json.dumps(ov), str(N)],
                               cwd=str(ROOT), capture_output=True, text=True)
        finally:
            restore()
        print(p.stderr.strip(), flush=True)
        line = [l for l in p.stdout.splitlines() if l.startswith("@@JSON@@")]
        if not line:
            print("  FAILED\n" + p.stdout[-2000:], flush=True)
            continue
        r = json.loads(line[0][8:])
        r["profile"] = prof
        r["patches"] = [o for o, _ in patches]
        r["param_overrides"] = ov
        out["results"][name] = r
        if r["n_survived"]:
            print(f"  AGG acc_x={r['base_acc_x_rms_mm']['mean']:.3f}+-{r['base_acc_x_rms_mm']['std']:.3f}  "
                  f"rep_x={r['base_rep_x_rms_mm']['mean']:.3f}+-{r['base_rep_x_rms_mm']['std']:.3f}  "
                  f"acc_xy={r['base_acc_xy_rms_mm']['mean']:.2f}  rep_xy={r['base_rep_xy_rms_mm']['mean']:.3f}  "
                  f"off=({r['base_off_x_mm']['mean']:.1f},{r['base_off_y_mm']['mean']:.1f})  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        json.dump(out, dest.open("w"), indent=2)
    print(f"\nSaved {dest}")


if __name__ == "__main__":
    main()
