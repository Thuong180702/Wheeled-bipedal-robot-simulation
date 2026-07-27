#!/usr/bin/env python3
"""Run structural ablations by temporarily patching k2_jax_controller.py.

Ablations:
  1. symmetric_EMA: use same slow coefficient for attack & release
  2. global_I: remove proximity gate on integral application
"""
import sys, os, shutil, subprocess, json, importlib

CTRL_FILE = "wheeled_biped/controllers/k2_jax_controller.py"
BACKUP = CTRL_FILE + ".ablation_backup"

def patch_and_run(ablation_name, old_str, new_str, output_file):
    """Patch the controller, run sweep, restore."""
    # Backup
    shutil.copy2(CTRL_FILE, BACKUP)

    # Read and patch
    with open(CTRL_FILE, 'r') as f:
        content = f.read()

    if old_str not in content:
        print(f"ERROR: old_str not found for {ablation_name}")
        shutil.copy2(BACKUP, CTRL_FILE)
        os.remove(BACKUP)
        return None

    content = content.replace(old_str, new_str)
    with open(CTRL_FILE, 'w') as f:
        f.write(content)

    # Clear Python caches
    for root, dirs, files in os.walk('wheeled_biped'):
        if '__pycache__' in dirs:
            cache_dir = os.path.join(root, '__pycache__')
            for f in os.listdir(cache_dir):
                if 'k2_jax_controller' in f:
                    os.remove(os.path.join(cache_dir, f))
                    print(f"  Removed cache: {os.path.join(cache_dir, f)}")

    # Run sweep (fresh process to pick up changes)
    print(f"\nRunning {ablation_name} sweep...")
    result = subprocess.run(
        [sys.executable, 'scripts/push_sweep_paper.py',
         '--quick', '--output', output_file],
        capture_output=True, text=True, timeout=1200,
    )
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-300:])

    # Restore original
    shutil.copy2(BACKUP, CTRL_FILE)
    os.remove(BACKUP)

    # Read results
    if os.path.exists(output_file):
        with open(output_file) as f:
            data = json.load(f)
        print(f"\n{ablation_name}: F_min={data['F_min_N']:.0f}N "
              f"F_med={data['F_med_N']:.0f}N F_max={data['F_max_N']:.0f}N")
        return data
    return None


if __name__ == '__main__':
    print("="*60)
    print("Structural Ablation Runner")
    print("="*60)

    # 1. Symmetric EMA
    sym_ema_old = "jnp.where(_act_dev > 0.0, 0.35, 0.0067) * _act_dev"
    sym_ema_new = "0.0067 * _act_dev"  # slow symmetric EMA
    r1 = patch_and_run("symmetric_EMA", sym_ema_old, sym_ema_new,
                       "outputs/push_sweep_sym_ema.json")

    # 2. Global I (no proximity gate)
    global_i_old = "_anchor_integ_applied = _anchor_integ * _anchor_prox"
    global_i_new = "_anchor_integ_applied = _anchor_integ  # global I (no prox gate)"
    r2 = patch_and_run("global_I", global_i_old, global_i_new,
                       "outputs/push_sweep_global_i.json")

    print("\n" + "="*60)
    print("RESULTS:")
    if r1:
        print(f"  symmetric_EMA: F_min={r1['F_min_N']:.0f}N F_med={r1['F_med_N']:.0f}N")
    if r2:
        print(f"  global_I:      F_min={r2['F_min_N']:.0f}N F_med={r2['F_med_N']:.0f}N")
    print("="*60)
