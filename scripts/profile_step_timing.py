"""Profile the force distributor to find the 128ms bottleneck."""
import sys, time, cProfile, pstats, io
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mujoco, numpy as np, jax.numpy as jnp

mj_model = mujoco.MjModel.from_xml_path("assets/robot/wheeled_biped_real.xml")
mj_data = mujoco.MjData(mj_model)
mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
mujoco.mj_forward(mj_model, mj_data)

from wheeled_biped.controllers.simple_force_distributor import SimpleForceDistributor

fd = SimpleForceDistributor(tau_hip_roll_max=15.0, max_force_asymmetry=60.0, min_wheel_force=20.0)

wrench = jnp.array([0.0, -4.2654, 79.46, 0.0, 0.0, 0.0])
wheel_left = jnp.array([0.05, -0.03, -0.48])
wheel_right = jnp.array([-0.05, -0.03, -0.48])

print("=== Force distributor timing (10 calls) ===")
for i in range(10):
    t0 = time.perf_counter()
    fl, fr, thr, diag = fd.distribute_wrench_contact_aware(
        wrench, left_contact=True, right_contact=True,
        wheel_pos_left=wheel_left, wheel_pos_right=wheel_right,
        hip_roll_authority_scale=1.0, distribution_mode="delta",
        max_delta_fz=30.0,
    )
    dt = (time.perf_counter() - t0) * 1000
    tag = "(first/warmup)" if i == 0 else ""
    print(f"  Call {i}: {dt:.3f} ms {tag} | {diag}")

# Now profile with cProfile
print("\n=== cProfile ===")
pr = cProfile.Profile()
pr.enable()
fl, fr, thr, diag = fd.distribute_wrench_contact_aware(
    wrench, left_contact=True, right_contact=True,
    wheel_pos_left=wheel_left, wheel_pos_right=wheel_right,
    hip_roll_authority_scale=1.0, distribution_mode="delta",
    max_delta_fz=30.0,
)
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
ps.print_stats(30)
print(s.getvalue())
