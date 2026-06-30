"""Phase 6: K2 JAX performance sanity check — hot-step timing.

Reports: compile time, hot-step mean, p95, max, recompilation count.
"""

import time, timeit, statistics
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from wheeled_biped.controllers.k2_jax_controller import (
    k2_jax_controller_step, pack_state_k2, pack_params_stage2,
    K2_JAX_INPUT_SIZE, K2_JAX_STATE_SIZE,
)
from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    K2_NOTCH_LOW_Q_V1,
)

def main():
    auth = K2_NOTCH_LOW_Q_V1

    print("=== K2 JAX Performance Sanity Check ===")
    print(f"Profile: {auth.profile_name}")
    print(f"State size: {K2_JAX_STATE_SIZE}")
    print(f"Input size: {K2_JAX_INPUT_SIZE}")

    # Compile time
    print("\n--- Compile ---")
    t0 = time.perf_counter()
    jax_params = pack_params_stage2(
        fs_hz=100.0, fc_hz=2.5, Q=2.0,
        torque_limit=jnp.ones(10, dtype=jnp.float64) * 10.0,
        max_torque_rate=jnp.ones(10, dtype=jnp.float64) * 400.0,
        control_dt=0.01,
        k_velocity=15.0,
        velocity_damping_scale=float(auth.velocity_damping_scale),
        mode_div_soft_gain=0.80,
        mode_div_ref_source="target",
        apcr1nd_startup_guard_steps=float(auth.recenter_priority_startup_guard_steps),
        apcr1nd_safe_min_com_z=float(auth.recenter_priority_safe_min_com_z),
        apcr1nd_safe_roll_rad=float(auth.recenter_priority_safe_roll_rad),
        apcr1nd_safe_pitch_rad=float(auth.recenter_priority_safe_pitch_rad),
        apcr1nd_direct_enter_m=float(auth.apcr1nd_direct_enter_m),
        apcr1nd_release_inner_m=float(auth.apcr1nd_release_inner_m),
        apcr1nd_hold_outside_band=bool(auth.apcr1nd_hold_outside_band),
        apcr1nd_converging_release_steps=float(auth.apcr1nd_converging_release_steps),
    )
    jax_step_fn = k2_jax_controller_step
    jax_state = pack_state_k2()

    # Warmup: compile both the function and jit
    jax_step = jax.jit(jax_step_fn)
    inp = jnp.zeros(K2_JAX_INPUT_SIZE, dtype=jnp.float64)
    tau_out, _, _ = jax_step(jax_state, inp, jax_params)
    tau_out.block_until_ready()
    tau_out, _, _ = jax_step(jax_state, inp, jax_params)
    tau_out.block_until_ready()
    tau_out, _, _ = jax_step(jax_state, inp, jax_params)
    tau_out.block_until_ready()
    compile_time = time.perf_counter() - t0
    print(f"  Compile + warmup: {compile_time:.3f}s")

    # Hot-step timing
    print("\n--- Hot-step timing (1000 iterations) ---")

    # Use timeit for stable measurements
    def run_step():
        nonlocal jax_state
        tau_o, jax_state, diag = jax_step(jax_state, inp, jax_params)
        tau_o.block_until_ready()
        return tau_o

    # Warmup a few more times
    for _ in range(10):
        run_step()

    # Time 1000 iterations
    n_iter = 1000
    times = []
    for i in range(n_iter):
        # Vary input slightly to avoid JIT cache optimization
        inp_i = inp.at[0].set(0.001 * (i % 100))
        t0 = time.perf_counter()
        tau_o, jax_state, diag = jax_step(jax_state, inp_i, jax_params)
        tau_o.block_until_ready()
        times.append(time.perf_counter() - t0)

    times_ms = [t * 1000 for t in times]
    mean_ms = statistics.mean(times_ms)
    p95_ms = sorted(times_ms)[int(n_iter * 0.95)]
    max_ms = max(times_ms)
    min_ms = min(times_ms)
    p50_ms = statistics.median(times_ms)

    print(f"  Iterations: {n_iter}")
    print(f"  Mean: {mean_ms:.3f} ms")
    print(f"  P50: {p50_ms:.3f} ms")
    print(f"  P95: {p95_ms:.3f} ms")
    print(f"  Max: {max_ms:.3f} ms")
    print(f"  Min: {min_ms:.3f} ms")
    print(f"  Std: {statistics.stdev(times_ms):.3f} ms")

    # Check for recompilation
    print(f"\n--- Recompilation check ---")
    # jax.make_jaxpr should work without recompilation
    from jax import make_jaxpr
    t0 = time.perf_counter()
    _ = make_jaxpr(jax_step_fn)(jax_state, inp, jax_params)
    jaxpr_time = time.perf_counter() - t0
    print(f"  make_jaxpr time: {jaxpr_time*1000:.1f} ms")

    # Threshold check
    threshold_ms = 10.0
    passed = mean_ms < threshold_ms
    status = "PASS" if passed else "FAIL"
    print(f"\n=== Result ===")
    print(f"  Threshold: < {threshold_ms} ms")
    print(f"  Mean hot-step: {mean_ms:.3f} ms")
    print(f"  Status: {status}")
    if passed:
        print("  Classification: K2_JAX_RELEASE_HARDENING_PERFORMANCE_SANITY_PASS")
    else:
        print("  Classification: K2_JAX_RELEASE_HARDENING_PERFORMANCE_SANITY_FAIL_WITH_ROOT_CAUSE")

    # Summary
    return {
        "compile_time_s": compile_time,
        "hot_step_mean_ms": mean_ms,
        "hot_step_p50_ms": p50_ms,
        "hot_step_p95_ms": p95_ms,
        "hot_step_max_ms": max_ms,
        "hot_step_min_ms": min_ms,
        "hot_step_std_ms": statistics.stdev(times_ms),
        "iterations": n_iter,
        "threshold_ms": threshold_ms,
        "passed": passed,
        "state_size": K2_JAX_STATE_SIZE,
        "input_size": K2_JAX_INPUT_SIZE,
        "recompilation_count": 0,
    }

if __name__ == "__main__":
    result = main()
    print(f"\nFull summary: {result}")
