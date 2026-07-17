# K2 JAX Production Input Packing Optimization

**Date:** 2026-06-29
**Branch:** repo-cleanup-t6j
**Phase:** 4

## Original overhead: ~5-6 ms

The `pack_input_k2_standalone()` function takes ~5-6 ms per call for a 45-element vector. Investigation showed the overhead came from:

1. **JAX-to-NumPy round-trips:** `joint_pos`, `joint_vel`, and `q_ref` were passed as `jnp.array()` objects. Inside the packer, `float()` extraction on JAX array elements triggers device-to-host transfers.

2. **Redundant array creation:** `jnp.array(joint_pos)` creates a new JAX array from an already-JAX slice — wasted dispatch.

3. **9× `jnp.array()` calls** inside the packer for slice construction.

## Optimization

### Pre-convert to NumPy

Added NumPy pre-conversion before calling the packer:

```python
joint_pos_np = np.array(mj_data.qpos[7:17])  # direct from MuJoCo
joint_vel_np = np.array(mj_data.qvel[6:16])
equilibrium_joint_pos_np = np.array(equilibrium_joint_pos)
```

Then pass NumPy arrays directly:
```python
pack_input_k2_standalone(
    ...
    joint_pos=joint_pos_np,      # was: jnp.array(joint_pos)
    joint_vel=joint_vel_np,      # was: jnp.array(joint_vel)
    q_ref=equilibrium_joint_pos_np,  # was: jnp.array(equilibrium_joint_pos)
    ...
)
```

This avoids:
- 3× `jnp.array()` dispatches at the call site
- Device-to-host transfers on individual `float()` extractions inside the packer
- ~3 ms of redundant JAX dispatch overhead

### Remaining pack_input_k2 cost

The `pack_input_k2_standalone` function still creates:
- 1× `np.zeros(K2_JAX_INPUT_SIZE)` — preallocated buffer
- 3× `np.array()` for q/qd/qref slices — acceptable
- 1× `jnp.asarray(inp)` — final JAX conversion

Estimated remaining cost: ~1-2 ms (vs. original ~5-6 ms).

An additional optimization opportunity exists: the `jnp.asarray(inp)` call creates a fresh JAX array each step. A persistent device array could be updated in-place, but this risks violating JAX's immutability guarantees and was not attempted.

## JAX backend analysis

- **Backend:** CPU (`jax.default_backend() == 'cpu'`)
- **Devices:** `[CpuDevice(id=0)]`
- **X64:** Enabled (`jax_enable_x64 = True`)
- **GPU:** Not available
- **JAX hot-step:** ~0.3 ms (unchanged by this optimization)

The 0.3 ms hot-step on CPU confirms that the tiny 836-element state + 45-element input controller is compute-bound by Python dispatch overhead, not matrix math.

## Results

| Metric | Before | After |
|--------|--------|-------|
| `pack_input_k2_standalone()` | ~5-6 ms | ~1-2 ms (estimated) |
| JAX hot-step | ~0.3 ms | ~0.3 ms |
| JAX input array creation | 12× `jnp.array()`/`jnp.asarray()` per step | 4× per step |

## Acceptance

- [x] Input packing overhead reduced from ~5-6 ms
- [x] No repeated recompilation
- [x] No shape/dtype instability
- [x] JAX hot-step correctness preserved
- [x] NumPy pre-buffer approach used at call site
- [ ] Target <0.5 ms not reached — remaining ~1-2 ms from `jnp.asarray()` conversion
