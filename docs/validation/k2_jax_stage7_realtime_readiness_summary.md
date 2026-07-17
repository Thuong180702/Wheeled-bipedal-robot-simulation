# Stage 7: Realtime-Readiness Summary

**Date:** 2026-06-27

## Key Metrics

- **JAX hot-step mean (average across scenarios):** 0.273 ms
- **JAX hot-step p95 (max across scenarios):** 0.345 ms
- **Meets 10 ms control budget:** YES

## Realtime Verdict

- **JAX hot-step meets 10ms budget:** [PASS] YES
- **No falls:** [PASS] YES
- **No NaN:** [PASS] YES
- **No per-step recompilation:** [PASS] YES

### Conclusion: Controller compute is realtime-ready [PASS]

The JAX JIT-compiled controller step executes in well under 1 ms with `block_until_ready()`, 
leaving ample headroom within the 10 ms control budget. The total simulation step time 
is dominated by Python-level operations (balance-core block, telemetry, state estimation) 
that run identically in both backends.

**Remaining blockers before changing default backend:**
1. Python balance-core computation runs in both backends for telemetry -- a 
   telemetry-decoupled mode would be needed for full JAX benefit.
2. Input packing (`pack_input_k2`) costs ~20 ms (Python->JAX device transfer).
3. Duplicate state estimation (control + log) adds ~1-2 ms overhead.
4. Visual/render overhead not yet benchmarked.

## Next Steps

1. Add visual benchmark runs (--visual mode)
2. Consider telemetry-decoupled mode for production JAX path
3. Consider moving input packing to JIT (pre-pack inputs)
4. Consider deduplicating state estimation
