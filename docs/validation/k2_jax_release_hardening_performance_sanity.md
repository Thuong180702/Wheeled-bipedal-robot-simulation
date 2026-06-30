# K2 JAX Release Hardening — Performance Sanity Check

**Date:** 2026-06-28
**Phase:** 6
**Classification:** K2_JAX_RELEASE_HARDENING_PERFORMANCE_SANITY_PASS

---

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Profile | k2_notch_low_q_v1 |
| State size | 835 |
| Input size | 42 |
| Params size | 41 (STAGE2) + 7 (EXT) = 48 |
| JIT backend | jax.jit with float64 |
| Iterations timed | 1000 |

---

## Results

### Compile Time
| Metric | Value |
|--------|-------|
| JIT compile + 3x warmup | **1.97 s** |

### Hot-Step Timing (1000 iterations)
| Metric | Value (ms) |
|--------|-----------|
| Mean | **0.185** |
| P50 (median) | 0.181 |
| P95 | 0.250 |
| Max | 0.523 |
| Min | 0.122 |
| Std | 0.040 |

### Threshold Check
| Metric | Value |
|--------|-------|
| Threshold | < 10.0 ms |
| Mean hot-step | 0.185 ms |
| Margin | 54× below threshold |
| Status | **PASS** |

### Recompilation
| Metric | Value |
|--------|-------|
| Recompilation count | **0** |
| make_jaxpr verification | 0.8 ms (no recompile) |

---

## Comparison with State Size History

| State Size | Approximate Hot-Step (estimated) | Notes |
|-----------|--------------------------------|-------|
| ~14 (STAGE2) | ~0.05 ms | Minimal state, no ABS ring buffer |
| ~332 (pre-ZC buffer) | ~0.10 ms | With ABS ring buffer (300+100) |
| 835 (current) | **0.185 ms** | With ZC buffer (500 entries) added |

The state size increase from 332 to 835 (ZC buffer addition) increased hot-step time from ~0.10 ms to ~0.185 ms. This is expected — the ZC buffer adds 500 float64 entries and the `_abs_count_zero_crossings_from_zc` scan operation. The 0.085 ms increase is proportional to the buffer size and scan complexity.

No performance regression attributable to both-synced-only state fields (`effective_max_position_tau_py`). This field is a single scalar read at step entry with negligible cost.

---

## Memory Footprint (Estimated)

| Buffer | Size (float64) | Bytes |
|--------|---------------|-------|
| State vector | 835 | 6,680 B |
| Input vector | 42 | 336 B |
| Params vector | 48 | 384 B |
| Diag vector | 45 | 360 B |
| **Total per-step** | **970** | **~7.8 KB** |

Comfortably within L1 cache (32 KB typical) for most XLA backends.

---

## Verdict

**Classification: K2_JAX_RELEASE_HARDENING_PERFORMANCE_SANITY_PASS**

Hot-step mean 0.185 ms is 54× below the 10 ms threshold. No recompilation. No unexpected regression from state size increase to 835. The ABS trim ring buffer and ZC buffer add proportional overhead as expected.
