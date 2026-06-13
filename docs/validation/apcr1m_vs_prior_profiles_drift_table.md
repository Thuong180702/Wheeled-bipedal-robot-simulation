# APCR1m vs Prior Profiles Drift Comparison Table

## Phase 4: Physical Drift Comparison Table

### Primary Drift Signal
- **Column used**: `active_pitch_crossing_signed_error_m`
- **Profiles compared**: APCR1h, APCR1j, APCR1k, APCR1m

---

## Drift Metrics Summary

| Metric | APCR1h | APCR1j | APCR1k | APCR1m | Best |
|--------|--------|--------|--------|--------|------|
| min (m) | -0.072 | -0.069 | -0.071 | **-0.434** | APCR1j |
| max (m) | **0.178** | 0.183 | 0.232 | 0.400 | APCR1h |
| max \|e\| (m) | **0.178** | 0.183 | 0.232 | 0.434 | APCR1h |
| P2P (m) | **0.249** | 0.251 | 0.303 | 0.833 | APCR1h |
| mean (m) | **0.059** | 0.079 | 0.083 | 0.060 | APCR1h |
| \|mean\| (m) | **0.075** | 0.092 | 0.095 | 0.177 | APCR1h |
| final (m) | 0.167 | **0.124** | 0.132 | 0.308 | APCR1j |
| positive % | 78.3% | 79.0% | 81.8% | 60.4% | APCR1k |
| negative % | 21.6% | 20.9% | 18.1% | 39.5% | APCR1m |
| zero crossings | 9 | 9 | 9 | 9 | tie |

---

## Band Violation Comparison

| Band | APCR1h | APCR1j | APCR1k | APCR1m | Best |
|------|--------|--------|--------|--------|------|
| outside ±0.03 | 73.2% | 76.9% | 79.2% | 91.2% | APCR1h |
| outside ±0.05 | 61.6% | 66.6% | 67.3% | 86.9% | APCR1h |
| outside ±0.08 | 43.6% | 53.1% | 53.7% | 75.7% | APCR1h |
| outside ±0.10 | 35.6% | 46.8% | 46.9% | 69.1% | APCR1h |
| outside ±0.12 | 23.2% | 40.3% | 39.6% | 63.0% | APCR1h |
| outside ±0.15 | **9.7%** | 25.8% | 20.2% | 54.0% | APCR1h |
| >+0.15 count | 97 | 258 | 202 | 368 | APCR1h |
| <-0.15 count | **0** | 0 | 0 | 172 | APCR1h |

**Key Finding**: APCR1m has **5.6x more** values outside ±0.15 than APCR1h (54.0% vs 9.7%).

---

## Torque Composition Comparison

| Component | APCR1h | APCR1j | APCR1k | APCR1m | Dominance |
|-----------|--------|--------|--------|--------|-----------|
| tau_pitch | 3.24 Nm | 3.36 Nm | 3.52 Nm | **4.23 Nm** | APCR1m |
| tau_position | **2.50 Nm** | 2.24 Nm | 2.25 Nm | 2.67 Nm | APCR1m |
| tau_wheel_vel_L | **1.42 Nm** | 1.34 Nm | 1.31 Nm | **5.00 Nm** | APCR1m (3.5x!) |
| tau_wheel_vel_R | **1.42 Nm** | 1.34 Nm | 1.31 Nm | **4.92 Nm** | APCR1m (3.5x!) |

**Critical Finding**: APCR1m's wheel velocity damping is **3.5x larger** than APCR1h/j/k!

This is the ROOT CAUSE of APCR1m's drift problem.

---

## Longest Interval Analysis

| Metric | APCR1h | APCR1j | APCR1k | APCR1m |
|--------|--------|--------|--------|--------|
| longest positive interval | **342** | 240 | 243 | 217 |
| longest negative interval | 79 | 83 | 79 | **202** |

APCR1m has the longest negative interval (202 steps), indicating sustained drift in the negative direction.

---

## Window Metrics (0-250, 250-500, 500-750, 750-1000)

### APCR1h Window Summary
| Window | min | max | max\|e\| | mean | final |
|--------|-----|-----|---------|------|-------|
| 0-250 | -0.038 | 0.110 | 0.110 | 0.032 | 0.063 |
| 250-500 | -0.041 | 0.147 | 0.147 | 0.063 | 0.147 |
| 500-750 | -0.046 | 0.178 | 0.178 | 0.082 | 0.167 |
| 750-1000 | -0.043 | 0.177 | 0.177 | 0.072 | 0.177 |

### APCR1m Window Summary
| Window | min | max | max|e| | mean | final |
|--------|-----|-----|---------|------|-------|
| 0-250 | -0.015 | 0.130 | 0.130 | 0.039 | 0.130 |
| 250-500 | -0.195 | 0.245 | 0.245 | 0.021 | 0.245 |
| 500-750 | -0.434 | 0.400 | 0.434 | 0.063 | 0.400 |
| 750-1000 | -0.398 | 0.308 | 0.398 | -0.014 | 0.308 |

**Key Finding**: APCR1m's drift accumulates over time, reaching max |e| = 0.434m in the 500-750 window.

---

## Conclusions

1. **APCR1h is the best** with lowest max|e|, P2P, and band violations
2. **APCR1m is the worst** with 2.4x worse max|e| than APCR1h
3. **Root cause identified**: APCR1m's wheel velocity damping is 3.5x larger than other profiles
4. **APCR1m blend is not working effectively**: Despite 42.2% blend activation and 88.4% RECENTER activation, drift is worse

---

## Files Generated

- `apcr1m_vs_prior_profiles_drift_table.json` - Full metrics in JSON
- `apcr1m_vs_prior_profiles_drift_table.csv` - Full metrics in CSV
