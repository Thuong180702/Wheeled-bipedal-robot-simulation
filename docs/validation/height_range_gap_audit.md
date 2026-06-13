# Phase 3: Height-Range Gap Audit

**Date:** 2026-06-06
**Phase:** HEIGHT_RANGE_GAP_AUDIT

---

## 1. Three Height Range Comparison

### A. Old Validated Dynamic Envelope

| Metric | Value |
|--------|-------|
| Min | **0.3933 m** |
| Max | **0.4128 m** |
| Span | **1.95 cm** |
| Variants | 5 (low_small, low_tiny, nominal, high_tiny, high_small) |
| Status | Step E/C PASS |

### B. Desired Extreme Targets

| Metric | Low | High |
|--------|-----|------|
| Target | **0.300 m** | **0.480 m** |
| Achieved | 0.2955 m | 0.4810 m |
| Gap from validated | **-9.93 cm** | **+6.72 cm** |

### C. Physical/Static Envelope

| Metric | Value |
|--------|-------|
| Physical Min | **0.2919 m** |
| Physical Max | **0.4908 m** |
| Physical Span | **19.89 cm** |
| Valid candidates | 61/61 (100%) |

---

## 2. Height-Range Evidence Table

### In Old Validated Range (0.394-0.414m)

| Height (m) | Source | Static Feasible | Step E Gate | Posture Gate | Support Gate | Step C | Evidence Status |
|------------|--------|-----------------|-------------|--------------|--------------|--------|-----------------|
| **0.394** | low_small variant | ✓ | **PASS** | **PASS** | **PASS** | **PASS** | VALIDATED_DYNAMIC_ENVELOPE |
| **0.399** | low_tiny variant | ✓ | **PASS** | **PASS** | **PASS** | **PASS** | VALIDATED_DYNAMIC_ENVELOPE |
| **0.404** | nominal variant | ✓ | **PASS** | **PASS** | **PASS** | **PASS** | VALIDATED_DYNAMIC_ENVELOPE |
| **0.409** | high_tiny variant | ✓ | **PASS** | **PASS** | **PASS** | **PASS** | VALIDATED_DYNAMIC_ENVELOPE |
| **0.414** | high_small variant | ✓ | **PASS** | **PASS** | **PASS** | **PASS** | VALIDATED_DYNAMIC_ENVELOPE |

### Intermediate Low Extension (0.300-0.380m)

| Height (m) | Source | Static Feasible | Step E Old Gate | Posture Gate | Support Gate | Step C | Evidence Status |
|------------|--------|-----------------|-----------------|--------------|--------------|--------|-----------------|
| **0.300** | low_0p300_setup | ✓ (0.295m achieved) | UNTESTED | **FAIL** (div 0.493) | DEFERRED | UNTESTED | CONTACT_HEIGHT_SAFE_BUT_POSTURE_FAILS |
| **0.330** | low_0p330_setup | ✓ (0.335m achieved) | UNTESTED | UNTESTED | DEFERRED | UNTESTED | UNKNOWN_REQUIRES_TEST |
| **0.360** | low_0p360_setup | ✓ (0.363m achieved) | UNTESTED | UNTESTED | DEFERRED | UNTESTED | UNKNOWN_REQUIRES_TEST |
| **0.380** | (interpolation) | ✓ | UNTESTED | UNTESTED | DEFERRED | UNTESTED | UNKNOWN_REQUIRES_TEST |

### Intermediate High Extension (0.420-0.490m)

| Height (m) | Source | Static Feasible | Step E Old Gate | Posture Gate | Support Gate | Step C | Evidence Status |
|------------|--------|-----------------|-----------------|--------------|--------------|--------|-----------------|
| **0.420** | (interpolation) | ✓ | UNTESTED | UNTESTED | DEFERRED | UNTESTED | UNKNOWN_REQUIRES_TEST |
| **0.430** | (interpolation) | ✓ | UNTESTED | UNTESTED | DEFERRED | UNTESTED | UNKNOWN_REQUIRES_TEST |
| **0.450** | high_0p450_setup | ✓ (0.451m achieved) | UNTESTED | UNTESTED | DEFERRED | UNTESTED | UNKNOWN_REQUIRES_TEST |
| **0.465** | (interpolation) | ✓ | UNTESTED | UNTESTED | DEFERRED | UNTESTED | UNKNOWN_REQUIRES_TEST |
| **0.480** | high_0p480_setup | ✓ (0.481m achieved) | UNTESTED | **FAIL** (div 0.340) | DEFERRED | UNTESTED | CONTACT_HEIGHT_SAFE_BUT_POSTURE_FAILS |

### Beyond Physical Min/Max

| Height (m) | Source | Static Feasible | Step E Old Gate | Posture Gate | Evidence Status |
|------------|--------|-----------------|-----------------|--------------|-----------------|
| **< 0.292** | physical_min | ✗ | N/A | N/A | OUTSIDE_CURRENT_EVIDENCE |
| **> 0.491** | physical_max | ✗ | N/A | N/A | OUTSIDE_CURRENT_EVIDENCE |

---

## 3. Gap Analysis Summary

### Low-Side Gap

| Boundary | Height (m) | Gap (cm) |
|----------|-----------|----------|
| Validated min | 0.3933 | - |
| Target low | 0.300 | -9.33 |
| Physical min | 0.2919 | -10.14 |
| Static-ready min | 0.330 | -6.33 |
| Setup file exists | 0.300, 0.330, 0.360 | - |

**Key insight:** Static feasibility exists down to 0.292m. Setup files exist for 0.300, 0.330, 0.360m. No dynamic validation between 0.394m and 0.300m.

### High-Side Gap

| Boundary | Height (m) | Gap (cm) |
|----------|-----------|----------|
| Validated max | 0.4128 | - |
| Target high | 0.480 | +6.72 |
| Physical max | 0.4908 | +7.80 |
| Static-ready max | 0.450 | +3.72 |
| Setup file exists | 0.450, 0.480 | - |

**Key insight:** Static feasibility exists up to 0.491m. Setup files exist for 0.450, 0.480m. No dynamic validation between 0.414m and 0.450m.

---

## 4. Region Classification Summary

| Region | Heights | Classification | Evidence |
|--------|---------|----------------|----------|
| Validated envelope | 0.394-0.414 | **VALIDATED_DYNAMIC_ENVELOPE** | Step E/C 5/5 PASS |
| Low intermediate | 0.300-0.380 | **UNKNOWN_REQUIRES_TEST** | Static feasible, no dynamic validation |
| Low extreme | 0.300 | **CONTACT_HEIGHT_SAFE_BUT_POSTURE_FAILS** | Survived 5000 steps but div 0.493 > target |
| High intermediate | 0.420-0.450 | **UNKNOWN_REQUIRES_TEST** | Static feasible, no dynamic validation |
| High extreme | 0.480 | **CONTACT_HEIGHT_SAFE_BUT_POSTURE_FAILS** | Survived 5000 steps but div 0.340 > target |

---

## 5. Key Findings

1. **Physical envelope is 10× broader** than validated dynamic envelope
2. **Low-side gap is larger** (9.93cm vs 6.72cm high-side)
3. **Intermediate heights are UNTESTED** — no dynamic validation between 0.394m and 0.300m, or 0.414m and 0.450m
4. **Extreme targets (0.300/0.480) survive** but diverge — contact/height safe, posture fails
5. **Setup files exist** for 0.330, 0.360, 0.450m — intermediate validation is feasible

---

## 6. Files Created

- `outputs/height_range_extension_strategy_audit/height_range_gap_audit.json`
- (this document)