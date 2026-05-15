# Phase B.9 Step 5.26 — Phase 1 Summary

**Date:** 2026-05-15  
**Phase:** Infrastructure (Week 1)  
**Status:** Complete

## Objectives

Build infrastructure for centroidal state estimation and height-dependent capture point computation to enable dynamic balance control.

## Deliverables

### 1. CentroidalState Dataclass
- **File:** `wheeled_biped/controllers/centroidal_state_estimator.py`
- **Purpose:** Data structure for centroidal state (CoM, capture point, momentum, contact)
- **Status:** ✓ Complete

### 2. CentroidalStateEstimator
- **File:** `wheeled_biped/controllers/centroidal_state_estimator.py`
- **Purpose:** Extract centroidal state from MJX simulation data
- **Features:**
  - CoM position extraction from `data.subtree_com[1]`
  - CoM velocity via finite difference (functional pattern with prev_com_pos parameter)
  - Contact force extraction from `data.contact.force`
  - Linear momentum computation
  - JAX/MJX-first architecture with frozen dataclasses
- **Status:** ✓ Complete

### 3. CapturePointEstimator
- **File:** `wheeled_biped/controllers/capture_point_estimator.py`
- **Purpose:** Compute height-dependent LIP capture point
- **Features:**
  - Height-dependent natural frequency: ω(h) = √(g/h)
  - Capture point: [x_cp, y_cp] = [x_com, y_com] + [vx, vy]/ω(h)
  - Divergence computation
  - Height clamping to avoid division by zero
- **Status:** ✓ Complete

### 4. Unit Tests
- **Files:** 
  - `tests/test_centroidal_state_estimator.py`
  - `tests/test_capture_point_estimator.py`
  - `tests/test_centroidal_integration.py`
- **Coverage:**
  - CentroidalState creation
  - CoM extraction from MJX data
  - First-call zero velocity handling
  - Velocity computation via finite difference
  - Contact force extraction
  - Capture point computation at various heights
  - Height dependency validation
  - 100-step no-NaN rollout integration test
- **Status:** ✓ Complete

## Validation Results

### Unit Tests
- All unit tests pass
- No NaN values in 100-step rollout (17 minutes runtime)
- Contact forces extracted correctly (non-zero values)
- Capture point computation matches analytical LIP model
- Height dependency verified (lower height → smaller capture point offset)

### Key Findings

1. **JAX/MJX Architecture Enforced**: Initial implementation used NumPy arrays and stateful mutations, which were caught by code quality review and fixed to use JAX arrays with functional patterns
2. **Contact Force Extraction Fixed**: Step 5.25's 0.0% contact activation issue resolved by properly extracting from `data.contact.force`
3. **Height-Dependent Capture Point Works**: Capture point correctly varies with CoM height as expected from LIP theory
4. **No NaN Issues**: 100-step rollout produces stable, valid values across all state fields
5. **Functional State Management**: Refactored to functional pattern where `estimate()` returns `(state, com_pos)` tuple for caller to manage state

## Next Steps

**Phase 2: Centroidal WBC Core (Week 2)**
- Implement CentroidalBalanceController skeleton
- Add CoM regulation with deadband control
- Add capture point tracking
- Integrate with existing height IK and roll stabilization
- Implement 60% authority budget clipping

## Files Created

```
wheeled_biped/controllers/
├── centroidal_state_estimator.py  (NEW)
└── capture_point_estimator.py     (NEW)

tests/
├── test_centroidal_state_estimator.py  (NEW)
├── test_capture_point_estimator.py     (NEW)
└── test_centroidal_integration.py      (NEW)

docs/
└── phase_b9_step5_26_phase1_summary.md  (NEW)
```

## Commits

1. `feat: add CentroidalState dataclass for dynamic balance` (0b76b77)
2. `fix: convert CentroidalState to JAX arrays for JIT compatibility` (d2b4272)
3. `feat: add CoM extraction from MJX data` (dd6d769)
4. `fix: refactor CentroidalStateEstimator to JAX functional pattern` (0237e93)
5. `feat: add contact force extraction from MJX data` (ef6a74d)
6. `feat: add CapturePointEstimator skeleton` (8c5e2e8)
7. `feat: implement height-dependent LIP capture point computation` (bca50c4)
8. `test: add 100-step no-NaN rollout integration test` (6fafc1c)

---

**Phase 1 Status:** ✓ Complete  
**Ready for Phase 2:** Yes
