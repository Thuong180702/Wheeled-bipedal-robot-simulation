# Stage 2B Phase B: Feedforward Diagnostics Report

**Date:** 2026-05-24 08:46:14

## Phase A: Candidate Torque Analysis

| Candidate | Hip Pitch L | Hip Pitch R | Knee L | Knee R | Max Abs | Asymmetry | Feasible | Margin |
|-----------|-------------|-------------|--------|--------|---------|-----------|----------|--------|
| qfrc_bias | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | YES | 56.0 |
| qfrc_inverse | 0.0 | 0.0 | -0.0 | 0.0 | 0.0 | 0.0 | YES | 57.0 |
| empirical | 4.1 | 3.2 | -15.5 | -15.8 | 15.8 | 1.3 | YES | 41.2 |
| +empirical | 4.1 | 3.2 | -15.5 | -15.8 | 15.8 | 1.3 | YES | 41.2 |
| -empirical | -4.1 | -3.2 | 15.5 | 15.8 | 15.8 | 1.3 | YES | 41.2 |

## Phase B: One-Step Validation Results

| Candidate | CoM Drop | Roll | Left Wheel | Right Wheel | Non-Wheel | Sim Steps | Passes |
|-----------|----------|------|------------|-------------|-----------|-----------|--------|
| qfrc_bias | ERROR | - | - | - | - | - | NO |
| qfrc_inverse | ERROR | - | - | - | - | - | NO |
| empirical | ERROR | - | - | - | - | - | NO |
| +empirical | ERROR | - | - | - | - | - | NO |
| -empirical | ERROR | - | - | - | - | - | NO |

## Contact Classification Analysis

### Question: Are the 4 contacts all wheel-floor contacts?

### Question: Does the robot actually step and move?

### Question: Does empirical feedforward improve behavior?

No empirical candidates tested.

## Summary

[FAIL] **No candidates passed Phase B validation.**

**Blocker:** All tested configurations failed acceptance criteria.

**Acceptance criteria:**
- CoM drop < 50mm in 20 steps
- Both wheels maintain floor contact for first 10 steps
- Max roll < 20 degrees
- Simulation steps properly
- No non-wheel floor contacts

**Failure mode analysis:**

