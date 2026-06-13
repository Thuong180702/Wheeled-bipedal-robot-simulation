# Pitch-Position Interaction Audit Report

**Date:** 2026-06-05
**Purpose:** Understand why J1-J3 improve support/hip-yaw but exceed pitch gate

## Executive Summary

**J0 Baseline:**
- pitch: 0.0951 rad (PASS gate)
- support: 0.2430 m (FAIL gate)
- hip_yaw: 0.1615 rad (FAIL gate)

**J2 (support cap + moderate damping):**
- pitch: 0.1444 rad (FAIL gate, +51.8%)
- support: 0.1128 m (PASS gate, -53.6%)
- hip_yaw: 0.0393 rad (PASS gate, -75.6%)

**J3 (support cap + strong damping):**
- pitch: 0.1395 rad (FAIL gate, +46.6%)
- support: 0.0971 m (PASS gate, -60.0%)
- hip_yaw: 0.0487 rad (PASS gate, -69.8%)

## Mechanism Classification

### Position Authority Induces Pitch Overshoot

**Confidence:** HIGH

J1 pitch 0.1632 > J0 pitch 0.0951 with k_position 80.0 vs 40.0

### Max Position Tau Too High

**Confidence:** MEDIUM

J1 max_position_tau 6.0 > J0 3.0, tau_position 6.00 vs 3.00

### K Velocity Helps Damp Pitch

**Confidence:** MEDIUM

J3 k_velocity 30.0 > J2 25.0, J3 pitch 0.1395 < J2 pitch 0.1444

## Recommendation

**Pitch-safe candidate strategy:**

Design candidates with reduced position authority at low heights:
- Reduce k_position low max from 80 to 65-70
- Reduce max_position_tau low max from 6.0 to 4.5-5.0
- Keep k_velocity at 25-30 for velocity damping

Target: preserve support/hip-yaw improvements while staying under pitch gate.

## Files Generated

- `pitch_position_interaction_summary.json` - comprehensive metrics for all profiles
- `pitch_peak_windows.csv` - telemetry windows around pitch peaks
- `support_peak_windows.csv` - telemetry windows around support peaks
- `torque_interaction_comparison.csv` - torque term comparison across profiles
- `pitch_failure_classification.json` - mechanism classification with confidence
