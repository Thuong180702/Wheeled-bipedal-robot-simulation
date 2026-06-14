# Stage 2B Phase C: Configuration Sweep Report

**Date:** 2026-05-24 09:54:37

## Empirical Feedforward

| Joint | Torque (Nm) |
|-------|-------------|
| l_hip_roll | -0.1 |
| l_hip_yaw | 0.0 |
| l_hip_pitch | 4.1 |
| l_knee | -15.5 |
| l_wheel | 0.0 |
| r_hip_roll | 0.1 |
| r_hip_yaw | -0.0 |
| r_hip_pitch | 3.2 |
| r_knee | -15.8 |
| r_wheel | 0.0 |

**Max abs:** 15.8 Nm

## Stage 1: 50-Step Screening Results

| Sign | Scale | Joint Group | Ramp | Survival | Reason | CoM Drop | Max Roll | Max Pitch | Saturation |
|------|-------|-------------|------|----------|--------|----------|----------|-----------|------------|
| +empirical | 0.25 | hip_pitch_knee | instant | 11/50 | wheel_contact_loss | 0.0mm | 0.5° | 0.0° | 0.0% |
| +empirical | 0.25 | hip_pitch_knee | short | 11/50 | wheel_contact_loss | 0.0mm | 0.5° | 0.0° | 0.0% |
| +empirical | 0.25 | hip_pitch_knee | medium | 11/50 | wheel_contact_loss | 0.0mm | 0.6° | 0.0° | 0.0% |
| +empirical | 0.25 | hip_pitch | instant | 11/50 | wheel_contact_loss | 0.0mm | 0.6° | 0.0° | 0.0% |
| +empirical | 0.25 | hip_pitch | short | 11/50 | wheel_contact_loss | 0.0mm | 0.6° | 0.0° | 0.0% |
| +empirical | 0.25 | hip_pitch | medium | 11/50 | wheel_contact_loss | 0.0mm | 0.6° | 0.0° | 0.0% |
| +empirical | 0.25 | knee | instant | 11/50 | wheel_contact_loss | 0.0mm | 0.5° | 0.0° | 0.0% |
| +empirical | 0.25 | knee | short | 11/50 | wheel_contact_loss | 0.0mm | 0.6° | 0.0° | 0.0% |
| +empirical | 0.25 | knee | medium | 11/50 | wheel_contact_loss | 0.0mm | 0.6° | 0.0° | 0.0% |
| +empirical | 0.50 | hip_pitch_knee | instant | 11/50 | wheel_contact_loss | 0.0mm | 0.4° | 0.0° | 0.0% |
| +empirical | 0.50 | hip_pitch_knee | short | 11/50 | wheel_contact_loss | 0.0mm | 0.5° | 0.0° | 0.0% |
| +empirical | 0.50 | hip_pitch_knee | medium | 11/50 | wheel_contact_loss | 0.0mm | 0.5° | 0.0° | 0.0% |
| +empirical | 0.50 | hip_pitch | instant | 11/50 | wheel_contact_loss | 0.0mm | 0.5° | 0.0° | 0.0% |
| +empirical | 0.50 | hip_pitch | short | 11/50 | wheel_contact_loss | 0.0mm | 0.6° | 0.0° | 0.0% |
| +empirical | 0.50 | hip_pitch | medium | 11/50 | wheel_contact_loss | 0.0mm | 0.6° | 0.0° | 0.0% |
| +empirical | 0.50 | knee | instant | 11/50 | wheel_contact_loss | 0.0mm | 0.5° | 0.0° | 0.0% |
| +empirical | 0.50 | knee | short | 11/50 | wheel_contact_loss | 0.0mm | 0.5° | 0.0° | 0.0% |
| +empirical | 0.50 | knee | medium | 11/50 | wheel_contact_loss | 0.0mm | 0.6° | 0.0° | 0.0% |
| +empirical | 0.75 | hip_pitch_knee | instant | 11/50 | wheel_contact_loss | 0.0mm | 0.3° | 0.0° | 0.0% |
| +empirical | 0.75 | hip_pitch_knee | short | 11/50 | wheel_contact_loss | 0.0mm | 0.4° | 0.0° | 0.0% |
| +empirical | 0.75 | hip_pitch_knee | medium | 11/50 | wheel_contact_loss | 0.0mm | 0.5° | 0.0° | 0.0% |
| +empirical | 0.75 | hip_pitch | instant | 11/50 | wheel_contact_loss | 0.0mm | 0.5° | 0.0° | 0.0% |
| +empirical | 0.75 | hip_pitch | short | 11/50 | wheel_contact_loss | 0.0mm | 0.6° | 0.0° | 0.0% |
| +empirical | 0.75 | hip_pitch | medium | 11/50 | wheel_contact_loss | 0.0mm | 0.6° | 0.0° | 0.0% |
| +empirical | 0.75 | knee | instant | 11/50 | wheel_contact_loss | 0.0mm | 0.4° | 0.0° | 0.0% |
| +empirical | 0.75 | knee | short | 11/50 | wheel_contact_loss | 0.0mm | 0.5° | 0.0° | 0.0% |
| +empirical | 0.75 | knee | medium | 11/50 | wheel_contact_loss | 0.0mm | 0.5° | 0.0° | 0.0% |
| +empirical | 1.00 | hip_pitch_knee | instant | 11/50 | wheel_contact_loss | 0.0mm | 0.3° | 0.0° | 1.8% |
| +empirical | 1.00 | hip_pitch_knee | short | 11/50 | wheel_contact_loss | 0.0mm | 0.4° | 0.0° | 0.0% |
| +empirical | 1.00 | hip_pitch_knee | medium | 11/50 | wheel_contact_loss | 0.0mm | 0.4° | 0.0° | 0.0% |
| +empirical | 1.00 | hip_pitch | instant | 11/50 | wheel_contact_loss | 0.0mm | 0.5° | 0.0° | 0.0% |
| +empirical | 1.00 | hip_pitch | short | 11/50 | wheel_contact_loss | 0.0mm | 0.5° | 0.0° | 0.0% |
| +empirical | 1.00 | hip_pitch | medium | 11/50 | wheel_contact_loss | 0.0mm | 0.6° | 0.0° | 0.0% |
| +empirical | 1.00 | knee | instant | 11/50 | wheel_contact_loss | 0.0mm | 0.4° | 0.0° | 0.0% |
| +empirical | 1.00 | knee | short | 11/50 | wheel_contact_loss | 0.0mm | 0.4° | 0.0° | 0.0% |
| +empirical | 1.00 | knee | medium | 11/50 | wheel_contact_loss | 0.0mm | 0.5° | 0.0° | 0.0% |
| -empirical | 0.25 | hip_pitch_knee | instant | 11/50 | wheel_contact_loss | 0.0mm | 0.8° | 0.0° | 0.0% |
| -empirical | 0.25 | hip_pitch_knee | short | 11/50 | wheel_contact_loss | 0.0mm | 0.7° | 0.0° | 0.0% |
| -empirical | 0.25 | hip_pitch_knee | medium | 11/50 | wheel_contact_loss | 0.0mm | 0.7° | 0.0° | 0.0% |
| -empirical | 0.25 | hip_pitch | instant | 11/50 | wheel_contact_loss | 0.0mm | 0.7° | 0.0° | 0.0% |
| -empirical | 0.25 | hip_pitch | short | 11/50 | wheel_contact_loss | 0.0mm | 0.7° | 0.0° | 0.0% |
| -empirical | 0.25 | hip_pitch | medium | 11/50 | wheel_contact_loss | 0.0mm | 0.7° | 0.0° | 0.0% |
| -empirical | 0.25 | knee | instant | 11/50 | wheel_contact_loss | 0.0mm | 0.7° | 0.0° | 0.0% |
| -empirical | 0.25 | knee | short | 11/50 | wheel_contact_loss | 0.0mm | 0.7° | 0.0° | 0.0% |
| -empirical | 0.25 | knee | medium | 11/50 | wheel_contact_loss | 0.0mm | 0.7° | 0.0° | 0.0% |
| -empirical | 0.50 | hip_pitch_knee | instant | 11/50 | wheel_contact_loss | 0.0mm | 0.9° | 0.0° | 0.0% |
| -empirical | 0.50 | hip_pitch_knee | short | 11/50 | wheel_contact_loss | 0.0mm | 0.8° | 0.0° | 0.0% |
| -empirical | 0.50 | hip_pitch_knee | medium | 11/50 | wheel_contact_loss | 0.0mm | 0.7° | 0.0° | 0.0% |
| -empirical | 0.50 | hip_pitch | instant | 11/50 | wheel_contact_loss | 0.0mm | 0.7° | 0.0° | 0.0% |
| -empirical | 0.50 | hip_pitch | short | 11/50 | wheel_contact_loss | 0.0mm | 0.7° | 0.0° | 0.0% |
| -empirical | 0.50 | hip_pitch | medium | 11/50 | wheel_contact_loss | 0.0mm | 0.7° | 0.0° | 0.0% |
| -empirical | 0.50 | knee | instant | 11/50 | wheel_contact_loss | 0.0mm | 0.8° | 0.0° | 0.0% |
| -empirical | 0.50 | knee | short | 11/50 | wheel_contact_loss | 0.0mm | 0.8° | 0.0° | 0.0% |
| -empirical | 0.50 | knee | medium | 11/50 | wheel_contact_loss | 0.0mm | 0.7° | 0.0° | 0.0% |
| -empirical | 0.75 | hip_pitch_knee | instant | 11/50 | wheel_contact_loss | 0.0mm | 1.0° | 0.0° | 0.0% |
| -empirical | 0.75 | hip_pitch_knee | short | 11/50 | wheel_contact_loss | 0.0mm | 0.9° | 0.0° | 0.0% |
| -empirical | 0.75 | hip_pitch_knee | medium | 11/50 | wheel_contact_loss | 0.0mm | 0.8° | 0.0° | 0.0% |
| -empirical | 0.75 | hip_pitch | instant | 11/50 | wheel_contact_loss | 0.0mm | 0.8° | 0.0° | 0.0% |
| -empirical | 0.75 | hip_pitch | short | 11/50 | wheel_contact_loss | 0.0mm | 0.7° | 0.0° | 0.0% |
| -empirical | 0.75 | hip_pitch | medium | 11/50 | wheel_contact_loss | 0.0mm | 0.7° | 0.0° | 0.0% |
| -empirical | 0.75 | knee | instant | 11/50 | wheel_contact_loss | 0.0mm | 0.9° | 0.0° | 0.0% |
| -empirical | 0.75 | knee | short | 11/50 | wheel_contact_loss | 0.0mm | 0.8° | 0.0° | 0.0% |
| -empirical | 0.75 | knee | medium | 11/50 | wheel_contact_loss | 0.0mm | 0.7° | 0.0° | 0.0% |
| -empirical | 1.00 | hip_pitch_knee | instant | 11/50 | wheel_contact_loss | 0.0mm | 1.2° | 0.0° | 0.0% |
| -empirical | 1.00 | hip_pitch_knee | short | 11/50 | wheel_contact_loss | 0.0mm | 1.0° | 0.0° | 0.0% |
| -empirical | 1.00 | hip_pitch_knee | medium | 11/50 | wheel_contact_loss | 0.0mm | 0.8° | 0.0° | 0.0% |
| -empirical | 1.00 | hip_pitch | instant | 11/50 | wheel_contact_loss | 0.0mm | 0.8° | 0.0° | 0.0% |
| -empirical | 1.00 | hip_pitch | short | 11/50 | wheel_contact_loss | 0.0mm | 0.8° | 0.0° | 0.0% |
| -empirical | 1.00 | hip_pitch | medium | 11/50 | wheel_contact_loss | 0.0mm | 0.7° | 0.0° | 0.0% |
| -empirical | 1.00 | knee | instant | 11/50 | wheel_contact_loss | 0.0mm | 1.0° | 0.0° | 0.0% |
| -empirical | 1.00 | knee | short | 11/50 | wheel_contact_loss | 0.0mm | 0.9° | 0.0° | 0.0% |
| -empirical | 1.00 | knee | medium | 11/50 | wheel_contact_loss | 0.0mm | 0.8° | 0.0° | 0.0% |

**Stage 1 survivors:** 0/72

## Stage 2: Validation

No configurations passed Stage 1 screening.

## Recommendation

[FAIL] **No safe feedforward configuration found.**

**Possible causes:**
1. Empirical feedforward magnitude incorrect
2. Roll instability dominates
3. Contact solver instability
4. Need different control approach

