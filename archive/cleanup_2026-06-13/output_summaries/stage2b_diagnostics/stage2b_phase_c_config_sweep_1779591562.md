# Stage 2B Phase C: Configuration Sweep Report

**Date:** 2026-05-24 09:59:22

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
| +empirical | 0.25 | hip_pitch_knee | instant | 50/50 | pass | 1.8mm | 0.4° | 0.0° | 0.0% |
| +empirical | 0.25 | hip_pitch_knee | short | 50/50 | pass | 2.0mm | 0.4° | 0.0° | 0.0% |
| +empirical | 0.25 | hip_pitch_knee | medium | 50/50 | pass | 2.1mm | 0.4° | 0.0° | 0.0% |
| +empirical | 0.25 | hip_pitch | instant | 50/50 | pass | 4.6mm | 0.8° | 0.0° | 0.0% |
| +empirical | 0.25 | hip_pitch | short | 50/50 | pass | 4.6mm | 0.8° | 0.0° | 0.0% |
| +empirical | 0.25 | hip_pitch | medium | 50/50 | pass | 4.6mm | 0.8° | 0.0° | 0.0% |
| +empirical | 0.25 | knee | instant | 50/50 | pass | 1.9mm | 0.5° | 0.0° | 0.0% |
| +empirical | 0.25 | knee | short | 50/50 | pass | 2.1mm | 0.5° | 0.0° | 0.0% |
| +empirical | 0.25 | knee | medium | 50/50 | pass | 2.2mm | 0.6° | 0.0° | 0.0% |
| +empirical | 0.50 | hip_pitch_knee | instant | 50/50 | pass | 0.0mm | 0.3° | 0.0° | 0.0% |
| +empirical | 0.50 | hip_pitch_knee | short | 50/50 | pass | 0.0mm | 0.2° | 0.0° | 0.0% |
| +empirical | 0.50 | hip_pitch_knee | medium | 50/50 | pass | 0.2mm | 0.1° | 0.0° | 0.0% |
| +empirical | 0.50 | hip_pitch | instant | 50/50 | pass | 4.5mm | 0.6° | 0.0° | 0.0% |
| +empirical | 0.50 | hip_pitch | short | 50/50 | pass | 4.6mm | 0.6° | 0.0° | 0.0% |
| +empirical | 0.50 | hip_pitch | medium | 50/50 | pass | 4.6mm | 0.7° | 0.0° | 0.0% |
| +empirical | 0.50 | knee | instant | 50/50 | pass | 0.0mm | 0.1° | 0.0° | 0.0% |
| +empirical | 0.50 | knee | short | 50/50 | pass | 0.0mm | 0.1° | 0.0° | 0.0% |
| +empirical | 0.50 | knee | medium | 50/50 | pass | 0.2mm | 0.2° | 0.0° | 0.0% |
| +empirical | 0.75 | hip_pitch_knee | instant | 50/50 | pass | 0.0mm | 0.9° | 0.0° | 0.0% |
| +empirical | 0.75 | hip_pitch_knee | short | 50/50 | pass | 0.0mm | 0.8° | 0.0° | 0.0% |
| +empirical | 0.75 | hip_pitch_knee | medium | 50/50 | pass | 0.0mm | 0.7° | 0.0° | 0.0% |
| +empirical | 0.75 | hip_pitch | instant | 50/50 | pass | 4.5mm | 0.5° | 0.0° | 0.0% |
| +empirical | 0.75 | hip_pitch | short | 50/50 | pass | 4.5mm | 0.5° | 0.0° | 0.0% |
| +empirical | 0.75 | hip_pitch | medium | 50/50 | pass | 4.5mm | 0.5° | 0.0° | 0.0% |
| +empirical | 0.75 | knee | instant | 50/50 | pass | 0.0mm | 0.4° | 0.0° | 0.0% |
| +empirical | 0.75 | knee | short | 50/50 | pass | 0.0mm | 0.3° | 0.0° | 0.0% |
| +empirical | 0.75 | knee | medium | 50/50 | pass | 0.0mm | 0.2° | 0.0° | 0.0% |
| +empirical | 1.00 | hip_pitch_knee | instant | 50/50 | pass | 0.0mm | 1.5° | 0.0° | 0.0% |
| +empirical | 1.00 | hip_pitch_knee | short | 50/50 | pass | 0.0mm | 1.4° | 0.0° | 0.0% |
| +empirical | 1.00 | hip_pitch_knee | medium | 50/50 | pass | 0.0mm | 1.3° | 0.0° | 0.0% |
| +empirical | 1.00 | hip_pitch | instant | 50/50 | pass | 4.4mm | 0.3° | 0.0° | 0.0% |
| +empirical | 1.00 | hip_pitch | short | 50/50 | pass | 4.4mm | 0.3° | 0.0° | 0.0% |
| +empirical | 1.00 | hip_pitch | medium | 50/50 | pass | 4.4mm | 0.3° | 0.0° | 0.0% |
| +empirical | 1.00 | knee | instant | 50/50 | pass | 0.0mm | 0.8° | 0.0° | 0.0% |
| +empirical | 1.00 | knee | short | 50/50 | pass | 0.0mm | 0.7° | 0.0° | 0.0% |
| +empirical | 1.00 | knee | medium | 50/50 | pass | 0.0mm | 0.6° | 0.0° | 0.0% |
| -empirical | 0.25 | hip_pitch_knee | instant | 50/50 | pass | 7.6mm | 1.6° | 0.0° | 0.0% |
| -empirical | 0.25 | hip_pitch_knee | short | 50/50 | pass | 7.5mm | 1.5° | 0.0° | 0.0% |
| -empirical | 0.25 | hip_pitch_knee | medium | 50/50 | pass | 7.4mm | 1.5° | 0.0° | 0.0% |
| -empirical | 0.25 | hip_pitch | instant | 50/50 | pass | 4.8mm | 1.1° | 0.0° | 0.0% |
| -empirical | 0.25 | hip_pitch | short | 50/50 | pass | 4.8mm | 1.1° | 0.0° | 0.0% |
| -empirical | 0.25 | hip_pitch | medium | 50/50 | pass | 4.8mm | 1.1° | 0.0° | 0.0% |
| -empirical | 0.25 | knee | instant | 50/50 | pass | 7.5mm | 1.4° | 0.0° | 0.0% |
| -empirical | 0.25 | knee | short | 50/50 | pass | 7.4mm | 1.4° | 0.0° | 0.0% |
| -empirical | 0.25 | knee | medium | 50/50 | pass | 7.3mm | 1.4° | 0.0° | 0.0% |
| -empirical | 0.50 | hip_pitch_knee | instant | 50/50 | pass | 10.5mm | 2.2° | 0.0° | 0.0% |
| -empirical | 0.50 | hip_pitch_knee | short | 50/50 | pass | 10.3mm | 2.1° | 0.0° | 0.0% |
| -empirical | 0.50 | hip_pitch_knee | medium | 50/50 | pass | 10.0mm | 2.1° | 0.0° | 0.0% |
| -empirical | 0.50 | hip_pitch | instant | 50/50 | pass | 4.9mm | 1.3° | 0.0° | 0.0% |
| -empirical | 0.50 | hip_pitch | short | 50/50 | pass | 4.9mm | 1.3° | 0.0° | 0.0% |
| -empirical | 0.50 | hip_pitch | medium | 50/50 | pass | 4.9mm | 1.3° | 0.0° | 0.0% |
| -empirical | 0.50 | knee | instant | 50/50 | pass | 10.4mm | 1.8° | 0.0° | 0.0% |
| -empirical | 0.50 | knee | short | 50/50 | pass | 10.1mm | 1.8° | 0.0° | 0.0% |
| -empirical | 0.50 | knee | medium | 50/50 | pass | 9.9mm | 1.8° | 0.0° | 0.0% |
| -empirical | 0.75 | hip_pitch_knee | instant | 50/50 | contact_unstable | 13.5mm | 2.7° | 0.0° | 0.0% |
| -empirical | 0.75 | hip_pitch_knee | short | 11/50 | wheel_contact_loss | 1.3mm | 0.3° | 0.0° | 0.0% |
| -empirical | 0.75 | hip_pitch_knee | medium | 50/50 | pass | 12.7mm | 2.6° | 0.0° | 0.0% |
| -empirical | 0.75 | hip_pitch | instant | 50/50 | pass | 5.0mm | 1.5° | 0.0° | 0.0% |
| -empirical | 0.75 | hip_pitch | short | 50/50 | pass | 5.0mm | 1.4° | 0.0° | 0.0% |
| -empirical | 0.75 | hip_pitch | medium | 50/50 | pass | 5.0mm | 1.4° | 0.0° | 0.0% |
| -empirical | 0.75 | knee | instant | 50/50 | contact_unstable | 13.2mm | 2.2° | 0.0° | 0.0% |
| -empirical | 0.75 | knee | short | 11/50 | wheel_contact_loss | 1.4mm | 0.2° | 0.0° | 0.0% |
| -empirical | 0.75 | knee | medium | 50/50 | pass | 12.5mm | 2.2° | 0.0° | 0.0% |
| -empirical | 1.00 | hip_pitch_knee | instant | 11/50 | wheel_contact_loss | 2.0mm | 0.5° | 0.0° | 0.0% |
| -empirical | 1.00 | hip_pitch_knee | short | 11/50 | wheel_contact_loss | 1.4mm | 0.4° | 0.0° | 0.0% |
| -empirical | 1.00 | hip_pitch_knee | medium | 11/50 | wheel_contact_loss | 1.2mm | 0.3° | 0.0° | 0.0% |
| -empirical | 1.00 | hip_pitch | instant | 50/50 | pass | 5.1mm | 1.6° | 0.0° | 0.0% |
| -empirical | 1.00 | hip_pitch | short | 50/50 | pass | 5.1mm | 1.6° | 0.0° | 0.0% |
| -empirical | 1.00 | hip_pitch | medium | 50/50 | pass | 5.0mm | 1.6° | 0.0° | 0.0% |
| -empirical | 1.00 | knee | instant | 11/50 | wheel_contact_loss | 2.0mm | 0.3° | 0.0° | 0.0% |
| -empirical | 1.00 | knee | short | 11/50 | wheel_contact_loss | 1.4mm | 0.3° | 0.0° | 0.0% |
| -empirical | 1.00 | knee | medium | 50/50 | pass | 15.1mm | 2.6° | 0.0° | 0.0% |

**Stage 1 survivors:** 63/72

## Stage 2: 100-Step Validation Results

| Sign | Scale | Joint Group | Ramp | Survival | Reason | CoM Drop | Max Roll | Max Pitch | Saturation |
|------|-------|-------------|------|----------|--------|----------|----------|-----------|------------|
| +empirical | 0.25 | hip_pitch_knee | instant | 100/100 | pass | 3.0mm | 0.5° | 0.0° | 0.0% |
| +empirical | 0.25 | hip_pitch_knee | short | 100/100 | pass | 3.0mm | 0.5° | 0.0° | 0.0% |
| +empirical | 0.25 | hip_pitch_knee | medium | 100/100 | pass | 3.1mm | 0.6° | 0.0° | 0.0% |
| +empirical | 0.25 | hip_pitch | instant | 100/100 | pass | 7.1mm | 1.2° | 0.0° | 0.0% |
| +empirical | 0.25 | hip_pitch | short | 100/100 | pass | 7.1mm | 1.2° | 0.0° | 0.0% |
| +empirical | 0.25 | hip_pitch | medium | 100/100 | pass | 7.1mm | 1.2° | 0.0° | 0.0% |
| +empirical | 0.25 | knee | instant | 100/100 | pass | 3.1mm | 0.8° | 0.0° | 0.0% |
| +empirical | 0.25 | knee | short | 100/100 | pass | 3.1mm | 0.8° | 0.0° | 0.0% |
| +empirical | 0.25 | knee | medium | 100/100 | pass | 3.2mm | 0.8° | 0.0° | 0.0% |
| +empirical | 0.50 | hip_pitch_knee | instant | 100/100 | pass | 0.0mm | 0.4° | 0.0° | 0.0% |
| +empirical | 0.50 | hip_pitch_knee | short | 100/100 | pass | 0.0mm | 0.3° | 0.0° | 0.0% |
| +empirical | 0.50 | hip_pitch_knee | medium | 100/100 | pass | 0.2mm | 0.3° | 0.0° | 0.0% |
| +empirical | 0.50 | hip_pitch | instant | 100/100 | pass | 6.9mm | 1.0° | 0.0° | 0.0% |
| +empirical | 0.50 | hip_pitch | short | 100/100 | pass | 6.9mm | 1.0° | 0.0° | 0.0% |
| +empirical | 0.50 | hip_pitch | medium | 100/100 | pass | 6.9mm | 1.0° | 0.0° | 0.0% |
| +empirical | 0.50 | knee | instant | 100/100 | pass | 0.0mm | 0.1° | 0.0° | 0.0% |
| +empirical | 0.50 | knee | short | 100/100 | pass | 0.0mm | 0.1° | 0.0° | 0.0% |
| +empirical | 0.50 | knee | medium | 100/100 | pass | 0.2mm | 0.2° | 0.0° | 0.0% |
| +empirical | 0.75 | hip_pitch_knee | instant | 100/100 | pass | 0.0mm | 1.3° | 0.0° | 0.0% |
| +empirical | 0.75 | hip_pitch_knee | short | 100/100 | pass | 0.0mm | 1.2° | 0.0° | 0.0% |
| +empirical | 0.75 | hip_pitch_knee | medium | 100/100 | pass | 0.0mm | 1.2° | 0.0° | 0.0% |
| +empirical | 0.75 | hip_pitch | instant | 100/100 | pass | 6.8mm | 0.7° | 0.0° | 0.0% |
| +empirical | 0.75 | hip_pitch | short | 100/100 | pass | 6.8mm | 0.7° | 0.0° | 0.0% |
| +empirical | 0.75 | hip_pitch | medium | 100/100 | pass | 6.8mm | 0.7° | 0.0° | 0.0% |
| +empirical | 0.75 | knee | instant | 100/100 | pass | 0.0mm | 0.5° | 0.0° | 0.0% |
| +empirical | 0.75 | knee | short | 100/100 | pass | 0.0mm | 0.5° | 0.0° | 0.0% |
| +empirical | 0.75 | knee | medium | 100/100 | pass | 0.0mm | 0.5° | 0.0° | 0.0% |
| +empirical | 1.00 | hip_pitch_knee | instant | 100/100 | pass | 0.0mm | 2.2° | 0.0° | 0.0% |
| +empirical | 1.00 | hip_pitch_knee | short | 100/100 | pass | 0.0mm | 2.1° | 0.0° | 0.0% |
| +empirical | 1.00 | hip_pitch_knee | medium | 100/100 | pass | 0.0mm | 2.1° | 0.0° | 0.0% |
| +empirical | 1.00 | hip_pitch | instant | 100/100 | pass | 6.7mm | 0.5° | 0.0° | 0.0% |
| +empirical | 1.00 | hip_pitch | short | 100/100 | pass | 6.7mm | 0.5° | 0.0° | 0.0% |
| +empirical | 1.00 | hip_pitch | medium | 100/100 | pass | 6.7mm | 0.5° | 0.0° | 0.0% |
| +empirical | 1.00 | knee | instant | 100/100 | pass | 0.0mm | 1.2° | 0.0° | 0.0% |
| +empirical | 1.00 | knee | short | 100/100 | pass | 0.0mm | 1.2° | 0.0° | 0.0% |
| +empirical | 1.00 | knee | medium | 100/100 | pass | 0.0mm | 1.1° | 0.0° | 0.0% |
| -empirical | 0.25 | hip_pitch_knee | instant | 100/100 | pass | 11.5mm | 2.3° | 0.0° | 0.0% |
| -empirical | 0.25 | hip_pitch_knee | short | 100/100 | pass | 11.4mm | 2.3° | 0.0° | 0.0% |
| -empirical | 0.25 | hip_pitch_knee | medium | 100/100 | pass | 11.4mm | 2.3° | 0.0° | 0.0% |
| -empirical | 0.25 | hip_pitch | instant | 100/100 | pass | 7.3mm | 1.7° | 0.0° | 0.0% |
| -empirical | 0.25 | hip_pitch | short | 100/100 | pass | 7.3mm | 1.7° | 0.0° | 0.0% |
| -empirical | 0.25 | hip_pitch | medium | 100/100 | pass | 7.3mm | 1.7° | 0.0° | 0.0% |
| -empirical | 0.25 | knee | instant | 100/100 | pass | 11.4mm | 2.1° | 0.0° | 0.0% |
| -empirical | 0.25 | knee | short | 100/100 | pass | 11.3mm | 2.1° | 0.0° | 0.0% |
| -empirical | 0.25 | knee | medium | 100/100 | pass | 11.3mm | 2.1° | 0.0° | 0.0% |
| -empirical | 0.50 | hip_pitch_knee | instant | 100/100 | pass | 15.8mm | 3.2° | 0.0° | 0.0% |
| -empirical | 0.50 | hip_pitch_knee | short | 100/100 | pass | 15.7mm | 3.2° | 0.0° | 0.0% |
| -empirical | 0.50 | hip_pitch_knee | medium | 100/100 | pass | 15.7mm | 3.2° | 0.0° | 0.0% |
| -empirical | 0.50 | hip_pitch | instant | 100/100 | pass | 7.5mm | 1.9° | 0.0° | 0.0% |
| -empirical | 0.50 | hip_pitch | short | 100/100 | pass | 7.5mm | 1.9° | 0.0° | 0.0% |
| -empirical | 0.50 | hip_pitch | medium | 100/100 | pass | 7.5mm | 1.9° | 0.0° | 0.0% |
| -empirical | 0.50 | knee | instant | 100/100 | pass | 15.6mm | 2.7° | 0.0° | 0.0% |
| -empirical | 0.50 | knee | short | 100/100 | pass | 15.5mm | 2.7° | 0.0° | 0.0% |
| -empirical | 0.50 | knee | medium | 100/100 | pass | 15.4mm | 2.7° | 0.0° | 0.0% |
| -empirical | 0.75 | hip_pitch_knee | medium | 100/100 | pass | 20.0mm | 4.1° | 0.0° | 0.0% |
| -empirical | 0.75 | hip_pitch | instant | 100/100 | pass | 7.6mm | 2.1° | 0.0° | 0.0% |
| -empirical | 0.75 | hip_pitch | short | 100/100 | pass | 7.6mm | 2.1° | 0.0° | 0.0% |
| -empirical | 0.75 | hip_pitch | medium | 100/100 | pass | 7.6mm | 2.1° | 0.0° | 0.0% |
| -empirical | 0.75 | knee | medium | 100/100 | pass | 19.5mm | 3.4° | 0.0° | 0.0% |
| -empirical | 1.00 | hip_pitch | instant | 100/100 | pass | 7.8mm | 2.4° | 0.0° | 0.0% |
| -empirical | 1.00 | hip_pitch | short | 100/100 | pass | 7.7mm | 2.4° | 0.0° | 0.0% |
| -empirical | 1.00 | hip_pitch | medium | 100/100 | pass | 7.7mm | 2.4° | 0.0° | 0.0% |
| -empirical | 1.00 | knee | medium | 100/100 | pass | 23.7mm | 4.0° | 0.0° | 0.0% |

**Stage 2 survivors:** 63/63

## Recommendation

[SUCCESS] **Best configuration found:**

- **Sign:** -empirical
- **Scale:** 0.25
- **Joint group:** hip_pitch_knee
- **Ramp mode:** medium
- **Survival:** 100/100 steps
- **CoM drop:** 11.4mm
- **Max roll:** 2.3°
- **Max pitch:** 0.0°
- **Mean saturation:** 0.0%
- **Mean torque:** 9.5 Nm

**Next step:** Implement StaticFeedforwardController with this configuration.

