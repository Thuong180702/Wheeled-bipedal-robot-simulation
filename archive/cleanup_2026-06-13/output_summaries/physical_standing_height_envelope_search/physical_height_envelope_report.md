# Physical Standing Height Envelope Search Report

**Verdict:** PHYSICAL_ENVELOPE_PASS

## Search summary

- Valid candidates: 61
- Invalid candidates: 0
- Total evaluated: 61

## Physical extrema

### Physical minimum height: 0.291919 m

- Hip pitch: 1.226052 rad
- Knee: 2.348364 rad
- Root z: 0.398301 m
- Joint limit margin: 0.351636 rad

### Physical maximum height: 0.490812 m

- Hip pitch: 0.626052 rad
- Knee: 1.148364 rad
- Root z: 0.642381 m
- Joint limit margin: 1.126052 rad

## Static revalidation

Verdict: PHYSICAL_ENVELOPE_PASS

Both extrema were revalidated by reloading setup JSON, rebuilding MuJoCo state, and recomputing static feasibility.

## Important notes

- This envelope is based on **static feasibility only**
- No controller constraints were applied
- No dynamic stability checks were performed
- Dynamic failure at these extrema does NOT invalidate the physical envelope
- The physical envelope quantifies kinematic workspace, not controller capability