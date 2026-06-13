# Step E No-Pitch-Reserve Revert Report

**Date:** 2026-06-01  
**Status:** pitch-reserve allocator reverted; no-pitch-reserve baseline restored  
**Scope:** Step E only; Step C remains blocked

## 1. Root cause for revert

`pitch_reserve_tau=2.0` is the wrong Step E fix family.

It preserved pitch torque by clipping counteracting `tau_position` too aggressively during the transient. The result was worse peak support-position error:

- no pitch reserve prior best: `0.3858 m`
- pitch reserve V3: `0.6657 m`
- final improvement was trivial: `0.0527 m -> 0.0448 m`

Therefore pitch reserve must not be used as the Step E final fix.

## 2. Restored allocator

The restored torque-budget-aware allocator is pure total-torque-bound allocation:

```text
tau_position_lower_bound = -max_tau_wheel - tau_balance_before_position
tau_position_upper_bound = +max_tau_wheel - tau_balance_before_position
tau_position = clip(tau_position_raw, lower, upper)
```

No pitch-reserve term is applied.

## 3. Validation runs

All runs used:

- `--controller-mode balance-core`
- `--sagittal-controller velocity-damped`
- `--vd-k-position 20.0`
- `--vd-k-velocity 15.0`
- `--vd-k-support-velocity 10.0`
- `--vd-enable-torque-budget-aware-position`
- no steady-state integral

Telemetry files:

- V1 1000: `outputs/hierarchical_controller_sim/telemetry_1780277761.csv`
- V2 2000: `outputs/hierarchical_controller_sim/telemetry_1780277894.csv`
- V3 5000: `outputs/hierarchical_controller_sim/telemetry_1780278172.csv`

## 4. Results

| Run | Rows | Max abs support error | Final abs support error | Termination |
|---|---:|---:|---:|---|
| V1 | 1000 | `0.121484 m` | `0.121484 m` | completed |
| V2 | 2000 | `0.385799 m` | `0.163960 m` | completed |
| V3 | 5000 | `0.385799 m` | `0.052668 m` | completed |

V3 restored the prior best no-pitch-reserve behavior exactly.

## 5. V3 diagnostics

- support error range: `[-0.008281, +0.385799] m`
- pitch range: `[-0.099, +10.401] deg`
- roll range: `[-2.397, +0.168] deg`
- CoM z range: `[0.363021, 0.408641] m`
- wheel velocity mean range: `[-4.754, +1.321] rad/s`
- `position_authority_reason`: `within_bounds` for all rows
- `tau_pitch_reserve_applied_max`: `0.0 Nm`
- ownership violations: `0`
- hidden torque norm: `0.0`

## 6. Gate result

- Preferred containment (`max_abs <= 0.10`, final `<= 0.05`): **fail**
- Fallback containment (`max_abs <= 0.15`, final `<= 0.10`): **fail**
- Hard minimum peak (`max_abs <= 0.30`): **fail**
- Hard minimum final (`final_abs <= 0.10`): **pass**

## 7. Interpretation

The revert fixed the pitch-reserve regression and restored the prior best baseline, but Step E is still not solved because the transient peak remains `0.3858 m`, above the hard peak gate of `0.30 m`.

Do **not** proceed to Step C.
