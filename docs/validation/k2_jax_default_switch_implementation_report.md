# K2 JAX Default Promotion — Implementation Report

**Date:** 2026-06-29
**Phase:** 2 — Controlled Default Switch

---

## 1. Changes Made

### File: `scripts/simulate_hierarchical_controller.py`

#### 1.1 Added scope constants and detection function (after line 2097)

```python
_K2_JAX_DEFAULT_SAGITTAL_CONTROLLER = "velocity-damped"
_K2_JAX_DEFAULT_AUTHORITY_PROFILE = "k2_notch_low_q_v1"

def _is_validated_k2_jax_scope(args) -> bool:
    """Check if the current configuration is within the validated K2 JAX scope."""
    mode = getattr(args, "controller_mode", "legacy")
    sagittal = getattr(args, "sagittal_controller", "baseline")
    profile = getattr(args, "vd_sagittal_authority_profile", "baseline")
    return (
        mode == "balance-core"
        and sagittal == _K2_JAX_DEFAULT_SAGITTAL_CONTROLLER
        and profile == _K2_JAX_DEFAULT_AUTHORITY_PROFILE
    )

def resolve_controller_backend(args) -> str:
    """Resolve the controller backend with K2 JAX default promotion."""
    explicit = getattr(args, "controller_backend", None)
    if explicit is not None:
        return explicit
    if _is_validated_k2_jax_scope(args):
        return "jax"
    return "python"
```

#### 1.2 Changed argparse `--controller-backend` definition

- `default`: `"python"` → `None` (resolved post-parse)
- Removed `type=str` (to keep `None` as `None`, not `"None"`)
- Updated help text to describe new default policy

#### 1.3 Added backend resolution after `parse_args()` (line ~3535)

- Calls `resolve_controller_backend(args)` 
- Logs backend decision with source (explicit override / K2 JAX default / fallback)

#### 1.4 Updated runtime fallback

- `getattr(args, "controller_backend", "python")` → `getattr(args, "controller_backend", None) or "python"`
- Safety net in case backend resolution is somehow bypassed

---

## 2. Backend Selection Behavior

| User Input | Profile | Result |
|-----------|---------|--------|
| (none) | K2 validated scope | `jax` (default for K2) |
| (none) | Non-K2 profile | `python` (fallback) |
| `--controller-backend python` | Any | `python` (explicit) |
| `--controller-backend jax` | Any | `jax` (explicit) |
| `--controller-backend both-synced` | Any | `both-synced` (explicit) |
| `--controller-backend both` | Any | `both` (explicit) |

---

## 3. Log Output Examples

- K2 default: `[BACKEND] Controller backend: jax (default for validated K2 balance-core + velocity-damped + k2_notch_low_q_v1 profile)`
- Explicit Python: `[BACKEND] Controller backend: python (explicit user override)`
- Fallback: `[BACKEND] Controller backend: python (default fallback for non-validated profile)`

---

## 4. What Was NOT Changed

- Python backend code — preserved entirely
- JAX backend code — preserved entirely  
- both-synced path — preserved entirely
- Controller gains, thresholds, torque limits, safety gates — no changes
- `--controller-backend jax` still requires balance-core mode (JAX requirement check preserved)
- Validation scripts with their own `--controller-backend` defaults — not changed (they have independent parsers)

---

## 5. Acceptance

- [x] JAX default applies only in validated K2 scope (balance-core + velocity-damped + k2_notch_low_q_v1)
- [x] Explicit Python override works (user sets `--controller-backend python`)
- [x] Explicit JAX override works (user sets `--controller-backend jax`)
- [x] both-synced still works
- [x] Unvalidated profiles do not silently switch to JAX
- [x] Backend decision logged with source
- [x] Python backend code preserved
