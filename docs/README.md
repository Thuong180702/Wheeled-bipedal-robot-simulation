# `docs/` — what is here, and what it is for

Nothing in this directory is needed to reproduce the paper. The reproduction
path is the table in the top-level [README](../README.md): each result names its
producing script and its raw result file under `outputs/`.

What is here is the engineering record behind those results.

## For a reader working through the paper

- [`setup_macos.md`](setup_macos.md) — Apple-silicon install notes, for when the
  CUDA-free path in `requirements.txt` still needs local adjustment.
- [`design/k2_coordinated_stability_controller_design.md`](design/k2_coordinated_stability_controller_design.md)
  — the design intent behind the controller the paper calls ACC and the code
  calls `K2` / `V3_ANCHOR`.
- [`specs/k2_stability_improvement_objective.md`](specs/k2_stability_improvement_objective.md)
  — the objective that design was written against.
- [`analysis/k2_current_controller_interaction_audit.md`](analysis/k2_current_controller_interaction_audit.md)
  — how the torque channels interact, and where they conflict.

## `validation/` — internal audit trail

Roughly forty reports written while the controller was being built: root-cause
audits, gain sweeps, sign-convention checks, promotion decisions. They use
internal profile names (`T5`, `T6F`, `T6I`, `T6J`, `APCR1*`, `K1`, `K2`,
`step_c`/`step_d`/`step_e`) that do not appear in the paper, and several of them
argue for designs that were later rejected.

They are kept for two reasons: parts of the test suite assert their existence as
evidence for specific fixes, and both manuscripts cite
[`validation/V3_vs_V3_Assist_comparison_report.md`](validation/V3_vs_V3_Assist_comparison_report.md)
by path.

Read them as a record of how the controller reached its current form, not as a
description of what it currently does. For the latter, read
`k2_jax_controller_step` in
[`wheeled_biped/controllers/k2_jax_controller.py`](../wheeled_biped/controllers/k2_jax_controller.py).

Some of these reports link to per-step telemetry CSVs that are not shipped —
they are large, and every figure the reports rely on is quoted inline.
