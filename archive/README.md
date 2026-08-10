# `archive/` — keep this directory

The name is misleading and the contents look disposable. They are not. Several
of the harnesses that produce the paper's numbers read calibration data from
here at startup, so deleting this directory breaks the evaluation pipeline.

## What is actually needed

```
archive/cleanup_2026-06-13/output_summaries/
├── balance_core_true_height_variants/
│   └── variant_nominal__variant_setup.json      ← the nominal standing posture
└── balance_core_extended_height_range/
    ├── dynamic_low_5cm__variant_setup.json      ← the two endpoints of the
    └── dynamic_high_5cm__variant_setup.json        teleop height command range
```

`variant_nominal__variant_setup.json` holds the solved standing equilibrium —
joint targets, root height, the pitch reference — that every ACC evaluation
starts from. Fifteen scripts load it, among them
`scripts/measure_control_step_cost.py`, `scripts/verify_body_axes.py`,
`scripts/acc_stability_certificate.py`, `scripts/diagnose_parking_offset.py`,
`scripts/idle_longhorizon.py` and `scripts/collect_robustness_sweep.py`.
`wheeled_biped/teleop_shaper.py` loads the other two.

The remaining files are the sibling variants and the summary reports from the
same calibration campaign, kept so the provenance of those numbers is legible
rather than a bare JSON with no context.

## Why it lives under a directory called `archive`

The path is baked into the scripts that read it, and those scripts are cited by
name in the paper and in the repository README as the producers of specific
tables. Moving the data would either break the citations or require touching
every harness, so the location stayed and this note was added instead.

Everything else from that 2026-06-13 cleanup — roughly 1450 files of telemetry
from abandoned research lines — was removed from version control on 2026-08-10
and is not part of this repository.
