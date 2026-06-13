# Balance-Core Extended Longevity Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing balance-core validation pipeline to run long nominal longevity studies at 10000/20000/50000/100000 steps with opt-in logging-only decimation, failure-window preservation, and whole-run summary reporting, without changing controller behavior.

**Architecture:** Keep [scripts/validate_balance_core.py](scripts/validate_balance_core.py), [wheeled_biped/validation/balance_core_validator.py](wheeled_biped/validation/balance_core_validator.py), [scripts/simulate_hierarchical_controller.py](scripts/simulate_hierarchical_controller.py), and [wheeled_biped/validation/study_aggregator.py](wheeled_biped/validation/study_aggregator.py) as the single pipeline. Add explicit long-run logging flags to the simulator, thread them through the validator, compute whole-run metrics from full-rate runtime accumulation rather than from decimated rows alone, and generate Step A summaries under `outputs/balance_core_extended_longevity/`.

**Tech Stack:** Python 3.10+, pandas, pytest, argparse, pathlib, existing balance-core telemetry schema/checker/classifier/reporting pipeline.

---

## File Structure

**Modify:**
- `scripts/validate_balance_core.py` — extend CLI for Step A duration lists, long-run logging flags, default output routing for extended longevity mode, and top-level summary generation through the existing study aggregator.
- `wheeled_biped/validation/balance_core_validator.py` — extend validator result types, long-run simulator argument forwarding, per-duration artifact tracking, classification source selection, and stop-at-first-failure behavior.
- `scripts/simulate_hierarchical_controller.py` — add opt-in logging-only decimation, rolling failure-window buffer, summary metric accumulation, and optional summary-sidecar emission without touching controller/simulation logic.
- `wheeled_biped/validation/study_aggregator.py` — enrich per-duration metrics, aggregate extended longevity results, emit Step A JSON/markdown summaries, and carry integrity/limitation notes.
- `wheeled_biped/validation/classification_report.py` — optionally include provenance fields such as classification telemetry source and fix-scope text in markdown/JSON reports if validator result wiring requires it.
- `tests/test_balance_core_validation_workflow.py` — add validator/aggregator/CLI orchestration coverage for arbitrary duration lists, stop-at-first-failure, continue-all, output routing, and summary generation.
- `tests/test_simulate_hierarchical_controller_telemetry.py` — add telemetry logging behavior tests for decimation flags, default behavior preservation, failure-window buffering, and summary metric accumulation.

**Keep unchanged:**
- `wheeled_biped/validation/failure_classifier.py`
- `wheeled_biped/validation/telemetry_schema_checker.py`
- `wheeled_biped/validation/structural_invariant_checker.py`

These files remain the same classifier/schema/invariant source of truth; the work should adapt inputs around them rather than duplicating or replacing them.

---

### Task 1: Extend the validator CLI for Step A orchestration

**Objective:** Make [scripts/validate_balance_core.py](scripts/validate_balance_core.py) the single Step A entry point for long-duration nominal validation while preserving current default behavior.

**Files:**
- Modify: `scripts/validate_balance_core.py`
- Test: `tests/test_balance_core_validation_workflow.py`

**Required behavior:**
- Support `python scripts/validate_balance_core.py --durations 10000,20000,50000,100000`
- Run durations in the order provided
- Stop at first failure by default
- Support `--continue-all`
- Route Step A outputs to `outputs/balance_core_extended_longevity/` when running extended nominal longevity mode unless user overrides `--output-dir`
- Preserve existing behavior for normal validation commands and root-z workflows
- Thread long-run logging flags to the validator without altering current defaults
- Keep the plan aligned with sidecar-backed simulated step counts so decimated CSV row counts never stand in for `actual_steps` or `survived_steps`

**Dependencies:** None

**Safety/Rollback notes:** Only CLI parsing and orchestration change. Do not touch controller code paths. Revert by removing newly added argparse flags and routing logic if needed.

- [ ] **Step 1.1: Add failing parser tests for long-run duration handling and output routing**

```python
# tests/test_balance_core_validation_workflow.py
from pathlib import Path
from scripts.validate_balance_core import _parse_durations


def test_parse_durations_accepts_extended_longevity_list():
    assert _parse_durations("10000,20000,50000,100000") == [10000, 20000, 50000, 100000]


def test_parse_durations_rejects_empty_list():
    import argparse
    import pytest

    with pytest.raises(argparse.ArgumentTypeError):
        _parse_durations(", ,")
```

- [ ] **Step 1.2: Add failing CLI orchestration tests for default stop-on-first-failure and continue-all**

```python
# tests/test_balance_core_validation_workflow.py
from pathlib import Path
from unittest.mock import patch


def test_cli_long_run_defaults_to_stop_on_first_failure(tmp_path):
    from scripts import validate_balance_core

    captured = {}

    class FakeValidator:
        def validate_ladder(self, output_dir, start_duration=None, durations=None, stop_on_first_failure=True, sim_args=None, long_run_options=None):
            captured["output_dir"] = output_dir
            captured["durations"] = list(durations)
            captured["stop_on_first_failure"] = stop_on_first_failure
            captured["long_run_options"] = long_run_options
            return []

    with patch.object(validate_balance_core, "BalanceCoreValidator", return_value=FakeValidator()):
        with patch.object(validate_balance_core.sys, "argv", [
            "validate_balance_core.py",
            "--durations", "10000,20000,50000,100000",
        ]):
            assert validate_balance_core.main() == 0

    assert captured["durations"] == [10000, 20000, 50000, 100000]
    assert captured["stop_on_first_failure"] is True
    assert Path(captured["output_dir"]).as_posix().endswith("outputs/balance_core_extended_longevity")


def test_cli_continue_all_disables_stop_on_first_failure(tmp_path):
    from scripts import validate_balance_core

    captured = {}

    class FakeValidator:
        def validate_ladder(self, output_dir, start_duration=None, durations=None, stop_on_first_failure=True, sim_args=None, long_run_options=None):
            captured["stop_on_first_failure"] = stop_on_first_failure
            return []

    with patch.object(validate_balance_core, "BalanceCoreValidator", return_value=FakeValidator()):
        with patch.object(validate_balance_core.sys, "argv", [
            "validate_balance_core.py",
            "--durations", "10000,20000,50000,100000",
            "--continue-all",
        ]):
            assert validate_balance_core.main() == 0

    assert captured["stop_on_first_failure"] is False
```

- [ ] **Step 1.3: Run the new targeted tests and confirm they fail before implementation**

Run:

```bash
pytest tests/test_balance_core_validation_workflow.py -q
```

Expected: FAIL because `validate_ladder()` does not accept `long_run_options`, and extended-longevity default output routing does not exist yet.

- [ ] **Step 1.4: Extend CLI parsing with explicit long-run logging flags**

Implement these flags in `scripts/validate_balance_core.py`:

```python
parser.add_argument(
    "--telemetry-decimation",
    type=int,
    help="Write every Nth telemetry row for long-run validation. Logging-only; default unchanged.",
)
parser.add_argument(
    "--failure-window-steps",
    type=int,
    default=500,
    help="Number of full-rate telemetry rows to preserve around failure in long-run mode.",
)
parser.add_argument(
    "--write-run-summary-sidecar",
    action="store_true",
    help="Write per-run summary sidecar with whole-run metrics for long-run validation.",
)
```

And add a small helper to build long-run options:

```python
def _build_long_run_options(args) -> dict | None:
    if (
        args.telemetry_decimation is None
        and args.failure_window_steps == 500
        and not args.write_run_summary_sidecar
    ):
        return None
    return {
        "telemetry_decimation": args.telemetry_decimation,
        "failure_window_steps": args.failure_window_steps,
        "write_run_summary_sidecar": args.write_run_summary_sidecar,
    }
```

- [ ] **Step 1.5: Add extended-longevity output directory routing without breaking existing modes**

Implement a helper in `scripts/validate_balance_core.py`:

```python
def _resolve_output_dir(args) -> Path:
    if args.output_dir != Path("outputs/balance_core_validation"):
        return args.output_dir
    if args.durations and any(duration >= 10000 for duration in args.durations):
        return Path("outputs/balance_core_extended_longevity")
    return args.output_dir
```

Use it in `main()` before constructing the validator call.

- [ ] **Step 1.6: Pass long-run options into the validator and keep current behavior intact**

Update the `validate_ladder()` call shape in `main()`:

```python
resolved_output_dir = _resolve_output_dir(args)
long_run_options = _build_long_run_options(args)
results = validator.validate_ladder(
    output_dir=str(resolved_output_dir),
    start_duration=args.start_duration,
    durations=args.durations,
    stop_on_first_failure=not args.continue_all,
    sim_args=sim_args,
    long_run_options=long_run_options,
)
```

- [ ] **Step 1.7: Add top-level Step A summary generation hook through the study aggregator**

After `validate_ladder()` returns, if running a duration list in extended longevity mode, build and write the summary:

```python
aggregator = StudyAggregator()
summary_output_dir = resolved_output_dir
aggregator.write_extended_longevity_summary(
    results=results,
    output_dir=summary_output_dir,
    required_max_duration=100000,
    command=" ".join(sys.argv),
)
```

- [ ] **Step 1.8: Re-run the CLI workflow tests**

Run:

```bash
pytest tests/test_balance_core_validation_workflow.py -q
```

Expected: the new CLI orchestration tests pass; later summary-related tests may still fail until Tasks 2–5 are complete.

**Acceptance criteria:**
- `--durations 10000,20000,50000,100000` works through the existing CLI
- `--continue-all` flips stop-on-first-failure semantics
- long-run flags are opt-in
- default output remains unchanged for non-Step-A usage
- Step A default output resolves to `outputs/balance_core_extended_longevity/`

---

### Task 2: Integrate long-run options and richer result fields into the validator

**Objective:** Extend [wheeled_biped/validation/balance_core_validator.py](wheeled_biped/validation/balance_core_validator.py) so it remains the single validator while carrying long-run logging options, richer per-duration artifact paths, and better classification source control.

**Files:**
- Modify: `wheeled_biped/validation/balance_core_validator.py`
- Test: `tests/test_balance_core_validation_workflow.py`

**Required behavior:**
- Accept arbitrary duration lists
- Pass long-run logging options to the simulator
- Preserve stop-at-first-failure semantics
- Preserve structural invariant and classifier behavior
- Expose richer fields: termination reason, summary sidecar path, failure-window path, classification source, summary metrics, artifact paths
- Ensure `ValidationResult.actual_steps` reflects simulated steps from summary sidecar data when available, never decimated CSV row counts

**Dependencies:** Task 1

**Safety/Rollback notes:** Validator remains orchestration-only. Do not modify the classifier thresholds or invariant checker semantics.

- [ ] **Step 2.1: Add failing tests for long-run option forwarding and enriched validation results**

```python
# tests/test_balance_core_validation_workflow.py

def test_long_run_options_are_forwarded_to_run_simulation(tmp_path):
    validator = BalanceCoreValidator()
    captured = {}

    def fake_run_simulation(steps, output_dir, sim_args=None, long_run_options=None):
        captured["steps"] = steps
        captured["long_run_options"] = dict(long_run_options or {})
        telemetry_path = Path(output_dir) / f"telemetry_{steps}.csv"
        df = TestBalanceCoreValidationWorkflow()._create_valid_telemetry(steps)
        df.to_csv(telemetry_path, index=False)
        return telemetry_path

    validator.run_simulation = fake_run_simulation
    validator.validate_ladder(
        output_dir=str(tmp_path),
        durations=[10000],
        long_run_options={"telemetry_decimation": 20, "failure_window_steps": 400},
    )

    assert captured["steps"] == 10000
    assert captured["long_run_options"]["telemetry_decimation"] == 20
    assert captured["long_run_options"]["failure_window_steps"] == 400
```

- [ ] **Step 2.2: Add failing test for classification source preferring failure-window telemetry when present and simulated-step counts preferring sidecar values**

```python
# tests/test_balance_core_validation_workflow.py
import json


def test_validate_duration_prefers_failure_window_for_classification(tmp_path):
    validator = BalanceCoreValidator()

    decimated = TestBalanceCoreValidationWorkflow()._create_valid_telemetry(1000)
    decimated_path = tmp_path / "telemetry_1000.csv"
    decimated.to_csv(decimated_path, index=False)

    failure_window = TestBalanceCoreValidationWorkflow()._create_valid_telemetry(200)
    failure_window.loc[150:, "pitch_x_rad"] = 0.35
    failure_window_path = tmp_path / "failure_window_1000.csv"
    failure_window.to_csv(failure_window_path, index=False)

    summary_sidecar_path = tmp_path / "telemetry_1000.summary.json"
    summary_sidecar_path.write_text(json.dumps({
        "requested_steps": 1000,
        "survived_steps": 1000,
        "actual_steps": 1000,
        "metric_integrity": {"source": "full_rate_online", "limitations": []},
    }), encoding="utf-8")

    result = validator.validate_duration(
        str(decimated_path),
        expected_steps=1000,
        failure_window_path=failure_window_path,
        summary_sidecar_path=summary_sidecar_path,
    )

    assert result.passed is False
    assert result.classification_result is not None
    assert result.classification_source == "failure_window"
    assert result.actual_steps == 1000
```

- [ ] **Step 2.3: Run validator workflow tests and confirm expected failure**

Run:

```bash
pytest tests/test_balance_core_validation_workflow.py -q
```

Expected: FAIL because the validator dataclass and methods do not accept the new parameters/fields yet.

- [ ] **Step 2.4: Extend `ValidationResult` with Step A fields**

Update the dataclass in `wheeled_biped/validation/balance_core_validator.py` to include:

```python
@dataclass
class ValidationResult:
    passed: bool
    duration_steps: int
    actual_steps: int
    structural_invariants_passed: bool
    failure_mode: Optional["FailureMode"]
    classification_result: Optional[ClassificationResult]
    telemetry_path: Path
    report_path: Optional[Path]
    termination_reason: Optional[str] = None
    failure_window_path: Optional[Path] = None
    summary_sidecar_path: Optional[Path] = None
    classification_source: str = "main_telemetry"
    summary_metrics: Optional[dict] = None
```

- [ ] **Step 2.5: Extend `run_simulation()` to accept `long_run_options` and forward logging-only flags**

Change the signature and append simulator flags only when explicitly set:

```python
def run_simulation(self, steps: int, output_dir: str, sim_args: Optional[Sequence[str]] = None, long_run_options: Optional[dict] = None) -> Path:
    ...
    if long_run_options:
        if long_run_options.get("telemetry_decimation") is not None:
            cmd.extend(["--telemetry-decimation", str(long_run_options["telemetry_decimation"])])
        if long_run_options.get("failure_window_steps") is not None:
            cmd.extend(["--failure-window-steps", str(long_run_options["failure_window_steps"])])
        if long_run_options.get("write_run_summary_sidecar"):
            cmd.append("--write-run-summary-sidecar")
```

- [ ] **Step 2.6: Extend `validate_duration()` to accept artifact paths, prefer sidecar-backed simulated step counts, and choose classification evidence source**

Add optional parameters:

```python
def validate_duration(self, telemetry_path: str, expected_steps: int, failure_window_path: Optional[Path] = None, summary_sidecar_path: Optional[Path] = None) -> ValidationResult:
```

Summary-sidecar loading and simulated-step selection logic:

```python
summary_metrics = None
actual_steps = len(df)
if summary_sidecar_path is not None and summary_sidecar_path.exists():
    summary_metrics = json.loads(summary_sidecar_path.read_text(encoding="utf-8"))
    actual_steps = int(
        summary_metrics.get(
            "actual_steps",
            summary_metrics.get("survived_steps", actual_steps),
        )
    )
```

Classification source logic:

```python
classification_df = df
classification_source = "main_telemetry"
if failure_window_path is not None and failure_window_path.exists():
    classification_df = pd.read_csv(failure_window_path)
    classification_source = "failure_window"
classification = self.failure_classifier.classify(classification_df)
```

Use `actual_steps` from the sidecar whenever available. Treat `len(df)` only as the count of written telemetry rows, not simulated steps, in decimated mode.

- [ ] **Step 2.7: Extend `validate_ladder()` to pass long-run options and collect artifact sidecars**

Update the signature:

```python
def validate_ladder(..., long_run_options: Optional[dict] = None) -> List[ValidationResult]:
```

When a simulation completes, derive predictable companion artifact paths inside the per-duration output directory:

```python
failure_window_path = Path(output_dir) / f"failure_window_{duration}.csv"
summary_sidecar_path = Path(output_dir) / f"telemetry_{duration}.summary.json"
result = self.validate_duration(
    str(telemetry_path),
    duration,
    failure_window_path=failure_window_path if failure_window_path.exists() else None,
    summary_sidecar_path=summary_sidecar_path if summary_sidecar_path.exists() else None,
)
```

- [ ] **Step 2.8: Re-run validator tests**

Run:

```bash
pytest tests/test_balance_core_validation_workflow.py -q
```

Expected: forwarding/classification-source tests pass; summary-generation assertions may still fail until Task 4 and Task 5 are complete.

**Acceptance criteria:**
- validator accepts arbitrary duration lists and long-run options
- `run_simulation()` forwards long-run flags only when explicitly set
- `validate_duration()` can classify from a failure-window artifact when present
- stop-at-first-failure and continue-all semantics remain intact
- result objects carry the artifact metadata needed by the study summary

---

### Task 3: Add logging-only decimation and failure-window preservation to the simulator

**Objective:** Extend [scripts/simulate_hierarchical_controller.py](scripts/simulate_hierarchical_controller.py) so long-run mode can decimate the main CSV, preserve a rolling full-rate failure window, and write whole-run summary metrics without touching control or simulation behavior.

**Files:**
- Modify: `scripts/simulate_hierarchical_controller.py`
- Test: `tests/test_simulate_hierarchical_controller_telemetry.py`

**Required behavior:**
- Default behavior unchanged when flags are omitted
- Decimation affects telemetry writing only
- No change to simulation timestep, controller timestep, controller inputs, torque computation, termination logic, balance-core stack, or ownership
- Support `--telemetry-decimation`, `--failure-window-steps`, and `--write-run-summary-sidecar`
- Write a full-rate failure-window CSV on failure when long-run mode is active
- Accumulate whole-run summary metrics over all steps

**Dependencies:** Task 1, Task 2

**Safety/Rollback notes:** This is the highest-sensitivity task because the simulator file is large. Keep all changes confined to telemetry storage/output plumbing. Do not edit any controller gain or torque-composition code.

- [ ] **Step 3.1: Add failing argument-parsing tests for new logging flags**

```python
# tests/test_simulate_hierarchical_controller_telemetry.py
from unittest.mock import patch


def test_main_accepts_long_run_logging_flags():
    from scripts import simulate_hierarchical_controller

    with patch.object(simulate_hierarchical_controller, "validate_balance_core_mode_args") as validate_args:
        with patch.object(simulate_hierarchical_controller.mujoco.MjModel, "from_xml_path", side_effect=RuntimeError("stop after parse")):
            with patch.object(simulate_hierarchical_controller.sys, "argv", [
                "simulate_hierarchical_controller.py",
                "--controller-mode", "balance-core",
                "--steps", "10000",
                "--telemetry-decimation", "20",
                "--failure-window-steps", "400",
                "--write-run-summary-sidecar",
            ]):
                try:
                    simulate_hierarchical_controller.main()
                except RuntimeError as exc:
                    assert str(exc) == "stop after parse"

    parsed_args = validate_args.call_args.args[0]
    assert parsed_args.telemetry_decimation == 20
    assert parsed_args.failure_window_steps == 400
    assert parsed_args.write_run_summary_sidecar is True
```

- [ ] **Step 3.2: Add failing unit tests for decimation behavior and default preservation**

```python
# tests/test_simulate_hierarchical_controller_telemetry.py

def test_decimation_keeps_every_nth_row():
    from scripts.simulate_hierarchical_controller import _select_rows_for_main_telemetry

    rows = [{"step": i} for i in range(10)]
    kept = _select_rows_for_main_telemetry(rows, telemetry_decimation=3)
    assert [row["step"] for row in kept] == [0, 3, 6, 9]


def test_decimation_none_preserves_all_rows():
    from scripts.simulate_hierarchical_controller import _select_rows_for_main_telemetry

    rows = [{"step": i} for i in range(5)]
    kept = _select_rows_for_main_telemetry(rows, telemetry_decimation=None)
    assert [row["step"] for row in kept] == [0, 1, 2, 3, 4]


def test_decimation_always_keeps_first_and_last_row():
    from scripts.simulate_hierarchical_controller import _select_rows_for_main_telemetry

    rows = [{"step": i} for i in range(10)]
    kept = _select_rows_for_main_telemetry(rows, telemetry_decimation=3)
    assert kept[0]["step"] == 0
    assert kept[-1]["step"] == 9
```

- [ ] **Step 3.3: Add failing unit test for rolling failure-window behavior**

```python
# tests/test_simulate_hierarchical_controller_telemetry.py

def test_failure_window_buffer_preserves_recent_full_rate_rows():
    from scripts.simulate_hierarchical_controller import _RollingFailureWindow

    window = _RollingFailureWindow(max_rows=4)
    for step in range(8):
        window.append({"step": step, "value": step})

    assert [row["step"] for row in window.rows()] == [4, 5, 6, 7]
```

- [ ] **Step 3.4: Run telemetry tests to confirm failure before implementation**

Run:

```bash
pytest tests/test_simulate_hierarchical_controller_telemetry.py -q
```

Expected: FAIL because helper functions/classes and argparse flags do not exist yet.

- [ ] **Step 3.5: Add explicit logging-only argparse flags and keep defaults inert**

In `scripts/simulate_hierarchical_controller.py`, add:

```python
parser.add_argument(
    "--telemetry-decimation",
    type=int,
    default=None,
    help="Write every Nth telemetry row to the main CSV. Logging-only; control/simulation behavior unchanged.",
)
parser.add_argument(
    "--failure-window-steps",
    type=int,
    default=500,
    help="Full-rate rolling telemetry window to preserve if termination occurs.",
)
parser.add_argument(
    "--write-run-summary-sidecar",
    action="store_true",
    default=False,
    help="Write whole-run summary metrics as a JSON sidecar for long-run validation.",
)
```

- [ ] **Step 3.6: Add minimal helper utilities for decimation and rolling failure windows**

Implement inside `scripts/simulate_hierarchical_controller.py` near telemetry helpers:

```python
from collections import deque


class _RollingFailureWindow:
    def __init__(self, max_rows: int):
        self._rows = deque(maxlen=max_rows)

    def append(self, row: dict) -> None:
        self._rows.append(dict(row))

    def rows(self) -> list[dict]:
        return list(self._rows)


def _select_rows_for_main_telemetry(rows: list[dict], telemetry_decimation: int | None) -> list[dict]:
    if telemetry_decimation is None or telemetry_decimation <= 1:
        return list(rows)
    selected = [row for index, row in enumerate(rows) if index % telemetry_decimation == 0]
    if rows and selected[-1] is not rows[-1]:
        selected.append(rows[-1])
    return selected
```

This keeps the first row, every Nth row, and the final row in the main CSV.
- [ ] **Step 3.7: Add whole-run summary accumulator helpers that operate on full-rate rows without retaining all full-rate telemetry**

Implement a small in-file accumulator with explicit metrics:

```python
class _RunSummaryAccumulator:
    def __init__(self):
        self.pitch_values = []
        self.roll_values = []
        self.com_z_values = []
        self.wheel_vel_mean_values = []
        self.contact_states = {}
        self.left_wheel_contact_all = True
        self.right_wheel_contact_all = True
        self.ownership_violation_count_max = 0
        self.hidden_torque_norm_max = 0.0
        self.tau_wbc_norm_max = 0.0
        self.torque_saturation_counts = None
        self.torque_rate_saturation_counts = None
        self.total_rows = 0

    def update(self, row: dict) -> None:
        ...

    def to_dict(self, terminated: bool, termination_reason: str | None, final_sim_time_s: float, requested_steps: int, telemetry_csv_path: str, failure_window_path: str | None) -> dict:
        ...
```

Use full-rate rows as they are generated, before decimation. In long-run mode, keep only decimated main-CSV rows plus the rolling failure-window buffer in memory; do not retain the complete full-rate row history unless `telemetry_decimation` is disabled.

- [ ] **Step 3.8: Capture one full-rate row dict per control step, feed accumulator and rolling buffer, and retain only the rows needed for the chosen logging mode**

Inside the telemetry-writing path, introduce a single `row_dict` per control step that is:
- appended to `_RollingFailureWindow`
- fed to `_RunSummaryAccumulator`
- appended to the in-memory main-row collection only if `args.telemetry_decimation` is `None` or the row is selected for the decimated main CSV
- force-appended to the main-row collection for the final row and any termination row if not already selected

Do **not** change control stepping or termination evaluation. In long-run mode, do not keep an `all_rows` copy of the full-rate telemetry stream in memory.

- [ ] **Step 3.9: Emit the decimated main CSV, optional failure-window CSV, and optional summary sidecar**

At write-out time:

```python
_write_rows_to_csv(main_csv_path, main_rows, fieldnames)

if terminated and args.telemetry_decimation not in (None, 0, 1):
    failure_rows = failure_window.rows()
    if failure_rows:
        if failure_rows[-1] != row_dict:
            failure_rows.append(dict(row_dict))
        _write_rows_to_csv(failure_window_path, failure_rows, fieldnames)

if args.write_run_summary_sidecar:
    summary_payload = summary_accumulator.to_dict(...)
    summary_sidecar_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
```

The main CSV must always contain the first row, every Nth row, the final row, and the termination row if terminated.
- [ ] **Step 3.10: Re-run telemetry behavior tests**

Run:

```bash
pytest tests/test_simulate_hierarchical_controller_telemetry.py -q
```

Expected: parsing, decimation, and failure-window tests pass.

**Acceptance criteria:**
- new logging flags are opt-in
- default simulator behavior remains unchanged when flags are omitted
- main telemetry CSV can be decimated without affecting simulation/control behavior
- a full-rate failure window is preserved on termination in long-run mode
- a whole-run summary sidecar can be emitted from full-rate accumulation

---

### Task 4: Compute whole-run metrics and mark limitations explicitly

**Objective:** Ensure required summary metrics are computed from all steps when possible, not merely from decimated rows, and are marked as approximate if any metric must fall back to reduced evidence.

**Files:**
- Modify: `scripts/simulate_hierarchical_controller.py`
- Modify: `wheeled_biped/validation/balance_core_validator.py`
- Modify: `wheeled_biped/validation/study_aggregator.py`
- Test: `tests/test_balance_core_validation_workflow.py`
- Test: `tests/test_simulate_hierarchical_controller_telemetry.py`

**Required behavior:**
- Whole-run metrics include requested/survived steps, actual simulated steps, termination, sim time, maxima/minima/RMS, trend, contact validity, saturation percentages, ownership violation max, hidden torque max, WBC norm max, and artifact paths
- Prefer full-rate online accumulation
- Structural invariant checks must prefer full-rate sidecar maxima when available for ownership violations, hidden torque, WBC norm, and saturation summaries
- If a sidecar is missing, mark metric integrity limitations explicitly rather than silently trusting decimated CSV maxima

**Dependencies:** Task 2, Task 3

**Safety/Rollback notes:** Metrics/reporting only. Do not alter pass/fail thresholds or controller state.

- [ ] **Step 4.1: Add failing tests for whole-run summary metrics and invariant maxima sourced from sidecar data rather than decimated telemetry**

```python
# tests/test_balance_core_validation_workflow.py
import json


def test_study_aggregator_prefers_summary_sidecar_metrics(tmp_path):
    aggregator = StudyAggregator()
    df = TestBalanceCoreValidationWorkflow()._create_valid_telemetry(3)
    telemetry_path = tmp_path / "telemetry.csv"
    df.to_csv(telemetry_path, index=False)

    summary_sidecar = tmp_path / "telemetry.summary.json"
    summary_sidecar.write_text(json.dumps({
        "requested_steps": 10000,
        "survived_steps": 10000,
        "actual_steps": 10000,
        "terminated": False,
        "termination_reason": None,
        "final_sim_time_s": 100.0,
        "pitch_x_rad_min": -0.04,
        "pitch_x_rad_max": 0.05,
        "pitch_x_rad_rms": 0.02,
        "roll_y_rad_min": -0.03,
        "roll_y_rad_max": 0.04,
        "roll_y_rad_rms": 0.01,
        "com_z_m_min": 0.43,
        "com_z_m_max": 0.46,
        "com_z_m_drift": -0.01,
        "wheel_vel_mean_rad_s_min": -1.0,
        "wheel_vel_mean_rad_s_max": 2.0,
        "wheel_vel_mean_rad_s_rms": 0.8,
        "wheel_velocity_trend": "stable",
        "contact_state_summary": {"DOUBLE_CONTACT": 10000},
        "left_wheel_contact_validity": True,
        "right_wheel_contact_validity": True,
        "ownership_violation_count_max": 0,
        "hidden_torque_norm_max": 0.0,
        "tau_wbc_norm_max": 0.0,
        "torque_saturation_percentage_per_joint": [0.0] * 10,
        "torque_rate_saturation_percentage_per_joint": [0.0] * 10,
        "metric_integrity": {"source": "full_rate_online", "limitations": []},
    }), encoding="utf-8")

    result = aggregator.evaluate_case_from_telemetry(
        case_id="longevity_10000",
        height_test_type="longevity",
        duration_steps=10000,
        telemetry_path=telemetry_path,
        summary_sidecar_path=summary_sidecar,
    )

    assert result.summary_metrics["pitch_x_rad_max"] == 0.05
    assert result.summary_metrics["actual_steps"] == 10000
    assert result.summary_metrics["metric_integrity"]["source"] == "full_rate_online"
```

- [ ] **Step 4.2: Add failing test for explicit limitation marking when only decimated evidence exists**

```python
# tests/test_balance_core_validation_workflow.py

def test_summary_marks_metrics_as_limited_when_sidecar_missing(tmp_path):
    aggregator = StudyAggregator()
    df = TestBalanceCoreValidationWorkflow()._create_valid_telemetry(100)
    telemetry_path = tmp_path / "telemetry.csv"
    df.to_csv(telemetry_path, index=False)

    result = aggregator.evaluate_case_from_telemetry(
        case_id="longevity_100",
        height_test_type="longevity",
        duration_steps=100,
        telemetry_path=telemetry_path,
    )

    assert "metric_integrity" in result.summary_metrics
    assert result.summary_metrics["metric_integrity"]["source"] in {"telemetry_csv", "full_rate_online"}
    assert result.summary_metrics["metric_integrity"]["limitations"]
```

- [ ] **Step 4.3: Run summary metric tests and confirm failure before implementation**

Run:

```bash
pytest tests/test_balance_core_validation_workflow.py -q
```

Expected: FAIL because `summary_sidecar_path` is not supported by the aggregator yet.

- [ ] **Step 4.4: Extend simulator summary sidecar payload to include all required Step A fields**

Have `_RunSummaryAccumulator.to_dict()` emit at least:

```python
{
    "requested_steps": requested_steps,
    "survived_steps": survived_steps,
    "actual_steps": survived_steps,
    "pass_fail": "PASS" if not terminated and survived_steps >= requested_steps else "FAIL",
    "terminated": terminated,
    "termination_reason": termination_reason,
    "final_sim_time_s": final_sim_time_s,
    "pitch_x_rad_min": ...,
    "pitch_x_rad_max": ...,
    "pitch_x_rad_rms": ...,
    "roll_y_rad_min": ...,
    "roll_y_rad_max": ...,
    "roll_y_rad_rms": ...,
    "com_z_m_min": ...,
    "com_z_m_max": ...,
    "com_z_m_drift": ...,
    "wheel_vel_mean_rad_s_min": ...,
    "wheel_vel_mean_rad_s_max": ...,
    "wheel_vel_mean_rad_s_rms": ...,
    "wheel_velocity_trend": ...,
    "contact_state_summary": ...,
    "left_wheel_contact_validity": ...,
    "right_wheel_contact_validity": ...,
    "ownership_violation_count_max": ...,
    "hidden_torque_norm_max": ...,
    "tau_wbc_norm_max": ...,
    "torque_saturation_percentage_per_joint": ...,
    "torque_rate_saturation_percentage_per_joint": ...,
    "telemetry_csv_path": telemetry_csv_path,
    "failure_window_path": failure_window_path,
    "metric_integrity": {"source": "full_rate_online", "limitations": []},
}
```

This sidecar is the authoritative source for simulated step counts in decimated mode.
- [ ] **Step 4.5: Teach the validator to load summary sidecars into `ValidationResult.summary_metrics` and use sidecar maxima for invariant-sensitive fields**

Inside `validate_duration()`:

```python
summary_metrics = None
if summary_sidecar_path is not None and summary_sidecar_path.exists():
    summary_metrics = json.loads(summary_sidecar_path.read_text(encoding="utf-8"))
```

When `summary_metrics` is present, prefer its full-rate values for:
- `ownership_violation_count_max`
- `hidden_torque_norm_max`
- `tau_wbc_norm_max`
- torque saturation / torque-rate saturation summaries if the invariant path needs them

If no sidecar is present, keep CSV-based schema/sample validation but record a metric-integrity limitation instead of implying full-rate certainty.

Then attach `summary_metrics=summary_metrics` to every returned `ValidationResult`.

- [ ] **Step 4.6: Extend the study aggregator to prefer sidecar metrics and fall back safely**

Modify `evaluate_case_from_telemetry()` and `_build_summary_metrics()` in `wheeled_biped/validation/study_aggregator.py`:

```python
def evaluate_case_from_telemetry(..., summary_sidecar_path: str | Path | None = None):
    ...
    summary_metrics = self._build_summary_metrics(df, duration_steps, summary_sidecar_path=summary_sidecar_path)
```

And inside `_build_summary_metrics()`:

```python
if summary_sidecar_path is not None and Path(summary_sidecar_path).exists():
    payload = json.loads(Path(summary_sidecar_path).read_text(encoding="utf-8"))
    payload.setdefault("metric_integrity", {"source": "full_rate_online", "limitations": []})
    return payload
```

Fallback payloads must set `metric_integrity.source = "telemetry_csv"`, add explicit limitations, and must not imply that decimated CSV maxima alone prove the full-run invariant-sensitive metrics.

- [ ] **Step 4.7: Re-run summary metric tests**

Run:

```bash
pytest tests/test_balance_core_validation_workflow.py -q
```

Expected: sidecar-preference and limitation-marking tests pass.

**Acceptance criteria:**
- required Step A metrics can be emitted from all-step accumulation
- summary payloads clearly indicate metric integrity/source
- fallback to CSV-derived metrics is explicit, not silent
- no whole-run claim is made from decimated rows without disclosure

---

### Task 5: Generate the extended longevity study summary and per-duration rows

**Objective:** Extend [wheeled_biped/validation/study_aggregator.py](wheeled_biped/validation/study_aggregator.py) to produce `extended_longevity_summary.json` and `extended_longevity_summary.md` with all Step A fields, verdicts, artifact paths, and no-controller-change confirmations.

**Files:**
- Modify: `wheeled_biped/validation/study_aggregator.py`
- Modify: `wheeled_biped/validation/balance_core_validator.py`
- Modify: `scripts/validate_balance_core.py`
- Test: `tests/test_balance_core_validation_workflow.py`

**Required behavior:**
- Write top-level summary files under `outputs/balance_core_extended_longevity/`
- Include maximum confirmed survival steps, whether failure occurred before 100000, first failing duration, primary failure mode, all per-duration rows, artifact paths, and fixed declarations about unchanged controller/gains/WBC/ownership/stack
- Emit `long_duration_survival_passed_up_to_100000_steps` if all required durations pass through 100000
- Do not claim infinite stability

**Dependencies:** Task 1, Task 2, Task 4

**Safety/Rollback notes:** Reporting only. Keep known-study summary support intact.

- [ ] **Step 5.1: Add failing tests for extended summary generation**

```python
# tests/test_balance_core_validation_workflow.py

def test_write_extended_longevity_summary_creates_json_and_markdown(tmp_path):
    aggregator = StudyAggregator()
    results = [
        ValidationResult(
            passed=True,
            duration_steps=10000,
            actual_steps=10000,
            structural_invariants_passed=True,
            failure_mode=None,
            classification_result=None,
            telemetry_path=tmp_path / "telemetry_10000.csv",
            report_path=None,
            termination_reason=None,
            classification_source="main_telemetry",
            summary_metrics={
                "requested_steps": 10000,
                "survived_steps": 10000,
                "pass_fail": "PASS",
                "terminated": False,
                "telemetry_csv_path": str(tmp_path / "telemetry_10000.csv"),
                "metric_integrity": {"source": "full_rate_online", "limitations": []},
            },
        )
    ]

    aggregator.write_extended_longevity_summary(
        results=results,
        output_dir=tmp_path,
        required_max_duration=100000,
        command="python scripts/validate_balance_core.py --durations 10000,20000,50000,100000",
    )

    assert (tmp_path / "extended_longevity_summary.json").exists()
    assert (tmp_path / "extended_longevity_summary.md").exists()
```

- [ ] **Step 5.2: Add failing test for success conclusion string**

```python
# tests/test_balance_core_validation_workflow.py
import json


def test_extended_longevity_summary_reports_success_up_to_100000(tmp_path):
    aggregator = StudyAggregator()
    results = []
    for duration in [10000, 20000, 50000, 100000]:
        results.append(
            ValidationResult(
                passed=True,
                duration_steps=duration,
                actual_steps=duration,
                structural_invariants_passed=True,
                failure_mode=None,
                classification_result=None,
                telemetry_path=tmp_path / f"telemetry_{duration}.csv",
                report_path=None,
                summary_metrics={
                    "requested_steps": duration,
                    "survived_steps": duration,
                    "pass_fail": "PASS",
                    "terminated": False,
                    "telemetry_csv_path": str(tmp_path / f"telemetry_{duration}.csv"),
                    "metric_integrity": {"source": "full_rate_online", "limitations": []},
                },
            )
        )

    aggregator.write_extended_longevity_summary(results, tmp_path, 100000, "cmd")
    payload = json.loads((tmp_path / "extended_longevity_summary.json").read_text(encoding="utf-8"))
    assert payload["conclusion"] == "long_duration_survival_passed_up_to_100000_steps"
```

- [ ] **Step 5.3: Run workflow tests and confirm failure before implementation**

Run:

```bash
pytest tests/test_balance_core_validation_workflow.py -q
```

Expected: FAIL because `write_extended_longevity_summary()` does not exist yet.

- [ ] **Step 5.4: Add a dedicated `write_extended_longevity_summary()` method to the study aggregator**

Implement in `wheeled_biped/validation/study_aggregator.py`:

```python
def write_extended_longevity_summary(self, results, output_dir, required_max_duration, command):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = self.build_extended_longevity_payload(results, required_max_duration, command)
    (output_dir / "extended_longevity_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "extended_longevity_summary.md").write_text(self._build_extended_longevity_markdown(payload), encoding="utf-8")
```

- [ ] **Step 5.5: Build the extended longevity payload with explicit Step A declarations**

Add:

```python
def build_extended_longevity_payload(self, results, required_max_duration, command):
    per_duration = [self._validation_result_to_extended_row(result) for result in results]
    passed_durations = [r.duration_steps for r in results if r.passed]
    first_failure = next((r for r in results if not r.passed), None)
    max_confirmed = max(passed_durations, default=0)
    success = required_max_duration in passed_durations and all(
        any(r.duration_steps == duration and r.passed for r in results)
        for duration in [10000, 20000, 50000, 100000]
    )
    conclusion = "long_duration_survival_passed_up_to_100000_steps" if success else "long_duration_survival_incomplete"
    return {
        "command": command,
        "output_directory": "outputs/balance_core_extended_longevity",
        "maximum_confirmed_survival_steps": max_confirmed,
        "failure_occurred_before_100000": first_failure is not None and first_failure.duration_steps < 100000,
        "first_failing_duration": first_failure.duration_steps if first_failure else None,
        "primary_failure_mode": first_failure.failure_mode.value if first_failure and first_failure.failure_mode else None,
        "conclusion": conclusion,
        "durations": per_duration,
        "controller_change_confirmation": False,
        "gains_tuned_confirmation": False,
        "wbc_reintroduced_confirmation": False,
        "legacy_torque_activated_confirmation": False,
        "torque_ownership_changed_confirmation": False,
        "four_source_stack_changed_confirmation": False,
        "declarations": {
            "no_controller_behavior_modified": True,
            "no_gains_tuned": True,
            "wbc_remained_off": True,
            "no_legacy_torque_source_activated": True,
            "torque_ownership_unchanged": True,
            "four_source_balance_core_stack_unchanged": True,
        },
    }
```

- [ ] **Step 5.6: Add markdown rendering for top-level summary and per-duration rows**

Create `_build_extended_longevity_markdown(payload)` with sections:
- Overview
- Verdict
- Per-duration table
- Artifact paths
- Required declarations
- Limitation notes if any metric integrity entries contain limitations

- [ ] **Step 5.7: Re-run summary generation tests**

Run:

```bash
pytest tests/test_balance_core_validation_workflow.py -q
```

Expected: JSON and markdown summary generation tests pass.

**Acceptance criteria:**
- both top-level summary files are generated
- verdict fields match the spec
- per-duration rows and artifact paths are included
- unchanged-controller/gains/WBC/ownership/stack declarations are present
- success string is exactly `long_duration_survival_passed_up_to_100000_steps` only when all required durations pass

---

### Task 6: Preserve failure reports and classify from the best available evidence

**Objective:** Ensure long-run failures preserve both the existing failure report and the best classification evidence, preferring full-rate failure windows when available.

**Files:**
- Modify: `wheeled_biped/validation/balance_core_validator.py`
- Modify: `wheeled_biped/validation/classification_report.py`
- Test: `tests/test_balance_core_validation_workflow.py`

**Required behavior:**
- If a duration fails, stop at that duration unless `--continue-all` is set
- Preserve failure report path
- Preserve failure-window telemetry path
- Classify through the existing temporal root-cause classifier
- Include next allowed fix scope from the classifier output
- Do not implement any controller fix

**Dependencies:** Task 2, Task 3

**Safety/Rollback notes:** Classification logic must stay in the existing classifier. Only evidence selection and report formatting are adjusted.

- [ ] **Step 6.1: Add failing test that failure reports include classification source and fix scope**

```python
# tests/test_balance_core_validation_workflow.py

def test_failure_report_includes_fix_scope_and_classification_source(tmp_path):
    validator = BalanceCoreValidator()
    failure_df = TestBalanceCoreValidationWorkflow()._create_valid_telemetry(200)
    failure_df.loc[150:, "pitch_x_rad"] = 0.35
    telemetry_path = tmp_path / "telemetry.csv"
    failure_df.to_csv(telemetry_path, index=False)

    result = validator.validate_duration(str(telemetry_path), expected_steps=200)
    assert result.report_path is not None
    report_text = result.report_path.read_text(encoding="utf-8")
    assert "Recommended Fix Scope" in report_text
```

- [ ] **Step 6.2: Run failure-report tests and confirm current gaps**

Run:

```bash
pytest tests/test_balance_core_validation_workflow.py -q
```

Expected: this may already partially pass for fix scope, but will not yet verify classification-source behavior consistently.

- [ ] **Step 6.3: Annotate reports with evidence provenance if needed**

If the existing report format lacks clarity, prepend a short provenance line in `classification_report.py` callers rather than rewriting the classifier:

```python
report = self.report_generator.to_markdown(classification)
if classification_source != "main_telemetry":
    report = f"**Classification telemetry source:** {classification_source}\n\n" + report
```

- [ ] **Step 6.4: Ensure validator stop behavior and artifact capture align**

In `validate_ladder()`, on failure:
- keep the `ValidationResult` in `results`
- preserve `report_path`, `failure_window_path`, and `summary_sidecar_path`
- break only if `stop_on_first_failure` is `True`

- [ ] **Step 6.5: Re-run validator workflow tests**

Run:

```bash
pytest tests/test_balance_core_validation_workflow.py -q
```

Expected: failure classification/reporting tests pass.

**Acceptance criteria:**
- failure reports are preserved
- failure-window artifact paths are preserved on long-run failures
- classification uses the best available evidence without changing classifier logic
- next allowed fix scope is present in outputs
- no controller fix is implemented as part of Step A

---

### Task 7: Expand tests for output routing, metric integrity, and unchanged default behavior

**Objective:** Finish the test surface required by the spec and verify that long-run mode is opt-in, logging-only, and compatible with the existing validator/invariant pipeline.

**Files:**
- Modify: `tests/test_balance_core_validation_workflow.py`
- Modify: `tests/test_simulate_hierarchical_controller_telemetry.py`

**Required behavior:**
- Cover arbitrary durations
- Cover stop-at-first-failure
- Cover continue-all
- Cover output directory routing
- Cover summary generation
- Cover schema preservation under decimation
- Cover structural invariant compatibility under decimation
- Cover failure-window preservation
- Cover whole-run metric behavior
- Cover unchanged default behavior when long-run flags are omitted
- Cover no controller behavior change required for long-run validation mode

**Dependencies:** Tasks 1–6

**Safety/Rollback notes:** Tests only.

- [ ] **Step 7.1: Add schema-preservation test for decimated telemetry**

```python
# tests/test_balance_core_validation_workflow.py

def test_decimated_telemetry_preserves_required_schema(tmp_path):
    validator = BalanceCoreValidator()
    df = TestBalanceCoreValidationWorkflow()._create_valid_telemetry(100)
    telemetry_path = tmp_path / "telemetry.csv"
    df.iloc[::10].to_csv(telemetry_path, index=False)
    result = validator.validate_duration(str(telemetry_path), expected_steps=10)
    assert result.structural_invariants_passed is True
```

- [ ] **Step 7.2: Add explicit unchanged-default-behavior test for simulator logging flags omitted**

```python
# tests/test_simulate_hierarchical_controller_telemetry.py

def test_default_logging_behavior_uses_full_rate_when_no_long_run_flags():
    from scripts.simulate_hierarchical_controller import _select_rows_for_main_telemetry

    rows = [{"step": i} for i in range(6)]
    kept = _select_rows_for_main_telemetry(rows, telemetry_decimation=None)
    assert len(kept) == 6
```

- [ ] **Step 7.3: Add whole-run metric behavior test for saturation percentages**

```python
# tests/test_simulate_hierarchical_controller_telemetry.py

def test_summary_accumulator_tracks_per_joint_saturation_percentages():
    from scripts.simulate_hierarchical_controller import _RunSummaryAccumulator

    acc = _RunSummaryAccumulator()
    for step in range(4):
        acc.update({
            "pitch_x_rad": 0.0,
            "roll_y_rad": 0.0,
            "com_z_m": 0.45,
            "wheel_vel_mean_rad_s": 0.0,
            "contact_supervisor_state": "DOUBLE_CONTACT",
            "left_wheel_contact": True,
            "right_wheel_contact": True,
            "ownership_violation_count": 0,
            "hidden_torque_norm": 0.0,
            "tau_wbc_norm": 0.0,
            "torque_saturation_mask_per_joint": "true,false,false,false,false,false,false,false,false,false",
            "torque_rate_saturation_mask_per_joint": "false,false,false,false,false,false,false,false,false,false",
        })

    payload = acc.to_dict(
        terminated=False,
        termination_reason=None,
        final_sim_time_s=0.04,
        requested_steps=4,
        telemetry_csv_path="telemetry.csv",
        failure_window_path=None,
    )
    assert payload["torque_saturation_percentage_per_joint"][0] == 100.0
```

- [ ] **Step 7.4: Run the full targeted test set**

Run:

```bash
pytest tests/test_balance_core_validation_workflow.py -q
pytest tests/test_simulate_hierarchical_controller_telemetry.py -q
```

Expected: PASS

- [ ] **Step 7.5: Run broader validation suite smoke test**

Run:

```bash
pytest tests -q
```

Expected: PASS, or if unrelated pre-existing failures exist, document them explicitly and confirm the new tests pass.

**Acceptance criteria:**
- test suite covers every spec requirement for Step A infrastructure
- decimation remains schema-compatible
- default behavior remains unchanged when long-run flags are omitted
- no tests imply or require controller/gain/ownership changes

---

### Task 8: Verify the end-to-end Step A workflow with exact commands

**Objective:** Run the exact verification commands required by the spec and confirm artifact generation and safety declarations before any claim of completion.

**Files:**
- No code changes expected
- Artifacts under: `outputs/balance_core_extended_longevity/`

**Required behavior:**
- Run targeted tests
- Run smoke validation
- Run extended longevity command
- Optionally run continue-all
- Check generated files and per-duration artifacts

**Dependencies:** Tasks 1–7

**Safety/Rollback notes:** This task executes validation only. It must not modify controllers or gains.

- [ ] **Step 8.1: Run the targeted validator workflow tests**

Run:

```bash
pytest tests/test_balance_core_validation_workflow.py -q
```

Expected: PASS

- [ ] **Step 8.2: Run the targeted simulator telemetry tests**

Run:

```bash
pytest tests/test_simulate_hierarchical_controller_telemetry.py -q
```

Expected: PASS

- [ ] **Step 8.3: Run the broader test suite**

Run:

```bash
pytest tests -q
```

Expected: PASS, or documented unrelated pre-existing failures only.

- [ ] **Step 8.4: Run smoke validation on a short nominal duration**

Run:

```bash
python scripts/validate_balance_core.py --single-duration 1000
```

Expected: existing short-run behavior still works; no long-run output routing unless explicitly requested.

- [ ] **Step 8.5: Run the Step A extended longevity command**

Run:

```bash
python scripts/validate_balance_core.py --durations 10000,20000,50000,100000 --telemetry-decimation 20 --failure-window-steps 500 --write-run-summary-sidecar
```

Expected:
- outputs under `outputs/balance_core_extended_longevity/`
- per-duration subdirectories and telemetry artifacts
- top-level `extended_longevity_summary.json`
- top-level `extended_longevity_summary.md`
- stop at first failure unless all pass

- [ ] **Step 8.6: Optionally run continue-all mode**

Run:

```bash
python scripts/validate_balance_core.py --durations 10000,20000,50000,100000 --continue-all --telemetry-decimation 20 --failure-window-steps 500 --write-run-summary-sidecar
```

Expected: all requested durations run, with failures preserved in per-duration artifacts.

- [ ] **Step 8.7: Verify the top-level summary payload fields explicitly**

Check that `extended_longevity_summary.json` includes:
- `maximum_confirmed_survival_steps`
- `failure_occurred_before_100000`
- `first_failing_duration`
- `primary_failure_mode`
- `durations`
- `declarations`
- exact success string `long_duration_survival_passed_up_to_100000_steps` only if all required durations pass

- [ ] **Step 8.8: Record final no-change confirmations**

Before marking Step A complete, confirm in the final report:
- no controller behavior was changed
- no gains were tuned
- WBC remains off
- no legacy torque source was activated
- torque ownership remains unchanged
- four-source balance-core stack remains unchanged

**Acceptance criteria:**
- arbitrary duration list runs work
- stop-at-first-failure works
- continue-all works
- telemetry decimation is opt-in and logging-only
- default telemetry behavior remains unchanged
- failure-window preservation works
- both top-level summary files are generated
- per-duration artifacts are generated
- required metrics are reported
- tests pass
- no controller/gain/WBC/ownership/stack changes occurred

---

## Spec Coverage Check

This plan covers:
- CLI and duration orchestration — Task 1
- Validator long-run integration — Task 2
- Simulator logging-only decimation — Task 3
- Whole-run summary metrics — Task 4
- Study summary generation — Task 5
- Failure-window preservation and classification — Task 6
- Tests — Task 7
- Validation commands and final acceptance — Task 8

No task modifies controller behavior, controller gains, WBC usage, torque ownership, or the four-source balance-core torque stack.

---

## Acceptance Criteria Summary

Step A is complete only when all of the following are true:
- arbitrary duration list runs work
- stop-at-first-failure works
- `--continue-all` works
- telemetry decimation is opt-in and logging-only
- default telemetry behavior remains unchanged
- failure-window preservation works
- `extended_longevity_summary.json` is generated
- `extended_longevity_summary.md` is generated
- per-duration artifacts are generated
- required metrics are reported
- tests pass
- no controller behavior was changed
- no gains were tuned
- WBC remains off
- no legacy torque source is activated
- torque ownership remains unchanged
- four-source balance-core stack remains unchanged
