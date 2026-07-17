import importlib
import pathlib
import csv
import sys


def test_run_step_d_all(monkeypatch, tmp_path):
    # Import the runner module and modify PUSH_CASES
    runner = importlib.import_module('scripts.run_outer_loop_step_d_push')
    dummy_case = [("TEST0", "low_0p330", 10, 0, 5, 150)]
    monkeypatch.setattr(runner, 'PUSH_CASES', dummy_case, raising=False)

    # Mock run_sim to avoid real simulation
    def fake_run_sim(label, steps, profile, out_dir, push_magnitude=0, push_duration=5, push_interval=150, push_sagittal=True):
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tel = out_dir / "telemetry_dummy.csv"
        tel.write_text("col\n1\n", encoding='utf-8')
        return tel, None
    monkeypatch.setattr(runner, 'run_sim', fake_run_sim, raising=False)

    # Mock analyze/safety_ok/fmt to avoid depending on real telemetry columns
    monkeypatch.setattr(runner, 'analyze', lambda path: {"metric": 42}, raising=False)
    monkeypatch.setattr(runner, 'safety_ok', lambda metrics: (True, 'safe'), raising=False)
    monkeypatch.setattr(runner, 'fmt', lambda metrics: 'ok', raising=False)

    # Import wrapper and redirect its output directory to tmp_path
    wrapper = importlib.import_module('scripts.run_step_d_all')
    monkeypatch.setattr(wrapper, 'OUT_BASE', tmp_path / 'step_d_all', raising=False)
    wrapper.main()

    # Verify the combined CSV exists and has at least one data row
    csv_path = wrapper.OUT_BASE / 'step_d_all_metrics.csv'
    assert csv_path.is_file(), f"CSV file not found at {csv_path}"
    with open(csv_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 1, "CSV should contain at least one data row"
