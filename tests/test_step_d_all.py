import importlib
import pathlib
import csv


def test_run_step_d_all(monkeypatch):
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

    # Mock analyze to return a simple metric dict
    monkeypatch.setattr(runner, 'analyze', lambda path: {"metric": 42}, raising=False)

    # Import and run the wrapper script
    wrapper = importlib.import_module('scripts.run_step_d_all')
    wrapper.main()

    # Verify the combined CSV exists and has at least one data row
    repo_root = pathlib.Path(__file__).parents[2]
    csv_path = repo_root / 'outputs' / 'step_d_all' / 'step_d_all_metrics.csv'
    assert csv_path.is_file(), f"CSV file not found at {csv_path}"
    with open(csv_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 1, "CSV should contain at least one data row"
