"""
Tests for wheeled_biped.eval.latex_table — LaTeX table generation.

No external dependencies beyond pytest.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wheeled_biped.eval.latex_table import (  # noqa: E402
    DEFAULT_COLUMNS,
    generate_latex_table,
    _fmt,
)


# ---------------------------------------------------------------------------
# _fmt helper
# ---------------------------------------------------------------------------


class TestFmt:
    def test_none_returns_dash(self):
        assert _fmt(None, ".2f", False) == "---"

    def test_nan_returns_dash(self):
        assert _fmt(float("nan"), ".2f", False) == "---"

    def test_inf_returns_dash(self):
        assert _fmt(float("inf"), ".2f", False) == "---"

    def test_float_formatting(self):
        assert _fmt(3.14159, ".2f", False) == "3.14"

    def test_percent_formatting(self):
        # 0.15 → 15 (with ".0f")
        assert _fmt(0.15, ".0f", True) == "15"

    def test_string_escapes_underscores(self):
        assert _fmt("push_sweep_60N", None, False) == r"push\_sweep\_60N"

    def test_int_formatting(self):
        assert _fmt(42, None, False) == "42"


# ---------------------------------------------------------------------------
# generate_latex_table
# ---------------------------------------------------------------------------


class TestGenerateLatexTable:
    def _sample_results(self) -> list[dict]:
        return [
            {
                "scenario": "nominal",
                "survival_time_mean_s": 20.0,
                "fall_rate": 0.0,
                "pitch_rms_deg": 1.23,
                "roll_rms_deg": 0.56,
                "height_rmse_m": 0.012,
                "torque_rms_nm": 3.45,
                "max_recoverable_push_n": float("nan"),
            },
            {
                "scenario": "push_recovery",
                "survival_time_mean_s": 17.2,
                "fall_rate": 0.15,
                "pitch_rms_deg": 3.67,
                "roll_rms_deg": 1.12,
                "height_rmse_m": 0.021,
                "torque_rms_nm": 5.23,
                "max_recoverable_push_n": 95.0,
            },
        ]

    def test_contains_toprule(self):
        tex = generate_latex_table(self._sample_results())
        assert r"\toprule" in tex

    def test_contains_bottomrule(self):
        tex = generate_latex_table(self._sample_results())
        assert r"\bottomrule" in tex

    def test_contains_midrule(self):
        tex = generate_latex_table(self._sample_results())
        assert r"\midrule" in tex

    def test_contains_begin_table(self):
        tex = generate_latex_table(self._sample_results())
        assert r"\begin{table}" in tex
        assert r"\end{table}" in tex

    def test_data_row_count(self):
        results = self._sample_results()
        tex = generate_latex_table(results)
        # Count lines with \\ that are between midrule and bottomrule
        lines = tex.split("\n")
        midrule_idx = next(i for i, l in enumerate(lines) if r"\midrule" in l)
        bottomrule_idx = next(i for i, l in enumerate(lines) if r"\bottomrule" in l)
        data_lines = [l for l in lines[midrule_idx + 1 : bottomrule_idx] if r"\\" in l]
        assert len(data_lines) == len(results)

    def test_empty_results(self):
        tex = generate_latex_table([])
        assert "No results" in tex
        # Should not crash
        assert isinstance(tex, str)

    def test_nan_renders_as_dash(self):
        tex = generate_latex_table(self._sample_results())
        # The nominal row has NaN for max_recoverable_push_n
        assert "---" in tex

    def test_custom_caption_and_label(self):
        tex = generate_latex_table(
            self._sample_results(),
            caption="My custom caption.",
            label="tab:custom",
        )
        assert "My custom caption." in tex
        assert "tab:custom" in tex

    def test_column_count_matches(self):
        """Each data row should have (n_cols - 1) ampersands."""
        results = self._sample_results()
        n_cols = len(DEFAULT_COLUMNS)
        tex = generate_latex_table(results)
        lines = tex.split("\n")
        midrule_idx = next(i for i, l in enumerate(lines) if r"\midrule" in l)
        bottomrule_idx = next(i for i, l in enumerate(lines) if r"\bottomrule" in l)
        for line in lines[midrule_idx + 1 : bottomrule_idx]:
            if r"\\" in line:
                assert line.count("&") == n_cols - 1

    def test_custom_columns(self):
        results = [{"scenario": "test", "fall_rate": 0.1}]
        custom_cols = [
            ("scenario", "Scenario", None, False),
            ("fall_rate", "Fall Rate", ".1f", True),
        ]
        tex = generate_latex_table(results, columns=custom_cols)
        assert "Scenario" in tex
        assert "Fall Rate" in tex
        # fall_rate 0.1 as percent = 10.0
        assert "10.0" in tex

    def test_underscore_escaping_in_scenario(self):
        results = [{"scenario": "push_sweep_60N", "fall_rate": 0.0}]
        tex = generate_latex_table(results)
        assert r"push\_sweep\_60N" in tex
