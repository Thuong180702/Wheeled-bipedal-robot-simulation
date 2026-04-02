"""
LaTeX table generator for balance evaluation results.

Produces a booktabs-style LaTeX table from ScenarioMetrics dicts
(as output by ``eval_balance.py`` → ``eval_results.json``).

Usage::

    from wheeled_biped.eval.latex_table import generate_latex_table
    import json

    with open("eval_results.json") as f:
        data = json.load(f)
    tex = generate_latex_table(data["results"])
    print(tex)

No external dependencies beyond Python stdlib.
"""

from __future__ import annotations

import math
from typing import Any

# Default columns shown in the paper table.
# Each entry: (dict_key, header_label, format_spec_or_None, percent_flag)
DEFAULT_COLUMNS: list[tuple[str, str, str | None, bool]] = [
    ("scenario", "Scenario", None, False),
    ("survival_time_mean_s", r"Surv.\,(s)", ".1f", False),
    ("fall_rate", r"Fall\,\%", ".0f", True),
    ("pitch_rms_deg", r"Pitch RMS ($^\circ$)", ".2f", False),
    ("roll_rms_deg", r"Roll RMS ($^\circ$)", ".2f", False),
    ("height_rmse_m", r"$h$ RMSE (m)", ".3f", False),
    ("torque_rms_nm", r"$\tau$ RMS (Nm)", ".2f", False),
    ("max_recoverable_push_n", r"Max Push (N)", ".1f", False),
]


def _fmt(val: Any, fmt: str | None, percent: bool) -> str:
    """Format a single value for LaTeX.

    Returns ``---`` for NaN / Inf / None / non-numeric values.
    """
    if val is None:
        return "---"
    if isinstance(val, str):
        # Escape underscores for LaTeX
        return val.replace("_", r"\_")
    if isinstance(val, (int, float)):
        if math.isnan(val) or math.isinf(val):
            return "---"
        if percent:
            return format(val * 100, fmt or ".0f")
        return format(val, fmt or "")
    return str(val).replace("_", r"\_")


def generate_latex_table(
    results: list[dict[str, Any]],
    columns: list[tuple[str, str, str | None, bool]] | None = None,
    caption: str = "Balance evaluation results.",
    label: str = "tab:balance_eval",
) -> str:
    r"""Generate a booktabs-style LaTeX table from result dicts.

    Args:
        results: List of ``ScenarioMetrics.to_dict()`` dicts.
        columns: Column specs as ``(key, header, format, is_percent)``.
                 Defaults to :data:`DEFAULT_COLUMNS`.
        caption: LaTeX table caption.
        label: LaTeX label for ``\ref{}``.

    Returns:
        Complete LaTeX table string (``\begin{table}`` … ``\end{table}``).
    """
    if columns is None:
        columns = DEFAULT_COLUMNS

    if not results:
        return "% No results to tabulate.\n"

    n_cols = len(columns)
    col_spec = "l" + "r" * (n_cols - 1)

    lines: list[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row
    headers = " & ".join(col[1] for col in columns)
    lines.append(headers + r" \\")
    lines.append(r"\midrule")

    # Data rows
    for row in results:
        cells: list[str] = []
        for key, _hdr, fmt, pct in columns:
            val = row.get(key)
            cells.append(_fmt(val, fmt, pct))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines) + "\n"
