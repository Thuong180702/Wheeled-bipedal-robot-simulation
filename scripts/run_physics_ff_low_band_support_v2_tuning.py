"""Tune and validate the PFF low-band support v2 opt-in profile.

The sweep phase runs the existing v1 low-band runtime path with explicit
continuous-parameter overrides.  It does not change v1/default behavior.

After the selected v2 profile is registered, the full phase runs v2 across the
Step C segment suite and fixed-height suite, then compares against unchanged
B2v2/current-PFF/v1 reference telemetry.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import run_physics_ff_low_band_support_v1_full_step_c_validation as base


ROOT = Path(__file__).resolve().parent.parent
SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups_centered"
OUT_BASE = ROOT / "outputs" / "physics_ff_low_band_support_v2_tuning"
REPORT_PATH = ROOT / "docs" / "validation" / "physics_ff_low_band_support_v2_tuning_report.md"
REFERENCE_V1_OUT = ROOT / "outputs" / "physics_ff_step_c_low_band_support_v1_full_step_c"

V1_PROFILE = "physics_equilibrium_feedforward_outer_loop_low_band_support_v1"
V2_PROFILE = "physics_equilibrium_feedforward_outer_loop_low_band_support_v2"

TRIM_SWEEP = [0.50, 0.65, 0.80, 1.00]
KP_SWEEP = [1.10, 1.20, 1.30, 1.40, 1.50]
SIGMA_SWEEP = [0.004, 0.006, 0.008]

FOCUSED_LOW_B2V2 = {
    "maxabs": 0.07152568473444086,
    "p2p": 0.14095128888914765,
    "out15_pct": 0.0,
}
FOCUSED_LOW_CURRENT_PFF = {
    "maxabs": 0.11582725658250322,
    "p2p": 0.16480270088952953,
    "out15_pct": 0.0,
}
FIXED_LOW_CURRENT_PFF = {
    "maxabs": 0.15485155294592035,
    "p2p": 0.24520564491828495,
    "out15_pct": 1.4507253626813407,
}


@dataclass(frozen=True)
class SimSpec:
    suite: str
    case_name: str
    seg_idx: int
    height: str
    steps: int
    tag: str
    profile: str
    profile_label: str
    out_dir: Path
    trim_deg_peak: float | None = None
    kp_eff_peak_deg_per_m: float | None = None
    sigma_m: float | None = None


def slug_float(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


def candidate_id(trim: float, kp: float, sigma: float) -> str:
    return f"trim{slug_float(trim)}_kp{slug_float(kp)}_sigma{slug_float(sigma)}"


def run_sim(spec: SimSpec, *, force: bool, timeout_s: int) -> dict[str, Any]:
    spec.out_dir.mkdir(parents=True, exist_ok=True)
    tel_dst = spec.out_dir / f"telemetry_{spec.steps}.csv"
    summary_dst = spec.out_dir / f"telemetry_{spec.steps}.summary.json"
    if tel_dst.exists() and tel_dst.stat().st_size > 1000 and not force:
        return {
            "spec": spec,
            "telemetry_path": str(tel_dst),
            "summary_path": str(summary_dst) if summary_dst.exists() else "",
            "returncode": 0,
            "cached": True,
            "error": "",
        }

    setup_path = SETUP_DIR / f"{spec.height}_setup.json"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "simulate_hierarchical_controller.py"),
        "--controller-mode",
        "balance-core",
        "--sagittal-controller",
        "velocity-damped",
        "--vd-sagittal-authority-profile",
        spec.profile,
        "--height-variant-setup",
        str(setup_path),
        "--steps",
        str(spec.steps),
        "--telemetry-decimation",
        "1",
        "--failure-window-steps",
        str(spec.steps),
        "--write-run-summary-sidecar",
        "--output-dir",
        str(spec.out_dir),
    ]
    if spec.trim_deg_peak is not None:
        cmd += ["--vd-low-band-support-pitch-ref-offset-peak-deg", str(spec.trim_deg_peak)]
    if spec.kp_eff_peak_deg_per_m is not None:
        cmd += ["--vd-low-band-support-kp-peak-deg-per-m", str(spec.kp_eff_peak_deg_per_m)]
    if spec.sigma_m is not None:
        cmd += ["--vd-low-band-support-sigma-m", str(spec.sigma_m)]

    (spec.out_dir / "command.json").write_text(json.dumps(cmd, indent=2), encoding="utf-8")
    try:
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout_s)
        returncode: int | None = result.returncode
        (spec.out_dir / "stdout.txt").write_text(result.stdout or "", encoding="utf-8", errors="replace")
        (spec.out_dir / "stderr.txt").write_text(result.stderr or "", encoding="utf-8", errors="replace")
        error = "" if result.returncode == 0 else f"returncode {result.returncode}"
    except subprocess.TimeoutExpired as exc:
        returncode = None
        (spec.out_dir / "stdout.txt").write_text(exc.stdout or "", encoding="utf-8", errors="replace")
        (spec.out_dir / "stderr.txt").write_text((exc.stderr or "") + "\nTIMEOUT\n", encoding="utf-8", errors="replace")
        error = "timeout"

    telemetry_candidates = sorted(spec.out_dir.glob("telemetry_*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if telemetry_candidates:
        newest = telemetry_candidates[0]
        if newest.resolve() != tel_dst.resolve():
            shutil.copy2(newest, tel_dst)
    sidecar_candidates = sorted(spec.out_dir.glob("telemetry_*.summary.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if sidecar_candidates:
        newest_sidecar = sidecar_candidates[0]
        if newest_sidecar.resolve() != summary_dst.resolve():
            shutil.copy2(newest_sidecar, summary_dst)

    return {
        "spec": spec,
        "telemetry_path": str(tel_dst) if tel_dst.exists() else "",
        "summary_path": str(summary_dst) if summary_dst.exists() else "",
        "returncode": returncode,
        "cached": False,
        "error": error,
    }


def analyze(result: dict[str, Any]) -> dict[str, Any]:
    spec: SimSpec = result["spec"]
    run_result = {
        **result,
        "spec": base.RunSpec(
            suite=spec.suite,
            case_name=spec.case_name,
            seg_idx=spec.seg_idx,
            height=spec.height,
            steps=spec.steps,
            tag=spec.tag,
            profile=spec.profile,
            profile_label=spec.profile_label,
        ),
    }
    row = base.analyze_telemetry(run_result)
    row["trim_deg_peak"] = spec.trim_deg_peak
    row["kp_eff_peak_deg_per_m"] = spec.kp_eff_peak_deg_per_m
    row["sigma_m"] = spec.sigma_m
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    base._write_csv(path, rows)


def load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_sweep_specs() -> list[SimSpec]:
    specs: list[SimSpec] = []
    for trim in TRIM_SWEEP:
        for kp in KP_SWEEP:
            for sigma in SIGMA_SWEEP:
                cid = candidate_id(trim, kp, sigma)
                base_dir = OUT_BASE / "sweep" / cid
                for case_name, height, steps in [
                    ("focused_low_0p320", "low_0p320", 300),
                    ("fixed_low_0p320", "low_0p320", 2000),
                ]:
                    specs.append(
                        SimSpec(
                            suite="sweep",
                            case_name=case_name,
                            seg_idx=0,
                            height=height,
                            steps=steps,
                            tag=cid,
                            profile=V1_PROFILE,
                            profile_label="sweep override",
                            out_dir=base_dir / case_name,
                            trim_deg_peak=trim,
                            kp_eff_peak_deg_per_m=kp,
                            sigma_m=sigma,
                        )
                    )
    return specs


def sweep_candidate_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["tag"], {})[row["case_name"]] = row
    summaries: list[dict[str, Any]] = []
    focused_maxabs_limit = FOCUSED_LOW_B2V2["maxabs"] + base.MAXABS_TOL_M
    focused_p2p_limit = FOCUSED_LOW_B2V2["p2p"] * base.P2P_FACTOR
    focused_out15_limit = FOCUSED_LOW_B2V2["out15_pct"] + base.OUT15_TOL_PP
    fixed_p2p_limit = FIXED_LOW_CURRENT_PFF["p2p"] * base.P2P_FACTOR
    for cid, by_case in sorted(grouped.items()):
        focused = by_case.get("focused_low_0p320")
        fixed = by_case.get("fixed_low_0p320")
        if not focused or not fixed:
            continue
        trim = float(focused["trim_deg_peak"])
        kp = float(focused["kp_eff_peak_deg_per_m"])
        sigma = float(focused["sigma_m"])
        reasons: list[str] = []
        if focused["any_fell"] or focused["any_unsafe"]:
            reasons.append("focused_safety")
        if fixed["any_fell"] or fixed["any_unsafe"]:
            reasons.append("fixed_safety")
        if float(focused["support_position_error_max_abs_m"]) > focused_maxabs_limit:
            reasons.append("focused_maxabs")
        if float(focused["support_position_error_p2p_m"]) > focused_p2p_limit:
            reasons.append("focused_p2p")
        if float(focused["out15_pct"]) > focused_out15_limit:
            reasons.append("focused_out15")
        if float(focused["support_position_error_max_abs_m"]) > FOCUSED_LOW_CURRENT_PFF["maxabs"]:
            reasons.append("focused_not_better_than_current_maxabs")
        if float(focused["support_position_error_p2p_m"]) > FOCUSED_LOW_CURRENT_PFF["p2p"]:
            reasons.append("focused_not_better_than_current_p2p")
        if float(fixed["support_position_error_p2p_m"]) > fixed_p2p_limit:
            reasons.append("fixed_p2p")
        if float(fixed["hip_yaw_abs_max"]) > 0.35:
            reasons.append("fixed_hip_yaw")
        target_margin = float(fixed["support_position_error_p2p_m"]) <= 0.275
        summaries.append(
            {
                "candidate_id": cid,
                "trim_deg_peak": trim,
                "kp_eff_peak_deg_per_m": kp,
                "sigma_m": sigma,
                "pass": not reasons,
                "target_margin_p2p_le_0p275": target_margin,
                "reasons": ";".join(reasons),
                "focused_maxabs": focused["support_position_error_max_abs_m"],
                "focused_p2p": focused["support_position_error_p2p_m"],
                "focused_out15_pct": focused["out15_pct"],
                "focused_pitch_max_deg": focused["pitch_max_abs_deg"],
                "focused_hip_yaw_max": focused["hip_yaw_abs_max"],
                "fixed_maxabs": fixed["support_position_error_max_abs_m"],
                "fixed_p2p": fixed["support_position_error_p2p_m"],
                "fixed_out15_pct": fixed["out15_pct"],
                "fixed_pitch_max_deg": fixed["pitch_max_abs_deg"],
                "fixed_hip_yaw_max": fixed["hip_yaw_abs_max"],
                "hidden_torque_max": max(float(focused["hidden_torque_max"]), float(fixed["hidden_torque_max"])),
                "ownership_violation_max": max(float(focused["ownership_violation_max"]), float(fixed["ownership_violation_max"])),
                "wbc_applied_rows": int(focused["wbc_applied_rows"]) + int(fixed["wbc_applied_rows"]),
                "low_band_scale_max": max(float(focused["low_band_scale_max"]), float(fixed["low_band_scale_max"])),
                "effective_support_kp_max": max(float(focused["effective_support_kp_max"]), float(fixed["effective_support_kp_max"])),
                "pitch_trim_deg_max_abs": max(float(focused["pitch_trim_deg_max_abs"]), float(fixed["pitch_trim_deg_max_abs"])),
                "selection_tuple": [0 if target_margin else 1, trim, kp, sigma, float(fixed["support_position_error_p2p_m"])],
            }
        )
    return summaries


def select_candidate(summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    passing = [row for row in summaries if row["pass"]]
    if not passing:
        return None
    with_margin = [row for row in passing if row["target_margin_p2p_le_0p275"]]
    pool = with_margin if with_margin else passing
    return sorted(pool, key=lambda row: tuple(row["selection_tuple"]))[0]


def run_sweep(*, force: bool, max_workers: int, timeout_s: int) -> dict[str, Any]:
    specs = make_sweep_specs()
    (OUT_BASE / "sweep").mkdir(parents=True, exist_ok=True)
    run_results: list[dict[str, Any]] = []
    print(f"Running v2 low-band sweep: {len(specs)} simulations", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        future_map = {executor.submit(run_sim, spec, force=force, timeout_s=timeout_s): spec for spec in specs}
        for idx, future in enumerate(as_completed(future_map), start=1):
            spec = future_map[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover
                result = {"spec": spec, "telemetry_path": "", "summary_path": "", "returncode": None, "cached": False, "error": repr(exc)}
            run_results.append(result)
            print(f"[sweep {idx:03d}/{len(specs):03d}] {spec.tag} {spec.case_name}: {'cached' if result.get('cached') else 'ran'} {result.get('error') or 'ok'}", flush=True)
    rows = [analyze(result) for result in run_results]
    rows.sort(key=lambda row: (row["tag"], row["case_name"]))
    summaries = sweep_candidate_summary(rows)
    selected = select_candidate(summaries)
    write_csv(OUT_BASE / "sweep_metrics.csv", rows)
    write_csv(OUT_BASE / "sweep_summary.csv", summaries)
    payload = {
        "selected": selected,
        "n_candidates": len(summaries),
        "n_passing": sum(1 for row in summaries if row["pass"]),
        "n_target_margin": sum(1 for row in summaries if row["pass"] and row["target_margin_p2p_le_0p275"]),
        "focused_limits": {
            "maxabs": FOCUSED_LOW_B2V2["maxabs"] + base.MAXABS_TOL_M,
            "p2p": FOCUSED_LOW_B2V2["p2p"] * base.P2P_FACTOR,
            "out15_pct": FOCUSED_LOW_B2V2["out15_pct"] + base.OUT15_TOL_PP,
        },
        "fixed_limits": {
            "p2p": FIXED_LOW_CURRENT_PFF["p2p"] * base.P2P_FACTOR,
            "target_margin_p2p": 0.275,
        },
    }
    (OUT_BASE / "selected_candidate.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def full_specs() -> list[SimSpec]:
    specs: list[SimSpec] = []
    for case_name, segments in base.STEP_C_CASES.items():
        for seg_idx, (height, steps) in enumerate(segments):
            specs.append(
                SimSpec(
                    suite="step_c",
                    case_name=case_name,
                    seg_idx=seg_idx,
                    height=height,
                    steps=steps,
                    tag="D_LOW_BAND_V2",
                    profile=V2_PROFILE,
                    profile_label="Low-band support v2",
                    out_dir=OUT_BASE / "step_c" / case_name / "D_LOW_BAND_V2" / f"seg{seg_idx:03d}_{height}_{steps}",
                )
            )
    for seg_idx, height in enumerate(base.HEIGHTS):
        specs.append(
            SimSpec(
                suite="fixed",
                case_name="fixed_height",
                seg_idx=seg_idx,
                height=height,
                steps=2000,
                tag="D_LOW_BAND_V2",
                profile=V2_PROFILE,
                profile_label="Low-band support v2",
                out_dir=OUT_BASE / "fixed_height" / "D_LOW_BAND_V2" / f"seg{seg_idx:03d}_{height}_2000",
            )
        )
    return specs


def normalize_reference_metric_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    if "seg_idx" in normalized:
        normalized["seg_idx"] = int(normalized["seg_idx"])
    if "steps_nominal" in normalized and normalized["steps_nominal"] != "":
        normalized["steps_nominal"] = int(float(normalized["steps_nominal"]))
    for key in (
        "returncode",
        "steps",
        "terminated_rows",
        "wbc_owner_rows",
        "wbc_authority_rows",
        "wbc_after_authority_clip_rows",
        "wbc_correction_rows_diagnostic",
        "wbc_applied_rows",
        "contact_invalid_rows_after_startup",
        "low_band_active_rows",
    ):
        if key in normalized and normalized[key] != "":
            normalized[key] = int(float(normalized[key]))
    for key in (
        "cached",
        "any_fell",
        "fell_short",
        "any_unsafe",
    ):
        if key in normalized:
            normalized[key] = str(normalized[key]).strip().lower() == "true"
    return normalized


def reference_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    step_rows = load_csv(REFERENCE_V1_OUT / "step_c_segment_metrics.csv")
    fixed_rows = load_csv(REFERENCE_V1_OUT / "fixed_height_metrics.csv")
    keep_tags = {"A_B2V2", "B_CURRENT_PFF", "C_LOW_BAND_V1"}
    return (
        [normalize_reference_metric_row(row) for row in step_rows if row.get("tag") in keep_tags],
        [normalize_reference_metric_row(row) for row in fixed_rows if row.get("tag") in keep_tags],
    )


def run_full_v2(*, force: bool, max_workers: int, timeout_s: int) -> dict[str, Any]:
    specs = full_specs()
    print(f"Running selected v2 full validation: {len(specs)} fresh v2 simulations", flush=True)
    run_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        future_map = {executor.submit(run_sim, spec, force=force, timeout_s=timeout_s): spec for spec in specs}
        for idx, future in enumerate(as_completed(future_map), start=1):
            spec = future_map[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover
                result = {"spec": spec, "telemetry_path": "", "summary_path": "", "returncode": None, "cached": False, "error": repr(exc)}
            run_results.append(result)
            print(f"[full {idx:03d}/{len(specs):03d}] {spec.suite} {spec.case_name} {spec.seg_idx:03d} {spec.height}: {'cached' if result.get('cached') else 'ran'} {result.get('error') or 'ok'}", flush=True)
    v2_rows = [analyze(result) for result in run_results]
    ref_step, ref_fixed = reference_rows()
    step_rows = ref_step + [row for row in v2_rows if row["suite"] == "step_c"]
    fixed_rows = ref_fixed + [row for row in v2_rows if row["suite"] == "fixed"]
    all_rows = step_rows + fixed_rows
    step_summary = base._aggregate(step_rows, suite="step_c")
    fixed_summary = base._aggregate(fixed_rows, suite="fixed")
    decision = evaluate_v2_decision(step_rows, fixed_rows, step_summary, fixed_summary)
    write_csv(OUT_BASE / "full_step_c_segment_metrics.csv", step_rows)
    write_csv(OUT_BASE / "full_step_c_case_summary.csv", step_summary)
    write_csv(OUT_BASE / "full_fixed_height_metrics.csv", fixed_rows)
    write_csv(OUT_BASE / "full_fixed_height_summary.csv", fixed_summary)
    (OUT_BASE / "decision_summary.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return {
        "decision": decision,
        "step_rows": step_rows,
        "fixed_rows": fixed_rows,
        "step_summary": step_summary,
        "fixed_summary": fixed_summary,
    }


def by_key(rows: list[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {tuple(row[key] for key in keys): row for row in rows}


def within(row: dict[str, Any], ref: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if float(row["support_position_error_max_abs_m"]) > float(ref["support_position_error_max_abs_m"]) + base.MAXABS_TOL_M:
        reasons.append("maxabs")
    if float(row["support_position_error_p2p_m"]) > float(ref["support_position_error_p2p_m"]) * base.P2P_FACTOR:
        reasons.append("p2p")
    if float(row["out15_pct"]) > float(ref["out15_pct"]) + base.OUT15_TOL_PP:
        reasons.append("out15")
    return not reasons, reasons


def evaluate_v2_decision(
    step_rows: list[dict[str, Any]],
    fixed_rows: list[dict[str, Any]],
    step_summary: list[dict[str, Any]],
    fixed_summary: list[dict[str, Any]],
) -> dict[str, Any]:
    failures: list[str] = []
    monitors: list[str] = []
    seg = by_key(step_rows + fixed_rows, "suite", "case_name", "seg_idx", "tag")
    summaries = by_key(step_summary + fixed_summary, "suite", "case_name", "tag")
    v2_rows = [row for row in step_rows + fixed_rows if row["tag"] == "D_LOW_BAND_V2"]
    for row in v2_rows:
        if row["any_fell"] or row["any_unsafe"]:
            failures.append(f"{row['suite']}:{row['case_name']}:{row['seg_idx']}:{row['height']}:{row.get('unsafe_reasons', '')}")
    for case_name in ["C1_slow_ladder_up_down", "C2_random_500dwell", "C3_random_200dwell", "C4_abrupt_stress", "C5_long_random"]:
        row = summaries.get(("step_c", case_name, "D_LOW_BAND_V2"))
        if not row:
            failures.append(f"{case_name}:missing_v2_summary")
        elif row["any_fell"] or row["any_unsafe"]:
            failures.append(f"{case_name}:v2_not_ok")

    focused = {
        tag: seg.get(("step_c", "focused_low_0p320", 0, tag))
        for tag in ("A_B2V2", "B_CURRENT_PFF", "C_LOW_BAND_V1", "D_LOW_BAND_V2")
    }
    if all(focused.values()):
        ok, reasons = within(focused["D_LOW_BAND_V2"], focused["A_B2V2"])
        if not ok:
            failures.append(f"focused_low_0p320:outside_B2v2:{','.join(reasons)}")
        ok, reasons = within(focused["D_LOW_BAND_V2"], focused["B_CURRENT_PFF"])
        if not ok:
            failures.append(f"focused_low_0p320:outside_current_PFF:{','.join(reasons)}")
        if (
            float(focused["D_LOW_BAND_V2"]["support_position_error_max_abs_m"])
            > float(focused["B_CURRENT_PFF"]["support_position_error_max_abs_m"])
        ):
            failures.append("focused_low_0p320:not_improved_vs_current_PFF_maxabs")
        if (
            float(focused["D_LOW_BAND_V2"]["support_position_error_p2p_m"])
            > float(focused["B_CURRENT_PFF"]["support_position_error_p2p_m"])
        ):
            failures.append("focused_low_0p320:not_improved_vs_current_PFF_p2p")
    else:
        failures.append("focused_low_0p320:missing_reference")

    high = {
        tag: seg.get(("step_c", "focused_high_0p480", 0, tag))
        for tag in ("B_CURRENT_PFF", "C_LOW_BAND_V1", "D_LOW_BAND_V2")
    }
    if all(high.values()):
        ok, reasons = within(high["D_LOW_BAND_V2"], high["B_CURRENT_PFF"])
        if not ok:
            failures.append(f"focused_high_0p480:outside_current_PFF:{','.join(reasons)}")
    else:
        failures.append("focused_high_0p480:missing_reference")

    fixed_low = {
        tag: seg.get(("fixed", "fixed_height", 1, tag))
        for tag in ("B_CURRENT_PFF", "D_LOW_BAND_V2")
    }
    if all(fixed_low.values()):
        fixed_p2p = float(fixed_low["D_LOW_BAND_V2"]["support_position_error_p2p_m"])
        limit = float(fixed_low["B_CURRENT_PFF"]["support_position_error_p2p_m"]) * base.P2P_FACTOR
        if fixed_p2p > limit:
            failures.append("fixed_low_0p320:p2p_vs_current_PFF")
        elif fixed_p2p > 0.275:
            monitors.append("fixed_low_0p320:p2p_above_target_margin_0p275")
    else:
        failures.append("fixed_low_0p320:missing_reference")

    for row in v2_rows:
        if row["height"] not in base.PROTECTED_HEIGHTS:
            continue
        for ref_tag in ("A_B2V2", "B_CURRENT_PFF", "C_LOW_BAND_V1"):
            ref = seg.get((row["suite"], row["case_name"], row["seg_idx"], ref_tag))
            if not ref:
                continue
            ok, reasons = within(row, ref)
            if not ok:
                message = f"{row['suite']}:{row['case_name']}:{row['seg_idx']}:{row['height']}:vs_{ref_tag}:{','.join(reasons)}"
                if ref_tag == "C_LOW_BAND_V1":
                    monitors.append(message)
                else:
                    failures.append(message)

    if failures:
        classification = "PHYSICS_FF_LOW_BAND_V2_STEP_C_FAIL"
    elif monitors:
        classification = "PHYSICS_FF_LOW_BAND_V2_STEP_C_PASS_WITH_MONITORING"
    else:
        classification = "PHYSICS_FF_LOW_BAND_V2_STEP_C_PASS"
    return {
        "classification": classification,
        "failures": failures,
        "monitors": monitors,
        "focused_low_0p320": focused,
        "fixed_low_0p320": fixed_low,
        "reference_source": str(REFERENCE_V1_OUT),
    }


def fmt(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


TAG_ORDER = {"A_B2V2": 0, "B_CURRENT_PFF": 1, "C_LOW_BAND_V1": 2, "D_LOW_BAND_V2": 3}
TAG_LABEL = {
    "A_B2V2": "B2v2",
    "B_CURRENT_PFF": "Current PFF",
    "C_LOW_BAND_V1": "Low-band v1",
    "D_LOW_BAND_V2": "Low-band v2",
}
CASE_ORDER = {
    name: idx
    for idx, name in enumerate(
        [
            "C1_slow_ladder_up_down",
            "C2_random_500dwell",
            "C3_random_200dwell",
            "C4_abrupt_stress",
            "C5_long_random",
            "focused_low_0p320",
            "focused_high_0p480",
        ]
    )
}
HEIGHT_ORDER = {height: idx for idx, height in enumerate(base.HEIGHTS)}


def render_report(sweep_payload: dict[str, Any] | None, full_payload: dict[str, Any] | None) -> str:
    selected = (sweep_payload or {}).get("selected")
    decision = (full_payload or {}).get("decision", {})
    classification = decision.get("classification", "PHYSICS_FF_LOW_BAND_V2_INCONCLUSIVE")
    lines = [
        "# Physics FF Low-Band Support v2 Tuning Report",
        "",
        "Task: `physics_ff_low_band_support_v2_tuning`",
        "",
        f"Classification: `{classification}`",
        "",
        "## Scope",
        "",
        "- No PFF promotion.",
        "- No Step D run.",
        "- No default-profile change.",
        "- No PFF source/calibration/interpolation change.",
        "- No setup-name branching; tuning uses continuous Gaussian height scale centered at 0.320 m.",
        "",
        "## Artifacts",
        "",
        f"- Output directory: `{OUT_BASE.relative_to(ROOT).as_posix()}/`",
        f"- Sweep metrics: `{(OUT_BASE / 'sweep_metrics.csv').relative_to(ROOT).as_posix()}`",
        f"- Sweep summary: `{(OUT_BASE / 'sweep_summary.csv').relative_to(ROOT).as_posix()}`",
        f"- Full Step C metrics: `{(OUT_BASE / 'full_step_c_segment_metrics.csv').relative_to(ROOT).as_posix()}`",
        f"- Fixed-height metrics: `{(OUT_BASE / 'full_fixed_height_metrics.csv').relative_to(ROOT).as_posix()}`",
    ]
    if selected:
        lines += [
            "",
            "## Selected v2 Parameters",
            "",
            f"- trim_deg_peak: `{fmt(selected['trim_deg_peak'], 2)}`",
            f"- kp_eff_peak_deg_per_m: `{fmt(selected['kp_eff_peak_deg_per_m'], 2)}`",
            f"- sigma_m: `{fmt(selected['sigma_m'], 3)}`",
            f"- focused maxabs: `{fmt(selected['focused_maxabs'], 10)} m`",
            f"- focused P2P: `{fmt(selected['focused_p2p'], 10)} m`",
            f"- fixed low_0p320 maxabs: `{fmt(selected['fixed_maxabs'], 10)} m`",
            f"- fixed low_0p320 P2P: `{fmt(selected['fixed_p2p'], 10)} m`",
            f"- target margin P2P <= 0.275 m: `{selected['target_margin_p2p_le_0p275']}`",
        ]
    if sweep_payload:
        lines += [
            "",
            "## Sweep Summary",
            "",
            f"- candidates evaluated: `{sweep_payload.get('n_candidates')}`",
            f"- passing candidates: `{sweep_payload.get('n_passing')}`",
            f"- passing target-margin candidates: `{sweep_payload.get('n_target_margin')}`",
        ]
    if full_payload:
        step_summary = full_payload["step_summary"]
        fixed_rows = full_payload["fixed_rows"]
        step_summary_sorted = sorted(
            step_summary,
            key=lambda row: (CASE_ORDER.get(row["case_name"], 999), TAG_ORDER.get(row["tag"], 999)),
        )
        fixed_rows_sorted = sorted(
            fixed_rows,
            key=lambda row: (HEIGHT_ORDER.get(row["height"], 999), TAG_ORDER.get(row["tag"], 999)),
        )
        lines += [
            "",
            "## Full Step C Comparison",
            "",
            "| Case | Profile | fell | unsafe | maxabs m | max trans m | p2p m | out15% | pitch deg | roll deg | hip-yaw | HY div | hidden | owner | WBC rows | scale max | Kp max | trim deg |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in step_summary_sorted:
            lines.append(
                f"| {row['case_name']} | {TAG_LABEL.get(row['tag'], row['tag'])} | "
                f"{row['any_fell']} | {row['any_unsafe']} | {fmt(row['max_maxabs'], 4)} | "
                f"{fmt(row['max_trans'], 4)} | {fmt(row['support_position_error_p2p_m'], 4)} | "
                f"{fmt(row['out15_pct'], 1)} | {fmt(row['pitch_max_abs_deg'], 2)} | "
                f"{fmt(row['roll_max_abs_deg'], 2)} | {fmt(row['hip_yaw_abs_max'], 4)} | "
                f"{fmt(row['hip_yaw_divergence_error'], 4)} | {fmt(row['hidden_torque_max'], 1)} | "
                f"{fmt(row['ownership_violation_max'], 1)} | {row['wbc_applied_rows']} | "
                f"{fmt(row['low_band_scale_max'], 4)} | {fmt(row['effective_support_kp_max'], 4)} | "
                f"{fmt(row['pitch_trim_deg_max_abs'], 4)} |"
            )
        lines += [
            "",
            "## Fixed-Height 10-Height Comparison",
            "",
            "| Height | Profile | fell | unsafe | maxabs m | max trans m | p2p m | out15% | pitch deg | roll deg | hip-yaw | HY div | hidden | owner | WBC rows | scale max | Kp max | trim deg |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in fixed_rows_sorted:
            lines.append(
                f"| {row['height']} | {TAG_LABEL.get(row['tag'], row['tag'])} | "
                f"{row['any_fell']} | {row['any_unsafe']} | "
                f"{fmt(row['support_position_error_max_abs_m'], 4)} | {fmt(row['max_trans_m'], 4)} | "
                f"{fmt(row['support_position_error_p2p_m'], 4)} | {fmt(row['out15_pct'], 1)} | "
                f"{fmt(row['pitch_max_abs_deg'], 2)} | {fmt(row['roll_max_abs_deg'], 2)} | "
                f"{fmt(row['hip_yaw_abs_max'], 4)} | {fmt(row['hip_yaw_divergence_error'], 4)} | "
                f"{fmt(row['hidden_torque_max'], 1)} | {fmt(row['ownership_violation_max'], 1)} | "
                f"{row['wbc_applied_rows']} | {fmt(row['low_band_scale_max'], 4)} | "
                f"{fmt(row['effective_support_kp_max'], 4)} | {fmt(row['pitch_trim_deg_max_abs'], 4)} |"
            )
        lines += [
            "",
            "## Decision",
            "",
            f"- Failures: `{len(decision.get('failures', []))}`",
            f"- Monitoring items: `{len(decision.get('monitors', []))}`",
        ]
        for item in decision.get("failures", [])[:40]:
            lines.append(f"- failure: {item}")
        for item in decision.get("monitors", [])[:40]:
            lines.append(f"- monitor: {item}")
        if classification in {"PHYSICS_FF_LOW_BAND_V2_STEP_C_PASS", "PHYSICS_FF_LOW_BAND_V2_STEP_C_PASS_WITH_MONITORING"}:
            lines += ["", "Step D may be run next. Step D was not run in this task."]
        else:
            lines += ["", "Step D was not run."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune and validate PFF low-band support v2")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--timeout-s", type=int, default=2400)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sweep-only", action="store_true")
    parser.add_argument("--full-only", action="store_true")
    args = parser.parse_args()

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    start = time.time()
    sweep_payload = None
    full_payload = None
    if not args.full_only:
        sweep_payload = run_sweep(force=args.force, max_workers=args.max_workers, timeout_s=args.timeout_s)
    else:
        selected_path = OUT_BASE / "selected_candidate.json"
        if selected_path.exists():
            sweep_payload = json.loads(selected_path.read_text(encoding="utf-8"))
    if not args.sweep_only:
        full_payload = run_full_v2(force=args.force, max_workers=args.max_workers, timeout_s=args.timeout_s)
    elapsed_s = time.time() - start
    summary = {
        "elapsed_s": elapsed_s,
        "sweep": sweep_payload,
        "decision": None if full_payload is None else full_payload["decision"],
    }
    (OUT_BASE / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(sweep_payload, full_payload), encoding="utf-8")
    classification = (full_payload or {}).get("decision", {}).get("classification", "PHYSICS_FF_LOW_BAND_V2_INCONCLUSIVE")
    print(f"Classification: {classification}", flush=True)
    print(f"Report: {REPORT_PATH}", flush=True)
    return 0 if not classification.endswith("_FAIL") else 2


if __name__ == "__main__":
    raise SystemExit(main())
