#!/usr/bin/env python
"""Ramp-up + step-down battery for V3_ANCHOR teleop (headless).

Terrain: flat run-up → 12° ramp → 1.2 m platform at height H → cliff edge
back down to the floor. The robot drives forward (teleop hold-to-drive),
climbs the ramp, crosses the platform, drives off the edge, and must land
and end standing (anchored) on the lower floor.

The height COMMAND follows the terrain: the controller measures CoM height
in absolute world z, so the harness adds the wheel-elevation offset to the
commanded height (hardware analog: leg-kinematics height is ground-relative).

Usage:
  python scripts/ramp_step_tests.py                    # 20,30,40,50 cm
  python scripts/ramp_step_tests.py --heights 20,30 --render
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import mujoco

import scripts.promote_v3_vs_assist as P
from scripts.teleop_scenario_tests import TeleopSim, DT, KEY_UP

ROOT = Path(__file__).resolve().parent.parent
MODEL_XML = ROOT / "assets/robot/wheeled_biped_real.xml"
SLOPE_RAD = np.radians(12.0)
FLAT_END_Y = 0.5      # ramp foot
PLATFORM_LEN = 1.2
TERRAIN_HALF_W = 0.8


def build_terrain_xml(h: float, sign: int = 1, edge_angle_deg: float = 0.0) -> tuple[Path, dict]:
    """Inject ramp + platform boxes into the robot XML; returns (path, geometry).

    sign=+1 lays the course toward +Y (forward); sign=-1 mirrors it toward -Y
    (the robot drives it BACKWARD). edge_angle_deg rotates the PLATFORM (and
    therefore the cliff edge) about the ramp-top seam, so a straight drive
    crosses the edge OBLIQUELY — the wheels leave the lip one after the
    other. The ramp itself stays square to the path. Geometry dict stores
    unsigned course-frame values (mirror handled by ``sign``).
    """
    ramp_len = h / np.tan(SLOPE_RAD)
    y0, y1 = FLAT_END_Y, FLAT_END_Y + ramp_len
    t = 0.05  # ramp slab half-thickness
    a = float(np.hypot(ramp_len, h)) / 2 + 0.02  # slight overlap into platform
    mid_y = (y0 + y1) / 2
    cy = mid_y + t * np.sin(SLOPE_RAD)
    cz = h / 2 - t * np.cos(SLOPE_RAD)
    phi = np.radians(edge_angle_deg)
    common = ('condim="6" contype="1" conaffinity="1" friction="0.8 0.005 0.0001" '
              'solref="0.002 1" solimp="0.95 0.99 0.001 0.5 2" rgba="0.55 0.45 0.35 1"')
    geoms = (
        f'    <geom name="ramp" type="box" size="{TERRAIN_HALF_W} {a} {t}" '
        f'pos="0 {sign * cy:.4f} {cz:.4f}" euler="{sign * SLOPE_RAD:.6f} 0 0" {common}/>\n'
    )
    if edge_angle_deg:
        # Square FILLER slab off the ramp top, then a ROTATED cliff slab whose
        # near (oblique) edge is buried inside the filler footprint — only the
        # far edge (the cliff) is exposed, crossing the path at ``phi``.
        # Rotating a single platform about the seam left its oblique near
        # FACE standing ~5 cm proud of the ramp surface mid-slope (wall).
        fw, Lf = 1.2, 1.6
        pw, pl = 1.2, 0.6
        seam2 = y1 + 1.3
        pcx = -np.sin(phi) * pl
        pcy = seam2 + np.cos(phi) * pl
        half = phi / 2  # mirrored course flips the rotation sign
        geoms += (
            f'    <geom name="filler" type="box" size="{fw} {Lf/2} {h/2}" '
            f'pos="0 {sign * (y1 + Lf/2):.4f} {h/2:.4f}" {common}/>\n'
            f'    <geom name="cliff" type="box" size="{pw} {pl} {h/2}" '
            f'pos="{pcx:.4f} {sign * pcy:.4f} {h/2:.4f}" '
            f'quat="{np.cos(half):.6f} 0 0 {np.sin(sign * half):.6f}" {common}/>\n'
        )
    else:
        fw, Lf = TERRAIN_HALF_W, PLATFORM_LEN
        pw, pl = TERRAIN_HALF_W, PLATFORM_LEN / 2
        pcx, pcy = 0.0, y1 + pl
        geoms += (
            f'    <geom name="platform" type="box" size="{pw} {pl} {h/2}" '
            f'pos="0 {sign * pcy:.4f} {h/2:.4f}" {common}/>\n'
        )
    xml = MODEL_XML.read_text()
    out = MODEL_XML.parent / f"_tmp_ramp_step_{int(h*100)}_{'m' if sign < 0 else 'p'}.xml"
    out.write_text(xml.replace("  </worldbody>", geoms + "  </worldbody>"))
    return out, dict(y0=y0, y1=y1, y_edge=y1 + Lf, h=h, sign=sign, phi=phi,
                     pcx=pcx, pcy=pcy, pw=pw, pl=pl, fw=fw, fy1=y1 + Lf)


def build_curb_xml(h: float, sign: int = -1) -> tuple[Path, dict]:
    """One-wheel curb: a narrow lead-in ramp + slab that ONLY the left wheel
    (x ≈ +0.14) rides; the right wheel (x ≈ −0.14) stays on the floor. The
    robot drives the whole stretch straddling the height difference — the
    per-leg terrain adapter must keep the torso level."""
    ramp_len = h / np.tan(SLOPE_RAD)
    y0, y1 = FLAT_END_Y, FLAT_END_Y + ramp_len
    y_end = y1 + 2.0                     # curb top length: 2 m
    t = 0.05
    a = float(np.hypot(ramp_len, h)) / 2 + 0.02
    cy = (y0 + y1) / 2 + t * np.sin(SLOPE_RAD)
    cz = h / 2 - t * np.cos(SLOPE_RAD)
    x_c, hw = 0.20, 0.18                 # lateral: covers the left wheel only
    common = ('condim="6" contype="1" conaffinity="1" friction="0.8 0.005 0.0001" '
              'solref="0.002 1" solimp="0.95 0.99 0.001 0.5 2" rgba="0.55 0.45 0.35 1"')
    geoms = (
        f'    <geom name="curb_ramp" type="box" size="{hw} {a} {t}" '
        f'pos="{x_c} {sign * cy:.4f} {cz:.4f}" euler="{sign * SLOPE_RAD:.6f} 0 0" {common}/>\n'
        f'    <geom name="curb" type="box" size="{hw} 1.0 {h/2}" '
        f'pos="{x_c} {sign * (y1 + 1.0):.4f} {h/2:.4f}" {common}/>\n'
    )
    xml = MODEL_XML.read_text()
    out = MODEL_XML.parent / f"_tmp_curb_{int(h*100)}.xml"
    out.write_text(xml.replace("  </worldbody>", geoms + "  </worldbody>"))
    return out, dict(y0=y0, y1=y1, y_end=y_end, h=h, sign=sign)


def run_curb(h: float, duration_s: float = 30.0, frames: list | None = None,
             frame_stride: int = 2, cam=None, renderer=None):
    """Drive the one-wheel curb end to end: mount (one-wheel ramp), straddle
    the full 2 m, dismount off the curb end, stop and anchor."""
    from scripts.teleop_scenario_tests import KEY_DOWN
    xml, g = build_curb_xml(h)
    try:
        sim = RampStepSim(xml)
    finally:
        xml.unlink()
    logs = {k: [] for k in ("pitch", "roll", "vxy")}
    released_k = None
    letgo_k = None
    settle_k = None
    roll_curb = 0.0        # max |roll| while straddling the curb top
    d_max = 0.0            # max ground split the adapter commanded
    n = int(duration_s / DT)
    for k in range(n):
        sy = float(0.5 * (sim.d.xpos[sim.lw][1] + sim.d.xpos[sim.rw][1]))
        eff_y = g["sign"] * sy
        if any(e == "SAFETY_LETGO" for e in sim.shaper.events):
            letgo_k = k
        if released_k is None and eff_y > g["y_end"] + 1.0:
            released_k = k
        held = set() if released_k is not None else {KEY_DOWN}
        if letgo_k is not None and k - letgo_k < 50:
            held = set()
        sig = sim.shaper.update_held(held)
        if sig == "ANCHOR":
            sx0, sy0 = sim.support_xy()
            sim.shaper.stop_here(sx0, sy0, sim.yaw())
        sim.shaper.events.clear()
        sim.step_teleop()
        r_deg, p_deg = sim.rpy()
        logs["pitch"].append(p_deg)
        logs["roll"].append(r_deg)
        logs["vxy"].append(float(np.hypot(sim.d.qvel[0], sim.d.qvel[1])))
        if g["y1"] < eff_y < g["y_end"]:
            roll_curb = max(roll_curb, abs(r_deg))
            d_max = max(d_max, abs(sim.terrain.g[0] - sim.terrain.g[1]))
        if float(sim.d.qpos[2]) < 0.15 or abs(p_deg) > 75 or abs(r_deg) > 60:
            return dict(h_cm=h * 100, fell=True, fall_t=k * DT, verdict="FALL")
        if settle_k is None and released_k is not None and k > released_k + 100:
            pw = np.array(logs["pitch"][k - 50:k])
            rw = np.array(logs["roll"][k - 50:k])
            if (pw.max() - pw.min() < 1.0 and rw.max() - rw.min() < 1.0
                    and np.abs(logs["vxy"][k - 50:k]).max() < 0.08):
                settle_k = k
        if frames is not None and k % frame_stride == 0:
            cam.lookat[1] = 0.6 * cam.lookat[1] + 0.4 * sy
            cam.lookat[2] = 0.6 * cam.lookat[2] + 0.4 * 0.35
            renderer.update_scene(sim.d, camera=cam)
            frames.append((renderer.render().copy(), k * DT, sim.rpy()[0],
                           float(sim.d.qpos[2])))
    L = {k2: np.array(v) for k2, v in logs.items()}
    tail = slice(-200, None)
    res = dict(
        h_cm=h * 100, fell=False,
        roll_curb_max=roll_curb, d_max=d_max,
        t_release=released_k * DT if released_k else None,
        settle_s=(settle_k - released_k) * DT if settle_k and released_k else None,
        pitch_tail=float(np.abs(L["pitch"][tail]).mean()),
        roll_tail=float(np.abs(L["roll"][tail]).mean()),
        still_vel=float(np.sqrt((L["vxy"][tail] ** 2).mean())),
    )
    res["verdict"] = "PASS" if (
        res["t_release"] is not None and res["settle_s"] is not None
        and res["pitch_tail"] <= 4.0 and res["still_vel"] <= 0.05
    ) else "FAIL"
    return res


def platform_local(sx: float, sy: float, g: dict) -> tuple[float, float]:
    """World point → platform-local (x', y'); far cliff edge is y' = +pl."""
    x, y = sx, g["sign"] * sy
    dx, dy = x - g["pcx"], y - g["pcy"]
    ca, sa = np.cos(g["phi"]), np.sin(g["phi"])
    return ca * dx + sa * dy, -sa * dx + ca * dy


def ground_z(y: float, g: dict) -> float:
    """Course-frame profile (straight courses / on-path queries)."""
    if y < g["y0"]:
        return 0.0
    if y < g["y1"]:
        return (y - g["y0"]) * np.tan(SLOPE_RAD)
    if y < g["y_edge"]:
        return g["h"]
    return 0.0


def ground_z_xy(sx: float, sy: float, g: dict) -> float:
    """Terrain height under a world (x, y) point — handles the rotated cliff slab."""
    xl, yl = platform_local(sx, sy, g)
    if abs(xl) <= g["pw"] and abs(yl) <= g["pl"]:
        return g["h"]
    y = g["sign"] * sy
    if g["y1"] <= y <= g["fy1"] and abs(sx) <= g["fw"]:  # filler slab
        return g["h"]
    if g["y0"] <= y <= g["y1"] and abs(sx) <= TERRAIN_HALF_W:
        return (y - g["y0"]) * np.tan(SLOPE_RAD)
    return 0.0


class RampStepSim(TeleopSim):
    """TeleopSim on injected terrain. Terrain-following height + per-leg
    adaptation live in the BASE class (LegTerrainAdapter) — one code path
    for the flat battery, the terrain courses, and the live viewer."""

    def __init__(self, xml_path: Path):
        orig = P.get_model_path
        P.get_model_path = lambda: xml_path
        try:
            super().__init__()
        finally:
            P.get_model_path = orig


def run_ramp_step(h: float, duration_s: float = 30.0, frames: list | None = None,
                  frame_stride: int = 2, cam=None, renderer=None, approach_vx: float = 1.0,
                  course: str = "up_off", edge_angle_deg: float = 30.0):
    """Courses:
    up_off   — forward up the ramp, across the platform, off the ledge (default)
    up_down  — forward up the ramp, anchor mid-platform 2.5 s, then REVERSE
               back down the ramp to the flat
    back_off — the mirrored course driven BACKWARD end to end (rear-first
               ledge drop)
    back_off_fast — back_off at max reverse speed (0.70 m/s) and KEEP DRIVING
               ~1.5 m past the landing before stopping
    diag_off — back_off but the cliff edge is rotated edge_angle_deg to the
               path: the wheels leave the lip one after the other (roll+yaw
               disturbance through the drop)
    """
    from scripts.teleop_scenario_tests import KEY_DOWN
    sign = -1 if course.startswith("back_off") or course == "diag_off" else 1
    xml, g = build_terrain_xml(
        h, sign=sign, edge_angle_deg=edge_angle_deg if course == "diag_off" else 0.0)
    try:
        sim = RampStepSim(xml)
    finally:
        xml.unlink()
    # Full-speed approach: at 0.8 m/s the trailing knee descends onto the
    # ledge lip ~13 cm past the edge (vy jumped 0.8→1.45 mid-flight in the
    # 40 cm trace) and the strike spins the robot nose-down. At 1.0 m/s the
    # knee clears the lip before it drops to platform level.
    if course == "up_down":
        approach_vx = min(approach_vx, 0.8)  # no ledge exit → no need for 1.0
    sim.shaper.VX_HOLD_FWD = approach_vx
    if course in ("back_off_fast", "diag_off"):
        # Max reverse speed. For diag_off it is REQUIRED at low ledges: at
        # 0.45 m/s the oblique 20 cm crossing settles into a one-wheel-up
        # straddle (roll −29° → SAFETY_LETGO kills the drive → slow tip);
        # at 0.70 the uphill wheel clears the lip before the straddle forms.
        sim.shaper.VX_HOLD_BACK = 0.70
    logs = {k: [] for k in ("y", "relz", "pitch", "roll", "vxy")}
    reached_top = False
    released_k = None
    landed_k = None
    settle_k = None
    peak_pitch_land = 0.0
    phase = 0          # up_down phase machine
    phase_k = 0
    letgo_k = None
    past_edge = False
    flew = False
    n = int(duration_s / DT)
    for k in range(n):
        sy = float(0.5 * (sim.d.xpos[sim.lw][1] + sim.d.xpos[sim.rw][1]))
        sup_x = float(0.5 * (sim.d.xpos[sim.lw][0] + sim.d.xpos[sim.rw][0]))
        eff_y = sign * sy
        # SAFETY_LETGO latches cruise off; a human releases and re-presses.
        # Emulate that: pause the held key 0.5 s after a letgo, then resume.
        # (Backing over the convex ramp crest trips the 12° pitch letgo.)
        if any(e == "SAFETY_LETGO" for e in sim.shaper.events):
            letgo_k = k
        # Key schedule per course. Release is POSITION-based, never on the
        # airborne gate (the ramp-crest micro-hop engages it for ~30 ms and
        # must not cancel the drive command).
        if course == "up_down":
            # Release AT the crest: the shaper brakes at ~0.6 m/s² so the
            # stop takes ~0.5-0.9 m — releasing mid-platform overran the far
            # edge (the 30 cm run sailed off the ledge while braking).
            if phase == 0 and eff_y > g["y1"] + 0.05:
                phase, phase_k = 1, k          # brake + anchor on the platform
            elif phase == 1 and k - phase_k >= 300:
                phase = 2                       # reverse back down the ramp
            elif phase == 2 and eff_y < g["y0"] - 0.3:
                phase = 3                       # anchor on the flat
                released_k = landed_k = k
            held = {KEY_UP} if phase == 0 else ({KEY_DOWN} if phase == 2 else set())
        else:
            drive = KEY_UP if course == "up_off" else KEY_DOWN
            overrun = 1.5 if course == "back_off_fast" else 0.05
            # BOTH wheels must clear the edge line before releasing: with an
            # oblique edge the support midpoint crosses while the uphill
            # wheel is still on top — releasing there parked the robot in a
            # one-wheel-up straddle (roll −29° held 0.6 s, then tipped).
            _, yl_l = platform_local(float(sim.d.xpos[sim.lw][0]),
                                     float(sim.d.xpos[sim.lw][1]), g)
            _, yl_r = platform_local(float(sim.d.xpos[sim.rw][0]),
                                     float(sim.d.xpos[sim.rw][1]), g)
            if released_k is None and min(yl_l, yl_r) > g["pl"] + overrun:
                released_k = k
            held = set() if released_k is not None else {drive}
        if letgo_k is not None and k - letgo_k < 50:
            held = set()
        sig = sim.shaper.update_held(held)
        if sig == "ANCHOR":
            sx0, sy0 = sim.support_xy()
            sim.shaper.stop_here(sx0, sy0, sim.yaw())
        sim.shaper.events.clear()
        sim.step_teleop()

        r_deg, p_deg = sim.rpy()
        gz = ground_z_xy(sup_x, sy, g)
        relz = float(sim.d.qpos[2]) - gz
        logs["y"].append(eff_y)
        logs["relz"].append(relz)
        logs["pitch"].append(p_deg)
        logs["roll"].append(r_deg)
        logs["vxy"].append(float(np.hypot(sim.d.qvel[0], sim.d.qvel[1])))
        xl_sup, yl_sup2 = platform_local(sup_x, sy, g)
        if abs(xl_sup) <= g["pw"] and abs(yl_sup2) <= g["pl"] and relz > 0.3:
            reached_top = True
        # Landing = the flight past the edge has ended (NOT keyed to release:
        # back_off_fast keeps driving well past the landing).
        if yl_sup2 > g["pl"] + 0.05:
            past_edge = True
        if past_edge and sim.ctx.get("airborne_mode", False):
            flew = True
        # A very oblique low ledge can be descended with NO flight phase (one
        # wheel always grounded), so "landed" = past the edge, gate idle, and
        # standing over the LOWER floor — flight optional.
        if course != "up_down" and landed_k is None \
                and (flew or (released_k is not None and gz == 0.0 and relz < 0.60)) \
                and not sim.ctx.get("airborne_mode", False) \
                and sim.ctx.get("airborne_frac", 0.0) == 0.0:
            landed_k = k
        if landed_k is not None or (course == "up_down" and phase >= 2):
            peak_pitch_land = max(peak_pitch_land, abs(p_deg))
        if relz < 0.15 or abs(p_deg) > 75 or abs(r_deg) > 60:
            return dict(h_cm=h * 100, fell=True, fall_t=k * DT, reached_top=reached_top,
                        verdict="FALL")
        if settle_k is None and landed_k is not None and k > landed_k + 60:
            pw = np.array(logs["pitch"][k - 50:k])
            rw = np.array(logs["roll"][k - 50:k])
            if (pw.max() - pw.min() < 1.0 and rw.max() - rw.min() < 1.0
                    and np.abs(logs["vxy"][k - 50:k]).max() < 0.08):
                settle_k = k
        if frames is not None and k % frame_stride == 0:
            cam.lookat[1] = 0.6 * cam.lookat[1] + 0.4 * sy
            cam.lookat[2] = 0.6 * cam.lookat[2] + 0.4 * (gz + 0.35)
            renderer.update_scene(sim.d, camera=cam)
            frames.append((renderer.render().copy(), k * DT, p_deg, relz))
    L = {k2: np.array(v) for k2, v in logs.items()}
    tail = slice(-200, None)
    res = dict(
        h_cm=h * 100, fell=False, reached_top=reached_top,
        t_release=released_k * DT if released_k else None,
        peak_pitch_land=peak_pitch_land,
        settle_s=(settle_k - landed_k) * DT if settle_k and landed_k else None,
        pitch_tail=float(np.abs(L["pitch"][tail]).mean()),
        relz_err_tail=float(abs(L["relz"][tail].mean() - 0.537)),
        still_vel=float(np.sqrt((L["vxy"][tail] ** 2).mean())),
        y_final=float(L["y"][-1]),
    )
    res["verdict"] = "PASS" if (
        reached_top and res["settle_s"] is not None
        and res["pitch_tail"] <= 4.0 and res["still_vel"] <= 0.05
    ) else "FAIL"
    return res


def _write_video(frames, path: Path, h: float):
    from PIL import Image, ImageDraw
    path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-s", "640x480", "-r", "50", "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(path)],
        stdin=subprocess.PIPE)
    for arr, t, roll, z in frames:
        img = Image.fromarray(arr)
        ImageDraw.Draw(img).text(
            (10, 10), f"buc {h*100:.0f} cm   t={t:5.2f}s   z={z:.2f}m   "
            f"roll={roll:+5.1f} deg", fill=(255, 255, 60))
        proc.stdin.write(np.asarray(img, dtype=np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"   wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heights", default="20,30,40,50", help="ledge heights in cm")
    ap.add_argument("--course", default="up_off",
                    choices=["up_off", "up_down", "back_off", "back_off_fast",
                             "diag_off", "curb"])
    ap.add_argument("--angle", type=float, default=30.0,
                    help="edge angle (deg) for diag_off")
    ap.add_argument("--render", action="store_true", help="write MP4 per height")
    ap.add_argument("--out-dir", default="outputs/visual")
    args = ap.parse_args()
    heights = [float(x) / 100.0 for x in args.heights.split(",")]
    print(f"course: {args.course}")
    if args.course == "curb":
        print(f"{'h(cm)':>6} {'verdict':>8} {'roll_max':>9} {'d_max_cm':>9} "
              f"{'settle_s':>8} {'pitch_tl':>8} {'roll_tl':>8} {'still':>6}")
    else:
        print(f"{'h(cm)':>6} {'verdict':>8} {'top?':>5} {'t_edge':>7} {'pk_pitch':>9} "
              f"{'settle_s':>8} {'pitch_tl':>8} {'herr_mm':>8} {'still':>6}")
    n_pass = 0
    for h in heights:
        frames = cam = renderer = None
        if args.render:
            from PIL import Image, ImageDraw
            if args.course == "curb":
                xml, _ = build_curb_xml(h)
            else:
                xml, _ = build_terrain_xml(
                    h, sign=-1 if args.course.startswith("back_off") or args.course == "diag_off" else 1,
                    edge_angle_deg=args.angle if args.course == "diag_off" else 0.0)
            m = mujoco.MjModel.from_xml_path(str(xml))
            xml.unlink()
            renderer = mujoco.Renderer(m, height=480, width=640)
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [0.0, 0.0, 0.4]
            cam.distance = 3.0
            cam.azimuth = 155.0
            cam.elevation = -12.0
            frames = []
        if args.course == "curb":
            r = run_curb(h, frames=frames, cam=cam, renderer=renderer)
            if r.get("fell"):
                print(f"{r['h_cm']:6.0f} {'FALL':>8} fall@{r['fall_t']:.1f}s")
            else:
                n_pass += r["verdict"] == "PASS"
                st = f"{r['settle_s']:.2f}" if r["settle_s"] else "never"
                print(f"{r['h_cm']:6.0f} {r['verdict']:>8} {r['roll_curb_max']:9.1f} "
                      f"{r['d_max']*100:9.1f} {st:>8} {r['pitch_tail']:8.1f} "
                      f"{r['roll_tail']:8.1f} {r['still_vel']:6.3f}")
            if frames:
                _write_video(frames, Path(args.out_dir) / f"curb_{h*100:.0f}cm.mp4", h)
                renderer.close()
            continue
        r = run_ramp_step(h, frames=frames, cam=cam, renderer=renderer,
                          course=args.course, edge_angle_deg=args.angle)
        if r.get("fell"):
            print(f"{r['h_cm']:6.0f} {'FALL':>8} {str(r['reached_top'])[:1]:>5} "
                  f"{'—':>7} {'—':>9} {'—':>8} {'—':>8} {'—':>8} fall@{r['fall_t']:.1f}s")
        else:
            n_pass += r["verdict"] == "PASS"
            st = f"{r['settle_s']:.2f}" if r["settle_s"] else "never"
            tr = f"{r['t_release']:.1f}" if r["t_release"] else "—"
            print(f"{r['h_cm']:6.0f} {r['verdict']:>8} {str(r['reached_top'])[:1]:>5} "
                  f"{tr:>7} {r['peak_pitch_land']:9.1f} {st:>8} {r['pitch_tail']:8.1f} "
                  f"{r['relz_err_tail']*1000:8.1f} {r['still_vel']:6.3f}")
        if frames:
            from PIL import Image, ImageDraw
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            suffix = "" if args.course == "up_off" else f"_{args.course}"
            if args.course == "diag_off":
                suffix += f"{args.angle:.0f}"
            path = out / f"ramp_step_{h*100:.0f}cm{suffix}.mp4"
            proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                 "-pix_fmt", "rgb24", "-s", "640x480", "-r", "50", "-i", "-",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(path)],
                stdin=subprocess.PIPE)
            for arr, t, p_deg, relz in frames:
                img = Image.fromarray(arr)
                ImageDraw.Draw(img).text(
                    (10, 10), f"buc {h*100:.0f} cm   t={t:5.2f}s   "
                    f"h_rel={relz:.2f}m   pitch={p_deg:+5.1f} deg",
                    fill=(255, 255, 60))
                proc.stdin.write(np.asarray(img, dtype=np.uint8).tobytes())
            proc.stdin.close()
            proc.wait()
            renderer.close()
            print(f"   wrote {path}")
    print(f"\n{n_pass}/{len(heights)} PASS")


if __name__ == "__main__":
    main()
