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


def build_terrain_xml(h: float, sign: int = 1) -> tuple[Path, dict]:
    """Inject ramp + platform boxes into the robot XML; returns (path, geometry).

    sign=+1 lays the course toward +Y (forward); sign=-1 mirrors it toward -Y
    (the robot drives it BACKWARD). Geometry dict stores unsigned distances.
    """
    ramp_len = h / np.tan(SLOPE_RAD)
    y0, y1 = FLAT_END_Y, FLAT_END_Y + ramp_len
    y_edge = y1 + PLATFORM_LEN
    t = 0.05  # ramp slab half-thickness
    a = float(np.hypot(ramp_len, h)) / 2 + 0.02  # slight overlap into platform
    mid_y, mid_z = (y0 + y1) / 2, h / 2
    cy = mid_y + t * np.sin(SLOPE_RAD)
    cz = mid_z - t * np.cos(SLOPE_RAD)
    common = ('condim="6" contype="1" conaffinity="1" friction="0.8 0.005 0.0001" '
              'solref="0.002 1" solimp="0.95 0.99 0.001 0.5 2" rgba="0.55 0.45 0.35 1"')
    geoms = (
        f'    <geom name="ramp" type="box" size="{TERRAIN_HALF_W} {a} {t}" '
        f'pos="0 {sign * cy:.4f} {cz:.4f}" euler="{sign * SLOPE_RAD:.6f} 0 0" {common}/>\n'
        f'    <geom name="platform" type="box" size="{TERRAIN_HALF_W} {PLATFORM_LEN/2} {h/2}" '
        f'pos="0 {sign * (y1 + PLATFORM_LEN/2):.4f} {h/2:.4f}" {common}/>\n'
    )
    xml = MODEL_XML.read_text()
    out = MODEL_XML.parent / f"_tmp_ramp_step_{int(h*100)}_{'m' if sign < 0 else 'p'}.xml"
    out.write_text(xml.replace("  </worldbody>", geoms + "  </worldbody>"))
    return out, dict(y0=y0, y1=y1, y_edge=y_edge, h=h, sign=sign)


def ground_z(y: float, g: dict) -> float:
    if y < g["y0"]:
        return 0.0
    if y < g["y1"]:
        return (y - g["y0"]) * np.tan(SLOPE_RAD)
    if y < g["y_edge"]:
        return g["h"]
    return 0.0


class RampStepSim(TeleopSim):
    def __init__(self, xml_path: Path):
        orig = P.get_model_path
        P.get_model_path = lambda: xml_path
        try:
            super().__init__()
        finally:
            P.get_model_path = orig
        # Wheel-center height on flat ground (settled) — terrain offset zero ref.
        self._wheel_z0 = float(0.5 * (self.d.xpos[self.lw][2] + self.d.xpos[self.rw][2]))

    def terrain_offset(self) -> float:
        wz = 0.5 * (self.d.xpos[self.lw][2] + self.d.xpos[self.rw][2])
        return float(wz - self._wheel_z0)

    def step_teleop(self):
        # TeleopSim.step_teleop with the height command terrain-following.
        off = self.terrain_offset()
        sx, sy = self.support_xy()
        r_deg, p_deg = self.rpy()
        cmd = self.shaper.step(DT, sx, sy, self.yaw(),
                               pitch_rad=np.radians(p_deg), roll_rad=np.radians(r_deg))
        self.ctx["height_ref"] = cmd["height_ref"] + off
        h_post = self.shaper.height_servo(float(self.d.subtree_com[0][2]) - off, DT,
                                          pitch_rad=np.radians(p_deg), roll_rad=np.radians(r_deg))
        self.ctx["eq_joint"] = self.hp.q_ref(h_post)
        self._step(cmd)
        return cmd


def run_ramp_step(h: float, duration_s: float = 30.0, frames: list | None = None,
                  frame_stride: int = 2, cam=None, renderer=None, approach_vx: float = 1.0,
                  course: str = "up_off"):
    """Courses:
    up_off   — forward up the ramp, across the platform, off the ledge (default)
    up_down  — forward up the ramp, anchor mid-platform 2.5 s, then REVERSE
               back down the ramp to the flat
    back_off — the mirrored course driven BACKWARD end to end (rear-first
               ledge drop)
    back_off_fast — back_off at max reverse speed (0.70 m/s) and KEEP DRIVING
               ~1.5 m past the landing before stopping
    """
    from scripts.teleop_scenario_tests import KEY_DOWN
    sign = -1 if course.startswith("back_off") else 1
    xml, g = build_terrain_xml(h, sign=sign)
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
    if course == "back_off_fast":
        sim.shaper.VX_HOLD_BACK = 0.70  # shaper max reverse speed
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
            if released_k is None and eff_y > g["y_edge"] + overrun:
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
        gz = ground_z(eff_y, g)
        relz = float(sim.d.qpos[2]) - gz
        logs["y"].append(eff_y)
        logs["relz"].append(relz)
        logs["pitch"].append(p_deg)
        logs["roll"].append(r_deg)
        logs["vxy"].append(float(np.hypot(sim.d.qvel[0], sim.d.qvel[1])))
        if g["y1"] + 0.2 < eff_y < g["y_edge"] and relz > 0.3:
            reached_top = True
        # Landing = the flight past the edge has ended (NOT keyed to release:
        # back_off_fast keeps driving well past the landing).
        if eff_y > g["y_edge"] + 0.05:
            past_edge = True
        if past_edge and sim.ctx.get("airborne_mode", False):
            flew = True
        if course != "up_down" and landed_k is None and flew \
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heights", default="20,30,40,50", help="ledge heights in cm")
    ap.add_argument("--course", default="up_off",
                    choices=["up_off", "up_down", "back_off", "back_off_fast"])
    ap.add_argument("--render", action="store_true", help="write MP4 per height")
    ap.add_argument("--out-dir", default="outputs/visual")
    args = ap.parse_args()
    heights = [float(x) / 100.0 for x in args.heights.split(",")]
    print(f"course: {args.course}")
    print(f"{'h(cm)':>6} {'verdict':>8} {'top?':>5} {'t_edge':>7} {'pk_pitch':>9} "
          f"{'settle_s':>8} {'pitch_tl':>8} {'herr_mm':>8} {'still':>6}")
    n_pass = 0
    for h in heights:
        frames = cam = renderer = None
        if args.render:
            from PIL import Image, ImageDraw
            xml, _ = build_terrain_xml(h, sign=-1 if args.course.startswith("back_off") else 1)
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
        r = run_ramp_step(h, frames=frames, cam=cam, renderer=renderer,
                          course=args.course)
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
