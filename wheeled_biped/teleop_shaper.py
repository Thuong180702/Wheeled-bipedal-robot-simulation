"""Teleop command shaper for the V3_ANCHOR controller.

Press-driven cruise model (teleop v1/v2 lesson: the MuJoCo passive viewer
delivers NO key-release events and unreliable auto-repeat, so any hold-to-drive
model fights the OS keyboard. Each key PRESS steps a latched setpoint instead):

    ↑ / ↓   step the velocity cruise setpoint (fwd/back)
    ← / →   step the yaw-rate cruise setpoint (CCW/CW)
    Space   full stop: zero cruise, target ← current pose (re-anchor here)
    PgUp    height +1 cm (fn+↑ on Mac keyboards)
    PgDn    height −1 cm (fn+↓)

The shaper integrates a world-frame TARGET pose (x, y, yaw) from the cruise
setpoints and leashes it to the robot (teleop v2 lesson: unbounded target
windup rails tau_position; asymmetric leash LEAD < saturation, TRAIL loose —
a tight symmetric leash removed the overshoot anchor and the WIP wandered).
The controller tracks the target through the ANCHOR stack.

Forward axis = Rz(yaw)·[0,1] = [-sin(yaw), cos(yaw)]  (verified teleop v1;
the [+sin, cos] form topples at 180°).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
_DX = _REPO / "archive/cleanup_2026-06-13/output_summaries/balance_core_extended_height_range"


class HeightPosture:
    """CoM height command → q_ref interpolated between the calibrated ±5 cm
    setups (same mechanism as scripts/viz_v3_homing_height.py)."""

    def __init__(self):
        lo = json.load(open(_DX / "dynamic_low_5cm__variant_setup.json"))
        hi = json.load(open(_DX / "dynamic_high_5cm__variant_setup.json"))
        def _posture(hs):
            return np.array([
                hs["hip_roll_left"], hs["hip_yaw_left"], hs["hip_pitch_ref"], hs["knee_ref"], 0.0,
                hs["hip_roll_right"], hs["hip_yaw_right"], hs["hip_pitch_ref"], hs["knee_ref"], 0.0])
        self.q_lo, self.q_hi = _posture(lo), _posture(hi)
        self.z_lo, self.z_hi = float(lo["target_com_z_m"]), float(hi["target_com_z_m"])

    def q_ref(self, h: float) -> np.ndarray:
        s = float(np.clip((h - self.z_lo) / (self.z_hi - self.z_lo), 0.0, 1.0))
        return self.q_lo + s * (self.q_hi - self.q_lo)

# GLFW keycodes seen by mujoco.viewer key_callback (letters A-Z are all bound
# to viewer render toggles — arrows/paging keys are safe; teleop v1 lesson).
KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262
KEY_PGUP, KEY_PGDN, KEY_SPACE, KEY_HOME = 266, 267, 32, 268
KEY_X, KEY_BACKSPACE = 88, 259


class TeleopShaper:
    # Cruise steps and limits
    VX_STEP = 0.10          # m/s per ↑/↓ press
    VX_MAX_FWD = 0.50
    VX_MAX_BACK = 0.35      # backward asymmetry above ~0.4 m/s (teleop v1)
    WZ_STEP = 0.20          # rad/s per ←/→ press
    WZ_MAX = 0.60
    ACC = 0.60              # m/s^2 slew of the applied vx toward the setpoint
    WACC = 2.0              # rad/s^2 slew of the applied yaw rate
    # Speed-vs-turn coupling cap (teleop v1: sustained circles tip at high
    # vx·wz; start conservative, battery re-validates on the fixed harness)
    TURN_CAP_FLOOR = 0.15   # m/s always allowed
    TURN_CAP_SLOPE = 0.55   # vx_cap = max(FLOOR, VX_MAX - SLOPE*|wz|/WZ_MAX * VX_MAX)
    # Robot-frame leash on the integrated target (v2 lesson #5)
    LEAD = 0.09             # m ahead (below tau_position saturation 4Nm/k40=0.10)
    TRAIL = 0.30            # m behind (loose — the overshoot anchor must stay)
    LAT = 0.05              # m lateral (wheels cannot close lateral error)
    # Height command
    H_STEP = 0.010          # m per PgUp/PgDn press
    H_MIN, H_MAX = 0.354, 0.454   # validated dynamic envelope (±5 cm)
    H_SLEW = 0.04           # m/s height ramp

    def __init__(self, x: float, y: float, yaw: float, height: float):
        self.vx_tgt = 0.0
        self.wz_tgt = 0.0
        self.vx = 0.0
        self.wz = 0.0
        self.tx = float(x)
        self.ty = float(y)
        self.tyaw = float(yaw)
        self.h_tgt = float(np.clip(height, self.H_MIN, self.H_MAX))
        self.h = self.h_tgt
        self.h_trim = 0.0
        self.events: list[str] = []

    @property
    def busy(self) -> bool:
        return self.vx_tgt != 0.0 or self.wz_tgt != 0.0

    def height_servo(self, com_z: float, dt: float,
                     pitch_rad: float = 0.0, roll_rad: float = 0.0) -> float:
        """Slow posture-height trim closing the standing CoM to the command.

        The leg posture PD has no integral action, so the standing joints rest
        at history-dependent equilibria: after any drive the legs settle
        ~0.03-0.05 rad away and the CoM ends ~7 mm off the commanded height
        (measured). This integrates the measured CoM error into the POSTURE
        interpolation height only (the controller's height_ref stays the raw
        command).

        Adaptation runs ONLY at a truly settled stance: frozen while driving
        AND while tilted. A tilt drops the CoM geometrically — adapting on it
        extends the legs while the robot leans, a roll-axis POSITIVE feedback
        that turned a recoverable 40 N lateral shove into a fall (measured:
        roll +2° → +52° in 1.6 s with the servo raising the posture).
        """
        if not self.busy and abs(pitch_rad) < 0.05 and abs(roll_rad) < 0.035:
            err = self.h - float(com_z)
            self.h_trim = float(np.clip(self.h_trim + 0.5 * err * dt, -0.025, 0.025))
        return float(np.clip(self.h + self.h_trim,
                             self.H_MIN - 0.02, self.H_MAX + 0.02))

    # ── key handling (call from the viewer key_callback or a scenario script) ──
    def on_key(self, keycode: int) -> str | None:
        if keycode == KEY_UP:
            self.vx_tgt = min(self.vx_tgt + self.VX_STEP, self.VX_MAX_FWD)
            ev = f"vx_tgt={self.vx_tgt:+.2f}"
        elif keycode == KEY_DOWN:
            self.vx_tgt = max(self.vx_tgt - self.VX_STEP, -self.VX_MAX_BACK)
            ev = f"vx_tgt={self.vx_tgt:+.2f}"
        elif keycode == KEY_LEFT:
            self.wz_tgt = min(self.wz_tgt + self.WZ_STEP, self.WZ_MAX)
            ev = f"wz_tgt={self.wz_tgt:+.2f}"
        elif keycode == KEY_RIGHT:
            self.wz_tgt = max(self.wz_tgt - self.WZ_STEP, -self.WZ_MAX)
            ev = f"wz_tgt={self.wz_tgt:+.2f}"
        elif keycode == KEY_SPACE:
            self.vx_tgt = self.wz_tgt = 0.0
            self.vx = self.wz = 0.0
            ev = "STOP"
        elif keycode == KEY_PGUP:
            self.h_tgt = min(self.h_tgt + self.H_STEP, self.H_MAX)
            ev = f"h_tgt={self.h_tgt:.3f}"
        elif keycode == KEY_PGDN:
            self.h_tgt = max(self.h_tgt - self.H_STEP, self.H_MIN)
            ev = f"h_tgt={self.h_tgt:.3f}"
        else:
            return None
        self.events.append(ev)
        return ev

    def stop_here(self, x: float, y: float, yaw: float) -> None:
        """Space semantics: re-anchor the target at the CURRENT pose."""
        self.tx, self.ty, self.tyaw = float(x), float(y), float(yaw)

    # Safety let-go: a human driver releases the stick when shoved; the cruise
    # must not keep driving THROUGH a disturbance recovery (measured: a 50 N
    # lateral push while cruising at 0.3 m/s fell where the same push at
    # standstill is far inside the envelope). When tilt exceeds the limits the
    # cruise zeroes and the target re-anchors at the current pose — the ANCHOR
    # stack recovers, the driver re-commands afterwards.
    LETGO_PITCH = 0.21      # rad (~12°, the stability-gate zero) — normal hard
    # braking from 0.5 m/s peaks ~10°; an 8° threshold fired spuriously on
    # every hard stop and silently aborted commanded chains (measured)
    LETGO_ROLL = 0.07       # rad (~4°) — roll rises FAST under a lateral shove
    # at speed (the weak axis); normal maneuvers stay ≤ ~3°

    # ── per-control-step update ──
    def step(self, dt: float, sup_x: float, sup_y: float, est_yaw: float,
             pitch_rad: float = 0.0, roll_rad: float = 0.0) -> dict:
        if (abs(pitch_rad) > self.LETGO_PITCH or abs(roll_rad) > self.LETGO_ROLL) \
                and (self.vx_tgt != 0.0 or self.wz_tgt != 0.0):
            self.vx_tgt = self.wz_tgt = 0.0
            self.vx = self.wz = 0.0
            self.stop_here(sup_x, sup_y, est_yaw)
            self.events.append("SAFETY_LETGO")
        # Slew applied cruise toward setpoints
        def slew(cur, tgt, rate):
            d = np.clip(tgt - cur, -rate * dt, rate * dt)
            return cur + d
        # Speed-vs-turn cap (continuous)
        vx_cap = max(self.TURN_CAP_FLOOR,
                     self.VX_MAX_FWD * (1.0 - self.TURN_CAP_SLOPE * abs(self.wz_tgt) / self.WZ_MAX))
        vx_goal = float(np.clip(self.vx_tgt, -min(self.VX_MAX_BACK, vx_cap), min(self.VX_MAX_FWD, vx_cap)))
        self.vx = float(slew(self.vx, vx_goal, self.ACC))
        self.wz = float(slew(self.wz, self.wz_tgt, self.WACC))

        # Integrate target pose along the TARGET heading
        self.tyaw = float(np.arctan2(np.sin(self.tyaw + self.wz * dt),
                                     np.cos(self.tyaw + self.wz * dt)))
        fwd = np.array([-np.sin(self.tyaw), np.cos(self.tyaw)])
        self.tx += float(fwd[0] * self.vx * dt)
        self.ty += float(fwd[1] * self.vx * dt)

        # Leash in the ROBOT frame (relative to the support center, using the
        # robot's estimated heading): lead/trail along forward, lat sideways.
        rel = np.array([self.tx - sup_x, self.ty - sup_y])
        rf = np.array([-np.sin(est_yaw), np.cos(est_yaw)])   # robot forward
        rl = np.array([-rf[1], rf[0]])                        # robot left
        a = float(rel @ rf)   # ahead(+)/behind(−)
        b = float(rel @ rl)   # left(+)/right(−)
        a_c = float(np.clip(a, -self.TRAIL, self.LEAD))
        b_c = float(np.clip(b, -self.LAT, self.LAT))
        if a_c != a or b_c != b:
            p = np.array([sup_x, sup_y]) + rf * a_c + rl * b_c
            self.tx, self.ty = float(p[0]), float(p[1])

        # Height slew
        dh = np.clip(self.h_tgt - self.h, -self.H_SLEW * dt, self.H_SLEW * dt)
        self.h = float(self.h + dh)

        return {
            "teleop_active": 1.0,
            "teleop_cmd_vx_m_s": self.vx,
            "teleop_target_x_m": self.tx,
            "teleop_target_y_m": self.ty,
            "teleop_target_yaw_rad": self.tyaw,
            "teleop_cmd_yaw_rate_rad_s": self.wz,
            "height_ref": self.h,
        }
