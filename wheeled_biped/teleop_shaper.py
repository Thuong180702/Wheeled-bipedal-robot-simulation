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
    setups (same mechanism as scripts/viz_v3_homing_height.py).

    Uses forward kinematics (cos/sin of hip_pitch + knee) to map between
    CoM height h and actual leg length. The old linear h→joint→length chain
    produced wrong leg lengths at extrapolated heights: at h=0.256 (98% below
    z_lo) linear joint extrapolation gave knee=2.64 rad (151°) which folded
    the shin back under the thigh — the actual length was 12 cm shorter than
    intended (2026-07-24 audit)."""

    # Leg link lengths from the MuJoCo model (wheeled_biped_real.xml):
    # l_thigh → l_knee_link: 0.26 m, l_knee_link → l_wheel_link: 0.28 m
    THIGH_LEN = 0.26   # m, hip_pitch to knee
    SHIN_LEN = 0.28    # m, knee to wheel centre

    def __init__(self):
        lo = json.load(open(_DX / "dynamic_low_5cm__variant_setup.json"))
        hi = json.load(open(_DX / "dynamic_high_5cm__variant_setup.json"))
        def _posture(hs):
            return np.array([
                hs["hip_roll_left"], hs["hip_yaw_left"], hs["hip_pitch_ref"], hs["knee_ref"], 0.0,
                hs["hip_roll_right"], hs["hip_yaw_right"], hs["hip_pitch_ref"], hs["knee_ref"], 0.0])
        self.q_lo, self.q_hi = _posture(lo), _posture(hi)
        self.z_lo, self.z_hi = float(lo["target_com_z_m"]), float(hi["target_com_z_m"])

        # Build h ↔ leg_length lookup using forward kinematics.
        # The old code used h directly as a proxy for leg length (linear in
        # joint space); actual leg length is a nonlinear function of the two
        # joint angles.
        self._build_length_table()

    @staticmethod
    def _leg_length_from_joints(hip_pitch: float, knee: float) -> float:
        """Forward kinematics: vertical distance from hip to wheel.

        The hip-pitch joint rotates the thigh; the knee joint bends the shin
        BACKWARD relative to the thigh (verified on the MuJoCo model: at the
        nominal h=0.404 pose hp=0.926 kn=1.734 → thigh contributes 0.156 m,
        shin contributes 0.193 m, total 0.349 m — the hip is ~35 cm above
        the wheel, matching the kinematics)."""
        return (HeightPosture.THIGH_LEN * np.cos(hip_pitch)
                + HeightPosture.SHIN_LEN * np.cos(hip_pitch - knee))

    def _build_length_table(self):
        """Precompute h → leg_length over the full extrapolated range."""
        margin = 0.20  # generous margin beyond z_lo/z_hi
        self._h_min = self.z_lo - margin
        self._h_max = self.z_hi + margin
        n = 2000
        self._h_table = np.linspace(self._h_min, self._h_max, n)
        self._len_table = np.zeros(n)
        for i in range(n):
            q = self.q_ref(float(self._h_table[i]), clip=False)
            # Use left leg joints (indices 2=hp, 3=knee) — same as right
            self._len_table[i] = self._leg_length_from_joints(
                float(q[2]), float(q[3]))
        # Verify monotonicity (longer h → longer leg)
        if not np.all(np.diff(self._len_table) > 0):
            # Find non-monotonic region
            diffs = np.diff(self._len_table)
            bad = np.where(diffs <= 0)[0]
            print(f"[HeightPosture] WARNING: leg_length not monotonic "
                  f"at {len(bad)}/{n} points (h={self._h_table[bad[:3]]})")

    def leg_length(self, h: float) -> float:
        """Actual leg length produced by q_ref(h)."""
        return float(np.interp(h, self._h_table, self._len_table))

    def h_from_leg_length(self, target_length: float) -> float:
        """Find h such that q_ref(h) produces target_length (within bounds)."""
        lo_len = self._len_table[0]
        hi_len = self._len_table[-1]
        clamped = float(np.clip(target_length, lo_len, hi_len))
        h = float(np.interp(clamped, self._len_table, self._h_table))
        return h

    def q_ref(self, h: float, clip: bool = True) -> np.ndarray:
        s = (h - self.z_lo) / (self.z_hi - self.z_lo)
        if clip:
            s = np.clip(s, 0.0, 1.0)
        return self.q_lo + float(s) * (self.q_hi - self.q_lo)

    def q_ref_pair(self, h_left: float, h_right: float) -> np.ndarray:
        """Per-leg posture: LEFT leg joints for h_left, RIGHT for h_right.

        q_ref_pair(h, h) == q_ref(h) exactly for h inside the calibrated
        range (flat ground degenerates to the symmetric posture). Outside it
        the table is EXTRAPOLATED (LegTerrainAdapter.EXT_M bounds this to
        ±5 cm; joint-limit margins verified: knee 2.35 < 2.7 at the fold
        extreme) so a leg pair can span up to 20 cm of lateral step."""
        return np.concatenate([self.q_ref(h_left, clip=False)[:5],
                               self.q_ref(h_right, clip=False)[5:]])


def measure_wheel_ground(model, data, lw_body: int, rw_body: int):
    """Per-wheel (load_N, ground_z or None) from MuJoCo contact data.

    Ground height = force-weighted mean CONTACT-POINT z. Deriving it from the
    wheel-center z is only valid near level: at 20-38° roll (a lateral-push
    catch) the loaded wheel's center rises ~2 cm geometrically, which read as
    phantom high ground and folded exactly the leg carrying all the weight
    (measured: roll ran 16°→62°). The contact point is exact at any tilt.
    Force sign via abs(): mj_contactForce's world sign flips with geom
    ordering (plane vs terrain-box contacts)."""
    import mujoco
    acc = {lw_body: [0.0, 0.0], rw_body: [0.0, 0.0]}   # [sum_F, sum_F*z]
    f6 = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        b = b1 if b1 in acc else (b2 if b2 in acc else None)
        if b is None:
            continue
        mujoco.mj_contactForce(model, data, i, f6)
        fz = abs(float((np.array(c.frame).reshape(3, 3).T @ f6[:3])[2]))
        acc[b][0] += fz
        acc[b][1] += fz * float(c.pos[2])
    def _gz(b):
        s, sz = acc[b]
        return (sz / s) if s > 1e-9 else None
    return acc[lw_body][0], acc[rw_body][0], _gz(lw_body), _gz(rw_body)


class LegTerrainAdapter:
    """Independent per-leg terrain adaptation (curbs, oblique ledges).

    Each leg tracks ITS OWN ground height estimate so the torso stays LEVEL
    across lateral height steps instead of rolling:

    - wheel in contact  → its ground is measured (wheel z − flat-stance z0),
      lightly low-passed against contact chatter;
    - ONE wheel airborne → its ground estimate slews DOWNWARD at SEEK_RATE:
      that leg extends looking for the ground while the grounded leg folds
      (the shared torso descends), until contact returns or the posture
      envelope clamps — the step-down reflex;
    - BOTH wheels airborne (ballistic flight) → both estimates track the mean
      wheel height, reproducing the pre-adapter symmetric behavior exactly.

    Output: per-leg posture heights (h_cmd ± Δground/2, clamped to the
    calibrated envelope), the mean ground offset for the controller's CoM
    height command, and the UNCOMPENSATED height difference (residual beyond
    the ±(z_hi−z_lo)/... envelope) which the torso must absorb as roll —
    harnesses subtract expected_roll from the letgo/servo roll inputs so a
    legitimate straddle is not mistaken for a fall.
    """

    SEEK_RATE = 0.50     # m/s base downward ground-seek while a wheel is airborne
    SEEK_K = 50.0        # 1/s proportional boost: rate = SEEK_RATE + drop_m * K
    # (0.35 lost the race on the 20 cm one-wheel curb dismount: the unloaded
    # wheel hung ~11 cm above the floor while the robot tipped at 24° —
    # the extending leg reached ground 0.14 s too late. Proportional seek
    # fixes this: a 15 cm drop gets ~8.0 m/s (16× faster), converging the
    # ground estimate in ~2 steps (0.02 s) — the leg extends DURING the
    # fall instead of waiting for contact.)
    G_LP = 0.35          # per-step low-pass on measured ground (chatter)
    TRACK_M = 0.278      # lateral wheel separation (measured at settle)
    EXT_M = 0.10         # per-leg extrapolation beyond the calibrated range
    # Raised from 0.08 (2026-07-24 audit): with the corrected split() formula
    # the curb leg needs h=0.254 (0.404−0.15 after clipping) at nominal stance.
    # 0.10 gives range [0.254, 0.554] = 0.30 m. Knee at 0.254→2.55 rad (146°),
    # verified under the 2.7 rad hardware limit. Full 20 cm curb at nominal
    # stance requires knee at 0.204→2.96 rad (170°) which VIOLATES the 2.7 rad
    # limit — the residual shows as ~10° expected_roll, handled by the letgo gates.

    LVL_RATE = 0.15      # m/(rad·s): closed-loop leveling integrator gain
    LVL_MAX = 0.06       # m: leveling trim authority

    def __init__(self, hp: HeightPosture):
        self.hp = hp
        self.g = [0.0, 0.0]        # est ground under [left, right] wheel
        self.lvl = 0.0             # leg-split leveling trim (m, +: left longer)
        self.expected_roll = 0.0   # last computed uncompensated-roll (rad)
        self._prev_d = 0.0         # previous ground diff for rate-of-change gate

    def update(self, dt: float, loaded_l: bool, loaded_r: bool,
               gz_l: float, gz_r: float, roll_rad: float = 0.0) -> dict:
        """Advance the per-wheel ground estimates.

        gz_i = wheel-center z minus the flat-stance wheel z0 (per wheel).
        loaded_i must mean REAL load (normal force ≥ ~0.2·mg), not mere
        touching: a lightly-touching unloading wheel lifts a few cm and
        polluted the estimate (+2.9 cm phantom ground on the 15 cm curb);
        and a 50-100 ms contact flap during a wobble must FREEZE the
        estimate, not trigger the down-seek (an instant seek dropped the
        curb wheel's ground 15→9 cm and pumped the legs the wrong way)."""
        meas = (gz_l, gz_r)
        con = (loaded_l, loaded_r)
        if not (con[0] or con[1]):
            # ballistic: both track the mean wheel height (pre-adapter
            # behavior); pre-arm the seek so a staggered landing extends the
            # still-airborne leg immediately.
            m = 0.5 * (meas[0] + meas[1])
            self.g[0] += self.G_LP * (m - self.g[0])
            self.g[1] += self.G_LP * (m - self.g[1])
        else:
            span = self.hp.z_hi - self.hp.z_lo + 2 * self.EXT_M
            for i in (0, 1):
                if con[i]:
                    self.g[i] += self.G_LP * (meas[i] - self.g[i])
                else:
                    # The ONLY physically certain "my ground is gone" signal
                    # is the wheel dropping BELOW its believed ground (>3 mm):
                    # step-offs and oblique-ledge exits all start that way.
                    # An unloaded wheel HOVERING at/above ground level is a
                    # maneuver (push recovery, turn roll) — a timer-based
                    # hanging-seek on that signature extended the leg mid
                    # push-catch and fell (flat battery lateral_push_cruise).
                    # Unloaded-but-not-fallen → FREEZE the estimate.
                    if meas[i] < self.g[i] - 0.003:
                        drop = self.g[i] - meas[i]  # > 0.003 m
                        # Proportional seek: a small 3 mm wobble gets the base
                        # SEEK_RATE; a 15 cm curb drop gets ~1.7 m/s (3.4×
                        # faster), converging the ground estimate in ~0.09 s
                        # instead of 0.30 s — the leg extends DURING the fall.
                        rate = self.SEEK_RATE + self.SEEK_K * drop
                        self.g[i] = max(min(self.g[i] - rate * dt,
                                            meas[i]),
                                        self.g[1 - i] - span)
        # Closed-loop leveling: the h→leg-length table is not perfectly
        # linear (and is extrapolated at the extremes), so a "level" split
        # left a +6° systematic torso roll on the 10 cm curb — enough to trip
        # the 4° letgo forever. Trim the split by the MEASURED roll instead
        # of trusting the table: positive roll (left side low) → left leg
        # longer. Integrates ONLY on a real ground step (|d| ≥ 2 cm) with
        # both wheels loaded — on flat ground the dynamic roll of turning is
        # NOT a terrain signal, and integrating it wound the legs into an
        # 8.6° roll bias through a 180° turn (flat battery regression).
        # Off-step the trim decays back to zero. Also gate on ground-estimate
        # stability: during a curb climb the transient roll is geometric, not
        # a leveling error — integrating it compresses the flat leg (2026-07-24 audit).
        _d_now = self.g[0] - self.g[1]
        _d_rate = abs(_d_now - self._prev_d) / max(dt, 1e-6)
        self._prev_d = _d_now
        _d_stable = _d_rate < 0.15  # m/s — ground estimate changing slower than 15 cm/s
        if con[0] and con[1] and abs(_d_now) >= 0.02 and _d_stable:
            self.lvl = float(np.clip(self.lvl + self.LVL_RATE * dt * roll_rad,
                                     -self.LVL_MAX, self.LVL_MAX))
        else:
            self.lvl -= self.lvl * min(1.0, dt / 1.0)
        return dict(g_mid=0.5 * (self.g[0] + self.g[1]),
                    d=_d_now)

    def split(self, h_cmd: float) -> tuple[float, float]:
        """Per-leg posture heights for a commanded torso height.

        Strategy (2026-07-24 audit, kinematic):
        1. Compute ACTUAL leg length of the flat-ground leg at h_cmd via
           forward kinematics (cos/sin of hip_pitch + knee), not the old
           linear h proxy.  The linear proxy was off by 12 cm at the
           extrapolated limit (h=0.256) — the knee folded back under the
           thigh and the leg was far too short.
        2. The higher-ground leg needs to be SHORTER by the full curb
           height |d|:  L_curb = L_flat - |d|.
        3. Find h_curb = h_from_leg_length(L_curb).  If that h falls below
           the joint-limit floor lo, RAISE the robot (extend the flat leg)
           until the curb leg fits — the taller stance gives the bent knee
           more room.
        4. If the flat leg reaches hi (max extension), accept residual
           roll — the robot physically cannot span this step level."""
        d = self.g[0] - self.g[1]          # left ground minus right ground
        lo, hi = self.hp.z_lo - self.EXT_M, self.hp.z_hi + self.EXT_M
        abs_d = abs(d)

        # Step 1: nominal leg length from the flat-ground leg at h_cmd
        L_flat = self.hp.leg_length(h_cmd)

        # Step 2: desired length for the leg on HIGHER ground
        L_curb_desired = L_flat - abs_d

        # Minimum leg length achievable (at the joint-limit floor lo)
        L_min = self.hp.leg_length(lo)

        # Step 3: auto-raise — if the curb leg would be shorter than
        # physically possible, extend the flat leg to raise the robot.
        if L_curb_desired < L_min:
            L_flat = L_min + abs_d  # raise until curb leg fits
            L_curb_desired = L_min

        # Map leg lengths back to h-space for q_ref lookup
        h_flat = float(np.clip(self.hp.h_from_leg_length(L_flat), lo, hi))
        h_curb = float(np.clip(self.hp.h_from_leg_length(L_curb_desired), lo, hi))

        # Step 4: assign to left/right + apply fine leveling trim
        if d > 0:   # left ground higher → left leg compressed
            h_l = float(np.clip(h_curb + 0.5 * self.lvl, lo, hi))
            h_r = float(np.clip(h_flat - 0.5 * self.lvl, lo, hi))
        else:       # right ground higher (or flat) → right leg compressed
            h_l = float(np.clip(h_flat + 0.5 * self.lvl, lo, hi))
            h_r = float(np.clip(h_curb - 0.5 * self.lvl, lo, hi))

        # Residual: height difference legs can't span, after FK correction.
        # Compare ACTUAL leg lengths.  d > 0 → left ground higher → left leg
        # must be shorter: perfect compensation means L_r − L_l = d.
        L_l = self.hp.leg_length(h_l)
        L_r = self.hp.leg_length(h_r)
        residual = d - (L_r - L_l)
        # Sign: left ground higher (residual > 0) tips the torso toward the
        # RIGHT = NEGATIVE measured roll (calibrated on the 15 cm curb trace).
        self.expected_roll = float(np.arctan2(-residual, self.TRACK_M))
        return h_l, h_r

# GLFW keycodes seen by mujoco.viewer key_callback (letters A-Z are all bound
# to viewer render toggles — arrows/paging keys are safe; teleop v1 lesson).
KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262
KEY_PGUP, KEY_PGDN, KEY_SPACE, KEY_HOME = 266, 267, 32, 268
KEY_X, KEY_BACKSPACE = 88, 259


class TeleopShaper:
    # Cruise steps and limits. Raised after the principled cruise fix
    # (wheel-velocity FF + position relief): the robot now tracks cmd_vx
    # accurately and stays stable well past the old 0.5 cap — swept to 1.5 m/s
    # fwd / 1.1 back with pitch_max only 5-6° (letgo at 12°) at every height.
    # Ceilings kept at 1.0/0.7 for a comfortable margin (pitch ~4.5° at 0.9).
    VX_STEP = 0.15          # m/s per ↑/↓ press
    VX_MAX_FWD = 1.00
    VX_MAX_BACK = 0.70
    WZ_STEP = 0.30          # rad/s per ←/→ press (raised from 0.20 per user request)
    WZ_MAX = 1.00            # rad/s (raised from 0.60)
    ACC = 1.00              # m/s^2 slew of the applied vx (raised from 0.60 for
    # fast stopping; 1.00 stops from 0.80 m/s in 0.80 s commanded ramp.
    # ponytail: if spin→drive→reverse chains fall, back off to 0.75)
    WACC = 2.0              # rad/s^2 slew of the applied yaw rate
    # Speed-vs-turn coupling cap: straight-line drive may use the full raised
    # VX_MAX_FWD, but turning scrubs the translate speed toward TURN_DRIVE_CAP
    # (an ABSOLUTE limit, NOT scaled by VX_MAX_FWD). Turn-while-drive builds
    # roll on the weak axis; when the cap was tied to the raised max it let
    # turn-drive hit 0.54 m/s and a spin→drive-turn chain rolled over (roll
    # 3°→72°, measured). At the 0.83 turn fraction of that chain this gives
    # ~0.27 m/s — the old validated turn-drive speed.
    TURN_CAP_FLOOR = 0.12   # m/s always allowed
    TURN_DRIVE_CAP = 0.12   # m/s translate ceiling at FULL turn (|wz|=WZ_MAX)
    # Robot-frame leash on the integrated target (v2 lesson #5)
    LEAD = 0.09             # m ahead (below tau_position saturation 4Nm/k40=0.10)
    TRAIL = 0.30            # m behind (loose — the overshoot anchor must stay)
    LAT = 0.05              # m lateral (wheels cannot close lateral error)
    # (The sagittal cruise surge/brake limit cycle is fixed IN THE CONTROLLER —
    # wheel-velocity FF + cruise position relief — not by tightening this leash.
    # An earlier tight-cruise-leash band-aid was reverted: the principled fix
    # works with the loose anchor leash, which also rides out a mid-cruise push
    # better than a tight leash that would fight the ballistic catch.)
    # Yaw leash: bound the target heading to the robot's ACTUAL heading, exactly
    # as LEAD/TRAIL bound position. A sustained turn under-delivers (the robot
    # sustains ~0.2 rad/s for a 0.5 command), so an unleashed target yaw runs
    # away, the heading error winds up and the achieved rate DECAYS (measured
    # 0.25→0.14 over 5 s). Leashing keeps the FF effective and the rate steady.
    YAW_LEAD = 0.30         # rad (~17°)
    # Height command
    H_STEP = 0.010          # m per PgUp/PgDn press
    # Command ceiling sits 9 mm below the posture-table edge (0.454): at the
    # edge the interpolation clips at s=1.0 and the height servo has no
    # posture headroom left to close the ~9 mm gravity sag (measured).
    H_MIN, H_MAX = 0.354, 0.445
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
        # Pitch gate 5°: the tall-stance EQUILIBRIUM pitch is ~3°+, which a
        # 2.9° gate mistook for motion and froze the servo at max height
        # (9 mm standing error, measured). Roll stays tight (2°) — that is
        # the positive-feedback axis.
        if not self.busy and abs(pitch_rad) < 0.09 and abs(roll_rad) < 0.035:
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
        """Re-anchor the target at the CURRENT pose."""
        self.tx, self.ty, self.tyaw = float(x), float(y), float(yaw)

    # ── hold-to-drive mode (real press/release via pynput; the MuJoCo viewer
    # delivers no release events, so hold semantics are impossible there) ──
    VX_HOLD_FWD = 0.80      # m/s while ↑ held (2× the old 0.40; stable to 1.5)
    VX_HOLD_BACK = 0.65     # m/s while ↓ held (raised from 0.45; higher damping
    # needs more speed to break through. ponytail: if spin→reverse chain falls,
    # back off to 0.55)
    WZ_HOLD = 0.50          # rad/s while ←/→ held

    def update_held(self, held: set) -> str | None:
        """Map the currently-held key set to cruise setpoints.

        Release-to-stop: when every drive key is released, the cruise zeroes
        and the caller must re-anchor at the robot's CURRENT pose (returns
        "ANCHOR" once on that transition). Height keys RAMP while held
        (h_tgt slews toward the envelope edge at H_SLEW via step()).
        """
        up, dn = KEY_UP in held, KEY_DOWN in held
        lf, rt = KEY_LEFT in held, KEY_RIGHT in held
        pu, pd = KEY_PGUP in held, KEY_PGDN in held
        # Safety let-go LATCH: after a let-go the drive stays suppressed until
        # every drive key is RELEASED (else a still-held ↑ re-applies the
        # cruise on the very next step and drives through the recovery —
        # measured fall). Like a motor-fault latch: lift off, then re-press.
        if getattr(self, "letgo_latch", False):
            if not (up or dn or lf or rt):
                self.letgo_latch = False
            else:
                up = dn = lf = rt = False
        was_driving = (self.vx_tgt != 0.0 or self.wz_tgt != 0.0)
        self.vx_tgt = (self.VX_HOLD_FWD if up and not dn
                       else -self.VX_HOLD_BACK if dn and not up else 0.0)
        self.wz_tgt = (self.WZ_HOLD if lf and not rt
                       else -self.WZ_HOLD if rt and not lf else 0.0)
        if pu and not pd:
            self.h_tgt = self.H_MAX
        elif pd and not pu:
            self.h_tgt = self.H_MIN
        else:
            self.h_tgt = self.h   # freeze height where the ramp stopped
        # Start-driving → re-anchor at current pose so the anchor integral
        # doesn't fight the initial movement (was latched at the old position).
        now_driving = (self.vx_tgt != 0.0 or self.wz_tgt != 0.0)
        if not was_driving and now_driving:
            self.events.append("START->ANCHOR")
            return "ANCHOR"
        if was_driving and not now_driving:
            self.events.append("RELEASE->ANCHOR")
            return "ANCHOR"
        return None

    # Safety let-go: a human driver releases the stick when shoved; the cruise
    # must not keep driving THROUGH a disturbance recovery (measured: a 50 N
    # lateral push while cruising at 0.3 m/s fell where the same push at
    # standstill is far inside the envelope). When tilt exceeds the limits the
    # cruise zeroes and the target re-anchors at the current pose — the ANCHOR
    # stack recovers, the driver re-commands afterwards.
    LETGO_PITCH = 0.40      # rad (~23°) — raised for curb ramp climbing (12° ramp + pitch)
    # acceleration with higher damping causes 13-15° pitch; forward braking stays
    # ~10°. A real disturbance spikes past 30° instantly — the extra 5° headroom
    # doesn't change catch timing.
    LETGO_ROLL = 0.07       # rad (~4°) — roll rises FAST under a lateral shove
    # at speed (the weak axis); normal STRAIGHT maneuvers stay ≤ ~3°
    LETGO_ROLL_TURN = 0.12  # rad (~7°) — a sustained turn builds ~4° roll on its
    # own (dynamic turn roll bias), which tripped the 4° straight-line limit and
    # killed every turn via the latch (measured); a real shove still spikes past
    # 7° much faster than turn roll accumulates

    # ── per-control-step update ──
    def step(self, dt: float, sup_x: float, sup_y: float, est_yaw: float,
             pitch_rad: float = 0.0, roll_rad: float = 0.0) -> dict:
        roll_lim = self.LETGO_ROLL_TURN if self.wz_tgt != 0.0 else self.LETGO_ROLL
        if (abs(pitch_rad) > self.LETGO_PITCH or abs(roll_rad) > roll_lim) \
                and (self.vx_tgt != 0.0 or self.wz_tgt != 0.0):
            self.vx_tgt = self.wz_tgt = 0.0
            self.vx = self.wz = 0.0
            self.stop_here(sup_x, sup_y, est_yaw)
            self.letgo_latch = True   # hold-mode: require full release to re-arm
            self.events.append("SAFETY_LETGO")
        # Slew applied cruise toward setpoints
        def slew(cur, tgt, rate):
            d = np.clip(tgt - cur, -rate * dt, rate * dt)
            return cur + d
        # Speed-vs-turn cap (continuous): full VX_MAX_FWD straight, scrubbed to
        # the absolute TURN_DRIVE_CAP at full turn (decoupled from the max).
        turn_frac = min(abs(self.wz_tgt) / self.WZ_MAX, 1.0)
        vx_cap = max(self.TURN_CAP_FLOOR,
                     self.VX_MAX_FWD - (self.VX_MAX_FWD - self.TURN_DRIVE_CAP) * turn_frac)
        vx_goal = float(np.clip(self.vx_tgt, -min(self.VX_MAX_BACK, vx_cap), min(self.VX_MAX_FWD, vx_cap)))
        self.vx = float(slew(self.vx, vx_goal, self.ACC))
        self.wz = float(slew(self.wz, self.wz_tgt, self.WACC))

        # Integrate target pose along the TARGET heading
        self.tyaw = float(np.arctan2(np.sin(self.tyaw + self.wz * dt),
                                     np.cos(self.tyaw + self.wz * dt)))
        # Yaw leash: bound the target heading to the robot's actual heading.
        dyaw = float(np.arctan2(np.sin(self.tyaw - est_yaw),
                                np.cos(self.tyaw - est_yaw)))
        dyaw = float(np.clip(dyaw, -self.YAW_LEAD, self.YAW_LEAD))
        self.tyaw = float(np.arctan2(np.sin(est_yaw + dyaw),
                                     np.cos(est_yaw + dyaw)))
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
