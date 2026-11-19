"""Behaviour layer that decides *what* to do before FOT decides *how*.

The previous node ran FOT whenever any kart was within 15 m and, if FOT
found nothing, published nothing -- which made simple_pure_pursuit snap
straight back onto the raceline at full speed, i.e. into the kart ahead.
There was no notion of "stay behind and wait".

This module is a small state machine in Frenet coordinates of the lap
(see raceline.py). It has no ROS dependency so it can be exercised in a
closed-loop simulation without AWSIM.

States
------
CRUISE    Nothing in the way. Publish nothing: the controller follows the
          untouched reference trajectory.
FOLLOW    A kart is in the way and overtaking is not allowed or not
          feasible here. Follow the raceline behind it with an ACC-style
          speed limit (this is what was missing and caused the rear-end
          crashes).
OVERTAKE  Inside a human-defined overtaking zone, with enough zone left to
          complete the pass and enough room on one side. FOT plans the
          lateral manoeuvre; its sampled lateral targets are restricted to
          the chosen passing side so it cannot "wait" on the raceline.
RETURN    The pass is complete (the other kart is clearly behind). Blend
          smoothly back onto the raceline, then go to CRUISE.

FOT is still the path generator for the overtaking manoeuvre (option A).
When it momentarily finds nothing, a raceline-offset fallback path keeps
the kart where it is laterally instead of snapping back.
"""
import math
from dataclasses import dataclass, field

import numpy as np


CRUISE = "CRUISE"
FOLLOW = "FOLLOW"
OVERTAKE = "OVERTAKE"
RETURN = "RETURN"
INACTIVE = "INACTIVE"


@dataclass
class PlannerParams:
    # racing_kart: length 2.064 m (Car.h), width 1.30 m
    vehicle_length: float = 2.064
    vehicle_half_width: float = 0.65
    drivable_area_margin: float = 0.2
    # Ego further than this from the loop (e.g. on the garage out-lap):
    # planner is inactive and the reference trajectory is followed.
    on_loop_max_offset: float = 4.0

    # Which karts matter (along-track distance from the ego, metres).
    detect_ahead: float = 35.0
    detect_behind: float = 6.0
    # Centre-to-centre lateral distance under which a kart is "in the way"
    # of a path: 2 * half width (1.30) + 0.30 m of clearance.
    lateral_block: float = 1.6

    # ---- FOLLOW (ACC) ----
    follow_gap_min: float = 3.0     # bumper-to-bumper standstill gap [m]
    follow_time_gap: float = 0.5    # extra gap per m/s of lead speed [s]
    follow_decel: float = 2.5       # comfortable braking used for the
                                    # speed envelope [m/s^2]
    controller_lag: float = 1.0     # 1 / speed_proportional_gain [s]
    follow_close_gain: float = 0.4  # [1/s] gap closing near the lead
    follow_close_max: float = 1.5   # [m/s] max closing speed near the lead
    path_length: float = 15.0       # length of FOLLOW/RETURN paths [m]
                                    # (pure pursuit only looks ~2.5-5 m
                                    # ahead; longer just clutters RViz)

    # ---- OVERTAKE decision ----
    min_speed_advantage: float = 1.5     # [m/s] need to be this much faster
    ego_accel: float = 2.5               # assumed acceleration for the
                                         # feasibility estimate [m/s^2]
    pass_done_gap: float = 1.5           # ego centre must lead the other
                                         # kart's centre by length + this
    zone_end_margin: float = 8.0         # [m] zone left after the pass,
                                         # for the swing back to the line
    lateral_pass: float = 1.8            # centre-to-centre lateral offset
                                         # targeted while passing
    lateral_pass_extra: float = 0.5      # FOT samples up to pass+extra
    lateral_min: float = 1.5             # FOT samples down to this while
                                         # passing (its collision check
                                         # still decides what is safe)
    side_room_margin: float = 0.2        # extra room required to START
    max_overtake_time: float = 10.0      # give up after this long [s]
    retry_cooldown: float = 1.5          # after an abort [s]
    min_track_samples: int = 3           # V2X samples before trusting v

    # ---- RETURN ----
    return_blend_min: float = 10.0       # [m]
    return_blend_time: float = 1.2       # [s] -> blend = max(min, v * t)
    return_done_offset: float = 0.3      # [m]
    lateral_goal_tolerance: float = 0.3  # [m] goal change that re-plans
    lateral_replan_error: float = 0.6    # [m] deviation that re-plans

    # ---- FOT ----
    fot_lookahead: float = 36.0
    fot_lookbehind: float = 6.0
    fot_waypoint_spacing: float = 2.0
    obstacle_half_size: float = 0.45     # C++ adds obstacle_clearance on top
    obstacle_longitudinal_offsets: tuple = (-0.6, 0.0, 0.6)
    obstacle_prediction_time: float = 1.0
    obstacle_prediction_step: float = 1.0
    max_start_position_error: float = 2.0
    fot_keep_time: float = 0.8           # [s] a FOT path whose first
                                         # off-track point is later than
                                         # this is truncated, not rejected
    fot_hyperparameters: dict = field(default_factory=lambda: {
        "max_speed": 15.0, "max_accel": 5.0, "max_curvature": 0.7,
        "max_road_width_l": 2.0, "max_road_width_r": 2.0, "d_road_w": 0.25,
        "dt": 0.1, "maxt": 2.0, "mint": 1.0, "d_t_s": 1.0,
        "n_s_sample": 1.0, "obstacle_clearance": 0.25, "kd": 1.0,
        "kv": 0.1, "ka": 0.1, "kj": 0.1, "kt": 0.1, "ko": 0.5,
        "klat": 1.0, "klon": 1.0, "num_threads": 0})


@dataclass
class Opponent:
    vid: str
    x: float
    y: float
    s: float
    d: float
    v: float          # along-track speed estimate [m/s]
    v_valid: bool
    delta: float = 0.0  # along-track distance from ego (+ = ahead)


@dataclass
class PlanOutput:
    state: str
    reason: str
    # None -> publish an empty trajectory (follow the reference).
    xs: np.ndarray = None
    ys: np.ndarray = None
    yaws: np.ndarray = None
    speeds: np.ndarray = None
    source: str = ""      # "fot" / "fallback" / "follow" / "return"
    target: str = ""


class OpponentTracker:
    """Frenet position and along-track speed of every other kart.

    V2X carries positions only, so speed is the filtered time derivative of
    the lap coordinate s (with lap wrap handled).
    """

    def __init__(self, raceline, stale_after=1.0, max_speed=30.0):
        self._rl = raceline
        self._stale = stale_after
        self._max_speed = max_speed
        self._tracks = {}

    def update(self, vid, x, y, t):
        tr = self._tracks.get(vid)
        hint = tr["s"] if tr is not None else None
        s, d = self._rl.project(x, y, hint)
        if tr is None or t - tr["t"] > self._stale:
            self._tracks[vid] = {"x": x, "y": y, "s": s, "d": d, "t": t,
                                 "v": 0.0, "n": 1}
            return
        dt = t - tr["t"]
        if dt < 1e-3:
            return
        v_raw = self._rl.delta_s(s, tr["s"]) / dt
        if abs(v_raw) > self._max_speed:   # teleport / reset
            self._tracks[vid] = {"x": x, "y": y, "s": s, "d": d, "t": t,
                                 "v": 0.0, "n": 1}
            return
        n = tr["n"] + 1
        v = v_raw if n == 2 else 0.6 * tr["v"] + 0.4 * v_raw
        self._tracks[vid] = {"x": x, "y": y, "s": s, "d": d, "t": t,
                             "v": v, "n": n}

    def opponents(self, now, min_samples):
        out = []
        for vid, tr in list(self._tracks.items()):
            age = now - tr["t"]
            if age > self._stale:
                del self._tracks[vid]
                continue
            # Extrapolate along the lap to "now" (V2X may be slower than
            # the planning rate).
            s = self._rl.wrap_s(tr["s"] + tr["v"] * max(age, 0.0))
            out.append(Opponent(vid, tr["x"], tr["y"], s, tr["d"], tr["v"],
                                tr["n"] >= min_samples))
        return out


class OvertakePlanner:

    def __init__(self, raceline, params=None, run_fot=None, logger=None):
        self.rl = raceline
        self.p = params or PlannerParams()
        self._run_fot = run_fot
        self._log = logger or (lambda msg: None)
        self.state = CRUISE
        self._target = None
        self._side = 0
        self._overtake_start = 0.0
        self._cooldown_until = -math.inf
        self._ego_s = None
        self._lat = None     # lateral plan (s0, d0, d1, blend), see
                             # _lateral_plan

    # ------------------------------------------------------------------
    def _set_state(self, state, reason):
        if state != self.state:
            self._log(f"{self.state} -> {state}: {reason}")
        self.state = state

    def _allowed(self, s_from, length):
        p = self.p
        return self.rl.min_allowed_offsets(
            s_from, length, p.vehicle_half_width, p.drivable_area_margin)

    def _acc_speed(self, v_ref, lead):
        """ACC speed limit behind ``lead`` (None -> v_ref)."""
        if lead is None:
            return v_ref
        p = self.p
        v_lead = max(lead.v, 0.0)
        gap = lead.delta - p.vehicle_length
        gap_des = p.follow_gap_min + p.follow_time_gap * v_lead
        if gap <= 0.5:
            return 0.0
        if gap < gap_des:
            v = v_lead * (gap - 0.5) / max(gap_des - 0.5, 1e-3)
        else:
            v_env = math.sqrt(v_lead ** 2 + 2.0 * p.follow_decel *
                              (gap - gap_des))
            # simple_pure_pursuit brakes with a = gain * (v_cmd - v), so it
            # only decelerates at follow_decel if the command is that much
            # *below* the envelope. Commanding the envelope itself makes
            # the kart lag behind it and run into a stopped kart.
            v_brake = v_env - p.follow_decel * p.controller_lag
            # Inside that band, close the remaining gap gently instead of
            # just matching speed (otherwise the gap settles several metres
            # too long and later passes need more zone than exists).
            v_close = v_lead + min(p.follow_close_gain * (gap - gap_des),
                                   p.follow_close_max)
            v = max(v_brake, v_close)
        return max(0.0, min(v_ref, v))

    def _lead_on_path(self, opps, path_s, path_d, ego_s):
        """Nearest kart that a path with offsets ``path_d`` would hit."""
        p = self.p
        lead = None
        deltas = self.rl.delta_s(path_s, ego_s)
        for o in opps:
            if o.delta <= -p.vehicle_length or o.delta > p.detect_ahead:
                continue
            if o.delta < deltas[0] or o.delta > deltas[-1]:
                d_here = path_d[-1] if o.delta > deltas[-1] else path_d[0]
            else:
                d_here = float(np.interp(o.delta, deltas, path_d))
            if abs(o.d - d_here) < p.lateral_block:
                if lead is None or o.delta < lead.delta:
                    lead = o
        return lead

    # ------------------------------------------------------------------
    def _pass_distance(self, ego_v, v_ref, opp):
        """Ego travel needed to complete a pass, or None if never."""
        p = self.p
        need = opp.delta + p.vehicle_length + p.pass_done_gap
        v = ego_v
        gained = 0.0
        travelled = 0.0
        dt = 0.05
        for _ in range(int(15.0 / dt)):
            v = min(v_ref, v + p.ego_accel * dt)
            gained += (v - opp.v) * dt
            travelled += v * dt
            if gained >= need:
                return travelled
        return None

    def _choose_side(self, ego_s, ego_d, opp, distance):
        """+1 (left) / -1 (right) / 0 (no room), for passing ``opp``."""
        p = self.p
        left, right = self._allowed(ego_s, distance)
        options = []
        need = p.lateral_pass + p.side_room_margin
        d_left = opp.d + need
        d_right = opp.d - need
        if d_left <= left:
            options.append((abs(d_left - ego_d), +1))
        if -d_right <= right:
            options.append((abs(d_right - ego_d), -1))
        if not options:
            return 0
        return min(options)[1]

    def _can_overtake(self, now, ego_s, ego_d, ego_v, v_ref, opp):
        p = self.p
        if now < self._cooldown_until:
            return False, 0, "cooling down after an aborted pass"
        remaining = self.rl.zone_remaining(ego_s)
        if remaining is None:
            return False, 0, "no-overtaking zone"
        if not opp.v_valid:
            return False, 0, "speed of the kart ahead not known yet"
        if v_ref - opp.v < p.min_speed_advantage:
            return False, 0, (f"not fast enough (ref {v_ref:.1f} vs "
                              f"{opp.v:.1f} m/s)")
        dist = self._pass_distance(ego_v, v_ref, opp)
        if dist is None or dist + p.zone_end_margin > remaining:
            need = "inf" if dist is None else f"{dist + p.zone_end_margin:.0f}"
            return False, 0, (f"zone too short ({remaining:.0f} m left, "
                              f"{need} m needed)")
        side = self._choose_side(ego_s, ego_d, opp, dist)
        if side == 0:
            return False, 0, "no room on either side"
        return True, side, (f"{'left' if side > 0 else 'right'} pass, "
                            f"{dist:.0f}/{remaining:.0f} m of zone")

    # ------------------------------------------------------------------
    def _offset_plan(self, ego_s, ego_d, d_goal, blend, opps, v_ref,
                     source, extra_lead=None):
        p = self.p
        s0, d0, d1, bl = self._lateral_plan(ego_s, ego_d, d_goal, blend)
        xs, ys, yaws, ss, dd = self.rl.offset_path(
            ego_s, p.path_length, d0, d1, bl,
            clamp=(p.vehicle_half_width, p.drivable_area_margin),
            anchor_s=s0)
        lead = self._lead_on_path(opps, ss, dd, ego_s)
        if extra_lead is not None and (lead is None or
                                       extra_lead.delta < lead.delta):
            lead = extra_lead
        v = self._acc_speed(v_ref, lead)
        return xs, ys, yaws, np.full(len(xs), v), lead

    def _planned_d(self, ego_s):
        """Offset of the current lateral plan at ``ego_s`` (None: no plan)."""
        if self._lat is None:
            return None
        s0, d0, d1, bl = self._lat
        into = self.rl.delta_s(ego_s, s0)
        if into < -1.0:
            return None
        r = min(max(into / max(bl, 1e-6), 0.0), 1.0)
        return d0 + (d1 - d0) * (1.0 - (1.0 - r) ** 2)

    def _lateral_plan(self, ego_s, ego_d, d_goal, blend):
        """Keep one lateral plan fixed in s; re-anchor only when needed.

        Re-anchoring at the kart's own offset every cycle (the first
        version) let the controller's corner-cutting accumulate -- in AWSIM
        the kart drifted from -0.35 m to -1.83 m through a hairpin while
        following and hit the wall. A plan fixed in s gives the controller
        something to correct back to.
        """
        p = self.p
        d_plan = self._planned_d(ego_s)
        if d_plan is not None and \
                abs(self._lat[2] - d_goal) <= p.lateral_goal_tolerance and \
                abs(d_plan - ego_d) <= p.lateral_replan_error:
            return self._lat
        # New plan. A small offset is the controller's tracking error, not
        # a position to plan from.
        d_start = 0.0 if abs(ego_d) < p.return_done_offset else ego_d
        if d_plan is not None and abs(d_plan - ego_d) <= p.lateral_replan_error:
            d_start = d_plan   # goal changed: continue from the old plan
        # Limit the lateral slope (<= ~0.33 at the start of the ease-out).
        blend = max(blend, 6.0 * abs(d_goal - d_start))
        self._lat = (ego_s, d_start, d_goal, blend)
        return self._lat

    def _obstacles(self, opps):
        p = self.p
        boxes = []
        r = p.obstacle_half_size
        for o in opps:
            if o.delta < -p.detect_behind or o.delta > p.detect_ahead:
                continue
            offsets = list(p.obstacle_longitudinal_offsets)
            t = p.obstacle_prediction_step
            while o.v > 0.1 and t <= p.obstacle_prediction_time + 1e-9:
                offsets.append(max(p.obstacle_longitudinal_offsets) +
                               o.v * t)
                t += p.obstacle_prediction_step
            for off in offsets:
                cx, cy, _ = self.rl.to_xy(o.s + off, o.d)
                boxes.append([cx - r, cy - r, cx + r, cy + r])
        return boxes

    def _fot_plan(self, ego, v_ref, opps, opp):
        """FOT path to the chosen side of ``opp``; (path, reason)."""
        p = self.p
        ego_x, ego_y, ego_s, ego_d, ego_vx, ego_vy = ego
        side = self._side
        d_a = opp.d + side * p.lateral_min
        d_b = opp.d + side * (p.lateral_pass + p.lateral_pass_extra)
        # Room is needed until we are past the other kart, not beyond.
        alongside_until = max(opp.delta + p.vehicle_length +
                              p.pass_done_gap, 5.0)
        left, right = self._allowed(ego_s, alongside_until)
        lo, hi = sorted((d_a, d_b))
        hi = min(hi, left)
        lo = max(lo, -right)
        if hi < lo:
            return None, "no room for the passing offset"

        wp = self.rl.sample_points(ego_s - p.fot_lookbehind,
                                   ego_s + p.fot_lookahead,
                                   p.fot_waypoint_spacing)
        hp = dict(p.fot_hyperparameters)
        # FOT samples di from -max_road_width_l to +max_road_width_r.
        hp["max_road_width_l"] = -lo
        hp["max_road_width_r"] = hi
        boxes = self._obstacles(opps)
        ic = {"ps": 0.0, "target_speed": v_ref,
              "pos": np.array([ego_x, ego_y]),
              "vel": np.array([ego_vx, ego_vy]),
              "wp": wp,
              "obs": np.array(boxes, dtype=np.float64).reshape(-1, 4)}
        (x_path, y_path, speeds, _ix, _iy, _iyaw, _d, _s, sx, sy,
         _params, _costs, success) = self._run_fot(ic, hp)
        if not success or len(x_path) < 2:
            return None, "FOT found no collision-free path"
        err = math.hypot(x_path[0] - ego_x, y_path[0] - ego_y)
        if err > p.max_start_position_error:
            return None, f"FOT path starts {err:.1f} m from the ego"
        bad = self.rl.check_path(x_path, y_path, ego_s,
                                 p.vehicle_half_width,
                                 p.drivable_area_margin)
        if bad is not None:
            dt = hp["dt"]
            if bad["path_index"] * dt < p.fot_keep_time:
                return None, (f"FOT path leaves the track ({bad['side']} "
                              f"{bad['offset']:.2f} > "
                              f"{bad['allowed']:.2f} m)")
            # Only the far end is off: keep the valid near part, the next
            # cycle (50 ms later) re-plans from closer anyway.
            k = bad["path_index"]
            x_path, y_path = x_path[:k], y_path[:k]
            speeds, sx, sy = speeds[:k], sx[:k], sy[:k]
        yaws = np.where(np.asarray(speeds) > 1e-2, np.arctan2(sy, sx),
                        np.arctan2(np.gradient(y_path), np.gradient(x_path)))
        # simple_pure_pursuit commands the speed of the path point CLOSEST
        # to the kart, i.e. FOT's first point, which is always the current
        # speed -- so a raw FOT speed profile never lets the kart speed up
        # to pass. Publish FOT's planned end speed on every point instead
        # (pure pursuit's max_acceleration still limits the ramp).
        v_goal = max(float(speeds[-1]), 0.0)
        return (np.asarray(x_path), np.asarray(y_path), yaws,
                np.full(len(x_path), v_goal)), "ok"

    # ------------------------------------------------------------------
    def step(self, now, ego_x, ego_y, ego_yaw, ego_v, opponents_raw):
        """One planning cycle. ``opponents_raw`` are Opponent objects."""
        p = self.p
        ego_s, ego_d = self.rl.project(ego_x, ego_y, self._ego_s)
        if abs(ego_d) > p.on_loop_max_offset:
            self._ego_s = None
            self._target = None
            self._set_state(CRUISE, "not on the raceline loop")
            return PlanOutput(INACTIVE, "ego is not on the raceline loop")
        self._ego_s = ego_s
        v_ref = self.rl.speed_at(ego_s)
        ego_vx = ego_v * math.cos(ego_yaw)
        ego_vy = ego_v * math.sin(ego_yaw)

        opps = []
        for o in opponents_raw:
            o.delta = self.rl.delta_s(o.s, ego_s)
            if -p.detect_behind <= o.delta <= p.detect_ahead:
                opps.append(o)

        # The kart currently in the way of the raceline (CRUISE path).
        def blocking(d_path=0.0):
            cand = [o for o in opps if o.delta > 0.0 and
                    abs(o.d - d_path) < p.lateral_block]
            return min(cand, key=lambda o: o.delta) if cand else None

        # ---------------- OVERTAKE ----------------
        if self.state == OVERTAKE:
            opp = next((o for o in opps if o.vid == self._target), None)
            if opp is None:
                self._set_state(RETURN, "overtaken kart out of range")
            elif opp.delta < -(p.vehicle_length + p.pass_done_gap):
                self._set_state(RETURN, f"passed {opp.vid}")
            elif self.rl.zone_remaining(ego_s) is None and opp.delta > 0.0:
                self._abort(now, "left the overtaking zone before passing")
            elif now - self._overtake_start > p.max_overtake_time:
                self._abort(now, "overtake took too long")
            else:
                return self._overtake_output(
                    (ego_x, ego_y, ego_s, ego_d, ego_vx, ego_vy),
                    v_ref, opps, opp)

        # ---------------- CRUISE / FOLLOW / RETURN ----------------
        lead = blocking(0.0)
        if lead is not None:
            ok, side, why = self._can_overtake(now, ego_s, ego_d, ego_v,
                                               v_ref, lead)
            if ok:
                self._target = lead.vid
                self._side = side
                self._overtake_start = now
                self._set_state(OVERTAKE, f"{lead.vid}: {why}")
                return self._overtake_output(
                    (ego_x, ego_y, ego_s, ego_d, ego_vx, ego_vy),
                    v_ref, opps, lead)
            self._set_state(FOLLOW, f"{lead.vid} ahead: {why}")
            return self._follow_output(ego_s, ego_d, ego_v, v_ref, opps,
                                       why, lead)

        # RETURN only follows a deliberate lateral manoeuvre. In CRUISE the
        # offset is just the controller's tracking error (it cuts corners
        # by more than return_done_offset), which is not ours to correct.
        if self.state in (OVERTAKE, FOLLOW, RETURN) and \
                abs(ego_d) > p.return_done_offset:
            self._set_state(RETURN, "back to the raceline")
            blend = max(p.return_blend_min, ego_v * p.return_blend_time)
            xs, ys, yaws, vs, lead2 = self._offset_plan(
                ego_s, ego_d, 0.0, blend, opps, v_ref, "return")
            return PlanOutput(RETURN, "blending back to the raceline", xs, ys,
                              yaws, vs, "return",
                              lead2.vid if lead2 else "")

        self._lat = None
        self._set_state(CRUISE, "clear")
        return PlanOutput(CRUISE, "nothing in the way")

    def _abort(self, now, reason):
        self._cooldown_until = now + self.p.retry_cooldown
        self._target = None
        self._set_state(FOLLOW, reason)

    def _follow_output(self, ego_s, ego_d, ego_v, v_ref, opps, why, lead):
        p = self.p
        # Do not swing back onto the raceline while still beside a kart.
        beside = any(abs(o.delta) < p.vehicle_length + 1.0 and
                     abs(o.d - ego_d) >= 0.8 * p.lateral_block
                     for o in opps)
        if beside:
            planned = self._planned_d(ego_s)
            d_goal = ego_d if planned is None else planned
        else:
            d_goal = 0.0
        blend = max(p.return_blend_min, ego_v * p.return_blend_time)
        xs, ys, yaws, vs, lead2 = self._offset_plan(
            ego_s, ego_d, d_goal, blend, opps, v_ref, "follow", lead)
        tgt = lead2.vid if lead2 else ""
        return PlanOutput(FOLLOW, why, xs, ys, yaws, vs, "follow", tgt)

    def _overtake_output(self, ego, v_ref, opps, opp):
        p = self.p
        ego_x, ego_y, ego_s, ego_d, _vx, _vy = ego
        path, why = self._fot_plan(ego, v_ref, opps, opp)
        if path is not None:
            xs, ys, yaws, vs = path
            return PlanOutput(OVERTAKE, "FOT", xs, ys, yaws, vs, "fot",
                              opp.vid)
        # Fallback: keep heading for the passing offset along the raceline
        # (never snap back to the raceline beside the other kart); slow down
        # if that offset path would run into someone.
        d_goal = opp.d + self._side * p.lateral_pass
        left, right = self._allowed(
            ego_s, max(opp.delta + p.vehicle_length + p.pass_done_gap, 5.0))
        d_goal = max(-right, min(left, d_goal))
        blend = max(8.0, 1.2 * math.hypot(ego[4], ego[5]))
        xs, ys, yaws, vs, lead = self._offset_plan(
            ego_s, ego_d, d_goal, blend, opps, v_ref, "fallback")
        return PlanOutput(OVERTAKE, f"fallback ({why})", xs, ys, yaws, vs,
                          "fallback", opp.vid)
