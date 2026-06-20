"""GMT whole-body G1: a 23-DOF motion-tracking policy steered by velocity.

Runs the pretrained GMT (General Motion Tracking) policy on the full 23-DOF
Unitree G1 (12 legs + 3 waist + 8 arms), so unlike the lower-body walker the
arms physically move. GMT tracks a reference motion clip; we make it WASD-
drivable by overriding the reference's commanded local velocity (vx, vy) and
yaw-rate while it tracks a walk gait. The policy only tracks base height, roll,
pitch, local velocity, and yaw-rate (not absolute x/y/yaw), which is what makes
velocity-steering work.

Assets (weights, MJCF, motion clips) live in $GMT_ROOT (the GMT repo). The
control logic (obs construction, PD gains, action scaling) matches GMT's
reference sim2sim.py; only the headless renderer + steering/serving are added.

Run:
    python -m ember.gmt --scene obstacles
    ember-gmt --motion dance.pkl
"""
from __future__ import annotations

import math
import sys
import threading
import time
from collections import deque

import mujoco
import numpy as np
import torch

from .config import (GMT_ROOT, RENDER_FPS, RENDER_H, RENDER_W, clamp)
from .streaming import FrameBuffer, create_app, encode_jpeg, serve

# GMT assets.
G1_DIR = GMT_ROOT / "assets" / "robots" / "g1"
MODEL_PATH = str(G1_DIR / "g1.xml")
OBSTACLE_PATH = str(G1_DIR / "g1_obstacles.xml")  # generated, see build_obstacle_scene
POLICY_PATH = str(GMT_ROOT / "assets" / "pretrained_checkpoints" / "pretrained.pt")
MOTION_DIR = GMT_ROOT / "assets" / "motions"
DEVICE = "cpu"  # tiny MLP; CPU sidesteps the Blackwell/sm_120 issue

MOTIONS = [
    "walk_stand.pkl", "basic_walk.pkl", "crouchwalk_stand.pkl",
    "kick_walk.pkl", "squat.pkl", "dance.pkl", "dance_waltz.pkl",
    "airkick_stand.pkl",
]

# WASD driving: override the reference's commanded local velocity + yaw-rate.
DRIVE_CLIP = "basic_walk.pkl"
VX_RANGE = (-0.4, 1.0)
VY_RANGE = (-0.3, 0.3)
YAW_RANGE = (-0.8, 0.8)
MOVE_EPS = 0.04   # below this on all axes -> stand in place
STAND_Z = 0.74    # reference pelvis height when standing

# MotionLib lives in the GMT repo, not pip-installable; add it to the path.
if str(GMT_ROOT) not in sys.path:
    sys.path.insert(0, str(GMT_ROOT))


@torch.jit.script
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3),
                          v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


def euler_from_quaternion(quat_angle):
    """Tensor [N,4] xyzw -> (roll, pitch, yaw). Matches GMT sim2sim."""
    x = quat_angle[:, 0]; y = quat_angle[:, 1]
    z = quat_angle[:, 2]; w = quat_angle[:, 3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)
    t2 = torch.clip(+2.0 * (w * y - z * x), -1, 1)
    pitch_y = torch.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)
    return roll_x, pitch_y, yaw_z


def quat_to_euler(quat):
    """np [w,x,y,z] -> [roll, pitch, yaw]. Matches GMT sim2sim."""
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    e = np.zeros(3)
    e[0] = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    sinp = 2 * (qw * qy - qz * qx)
    e[1] = np.copysign(np.pi / 2, sinp) if abs(sinp) >= 1 else np.arcsin(sinp)
    e[2] = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return e


def build_obstacle_scene(out_path=OBSTACLE_PATH, src_path=MODEL_PATH,
                         bump_x=1.4, bump_radius=0.40, bump_expose=0.02,
                         ramp_x=2.6, ramp_run=1.8, rise=0.07,
                         plat_depth=1.4, width=1.8):
    """Write a copy of GMT's g1.xml with a GENTLE obstacle course injected. GMT
    is blind and tracks a flat-ground walk gait, so it only clears smooth,
    rounded, low obstacles: rounded log -> up-ramp -> platform -> down-ramp."""
    g = [
        f'<geom name="log" type="cylinder" size="{bump_radius} {width/2}" '
        f'pos="{bump_x} 0 {bump_expose - bump_radius}" euler="1.5708 0 0" '
        f'rgba="0.45 0.30 0.15 1" condim="3" friction="1.3 0.01 0.001"/>',
    ]
    theta = math.atan2(rise, ramp_run)
    length = math.hypot(ramp_run, rise) + 0.1
    g.append(
        f'<geom name="ramp_up" type="box" size="{length/2} {width/2} 0.03" '
        f'pos="{ramp_x + ramp_run/2} 0 {rise/2}" euler="0 {-theta} 0" '
        f'rgba="0.35 0.35 0.4 1" condim="3" friction="1.1 0.005 0.0001"/>')
    plat_x0 = ramp_x + ramp_run
    g.append(
        f'<geom name="platform" type="box" size="{plat_depth/2} {width/2} {rise/2}" '
        f'pos="{plat_x0 + plat_depth/2} 0 {rise/2}" '
        f'rgba="0.30 0.40 0.30 1" condim="3" friction="1.0 0.005 0.0001"/>')
    down_x = plat_x0 + plat_depth
    g.append(
        f'<geom name="ramp_down" type="box" size="{length/2} {width/2} 0.03" '
        f'pos="{down_x + ramp_run/2} 0 {rise/2}" euler="0 {theta} 0" '
        f'rgba="0.35 0.35 0.4 1" condim="3" friction="1.1 0.005 0.0001"/>')
    geom_xml = "\n    ".join(g)

    with open(src_path) as f:
        xml = f.read()
    marker = '<light pos="0 0 1000"'
    if marker not in xml:
        raise RuntimeError("scene light marker not found in GMT g1.xml")
    xml = xml.replace(marker, geom_xml + "\n    " + marker, 1)
    with open(out_path, "w") as f:
        f.write(xml)
    return out_path


class GMTSim:
    """Headless GMT whole-body G1, steerable by velocity. Control logic per
    GMT's sim2sim.py."""

    def __init__(self, motion=DRIVE_CLIP, scene="flat"):
        # Robot constants (g1, 23 DOF) -- verbatim from GMT sim2sim.py.
        self.stiffness = np.array([
            100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40,
            150, 150, 150, 40, 40, 40, 40, 40, 40, 40, 40], dtype=np.float32)
        self.damping = np.array([
            2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2,
            4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5], dtype=np.float32)
        self.torque_limits = np.array([
            88, 139, 88, 139, 50, 50, 88, 139, 88, 139, 50, 50,
            88, 50, 50, 25, 25, 25, 25, 25, 25, 25, 25], dtype=np.float32)
        self.default_dof_pos = np.array([
            -0.2, 0.0, 0.0, 0.4, -0.2, 0.0, -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 1.2, 0.0, -0.4, 0.0, 1.2], dtype=np.float32)
        self.num_actions = 23
        self.num_dofs = 23
        self.action_scale = 0.5

        self.sim_dt = 0.001
        self.sim_decimation = 20             # -> 50 Hz control
        self.control_dt = self.sim_dt * self.sim_decimation

        self.tar_obs_steps = torch.tensor(
            [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
             50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
            device=DEVICE, dtype=torch.int)
        self.n_proprio = 3 + 2 + 3 * self.num_actions   # 74
        self.history_len = 20
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        self.ang_vel_scale = 0.25

        if scene == "obstacles":
            model_path = build_obstacle_scene()
        else:
            model_path = MODEL_PATH
        self.scene = scene
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.sim_dt
        # GMT's g1.xml has no <global offwidth/offheight>; enlarge for our size.
        self.model.vis.global_.offwidth = RENDER_W
        self.model.vis.global_.offheight = RENDER_H
        self.data = mujoco.MjData(self.model)

        torch.set_num_threads(1)
        self.policy = torch.jit.load(POLICY_PATH, map_location=DEVICE)
        self.policy.eval()

        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.proprio_history_buf = deque(maxlen=self.history_len)
        self.counter = 0
        self.motion_name = motion
        self._pending_motion = None
        self._fell = False

        # WASD driving command, applied as a reference override.
        self.drive = (motion == DRIVE_CLIP)
        self.cmd = np.zeros(3, dtype=np.float32)
        self._cmd_lock = threading.Lock()
        self.heading_hold = False
        self.heading_target = 0.0
        self.lateral_target = 0.0
        self._manual_yaw = False
        self._manual_vy = False

        self.renderer = None
        self.cam = mujoco.MjvCamera()
        self.cam.distance = 3.5
        self.cam.elevation = -15.0
        self.cam.azimuth = 135.0
        self.frames = FrameBuffer()
        self._running = False

        self._load_motion(motion)
        self._reset()

    # -- motion / reset ------------------------------------------------------ #
    def _load_motion(self, name):
        from utils.motion_lib import MotionLib
        self._motion_lib = MotionLib(str(MOTION_DIR / name), DEVICE)
        self.motion_name = name

    def request_motion(self, name):
        if name in MOTIONS:
            self._pending_motion = name
            return True
        return False

    def reset(self):
        with self._cmd_lock:
            self.cmd[:] = 0.0
        self._manual_yaw = False
        self._manual_vy = False
        self._reset()
        return True

    def _reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self.last_action[:] = 0.0
        self.proprio_history_buf.clear()
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_proprio))
        self.counter = 0
        self._fell = False

    # -- WASD driving -------------------------------------------------------- #
    def set_command(self, vx=None, vy=None, yaw=None):
        with self._cmd_lock:
            if vx is not None:
                self.cmd[0] = clamp(vx, *VX_RANGE)
            if vy is not None:
                self._manual_vy = True
                self.cmd[1] = clamp(vy, *VY_RANGE)
            if yaw is not None:
                self._manual_yaw = True
                self.cmd[2] = clamp(yaw, *YAW_RANGE)
            return self.cmd.copy()

    def set_heading_hold(self, on=True):
        self.heading_hold = on
        self._manual_yaw = not on
        self._manual_vy = not on
        if on:
            self.heading_target = self._base_yaw()
            self.lateral_target = float(self.data.qpos[1])
        return on

    def _base_yaw(self):
        w, x, y, z = self.data.qpos[3:7]
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def get_state(self):
        return {
            "pos": self.data.qpos[0:3].copy(),
            "quat": self.data.qpos[3:7].copy(),
            "motion": self.motion_name,
            "drive": self.drive,
            "cmd": self.cmd.copy(),
            "heading_hold": self.heading_hold,
            "fell": self._fell,
            "sim_time": self.data.time,
        }

    # -- observation (GMT logic, with WASD override) ------------------------- #
    def _get_mimic_obs(self, curr_time_step):
        num_steps = len(self.tar_obs_steps)
        motion_times = torch.tensor(
            [curr_time_step * self.control_dt], device=DEVICE).unsqueeze(-1)
        obs_motion_times = (self.tar_obs_steps * self.control_dt
                            + motion_times).flatten()
        motion_ids = torch.zeros(num_steps, dtype=torch.int, device=DEVICE)
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, _ = \
            self._motion_lib.calc_motion_frame(motion_ids, obs_motion_times)

        roll, pitch, _ = euler_from_quaternion(root_rot)
        roll = roll.reshape(1, num_steps, 1)
        pitch = pitch.reshape(1, num_steps, 1)
        root_vel = quat_rotate_inverse(root_rot, root_vel).reshape(1, num_steps, 3)
        root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel).reshape(1, num_steps, 3)
        root_pos = root_pos.reshape(1, num_steps, 3)
        dof_pos = dof_pos.reshape(1, num_steps, -1)

        mimic = torch.cat((
            root_pos[..., 2:3], roll, pitch, root_vel,
            root_ang_vel[..., 2:3], dof_pos,
        ), dim=-1).reshape(1, -1)
        m = mimic.detach().cpu().numpy().squeeze()

        # WASD override. Per-step layout (30): [z, roll, pitch, vx, vy, vz,
        # yaw_rate, dof(23)].
        if self.drive:
            with self._cmd_lock:
                vx, vy, yaw = self.cmd
            a = m.reshape(num_steps, 30)
            if abs(vx) < MOVE_EPS and abs(vy) < MOVE_EPS and abs(yaw) < MOVE_EPS:
                # Stand in place: upright, zero velocity, default pose.
                a[:, 0] = STAND_Z
                a[:, 1] = 0.0; a[:, 2] = 0.0
                a[:, 3:7] = 0.0
                a[:, 7:30] = self.default_dof_pos
            else:
                a[:, 3] = vx; a[:, 4] = vy; a[:, 6] = yaw
            m = a.reshape(-1)
        return m

    def _extract(self):
        dof_pos = self.data.qpos.astype(np.float32)[-self.num_dofs:]
        dof_vel = self.data.qvel.astype(np.float32)[-self.num_dofs:]
        quat = self.data.sensor("orientation").data.astype(np.float32)
        ang_vel = self.data.sensor("angular-velocity").data.astype(np.float32)
        return dof_pos, dof_vel, quat, ang_vel

    def _compute_pd_target(self):
        dof_pos, dof_vel, quat, ang_vel = self._extract()
        mimic_obs = self._get_mimic_obs(self.counter)
        rpy = quat_to_euler(quat)
        obs_dof_vel = dof_vel.copy()
        obs_dof_vel[[4, 5, 10, 11]] = 0.0   # zero ankle vels (GMT)
        obs_prop = np.concatenate([
            ang_vel * self.ang_vel_scale,
            rpy[:2],
            (dof_pos - self.default_dof_pos) * self.dof_pos_scale,
            obs_dof_vel * self.dof_vel_scale,
            self.last_action,
        ])
        obs_hist = np.array(self.proprio_history_buf).flatten()
        obs_buf = np.concatenate([mimic_obs, obs_prop, obs_hist])
        obs_t = torch.from_numpy(obs_buf).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            raw = self.policy(obs_t).cpu().numpy().squeeze()
        self.last_action = raw.copy()
        scaled = np.clip(raw, -10.0, 10.0) * self.action_scale
        self.proprio_history_buf.append(obs_prop)
        return scaled + self.default_dof_pos

    # -- rendering ----------------------------------------------------------- #
    def render_once(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, RENDER_H, RENDER_W)
        self.cam.lookat[:] = self.data.qpos[0:3]
        self.renderer.update_scene(self.data, self.cam)
        self.frames.set(encode_jpeg(self.renderer.render()))

    def _render_loop(self):
        interval = 1.0 / RENDER_FPS
        while self._running:
            t0 = time.time()
            try:
                self.render_once()
            except Exception as e:
                print("render error:", e)
            sleep = interval - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)

    # -- main loop ----------------------------------------------------------- #
    def run(self, render=True):
        self._running = True
        if render:
            threading.Thread(target=self._render_loop, daemon=True).start()
        dec = self.sim_decimation
        try:
            while self._running:
                tick_start = time.time()

                if self._pending_motion is not None:
                    self._load_motion(self._pending_motion)
                    self.drive = (self._pending_motion == DRIVE_CLIP)
                    self._pending_motion = None
                    self._reset()

                # Heading-hold keeps the robot on the course centerline. Only
                # correct laterally while roughly facing the target heading,
                # else the cos(yaw) sign can flip into a runaway.
                if self.drive and self.heading_hold:
                    yaw = self._base_yaw()
                    h_err = np.arctan2(np.sin(self.heading_target - yaw),
                                       np.cos(self.heading_target - yaw))
                    y_err = self.lateral_target - self.data.qpos[1]
                    vy_body = y_err * np.cos(yaw)
                    with self._cmd_lock:
                        if not self._manual_yaw:
                            self.cmd[2] = clamp(2.0 * h_err, *YAW_RANGE)
                        if not self._manual_vy:
                            self.cmd[1] = (clamp(1.2 * vy_body, *VY_RANGE)
                                           if abs(h_err) < 0.6 else 0.0)

                pd_target = self._compute_pd_target()
                for _ in range(dec):
                    dof_pos = self.data.qpos.astype(np.float32)[-self.num_dofs:]
                    dof_vel = self.data.qvel.astype(np.float32)[-self.num_dofs:]
                    torque = np.clip(
                        (pd_target - dof_pos) * self.stiffness - dof_vel * self.damping,
                        -self.torque_limits, self.torque_limits)
                    self.data.ctrl[:] = torque
                    mujoco.mj_step(self.model, self.data)
                self.counter += 1
                self._fell = self.data.qpos[2] < 0.4

                sleep = self.control_dt - (time.time() - tick_start)
                if sleep > 0:
                    time.sleep(sleep)
        finally:
            self._running = False

    def stop(self):
        self._running = False


# --------------------------------------------------------------------------- #
# Module singleton + web viewer
# --------------------------------------------------------------------------- #
_SIM: GMTSim | None = None


def get_sim(motion=DRIVE_CLIP, scene="flat") -> GMTSim:
    global _SIM
    if _SIM is None:
        _SIM = GMTSim(motion=motion, scene=scene)
    return _SIM


def send_velocity_command(vx=None, vy=None, yaw=None):
    return get_sim().set_command(vx=vx, vy=vy, yaw=yaw)


def get_state():
    return get_sim().get_state()


def _page_html():
    btns = "".join(
        f'<button onclick="setMotion(\'{m}\')">{m.split(".")[0]}</button> '
        for m in MOTIONS)
    return """<!doctype html><html><head><title>GMT G1</title>
<style>body{background:#111;color:#ddd;font-family:monospace;margin:0;text-align:center}
img{max-width:100%;height:auto;background:#000}
#hud{position:fixed;top:8px;left:8px;background:rgba(0,0,0,.6);padding:8px;text-align:left;font-size:13px;border-radius:4px;max-width:46vw}
kbd{background:#333;padding:1px 5px;border-radius:3px}
button{background:#222;color:#ddd;border:1px solid #555;border-radius:4px;padding:3px 6px;margin:2px;cursor:pointer;font-family:monospace;font-size:12px}
button:hover{background:#345}</style></head>
<body>
<div id="hud">
<b>GMT whole-body G1</b> (23-DOF, legs+waist+arms)<br>
<kbd>W</kbd>/<kbd>S</kbd> fwd/back &nbsp; <kbd>A</kbd>/<kbd>D</kbd> turn &nbsp;
<kbd>Q</kbd>/<kbd>E</kbd> strafe &nbsp; <kbd>Space</kbd> stop<br>
<kbd>H</kbd> auto-steer (walk straight) <span id="hh">[off]</span> &nbsp; <kbd>R</kbd> reset to start<br>
cmd: <span id="cmd">0,0,0</span> &nbsp; <span id="dr"></span><br>
Reference clip (driving works on basic_walk):<br>__BTNS__<br>
motion: <span id="mo">?</span><br>
<span id="st"></span>
</div>
<img src="/stream"/>
<script>
let vx=0,vy=0,yaw=0;
function send(){fetch(`/cmd?vx=${vx}&vy=${vy}&yaw=${yaw}`).then(r=>r.json()).then(d=>{
  document.getElementById('cmd').textContent=d.cmd.map(x=>x.toFixed(2)).join(', ');});}
function setMotion(m){fetch('/motion?name='+m).then(r=>r.json()).then(d=>{
  document.getElementById('mo').textContent=d.motion;});}
document.addEventListener('keydown',e=>{
  if(e.repeat)return;
  switch(e.key.toLowerCase()){
    case'w':vx=0.8;break; case's':vx=-0.3;break;
    case'a':yaw=0.6;break; case'd':yaw=-0.6;break;
    case'q':vy=0.25;break; case'e':vy=-0.25;break;
    case' ':vx=vy=yaw=0;break;
    case'h':fetch('/heading?on=1').then(r=>r.json()).then(d=>{
      document.getElementById('hh').textContent=d.heading_hold?'[ON]':'[off]';});return;
    case'r':vx=vy=yaw=0;fetch('/reset');return;
    default:return;}
  send();});
document.addEventListener('keyup',e=>{
  switch(e.key.toLowerCase()){
    case'w':case's':vx=0;break;
    case'a':case'd':yaw=0;break;
    case'q':case'e':vy=0;break; default:return;}
  send();});
setInterval(()=>{fetch('/state').then(r=>r.json()).then(d=>{
  document.getElementById('mo').textContent=d.motion;
  document.getElementById('dr').textContent=d.drive?'(drivable)':'(clip playback)';
  document.getElementById('hh').textContent=d.heading_hold?'[ON]':'[off]';
  document.getElementById('st').textContent=
    `pos ${d.pos.map(x=>x.toFixed(2))} ${d.fell?'\u26a0 FELL':'ok'}`;});},400);
</script></body></html>""".replace("__BTNS__", btns)


def _make_app(sim: GMTSim):
    from flask import jsonify, request

    def register(app):
        @app.route("/cmd")
        def cmd():
            def f(name):
                v = request.args.get(name)
                return float(v) if v is not None else None
            return jsonify(cmd=sim.set_command(vx=f("vx"), vy=f("vy"),
                                               yaw=f("yaw")).tolist())

        @app.route("/motion")
        def motion():
            name = request.args.get("name", "")
            ok = sim.request_motion(name)
            return jsonify(ok=ok, motion=name if ok else sim.motion_name)

        @app.route("/heading")
        def heading():
            on = request.args.get("on", "1") not in ("0", "false", "")
            sim.set_heading_hold(on)
            return jsonify(heading_hold=on)

        @app.route("/reset")
        def reset():
            sim.reset()
            return jsonify(ok=True)

    def state_fn():
        s = sim.get_state()
        return dict(pos=s["pos"].tolist(), motion=s["motion"],
                    drive=bool(s["drive"]), cmd=s["cmd"].tolist(),
                    heading_hold=bool(s["heading_hold"]),
                    fell=bool(s["fell"]), sim_time=float(s["sim_time"]))

    return create_app(page_html=_page_html(), frame_buffer=sim.frames,
                      state_fn=state_fn, register_routes=register)


def start(block=True, serve_web=True, port=8089, render=True, motion=DRIVE_CLIP,
          scene="flat"):
    sim = get_sim(motion=motion, scene=scene)
    if serve_web:
        serve(_make_app(sim), port, label="GMT whole-body G1")
    if block:
        sim.run(render=render)
    else:
        threading.Thread(target=lambda: sim.run(render=render), daemon=True).start()
    return sim


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--motion", default=DRIVE_CLIP,
                   help="reference clip: " + ", ".join(MOTIONS))
    p.add_argument("--scene", default="flat", choices=["flat", "obstacles"])
    p.add_argument("--port", type=int, default=8089)
    args = p.parse_args()
    start(block=True, serve_web=True, port=args.port, motion=args.motion,
          scene=args.scene)


if __name__ == "__main__":
    main()
