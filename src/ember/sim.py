"""Unitree G1 12-DOF RL walker with a kinematic full-body arm overlay.

Physics: Unitree's pretrained 12-DOF lower-body locomotion policy
(``deploy/pre_train/g1/motion.pt``) in a plain MuJoCo sim. Its command vector is
exactly ``(vx, vy, yaw_rate)``, so velocity control reduces to writing that
vector -- see :meth:`G1Sim.set_command`.

Arms: the 12-DOF model's arms are welded decoration, so for rendering we drive a
full 29-DOF (with hands) body whose base + 12 legs are copied from the physics
sim every frame, with the arms posed kinematically (carry hose / aim) -- an
articulated, posable upper body on the bulletproof walker, no extra physics.

The interactive web viewer that hot-swaps and serves this sim lives in
:mod:`ember.viewer`.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque

import mujoco
import numpy as np
import torch
import yaml

from . import arms, effects, scenes
from .config import (CAM_FOVY, CAM_H, CAM_W, DEPLOY_CONFIG_PATH, G1_MODEL_DIR,
                     POLICY_PATH, RENDER_FPS, RENDER_H, RENDER_W, VX_RANGE,
                     VY_RANGE, YAW_RANGE, clamp, gravity_orientation, quat_yaw)
from .streaming import FrameBuffer, encode_jpeg

# Ego ("helmet") camera mount, in the torso_link frame: a bit forward and up to
# head height, looking forward and tilted down to see the ground ahead.
EGO_CAM_POS = (0.10, 0.0, 0.45)
EGO_CAM_TILT_DEG = 18.0


def _ego_camera_quat(tilt_deg: float) -> list[float]:
    """Quaternion orienting a torso-mounted camera to look forward (+x body)
    and ``tilt_deg`` downward. MuJoCo cameras look down their local -Z."""
    t = math.radians(tilt_deg)
    # Columns are the camera axes (x-right, y-up, -z-forward) in the body frame.
    R = np.array([[0.0, math.sin(t), -math.cos(t)],
                  [-1.0, 0.0, 0.0],
                  [0.0, math.cos(t), math.sin(t)]], dtype=float)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q.tolist()


# Fog-nozzle prop, welded under torso_link (the hands hold it in the "carry"
# pose); offset/orientation tuned by rendering against that pose.
HOSE_TORSO_OFFSET = (0.439, 0.101, 0.037)
HOSE_BARREL_TILT = 1.65   # barrel axis, rad from +Z toward +X (slight down-aim)
NOZZLE_REACH = 0.21       # hose origin -> muzzle tip, along the barrel
# Nozzle elevation hinge (about torso +Y; negative raises the barrel), posed
# kinematically since the overlay is never stepped.
NOZZLE_JOINT = "nozzle_pitch_joint"
NOZZLE_PITCH_RANGE = (-1.0, 0.3)
NOZZLE_REST_PITCH = -0.08  # fixed ~level elevation (no auto-aim)

# Fire engagement: the jet must pass within HIT_RADIUS of the fire centre, and
# the fire goes out only after EXTINGUISH_TIME of cumulative on-target hits.
FIRE_AIM_HEIGHT = 0.45
HIT_RADIUS = 0.25
EXTINGUISH_TIME = 10.0

BLOCKED_WINDOW = 2.0
BLOCKED_DISP_THRESH = 0.05
BLOCKED_CMD_THRESH = 0.05


def _y_quat(phi: float) -> list[float]:
    """Quaternion for a rotation of ``phi`` rad about +Y (tilts +Z toward +X)."""
    return [math.cos(phi / 2), 0.0, math.sin(phi / 2), 0.0]


def _attach_hose(spec) -> None:
    """Add the fog-nozzle prop as a non-colliding body under ``torso_link``."""
    GT = mujoco.mjtGeom
    body = spec.body("torso_link").add_body()
    body.name = "hose"
    body.pos = list(HOSE_TORSO_OFFSET)
    j = body.add_joint()
    j.name = NOZZLE_JOINT
    j.type = mujoco.mjtJoint.mjJNT_HINGE
    j.axis = [0, 1, 0]
    j.range = list(NOZZLE_PITCH_RANGE)
    barrel = HOSE_BARREL_TILT
    bq = _y_quat(barrel)
    d = (math.sin(barrel), 0.0, math.cos(barrel))

    def along(s):  # point a distance s along the barrel axis from the grip centre
        return [d[0] * s, d[1] * s, d[2] * s]

    # name, type, size, pos, quat, rgba
    parts = [
        ("hose_barrel", GT.mjGEOM_CYLINDER, [0.026, 0.12, 0], [0, 0, 0], bq,
         [0.15, 0.15, 0.17, 1]),
        ("hose_head", GT.mjGEOM_CYLINDER, [0.045, 0.035, 0], along(0.15), bq,
         [0.10, 0.10, 0.12, 1]),
        ("hose_face", GT.mjGEOM_CYLINDER, [0.042, 0.006, 0], along(0.188), bq,
         [0.85, 0.68, 0.12, 1]),
        ("hose_coupling", GT.mjGEOM_CYLINDER, [0.030, 0.020, 0], along(-0.13), bq,
         [0.72, 0.55, 0.20, 1]),
        ("hose_bale", GT.mjGEOM_CYLINDER, [0.011, 0.05, 0],
         [d[0] * -0.05, 0, d[2] * -0.05 + 0.055], _y_quat(-0.40),
         [0.12, 0.12, 0.14, 1]),
        ("hose_tail", GT.mjGEOM_CYLINDER, [0.024, 0.16, 0], [-0.14, 0, -0.16],
         _y_quat(3.54), [0.52, 0.16, 0.12, 1]),
    ]
    for name, typ, size, pos, quat, rgba in parts:
        g = body.add_geom()
        g.name = name
        g.type = typ
        g.size = size
        g.pos = pos
        if quat is not None:
            g.quat = quat
        g.rgba = rgba
        g.contype = 0
        g.conaffinity = 0

# 12-DOF physics scenes, and their matching 29-DOF render-overlay scenes.
SCENES = {
    "flat": str(G1_MODEL_DIR / scenes.FLAT_12),
    "obstacles": str(G1_MODEL_DIR / scenes.OBSTACLES_12),
    "fire": str(G1_MODEL_DIR / scenes.FIRE_12),
}
OVERLAY_SCENES = {
    SCENES["flat"]: str(G1_MODEL_DIR / scenes.FLAT_29),
    SCENES["obstacles"]: str(G1_MODEL_DIR / scenes.OBSTACLES_29),
    SCENES["fire"]: str(G1_MODEL_DIR / scenes.FIRE_29),
}

# The 12 leg joints, in the order the policy/config expect. The policy is
# indexed by these (by name), so it works on any G1 model layout.
LEG_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


class G1Sim:
    """12-DOF locomotion sim + kinematic full-body render overlay."""

    def __init__(self, scene_path: str | None = None, overlay_scene: str | None = None,
                 spec=None):
        self.scene_path = scene_path or SCENES["flat"]
        self.overlay_scene = overlay_scene
        self.spec = spec

        with open(DEPLOY_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        self.sim_dt = cfg["simulation_dt"]                  # 0.002 -> 500 Hz
        self.control_decimation = cfg["control_decimation"]  # 10 -> 50 Hz policy
        self.kps = np.array(cfg["kps"], dtype=np.float32)
        self.kds = np.array(cfg["kds"], dtype=np.float32)
        self.default_angles = np.array(cfg["default_angles"], dtype=np.float32)
        self.ang_vel_scale = cfg["ang_vel_scale"]
        self.dof_pos_scale = cfg["dof_pos_scale"]
        self.dof_vel_scale = cfg["dof_vel_scale"]
        self.action_scale = cfg["action_scale"]
        self.cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)
        self.num_actions = cfg["num_actions"]
        self.num_obs = cfg["num_obs"]

        # Live control state.
        self.cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # vx, vy, yaw
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.counter = 0
        self._fell = False

        # MuJoCo physics model (12-DOF).
        self.model = mujoco.MjModel.from_xml_path(self.scene_path)
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)
        self._build_leg_maps()

        self._spawn_qpos = self.data.qpos.copy()
        self._spawn_qpos[self.leg_qadr] = self.default_angles
        if spec is not None:
            # Spawn the base at the SceneSpec start (x, y, yaw); keep the model's
            # default standing height in z. Quaternion is [w, x, y, z].
            sx, sy, syaw = spec.start
            self._spawn_qpos[0] = sx
            self._spawn_qpos[1] = sy
            self._spawn_qpos[3:7] = [np.cos(syaw / 2), 0.0, 0.0, np.sin(syaw / 2)]
        self._reset_state()
        mujoco.mj_forward(self.model, self.data)

        # Demo loop: snap back to start on a fall or after clearing the course.
        self.auto_reset = False
        self.reset_x = (spec.bounds[1] - 0.5) if spec is not None else 6.5

        # Heading-hold: outer P loop keeping the robot straight along y=0 facing
        # +x (the blind policy curves otherwise). Manual A/D overrides it.
        self.heading_hold = False
        self.heading_target = 0.0
        self.lateral_target = 0.0
        self._manual_yaw = False

        torch.set_num_threads(1)  # tiny MLP; keep it off the sim/render threads
        self.policy = torch.jit.load(str(POLICY_PATH))
        self.policy.eval()

        # Rendering (model/renderer created lazily in the render thread).
        self.render_model = None
        self.render_data = None
        self.overlay_arm_pose = "carry"  # preset name OR {joint: angle} dict
        self.spraying = False            # fire scene: water jet on/off
        self.jet_speed = 7.0             # m/s muzzle speed of the water
        self.aim_pitch = NOZZLE_REST_PITCH   # fixed nozzle elevation (no auto-aim)
        if spec is not None:
            self.fire_positions = [(fx, fy, 0.0) for fx, fy in spec.fires]
        else:
            self.fire_positions = [scenes.FIRE_POSITION_WORLD]
        self.fire_health = [1.0] * len(self.fire_positions)  # 1 = burning, 0 = out
        if spec is not None:
            from .nav import nearest_burning_fire
            idx = nearest_burning_fire(spec.fires, self.fire_health,
                                       spec.start[0], spec.start[1])
            self.targeted_fire = idx if idx is not None else 0
        else:
            self.targeted_fire = 0
        self._hitting = False                # is the jet currently on the fire?
        self.fx = None                   # effects.SceneFX (fire scene only)
        self._hose_bid = -1
        self._water = None               # cached per-frame jet path (for ego view)
        self._water_landing = None
        self._water_hit_idx = None
        self.renderer = None
        self.cam = mujoco.MjvCamera()
        self.cam.distance = 3.0
        self.cam.elevation = -15.0
        self.cam.azimuth = 135.0

        # Robot-mounted ego camera (overlay only): separate renderer + frame
        # buffer + cached world pose (for later ground-plane projection).
        self.cam_renderer = None
        self.ego_cam_id = -1
        self._last_cam_rgb = None
        self._ego_pose = None  # (pos[3], rot[3x3] world<-cam) or None

        self.frames = FrameBuffer()
        self.cam_frames = FrameBuffer()
        self._cmd_lock = threading.Lock()
        self._running = False
        self.navigator = None
        self._approach_enabled = False
        self._approach_ready = False
        self._nav_costmap = None
        self._nav_path: list[tuple[float, float]] = []
        self._blocked = False
        self._pos_history: deque[tuple[float, float, float]] = deque()
        self._blocked_replan_done = False

    # -- public control API ------------------------------------------------- #
    def set_command(self, vx=None, vy=None, yaw=None, _nav=False):
        with self._cmd_lock:
            if not _nav:
                self.navigator = None
                self._approach_ready = False
            if vx is not None:
                self.cmd[0] = clamp(vx, *VX_RANGE)
            if vy is not None:
                self.cmd[1] = clamp(vy, *VY_RANGE)
            if yaw is not None:
                # A manual yaw input takes over from heading-hold so the user
                # can steer; set_heading_hold re-enables auto-steer.
                self._manual_yaw = True
                self.cmd[2] = clamp(yaw, *YAW_RANGE)
            return self.cmd.copy()

    def set_heading_hold(self, on=True):
        self.heading_hold = on
        self._manual_yaw = not on
        return on

    def set_overlay_arm_pose(self, pose):
        """Pose the overlay arms.

        ``pose`` is either a preset name -- ``'carry'`` (hold hose), ``'aim'``,
        ``'down'`` -- or a dict of ``{joint_name: angle_rad}`` for continuous
        control. Joint names may omit the ``_joint`` suffix; unspecified arm
        joints go to 0. Invalid input is ignored (the previous pose is kept).
        Returns the active pose."""
        if isinstance(pose, str):
            if pose in arms.ARM_POSES:
                self.overlay_arm_pose = pose
        elif isinstance(pose, dict):
            try:
                self.overlay_arm_pose = {str(k): float(v) for k, v in pose.items()}
            except (TypeError, ValueError):
                pass
        return self.overlay_arm_pose

    def set_spray(self, on=None):
        """Toggle (or set) the water jet (fire scene only). Returns the state."""
        self.spraying = (not self.spraying) if on is None else bool(on)
        return self.spraying

    def reignite(self):
        """Restore all fires to full health (for repeated demo runs)."""
        self.fire_health = [1.0] * len(self.fire_positions)
        return self.fire_health[self.targeted_fire]

    @property
    def arm_pose_label(self) -> str:
        """Short label for the active pose (preset name, or 'custom')."""
        return (self.overlay_arm_pose if isinstance(self.overlay_arm_pose, str)
                else "custom")

    def get_state(self):
        """Pelvis pose/vel (world frame) + command + fallen flag + arm pose."""
        return {
            "pos": self.data.qpos[0:3].copy(),
            "quat": self.data.qpos[3:7].copy(),
            "lin_vel": self.data.qvel[0:3].copy(),
            "ang_vel": self.data.qvel[3:6].copy(),
            "cmd": self.cmd.copy(),
            "fell": self._fell,
            "sim_time": self.data.time,
            "arm_pose": self.arm_pose_label if self.overlay_scene else None,
            "spraying": self.spraying,
            "hitting": self._hitting,
            "fires": [{"pos": p, "health": h}
                      for p, h in zip(self.fire_positions, self.fire_health)],
            "targeted_fire": self.targeted_fire,
            "fire_health": self.fire_health[self.targeted_fire],
            "blocked": self._blocked,
            "approach_phase": (self.navigator.phase
                               if self._is_approach_controller()
                               else ("ready" if self._approach_ready else None)),
        }

    # -- navigation / autonomous approach ----------------------------------- #
    def _ensure_nav_costmap(self) -> None:
        if self._nav_costmap is None and self.spec is not None:
            from .nav import Costmap
            self._nav_costmap = Costmap.from_spec(self.spec)

    def _is_approach_controller(self) -> bool:
        from .nav import ApproachController
        return isinstance(self.navigator, ApproachController)

    def _advance_targeted_fire(self) -> None:
        from .nav import nearest_burning_fire
        x, y = float(self.data.qpos[0]), float(self.data.qpos[1])
        nxt = nearest_burning_fire(self.fire_positions, self.fire_health, x, y)
        if nxt is None:
            return
        if nxt == self.targeted_fire:
            return
        self.targeted_fire = nxt
        self._nav_path = []
        self._replan_approach_if_active()

    def _replan_approach_if_active(self) -> None:
        if not self._approach_enabled:
            return
        if not self.set_approach(True):
            self.navigator = None
            self._approach_enabled = False

    def set_targeted_fire(self, idx: int) -> int:
        if idx < 0 or idx >= len(self.fire_positions):
            return self.targeted_fire
        if self.fire_health[idx] <= 0:
            return self.targeted_fire
        self.targeted_fire = idx
        self._nav_path = []
        self._replan_approach_if_active()
        return self.targeted_fire

    def _nav_spray_goal(self) -> tuple[float, float] | None:
        if self._is_approach_controller():
            return self.navigator.spray_goal
        if self.spec is None:
            return None
        self._ensure_nav_costmap()
        from .nav import safe_spray_point
        cm = self._nav_costmap
        if cm is None:
            return None
        state = self.get_state()
        x, y = float(state["pos"][0]), float(state["pos"][1])
        tf = int(state["targeted_fire"])
        fx, fy = self.fire_positions[tf][0], self.fire_positions[tf][1]
        return safe_spray_point(cm, (fx, fy), self.fire_positions, (x, y),
                                target_idx=tf)

    def plan_nav_path(self) -> list[tuple[float, float]]:
        """A* from current pose to safe spray point; caches costmap + path."""
        if self.spec is None:
            self._nav_costmap = None
            self._nav_path = []
            return []
        if self._is_approach_controller():
            return list(self.navigator.path)
        self._ensure_nav_costmap()
        from .nav import astar
        cm = self._nav_costmap
        state = self.get_state()
        x, y = float(state["pos"][0]), float(state["pos"][1])
        goal = self._nav_spray_goal()
        if goal is None:
            self._nav_path = []
            return []
        self._nav_path = astar(cm, (x, y), goal)
        return self._nav_path

    def set_approach(self, on: bool) -> bool:
        """Autonomously approach targeted fire: navigate, face, ready."""
        self._approach_enabled = bool(on)
        self._approach_ready = False
        if not self._approach_enabled:
            self.navigator = None
            return False
        if self.spec is None:
            self._approach_enabled = False
            return False
        from .nav import ApproachController, astar, nearest_burning_fire, safe_spray_point
        self._ensure_nav_costmap()
        cm = self._nav_costmap
        if cm is None:
            self._approach_enabled = False
            return False
        tf = self.targeted_fire
        if tf >= len(self.fire_health) or self.fire_health[tf] <= 0:
            nxt = nearest_burning_fire(
                self.fire_positions, self.fire_health,
                float(self.data.qpos[0]), float(self.data.qpos[1]))
            if nxt is None:
                self._approach_enabled = False
                return False
            self.targeted_fire = nxt
            tf = nxt
        x, y = float(self.data.qpos[0]), float(self.data.qpos[1])
        fx, fy = self.fire_positions[tf][0], self.fire_positions[tf][1]
        goal = safe_spray_point(cm, (fx, fy), self.fire_positions, (x, y), target_idx=tf)
        if goal is None:
            self._approach_enabled = False
            return False
        path = astar(cm, (x, y), goal)
        if not path:
            self._approach_enabled = False
            return False
        self._nav_path = path
        self.navigator = ApproachController(self, path, (fx, fy), tf)
        self.heading_hold = False
        self._manual_yaw = False
        return True

    def nav_snapshot(self) -> dict | None:
        """Top-down nav overlay payload for the web viewer."""
        if self.spec is None:
            return None
        self._ensure_nav_costmap()
        cm = self._nav_costmap
        if cm is None:
            return None
        state = self.get_state()
        if self._is_approach_controller():
            path = self.navigator.path
            phase = self.navigator.phase
            spray_goal = self.navigator.spray_goal
        elif self._approach_enabled and self.navigator is not None:
            path = self.navigator.path
            phase = None
            spray_goal = getattr(self.navigator, "spray_goal", None)
        else:
            path = self.plan_nav_path()
            phase = None
            spray_goal = self._nav_spray_goal()
        ny, nx = cm.shape
        occ = cm.occupied.reshape(-1).tolist()
        yaw = quat_yaw(state["quat"])
        return {
            "origin": [cm.xmin, cm.ymin],
            "res": cm.res,
            "shape": [nx, ny],
            "occupied": occ,
            "path": [[p[0], p[1]] for p in path],
            "spray_goal": list(spray_goal) if spray_goal else None,
            "phase": phase,
            "fires": [{"pos": list(f["pos"]), "health": float(f["health"])}
                      for f in state["fires"]],
            "home": list(self.spec.home),
            "robot": {"pos": state["pos"].tolist(), "yaw": float(yaw)},
            "targeted_fire": int(state["targeted_fire"]),
            "approach": bool(self._approach_enabled),
        }

    def _update_blocked(self) -> None:
        t = float(self.data.time)
        x, y = float(self.data.qpos[0]), float(self.data.qpos[1])
        self._pos_history.append((t, x, y))
        while self._pos_history and t - self._pos_history[0][0] > BLOCKED_WINDOW:
            self._pos_history.popleft()

        with self._cmd_lock:
            cmd = self.cmd.copy()
        commanded = (self._approach_enabled or self.navigator is not None
                     or abs(float(cmd[0])) > BLOCKED_CMD_THRESH
                     or abs(float(cmd[1])) > BLOCKED_CMD_THRESH)

        if not commanded or len(self._pos_history) < 2:
            self._blocked = False
            return

        t0, x0, y0 = self._pos_history[0]
        if t - t0 < BLOCKED_WINDOW * 0.9:
            self._blocked = False
            return

        disp = math.hypot(x - x0, y - y0)
        self._blocked = disp < BLOCKED_DISP_THRESH

    # -- robot ego camera ---------------------------------------------------- #
    @property
    def has_camera(self) -> bool:
        return bool(self.overlay_scene)

    def get_camera_frame(self, encode: bool = False):
        """Latest robot-POV frame. ``encode=False`` -> RGB ndarray (feed the
        flame detector); ``encode=True`` -> JPEG bytes (for streaming). Returns
        None until the first ego frame has rendered."""
        return self.cam_frames.get() if encode else self._last_cam_rgb

    def get_camera_pose(self):
        """Ego camera world pose as ``(pos[3], rot[3x3])`` or None. ``rot``
        columns are the camera axes in world (MuJoCo cam looks along -Z)."""
        return self._ego_pose

    def get_camera_intrinsics(self):
        """(fovy_deg, width, height) of the ego camera."""
        return (CAM_FOVY, CAM_W, CAM_H)

    # -- joint maps & policy ------------------------------------------------- #
    def _build_leg_maps(self):
        m = self.model
        JNT, ACT = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_ACTUATOR
        nid = lambda obj, n: mujoco.mj_name2id(m, obj, n)
        self.leg_qadr = np.array([m.jnt_qposadr[nid(JNT, n)] for n in LEG_JOINTS])
        self.leg_vadr = np.array([m.jnt_dofadr[nid(JNT, n)] for n in LEG_JOINTS])
        self.leg_act = np.array([nid(ACT, n) for n in LEG_JOINTS])

    def _compute_action(self):
        q = self.data.qpos[self.leg_qadr]
        dq = self.data.qvel[self.leg_vadr]
        quat = self.data.qpos[3:7]
        omega = self.data.qvel[3:6]

        qj = (q - self.default_angles) * self.dof_pos_scale
        dqj = dq * self.dof_vel_scale
        grav = gravity_orientation(quat)
        omega = omega * self.ang_vel_scale

        # The blind policy balances *by* stepping (it marks time at zero
        # command); freezing the gait makes it drift/fall, so the clock always
        # advances. heading_hold cancels positional wander.
        period = 0.8
        t = self.counter * self.sim_dt
        phase = (t % period) / period
        sin_phase, cos_phase = np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)

        with self._cmd_lock:
            cmd = self.cmd.copy()

        n = self.num_actions
        self.obs[:3] = omega
        self.obs[3:6] = grav
        self.obs[6:9] = cmd * self.cmd_scale
        self.obs[9:9 + n] = qj
        self.obs[9 + n:9 + 2 * n] = dqj
        self.obs[9 + 2 * n:9 + 3 * n] = self.action
        self.obs[9 + 3 * n:9 + 3 * n + 2] = [sin_phase, cos_phase]

        with torch.no_grad():
            obs_t = torch.from_numpy(self.obs).unsqueeze(0)
            self.action = self.policy(obs_t).detach().numpy().squeeze()
        self.target_dof_pos = self.action * self.action_scale + self.default_angles

    # -- kinematic full-body render overlay ---------------------------------- #
    def _build_render_maps(self):
        """Index the overlay body's legs/arms. Runs in the render thread."""
        m = self.render_model
        JNT = mujoco.mjtObj.mjOBJ_JOINT
        nid = lambda n: mujoco.mj_name2id(m, JNT, n)
        self._render_leg_qadr = np.array([m.jnt_qposadr[nid(n)] for n in LEG_JOINTS])
        # The nozzle joint is driven separately (aim), not by arm presets.
        skip = set(LEG_JOINTS) | {NOZZLE_JOINT}
        self._render_aux_names, qadr = [], []
        for i in range(m.njnt):
            if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            name = mujoco.mj_id2name(m, JNT, i)
            if name in skip:
                continue
            self._render_aux_names.append(name)
            qadr.append(m.jnt_qposadr[i])
        self._render_aux_qadr = np.array(qadr, dtype=int)
        nid_noz = nid(NOZZLE_JOINT)
        self._nozzle_qadr = m.jnt_qposadr[nid_noz] if nid_noz >= 0 else -1

    def _init_render_model(self):
        """Build the 29-DOF overlay model with a torso-mounted ego camera, and
        create both the third-person and ego renderers. Runs in the render
        thread (owns the GL context)."""
        spec = mujoco.MjSpec.from_file(self.overlay_scene)
        try:
            cam = spec.body("torso_link").add_camera()
            cam.name = "ego"
            cam.pos = list(EGO_CAM_POS)
            cam.fovy = CAM_FOVY
            cam.quat = _ego_camera_quat(EGO_CAM_TILT_DEG)
        except Exception as e:  # overlay still works without the ego cam
            print("ego camera attach failed:", e)
        try:
            _attach_hose(spec)
        except Exception as e:  # overlay still works without the hose prop
            print("hose attach failed:", e)
        self.render_model = spec.compile()
        # One offscreen buffer must hold the larger of the two render sizes.
        self.render_model.vis.global_.offwidth = max(RENDER_W, CAM_W)
        self.render_model.vis.global_.offheight = max(RENDER_H, CAM_H)
        self.render_data = mujoco.MjData(self.render_model)
        self._build_render_maps()
        self.renderer = mujoco.Renderer(self.render_model, RENDER_H, RENDER_W)
        self.ego_cam_id = mujoco.mj_name2id(
            self.render_model, mujoco.mjtObj.mjOBJ_CAMERA, "ego")
        if self.ego_cam_id >= 0:
            self.cam_renderer = mujoco.Renderer(self.render_model, CAM_H, CAM_W)
        self._hose_bid = mujoco.mj_name2id(
            self.render_model, mujoco.mjtObj.mjOBJ_BODY, "hose")
        self.fx = effects.SceneFX(self.render_model)

    def _muzzle_world(self):
        """Nozzle muzzle world ``(position, unit_direction)``, or ``None``. The
        direction is the real barrel axis, so the jet goes where the nozzle
        points. Call after the overlay kinematics are forwarded."""
        if self._hose_bid < 0:
            return None
        rd = self.render_data
        rot = rd.xmat[self._hose_bid].reshape(3, 3)
        d_local = np.array([math.sin(HOSE_BARREL_TILT), 0.0, math.cos(HOSE_BARREL_TILT)])
        direction = rot @ d_local
        pos = rd.xpos[self._hose_bid] + direction * NOZZLE_REACH
        return pos, direction / (np.linalg.norm(direction) + 1e-9)

    def _update_jet(self):
        """Per-frame water ballistics, hit test and extinguish progress (fire
        scene only). Caches the jet path so the ego view draws the same arc. The
        nozzle sits at a fixed elevation -- there is no closed-loop aiming."""
        self._water = self._water_landing = self._water_hit_idx = None
        self._hitting = False
        if not (self.fx and self.fx.active and self.spraying):
            return
        mz = self._muzzle_world()
        if mz is None:
            return
        muzzle, direction = mz
        pts, landing = effects.trajectory(muzzle, direction, self.jet_speed)
        tidx = self.targeted_fire
        if tidx >= len(self.fire_positions):
            tidx = 0
        pos = self.fire_positions[tidx]
        center = np.asarray(pos, float) + [0, 0, FIRE_AIM_HEIGHT]
        idx, hit, _ = effects.hit_index(pts, center, HIT_RADIUS)
        self._water, self._water_landing, self._water_hit_idx = pts, landing, idx
        health = self.fire_health[tidx]
        self._hitting = hit = hit and health > 0.0
        if hit:
            self.fire_health[tidx] = max(
                0.0, health - 1.0 / (EXTINGUISH_TIME * RENDER_FPS))
            if health > 0.0 and self.fire_health[tidx] <= 0.0:
                self._advance_targeted_fire()

    def _decorate(self, scene):
        fires = list(zip(self.fire_positions, self.fire_health))
        self.fx.decorate(scene, fires=fires, water=self._water,
                         landing=self._water_landing, hit_idx=self._water_hit_idx,
                         hitting=self._hitting, targeted_idx=self.targeted_fire)

    def render_once(self):
        """Render the current frame to the shared frame buffer. Must run on the
        thread that owns the GL context (see :meth:`run`)."""
        if self.overlay_scene:
            if self.render_model is None:
                self._init_render_model()
            rd = self.render_data
            # Drive the overlay body from physics: base + legs copied, arms posed.
            rd.qpos[0:7] = self.data.qpos[0:7]
            rd.qpos[self._render_leg_qadr] = self.data.qpos[self.leg_qadr]
            targets = arms.resolve_pose(self.overlay_arm_pose, self._render_aux_names)
            for name, adr in zip(self._render_aux_names, self._render_aux_qadr):
                rd.qpos[adr] = targets[name]
            if self._nozzle_qadr >= 0:
                rd.qpos[self._nozzle_qadr] = self.aim_pitch
            self.fx.animate(self.render_model, self.fire_health)
            mujoco.mj_forward(self.render_model, rd)
            self._update_jet()
            self.cam.lookat[:] = rd.qpos[0:3]
            self.renderer.update_scene(rd, self.cam)
            self._decorate(self.renderer.scene)
        else:
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, RENDER_H, RENDER_W)
            self.cam.lookat[:] = self.data.qpos[0:3]
            self.renderer.update_scene(self.data, self.cam)
        self.frames.set(encode_jpeg(self.renderer.render()))

    def render_camera_once(self):
        """Render the robot's ego view (overlay only). Call after
        :meth:`render_once` so the overlay kinematics are already forwarded."""
        if self.cam_renderer is None or self.ego_cam_id < 0:
            return
        self.cam_renderer.update_scene(self.render_data, camera=self.ego_cam_id)
        self._decorate(self.cam_renderer.scene)
        rgb = self.cam_renderer.render()
        self._last_cam_rgb = rgb
        self.cam_frames.set(encode_jpeg(rgb))
        cid = self.ego_cam_id
        self._ego_pose = (self.render_data.cam_xpos[cid].copy(),
                          self.render_data.cam_xmat[cid].reshape(3, 3).copy())

    def _render_loop(self):
        interval = 1.0 / RENDER_FPS
        while self._running:
            t0 = time.time()
            try:
                self.render_once()
                self.render_camera_once()
            except Exception as e:  # a render hiccup must not kill the demo
                print("render error:", e)
            sleep = interval - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)

    # -- main loop ----------------------------------------------------------- #
    def run(self, render=True):
        """Blocking physics loop: 500 Hz sim, 50 Hz policy, paced to real time.
        Rendering runs on its own thread with its own GL context."""
        self._running = True
        render_thread = None
        if render:
            render_thread = threading.Thread(
                target=self._render_loop, daemon=True, name="g1-render")
            render_thread.start()
        dec = self.control_decimation
        try:
            while self._running:
                tick_start = time.time()
                for _ in range(dec):
                    leg_tau = pd_control(
                        self.target_dof_pos, self.data.qpos[self.leg_qadr],
                        self.kps, np.zeros_like(self.kds),
                        self.data.qvel[self.leg_vadr], self.kds,
                    )
                    self.data.ctrl[self.leg_act] = leg_tau
                    mujoco.mj_step(self.model, self.data)
                    self.counter += 1

                if self.navigator is not None:
                    if self.navigator.update():
                        if (self._is_approach_controller()
                                and self.navigator.phase == "ready"):
                            self._approach_ready = True
                        self.navigator = None
                        self._approach_enabled = False
                elif self.heading_hold and not self._manual_yaw:
                    self.cmd[2] = self._steer()
                self._update_blocked()
                if (self._approach_enabled and self._blocked
                        and not self._blocked_replan_done):
                    if self._is_approach_controller():
                        if not self.navigator.replan():
                            self.navigator = None
                            self._approach_enabled = False
                    else:
                        old_path = list(self.navigator.path) if self.navigator else []
                        path = self.plan_nav_path()
                        if path and path != old_path:
                            from .nav import WaypointFollower
                            self.navigator = WaypointFollower(self, path)
                        else:
                            self.navigator = None
                            self._approach_enabled = False
                    self._blocked_replan_done = True
                elif not self._blocked:
                    self._blocked_replan_done = False
                self._compute_action()
                self._fell = self.data.qpos[2] < 0.4

                if self.auto_reset and (self._fell or self.data.qpos[0] > self.reset_x):
                    time.sleep(0.4)
                    self._reset_state()

                sleep = dec * self.sim_dt - (time.time() - tick_start)
                if sleep > 0:
                    time.sleep(sleep)
        finally:
            self._running = False
            if render_thread is not None:
                render_thread.join(timeout=3.0)
            self._release_render()

    def _steer(self):
        """Yaw rate driving heading->heading_target and y->lateral_target.

        Only meaningful on the flat/obstacle demo corridors; procedural scenes
        manage heading via the navigator, so hold the current heading there."""
        if self.spec is not None:
            return 0.0
        yaw = quat_yaw(self.data.qpos[3:7])
        heading_err = np.arctan2(np.sin(self.heading_target - yaw),
                                 np.cos(self.heading_target - yaw))
        lateral_err = self.lateral_target - self.data.qpos[1]
        return clamp(1.8 * heading_err + 0.9 * lateral_err, *YAW_RANGE)

    def _reset_state(self):
        self.data.qpos[:] = self._spawn_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.action[:] = 0.0
        self.target_dof_pos = self.default_angles.copy()
        self.counter = 0
        self._fell = False
        mujoco.mj_forward(self.model, self.data)

    def reset_to_start(self):
        """Teleport the robot back to its spawn pose and clear autonomy state."""
        with self._cmd_lock:
            self.cmd[:] = 0.0
            self.navigator = None
            self._approach_enabled = False
            self._approach_ready = False
            self._manual_yaw = False
            self._blocked = False
            self._blocked_replan_done = False
            self._pos_history.clear()
            self._nav_path = []
            self._reset_state()
        return self.data.qpos[0:3].copy()

    def stop(self):
        self._running = False

    def _release_render(self):
        """Drop GL resources after the render thread has exited."""
        self.renderer = None
        self.cam_renderer = None
        self.render_model = None
        self.render_data = None
        self.fx = None
