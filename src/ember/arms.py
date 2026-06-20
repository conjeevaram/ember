"""Named G1 arm poses for the kinematic hose-carry / aim overlay.

Sign conventions verified by rendering on the G1: **negative** shoulder_pitch
raises the arm forward; the right side mirrors the roll/yaw signs so the two
hands meet in front. Fingers close to a loose grip. Any joint not listed in a
pose renders at 0 (waist upright, wrists neutral).
"""
from __future__ import annotations

GRIP_FINGER_ANGLE = 0.6  # rad; all finger joints -> loose closed grip

ARM_POSES: dict[str, dict[str, float]] = {
    # Arms hang at the sides.
    "down": {},
    # Arms forward & slightly down, elbows ~90 deg, hands meeting (hold a hose).
    "carry": {
        "left_shoulder_pitch_joint": -0.55, "right_shoulder_pitch_joint": -0.55,
        "left_shoulder_roll_joint": 0.18, "right_shoulder_roll_joint": -0.18,
        "left_shoulder_yaw_joint": 0.15, "right_shoulder_yaw_joint": -0.15,
        "left_elbow_joint": 1.30, "right_elbow_joint": 1.30,
    },
    # Both arms extended forward, elbows less bent (nozzle aimed).
    "aim": {
        "left_shoulder_pitch_joint": -0.95, "right_shoulder_pitch_joint": -0.95,
        "left_shoulder_roll_joint": 0.10, "right_shoulder_roll_joint": -0.10,
        "left_shoulder_yaw_joint": 0.10, "right_shoulder_yaw_joint": -0.10,
        "left_elbow_joint": 0.55, "right_elbow_joint": 0.55,
    },
}

POSE_NAMES = tuple(ARM_POSES)


def resolve_pose(name: str, joint_names) -> dict[str, float]:
    """Resolve a pose name to {joint_name: angle} over the given joints, with
    fingers gripping and everything else at 0."""
    targets = {n: 0.0 for n in joint_names}
    for n, ang in ARM_POSES[name].items():
        if n in targets:
            targets[n] = ang
    for n in joint_names:
        if "hand" in n:
            targets[n] = GRIP_FINGER_ANGLE
    return targets
