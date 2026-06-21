"""G1 arm + hand poses for the kinematic hose-carry / aim overlay.

Two ways to drive the overlay arms (see :func:`resolve_pose`):

- **Named presets** (``"carry"``, ``"aim"``, ``"down"``) -- canned demo poses.
- **Continuous joint dict** ``{"left_shoulder_pitch": -0.5, "left_elbow": 1.2}``
  -- the actuator handle the fire controller writes to. Joint names may omit the
  ``_joint`` suffix; unspecified arm joints render at 0 and fingers at a neutral
  curl.

Sign conventions (verified by rendering on the G1):

- **shoulder_pitch** (axis Y): negative raises the arm forward/up.
- The right arm mirrors the roll/yaw signs so the two hands meet at the centre.
- **Hands mirror**: left finger-curl joints have a negative range, right ones
  positive, so each hand gets its own explicit angle set below.
"""
from __future__ import annotations

from typing import Mapping, Union

# What set_overlay_arm_pose / resolve_pose accept.
Pose = Union[str, Mapping[str, float]]

# --- Hand shapes -----------------------------------------------------------
# Both hands wrap a horizontal nozzle barrel (fingers curl around it, thumb
# opposed). The hands are mirror images, so the curl signs flip per side.
LEFT_GRIP: dict[str, float] = {
    "left_hand_index_0_joint": -1.20, "left_hand_index_1_joint": -1.30,
    "left_hand_middle_0_joint": -1.20, "left_hand_middle_1_joint": -1.30,
    "left_hand_thumb_0_joint": 0.40, "left_hand_thumb_1_joint": 0.60,
    "left_hand_thumb_2_joint": 0.90,
}
RIGHT_GRIP: dict[str, float] = {
    "right_hand_index_0_joint": 1.20, "right_hand_index_1_joint": 1.30,
    "right_hand_middle_0_joint": 1.20, "right_hand_middle_1_joint": 1.30,
    "right_hand_thumb_0_joint": -0.40, "right_hand_thumb_1_joint": -0.60,
    "right_hand_thumb_2_joint": -0.90,
}
# Neutral light curl used to fill any finger a pose leaves unspecified
# (sign-correct per hand, so nothing ever renders bent backwards).
DEFAULT_FINGERS: dict[str, float] = {
    "left_hand_index_0_joint": -0.40, "left_hand_index_1_joint": -0.40,
    "left_hand_middle_0_joint": -0.40, "left_hand_middle_1_joint": -0.40,
    "left_hand_thumb_0_joint": 0.20, "left_hand_thumb_1_joint": 0.30,
    "left_hand_thumb_2_joint": 0.50,
    "right_hand_index_0_joint": 0.40, "right_hand_index_1_joint": 0.40,
    "right_hand_middle_0_joint": 0.40, "right_hand_middle_1_joint": 0.40,
    "right_hand_thumb_0_joint": -0.20, "right_hand_thumb_1_joint": -0.30,
    "right_hand_thumb_2_joint": -0.50,
}

# --- Arm presets -----------------------------------------------------------
ARM_POSES: dict[str, dict[str, float]] = {
    # Arms hang at the sides.
    "down": {},
    # Firefighter nozzle carry: both hands clustered on the nozzle at chest
    # height, barrel pointing forward (braced against the pressure). Right hand
    # higher/back on the bale, left hand lower/front on the barrel.
    "carry": {
        "left_shoulder_pitch_joint": -1.00, "right_shoulder_pitch_joint": -1.05,
        "left_shoulder_roll_joint": 0.11, "right_shoulder_roll_joint": 0.27,
        "left_shoulder_yaw_joint": 0.30, "right_shoulder_yaw_joint": -0.05,
        "left_elbow_joint": 1.60, "right_elbow_joint": 1.25,
        **LEFT_GRIP, **RIGHT_GRIP,
    },
    # Both arms extended forward, elbows less bent (nozzle pushed out to aim).
    "aim": {
        "left_shoulder_pitch_joint": -0.95, "right_shoulder_pitch_joint": -0.95,
        "left_shoulder_roll_joint": 0.10, "right_shoulder_roll_joint": -0.10,
        "left_shoulder_yaw_joint": 0.10, "right_shoulder_yaw_joint": -0.10,
        "left_elbow_joint": 0.55, "right_elbow_joint": 0.55,
        **LEFT_GRIP, **RIGHT_GRIP,
    },
}

POSE_NAMES = tuple(ARM_POSES)


def _match_joint(name: str, valid: set[str]) -> str | None:
    """Map a (possibly suffix-less) joint name to one present on the model."""
    if name in valid:
        return name
    suffixed = name if name.endswith("_joint") else name + "_joint"
    return suffixed if suffixed in valid else None


def resolve_pose(pose: Pose, joint_names) -> dict[str, float]:
    """Resolve a pose to a full {joint_name: angle} target over ``joint_names``.

    ``pose`` is either a preset name (looked up in :data:`ARM_POSES`) or a dict
    of joint angles (names may omit the ``_joint`` suffix). Every joint defaults
    to 0 except fingers, which default to a neutral, sign-correct curl
    (:data:`DEFAULT_FINGERS`); the pose's entries then override those, so a dict
    may also set fingers explicitly.

    Raises:
        KeyError: unknown preset name.
        TypeError: pose is neither a str nor a mapping.
    """
    valid = set(joint_names)
    targets = {n: 0.0 for n in joint_names}
    for name, angle in DEFAULT_FINGERS.items():
        joint = _match_joint(name, valid)
        if joint is not None:
            targets[joint] = angle

    if isinstance(pose, str):
        if pose not in ARM_POSES:
            raise KeyError(f"unknown arm preset {pose!r}; "
                           f"choices: {', '.join(POSE_NAMES)}")
        overrides: Mapping[str, float] = ARM_POSES[pose]
    elif isinstance(pose, Mapping):
        overrides = pose
    else:
        raise TypeError(f"pose must be a preset name or joint dict, got {type(pose)}")

    for name, angle in overrides.items():
        joint = _match_joint(name, valid)
        if joint is not None:
            targets[joint] = float(angle)
    return targets
