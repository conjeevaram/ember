"""G1 arm + hand pose for the kinematic hose-carry overlay.

The overlay arms are **locked**: they always render in a single firefighter
nozzle-carry pose (both hands clustered on the barrel at chest height). There is
no runtime pose switching -- aiming is done by the body yaw and the nozzle
elevation, not by moving the arms.

Sign conventions (verified by rendering on the G1):

- **shoulder_pitch** (axis Y): negative raises the arm forward/up.
- The right arm mirrors the roll/yaw signs so the two hands meet at the centre.
- **Hands mirror**: left finger-curl joints have a negative range, right ones
  positive, so each hand gets its own explicit angle set below.
"""
from __future__ import annotations

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

# --- Locked carry pose -----------------------------------------------------
# Firefighter nozzle carry: both hands clustered on the nozzle at chest height,
# barrel pointing forward (braced against the pressure). Right hand higher/back
# on the bale, left hand lower/front on the barrel. This is the only pose the
# overlay ever renders.
LOCKED_POSE: dict[str, float] = {
    "left_shoulder_pitch_joint": -1.00, "right_shoulder_pitch_joint": -1.05,
    "left_shoulder_roll_joint": 0.11, "right_shoulder_roll_joint": 0.27,
    "left_shoulder_yaw_joint": 0.30, "right_shoulder_yaw_joint": -0.05,
    "left_elbow_joint": 1.60, "right_elbow_joint": 1.25,
    **LEFT_GRIP, **RIGHT_GRIP,
}


def _match_joint(name: str, valid: set[str]) -> str | None:
    """Map a (possibly suffix-less) joint name to one present on the model."""
    if name in valid:
        return name
    suffixed = name if name.endswith("_joint") else name + "_joint"
    return suffixed if suffixed in valid else None


def locked_pose(joint_names) -> dict[str, float]:
    """Resolve the locked carry pose to a full {joint_name: angle} target.

    Every joint defaults to 0; the locked carry angles (arms + finger grips)
    then override those. Joint names may omit the ``_joint`` suffix.
    """
    valid = set(joint_names)
    targets = {n: 0.0 for n in joint_names}
    for name, angle in LOCKED_POSE.items():
        joint = _match_joint(name, valid)
        if joint is not None:
            targets[joint] = float(angle)
    return targets
