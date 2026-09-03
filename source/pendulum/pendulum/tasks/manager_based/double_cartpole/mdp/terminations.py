from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

from .rewards import L1_DEFAULT, L2_DEFAULT, tip_height

__all__ = ["tip_below_height"]


def tip_below_height(
    env,
    ipole_cfg: SceneEntityCfg,
    opole_cfg: SceneEntityCfg,
    min_height_frac: float = 0.5,
    l1: float = L1_DEFAULT,
    l2: float = L2_DEFAULT,
) -> torch.Tensor:
    robot: Articulation = env.scene[ipole_cfg.name]
    th1 = wrap_to_pi(robot.data.joint_pos[:, ipole_cfg.joint_ids[0]])
    th2 = wrap_to_pi(robot.data.joint_pos[:, opole_cfg.joint_ids[0]])
    return tip_height(th1, th2, l1, l2) < min_height_frac * (l1 + l2)
