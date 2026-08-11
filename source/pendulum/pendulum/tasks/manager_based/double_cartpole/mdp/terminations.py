# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms specific to the double cart-pole."""

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
    """True once the outer link's tip drops below `min_height_frac * (l1 + l2)`.

    One condition covering both links, which is what Gymnasium's InvertedDoublePendulum uses for this
    same system (it terminates at tip height <= half of full extension). A per-joint angle limit does
    not work here: th2 is RELATIVE, so |th2| small only means "outer aligned with inner" and would pass
    a fully tipped-over but straight assembly.

    At the 0.5 default the threshold is 0.125 m of 0.25 m max, which fires when a rigid pair passes
    ~60 deg from vertical, or when the outer link reaches horizontal with the inner still up. Raise
    toward 0.7 (0.175 m, ~45 deg rigid) for shorter episodes and a denser learning signal, at the cost
    of cutting off states that were still recoverable.

    Safe to use because the reward is POSITIVE: ending early forfeits the remaining stream, so falling
    is a real loss. With a non-positive reward this would be exploitable.
    """
    robot: Articulation = env.scene[ipole_cfg.name]
    th1 = wrap_to_pi(robot.data.joint_pos[:, ipole_cfg.joint_ids[0]])
    th2 = wrap_to_pi(robot.data.joint_pos[:, opole_cfg.joint_ids[0]])
    return tip_height(th1, th2, l1, l2) < min_height_frac * (l1 + l2)
