# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def double_swingup_reward_quadratic(
    env: ManagerBasedRLEnv,
    ipole_cfg: SceneEntityCfg,
    opole_cfg: SceneEntityCfg,
    cart_cfg: SceneEntityCfg,
    w_ipole: float = 0.35,
    w_opole: float = 0.35,
    w_cart: float = 0.15,
    w_effort: float = 0.15,
    x_max: float = 0.35,
) -> torch.Tensor:
    """Dense LQR-like cost for single-net swing-up + balance of the DOUBLE pendulum.

    reward = 1 - (w_ipole*(th1/pi)^2 + w_opole*(th2/pi)^2 + w_cart*(x/x_max)^2 + w_effort*u^2).
    Both joint angles are 0 when the two links point straight up (fully inverted), so driving th1,th2
    toward 0 makes 'both up, centered, low effort' the unique optimum -- same anti-limit-cycle argument
    as the single cartpole quadratic. Weights are on normalized [0,1] terms (sum ~1 -> reward ~[0,1]).
    VERIFY the rig convention: assumes cart_to_ipole=0 and ipole_to_opole=0 == fully upright.
    """
    ipole: Articulation = env.scene[ipole_cfg.name]
    th1 = wrap_to_pi(ipole.data.joint_pos[:, ipole_cfg.joint_ids[0]])
    th2 = wrap_to_pi(env.scene[opole_cfg.name].data.joint_pos[:, opole_cfg.joint_ids[0]])
    x = env.scene[cart_cfg.name].data.joint_pos[:, cart_cfg.joint_ids[0]]
    u = env.action_manager.action[:, 0]

    cost = w_ipole * (th1 / math.pi) ** 2 + w_opole * (th2 / math.pi) ** 2 + w_cart * (x / x_max) ** 2 + w_effort * u**2
    return 1.0 - cost
