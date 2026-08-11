# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the double cart-pole swing-up.

The reward is the multiplicative form from Lee, Ju & Lee, "Transition Control of a Double-Inverted
Pendulum System Using Sim2Real Reinforcement Learning", Machines 2025, 13, 186 -- the only published
end-to-end RL swing-up of a CART double pendulum on real hardware, at a 100 Hz policy rate, with no
LQR handoff. Constants are theirs where the units match and rescaled where they don't.
Rewards read ground-truth joint_pos, so observation noise never enters here.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

L1_DEFAULT = 0.125  # inner pivot -> outer pivot
L2_DEFAULT = 0.125  # outer pivot -> weight


def _unpack(env, ipole_cfg, opole_cfg, cart_cfg):
    """(th1, th2_rel, w1, w2_rel, x, u, u_prev). Angles wrapped, th1=0 is upright."""
    robot: Articulation = env.scene[ipole_cfg.name]
    i, o, c = ipole_cfg.joint_ids[0], opole_cfg.joint_ids[0], cart_cfg.joint_ids[0]
    return (
        wrap_to_pi(robot.data.joint_pos[:, i]),
        wrap_to_pi(robot.data.joint_pos[:, o]),
        robot.data.joint_vel[:, i],
        robot.data.joint_vel[:, o],
        robot.data.joint_pos[:, c],
        env.action_manager.action[:, 0],
        env.action_manager.prev_action[:, 0],
    )


def tip_height(th1: torch.Tensor, th2: torch.Tensor, l1: float, l2: float) -> torch.Tensor:
    """Height of the outer link tip above the inner pivot, metres. +(l1+l2) fully upright."""
    return l1 * torch.cos(th1) + l2 * torch.cos(th1 + th2)


def tip_height_norm(th1: torch.Tensor, th2: torch.Tensor, l1: float, l2: float) -> torch.Tensor:
    """tip_height mapped to [0, 1]; 1 = both links upright. Equals dm_control upright.mean() when l1==l2."""
    return (tip_height(th1, th2, l1, l2) + (l1 + l2)) / (2.0 * (l1 + l2))


def multiplicative_swingup_reward(
    env: ManagerBasedRLEnv,
    ipole_cfg: SceneEntityCfg,
    opole_cfg: SceneEntityCfg,
    cart_cfg: SceneEntityCfg,
    # k_effort: float = 0.7792,
    k_effort: float = 0.25,
    k_cart: float = 0.5,
    k_ivel: float = 0.02,
    k_ovel: float = 0.02,
) -> torch.Tensor:
    """Lee et al. (Machines 2025) eq. (8)-(9): the product of six [0,1] terms, so reward is in [0,1].

        Ru = exp(-k_effort*|u|)          Ry  = exp(-k_cart*|x|)
        Rt1 = 0.5 + 0.5*cos(th1)         Rt2 = 0.5 + 0.5*cos(th2)
        Rw1 = exp(-k_ivel*|w1|)          Rw2 = exp(-k_ovel*|w2|)
        reward = Ru * Ry * Rt1 * Rt2 * Rw1 * Rw2

    Their targets for the up-up equilibrium are th1* = th2* = 0, which is already our convention, and
    th2 is RELATIVE for them too. The product is what makes the relative angle safe here: it is 1 only
    when both links are up, and ~0 at every non-goal corner, so unlike an additive quadratic it never
    ranks a higher configuration as worse. It also self-gates the velocity terms -- near the bottom
    Rt1 ~ 0 swamps them, so pumping is effectively free without needing an explicit height gate.

    Constants: k_ivel, k_ovel and k_cart are theirs verbatim (rad/s and metres match our units).
    k_effort is CONVERTED, not guessed: their u is a cart acceleration in m/s^2, ours is a normalized
    action, and |u| = (F_max / m_translating) * |a| = (40 / 0.77) * |a| = 51.95 * |a|, so
    k_effort = 0.015 * 51.95 = 0.7792. Full effort then gives Ru = 0.459.
    POSITIVE reward, so it pairs with the rail termination.
    """
    th1, th2, w1, w2, x, u, _ = _unpack(env, ipole_cfg, opole_cfg, cart_cfg)

    r_u = torch.exp(-k_effort * u.abs())
    # r_u = torch.exp(-k_effort * u.abs() * u.abs())
    r_x = torch.exp(-k_cart * x.abs())
    r_t1 = 0.5 + 0.5 * torch.cos(th1)
    r_t2 = 0.5 + 0.5 * torch.cos(th2)
    r_w1 = torch.exp(-k_ivel * w1.abs())
    r_w2 = torch.exp(-k_ovel * w2.abs())

    return r_u * r_x * r_t1 * r_t2 * r_w1 * r_w2


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
    """Original quadratic on RELATIVE joint angles. Unused -- kept only as a fallback reference.

    Known defect: th2 is relative, so this ranks ~32% of state pairs backwards versus tip height and
    scores 'inner down / outer up' as worse than fully hanging.
    """
    ipole: Articulation = env.scene[ipole_cfg.name]
    th1 = wrap_to_pi(ipole.data.joint_pos[:, ipole_cfg.joint_ids[0]])
    th2 = wrap_to_pi(env.scene[opole_cfg.name].data.joint_pos[:, opole_cfg.joint_ids[0]])
    x = env.scene[cart_cfg.name].data.joint_pos[:, cart_cfg.joint_ids[0]]
    u = env.action_manager.action[:, 0]

    cost = w_ipole * (th1 / math.pi) ** 2 + w_opole * (th2 / math.pi) ** 2 + w_cart * (x / x_max) ** 2 + w_effort * u**2
    return 1.0 - cost
