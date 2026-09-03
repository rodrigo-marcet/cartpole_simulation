from __future__ import annotations

import math
from functools import cache
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
    k_effort: float = 0.7792,
    # k_effort: float = 0.25,
    k_cart: float = 0.5,
    k_ivel: float = 0.02,
    k_ovel: float = 0.02,
) -> torch.Tensor:
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


X_LIM_DEFAULT = 0.35  # cart termination bound; sets the reachable region the reward is scaled over


@cache
def _goal_dist_max(w_x: float, goal_half_width: float, x_lim: float, l1: float, l2: float) -> float:
    h = l1 + l2
    phi = torch.linspace(0.0, math.pi / 2, 2001, dtype=torch.float64)
    dx = (x_lim + h * torch.cos(phi) - goal_half_width).clamp_min(0.0)
    return float(torch.sqrt(w_x * dx**2 + (-h * torch.sin(phi) - h) ** 2).max())


def tip_position_reward(
    env: ManagerBasedRLEnv,
    ipole_cfg: SceneEntityCfg,
    opole_cfg: SceneEntityCfg,
    cart_cfg: SceneEntityCfg,
    w_x: float = 0.178,
    goal_half_width: float = 0.05,
    l1: float = L1_DEFAULT,
    l2: float = L2_DEFAULT,
    x_lim: float = X_LIM_DEFAULT,
) -> torch.Tensor:
    th1, th2, _, _, x, _, _ = _unpack(env, ipole_cfg, opole_cfg, cart_cfg)

    tip_x = x + l1 * torch.sin(th1) + l2 * torch.sin(th1 + th2)
    tip_y = l1 * torch.cos(th1) + l2 * torch.cos(th1 + th2)

    dx = (tip_x.abs() - goal_half_width).clamp_min(0.0)
    d = torch.sqrt(w_x * dx**2 + (tip_y - (l1 + l2)) ** 2)

    return (1.0 - d / _goal_dist_max(w_x, goal_half_width, x_lim, l1, l2)).clamp_min(0.0)
