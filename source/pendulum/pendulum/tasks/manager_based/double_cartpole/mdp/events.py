# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events specific to the double cart-pole."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = ["reset_double_poles_graded", "reset_double_poles_uniform"]


def reset_double_poles_graded(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    ipole_cfg: SceneEntityCfg,
    opole_cfg: SceneEntityCfg,
    near_fraction: float = 0.2,
    mid_fraction: float = 0.2,
    # near_pos_range: tuple[float, float] = (-0.15, 0.15),
    # near_vel_range: tuple[float, float] = (-0.5, 0.5),
    near_pos_range: tuple[float, float] = (-0.1, 0.1),
    near_vel_range: tuple[float, float] = (-0.2, 0.2),
    mid_ipos_range: tuple[float, float] = (0.5 * math.pi, 1.5 * math.pi),
    mid_opos_range: tuple[float, float] = (-0.5, 0.5),
    mid_vel_range: tuple[float, float] = (-3.0, 3.0),
    far_ipos_mean: float = math.pi,
    far_ipos_std: float = 0.05,
    far_opos_std: float = 0.10,
    far_vel_std: float = 0.05,
) -> None:
    """Three-tier reverse curriculum on both revolute joints, sampled jointly per env.

    near : both links close to upright -> the policy practises the catch every episode
    mid  : inner anywhere in the upper/lower half, outer roughly aligned -> bridges the two
    far  : hanging with symmetry-breaking noise (dm_control's swing_up reset, incl. its wider
           sigma on the second link)

    Deliberately NOT a full-circle uniform reset: sampling both angles uniformly with large
    velocities is mostly unrecoverable chaos and wastes the batch. Offsets are added to the joints'
    default positions (0 == upright), so `far` centres the inner link at pi. The cart is left to a
    separate reset term.

    Using this REQUIRES removing `pole_fell` from the terminations: the mid and far tiers start at or
    below the tip threshold, so every one of those episodes would terminate on step 1.
    """
    robot: Articulation = env.scene[ipole_cfg.name]
    i_idx = robot.find_joints(ipole_cfg.joint_names)[0][0]
    o_idx = robot.find_joints(opole_cfg.joint_names)[0][0]

    n = len(env_ids)
    device = robot.device
    tier = torch.rand(n, device=device)
    near = tier < near_fraction
    mid = (tier >= near_fraction) & (tier < near_fraction + mid_fraction)

    def _u(lo: float, hi: float) -> torch.Tensor:
        return torch.empty(n, device=device).uniform_(lo, hi)

    def _n(mean: float, std: float) -> torch.Tensor:
        return torch.normal(mean=mean, std=std, size=(n,), device=device)

    th1 = torch.where(near, _u(*near_pos_range), torch.where(mid, _u(*mid_ipos_range), _n(far_ipos_mean, far_ipos_std)))
    th2 = torch.where(near, _u(*near_pos_range), torch.where(mid, _u(*mid_opos_range), _n(0.0, far_opos_std)))
    w1 = torch.where(near, _u(*near_vel_range), torch.where(mid, _u(*mid_vel_range), _n(0.0, far_vel_std)))
    w2 = torch.where(near, _u(*near_vel_range), torch.where(mid, _u(*mid_vel_range), _n(0.0, far_vel_std)))

    for idx, pos_off, vel_off in ((i_idx, th1, w1), (o_idx, th2, w2)):
        pos = robot.data.default_joint_pos[env_ids][:, [idx]] + pos_off.unsqueeze(-1)
        vel = robot.data.default_joint_vel[env_ids][:, [idx]] + vel_off.unsqueeze(-1)
        robot.write_joint_state_to_sim(pos, vel, joint_ids=[idx], env_ids=env_ids)


def reset_double_poles_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    ipole_cfg: SceneEntityCfg,
    opole_cfg: SceneEntityCfg,
    near_fraction: float = 0.15,
    hang_fraction: float = 0.15,
    near_pos_range: tuple[float, float] = (-0.15, 0.15),
    near_vel_range: tuple[float, float] = (-0.5, 0.5),
    hang_ipos_std: float = 0.05,
    hang_opos_std: float = 0.10,
    hang_vel_std: float = 0.05,
    ipos_range: tuple[float, float] = (-math.pi, math.pi),
    opos_range: tuple[float, float] = (-math.pi, math.pi),
    ivel_range: tuple[float, float] = (-10.0, 10.0),
    ovel_range: tuple[float, float] = (-20.0, 20.0),
) -> None:
    """Lee et al. Eq. (10) full-state reset, with an upright and a hanging tier carved out.

    near : both links upright -> keeps the catch in every batch (PPO has no replay buffer, so
           without this anchor the existing catch competence decays before coverage repays it)
    hang : the classic swing-up start, dm_control's symmetry-breaking noise
    rest : uniform over BOTH full circles with per-joint velocity ranges

    The uniform tier is the point: `reset_double_poles_graded` never samples |th2| above 28.6 deg
    or |w2| above 3 rad/s, so it only ever visits the down-down <-> up-up axis and the policy sees
    folded states solely as ones it produced itself. Velocity defaults are Lee's (+/-10, +/-20);
    the outer link needs roughly double because it is lighter and spins ~30 rad/s mid-swing-up.
    Well inside the 6000 deg/s rigid-body clamp in configurations.py.

    Uniform angles mean most episodes start "fallen", so like the graded reset this REQUIRES that
    `pole_fell` stay out of the terminations. Offsets are added to the default positions
    (0 == upright); the cart is left to a separate reset term.
    """
    robot: Articulation = env.scene[ipole_cfg.name]
    i_idx = robot.find_joints(ipole_cfg.joint_names)[0][0]
    o_idx = robot.find_joints(opole_cfg.joint_names)[0][0]

    n = len(env_ids)
    device = robot.device
    tier = torch.rand(n, device=device)
    near = tier < near_fraction
    hang = (tier >= near_fraction) & (tier < near_fraction + hang_fraction)

    def _u(lo: float, hi: float) -> torch.Tensor:
        return torch.empty(n, device=device).uniform_(lo, hi)

    def _n(mean: float, std: float) -> torch.Tensor:
        return torch.normal(mean=mean, std=std, size=(n,), device=device)

    th1 = torch.where(near, _u(*near_pos_range), torch.where(hang, _n(math.pi, hang_ipos_std), _u(*ipos_range)))
    th2 = torch.where(near, _u(*near_pos_range), torch.where(hang, _n(0.0, hang_opos_std), _u(*opos_range)))
    w1 = torch.where(near, _u(*near_vel_range), torch.where(hang, _n(0.0, hang_vel_std), _u(*ivel_range)))
    w2 = torch.where(near, _u(*near_vel_range), torch.where(hang, _n(0.0, hang_vel_std), _u(*ovel_range)))

    for idx, pos_off, vel_off in ((i_idx, th1, w1), (o_idx, th2, w2)):
        pos = robot.data.default_joint_pos[env_ids][:, [idx]] + pos_off.unsqueeze(-1)
        vel = robot.data.default_joint_vel[env_ids][:, [idx]] + vel_off.unsqueeze(-1)
        robot.write_joint_state_to_sim(pos, vel, joint_ids=[idx], env_ids=env_ids)
