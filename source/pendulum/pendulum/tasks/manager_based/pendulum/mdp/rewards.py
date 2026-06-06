# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l1(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value using L1 norm."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    return torch.sum(torch.abs(joint_pos - target), dim=1)


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def joint_pos_target_exp(
    env: ManagerBasedRLEnv, target: float, sigma: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward joint position proximity to a target value using a gaussian kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    error = torch.sum(torch.square(joint_pos - target), dim=1)
    return torch.exp(-error / sigma**2)


def joint_effort_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize large effort commands (action magnitude) using L2."""
    return torch.sum(env.action_manager.action**2, dim=1)


def action_rate_l2(env) -> torch.Tensor:
    """Penalize rapid changes between consecutive actions."""
    return torch.sum(
        (env.action_manager.action - env.action_manager.prev_action) ** 2,
        dim=1,
    )
