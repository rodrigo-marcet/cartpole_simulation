from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_pole_position_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    initial_range: float = 0.2,
    final_range: float = math.pi,
    curriculum_steps: int = 50000,
) -> None:
    asset: Articulation = env.scene[asset_cfg.name]

    progress = min(env.common_step_counter / curriculum_steps, 1.0)
    current_range = initial_range + (final_range - initial_range) * progress

    if asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids

    joint_pos = asset.data.default_joint_pos[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_vel = asset.data.default_joint_vel[iter_env_ids, asset_cfg.joint_ids].clone()

    joint_pos += torch.zeros_like(joint_pos).uniform_(-current_range, current_range)

    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
