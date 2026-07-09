from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = [
    "randomize_slider_friction_effort",
    "randomize_pole_vel_noise_std",
    "reset_joints_by_offset_unclamped",
]


def randomize_slider_friction_effort(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    static_range: tuple[float, float] = (1.7, 2.0),  # was (1.5, 2.0)
    dynamic_params: tuple[float, float] = (1.7, 0.17),  # was (1.18, 0.12); scales with effective mass
    viscous_value: float = 0.0,
) -> None:
    asset: Articulation = env.scene[asset_cfg.name]

    slider_ids, _ = asset.find_joints(asset_cfg.joint_names)
    slider_idx = slider_ids[0]

    n = len(env_ids)
    device = asset.device

    static = torch.empty(n, device=device).uniform_(static_range[0], static_range[1])
    dynamic = torch.normal(mean=dynamic_params[0], std=dynamic_params[1], size=(n,), device=device)
    dynamic = dynamic.clamp_min(0.0)

    dynamic = torch.minimum(dynamic, static)
    viscous = torch.full((n,), float(viscous_value), device=device)

    asset.write_joint_friction_coefficient_to_sim(
        static.unsqueeze(-1),
        joint_dynamic_friction_coeff=dynamic.unsqueeze(-1),
        joint_viscous_friction_coeff=viscous.unsqueeze(-1),
        joint_ids=[slider_idx],
        env_ids=env_ids,
    )


def randomize_pole_vel_noise_std(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mean: float = 0.4,
    std: float = 0.122,
    clip: tuple[float, float] = (0.05, 0.8),
) -> None:
    device = env.device
    if getattr(env, "pole_vel_noise_std", None) is None:
        env.pole_vel_noise_std = torch.full((env.num_envs, 1), float(mean), device=device)
    n = len(env_ids)
    samp = torch.normal(mean=mean, std=std, size=(n, 1), device=device).clamp_(clip[0], clip[1])
    env.pole_vel_noise_std[env_ids] = samp


def reset_joints_by_offset_unclamped(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
) -> None:
    asset: Articulation = env.scene[asset_cfg.name]
    ids, _ = asset.find_joints(asset_cfg.joint_names)
    pos = asset.data.default_joint_pos[env_ids][:, ids].clone()
    vel = asset.data.default_joint_vel[env_ids][:, ids].clone()
    pos += torch.empty_like(pos).uniform_(position_range[0], position_range[1])
    vel += torch.empty_like(vel).uniform_(velocity_range[0], velocity_range[1])
    asset.write_joint_state_to_sim(pos, vel, joint_ids=ids, env_ids=env_ids)
