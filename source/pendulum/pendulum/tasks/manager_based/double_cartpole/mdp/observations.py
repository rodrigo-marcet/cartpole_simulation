from __future__ import annotations

import math

import torch

from isaaclab.envs.mdp import joint_pos_rel
from isaaclab.managers import SceneEntityCfg

from .rewards import L1_DEFAULT, L2_DEFAULT

__all__ = ["opole_angle_sin_abs_quantized", "opole_angle_cos_abs_quantized", "tip_offset_x", "tip_offset_y"]


def opole_angle_sin_abs_quantized(
    env, ipole_cfg: SceneEntityCfg, opole_cfg: SceneEntityCfg, ticks_per_rev: int = 4096
) -> torch.Tensor:
    return torch.sin(_abs_outer_angle(env, ipole_cfg, opole_cfg, ticks_per_rev))


def opole_angle_cos_abs_quantized(
    env, ipole_cfg: SceneEntityCfg, opole_cfg: SceneEntityCfg, ticks_per_rev: int = 4096
) -> torch.Tensor:
    return torch.cos(_abs_outer_angle(env, ipole_cfg, opole_cfg, ticks_per_rev))


def _quantized_angles(
    env, ipole_cfg: SceneEntityCfg, opole_cfg: SceneEntityCfg, ticks_per_rev: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """(th1, th1 + th2), each encoder quantized independently the way the two AS5600s are."""
    resolution = (2 * math.pi) / ticks_per_rev
    th1 = torch.round(joint_pos_rel(env, ipole_cfg) / resolution) * resolution
    th2 = torch.round(joint_pos_rel(env, opole_cfg) / resolution) * resolution
    return th1, th1 + th2


def _abs_outer_angle(env, ipole_cfg: SceneEntityCfg, opole_cfg: SceneEntityCfg, ticks_per_rev: int) -> torch.Tensor:
    """th1 + th2, each quantized independently the way the two AS5600s are."""
    return _quantized_angles(env, ipole_cfg, opole_cfg, ticks_per_rev)[1]


def tip_offset_x(
    env,
    ipole_cfg: SceneEntityCfg,
    opole_cfg: SceneEntityCfg,
    l1: float = L1_DEFAULT,
    l2: float = L2_DEFAULT,
    ticks_per_rev: int = 4096,
) -> torch.Tensor:
    th1, th2_abs = _quantized_angles(env, ipole_cfg, opole_cfg, ticks_per_rev)
    return l1 * torch.sin(th1) + l2 * torch.sin(th2_abs)


def tip_offset_y(
    env,
    ipole_cfg: SceneEntityCfg,
    opole_cfg: SceneEntityCfg,
    l1: float = L1_DEFAULT,
    l2: float = L2_DEFAULT,
    ticks_per_rev: int = 4096,
) -> torch.Tensor:
    th1, th2_abs = _quantized_angles(env, ipole_cfg, opole_cfg, ticks_per_rev)
    return l1 * torch.cos(th1) + l2 * torch.cos(th2_abs)
