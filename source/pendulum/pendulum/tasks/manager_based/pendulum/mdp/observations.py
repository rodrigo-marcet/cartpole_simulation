# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .. import mdp

if TYPE_CHECKING:
    pass


def add_encoder_tick_noise(value: torch.Tensor, resolution: float, max_ticks: int = 3) -> torch.Tensor:
    """Add discrete tick noise with center-peaked distribution."""
    # weights: [1, 2, 3, 4, 3, 2, 1] for max_ticks=3 (triangular)
    ticks = torch.arange(-max_ticks, max_ticks + 1)  # [-3, -2, -1, 0, 1, 2, 3]
    weights = (max_ticks + 1 - ticks.abs()).float()  # triangular weights
    probs = weights / weights.sum()

    indices = (
        torch.multinomial(probs.unsqueeze(0).expand(value.numel(), -1), num_samples=1).squeeze(-1).reshape(value.shape)
    )

    offsets = ticks[indices].to(value.device).float() * resolution
    return value + offsets


def pole_angle_sin(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_ids=[1])) -> torch.Tensor:
    angle = mdp.joint_pos_rel(env, asset_cfg)
    # print("angle shape:", angle.shape)
    # result = torch.sin(angle)
    # print("sin shape:", result.shape)
    # return result
    return torch.sin(angle)


def pole_angle_cos(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_ids=[1])) -> torch.Tensor:
    angle = mdp.joint_pos_rel(env, asset_cfg)
    return torch.cos(angle)


def pole_angle_sin_quantized(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_ids=[1]), ticks_per_rev: int = 4096
) -> torch.Tensor:
    angle = mdp.joint_pos_rel(env, asset_cfg)
    resolution = (2 * math.pi) / ticks_per_rev
    angle_q = torch.round(angle / resolution) * resolution
    return torch.sin(angle_q)


def pole_angle_cos_quantized(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_ids=[1]), ticks_per_rev: int = 4096
) -> torch.Tensor:
    angle = mdp.joint_pos_rel(env, asset_cfg)
    resolution = (2 * math.pi) / ticks_per_rev
    angle_q = torch.round(angle / resolution) * resolution
    return torch.cos(angle_q)


def cart_pos_quantized(env, asset_cfg: SceneEntityCfg, ticks: int = 16384, range_m: float = 0.8) -> torch.Tensor:
    """Quantize cart position to 14-bit encoder resolution."""
    asset = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos[:, asset_cfg.joint_ids[0]]
    resolution = range_m / ticks
    return (torch.round(pos / resolution) * resolution).unsqueeze(-1)


def cart_pos_noisy(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
    ticks: int = 16384,
    pulley_radius_m: float = 0.01,
) -> torch.Tensor:
    pos = mdp.joint_pos_rel(env, asset_cfg)
    tick_size_m = 2.0 * math.pi * pulley_radius_m / ticks  # ≈ 3.83e-6 m
    return add_encoder_tick_noise(pos, tick_size_m, max_ticks=3)


def cart_vel_noisy(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
    ticks: int = 16384,
    pulley_radius_m: float = 0.01,
    encoder_bandwidth_hz: float = 1000.0,
    max_pos_ticks: int = 3,
) -> torch.Tensor:
    vel = mdp.joint_vel_rel(env, asset_cfg)

    # Correct resolution: one tick = one full motor revolution / CPR, scaled by pulley circumference
    tick_size_m = 2.0 * math.pi * pulley_radius_m / ticks  # ≈ 3.83e-6 m

    # ODrive PLL: kp = 2 * bandwidth, critically damped
    # Vel noise ≈ pll_kp * position_noise
    pll_kp = 2.0 * encoder_bandwidth_hz
    vel_noise_per_tick = pll_kp * tick_size_m  # ≈ 0.00766 m/s per tick

    return add_encoder_tick_noise(vel, vel_noise_per_tick, max_ticks=max_pos_ticks)


def pole_angular_vel_noisy(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_ids=[1]), ticks_per_rev: int = 4096
) -> torch.Tensor:
    vel = mdp.joint_vel_rel(env, asset_cfg)
    resolution = (2 * math.pi) / ticks_per_rev
    return add_encoder_tick_noise(vel, resolution)
