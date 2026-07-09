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


def pole_angle_sin_noisy(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_ids=[1]), ticks_per_rev: int = 4096
) -> torch.Tensor:
    angle = mdp.joint_pos_rel(env, asset_cfg)
    resolution = (2 * math.pi) / ticks_per_rev
    angle_q = torch.round(angle / resolution) * resolution
    angle_q = add_encoder_tick_noise(angle_q, resolution)
    return torch.sin(angle_q)


def pole_angle_cos_noisy(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_ids=[1]), ticks_per_rev: int = 4096
) -> torch.Tensor:
    angle = mdp.joint_pos_rel(env, asset_cfg)
    resolution = (2 * math.pi) / ticks_per_rev
    angle_q = torch.round(angle / resolution) * resolution
    angle_q = add_encoder_tick_noise(angle_q, resolution)
    return torch.cos(angle_q)


def pole_angular_vel_noisy(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_ids=[1]),
    ticks_per_rev: int = 4096,  # kept for call-site compatibility; unused (see below)
    # noise_std: float = 0.28,  # rad/s -- measured from the rig; DR-able later
    noise_std: float = 0.4,  # rad/s -- measured from the rig; DR-able later
) -> torch.Tensor:
    """Pole angular velocity with REALISTIC additive Gaussian sensor noise.

    The pole joint is passive (no motor / no ODrive PLL): the rig derives its angular
    velocity by software finite-difference of the 12-bit absolute encoder at the ~100 Hz
    control rate. That differentiation amplifies the position tick by ~1/dt, and the result
    is dominated by broadband mechanical/electrical noise -- NOT clean quantization ticks.

    Measured from the rig free-swing (analysis/data/pole_analysis/ground_truth/001.csv,
    12 Hz high-pass): noise std ~= 0.28 rad/s, essentially FLAT with speed (only +12% at
    18 rad/s), and ~4x larger than the pure encoder-quantization-differencing prediction.
    So the right model is additive Gaussian(0, ~0.28), not the discrete tick model.

    The OLD model added ``add_encoder_tick_noise`` at the raw ANGLE resolution
    (2*pi/4096 -> ~0.0024 rad/s std) -- about 115x too small. It forgot the
    finite-difference amplification (1/dt) that ``cart_vel_noisy`` DOES apply via its
    ``pll_kp`` factor. Training on a near-perfect pole velocity is a prime sim2real gap;
    even for balancing, the controller keys off the pole angular velocity.
    """
    vel = mdp.joint_vel_rel(env, asset_cfg)
    # Per-episode DR: randomize_pole_vel_noise_std sets env.pole_vel_noise_std (num_envs, 1).
    # Use it when present; otherwise fall back to the scalar noise_std (bare probes / no events).
    std = getattr(env, "pole_vel_noise_std", None)
    if std is None:
        std = noise_std
    return vel + torch.randn_like(vel) * std
