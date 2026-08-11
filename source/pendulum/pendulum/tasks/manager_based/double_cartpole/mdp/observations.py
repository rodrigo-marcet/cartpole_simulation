# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Absolute outer-link orientation.

The outer AS5600 measures th2 RELATIVE to the inner link, so the absolute orientation th1 + th2 is a
derived quantity. Computing it here rather than letting the network do it matters because
sin(th1+th2) = sin(th1)cos(th2) + cos(th1)sin(th2) is a PRODUCT of inputs, which an ELU MLP has to
spend capacity approximating. Stacking the two encoders' quantization costs 0.016% of tip-height
range -- negligible. Velocities are deliberately NOT stacked (see ObservationsCfg).
"""

from __future__ import annotations

import math

import torch

from isaaclab.envs.mdp import joint_pos_rel
from isaaclab.managers import SceneEntityCfg

__all__ = ["opole_angle_sin_abs_quantized", "opole_angle_cos_abs_quantized"]


def _abs_outer_angle(env, ipole_cfg: SceneEntityCfg, opole_cfg: SceneEntityCfg, ticks_per_rev: int) -> torch.Tensor:
    """th1 + th2, each quantized independently the way the two AS5600s are."""
    resolution = (2 * math.pi) / ticks_per_rev
    th1 = joint_pos_rel(env, ipole_cfg)
    th2 = joint_pos_rel(env, opole_cfg)
    return torch.round(th1 / resolution) * resolution + torch.round(th2 / resolution) * resolution


def opole_angle_sin_abs_quantized(
    env, ipole_cfg: SceneEntityCfg, opole_cfg: SceneEntityCfg, ticks_per_rev: int = 4096
) -> torch.Tensor:
    return torch.sin(_abs_outer_angle(env, ipole_cfg, opole_cfg, ticks_per_rev))


def opole_angle_cos_abs_quantized(
    env, ipole_cfg: SceneEntityCfg, opole_cfg: SceneEntityCfg, ticks_per_rev: int = 4096
) -> torch.Tensor:
    return torch.cos(_abs_outer_angle(env, ipole_cfg, opole_cfg, ticks_per_rev))
