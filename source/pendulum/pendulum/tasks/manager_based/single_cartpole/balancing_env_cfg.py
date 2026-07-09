# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Balancing task: pole starts near-upright, reward keeps it upright and the cart centered.

Everything shared lives in cartpole_env_cfg.py; this module only defines the balancing deltas,
including bumping the action scale to 40 (vs the base/swing-up 30).
"""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from .cartpole_env_cfg import CartpoleEnvCfg, CartpoleEventCfg


@configclass
class BalancingEventCfg(CartpoleEventCfg):
    """Shared events + pole starts NEAR-UPRIGHT (theta ~ 0, within +/-45 deg)."""

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
        },
    )


@configclass
class BalancingRewardsCfg:
    """Upright-balance shaping: keep the pole up, the cart centered, motions small."""

    # (1) constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) primary task: keep pole upright
    pole_pos = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    )
    # (4) lower cart velocity
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )
    # (5) lower pole angular velocity
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    )
    # (6) keep the cart near the middle
    cart_pos = RewTerm(
        func=mdp.joint_pos_target_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "target": 0.0},
    )
    # (7) penalize large effort commands directly
    effort_penalty = RewTerm(
        func=mdp.joint_effort_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )
    # (8) penalize rapid changes between consecutive actions (jerk)
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )


@configclass
class BalancingTerminationsCfg:
    """Time-out + cart off the track + pole falls past +/-90 deg."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-0.4, 0.4)},
    )
    pole_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "bounds": (-(math.pi / 2.0), (math.pi / 2.0)),
        },
    )


@configclass
class BalancingEnvCfg(CartpoleEnvCfg):
    events: BalancingEventCfg = BalancingEventCfg()
    rewards: BalancingRewardsCfg = BalancingRewardsCfg()
    terminations: BalancingTerminationsCfg = BalancingTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # balancing drives harder than swing-up
        self.actions.joint_effort.scale = 40.0
