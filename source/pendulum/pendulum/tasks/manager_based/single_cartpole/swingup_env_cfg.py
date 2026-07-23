# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Swing-up task: pole starts hanging, energy-pump reward brings it up for the PID to catch.

Everything shared lives in cartpole_env_cfg.py; this module only defines the swing-up deltas.
Action scale stays at the base 30 (the value the deployed policy was trained with).
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
class SwingupEventCfg(CartpoleEventCfg):
    """Shared events + pole starts HANGING (theta ~ pi) with a small velocity kick."""

    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            # "position_range": (-0.1, 0.1),
            # "velocity_range": (-0.5, 0.5),
            "position_range": (-0.3, 0.3),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # reset_pole_position = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
    #         # "position_range": (math.pi - 0.5, math.pi + 0.5),
    #         # "velocity_range": (-1.0, 1.0),
    #         "position_range": (0.0, 2 * math.pi),
    #         "velocity_range": (-25.0, 25.0),
    #     },
    # )

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset_mixed,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "near_fraction": 0.3,  # 30% start near upright
            "near_pos_range": (-0.2, 0.2),  # ~ +/-11 deg from upright
            "near_vel_range": (-1.0, 1.0),
            "far_pos_range": (0.0, 2 * math.pi),  # full-circle swing-up (previous behavior)
            "far_vel_range": (-25.0, 25.0),
        },
    )


@configclass
class SwingupRewardsCfg:
    """Energy-shaping swing-up reward (pumps the pole onto the homoclinic orbit for the PID handoff)."""

    # (1) constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)

    # (2) primary task: energy-pump swing-up
    swingup = RewTerm(
        func=mdp.swingup_reward_energy_pump,
        weight=4.0,
        params={
            "pole_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "cart_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
        },
    )
    # swingup = RewTerm(
    #     func=mdp.swingup_reward_unified,
    #     weight=4.0,
    #     params={
    #         "pole_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
    #         "cart_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
    #     },
    # )

    # (3) penalize large effort commands directly
    effort_penalty = RewTerm(
        func=mdp.joint_effort_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )
    # (4) penalize rapid changes between consecutive actions (jerk)
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )


@configclass
class SwingupTerminationsCfg:
    """Time-out + cart off the (safety-margined) track."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-0.35, 0.35)},
    )


@configclass
class SwingupEnvCfg(CartpoleEnvCfg):
    events: SwingupEventCfg = SwingupEventCfg()
    rewards: SwingupRewardsCfg = SwingupRewardsCfg()
    terminations: SwingupTerminationsCfg = SwingupTerminationsCfg()
