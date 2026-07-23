# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Double cart-pole swing-up + balance: bring both links from hanging to fully upright and hold.

Everything shared lives in double_cartpole_env_cfg.py; this module only defines the swing-up deltas
(reset, reward, terminations). The reward drives both joint angles to 0 (fully inverted).
"""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from .double_cartpole_env_cfg import DoubleCartpoleEnvCfg, DoubleCartpoleEventCfg


@configclass
class SwingupEventCfg(DoubleCartpoleEventCfg):
    """Shared events + cart/pole resets."""

    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-0.3, 0.3),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # Reverse-curriculum reset on BOTH revolute joints together: a fraction of envs start near-upright
    # (both angles ~0 -> the policy practices balancing), the rest are full-circle random (rich swing-up
    # starts). One coupled term so 'both up' happens together, not independently.
    # A/B (classic hanging-rest start): replace with two reset_joints_by_offset terms --
    #   cart_to_ipole position_range=(pi-0.3, pi+0.3), ipole_to_opole position_range=(-0.3, 0.3).
    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset_mixed,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_ipole", "ipole_to_opole"]),
            "near_fraction": 0.2,  # 20% start near fully-upright
            "near_pos_range": (-0.2, 0.2),  # ~ +/-11 deg from upright, both joints
            "near_vel_range": (-1.0, 1.0),
            "far_pos_range": (0.0, 2 * math.pi),  # full-circle random starts
            "far_vel_range": (-10.0, 10.0),
        },
    )


@configclass
class SwingupRewardsCfg:
    """Dense LQR-like quadratic cost -> single-net swing-up + balance of the double pendulum."""

    # (1) constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) failure penalty: leaving the track is strictly bad
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) primary task: both links upright, cart centered, low effort
    swingup = RewTerm(
        func=mdp.double_swingup_reward_quadratic,
        weight=4.0,
        params={
            "ipole_cfg": SceneEntityCfg("robot", joint_names=["cart_to_ipole"]),
            "opole_cfg": SceneEntityCfg("robot", joint_names=["ipole_to_opole"]),
            "cart_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
        },
    )
    # (4) penalize large effort commands directly
    effort_penalty = RewTerm(
        func=mdp.joint_effort_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )
    # (5) penalize rapid changes between consecutive actions (jerk)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


@configclass
class SwingupTerminationsCfg:
    """Time-out + cart off the (safety-margined) track. No pole-angle limit (swing-up)."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-0.35, 0.35)},
    )


@configclass
class SwingupEnvCfg(DoubleCartpoleEnvCfg):
    events: SwingupEventCfg = SwingupEventCfg()
    rewards: SwingupRewardsCfg = SwingupRewardsCfg()
    terminations: SwingupTerminationsCfg = SwingupTerminationsCfg()
