# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Double cart-pole swing-up + balance: bring both links from hanging to fully upright and hold.

One task, one policy, replicating Lee, Ju & Lee (Machines 2025, 13, 186) -- the only published
end-to-end RL swing-up of a CART double pendulum on real hardware, and notably at a 100 Hz policy
rate with no LQR handoff. Multiplicative [0,1] reward, positive, with a rail termination.
Alternatives considered: scripts/double_swingup_policies.md.
Everything shared (scene, actions, observations, DR) lives in double_cartpole_env_cfg.py.

For a BALANCE-ONLY run: set near_fraction=1.0 / mid_fraction=0.0 in `reset_poles` and add
    pole_fell = DoneTerm(func=mdp.tip_below_height,
                         params={"ipole_cfg": IPOLE, "opole_cfg": OPOLE, "min_height_frac": 0.5})
to SwingupTerminationsCfg. Both together, never one without the other.
"""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from .double_cartpole_env_cfg import DoubleCartpoleEnvCfg, DoubleCartpoleEventCfg

IPOLE = SceneEntityCfg("robot", joint_names=["cart_to_ipole"])
OPOLE = SceneEntityCfg("robot", joint_names=["ipole_to_opole"])
CART = SceneEntityCfg("robot", joint_names=["slider_to_cart"])


@configclass
class SwingupEventCfg(DoubleCartpoleEventCfg):
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"asset_cfg": CART, "position_range": (-0.1, 0.1), "velocity_range": (-1.0, 1.0)},
    )
    reset_poles = EventTerm(
        func=mdp.reset_double_poles_graded,
        mode="reset",
        params={
            "ipole_cfg": IPOLE,
            "opole_cfg": OPOLE,
            "near_fraction": 1.0,  # balancing
            "mid_fraction": 0.0,  # balancing
            # "near_fraction": 0.5,     #initial swingup
            # "mid_fraction": 0.25,     #initial swingup
            # "near_fraction": 0.2,       #final swingup
            # "mid_fraction": 0.2,        #final swingup
            "near_pos_range": (-0.15, 0.15),
            "near_vel_range": (-0.5, 0.5),
            "mid_ipos_range": (0.5 * math.pi, 1.5 * math.pi),
            "mid_opos_range": (-0.5, 0.5),
            "mid_vel_range": (-3.0, 3.0),
            "far_ipos_mean": math.pi,
            "far_ipos_std": 0.05,
            "far_opos_std": 0.10,
            "far_vel_std": 0.05,
        },
    )
    # reset_poles = EventTerm(
    #     func=mdp.reset_double_poles_uniform,
    #     mode="reset",
    #     params={
    #         "ipole_cfg": IPOLE,
    #         "opole_cfg": OPOLE,
    #         "near_fraction": 0.2,
    #         "hang_fraction": 0.2,
    #         "near_pos_range": (-0.15, 0.15),
    #         "near_vel_range": (-0.5, 0.5),
    #         "hang_ipos_std": 0.05,
    #         "hang_opos_std": 0.10,
    #         "hang_vel_std": 0.05,
    #         "ipos_range": (-math.pi, math.pi),
    #         "opos_range": (-math.pi, math.pi),
    #         "ivel_range": (-10.0, 10.0),
    #         "ovel_range": (-20.0, 20.0),
    #     },
    # )

    def __post_init__(self) -> None:
        # reset_poles owns both revolute joints; drop the base hard-pi inner reset
        self.reset_ipole_position = None


@configclass
class SwingupRewardsCfg:
    """Single self-contained term, the Lee et al. product of six [0,1] factors.

    Constants deliberately left to the function defaults in mdp/rewards.py so there is one source of
    truth -- k_effort in particular is derived from their exp(-0.015|u|) with u in m/s^2, not guessed.
    """

    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)

    swingup = RewTerm(
        func=mdp.multiplicative_swingup_reward,
        weight=1.0,
        params={"ipole_cfg": IPOLE, "opole_cfg": OPOLE, "cart_cfg": CART},
    )


@configclass
class SwingupTerminationsCfg:
    """Time-out + cart off the (safety-margined) track. No pole-angle limit -- this is the swing-up.

    Adding a fall termination here would make the goal unreachable from hanging, and would also prune
    the pumping motion (a swing-up has to dip lower before it can come up). See the docstring on
    mdp.reset_double_poles_graded.

    The rail termination is correct BECAUSE the reward is positive: ending early forfeits the
    remaining stream, so leaving the track is a real loss. Lee et al. do the same (|y| > 0.4 m).
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": CART, "bounds": (-0.35, 0.35)},
    )

    # Only for balancing task
    pole_fell = DoneTerm(
        func=mdp.tip_below_height,
        params={"ipole_cfg": IPOLE, "opole_cfg": OPOLE, "min_height_frac": 0.5},
    )


@configclass
class SwingupEnvCfg(DoubleCartpoleEnvCfg):
    events: SwingupEventCfg = SwingupEventCfg()
    rewards: SwingupRewardsCfg = SwingupRewardsCfg()
    terminations: SwingupTerminationsCfg = SwingupTerminationsCfg()
