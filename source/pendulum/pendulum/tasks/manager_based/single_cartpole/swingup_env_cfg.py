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
            "position_range": (-0.3, 0.3),
            "velocity_range": (-1.0, 1.0),
        },
    )
    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset_mixed,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "near_fraction": 0.3,
            "near_pos_range": (-0.2, 0.2),
            "near_vel_range": (-1.0, 1.0),
            "far_pos_range": (0.0, 2 * math.pi),
            "far_vel_range": (-25.0, 25.0),
        },
    )


@configclass
class SwingupRewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    swingup = RewTerm(
        func=mdp.swingup_reward_energy_pump,
        weight=4.0,
        params={
            "pole_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "cart_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
        },
    )
    effort_penalty = RewTerm(
        func=mdp.joint_effort_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )


@configclass
class SwingupTerminationsCfg:
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
