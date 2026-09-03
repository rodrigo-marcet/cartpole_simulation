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
            # "near_fraction": 0.3,       #final swingup
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

    def __post_init__(self) -> None:
        self.reset_ipole_position = None


@configclass
class SwingupRewardsCfg:
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    swingup = RewTerm(
        func=mdp.double_swingup_reward_quadratic,
        weight=1.0,
        params={"ipole_cfg": IPOLE, "opole_cfg": OPOLE, "cart_cfg": CART},
    )


@configclass
class SwingupTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": CART, "bounds": (-0.35, 0.35)},
    )

    # # Only for balancing task
    # pole_fell = DoneTerm(
    #     func=mdp.tip_below_height,
    #     params={"ipole_cfg": IPOLE, "opole_cfg": OPOLE, "min_height_frac": 0.5},
    # )


@configclass
class SwingupEnvCfg(DoubleCartpoleEnvCfg):
    events: SwingupEventCfg = SwingupEventCfg()
    rewards: SwingupRewardsCfg = SwingupRewardsCfg()
    terminations: SwingupTerminationsCfg = SwingupTerminationsCfg()
