# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp

##
# Scene definition
##


@configclass
class PendulumSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot: ArticulationCfg = mdp.MY_CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot: ArticulationCfg = mdp.FUSION_CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=30.0)


# @configclass
# class ActionsCfg:
#     joint_effort = mdp.SmoothedJointEffortActionCfg(
#         asset_name="robot",
#         joint_names=["slider_to_cart"],
#         scale=40.0,
#         alpha=0.3,   # <-- tune this
#     )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # cart_vel = ObsTerm(
        #     func=mdp.joint_vel_rel,
        #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
        # )
        # pole_angular_vel = ObsTerm(
        #     func=mdp.joint_vel_rel,
        #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
        # )
        # pole_sin = ObsTerm(func=mdp.pole_angle_sin)
        # pole_cos = ObsTerm(func=mdp.pole_angle_cos)

        cart_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
        )
        cart_vel = ObsTerm(
            func=mdp.cart_vel_noisy,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
                "ticks": 16384,
                "range_m": 0.8,
            },
        )
        pole_sin = ObsTerm(func=mdp.pole_angle_sin_quantized, params={"ticks_per_rev": 4096})
        pole_cos = ObsTerm(func=mdp.pole_angle_cos_quantized, params={"ticks_per_rev": 4096})
        pole_angular_vel = ObsTerm(
            func=mdp.pole_angular_vel_noisy,
            params={"asset_cfg": SceneEntityCfg("robot", joint_ids=[1])},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.2, 0.2),
            # "position_range": (-1.0, 1.0),
            # "velocity_range": (-0.5, 0.5),
        },
    )

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            # "position_range": (-0.1 * math.pi, 0.1 * math.pi),
            # "velocity_range": (-0.1 * math.pi, 0.1 * math.pi),
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
        },
    )

    randomize_cart_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["cart_1"]),
            # "mass_distribution_params": (0.18, 0.22),
            "mass_distribution_params": (0.20, 0.01),  # cart: ±10% ≈ ±0.02, so std=0.01 keeps it tight
            "operation": "abs",  # or "scale" to multiply existing mass
        },
    )
    randomize_shaft_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["shaft_1"]),
            # "mass_distribution_params": (0.072, 0.088),
            "mass_distribution_params": (0.080, 0.004),  # shaft
            "operation": "abs",  # or "scale" to multiply existing mass
        },
    )
    randomize_pole_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["pendulum_1"]),
            # "mass_distribution_params": (0.018, 0.022),
            "mass_distribution_params": (0.020, 0.001),  # pole
            "operation": "abs",  # or "scale" to multiply existing mass
        },
    )
    randomize_weight_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["weight_1"]),
            # "mass_distribution_params": (0.027, 0.033),
            "mass_distribution_params": (0.030, 0.0015),  # weight
            "operation": "abs",  # or "scale" to multiply existing mass
        },
    )

    randomize_slider_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            # "friction_distribution_params": (0.47, 0.70),
            # "friction_distribution_params": (0.3, 0.5),
            "friction_distribution_params": (0.4, 0.05),  # mean=0.4, std=0.05 → ~95% of samples within (0.3, 0.5)
            "operation": "abs",
        },
    )

    randomize_pole_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            # "friction_distribution_params": (0.0001, 0.00001),
            "friction_distribution_params": (0.00005, 0.00002),  # tiny mean, tiny std
            "operation": "abs",
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: keep pole upright
    pole_pos = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    )
    # (4) Shaping tasks: lower cart velocity
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )
    # (5) Shaping tasks: lower pole angular velocity
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    )
    cart_pos = RewTerm(
        func=mdp.joint_pos_target_l1,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "target": 0.0},
    )

    # Penalize large effort commands directly
    effort_penalty = RewTerm(
        func=mdp.joint_effort_l2,  # or l1
        weight=-0.05,  # fixed_dr
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )

    # Penalize rapid changes between consecutive actions (jerk)
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.5,  # fixed_dr
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        # params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-0.4, 0.4)},
    )


##
# Environment configuration
##


@configclass
class PendulumEnvCfg(ManagerBasedRLEnvCfg):
    scene: PendulumSceneCfg = PendulumSceneCfg(num_envs=4096, env_spacing=1.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 10
        self.episode_length_s = 10
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 1 / 1000
        self.sim.render_interval = self.decimation
