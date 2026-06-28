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


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        cart_pos = ObsTerm(
            func=mdp.cart_pos_noisy,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
                "ticks": 16384,
            },
        )
        cart_vel = ObsTerm(
            func=mdp.cart_vel_noisy,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
                "ticks": 16384,
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
            "position_range": (-0.0, 0.0),
            "velocity_range": (-0.0, 0.0),
            # "position_range": (-0.1, 0.1),
            # "velocity_range": (-1.0, 1.0),
        },
    )

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            # "position_range": (-2 * math.pi, 2 * math.pi),
            # "velocity_range": (-2 * math.pi * 15.0, 2 * math.pi * 15.0),
            "position_range": (math.pi - 0.1, math.pi + 0.1),
            "velocity_range": (-0.0, 0.0),
        },
    )

    randomize_cart_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["cart_1"]),
            "mass_distribution_params": (0.20, 0.01),
            "operation": "abs",
            "distribution": "gaussian",
        },
    )
    randomize_shaft_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["shaft_1"]),
            "mass_distribution_params": (0.080, 0.004),
            "operation": "abs",
            "distribution": "gaussian",
        },
    )
    randomize_pole_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["pendulum_1"]),
            "mass_distribution_params": (0.020, 0.001),
            "operation": "abs",
            "distribution": "gaussian",
        },
    )
    randomize_weight_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["weight_1"]),
            "mass_distribution_params": (0.030, 0.0015),
            "operation": "abs",
            "distribution": "gaussian",
        },
    )

    randomize_slider_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "friction_distribution_params": (0.4, 0.05),
            "operation": "abs",
            "distribution": "gaussian",
        },
    )

    randomize_pole_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            # "friction_distribution_params": (0.00005, 0.00002),
            "friction_distribution_params": (0.00007, 0.000018),
            "operation": "abs",
            "distribution": "gaussian",
        },
    )

    randomize_pole_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "damping_distribution_params": (0.00001, 3.04e-06),
            "operation": "abs",
            "distribution": "gaussian",
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
    # (6) Shaping tasks: keep cart near the middle
    cart_pos = RewTerm(
        # func=mdp.joint_pos_target_l2, # Best so far
        # weight=-0.1,
        func=mdp.joint_pos_target_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "target": 0.0},
    )
    # (7) Shaping tasks: penalize large effort commands directly
    effort_penalty = RewTerm(
        func=mdp.joint_effort_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )
    # (8) Shaping tasks: penalize rapid changes between consecutive actions (jerk)
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        # weight=-0.5, # Best so far
        # weight=-0.05, # Best long so far
        weight=-0.01,
    )


# @configclass
# class RewardsCfg:
#     alive = RewTerm(func=mdp.is_alive, weight=1.0)
#     swingup = RewTerm(
#         func=mdp.swingup_reward_energy_pump,
#         # func=mdp.swingup_reward_gated,
#         weight=4.0,
#         params={
#             "pole_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
#             "cart_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
#         },
#     )
#     # (7) Shaping tasks: penalize large effort commands directly
#     effort_penalty = RewTerm(
#         func=mdp.joint_effort_l2,
#         # weight=-0.05,
#         weight=-0.5,
#         params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
#     )
#     # (8) Shaping tasks: penalize rapid changes between consecutive actions (jerk)
#     action_rate = RewTerm(
#         func=mdp.action_rate_l2,
#         # weight=-0.5, # Best so far
#         weight=-0.05, # Best long so far
#         # weight=-0.01,
#     )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-0.2, 0.2)},
    )


# @configclass
# class CurriculumsCfg:
#     """Curriculum terms for the MDP."""

#     narrow_cart_bounds = CurriculumTermCfg(
#         func=mdp.NarrowCartBoundsCurriculum,
#         params={
#             "term_name": "cart_out_of_bounds",
#             "start_bound": 0.4,
#             "end_bound": 0.2,
#             "num_steps": 7_500,  # adjust to 75% of your total training steps
#         },
#     )


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
    # curriculum: CurriculumsCfg = CurriculumsCfg()  # <-- add this line

    def __post_init__(self) -> None:
        self.decimation = 10
        self.episode_length_s = 5
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 1 / 1000
        self.sim.render_interval = self.decimation
