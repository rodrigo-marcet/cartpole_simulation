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
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.5, 0.5),
        },
    )

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (math.pi - 0.5, math.pi + 0.5),
            "velocity_range": (-1.0, 1.0),
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
        func=mdp.randomize_slider_friction_effort,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "static_range": (1.7, 2.0),
            "dynamic_params": (1.41, 0.15),
            "viscous_value": 0.48,
        },
    )

    randomize_slider_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "armature_distribution_params": (0.37, 0.05),
            "operation": "abs",
            "distribution": "gaussian",
        },
    )

    randomize_pole_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "friction_distribution_params": (1.5e-5, 3.9e-6),
            "operation": "abs",
            "distribution": "gaussian",
        },
    )

    randomize_pole_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "damping_distribution_params": (7.5e-6, 2.25e-06),
            "operation": "abs",
            "distribution": "gaussian",
        },
    )

    randomize_pole_vel_noise = EventTerm(
        func=mdp.randomize_pole_vel_noise_std,
        mode="reset",
        params={"mean": 0.4, "std": 0.122, "clip": (0.05, 0.8)},
    )


@configclass
class RewardsCfg:
    # (1) Shaping tasks: penalize large effort commands directly
    alive = RewTerm(func=mdp.is_alive, weight=1.0)

    # (2) Shaping tasks: penalize large effort commands directly
    swingup = RewTerm(
        func=mdp.swingup_reward_energy_pump,
        weight=4.0,
        params={
            "pole_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "cart_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
        },
    )
    # (3) Shaping tasks: penalize large effort commands directly
    effort_penalty = RewTerm(
        func=mdp.joint_effort_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )
    # (4) Shaping tasks: penalize rapid changes between consecutive actions (jerk)
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-0.3, 0.3)},
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
        self.episode_length_s = 5
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 1 / 1000
        self.sim.render_interval = self.decimation
