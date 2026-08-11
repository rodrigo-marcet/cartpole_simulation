# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared config for the double cart-pole task family.

Same rig / observations / actions / domain randomization / sim settings live here; each task
(currently swingup_env_cfg.py) only defines its own reset, rewards and terminations. Underactuated:
only the slider is driven; both revolute joints (cart_to_ipole, ipole_to_opole) are passive.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp


##
# Scene definition
##
@configclass
class DoubleCartpoleSceneCfg(InteractiveSceneCfg):
    """Configuration for a double cart-pole scene."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    robot: ArticulationCfg = mdp.DOUBLE_PENDULUM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Only the slider is actuated (same cart + motor as the single cartpole)."""

    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=30.0)


@configclass
class ObservationsCfg:
    """Cart + BOTH poles, 8-dim; same noise/sensor model as the single cartpole.

    Declaration order IS the vector layout, and it matches double_pendulum_policy() in
    ../cartpole/pendulum/src/states/running/double_pendulum.cpp exactly as it already stands:

        0 x            4 w1
        1 xdot         5 sin(th2)  RELATIVE
        2 sin(th1)     6 cos(th2)  RELATIVE
        3 cos(th1)     7 w2        RELATIVE

    This mirrors Lee et al.'s state [y, th1, th2, ydot, th1dot, th2dot] -- relative th2, no previous
    action. The one deliberate difference is the sin/cos lifting of the angles instead of raw angle:
    the AS5600 wraps at 0/2pi, so a raw angle has a discontinuity the network would have to learn
    around.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        cart_pos = ObsTerm(
            func=mdp.cart_pos_noisy,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "ticks": 16384},
        )
        cart_vel = ObsTerm(
            func=mdp.cart_vel_noisy,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "ticks": 16384},
        )
        # inner pole (cart_to_ipole) -- already absolute, measured off the cart
        ipole_sin = ObsTerm(
            func=mdp.pole_angle_sin_quantized,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_ipole"]), "ticks_per_rev": 4096},
        )
        ipole_cos = ObsTerm(
            func=mdp.pole_angle_cos_quantized,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_ipole"]), "ticks_per_rev": 4096},
        )
        ipole_vel = ObsTerm(
            func=mdp.pole_angular_vel_noisy,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_ipole"])},
        )
        # outer pole -- RELATIVE angle, same sensor/noise as the inner joint
        opole_sin = ObsTerm(
            func=mdp.pole_angle_sin_quantized,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["ipole_to_opole"]), "ticks_per_rev": 4096},
        )
        opole_cos = ObsTerm(
            func=mdp.pole_angle_cos_quantized,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["ipole_to_opole"]), "ticks_per_rev": 4096},
        )
        opole_vel = ObsTerm(
            func=mdp.pole_angular_vel_noisy,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["ipole_to_opole"])},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class DoubleCartpoleEventCfg:
    """Shared reset + domain-randomization events (per-task resets live in the task cfg).

    Masses are centered on the USDA-authored values (see double_pendulum_physics.usda) so the mass DR
    does not fight recompute_inertia. Spreads are starting placeholders -- retune for the double rig.
    The friction-effort term MUST stay before the slider-armature term (see single_cartpole).

    DR IS CURRENTLY OFF: every spread is set to 0 so all envs are the nominal calibrated rig, and
    the original values sit commented directly above each one. Lee et al. also train without DR.
    While debugging in sim, DR only adds variance. Restore the commented lines before any
    transfer run. Observation noise is deliberately still ON at its 0.4 rad/s mean -- only the
    per-episode randomization of that std is disabled.
    """

    randomize_cart_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["cart_1"]),
            "mass_distribution_params": (0.20, 0.01),  # DR spread -- restore to re-enable
            # "mass_distribution_params": (0.20, 0.0),
            "operation": "abs",
            "distribution": "gaussian",
        },
    )
    randomize_ishaft_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["ishaft_1"]),
            "mass_distribution_params": (0.080, 0.004),
            # "mass_distribution_params": (0.080, 0.0),
            "operation": "abs",
            "distribution": "gaussian",
        },
    )
    randomize_ipole_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["ipole_1"]),
            "mass_distribution_params": (0.060, 0.003),
            # "mass_distribution_params": (0.060, 0.0),
            "operation": "abs",
            "distribution": "gaussian",
        },
    )
    randomize_oshaft_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["oshaft_1"]),
            "mass_distribution_params": (0.060, 0.003),
            # "mass_distribution_params": (0.060, 0.0),
            "operation": "abs",
            "distribution": "gaussian",
        },
    )
    randomize_opole_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["opole_1"]),
            "mass_distribution_params": (0.020, 0.001),
            # "mass_distribution_params": (0.020, 0.0),
            "operation": "abs",
            "distribution": "gaussian",
        },
    )
    randomize_weight_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["weight_1"]),
            "mass_distribution_params": (0.050, 0.0025),
            # "mass_distribution_params": (0.050, 0.0),
            # "mass_distribution_params": (0.0, 0.0),
            "operation": "abs",
            "distribution": "gaussian",
        },
    )

    randomize_slider_friction = EventTerm(
        func=mdp.randomize_slider_friction_effort,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "static_range": (3.0, 3.5),  # measured double breakaway 0.0324 N*m (= 3.24 N); ~+/-8%
            "dynamic_params": (2.5, 0.27),  # motor-curve calibrated Coulomb; ~11% spread
            # "static_range": (3.25, 3.25),       # midpoint of the measured range
            # "dynamic_params": (2.5, 0.0),
            # "viscous_value": 0.48,
            "viscous_value": 0.2,  # calibrated to the motor-curve match (with armature 0.3)
        },
    )
    randomize_slider_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "armature_distribution_params": (0.37, 0.05),  # calibrated to the motor-curve match
            # "armature_distribution_params": (0.3, 0.05),  # calibrated to the motor-curve match
            # "armature_distribution_params": (0.3, 0.0),
            "operation": "abs",
            "distribution": "gaussian",
        },
    )

    # Per-joint bearings: the inner (cart_to_ipole) has the slip ring -> much higher friction/damping than
    # the outer (ipole_to_opole). Values from the free-swing calibration; ~25% gaussian spread.
    randomize_inner_pole_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_ipole"]),
            "friction_distribution_params": (1.0e-4, 2.5e-5),
            # "friction_distribution_params": (1.0e-4, 0.0),
            "operation": "abs",
            "distribution": "gaussian",
        },
    )
    randomize_outer_pole_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["ipole_to_opole"]),
            "friction_distribution_params": (1.5e-5, 3.9e-6),
            # "friction_distribution_params": (1.5e-5, 0.0),
            "operation": "abs",
            "distribution": "gaussian",
        },
    )
    randomize_inner_pole_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_ipole"]),
            "damping_distribution_params": (1.5e-4, 3.75e-5),
            # "damping_distribution_params": (1.5e-4, 0.0),
            "operation": "abs",
            "distribution": "gaussian",
        },
    )
    randomize_outer_pole_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["ipole_to_opole"]),
            "damping_distribution_params": (1.1e-5, 2.75e-6),
            # "damping_distribution_params": (1.1e-5, 0.0),
            "operation": "abs",
            "distribution": "gaussian",
        },
    )

    randomize_pole_vel_noise = EventTerm(
        func=mdp.randomize_pole_vel_noise_std,
        mode="reset",
        params={"mean": 0.4, "std": 0.122, "clip": (0.05, 0.8)},
        # params={"mean": 0.0, "std": 0.0, "clip": (0.05, 0.8)},
    )


##
# Base environment configuration
##
@configclass
class DoubleCartpoleEnvCfg(ManagerBasedRLEnvCfg):
    """Shared base env cfg. Subclasses set `events`, `rewards`, `terminations`."""

    scene: DoubleCartpoleSceneCfg = DoubleCartpoleSceneCfg(num_envs=4096, env_spacing=1.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self) -> None:
        self.decimation = 10
        self.episode_length_s = 5  # balancing transfer
        # self.episode_length_s = 10    # swingup
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 1 / 1000
        self.sim.render_interval = self.decimation
