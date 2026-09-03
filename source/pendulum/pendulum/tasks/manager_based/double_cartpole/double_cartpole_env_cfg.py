import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp


def euler_deg_to_quat(roll_deg: float, pitch_deg: float, yaw_deg: float) -> tuple[float, float, float, float]:
    """Convert Euler angles in degrees (XYZ order) to a (w, x, y, z) quaternion."""
    roll, pitch, yaw = torch.deg2rad(torch.tensor([roll_deg, pitch_deg, yaw_deg]))
    quat = math_utils.quat_from_euler_xyz(roll.unsqueeze(0), pitch.unsqueeze(0), yaw.unsqueeze(0))
    return tuple(quat.squeeze(0).tolist())


##
# Scene definition
##
@configclass
class DoubleCartpoleSceneCfg(InteractiveSceneCfg):
    """Configuration for a double cart-pole scene."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.MeshCuboidCfg(
            size=(1000.0, 1000.0, 0.05),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path="E:/youtube/videos/8.Pendulum/media/materials/ground_material.mdl",
                project_uvw=True,
                texture_scale=(0.1, 0.1),
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        ),
    )

    robot: ArticulationCfg = mdp.DOUBLE_PENDULUM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            texture_file="E:/youtube/videos/8.Pendulum/media/materials/sky.hdr",
            texture_format="latlong",
        ),
    )

    front_light = AssetBaseCfg(
        prim_path="/World/FrontLight",
        spawn=sim_utils.DistantLightCfg(
            color=(2.0, 2.0, 2.0),
            intensity=3000.0,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.71488, 0.32137, 0.40919, -0.46715)),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Only the slider is actuated (same cart + motor as the single cartpole)."""

    # joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=30.0)
    joint_effort = mdp.LaggedJointEffortActionCfg(
        asset_name="robot",
        joint_names=["slider_to_cart"],
        scale=30.0,
        alpha=0.414,
    )


OBS_SCALE = (2.5, 0.4, 1.0, 1.0, 1.0 / 15.0, 1.0, 1.0, 1.0 / 15.0, 4.0, 4.0)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        cart_pos = ObsTerm(
            func=mdp.cart_pos_noisy,
            scale=OBS_SCALE[0],
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "ticks": 16384},
        )
        cart_vel = ObsTerm(
            func=mdp.cart_vel_noisy,
            scale=OBS_SCALE[1],
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "ticks": 16384},
        )
        # inner pole (cart_to_ipole) -- already absolute, measured off the cart
        ipole_sin = ObsTerm(
            func=mdp.pole_angle_sin_quantized,
            scale=OBS_SCALE[2],
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_ipole"]), "ticks_per_rev": 4096},
        )
        ipole_cos = ObsTerm(
            func=mdp.pole_angle_cos_quantized,
            scale=OBS_SCALE[3],
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_ipole"]), "ticks_per_rev": 4096},
        )
        ipole_vel = ObsTerm(
            func=mdp.pole_angular_vel_noisy,
            scale=OBS_SCALE[4],
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_ipole"])},
        )
        # outer pole -- RELATIVE angle, same sensor/noise as the inner joint
        opole_sin = ObsTerm(
            func=mdp.pole_angle_sin_quantized,
            scale=OBS_SCALE[5],
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["ipole_to_opole"]), "ticks_per_rev": 4096},
        )
        opole_cos = ObsTerm(
            func=mdp.pole_angle_cos_quantized,
            scale=OBS_SCALE[6],
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["ipole_to_opole"]), "ticks_per_rev": 4096},
        )
        opole_vel = ObsTerm(
            func=mdp.pole_angular_vel_noisy,
            scale=OBS_SCALE[7],
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["ipole_to_opole"])},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class DoubleCartpoleEventCfg:
    randomize_cart_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["cart_1"]),
            "mass_distribution_params": (0.20, 0.01),
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
            "operation": "abs",
            "distribution": "gaussian",
        },
    )

    randomize_slider_friction = EventTerm(
        func=mdp.randomize_slider_friction_effort,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "static_range": (3.0, 3.5),
            "dynamic_params": (2.5, 0.27),
            "viscous_value": 0.2,
            # "static_range": (3.25, 3.25),
            # "dynamic_params": (2.5, 0.0),
            # "viscous_value": 0.2,
        },
    )
    randomize_slider_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "armature_distribution_params": (0.43, 0.05),
            # "armature_distribution_params": (0.43, 0.0),
            "operation": "abs",
            "distribution": "gaussian",
        },
    )

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
        params={"mean": 0.3, "std": 0.122, "clip": (0.05, 0.8)},
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
        self.episode_length_s = 10
        # self.viewer.eye = (0.51, -2.5, 2.2)
        # self.viewer.lookat = (0.51, 0.0, 2.0)
        self.viewer.eye = (0.0, -1.1, 2.1)
        self.viewer.lookat = (0.0, 0.0, 2.0)
        self.viewer.resolution = (1920, 1080)
        # self.viewer.resolution = (1280, 720)

        # self.sim.dt = 1 / 1000
        self.sim.dt = 0.0010251
        self.sim.render_interval = self.decimation
