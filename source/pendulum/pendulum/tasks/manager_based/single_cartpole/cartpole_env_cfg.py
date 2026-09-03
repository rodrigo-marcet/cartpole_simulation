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
class CartpoleSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

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

    robot: ArticulationCfg = mdp.FUSION_CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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
        init_state=AssetBaseCfg.InitialStateCfg(
            # rot=euler_deg_to_quat(67.0, -14.0, 90.0),
            rot=(0.71488, 0.32137, 0.40919, -0.46715)
            # rot=(67.0, -14.0, 90.0),
        ),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=40.0)


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
class CartpoleEventCfg:
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

    randomize_shaft_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["shaft_1"]),
            "mass_distribution_params": (0.080, 0.004),
            # "mass_distribution_params": (0.080, 0.0),
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
            "mass_distribution_params": (0.030, 0.0015),
            # "mass_distribution_params": (0.030, 0.0),
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
            # "static_range": (1.85, 1.85),
            # "dynamic_params": (1.41, 0.0),
            # "viscous_value": 0.48,
        },
    )

    randomize_slider_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "armature_distribution_params": (0.37, 0.05),
            # "armature_distribution_params": (0.37, 0.0),
            "operation": "abs",
            "distribution": "gaussian",
        },
    )

    randomize_pole_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            # "friction_distribution_params": (1.5e-5, 3.9e-6),
            "friction_distribution_params": (1.5e-5, 0.0),
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
            # "damping_distribution_params": (7.5e-6, 0.0),
            "operation": "abs",
            "distribution": "gaussian",
        },
    )

    randomize_pole_vel_noise = EventTerm(
        func=mdp.randomize_pole_vel_noise_std,
        mode="reset",
        params={"mean": 0.4, "std": 0.122, "clip": (0.05, 0.8)},
        # params={"mean": 0.4, "std": 0.0, "clip": (0.05, 0.8)},
    )


##
# Base environment configuration
##
@configclass
class CartpoleEnvCfg(ManagerBasedRLEnvCfg):
    scene: CartpoleSceneCfg = CartpoleSceneCfg(num_envs=4096, env_spacing=1.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self) -> None:
        self.decimation = 10
        self.episode_length_s = 5
        self.viewer.eye = (0.0, -1.0, 2.1)
        self.viewer.lookat = (0.0, 0.0, 2.0)
        self.viewer.resolution = (1920, 1080)
        self.sim.dt = 1 / 1000
        self.sim.render = sim_utils.RenderCfg(
            rendering_mode="quality",
            antialiasing_mode="DLAA",
            enable_dl_denoiser=True,
        )
        self.sim.render_interval = self.decimation
