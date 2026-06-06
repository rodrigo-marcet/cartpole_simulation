"""Configuration for a simple Cartpole robot."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

FUSION_CARTPOLE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Classic/Cartpole/cartpole.usd",
        usd_path=os.path.join(os.path.dirname(__file__), "assets/cartpole_acc/cartpole_acc.usda"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.0), joint_pos={"slider_to_cart": 0.0, "cart_to_pole": 0.0}
    ),
    actuators={
        "cart_actuator": DCMotorCfg(
            joint_names_expr=["slider_to_cart"],
            saturation_effort=30.0,  # peak stall force
            velocity_limit=3.3,  # no-load speed
            stiffness=0.0,
            damping=0.0,
            effort_limit_sim=30.0,
        ),
        "pole_actuator": ImplicitActuatorCfg(
            joint_names_expr=["cart_to_pole"], effort_limit_sim=0.0, stiffness=0.0, damping=0.0
        ),
    },
)

"""Configuration for a simple Cartpole robot."""
