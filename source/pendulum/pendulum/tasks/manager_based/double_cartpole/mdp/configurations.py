"""Configuration for the double cart-pole robot (double_pendulum.usda)."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Configuration
##

DOUBLE_PENDULUM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(os.path.dirname(__file__), "assets/double_pendulum.usda"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            # deg/s -- keep high; 1000 (=17.45 rad/s) clamps swing-up. See single_cartpole.
            max_angular_velocity=6000.0,
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
    # joint_pos 0 for both revolute joints == fully upright; keep 0 so joint_pos_rel == absolute angle.
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.0),
        joint_pos={"slider_to_cart": 0.0, "cart_to_ipole": 0.0, "ipole_to_opole": 0.0},
    ),
    actuators={
        # same physical cart + motor as the single cartpole
        "cart_actuator": DCMotorCfg(
            joint_names_expr=["slider_to_cart"],
            saturation_effort=62.8,
            velocity_limit=2.65,
            stiffness=0.0,
            damping=0.0,
            effort_limit_sim=40.0,
            armature=0.37,
        ),
        # both pendulum joints are passive (underactuated); tiny damping, DR varies it per-episode
        "pole_actuator": ImplicitActuatorCfg(
            joint_names_expr=["cart_to_ipole", "ipole_to_opole"],
            effort_limit_sim=1000.0,
            stiffness=0.0,
            damping=0.00001,
        ),
    },
)
