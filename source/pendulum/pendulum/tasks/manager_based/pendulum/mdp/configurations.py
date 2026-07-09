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
        usd_path=os.path.join(os.path.dirname(__file__), "assets/cartpole/cartpole.usda"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            # UNITS GOTCHA: RigidBodyPropertiesCfg.max_angular_velocity is in DEG/s (Isaac Lab schema
            # docs). The old 1000 was really 1000 deg/s = 17.45 rad/s, which CLAMPED the pole just below
            # the ~17.7 rad/s (2*omega_n) a swing-up needs -> this was THE high-omega "energy loss" that
            # broke swing-up transfer (proven in scripts/skrl/play_energy_sweep.py). Balancing stays near
            # upright (low omega) so this clamp didn't bite balance, but we sync it anyway to run the same
            # faithful rig. 6000 deg/s = ~105 rad/s: ample headroom, keeps a stability cap.
            max_angular_velocity=6000.0,  # was 1000.0 (=17.45 rad/s clamp) -- deg/s!
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            # 4/0 are the known-good balance values (32/64 vs 4/0 were byte-identical in the swing-up
            # sweep -- iterations were never the issue). Keep them.
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
            # Fitted to the rig FIXED_TORQUE torque-speed curves (analysis/data/fixed_torque,
            # fit_dcmotor.py): flat shelves -> effective mass ~0.70 kg & Coulomb ~1.4 N; the
            # force-dependent knees -> saturation_effort; the ~2.6 m/s velocity cap -> no-load speed.
            saturation_effort=62.8,  # stall force; sets the torque-speed droop slope (was 30.0)
            velocity_limit=2.65,  # no-load speed = measured cart velocity cap (was 3.3)
            stiffness=0.0,
            damping=0.0,
            effort_limit_sim=40.0,  # current-limit force ~ 10 A * 0.04 Nm/A / 0.01 m (was 30.0)
            # OPERATIVE armature is set by randomize_slider_armature in pendulum_env_cfg.py (it
            # overwrites this at reset) -- keep the two in sync. Targets effective mass ~0.70 kg.
            armature=0.37,  # kg (was absent; DR center in pendulum_env_cfg.py must match)
        ),
        "pole_actuator": ImplicitActuatorCfg(
            joint_names_expr=["cart_to_pole"],
            effort_limit_sim=1000.0,
            stiffness=0.0,
            damping=0.00001,  # BEST ONE SO FAR
        ),
    },
)

"""Configuration for a simple Cartpole robot."""
