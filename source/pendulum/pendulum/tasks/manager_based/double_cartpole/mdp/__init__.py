# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for the double_cartpole task.

The observation noise and the custom reset/friction events are REUSED from single_cartpole: it is
the same physical rig and the same encoders (the ipole_to_opole sensor is identical to cart_to_ipole
and to the single cartpole's revolute-joint sensor), so the noise profiles are identical. Only the
robot configuration (double_pendulum.usda) and the double-pendulum rewards are defined locally.
"""

from isaaclab.envs.mdp import (
    JointEffortActionCfg,
    joint_pos_rel,
    joint_vel_rel,
    reset_joints_by_offset,
    randomize_joint_parameters,
    randomize_rigid_body_mass,
    randomize_actuator_gains,
    is_alive,
    is_terminated,
    last_action,
    joint_vel_l1,
    joint_vel_l2,
    time_out,
    joint_pos_out_of_manual_limit,
    apply_external_force_torque,
)

# --- reused from single_cartpole (identical hardware / sensors) ---
from ...single_cartpole.mdp.observations import (
    add_encoder_tick_noise,
    cart_pos_noisy,
    cart_vel_noisy,
    pole_angle_sin_quantized,
    pole_angle_cos_quantized,
    pole_angle_sin_noisy,
    pole_angle_cos_noisy,
    pole_angular_vel_noisy,
)
from ...single_cartpole.mdp.events import (
    randomize_slider_friction_effort,
    randomize_pole_vel_noise_std,
    reset_joints_by_offset_mixed,
    reset_joints_by_offset_unclamped,
)
from ...single_cartpole.mdp.rewards import (
    joint_pos_target_l1,
    joint_pos_target_l2,
    joint_effort_l2,
    action_rate_l2,
)

# --- double_cartpole specific ---
from .rewards import *  # noqa: F401, F403
from .configurations import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
from .events import *  # noqa: F401, F403
