# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Swingup-DoubleCartpole-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.swingup_env_cfg:SwingupEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Single-pole jig for validating each pole's inertia in sim (see jig_env_cfg.py). Reuses the
# single_cartpole 5-dim agent cfg since the probe runs open-loop (no policy is loaded).
gym.register(
    id="FreeSwing-DoubleCartpoleJig-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jig_env_cfg:JigEnvCfg",
        "skrl_cfg_entry_point": "pendulum.tasks.manager_based.single_cartpole.agents:skrl_ppo_cfg.yaml",
    },
)
