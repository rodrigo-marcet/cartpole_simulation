# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-pole free-swing JIG for validating the two double-cartpole pole inertias in sim.

Reuses the single_cartpole balancing env (identical rig / MDP / sensors) but spawns the
co-located jig USD (a copy of the single cartpole) under assets/single_cartpole/. The
double free_swing probe overrides the swinging assembly's (mass, CoM, inertia) to each
double pole and compares the sim swing to the real bench swing. Task: FreeSwing-DoubleCartpoleJig-v0.
"""

import os

from isaaclab.utils import configclass

from ..single_cartpole.balancing_env_cfg import BalancingEnvCfg

_JIG_USD = os.path.join(os.path.dirname(__file__), "mdp/assets/single_cartpole/cartpole/cartpole.usda")


@configclass
class JigEnvCfg(BalancingEnvCfg):
    """Balancing rig pointed at the jig USD, with the pole-angle / tight-cart terminations removed
    so a full free swing runs without resetting. The probe seeds the release state itself."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.robot.spawn.usd_path = _JIG_USD
        self.terminations.pole_out_of_bounds = None
        self.terminations.cart_out_of_bounds.params["bounds"] = (-1.0, 1.0)
        self.episode_length_s = 120.0
        # Use the USD masses as authored (the physics file holds the pole values); the single-cartpole
        # mass-DR would otherwise reset them to its own centers every episode.
        self.events.randomize_cart_mass = None
        self.events.randomize_shaft_mass = None
        self.events.randomize_pole_mass = None
        self.events.randomize_weight_mass = None
