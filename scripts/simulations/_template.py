# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Template for the scripts/skrl/simulations probes. Copy it and fill the four ★SLOTS★: ★1 KNOBS,
★2 _freeze_dr, ★3 PROBE BODY (one of the 3 patterns marked below), ★4 this docstring (keep it a few
lines: what it measures, how to read the result, output file). Everything else is the shared harness
-- keep it identical across probes. Keep docstrings/comments short.
"""

"""Launch Isaac Sim Simulator first."""

# ══════════════════════════════════════════════════════════════════════════════════════
# SHARED HARNESS -- launch  (identical in every probe; full standard skrl header)
# ══════════════════════════════════════════════════════════════════════════════════════
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="<PROBE NAME> probe inside play's env.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default=None, help="Name of the RL agent configuration entry point.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (unused; parity only).")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"])
parser.add_argument("--algorithm", type=str, default="PPO", choices=["AMP", "PPO", "IPPO", "MAPPO"])
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# ★ add probe-specific flags here (e.g. --logs, --forces, --w_min ...) ★

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# ══════════════════════════════════════════════════════════════════════════════════════
# SHARED HARNESS -- post-launch imports  (identical in every probe)
# ══════════════════════════════════════════════════════════════════════════════════════
import csv
import math
import os
import random

import gymnasium as gym
import numpy as np
import skrl
import torch
from packaging import version

SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(f"Unsupported skrl version: {skrl.__version__}. Install skrl>={SKRL_VERSION}")
    exit()

import pendulum.tasks  # noqa: F401

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# ══════════════════════════════════════════════════════════════════════════════════════
# ★1 KNOBS -- probe constants (per-probe)
# ══════════════════════════════════════════════════════════════════════════════════════
CSV_PATH = "<probe>.csv"
CSV_PATH_NOISY = "<probe>_noisy.csv"  # policy-observed sensor values (standardized across probes)
# ... probe targets / initial conditions ...


def _out_dir(name):
    """scripts/skrl/simulations/output/<name>/ (created), anchored to the repo, not cwd."""
    d = os.path.dirname(os.path.abspath(__file__))
    while os.path.basename(d) != "simulations" and os.path.dirname(d) != d:
        d = os.path.dirname(d)
    out = os.path.join(d, "output", name)
    os.makedirs(out, exist_ok=True)
    return out


OUT_DIR = _out_dir("<probe>")  # write CSVs here, e.g. os.path.join(OUT_DIR, CSV_PATH)

# SHARED HARNESS -- agent entry-point resolution (identical in every probe)
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


# ══════════════════════════════════════════════════════════════════════════════════════
# SHARED HARNESS -- standardized policy-observed sensor readout (the "noisy" CSV)
# Returns the 5 values the POLICY sees for env 0: cart_pos, cart_vel, pole_angle, pole_vel.
# (obs index map: 0=cart_pos 1=cart_vel 2=pole_sin 3=pole_cos 4=pole_vel)
# ══════════════════════════════════════════════════════════════════════════════════════
def policy_obs_env0(env):
    o = env.unwrapped.observation_manager.compute()["policy"][0]
    return float(o[0]), float(o[1]), math.atan2(float(o[2]), float(o[3])), float(o[4])


def write_csv(path, header, rows):
    with open(os.path.abspath(path), "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(header)
        wtr.writerows(rows)
    return os.path.abspath(path)


# ══════════════════════════════════════════════════════════════════════════════════════
# ★2 _freeze_dr -- collapse the DR terms this probe cares about to their nominal center
# ══════════════════════════════════════════════════════════════════════════════════════
def _freeze_dr(env_cfg):
    ev = env_cfg.events
    # example (slider calibration probes):
    #   dyn = ev.randomize_slider_friction.params["dynamic_params"]
    #   ev.randomize_slider_friction.params["dynamic_params"] = (dyn[0], 0.0)
    #   ev.randomize_slider_friction.params["static_range"] = (1.8, 1.8)
    #   arm = ev.randomize_slider_armature.params["armature_distribution_params"]
    #   ev.randomize_slider_armature.params["armature_distribution_params"] = (arm[0], 0.0)
    env_cfg.terminations.cart_out_of_bounds.params["bounds"] = (-1.0, 1.0)  # don't reset mid-probe
    env_cfg.episode_length_s = 30.0
    return ev


# ══════════════════════════════════════════════════════════════════════════════════════
# SHARED HARNESS -- main() scaffold (identical up to the ★3 PROBE BODY★)
# ══════════════════════════════════════════════════════════════════════════════════════
@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    _freeze_dr(env_cfg)

    # log_dir parity (policy is NOT loaded; kept for parity with train/play)
    log_root_path = os.path.abspath(os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"]))
    try:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
        env_cfg.log_dir = os.path.dirname(os.path.dirname(resume_path))
    except Exception:
        env_cfg.log_dir = log_root_path

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    obs, _ = env.reset()
    dev = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    _rb = env.unwrapped.scene["robot"]
    _view = _rb.root_physx_view
    _sidx = _rb.find_joints("slider_to_cart")[0][0]
    _pidx = _rb.find_joints("cart_to_pole")[0][0]
    dt_ctrl = getattr(env.unwrapped, "step_dt", env.unwrapped.physics_dt * env_cfg.decimation)

    print("=" * 78)
    print("<PROBE NAME>")
    print(f"  num_envs={num_envs}  device={dev}  step_dt={dt_ctrl:.4f}s")
    print("=" * 78)

    # ══════════════════════════════════════════════════════════════════════════════════
    # ★3 PROBE BODY -- pick ONE pattern:
    #
    # PATTERN A (single-env probe: coast_down, breakaway, free_swing, replay)
    #   - seed env 0 state, then step loop feeding an action;
    #   - log BOTH the true sim state (CSV_PATH) and policy_obs_env0(env) (CSV_PATH_NOISY);
    #   - fit / verdict at the end.
    #
    # PATTERN B (sweep: motor_curve, energy_conservation)
    #   - loop over a swept parameter (forces / omega); one short run each; collect a row per value.
    #
    # PATTERN C (multi-env inspect: dr_spread)
    #   - one reset, read _view.get_dof_*()/get_masses() across ALL envs, print per-term stats.
    #     (does NOT _freeze_dr and writes no CSV -- it is measuring the spread itself.)
    # ══════════════════════════════════════════════════════════════════════════════════

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
