# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--freeze_dr",
    action="store_true",
    default=False,
    help="Collapse domain randomization to its MIDPOINT (every env == the nominal calibrated rig).",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import random
import time

import gymnasium as gym
import skrl
import torch
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

import pendulum.tasks  # noqa: F401

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


def _freeze_dr(env_cfg):
    """Freeze the RIG (physics + sensor noise) DR to its nominal midpoint, per env.

    The START-STATE resets (pole position/velocity, cart position) are LEFT RANDOMIZED on
    purpose, so the policy is still tested from the normal variety of initial conditions --
    only the rig parameters are pinned. static range -> midpoint; gaussians -> (center, std=0);
    the per-episode pole-noise std -> its mean (noise stays ON, just not randomized). Edits
    env_cfg in memory, so the config file is untouched.
    """
    ev = getattr(env_cfg, "events", None)
    if ev is None:
        print("[freeze_dr] no events on env_cfg; nothing to freeze.")
        return

    def mid_range(term, key):
        try:
            a, b = term.params[key]
            m = 0.5 * (float(a) + float(b))
            term.params[key] = (m, m)
        except Exception as e:  # noqa: BLE001
            print(f"[freeze_dr] skip {key}: {e}")

    def zero_std(term, key):
        try:
            p = term.params[key]
            term.params[key] = (p[0], 0.0)
        except Exception as e:  # noqa: BLE001
            print(f"[freeze_dr] skip {key}: {e}")

    def get(name):
        return getattr(ev, name, None)

    # NOTE: reset_cart_position / reset_pole_position are deliberately NOT frozen -- the pole
    # still starts across its configured range (pi +/- 0.5) so the policy faces varied swing-ups.
    # slider friction: static range -> midpoint; dynamic -> center (std 0); viscous unchanged
    t = get("randomize_slider_friction")
    if t:
        mid_range(t, "static_range")
        zero_std(t, "dynamic_params")
    # gaussian (center, std) terms -> (center, 0)
    for nm, key in (
        ("randomize_slider_armature", "armature_distribution_params"),
        ("randomize_cart_mass", "mass_distribution_params"),
        ("randomize_shaft_mass", "mass_distribution_params"),
        ("randomize_pole_mass", "mass_distribution_params"),
        ("randomize_weight_mass", "mass_distribution_params"),
        ("randomize_pole_friction", "friction_distribution_params"),
        ("randomize_pole_damping", "damping_distribution_params"),
    ):
        t = get(nm)
        if t:
            zero_std(t, key)
    # per-episode pole-velocity noise std -> its mean (noise still applied at the midpoint level)
    t = get("randomize_pole_vel_noise")
    if t and "std" in t.params:
        t.params["std"] = 0.0
    print("[freeze_dr] DR frozen to midpoint -- every env is the nominal rig (noise kept at its mean).")


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Play with skrl agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

        # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # optional: collapse DR to its midpoint so the policy is evaluated in the nominal rig
    if args_cli.freeze_dr:
        _freeze_dr(env_cfg)

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")
    probe = torch.tensor([[-0.345092, -2.589072, -0.53195, 0.84678, 24.96285]], device=runner.agent.device)
    sp = runner.agent._state_preprocessor
    print("CLIP_THRESHOLD:", getattr(sp, "clip_threshold", None))  # is it really 5.0?
    print("EPSILON       :", getattr(sp, "epsilon", None))
    print("MEAN:", sp.running_mean.flatten().tolist())
    print("VAR :", sp.running_variance.flatten().tolist())
    print("PLAY SCALED:", sp(probe).flatten().tolist())  # what the net actually receives
    print(runner.agent.policy)  # the REAL architecture
    e = env.unwrapped
    print("PLAY decimation:", e.cfg.decimation, "| sim.dt:", e.physics_dt, "| step_dt:", e.step_dt)
    print("PLAY resume_path:", resume_path)

    # reset environment
    obs, _ = env.reset()
    print("play masses:", env.unwrapped.scene["robot"].root_physx_view.get_masses()[:5].tolist())
    print("play inertias:", env.unwrapped.scene["robot"].root_physx_view.get_inertias()[:5].tolist())
    rb = env.unwrapped.scene["robot"]
    v = rb.root_physx_view
    print("PLAY friction :", v.get_dof_friction_coefficients()[0].tolist())
    print("PLAY armature :", v.get_dof_armatures()[0].tolist())
    print("PLAY stiffness:", v.get_dof_stiffnesses()[0].tolist())
    print("PLAY damping  :", v.get_dof_dampings()[0].tolist())
    print("PLAY max_vel  :", v.get_dof_max_velocities()[0].tolist())
    print("PLAY max_force:", v.get_dof_max_forces()[0].tolist())

    v = env.unwrapped.scene["robot"].root_physx_view
    try:
        print("PLAY solver_pos_iters:", int(v.get_solver_position_iteration_counts()[0]))
        print("PLAY solver_vel_iters:", int(v.get_solver_velocity_iteration_counts()[0]))
    except Exception as e:
        print("PLAY solver iters: <unavailable>", e)
    try:
        pc = env.unwrapped.sim.get_physics_context()
        print(
            "PLAY gpu_dynamics:",
            pc.is_gpu_dynamics_enabled(),
            "| solver_type:",
            pc.get_solver_type(),
            "| gravity:",
            pc.get_gravity(),
        )
    except Exception as e:
        print("PLAY physics_context: <unavailable>", e)

    timestep = 0
    # simulate environment

    sidx = env.unwrapped.scene["robot"].find_joints("slider_to_cart")[0][0]  # MINE

    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, _, _, _, _ = env.step(actions)

            rb = env.unwrapped.scene["robot"]
            sidx = rb.find_joints("slider_to_cart")[0][0]  # MINE
            print(
                "PLAY cart_vel:", rb.data.joint_vel[0, sidx].item(), "applied:", rb.data.applied_torque[0, sidx].item()
            )  # MINE

        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
