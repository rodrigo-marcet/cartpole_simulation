# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Constant-force cart response (sim counterpart of the rig FIXED_TORQUE test): drive the cart at
each --forces level with the pole hanging still, and log the torque-speed curve (initial accel +
terminal no-load speed) in play's env. Compare v_max to the rig ~2.63 m/s cap (sim higher =>
velocity_limit too high). Writes one <F>n.csv per force into --out_dir, columns matching the rig
CSVs so they overlay directly. Pole angle 0 = upright, +/-pi = hanging.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Constant-force cart probe (sim counterpart of rig FIXED_TORQUE).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (only env 0 is used).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default=None, help="Name of the RL agent configuration entry point.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (unused; parity only).")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"])
parser.add_argument("--algorithm", type=str, default="PPO", choices=["AMP", "PPO", "IPPO", "MAPPO"])
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--forces", type=str, default="5,10,15,20,25,30", help="Comma-separated force levels [N] to sweep.")
parser.add_argument("--start_pos", type=float, default=-0.26, help="Cart start position [m] (mirrors rig ~+0.26).")
parser.add_argument(
    "--stop_pos", type=float, default=0.25, help="Stop when cart_pos reaches this [m] (mirrors rig -0.25)."
)
parser.add_argument("--max_steps", type=int, default=250, help="Safety cap on steps per force.")
parser.add_argument(
    "--out_dir", type=str, default=None, help="Output folder (default: single_cartpole/output/motor_curve/)."
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import math
import os
import random

import gymnasium as gym
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

# ============================================================================
# PROBE KNOBS
# ============================================================================
POLE_HANG = math.pi  # pole hanging straight down (0 = upright)


def _out_dir(name):
    """scripts/simulations/single_cartpole/output/<name>/ (created), anchored to the repo, not cwd."""
    d = os.path.dirname(os.path.abspath(__file__))
    while os.path.basename(d) != "single_cartpole" and os.path.dirname(d) != d:
        d = os.path.dirname(d)
    out = os.path.join(d, "output", name)
    os.makedirs(out, exist_ok=True)
    return out


if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


def _zero_std(term, key):
    p = term.params[key]
    term.params[key] = (p[0], 0.0)


def _freeze_dr(env_cfg):
    """Freeze every randomized PHYSICS term to its configured center so env 0 == the calibrated
    nominal cart. Pole stays present & passive (its hanging inertia loads the cart, like the rig).
    Widen the cart bound so env.step() never resets during the ~0.5 m drive.
    """
    ev = env_cfg.events
    for term_name, key in [
        ("randomize_pole_friction", "friction_distribution_params"),
        ("randomize_pole_damping", "damping_distribution_params"),
        ("randomize_slider_armature", "armature_distribution_params"),
        ("randomize_cart_mass", "mass_distribution_params"),
        ("randomize_shaft_mass", "mass_distribution_params"),
        ("randomize_pole_mass", "mass_distribution_params"),
        ("randomize_weight_mass", "mass_distribution_params"),
    ]:
        try:
            _zero_std(getattr(ev, term_name), key)
        except Exception as e:  # noqa: BLE001
            print(f"[freeze_dr] skip {term_name}.{key}: {e}")
    try:
        fr = ev.randomize_slider_friction.params
        dyn = fr["dynamic_params"]
        fr["dynamic_params"] = (dyn[0], 0.0)
        lo, hi = fr["static_range"]
        mid = 0.5 * (lo + hi)
        fr["static_range"] = (mid, mid)
    except Exception as e:  # noqa: BLE001
        print(f"[freeze_dr] skip randomize_slider_friction: {e}")
    try:
        env_cfg.terminations.cart_out_of_bounds.params["bounds"] = (-5.0, 5.0)
    except Exception as e:  # noqa: BLE001
        print(f"[freeze_dr] could not widen cart bounds: {e}")
    env_cfg.episode_length_s = 1.0e6
    return ev


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

    forces = [float(x) for x in args_cli.forces.split(",")]
    action_scale = float(getattr(env_cfg.actions.joint_effort, "scale", 30.0))
    _freeze_dr(env_cfg)

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

    env.reset()
    dev = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    _rb = env.unwrapped.scene["robot"]
    _sidx = _rb.find_joints("slider_to_cart")[0][0]
    _pidx = _rb.find_joints("cart_to_pole")[0][0]
    dt = getattr(env.unwrapped, "step_dt", env.unwrapped.physics_dt * env_cfg.decimation)
    e0 = torch.tensor([0], device=dev)

    print("=" * 78)
    print("MOTOR-CURVE (constant-force cart response, pole hanging)")
    print(f"  num_envs={num_envs}  device={dev}  step_dt={dt:.4f}s  action_scale={action_scale} N")
    print(
        f"  forces [N] = {forces}   drive {args_cli.start_pos:+.2f} -> {args_cli.stop_pos:+.2f} m  "
        f"(max {args_cli.max_steps} steps)"
    )
    print("=" * 78)

    def reset_cart():
        """Cart at start_pos (at rest); pole hanging straight down and still."""
        with torch.inference_mode():
            jp = _rb.data.joint_pos[0:1].clone()
            jv = _rb.data.joint_vel[0:1].clone()
            jp[0, _sidx] = args_cli.start_pos
            jp[0, _pidx] = POLE_HANG
            jv[0, _sidx] = 0.0
            jv[0, _pidx] = 0.0
            _rb.write_joint_state_to_sim(jp, jv, env_ids=e0)
            _rb.update(env.unwrapped.physics_dt)

    def cart_pos():
        return float(_rb.data.joint_pos[0, _sidx])

    def cart_vel():
        return float(_rb.data.joint_vel[0, _sidx])

    def pole_ang():
        return float(_rb.data.joint_pos[0, _pidx])

    out_dir = os.path.abspath(args_cli.out_dir) if args_cli.out_dir else _out_dir("motor_curve")
    os.makedirs(out_dir, exist_ok=True)
    summary = []  # (force, a0, v_max)
    for F in forces:
        reset_cart()
        act = torch.zeros((num_envs, 1), dtype=torch.float32, device=dev)
        act[0, 0] = F / action_scale  # push +; drives start_pos -> stop_pos
        frows = []  # this force's per-step rows: (step, t, pos, vel, pole)
        speeds = []
        vmax = 0.0
        for k in range(args_cli.max_steps):
            p, v = cart_pos(), cart_vel()
            speeds.append(abs(v))
            vmax = max(vmax, abs(v))
            frows.append((k, k * dt, p, v, pole_ang()))
            if p >= args_cli.stop_pos:  # reached the far end (rig brake point)
                break
            with torch.inference_mode():
                _, _, _, dones, _ = env.step(act)
            if bool(dones[0]):
                print(f"  [WARN] env reset during F={F} at step {k}.")
                break
        a0 = (speeds[2] - speeds[1]) / dt if len(speeds) >= 3 else float("nan")
        summary.append((F, a0, vmax))
        # one CSV per force, named/columned like the rig files ground_truth/<F>n.csv
        with open(os.path.join(out_dir, f"{F:.0f}n.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["force_n", "step", "t_s", "cart_pos_m", "cart_vel_mps", "pole_ang_rad"])
            for k, t, p, v, pole in frows:
                w.writerow([f"{F:.0f}", k, f"{t:.5f}", f"{p:.6f}", f"{v:.6f}", f"{pole:.6f}"])

    # --- table + effective-mass fit ---
    print("=" * 78)
    print(f"  {'F (N)':>6} | {'a0 init (m/s^2)':>15} | {'v_max plateau (m/s)':>19}")
    Fs, a0s = [], []
    for F, a0, vmax in summary:
        print(f"  {F:>6.0f} | {a0:>15.2f} | {vmax:>19.2f}")
        Fs.append(F)
        a0s.append(a0)
    if len(Fs) >= 2:
        n = len(Fs)
        mx, my = sum(Fs) / n, sum(a0s) / n
        sxx = sum((x - mx) ** 2 for x in Fs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(Fs, a0s))
        slope = sxy / sxx
        m_eff = 1.0 / slope
        coulomb = -(my - slope * mx) / slope
        print("-" * 78)
        print(f"  SIM effective mass (cart+drivetrain+hanging pole) = {m_eff:.3f} kg   Coulomb ~ {coulomb:.2f} N")
    print("  v_max should match the RIG's ~2.63 m/s cap; sim HIGHER => DCMotor velocity_limit too high.")
    print(f"  wrote {len(forces)} files -> {out_dir}{os.sep}<F>n.csv")
    print("=" * 78)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
