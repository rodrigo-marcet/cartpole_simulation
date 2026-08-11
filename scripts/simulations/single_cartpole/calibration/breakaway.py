# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Breakaway probe: ramp the cart force until it starts moving, in play's env. Reports the
breakaway torque (rig ~0.018 N*m = static friction) and the post-breakaway acceleration (checks
effective mass + dynamic friction). Writes breakaway.csv (+ _noisy: policy-observed sensor values).
Run with the SAME --task / --num_envs you train with.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Breakaway (static friction) + accel probe inside play's env.")
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

# ============================================================================
# PROBE KNOBS
# ============================================================================
RAMP = 0.005  # torque ramp rate [N*m/s] (matches simulations/torque.py and the rig test)
R_PULLEY = 0.01  # m  (force = torque / r)
SCALE = 30.0  # JointEffortActionCfg scale (action -> effort). MUST match ActionsCfg.
X0 = -0.3  # start pos [m]; cart creeps toward + within the +/-0.5 joint limit
POLE_ANGLE = 0.0  # seed pole here. 0.0 when the pole joint is LOCKED (USDA limits (0,0)); math.pi
# only if the pole limits are WIDE. A pi seed against a (0,0) limit yanks the cart.
RIG_BREAKAWAY = 0.018  # N*m target
RIG_V_AT_021 = 0.134  # m/s the rig reaches ~0.21 s after breakaway (for the accel check)
CSV_PATH = "breakaway.csv"
CSV_PATH_NOISY = "breakaway_noisy.csv"  # policy-observed sensor values


def _out_dir(name):
    """scripts/simulations/single_cartpole/output/<name>/ (created), anchored to the repo, not cwd."""
    d = os.path.dirname(os.path.abspath(__file__))
    while os.path.basename(d) != "single_cartpole" and os.path.dirname(d) != d:
        d = os.path.dirname(d)
    out = os.path.join(d, "output", name)
    os.makedirs(out, exist_ok=True)
    return out


OUT_DIR = _out_dir("breakaway")

if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


def _freeze_dr(env_cfg):
    ev = env_cfg.events
    dyn = ev.randomize_slider_friction.params["dynamic_params"]
    ev.randomize_slider_friction.params["dynamic_params"] = (dyn[0], 0.0)  # keep center, zero std
    ev.randomize_slider_friction.params["static_range"] = (1.8, 1.8)  # calibrated breakaway nominal
    arm = ev.randomize_slider_armature.params["armature_distribution_params"]
    ev.randomize_slider_armature.params["armature_distribution_params"] = (arm[0], 0.0)
    env_cfg.terminations.cart_out_of_bounds.params["bounds"] = (-1.0, 1.0)  # don't reset mid-ramp
    env_cfg.episode_length_s = 30.0  # ramp reaches breakaway ~3.6 s
    return env_cfg.events.randomize_slider_friction.params, env_cfg.events.randomize_slider_armature.params


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

    fr_params, arm_params = _freeze_dr(env_cfg)

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
    print("BREAKAWAY (static friction + post-breakaway accel)")
    print(f"  num_envs={num_envs}  device={dev}  step_dt={dt_ctrl:.4f}s  ramp={RAMP} N*m/s  scale={SCALE}")
    print(
        f"  frozen DR -> dynamic={fr_params['dynamic_params']}  static={fr_params['static_range']}  "
        f"armature={arm_params['armature_distribution_params']}"
    )
    try:
        print("  env0 armature read-back :", _view.get_dof_armatures()[0].tolist())
        print(
            "  env0 friction_props     :",
            _view.get_dof_friction_properties()[0].tolist(),
            "(may lie; behavior is truth)",
        )
    except Exception as e:
        print("  read-back unavailable:", e)
    print("=" * 78)

    # seed env 0 at rest, pole hanging at pi
    jp = _rb.data.joint_pos[0:1].clone()
    jv = _rb.data.joint_vel[0:1].clone()
    jp[0, _sidx] = X0
    jp[0, _pidx] = POLE_ANGLE
    jv[0, _sidx] = 0.0
    jv[0, _pidx] = 0.0
    _rb.write_joint_state_to_sim(jp, jv, env_ids=torch.tensor([0], device=dev))
    _rb.update(env.unwrapped.physics_dt)

    # Probe obs layout once; the noisy log reads the obs manager (= what the policy sees).
    try:
        _probe = env.unwrapped.observation_manager.compute()["policy"]
        noisy_ok = True
        print(
            f"  obs layout: shape={tuple(_probe.shape)}  "
            "(index map: 0=cart_pos 1=cart_vel 2=pole_sin 3=pole_cos 4=pole_vel)"
        )
    except Exception as _e:  # noqa: BLE001
        noisy_ok = False
        print("  [!] observation_manager unavailable -> noisy CSV skipped:", _e)

    ts, taus, vs, xs, applied, pas, pvs = [], [], [], [], [], [], []
    n_vs, n_xs, n_pas, n_pvs = [], [], [], []  # NOISY sensor logs (what the policy observes)
    _act = torch.zeros((num_envs, 1), dtype=torch.float32, device=dev)
    breakaway = None
    k = 0
    while True:
        t = k * dt_ctrl
        v = _rb.data.joint_vel[0, _sidx].item()
        x = _rb.data.joint_pos[0, _sidx].item()
        tau = RAMP * t  # commanded torque [N*m]
        force = tau / R_PULLEY  # commanded force [N]
        ts.append(t)
        taus.append(tau)
        vs.append(v)
        xs.append(x)
        applied.append(_rb.data.applied_torque[0, _sidx].item())
        pas.append(_rb.data.joint_pos[0, _pidx].item())
        pvs.append(_rb.data.joint_vel[0, _pidx].item())
        if noisy_ok:
            _o = env.unwrapped.observation_manager.compute()["policy"][0]
            n_xs.append(float(_o[0]))
            n_vs.append(float(_o[1]))
            n_pas.append(math.atan2(float(_o[2]), float(_o[3])))
            n_pvs.append(float(_o[4]))

        if breakaway is None and abs(v) > 0.05:
            breakaway = (t, tau, k)
            print(f"  >> BREAKAWAY at t={t:.3f}s  tau={tau:.5f} N*m  (F={force:.3f} N)   [rig ~{RIG_BREAKAWAY}]")

        if (breakaway is not None and abs(v) > 0.5) or abs(x) > 0.45 or k > 4000:
            break

        _act[0, 0] = force / SCALE
        with torch.inference_mode():
            obs, _, _, dones, _ = env.step(_act)
        if bool(dones[0]):
            print(f"[WARN] env 0 reset at step {k}; ramp truncated.")
            break
        k += 1

    # write the full trajectory so it can be inspected / plotted
    csv_path = os.path.join(OUT_DIR, CSV_PATH)
    with open(csv_path, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["time", "torque", "cart_vel", "cart_pos", "applied_force", "pole_angle", "pole_vel"])
        for i in range(len(ts)):
            wtr.writerow(
                [
                    f"{ts[i]:.5f}",
                    f"{taus[i]:.6f}",
                    f"{vs[i]:.6f}",
                    f"{xs[i]:.6f}",
                    f"{applied[i]:.6f}",
                    f"{pas[i]:.6f}",
                    f"{pvs[i]:.6f}",
                ]
            )

    # NOISY trajectory: sensor values the POLICY sees (cart/pole noise). torque = commanded ref.
    csv_path_noisy = os.path.join(OUT_DIR, CSV_PATH_NOISY)
    if noisy_ok:
        with open(csv_path_noisy, "w", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(["time", "torque", "cart_vel", "cart_pos", "pole_angle", "pole_vel"])
            for i in range(len(n_vs)):
                wtr.writerow(
                    [
                        f"{ts[i]:.5f}",
                        f"{taus[i]:.6f}",
                        f"{n_vs[i]:.6f}",
                        f"{n_xs[i]:.6f}",
                        f"{n_pas[i]:.6f}",
                        f"{n_pvs[i]:.6f}",
                    ]
                )

    print("=" * 78)
    if breakaway is None:
        print("RESULT (play torque): NO breakaway detected -- static friction too high, or force not applied.")
        print(f"  max commanded tau={taus[-1]:.5f} N*m,  max applied force={max(applied):.3f} N")
    else:
        tb, taub, kb = breakaway
        print(f"RESULT (play torque): breakaway tau = {taub:.5f} N*m   (rig ~{RIG_BREAKAWAY})")
        # post-breakaway velocities at fixed offsets, to compare the acceleration to the rig
        t = np.array(ts)
        v = np.abs(np.array(vs))
        print(f"  post-breakaway velocity (rig: ~{RIG_V_AT_021:.3f} m/s at +0.21 s):")
        for off in (0.10, 0.21, 0.30, 0.50):
            j = int(np.argmin(np.abs(t - (tb + off))))
            if t[j] <= tb + off + dt_ctrl:
                fj = taus[j] / R_PULLEY
                print(f"     +{off:.2f}s : v={v[j]:.4f} m/s   (F_applied~{fj:.3f} N)")
        # mean accel over the first 0.2 s after breakaway
        seg = (t >= tb) & (t <= tb + 0.20)
        if seg.sum() >= 3:
            a_mean = (v[seg][-1] - v[seg][0]) / (t[seg][-1] - t[seg][0])
            print(f"  mean accel over first 0.20 s after breakaway = {a_mean:.3f} m/s^2")
        print()
        print("  Interpretation:")
        print("   - breakaway ~0.018 N*m           -> static friction is live on the GPU path.")
        print("   - v(+0.21s) ~0.13 m/s (like rig) -> effective mass (body + armature) + dynamic friction correct.")
        print("   - if it accelerates much faster  -> armature likely NOT applied (check read-back ~0.15).")
    pole_swing = float(np.max(np.abs(np.array(pas) - pas[0]))) if pas else float("nan")
    print(f"  pole swing during ramp = {pole_swing:.4f} rad   (want ~0; large => pole perturbing the cart)")
    print(f"  CSV -> {csv_path}")
    if noisy_ok:
        inj_v = np.array(n_vs) - np.array(vs)
        inj_x = np.array(n_xs) - np.array(xs)
        print(f"  NOISY CSV -> {csv_path_noisy}")
        print(f"  injected cart_vel noise std = {inj_v.std():.4f} m/s   (model ~0.012; rig standstill ~0.003-0.005)")
        print(f"  injected cart_pos noise std = {inj_x.std():.2e} m     (model ~6e-6 m)")
        print("  the pre-breakaway (standstill) segment is the cleanest read of the cart_vel noise floor.")
    print("=" * 78)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
