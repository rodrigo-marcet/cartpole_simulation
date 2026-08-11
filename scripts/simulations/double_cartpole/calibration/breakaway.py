# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Breakaway probe on the real double pendulum (Swingup-DoubleCartpole-v0, double_pendulum.usda): ramp the
cart force until it starts moving; the torque at lift-off is the slider static friction. Reproduces the value
measured on the rig (6-run avg 0.0324 N*m) with SLIDER_STATIC = 0.0324 / R_PULLEY = 3.24 N.

USDA setup (revert the free-swing freeze in double_pendulum_physics.usda before running): slider_to_cart must
be FREE (uncomment the -0.5/0.5 limits) and cart_to_ipole FREE (back to wide); ipole_to_opole is already free.
The probe seeds both poles hanging (inner=pi, outer=0) and lets them rotate -- the breakaway happens in the
first ~7 s while the cart is still near rest, so the free poles don't perturb it. STATIC ONLY: dynamic/viscous
are still the single-pole values, so the breakaway POINT matches the rig but the post-breakaway acceleration
will not until those are recalibrated separately. Writes the canonical time_s,torque_nm,cart_vel_mps so it
drops straight into analysis/double_pole/calibration/breakaway/data/sim/.
Run:  python breakaway.py --task Swingup-DoubleCartpole-v0 --num_envs 1 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Breakaway (static friction) probe on the double pendulum.")
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
RAMP = 0.005  # torque ramp rate [N*m/s] (matches the rig breakaway ~0.0051)
R_PULLEY = 0.01  # m  (force = torque / r; same cart hardware as the single cartpole)
SCALE = 40.0  # JointEffortActionCfg scale in the double env (MUST match ActionsCfg).
X0 = -0.3  # cart start [m]; creeps toward + within the freed slider limit
INNER_HANG = math.pi  # cart_to_ipole seed [rad]; inner straight down (0 = upright). Poles left free.
OUTER_HANG = 0.0  # ipole_to_opole seed [rad]; outer aligned below the inner (0 = hanging)
SLIDER_STATIC = 3.24  # slider static friction [N] = measured 0.0324 N*m breakaway / R_PULLEY  <-- calculated
WEIGHT_MASS = 0.05  # restore real weight_1 mass [kg] (env DR zeroes it for the free-swing test)
RIG_BREAKAWAY = 0.0324  # N*m measured on the real double rig (6-run avg)
CSV_PATH = "breakaway.csv"


def _out_dir(name):
    """scripts/simulations/double_cartpole/output/<name>/ (created), anchored to the repo, not cwd."""
    d = os.path.dirname(os.path.abspath(__file__))
    while os.path.basename(d) != "double_cartpole" and os.path.dirname(d) != d:
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
    """Pin the slider friction to the calibrated static value, freeze the rest of the DR to nominal, and drop
    the random cart reset so the probe controls x0. Static only: dynamic/viscous keep their single-pole centers."""
    ev = env_cfg.events
    fr = ev.randomize_slider_friction.params
    dyn = fr["dynamic_params"]
    fr["dynamic_params"] = (dyn[0], 0.0)  # keep single-pole dynamic center, zero std
    fr["static_range"] = (SLIDER_STATIC, SLIDER_STATIC)  # calibrated double breakaway
    arm = ev.randomize_slider_armature.params["armature_distribution_params"]
    ev.randomize_slider_armature.params["armature_distribution_params"] = (arm[0], 0.0)
    ev.randomize_weight_mass.params["mass_distribution_params"] = (WEIGHT_MASS, 0.0)  # real weight, deterministic
    for t in (
        "randomize_cart_mass",
        "randomize_ishaft_mass",
        "randomize_ipole_mass",
        "randomize_oshaft_mass",
        "randomize_opole_mass",
    ):
        try:
            p = getattr(ev, t).params["mass_distribution_params"]
            getattr(ev, t).params["mass_distribution_params"] = (p[0], 0.0)
        except Exception as e:  # noqa: BLE001
            print(f"[freeze_dr] skip {t}: {e}")
    try:
        ev.reset_cart_position = None  # probe seeds x0; inner reset to pi is harmless
    except Exception as e:  # noqa: BLE001
        print(f"[freeze_dr] reset_cart_position: {e}")
    env_cfg.terminations.cart_out_of_bounds.params["bounds"] = (-1.0, 1.0)  # don't reset mid-ramp
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), 30.0)  # ramp reaches breakaway ~6.5 s
    return fr, ev.randomize_slider_armature.params


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

    env.reset()
    dev = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    _rb = env.unwrapped.scene["robot"]
    _view = _rb.root_physx_view
    _sidx = _rb.find_joints("slider_to_cart")[0][0]
    _iidx = _rb.find_joints("cart_to_ipole")[0][0]  # inner: seeded hanging, free
    _oidx = _rb.find_joints("ipole_to_opole")[0][0]  # outer: seeded hanging, free
    dt_ctrl = getattr(env.unwrapped, "step_dt", env.unwrapped.physics_dt * env_cfg.decimation)

    print("=" * 78)
    print("BREAKAWAY (static friction) -- double pendulum")
    print(f"  num_envs={num_envs}  device={dev}  step_dt={dt_ctrl:.4f}s  ramp={RAMP} N*m/s  scale={SCALE}")
    print(f"  slider static -> {SLIDER_STATIC} N (= {SLIDER_STATIC * R_PULLEY:.4f} N*m)   [rig {RIG_BREAKAWAY} N*m]")
    print(
        f"  frozen DR -> dynamic={fr_params['dynamic_params']}  static={fr_params['static_range']}  "
        f"viscous={fr_params.get('viscous_value')}  armature={arm_params['armature_distribution_params']}"
    )
    try:
        print(
            "  env0 friction_props :", _view.get_dof_friction_properties()[0].tolist(), "(may lie; behavior is truth)"
        )
    except Exception as e:  # noqa: BLE001
        print("  read-back unavailable:", e)
    print("=" * 78)

    # seed env 0 at rest: cart at x0, both poles hanging (free thereafter)
    jp = _rb.data.joint_pos[0:1].clone()
    jv = _rb.data.joint_vel[0:1].clone()
    jp[0, _sidx] = X0
    jp[0, _iidx] = INNER_HANG
    jp[0, _oidx] = OUTER_HANG
    jv[0, _sidx] = 0.0
    jv[0, _iidx] = 0.0
    jv[0, _oidx] = 0.0
    _rb.write_joint_state_to_sim(jp, jv, env_ids=torch.tensor([0], device=dev))
    _rb.update(env.unwrapped.physics_dt)

    ts, taus, vs, xs, ipas, opas = [], [], [], [], [], []
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
        ipas.append(_rb.data.joint_pos[0, _iidx].item())
        opas.append(_rb.data.joint_pos[0, _oidx].item())

        if breakaway is None and abs(v) > 0.05:
            breakaway = (t, tau, k)
            print(f"  >> BREAKAWAY at t={t:.3f}s  tau={tau:.5f} N*m  (F={force:.3f} N)   [rig ~{RIG_BREAKAWAY}]")

        if (breakaway is not None and abs(v) > 0.5) or abs(x) > 0.45 or k > 4000:
            break

        _act[0, 0] = force / SCALE
        with torch.inference_mode():
            _, _, _, dones, _ = env.step(_act)
        if bool(dones[0]):
            print(f"[WARN] env 0 reset at step {k}; ramp truncated.")
            break
        k += 1

    # canonical columns -> overlay directly on the rig ground truth
    csv_path = os.path.join(OUT_DIR, CSV_PATH)
    with open(csv_path, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["time_s", "torque_nm", "cart_vel_mps"])
        for i in range(len(ts)):
            wtr.writerow([f"{ts[i]:.5f}", f"{taus[i]:.6f}", f"{vs[i]:.6f}"])

    print("=" * 78)
    if breakaway is None:
        print("RESULT: NO breakaway detected -- static friction too high, or force not applied.")
        print(f"  max commanded tau={taus[-1]:.5f} N*m")
    else:
        tb, taub, kb = breakaway
        print(f"RESULT: breakaway tau = {taub:.5f} N*m   (rig {RIG_BREAKAWAY};  set static = {SLIDER_STATIC} N)")
        t = np.array(ts)
        v = np.abs(np.array(vs))
        print("  post-breakaway velocity (STATIC-ONLY: accel not matched to the rig yet):")
        for off in (0.10, 0.21, 0.30, 0.50):
            j = int(np.argmin(np.abs(t - (tb + off))))
            if t[j] <= tb + off + dt_ctrl:
                print(f"     +{off:.2f}s : v={v[j]:.4f} m/s")
        seg = (t >= tb) & (t <= tb + 0.20)
        if seg.sum() >= 3:
            a_mean = (v[seg][-1] - v[seg][0]) / (t[seg][-1] - t[seg][0])
            print(f"  mean accel over first 0.20 s after breakaway = {a_mean:.3f} m/s^2")
    iswing = float(np.max(np.abs(np.array(ipas) - ipas[0]))) if ipas else float("nan")
    oswing = float(np.max(np.abs(np.array(opas) - opas[0]))) if opas else float("nan")
    print(f"  pole swing during ramp: inner={iswing:.4f} rad  outer={oswing:.4f} rad  (want ~0 before breakaway)")
    print(f"  CSV -> {csv_path}")
    print("  copy into analysis/double_pole/calibration/breakaway/data/sim/001.csv and re-run the analysis plot.")
    print("=" * 78)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
