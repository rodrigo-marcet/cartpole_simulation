# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coast-down probe: seed the cart with an initial velocity, feed zero action, and fit the
deceleration in play's env. ~3.57 m/s^2 (the rig target) => dynamic friction + drivetrain armature
are live on the GPU path. Writes coast_down.csv (+ _noisy: policy-observed sensor values).
Run with the SAME --task / --num_envs you train with.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Coast-down (dynamic friction) probe inside play's env.")
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
V0 = 2.1  # initial coast velocity [m/s] (matches simulations/friction.py)
X0 = -0.4  # start pos [m]; +V0 -> coasts toward + within the +/-0.5 joint limit
POLE_ANGLE = 0.0  # where to seed the pole. Use 0.0 when the pole joint is LOCKED (USDA limits
# (0,0)): seeding it at pi against a (0,0) limit makes PhysX violently yank the
# pole and wreck the coast. Use math.pi only if the pole limits are WIDE (hangs).
RIG_DECEL = 3.57  # m/s^2 target (rig pooled coast mean)
CSV_PATH = "coast_down.csv"
CSV_PATH_NOISY = "coast_down_noisy.csv"  # policy-observed sensor values


def _out_dir(name):
    """scripts/simulations/single_cartpole/output/<name>/ (created), anchored to the repo, not cwd."""
    d = os.path.dirname(os.path.abspath(__file__))
    while os.path.basename(d) != "single_cartpole" and os.path.dirname(d) != d:
        d = os.path.dirname(d)
    out = os.path.join(d, "output", name)
    os.makedirs(out, exist_ok=True)
    return out


OUT_DIR = _out_dir("coast_down")

if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


def _freeze_dr(env_cfg):
    """Collapse every randomized slider term to its nominal so every env is identical."""
    ev = env_cfg.events
    dyn = ev.randomize_slider_friction.params["dynamic_params"]
    ev.randomize_slider_friction.params["dynamic_params"] = (dyn[0], 0.0)  # keep center, zero std
    ev.randomize_slider_friction.params["static_range"] = (1.8, 1.8)  # calibrated breakaway nominal
    arm = ev.randomize_slider_armature.params["armature_distribution_params"]
    ev.randomize_slider_armature.params["armature_distribution_params"] = (arm[0], 0.0)
    # room to coast: play terminates the episode at +/-0.15 m; widen so it doesn't reset mid-coast.
    env_cfg.terminations.cart_out_of_bounds.params["bounds"] = (-1.0, 1.0)
    env_cfg.episode_length_s = 30.0
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

    # log_dir parity only (policy is NOT loaded -- this is an open-loop physics probe)
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
    print("COAST-DOWN (dynamic friction + armature)")
    print(f"  num_envs={num_envs}  device={dev}  step_dt={dt_ctrl:.4f}s  V0={V0} m/s")
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

    # seed env 0: coast from V0, pole hanging at pi (stable) so it barely perturbs the cart
    jp = _rb.data.joint_pos[0:1].clone()
    jv = _rb.data.joint_vel[0:1].clone()
    jp[0, _sidx] = X0
    jp[0, _pidx] = POLE_ANGLE
    jv[0, _sidx] = V0
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

    ts, vsig, xs, pas, pvs = [], [], [], [], []
    n_xs, n_vsig, n_pas, n_pvs = [], [], [], []  # NOISY sensor logs (what the policy observes)
    _zero = torch.zeros((num_envs, 1), dtype=torch.float32, device=dev)
    k = 0
    while True:
        v = _rb.data.joint_vel[0, _sidx].item()
        x = _rb.data.joint_pos[0, _sidx].item()
        pa = _rb.data.joint_pos[0, _pidx].item()
        pv = _rb.data.joint_vel[0, _pidx].item()
        ts.append((k + 1) * dt_ctrl)  # +1 so the first sample is t=dt (0.01s), matching the rig CSVs
        vsig.append(v)
        xs.append(x)
        pas.append(pa)
        pvs.append(pv)
        if noisy_ok:
            _o = env.unwrapped.observation_manager.compute()["policy"][0]
            n_xs.append(float(_o[0]))
            n_vsig.append(float(_o[1]))
            n_pas.append(math.atan2(float(_o[2]), float(_o[3])))
            n_pvs.append(float(_o[4]))
        if abs(v) < 0.05 or abs(x) > 0.45 or k > 800:
            break
        with torch.inference_mode():
            obs, _, _, dones, _ = env.step(_zero)
        if bool(dones[0]):
            print(f"[WARN] env 0 reset at step {k}; coast truncated (widen bounds / lower V0).")
            break
        k += 1

    # write the full trajectory so it can be inspected / plotted
    csv_path = os.path.join(OUT_DIR, CSV_PATH)
    with open(csv_path, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["time", "cart_pos", "cart_vel", "pole_angle", "pole_vel"])
        for i in range(len(ts)):
            wtr.writerow([f"{ts[i]:.5f}", f"{xs[i]:.6f}", f"{vsig[i]:.6f}", f"{pas[i]:.6f}", f"{pvs[i]:.6f}"])

    # NOISY trajectory: same columns, values the POLICY sees (cart/pole sensor noise applied).
    csv_path_noisy = os.path.join(OUT_DIR, CSV_PATH_NOISY)
    if noisy_ok:
        with open(csv_path_noisy, "w", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(["time", "cart_pos", "cart_vel", "pole_angle", "pole_vel"])
            for i in range(len(n_vsig)):
                wtr.writerow(
                    [f"{ts[i]:.5f}", f"{n_xs[i]:.6f}", f"{n_vsig[i]:.6f}", f"{n_pas[i]:.6f}", f"{n_pvs[i]:.6f}"]
                )

    t = np.array(ts)
    v = np.abs(np.array(vsig))
    v0 = float(v[0])
    mask = v > 0.15 * v0
    mask[0] = False  # drop the release transient
    if mask.sum() >= 4:
        slope, intc = np.polyfit(t[mask], v[mask], 1)
        pred = slope * t[mask] + intc
        ss_res = float(np.sum((v[mask] - pred) ** 2))
        ss_tot = float(np.sum((v[mask] - v[mask].mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        decel = -float(slope)
    else:
        decel, r2 = float("nan"), float("nan")

    # pole-motion diagnostic: a clean coast needs the pole ~still. Big swing => contaminated.
    pole_swing = float(np.max(np.abs(np.array(pas) - pas[0]))) if pas else float("nan")

    print("=" * 78)
    print(f"RESULT (play coast) : decel = {decel:.3f} m/s^2   R^2 = {r2:.4f}")
    print(f"  rig target ~{RIG_DECEL:.2f} m/s^2   |   samples={len(ts)}  V0={v0:.3f}  travel={xs[-1] - xs[0]:+.3f} m")
    print(f"  pole swing during coast = {pole_swing:.4f} rad   (want ~0; large => pole is perturbing the cart)")
    print(f"  CSV -> {csv_path}")
    if noisy_ok:
        inj_v = np.array(n_vsig) - np.array(vsig)
        inj_x = np.array(n_xs) - np.array(xs)
        print(f"  NOISY CSV -> {csv_path_noisy}")
        print(f"  injected cart_vel noise std = {inj_v.std():.4f} m/s   (model ~0.012; rig coast ~0.003-0.024)")
        print(f"  injected cart_pos noise std = {inj_x.std():.2e} m     (model ~6e-6 m)")
        print("  overlay play_friction_coast_noisy.csv 'cart_vel' on the rig coastdown to judge the jitter.")
    if not math.isnan(r2) and r2 < 0.95:
        print("  [!] R^2 < 0.95 -> the coast is NOT a clean line, the number is unreliable.")
        print("      Check pole_swing / the CSV: pole yank (bad lock+seed) or joint-limit contact.")
    elif not math.isnan(decel):
        if 3.3 <= decel <= 3.8:
            print("  => MATCH: play's dynamic friction + armature are LIVE on the GPU path. ✓")
        elif decel > 4.5:
            print("  => decel too HIGH: friction ok but armature likely NOT in the effective mass")
            print("     (decel ~= F_dyn/0.33 instead of F_dyn/0.48; check the armature read-back).")
        else:
            print("  => off target: recheck the frozen DR values / armature.")
    print("=" * 78)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
