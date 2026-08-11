# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Free-swing OUTER-pole VALIDATION on the real double pendulum (Swingup-DoubleCartpole-v0, double_pendulum.usda).

All physics (friction, damping, inertia, masses) comes straight from the env cfg + USDA -- the exact values
training uses -- so this run validates them against the outer bench swing. The ONLY value this script forces is
weight_1 mass = 0.0 (the outer bench had no weight). Freeze slider_to_cart and cart_to_ipole by hand in the USDA
(inner straight down) and set INNER_HOLD/CART_HOLD below to match. The probe seeds the outer at POLE_ANGLE0 off the
hanging bottom, feeds zero action, and logs outer angle/vel (0 = hanging bottom, like the bench encoder). On startup
it prints the EFFECTIVE pole friction/damping + swing inertia read back from PhysX, to confirm the cfg is applied.
Run:  python outer_free_swing.py --task Swingup-DoubleCartpole-v0 --num_envs 1 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Free-swing (outer pole) validation on the double pendulum.")
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
# THINGS TO TUNE  (release IC + the by-hand USDA holds; ALL physics comes from the cfg/USDA)
# ============================================================================
POLE_ANGLE0 = 2.535670  # outer release angle (rad off the hanging bottom; 0 = bottom, like the bench encoder)
POLE_VEL0 = -0.151502  # outer release angular velocity (rad/s)
DURATION = 15.0  # s to simulate
INNER_HOLD = -math.pi  # cart_to_ipole frozen angle (rad); MATCH your USDA limit (inner straight down)
CART_HOLD = 0.0  # slider_to_cart frozen position (m); MATCH your USDA limit
CSV_PATH = "outer_free_swing.csv"


def _out_dir(name):
    """scripts/simulations/double_cartpole/output/<name>/ (created), anchored to the repo, not cwd."""
    d = os.path.dirname(os.path.abspath(__file__))
    while os.path.basename(d) != "double_cartpole" and os.path.dirname(d) != d:
        d = os.path.dirname(d)
    out = os.path.join(d, "output", name)
    os.makedirs(out, exist_ok=True)
    return out


OUT_DIR = _out_dir("free_swing")


def _quat_to_R(q):
    """Rotation matrix from a wxyz quaternion (numpy)."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _diagnose_inertia(rb, view):
    """Effective swing inertia about ipole_to_opole from the loaded USD: per swinging body the CoM perpendicular
    distance to the joint axis and its I_com_axis + m*d_perp^2. weight_1 is forced to 0 here so it reads ~0."""
    masses = view.get_masses()[0].cpu().numpy()
    coms = view.get_coms()[0][:, 0:3].cpu().numpy()
    iner = view.get_inertias()[0].cpu().numpy().reshape(-1, 3, 3)
    pos_w = rb.data.body_pos_w[0].cpu().numpy()
    quat_w = rb.data.body_quat_w[0].cpu().numpy()
    bi = {n: rb.find_bodies(n)[0][0] for n in ("oshaft_1", "opole_1", "weight_1")}
    pivot = pos_w[bi["oshaft_1"]]  # ipole_to_opole sits at oshaft_1's origin
    axis = _quat_to_R(quat_w[bi["oshaft_1"]]) @ np.array([0.0, 1.0, 0.0])
    axis = axis / np.linalg.norm(axis)
    print("  --- effective swing inertia about ipole_to_opole (from loaded USD) ---")
    total = 0.0
    for name, b in bi.items():
        R = _quat_to_R(quat_w[b])
        off = (pos_w[b] + R @ coms[b]) - pivot
        d_perp = float(np.linalg.norm(off - np.dot(off, axis) * axis))
        i_axis = float(axis @ (R @ iner[b] @ R.T) @ axis)
        contrib = i_axis + masses[b] * d_perp**2
        total += contrib
        print(
            f"    {name:11s} m={masses[b]:.4f}  d_perp={d_perp:.4f} m  I_com_axis={i_axis:.3e}  contrib={contrib:.3e}"
        )
    print(f"    TOTAL effective I_pivot = {total:.3e}   (target 1.34e-4)")


def _print_effective(rb, view):
    """Effective per-joint pole friction/damping read back from PhysX -- confirms the cfg values are applied."""
    ii = rb.find_joints("cart_to_ipole")[0][0]
    oi = rb.find_joints("ipole_to_opole")[0][0]
    print("  --- effective pole friction/damping read back (env 0) ---")
    try:
        fp = view.get_dof_friction_properties()[0]
        print(f"    joint friction  inner(cart_to_ipole)={fp[ii].tolist()}   outer(ipole_to_opole)={fp[oi].tolist()}")
    except Exception as e:  # noqa: BLE001
        print(f"    joint friction read failed: {e}")
    got = False
    for meth in ("get_dof_dampings", "get_dof_damping_coefficients"):
        if hasattr(view, meth):
            try:
                d = getattr(view, meth)()[0]
                print(f"    joint damping   inner={d[ii].item():.3e}   outer={d[oi].item():.3e}")
                got = True
                break
            except Exception as e:  # noqa: BLE001
                print(f"    {meth} failed: {e}")
    if not got:
        try:
            print(f"    pole_actuator damping (per-joint) = {rb.actuators['pole_actuator'].damping[0].tolist()}")
        except Exception as e:  # noqa: BLE001
            print(f"    damping read failed: {e}")


if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


def _zero_std(term, key):
    """Collapse a (center, std) distribution-params tuple to (center, 0.0) so env 0 uses the cfg center."""
    p = term.params[key]
    term.params[key] = (p[0], 0.0)


def _freeze_dr(env_cfg):
    """Zero-std every DR term so env 0 uses the cfg CENTERS (the training values). The ONLY forced value is
    weight_1 mass = 0.0 (the outer bench had no weight). Drop the random resets and mirror the holds into
    init_state so Isaac's construction-time limit check passes."""
    ev = env_cfg.events
    for term_name, key in [
        ("randomize_inner_pole_friction", "friction_distribution_params"),
        ("randomize_outer_pole_friction", "friction_distribution_params"),
        ("randomize_inner_pole_damping", "damping_distribution_params"),
        ("randomize_outer_pole_damping", "damping_distribution_params"),
        ("randomize_slider_armature", "armature_distribution_params"),
        ("randomize_cart_mass", "mass_distribution_params"),
        ("randomize_ishaft_mass", "mass_distribution_params"),
        ("randomize_ipole_mass", "mass_distribution_params"),
        ("randomize_oshaft_mass", "mass_distribution_params"),
        ("randomize_opole_mass", "mass_distribution_params"),
    ]:
        try:
            _zero_std(getattr(ev, term_name), key)
        except Exception as e:  # noqa: BLE001
            print(f"[freeze_dr] skip {term_name}.{key}: {e}")
    ev.randomize_weight_mass.params["mass_distribution_params"] = (0.0, 0.0)  # forced: outer bench had no weight
    for term_name in ("reset_cart_position", "reset_ipole_position", "reset_pole_position"):
        try:
            setattr(ev, term_name, None)
        except Exception as e:  # noqa: BLE001
            print(f"[freeze_dr] skip {term_name}: {e}")
    try:
        env_cfg.terminations.cart_out_of_bounds.params["bounds"] = (-1.0, 1.0)
    except Exception as e:  # noqa: BLE001
        print(f"[freeze_dr] cart bounds: {e}")
    try:
        env_cfg.scene.robot.init_state.joint_pos["cart_to_ipole"] = INNER_HOLD
        env_cfg.scene.robot.init_state.joint_pos["slider_to_cart"] = CART_HOLD
    except Exception as e:  # noqa: BLE001
        print(f"[freeze_dr] init_state: {e}")
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), DURATION + 5.0)
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

    _freeze_dr(env_cfg)
    fr = env_cfg.events.randomize_outer_pole_friction.params["friction_distribution_params"][0]
    dmp = env_cfg.events.randomize_outer_pole_damping.params["damping_distribution_params"][0]

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
    _iidx = _rb.find_joints("cart_to_ipole")[0][0]  # inner: frozen by hand in the USD
    _oidx = _rb.find_joints("ipole_to_opole")[0][0]  # outer: the swinging joint we log
    dt_ctrl = getattr(env.unwrapped, "step_dt", env.unwrapped.physics_dt * env_cfg.decimation)

    bottom = (math.pi - INNER_HOLD) % (2 * math.pi)  # outer raw angle at the hanging bottom

    print("=" * 78)
    print("FREE-SWING (outer pole on the double pendulum) -- validating the cfg/USDA values")
    print(f" angle0={POLE_ANGLE0:.4f} rad ({math.degrees(POLE_ANGLE0):.1f} deg off bottom)  vel0={POLE_VEL0:.4f} rad/s")
    print(f"  inner frozen at {INNER_HOLD:.4f} rad  cart at {CART_HOLD:.4f} m  -> outer bottom at raw {bottom:.4f} rad")
    _print_effective(_rb, _view)
    _diagnose_inertia(_rb, _view)
    print(f"  outer friction(cfg)={fr}  damping(cfg)={dmp}  duration={DURATION}s")
    print(f"  num_envs={num_envs}  device={dev}  step_dt={dt_ctrl:.4f}s")
    print("=" * 78)

    # seed env 0 to the frozen hold + the outer release IC
    _eid = torch.tensor([0], device=dev)
    jp = _rb.data.joint_pos[0:1].clone()
    jv = _rb.data.joint_vel[0:1].clone()
    jp[0, _sidx] = CART_HOLD
    jp[0, _iidx] = INNER_HOLD
    jp[0, _oidx] = bottom + POLE_ANGLE0
    jv[0, _sidx] = 0.0
    jv[0, _iidx] = 0.0
    jv[0, _oidx] = POLE_VEL0
    _rb.write_joint_state_to_sim(jp, jv, env_ids=_eid)
    _rb.update(env.unwrapped.physics_dt)

    _zero = torch.zeros((num_envs, 1), dtype=torch.float32, device=dev)
    max_steps = int(DURATION / dt_ctrl) + 5
    ts, coss, sins, angs, avs, cxs = [], [], [], [], [], []
    k = 0
    while True:
        phi = _rb.data.joint_pos[0, _oidx].item() - bottom  # deviation off the hanging bottom (0 at rest)
        w = _rb.data.joint_vel[0, _oidx].item()
        cx = _rb.data.joint_pos[0, _sidx].item()
        t = k * dt_ctrl
        c, s = math.cos(phi), math.sin(phi)
        ts.append(t)
        coss.append(c)
        sins.append(s)
        angs.append(math.atan2(s, c) % (2 * math.pi))  # 0 = hanging bottom, like the outer bench encoder
        avs.append(w)
        cxs.append(cx)
        if t >= DURATION or k > max_steps:
            break
        with torch.inference_mode():
            _, _, _, dones, _ = env.step(_zero)  # USD limits hold the inner + cart; no teleport, no injection
        if bool(dones[0]):
            print(f"[WARN] env 0 reset at step {k} (t={t:.2f}s); swing truncated.")
            break
        k += 1

    # ONE CSV, columns match the bench CSVs -> overlay directly on the outer ground truth.
    csv_path = os.path.join(OUT_DIR, CSV_PATH)
    with open(csv_path, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["time_s", "pole_cos", "pole_sin", "pole_angle_rad", "pole_vel_radps"])
        for i in range(len(ts)):
            wtr.writerow([f"{ts[i]:.6f}", f"{coss[i]:.6f}", f"{sins[i]:.6f}", f"{angs[i]:.6f}", f"{avs[i]:.6f}"])

    pole_sweep = float(np.max(angs) - np.min(angs)) if angs else float("nan")
    cart_travel = float(np.max(cxs) - np.min(cxs)) if cxs else float("nan")
    print("=" * 78)
    print(f"RESULT  samples={len(ts)}  duration={ts[-1]:.2f}s")
    print(f"  outer sweep = {pole_sweep:.3f} rad   cart travel = {cart_travel:.4f} m (should be ~0, cart locked)")
    if pole_sweep < 0.2:
        print("  [!] outer barely moved -> ipole_to_opole may be locked/over-damped, or IC not applied.")
    print(f"  CSV -> {csv_path}")
    print("  overlay on the outer bench ground truth (same columns) and eyeball the decay.")
    print("=" * 78)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
