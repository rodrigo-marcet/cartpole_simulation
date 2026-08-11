# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sim2real one-step residual: teacher-force each rig-log row into play's env, step with the logged
force, and compare the sim's TRUE next state to the rig's next state. A per-dim bias that survives
averaging = a real dynamics gap; a large negative cart_vel residual at high |v| means the DCMotor
velocity_limit under-drives. Force is always flipped (NN-output -> sensor frame, mirroring the rig).
Writes replay_onestep.csv and replay_rollout.csv. Pole angle 0 = upright, +/-pi = hanging.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sim2real one-step residual probe inside play's env.")
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
parser.add_argument(
    "--logs", type=str, default=None, help="Rig CSV from logs_to_csv.py (default: simulations/inputs/logs.csv)."
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
CSV_ONESTEP = "onestep.csv"
CSV_ROLLOUT = "rollout.csv"


def _out_dir(name):
    """scripts/simulations/single_cartpole/output/<name>/ (created), anchored to the repo, not cwd."""
    d = os.path.dirname(os.path.abspath(__file__))
    while os.path.basename(d) != "single_cartpole" and os.path.dirname(d) != d:
        d = os.path.dirname(d)
    out = os.path.join(d, "output", name)
    os.makedirs(out, exist_ok=True)
    return out


def _in_dir():
    """scripts/simulations/inputs/ , anchored to the repo, not cwd."""
    d = os.path.dirname(os.path.abspath(__file__))
    while os.path.basename(d) != "single_cartpole" and os.path.dirname(d) != d:
        d = os.path.dirname(d)
    return os.path.join(d, "inputs")


OUT_DIR = _out_dir("replay")
SKIP_FIRST = 1  # transition 0 is a finite-diff startup artifact (pole_vel=-49 at rest) -> drop it
DIM_NAMES = ["cart_pos", "cart_vel", "pole_ang", "pole_vel"]
DIM_UNITS = ["m", "m/s", "rad", "rad/s"]
# rough per-step sensor noise floor of the RIG log (for interpreting residual bias vs noise)
NOISE_FLOOR = {"cart_pos": 5e-6, "cart_vel": 0.011, "pole_ang": 1e-3, "pole_vel": 0.57}


if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


def _wrap(a):
    """Wrap an angle (or array) to [-pi, pi]."""
    return np.arctan2(np.sin(a), np.cos(a))


def _zero_std(term, key):
    """Collapse a (center, std) distribution-params tuple to (center, 0.0) -- keeps the mean."""
    p = term.params[key]
    term.params[key] = (p[0], 0.0)


def _freeze_dr(env_cfg):
    """Freeze every randomized physics term to its configured center so env 0 == the calibrated
    model, and widen bounds / stretch the episode so env.step() never auto-resets mid-replay."""
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
        env_cfg.terminations.cart_out_of_bounds.params["bounds"] = (-2.0, 2.0)
    except Exception as e:  # noqa: BLE001
        print(f"[freeze_dr] could not widen cart bounds: {e}")
    env_cfg.episode_length_s = 1.0e6
    return ev


def _load_logs(path):
    """Load logs.csv -> dict of float arrays (cart_pos, cart_vel, pole_ang, pole_vel, force)."""
    ap = os.path.abspath(path)
    if not os.path.isfile(ap):
        raise FileNotFoundError(
            f"logs CSV not found: {ap}\nRun simulations/inputs/logs_to_csv.py first, or pass --logs <path>."
        )
    cp, cv, pa, pv, fn, dt = [], [], [], [], [], []
    with open(ap) as fh:
        r = csv.DictReader(fh)
        for row in r:
            cp.append(float(row["cart_pos_m"]))
            cv.append(float(row["cart_vel_mps"]))
            pa.append(float(row["pole_angle_rad"]))
            pv.append(float(row["pole_vel_radps"]))
            fn.append(float(row["force_n"]))
            if row.get("dt_s"):
                dt.append(float(row["dt_s"]))
    return {
        "path": ap,
        "cart_pos": np.array(cp),
        "cart_vel": np.array(cv),
        "pole_ang": np.array(pa),
        "pole_vel": np.array(pv),
        "force": np.array(fn),
        "dt_mean": (float(np.mean(dt)) if dt else None),
    }


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

    logs_path = args_cli.logs or os.path.join(_in_dir(), "logs.csv")
    rig = _load_logs(logs_path)
    N = len(rig["force"])

    # ALWAYS flip force_n from the NN-output frame to the sensor frame (see module docstring).
    rig["force"] = -rig["force"]

    action_scale = float(getattr(env_cfg.actions.joint_effort, "scale", 30.0))

    # match sim.dt to the rig log's mean loop dt so step_dt == the real transition
    if rig.get("dt_mean"):
        env_cfg.sim.dt = rig["dt_mean"] / env_cfg.decimation
        print(
            f"[dt-match] rig mean dt = {rig['dt_mean'] * 1000:.2f} ms -> sim.dt = {env_cfg.sim.dt * 1e6:.0f} us "
            f"x{env_cfg.decimation} = {env_cfg.decimation * env_cfg.sim.dt * 1000:.2f} ms/step"
        )

    _freeze_dr(env_cfg)

    # log_dir parity only (no policy loaded -- this is an open-loop physics probe)
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
    dt_ctrl = getattr(env.unwrapped, "step_dt", env.unwrapped.physics_dt * env_cfg.decimation)
    e0 = torch.tensor([0], device=dev)

    print("=" * 78)
    print("REPLAY (sim2real one-step residual)")
    print(f"  logs: {rig['path']}  ({N} rows, {N * dt_ctrl:.2f} s @ {1 / dt_ctrl:.0f} Hz)")
    print(f"  num_envs={num_envs}  device={dev}  step_dt={dt_ctrl:.4f}s  action_scale={action_scale} N")
    print(
        f"  force sign: FLIPPED (NN-output -> sensor frame; standard).  "
        f"rig |cart_v|max={np.max(np.abs(rig['cart_vel'])):.2f} m/s"
    )
    print(f"  skipping transition 0 (startup artifact); {N - 1 - SKIP_FIRST} transitions used")
    print("=" * 78)

    def set_state(i):
        """Hard-set env 0 to rig row i. inference_mode required: env.step() marks the data buffers
        as inference tensors, so the in-place joint-state write must run inside it too."""
        with torch.inference_mode():
            jp = _rb.data.joint_pos[0:1].clone()
            jv = _rb.data.joint_vel[0:1].clone()
            jp[0, _sidx] = float(rig["cart_pos"][i])
            jp[0, _pidx] = float(rig["pole_ang"][i])
            jv[0, _sidx] = float(rig["cart_vel"][i])
            jv[0, _pidx] = float(rig["pole_vel"][i])
            _rb.write_joint_state_to_sim(jp, jv, env_ids=e0)
            _rb.update(env.unwrapped.physics_dt)

    def step_force(force_n):
        """Apply force_n [N] to the slider for one control step; return sim's TRUE next state."""
        act = torch.zeros((num_envs, 1), dtype=torch.float32, device=dev)
        act[0, 0] = force_n / action_scale
        with torch.inference_mode():
            _, _, _, dones, _ = env.step(act)
        return (
            float(_rb.data.joint_pos[0, _sidx]),
            float(_rb.data.joint_vel[0, _sidx]),
            float(_rb.data.joint_pos[0, _pidx]),
            float(_rb.data.joint_vel[0, _pidx]),
            bool(dones[0]),
        )

    # ---- one-step residual: drive i -> i+1 with force[i], compare sim next-state to rig next ----
    res = []
    applied = []  # (idx, sim_next, rig_next, force_applied, cart_vel_start) for the CSV dump
    for i in range(SKIP_FIRST, N - 1):
        set_state(i)
        fi = rig["force"][i]
        scp, scv, spa, spv, done = step_force(fi)
        rcp, rcv, rpa, rpv = (
            rig["cart_pos"][i + 1],
            rig["cart_vel"][i + 1],
            rig["pole_ang"][i + 1],
            rig["pole_vel"][i + 1],
        )
        res.append([scp - rcp, scv - rcv, float(_wrap(spa - rpa)), spv - rpv])
        applied.append(
            (
                i + 1,
                (scp, scv, float(_wrap(spa)), spv),
                (rcp, rcv, float(rpa), rpv),
                float(fi),
                float(rig["cart_vel"][i]),
            )
        )
        if done:
            print(f"  [WARN] env reset during transition {i} (widen bounds / episode_length).")
    res = np.array(res)

    # ---- one-step residual detail (bias = systematic gap vs the rig noise floor) ----
    mean = np.mean(res, axis=0)
    std = np.std(res, axis=0)
    print("ONE-STEP RESIDUAL (sim_next - rig_next)   [mean = systematic bias]")
    for k, name in enumerate(DIM_NAMES):
        floor = NOISE_FLOOR[name]
        verdict = "significant" if abs(mean[k]) > floor else "within noise floor"
        print(
            f"  {name:>9} [{DIM_UNITS[k]:>5}] : bias={mean[k]:+.5f}  std={std[k]:.5f}  "
            f"(noise floor ~{floor:g}) -> {verdict}"
        )

    # whirl (pole >=20 rad/s) is chaotic; the swing-up/balance regime is what matters for transfer
    print("-" * 78)
    print("BY REGIME (split on pole speed at transition start; whirl is chaotic -> expect big RMSE):")
    pv_start = np.abs(rig["pole_vel"][SKIP_FIRST : N - 1])
    for label, m in [("swing-up/balance |w|<20", pv_start < 20.0), ("runaway whirl  |w|>=20", pv_start >= 20.0)]:
        if m.any():
            mm = np.mean(res[m], axis=0)
            rr = np.sqrt(np.mean(res[m] ** 2, axis=0))
            print(
                f"  {label} : n={int(m.sum()):>3}  "
                f"cart_vel bias={mm[1]:+.3f} rmse={rr[1]:.3f}  |  pole_vel bias={mm[3]:+.2f} rmse={rr[3]:.2f}"
            )

    # cart_vel residual binned by |cart_vel| -> exposes DCMotor velocity_limit derating
    print("-" * 78)
    print("CART_VEL residual binned by |cart_vel| (exposes DCMotor velocity_limit=2.65 derating):")
    cv_at = np.abs(rig["cart_vel"][SKIP_FIRST : N - 1])  # speed at the START of each transition
    edges = [0.0, 0.5, 1.0, 1.5, 2.0, 10.0]
    cv_res = res[:, 1]
    for a, b in zip(edges[:-1], edges[1:]):
        m = (cv_at >= a) & (cv_at < b)
        if m.any():
            print(f"  |v| in [{a:.1f},{b:.1f}) m/s : n={m.sum():>3}  mean cart_vel resid = {cv_res[m].mean():+.4f} m/s")
    print("  (residual trending NEGATIVE as |v| rises => sim under-drives => raise velocity_limit /")
    print("   the real motor's no-load speed is above 2.65 m/s.)")

    # DCMotor derating only weakens the motor when accelerating (F*v>0); split drive vs brake so the
    # signal isn't cancelled. push err = sign(F)*(sim_v - rig_v): +ve => sim harder, -ve => weaker.
    print("-" * 78)
    print("DRIVE vs BRAKE (sign of force*velocity; isolates the DCMotor velocity_limit=2.65 derating):")
    f_app = rig["force"][SKIP_FIRST : N - 1]  # force applied at each transition
    v0 = rig["cart_vel"][SKIP_FIRST : N - 1]  # cart velocity at the start of the step
    fv = f_app * v0
    push = np.sign(f_app) * res[:, 1]  # +: sim stronger, -: sim weaker (under-drove)
    speed = np.abs(v0)
    brk = fv < 0.0
    drv = ~brk
    if brk.any():
        print(
            f"  BRAKING      (F*v<0, derating N/A)  : n={int(brk.sum()):>3}  "
            f"mean push err = {push[brk].mean():+.4f} m/s   (expect ~0 if the motor curve is the issue)"
        )
    print("  ACCELERATING (F*v>0, derating bites) -- by cart speed:")
    for a, b in [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 10.0)]:
        m = drv & (speed >= a) & (speed < b)
        if m.any():
            print(f"    |v| in [{a:.1f},{b:.1f}) m/s : n={int(m.sum()):>3}  mean push err = {push[m].mean():+.4f} m/s")
    print("  Fingerprint: BRAKING ~0 while ACCELERATING goes NEGATIVE and grows with speed =>")
    print("  sim under-drives => velocity_limit too low (raise it). Both similar => not the motor")
    print("  curve (look at cart mass/friction). ACCELERATING positive => velocity_limit too high.")

    # ---- one-step CSV dump ----
    with open(os.path.join(OUT_DIR, CSV_ONESTEP), "w", newline="") as f:
        w = csv.writer(f)
        # force_n applied + cart_vel_start let you plot the residual vs force and vs speed
        cols = ["step", "t_s", "force_n", "cart_vel_start"]
        for name in DIM_NAMES:
            cols += [f"rig_{name}", f"sim_{name}", f"resid_{name}"]
        w.writerow(cols)
        for idx, sim_next, rig_next, force_n, cart_vel_start in applied:
            row = [idx, round(idx * dt_ctrl, 5), round(force_n, 5), round(cart_vel_start, 5)]
            for k in range(4):
                rr = float(_wrap(sim_next[k] - rig_next[k])) if k == 2 else sim_next[k] - rig_next[k]
                row += [rig_next[k], sim_next[k], rr]
            w.writerow(row)

    # ---- open-loop divergence (seed once, replay forces free) ----
    print("=" * 78)
    print(f"OPEN-LOOP DIVERGENCE  (seed at row {SKIP_FIRST}, replay forces free)")
    set_state(SKIP_FIRST)
    roll = []  # (step, rig tuple, sim tuple)
    for i in range(SKIP_FIRST, N - 1):
        scp, scv, spa, spv, done = step_force(rig["force"][i])
        roll.append(
            (
                i + 1,
                (rig["cart_pos"][i + 1], rig["cart_vel"][i + 1], rig["pole_ang"][i + 1], rig["pole_vel"][i + 1]),
                (scp, scv, float(_wrap(spa)), spv),
            )
        )
        if done:
            print(f"  [WARN] env reset during open-loop step {i}.")
            break

    with open(os.path.join(OUT_DIR, CSV_ROLLOUT), "w", newline="") as f:
        w = csv.writer(f)
        cols = ["step", "t_s"]
        for name in DIM_NAMES:
            cols += [f"rig_{name}", f"sim_{name}"]
        w.writerow(cols)
        for idx, rt, st in roll:
            row = [idx, round(idx * dt_ctrl, 5)]
            for k in range(4):
                row += [rt[k], st[k]]
            w.writerow(row)

    for h in [1, 5, 10, 20, 40]:
        if h <= len(roll):
            idx, rt, st = roll[h - 1]
            err = [float(_wrap(st[2] - rt[2])) if k == 2 else st[k] - rt[k] for k in range(4)]
            print(
                f"  after {h:>2} steps ({h * dt_ctrl * 1000:>4.0f} ms): "
                f"d_cart_pos={err[0]:+.4f} m  d_cart_vel={err[1]:+.4f} m/s  "
                f"d_pole_ang={err[2]:+.4f} rad  d_pole_vel={err[3]:+.3f} rad/s"
            )

    print("=" * 78)
    print(f"  one-step CSV -> {os.path.join(OUT_DIR, CSV_ONESTEP)}")
    print(f"  rollout  CSV -> {os.path.join(OUT_DIR, CSV_ROLLOUT)}")
    print("  Reading it: a small pole_vel/cart_vel BIAS that survives averaging = a real dynamics")
    print("  gap. Big NEGATIVE cart_vel residual at high |v| = DCMotor velocity_limit too low.")
    print("=" * 78)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
