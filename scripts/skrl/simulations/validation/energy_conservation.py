# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pole rotational-energy conservation sweep: gravity off, cart locked, pole damping 0; spin the
pole at each w0 (--w_min..--w_max) and check %E kept = (w_end/w0)^2 in play's env. A faithful sim
holds ~100% at every w. The go-to validator when the model changes (esp. the double pendulum) -- a
units/clamp bug that re-caps the pole shows up here. Writes energy_conservation.csv.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isolated pole rotational-energy conservation sweep (play env).")
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
parser.add_argument("--w_min", type=float, default=1.0, help="min initial pole speed (rad/s).")
parser.add_argument("--w_max", type=float, default=50.0, help="max initial pole speed (rad/s).")
parser.add_argument("--w_step", type=float, default=1.0, help="pole speed increment (rad/s).")
parser.add_argument("--steps", type=int, default=100, help="physics steps per angular-velocity trial.")
parser.add_argument("--gravity_on", action="store_true", default=False, help="enable gravity (default OFF).")

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
CSV_OUT = "energy_conservation.csv"
CART_LOCK_ARMATURE = 1.0e6  # slider armature forced huge to LOCK the cart (isolation)


def _out_dir(name):
    """scripts/skrl/simulations/output/<name>/ (created), anchored to the repo, not cwd."""
    d = os.path.dirname(os.path.abspath(__file__))
    while os.path.basename(d) != "simulations" and os.path.dirname(d) != d:
        d = os.path.dirname(d)
    out = os.path.join(d, "output", name)
    os.makedirs(out, exist_ok=True)
    return out


OUT_DIR = _out_dir("energy_conservation")


if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


def _scalar(x):
    """Best-effort first-element float of a tensor/array/scalar."""
    try:
        return float(np.asarray(x.detach().cpu() if hasattr(x, "detach") else x).reshape(-1)[0])
    except Exception:  # noqa: BLE001
        return x


def _zero_std(term, key):
    p = term.params[key]
    term.params[key] = (p[0], 0.0)


def _freeze_dr(env_cfg):
    """Freeze DR to centers, force the two isolation knobs (cart armature huge = locked, pole
    damping 0), and widen bounds / stretch the episode so env.step() never resets mid-sweep."""
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
        fr["static_range"] = (0.5 * (lo + hi), 0.5 * (lo + hi))
    except Exception as e:  # noqa: BLE001
        print(f"[freeze_dr] skip randomize_slider_friction: {e}")

    # --- the two forced knobs for this test ---
    try:
        ev.randomize_slider_armature.params["armature_distribution_params"] = (CART_LOCK_ARMATURE, 0.0)
    except Exception as e:  # noqa: BLE001
        print(f"[freeze_dr] could not force cart armature: {e}")
    try:
        ev.randomize_pole_damping.params["damping_distribution_params"] = (0.0, 0.0)
    except Exception as e:  # noqa: BLE001
        print(f"[freeze_dr] could not zero pole damping: {e}")

    try:
        env_cfg.terminations.cart_out_of_bounds.params["bounds"] = (-2.0, 2.0)
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

    # ── the isolation conditions (gravity off by default; TGS is the trained-config solver) ──
    if not args_cli.gravity_on:
        env_cfg.sim.gravity = (0.0, 0.0, 0.0)
    _freeze_dr(env_cfg)  # freezes DR + forces cart armature huge + pole damping 0

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
    physics_dt = env.unwrapped.physics_dt
    dt_ctrl = getattr(env.unwrapped, "step_dt", physics_dt * env_cfg.decimation)
    e0 = torch.tensor([0], device=dev)

    # ── confirm the isolation conditions ACTUALLY applied ───────────────────────────────────────
    print("=" * 78)
    print(f"ENERGY CONSERVATION SWEEP (play env)   gravity={'ON' if args_cli.gravity_on else 'OFF'}")
    print(
        f"  num_envs={num_envs}  device={dev}  physics_dt={physics_dt * 1000:.3f} ms  step_dt={dt_ctrl * 1000:.2f} ms  "
        f"steps/trial={args_cli.steps}"
    )
    print(
        f"  env_cfg.sim.gravity={tuple(env_cfg.sim.gravity)}  solver_type={env_cfg.sim.physx.solver_type} (1=TGS,0=PGS)"
    )
    try:
        maxv = _rb.root_physx_view.get_dof_max_velocities()
        print(f"  dof_max_velocity: cart={_scalar(maxv[0, _sidx]):.4g} pole={_scalar(maxv[0, _pidx]):.4g}")
    except Exception as e:  # noqa: BLE001
        print(f"  dof_max_velocity unavailable: {e}")
    for label, cands in [
        ("armature", ["get_dof_armatures"]),
        ("damping", ["get_dof_dampings"]),
        ("friction", ["get_dof_friction_coefficients", "get_dof_frictions"]),
    ]:
        for gn in cands:
            g = getattr(_rb.root_physx_view, gn, None)
            if g is not None:
                try:
                    arr = g()
                    print(f"  dof {label}: cart={_scalar(arr[0, _sidx]):.6g} pole={_scalar(arr[0, _pidx]):.6g}")
                    break
                except Exception:  # noqa: BLE001
                    continue
    try:
        import re as _re

        import omni.usd
        from pxr import PhysxSchema, Usd

        stage = omni.usd.get_context().get_stage()
        rp = _re.sub(r"env_[^/]*", "env_0", env_cfg.scene.robot.prim_path)
        # THE pole spin clamp: rigid-body maxAngularVelocity (stored in DEG/s) on a pole body.
        # Confirm it's HIGH (fix = 6000 deg/s = 105 rad/s); a low value here would re-cap the pole.
        mav, mav_name = None, None
        for prim in Usd.PrimRange(stage.GetPrimAtPath(rp)):
            if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                a = PhysxSchema.PhysxRigidBodyAPI(prim).GetMaxAngularVelocityAttr()
                if a and a.Get() is not None:
                    mav, mav_name = a.Get(), prim.GetName()
                    if prim.GetName() in ("pendulum_1", "weight_1"):
                        break  # prefer a pole body
        if mav is not None:
            print(
                f"  maxAngularVelocity STAGE: {mav:.1f} deg/s = {mav * math.pi / 180.0:.2f} rad/s "
                f"(on '{mav_name}')  <-- the pole spin clamp; must be well above the swept w range"
            )
    except Exception as e:  # noqa: BLE001
        print(f"  stage clamp readout failed: {e}")
    print("  cart LOCKED (armature forced huge), pole damping 0, gravity off => a free pole MUST hold w.")
    print("  Any decay = numerical loss.  E ~ w^2  =>  %E kept = (w_end/w0)^2.")
    print("=" * 78)

    def set_pole(w0):
        """Spin the pole at w0 (angle 0), cart at rest at 0."""
        with torch.inference_mode():
            jp = _rb.data.joint_pos[0:1].clone()
            jv = _rb.data.joint_vel[0:1].clone()
            jp[0, _sidx] = 0.0
            jv[0, _sidx] = 0.0
            jp[0, _pidx] = 0.0
            jv[0, _pidx] = float(w0)
            _rb.write_joint_state_to_sim(jp, jv, env_ids=e0)
            _rb.update(physics_dt)

    zero_act = torch.zeros((num_envs, 1), dtype=torch.float32, device=dev)

    # build sweep
    ws, w = [], args_cli.w_min
    while w <= args_cli.w_max + 1e-9:
        ws.append(round(w, 4))
        w += args_cli.w_step

    rows, summary = [], []
    for w0 in ws:
        set_pole(w0)
        w_end = w0
        cart_max = 0.0
        for s in range(args_cli.steps):
            with torch.inference_mode():
                env.step(zero_act)
            w_end = float(_rb.data.joint_vel[0, _pidx])
            cart = float(_rb.data.joint_pos[0, _sidx])
            cart_max = max(cart_max, abs(cart))
            rows.append([w0, s, round((s + 1) * physics_dt, 6), w_end, float(_rb.data.joint_pos[0, _pidx]), cart])
        w_ret = abs(w_end) / w0
        e_ret = w_ret**2
        per_step = (1.0 - w_ret ** (1.0 / args_cli.steps)) * 100.0
        summary.append((w0, w_end, w_ret * 100.0, e_ret * 100.0, per_step, cart_max))

    with open(os.path.join(OUT_DIR, CSV_OUT), "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["w_init", "step", "t_s", "pole_vel", "pole_angle_rad", "cart_pos"])
        wtr.writerows(rows)

    print(f"\n{'w_init':>7} {'w_final':>9} {'%w kept':>8} {'%E kept':>8} {'loss/step':>10} {'|cart|max':>9}")
    for w0, wf, wr, er, ps, cm in summary:
        flag = "" if er > 99.0 else ("  <- >1% E loss" if er > 90.0 else "  <-- BIG loss")
        print(f"{w0:>7.1f} {wf:>9.3f} {wr:>7.1f}% {er:>7.1f}% {ps:>9.3f}% {cm:>9.4f}{flag}")
    print(f"\nsaved {os.path.join(OUT_DIR, CSV_OUT)}  ({len(rows)} rows)")
    print("READ: %E kept ~100% => sim conserves at that w. A w where it drops = a numerical/clamp ceiling.")
    print("|cart|max should be ~0 (confirms the cart stayed locked).")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
