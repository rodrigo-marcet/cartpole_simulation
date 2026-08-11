# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Free-swing pole energy decay: seed the pole at the rig release IC, feed zero action, and log
angle/vel/energy each step in play's env so it overlays on the rig ground-truth. Used to calibrate
pole friction/damping (the FREE-SWING DECAY KNOBS below). Writes free_swing.csv (+ _noisy:
policy-observed sensor values). Pole angle 0 = upright, pi = hanging.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Free-swing (pole energy decay) probe inside play's env.")
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
# Release IC = row 1 of analysis/data/pole_analysis/ground_truth/001.csv (the rig free-swing).
POLE_ANGLE0 = 0.067465  # rad from upright (0 = upright, pi = hanging)
POLE_VEL0 = 0.288913  # rad/s
DURATION = 60.0  # s to simulate (rig trace is ~128 s; raise for a full-length overlay)
ENERGY_THRESHOLD = 1e-4  # J -- stop early once the pendulum has essentially stopped
CSV_PATH = "free_swing.csv"
CSV_PATH_NOISY = "free_swing_noisy.csv"  # policy-observed sensor values


def _out_dir(name):
    """scripts/simulations/single_cartpole/output/<name>/ (created), anchored to the repo, not cwd."""
    d = os.path.dirname(os.path.abspath(__file__))
    while os.path.basename(d) != "single_cartpole" and os.path.dirname(d) != d:
        d = os.path.dirname(d)
    out = os.path.join(d, "output", name)
    os.makedirs(out, exist_ok=True)
    return out


OUT_DIR = _out_dir("free_swing")

# ---------------------------------------------------------------------------
# FREE-SWING DECAY KNOBS  <-- change these to tune the energy decay, then re-run
# ---------------------------------------------------------------------------
# These override the frozen DR CENTERS for env 0. LOWER => SLOWER decay (less loss).
# The POLE terms dominate the decay; the SLIDER (cart-reaction) terms barely matter
# in a free swing. Once a pair matches ground_truth/001.csv, port them into
# pendulum_env_cfg.py (randomize_pole_friction / randomize_pole_damping centers).
POLE_FRICTION = 0.00007  # cart_to_pole Coulomb friction (effort, N*m) -> randomize_pole_friction
POLE_DAMPING = 0.00001  # cart_to_pole viscous damping               -> randomize_pole_damping
SLIDER_STATIC = 1.8  # cart breakaway effort [N]                  -> randomize_slider_friction
SLIDER_DYNAMIC = 1.7  # cart Coulomb effort [N]                    -> randomize_slider_friction

# Single-pendulum energy params -- MUST match energy_pendulum.py + compare_energies.py.
PEND_M = 0.03  # kg (point mass at tip)
PEND_L = 0.125  # m
PEND_G = 9.81  # m/s^2
PEND_I = PEND_M * PEND_L**2
PEND_MGL = PEND_M * PEND_G * PEND_L


def _energy(angle: float, ang_vel: float) -> float:
    """PE = mgL*(1+cos): 0 at bottom, 2*mgL at top.  KE = 0.5*I*w^2.  (0 = upright.)"""
    return PEND_MGL * (1.0 + math.cos(angle)) + 0.5 * PEND_I * ang_vel**2


if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


def _zero_std(term, key):
    """Collapse a (center, std) distribution-params tuple to (center, 0.0)."""
    p = term.params[key]
    term.params[key] = (p[0], 0.0)


def _freeze_dr(env_cfg):
    """Freeze EVERY randomized term to its nominal so env 0 == the calibrated rig.

    The free-swing decay depends on pole friction, pole damping, pole/weight/shaft/cart mass,
    the slider friction (cart reaction) and the drivetrain armature -- so all of them are pinned.
    """
    ev = env_cfg.events
    # THE decay knobs (from the constants at the top): pole friction + pole damping.
    ev.randomize_pole_friction.params["friction_distribution_params"] = (POLE_FRICTION, 0.0)
    ev.randomize_pole_damping.params["damping_distribution_params"] = (POLE_DAMPING, 0.0)
    # slider (cart-reaction) friction -> explicit point values (minor effect in free swing).
    ev.randomize_slider_friction.params["dynamic_params"] = (SLIDER_DYNAMIC, 0.0)
    ev.randomize_slider_friction.params["static_range"] = (SLIDER_STATIC, SLIDER_STATIC)
    # everything else -> center (std 0). Guarded so a renamed term won't crash the probe.
    for term_name, key in [
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
    # don't let the episode reset mid-swing: widen the cart bound and outlast DURATION.
    try:
        env_cfg.terminations.cart_out_of_bounds.params["bounds"] = (-1.0, 1.0)
    except Exception as e:  # noqa: BLE001
        print(f"[freeze_dr] could not widen cart bounds: {e}")
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

    env.reset()
    dev = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    _rb = env.unwrapped.scene["robot"]
    _view = _rb.root_physx_view
    _sidx = _rb.find_joints("slider_to_cart")[0][0]
    _pidx = _rb.find_joints("cart_to_pole")[0][0]
    dt_ctrl = getattr(env.unwrapped, "step_dt", env.unwrapped.physics_dt * env_cfg.decimation)

    print("=" * 78)
    print("FREE-SWING (pole energy decay)")
    print(f"  num_envs={num_envs}  device={dev}  step_dt={dt_ctrl:.4f}s  (log rate {1 / dt_ctrl:.0f} Hz)")
    print(
        f"  release IC -> angle={POLE_ANGLE0:.6f} rad ({math.degrees(POLE_ANGLE0):.2f} deg from upright)  "
        f"vel={POLE_VEL0:.6f} rad/s"
    )
    print(f"  E(0) target = {_energy(POLE_ANGLE0, POLE_VEL0):.6e} J   |   E_max (2*mgL) = {2 * PEND_MGL:.6e} J")
    try:
        print(
            "  env0 pole friction_props:", _view.get_dof_friction_properties()[0].tolist(), "(static,dyn,visc; may lie)"
        )
        print("  env0 damping read-back  :", _view.get_dof_dampings()[0].tolist())
        print("  env0 masses             :", _view.get_masses()[0].tolist())
    except Exception as e:  # noqa: BLE001
        print("  read-back unavailable:", e)
    print("=" * 78)

    # seed env 0: pole at the rig release IC, cart parked at 0.
    jp = _rb.data.joint_pos[0:1].clone()
    jv = _rb.data.joint_vel[0:1].clone()
    jp[0, _sidx] = 0.0
    jp[0, _pidx] = POLE_ANGLE0
    jv[0, _sidx] = 0.0
    jv[0, _pidx] = POLE_VEL0
    _rb.write_joint_state_to_sim(jp, jv, env_ids=torch.tensor([0], device=dev))
    _rb.update(env.unwrapped.physics_dt)

    # Probe the observation layout once; the noisy log reads the obs manager (= what the policy
    # actually sees, with the baked-in encoder/velocity noise). Guarded so the clean run survives
    # even if the obs layout is unexpected.
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

    ts, coss, sins, angs, avs, es = [], [], [], [], [], []
    cxs, cvs = [], []  # cart pos/vel: if the cart moves, slider friction is draining pole energy
    # NOISY sensor logs = exactly what the policy observes (encoder + 0.28 rad/s vel noise applied)
    n_coss, n_sins, n_angs, n_avs, n_es, n_cxs, n_cvs = [], [], [], [], [], [], []
    _zero = torch.zeros((num_envs, 1), dtype=torch.float32, device=dev)
    max_steps = int(DURATION / dt_ctrl) + 5
    k = 0
    while True:
        ang = _rb.data.joint_pos[0, _pidx].item()
        w = _rb.data.joint_vel[0, _pidx].item()
        cx = _rb.data.joint_pos[0, _sidx].item()
        cv = _rb.data.joint_vel[0, _sidx].item()
        t = k * dt_ctrl
        e = _energy(ang, w)
        ts.append(t)
        coss.append(math.cos(ang))
        sins.append(math.sin(ang))
        angs.append(ang)
        avs.append(w)
        es.append(e)
        cxs.append(cx)
        cvs.append(cv)
        if noisy_ok:
            # re-compute the obs at THIS (clean) state -> row-aligned with the clean CSV.
            _o = env.unwrapped.observation_manager.compute()["policy"][0]
            n_sin, n_cos, n_pv = float(_o[2]), float(_o[3]), float(_o[4])
            n_ang = math.atan2(n_sin, n_cos)
            n_coss.append(n_cos)
            n_sins.append(n_sin)
            n_angs.append(n_ang)
            n_avs.append(n_pv)
            n_es.append(_energy(n_ang, n_pv))
            n_cxs.append(float(_o[0]))
            n_cvs.append(float(_o[1]))
        if t >= DURATION or e < ENERGY_THRESHOLD or k > max_steps:
            break
        with torch.inference_mode():
            _, _, _, dones, _ = env.step(_zero)
        if bool(dones[0]):
            print(
                f"[WARN] env 0 reset at step {k} (t={t:.2f}s); swing truncated. Widen bounds / raise episode_length_s."
            )
            break
        k += 1

    # write the full trajectory (columns match ground_truth/001.csv + an energy column)
    csv_path = os.path.join(OUT_DIR, CSV_PATH)
    with open(csv_path, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["time", "cos", "sin", "angle", "angular_vel", "energy", "cart_pos", "cart_vel"])
        for i in range(len(ts)):
            wtr.writerow(
                [
                    f"{ts[i]:.6f}",
                    f"{coss[i]:.6f}",
                    f"{sins[i]:.6f}",
                    f"{angs[i]:.6f}",
                    f"{avs[i]:.6f}",
                    f"{es[i]:.6e}",
                    f"{cxs[i]:.6f}",
                    f"{cvs[i]:.6f}",
                ]
            )

    # NOISY trajectory: same columns, but the values the POLICY sees (sensor noise applied).
    csv_path_noisy = os.path.join(OUT_DIR, CSV_PATH_NOISY)
    if noisy_ok:
        with open(csv_path_noisy, "w", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(["time", "cos", "sin", "angle", "angular_vel", "energy", "cart_pos", "cart_vel"])
            for i in range(len(n_avs)):
                wtr.writerow(
                    [
                        f"{ts[i]:.6f}",
                        f"{n_coss[i]:.6f}",
                        f"{n_sins[i]:.6f}",
                        f"{n_angs[i]:.6f}",
                        f"{n_avs[i]:.6f}",
                        f"{n_es[i]:.6e}",
                        f"{n_cxs[i]:.6f}",
                        f"{n_cvs[i]:.6f}",
                    ]
                )

    # --- diagnostics ---
    e = np.array(es)
    e0 = float(e[0])
    e_end = float(e[-1])
    # half-energy decay time: first t where E drops below E0/2 (envelope proxy)
    half_idx = np.where(e < 0.5 * e0)[0]
    t_half = ts[half_idx[0]] if len(half_idx) else float("nan")
    # pole must actually swing; ~0 sweep => the cart_to_pole joint is LOCKED in the USD.
    pole_sweep = float(np.max(angs) - np.min(angs)) if angs else float("nan")
    # cart motion: if the cart slips, slider friction (1.8 N) drains pole energy -> fast decay
    # that NO pole-friction/damping value can fix. A clean pole test wants the cart ~still.
    cart_travel = float(np.max(cxs) - np.min(cxs)) if cxs else float("nan")
    cart_vmax = float(np.max(np.abs(cvs))) if cvs else float("nan")

    print("=" * 78)
    print(f"RESULT (play free-swing) : samples={len(ts)}  duration={ts[-1]:.2f}s")
    print(f"  E(0)   = {e0:.6e} J   (target {_energy(POLE_ANGLE0, POLE_VEL0):.6e})")
    print(f"  E(end) = {e_end:.6e} J   ({100 * e_end / e0:.1f}% of E0)")
    print(f"  half-energy time t(E<E0/2) = {t_half:.2f} s   <-- RIG target ~20 s (viscous, tau_E~24 s)")
    print(f"  pole angle sweep = {pole_sweep:.3f} rad  (want LARGE; ~0 => pole joint is LOCKED in the USD)")
    print(f"  cart travel = {cart_travel:.4f} m   cart |v|max = {cart_vmax:.3f} m/s")
    if cart_travel > 0.02:
        print("  [!] cart is MOVING -> slider friction is a major energy sink here. Lowering the pole")
        print("      knobs won't fix it; the fast decay is the cart, not the pole.")
    print(f"  CSV -> {csv_path}")
    if noisy_ok:
        inj = np.array(n_avs) - np.array(avs)
        print(f"  NOISY CSV -> {csv_path_noisy}")
        print(f"  injected pole-vel noise std = {inj.std():.3f} rad/s   (RIG measured ~0.28 rad/s)")
        print("  overlay play_energy_noisy.csv 'angular_vel' on ground_truth/001.csv to judge it:")
        print("    similar jitter => spot-on;  smoother => too little;  jumpier => too much.")
    if pole_sweep < 0.2:
        print("  [!] pole barely moved -> cart_to_pole is LOCKED (USDA lowerLimit==upperLimit).")
        print("      Revert the pole joint limits to +/-1e6 before comparing / training.")
    else:
        print("  Compare against analysis/data/pole_analysis/ground_truth/001.csv")
        print("  (same columns; loadable by pole_analysis/compare_energies.py). Overlay E(t):")
        print("    MATCH  => pole friction/damping/inertia are faithful on the GPU path.")
        print("    decays TOO FAST => too much pole friction/damping in the env.")
        print("    decays TOO SLOW => too little (or a term got clobbered at reset).")
    print("=" * 78)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
