# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DR spread check: with domain randomization ON, reset once and read back each randomized term
across all envs (slider friction/armature, body masses, sensor-noise std), then print observed
stats next to the configured distribution to confirm each term actually varies per env. Flags any
term that is fixed when it should spread. Console + per-env CSV; nothing frozen, no stepping.
Run with the SAME --task / --num_envs you train with.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check that DR actually spreads per-env in play.")
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
import os
import random

import gymnasium as gym
import numpy as np
import skrl
import torch  # noqa: F401  (imported for parity / device handling)
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

CSV_PATH = "dr_spread.csv"


def _out_dir(name):
    """scripts/simulations/single_cartpole/output/<name>/ (created), anchored to the repo, not cwd."""
    d = os.path.dirname(os.path.abspath(__file__))
    while os.path.basename(d) != "single_cartpole" and os.path.dirname(d) != d:
        d = os.path.dirname(d)
    out = os.path.join(d, "output", name)
    os.makedirs(out, exist_ok=True)
    return out


OUT_DIR = _out_dir("dr_spread")

if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


def _stat_line(name, arr, configured):
    """Print observed distribution stats for one quantity across all envs."""
    a = np.asarray(arr, dtype=float).ravel()
    spread = float(a.max() - a.min())
    tag = "SPREAD" if spread > 1e-9 else "FIXED "
    print(
        f"  {name:15s} {configured:30s} | mean={a.mean():.5f} std={a.std():.5f} "
        f"min={a.min():.5f} max={a.max():.5f} p5={np.percentile(a, 5):.5f} p95={np.percentile(a, 95):.5f} [{tag}]"
    )
    return spread


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

    # DR is left ON on purpose -- we want to SEE the per-env spread the policy trains on.

    # read the CONFIGURED distributions so we can compare observed vs intended
    ev = env_cfg.events
    fr_params = ev.randomize_slider_friction.params
    cfg_static = fr_params.get("static_range")
    cfg_dyn = fr_params.get("dynamic_params")
    cfg_arm = ev.randomize_slider_armature.params.get("armature_distribution_params")
    _pf = getattr(ev, "randomize_pole_friction", None)
    cfg_pole_fric = _pf.params.get("friction_distribution_params") if _pf else None
    _pd = getattr(ev, "randomize_pole_damping", None)
    cfg_pole_damp = _pd.params.get("damping_distribution_params") if _pd else None
    _nz = getattr(ev, "randomize_pole_vel_noise", None)
    cfg_noise = (_nz.params.get("mean"), _nz.params.get("std")) if _nz else None
    _cr = getattr(ev, "reset_cart_position", None)
    cfg_cart_pos = _cr.params.get("position_range") if _cr else None
    cfg_cart_vel = _cr.params.get("velocity_range") if _cr else None
    _pr = getattr(ev, "reset_pole_position", None)
    cfg_pole_pos = _pr.params.get("position_range") if _pr else None
    cfg_pole_vel = _pr.params.get("velocity_range") if _pr else None
    mass_terms = {
        "cart_1": getattr(ev, "randomize_cart_mass", None),
        "shaft_1": getattr(ev, "randomize_shaft_mass", None),
        "pendulum_1": getattr(ev, "randomize_pole_mass", None),
        "weight_1": getattr(ev, "randomize_weight_mass", None),
    }

    # log_dir parity only (policy is NOT loaded)
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

    # one reset -> every env draws its own reset-event samples
    obs, _ = env.reset()
    num_envs = env.unwrapped.num_envs
    _rb = env.unwrapped.scene["robot"]
    _view = _rb.root_physx_view
    _sidx = _rb.find_joints("slider_to_cart")[0][0]
    _pidx = _rb.find_joints("cart_to_pole")[0][0]
    _rb.update(env.unwrapped.physics_dt)

    fp = _view.get_dof_friction_properties().cpu().numpy()  # (N, dofs, 3): static, dynamic, viscous
    arm = _view.get_dof_armatures().cpu().numpy()  # (N, dofs)
    damp = _view.get_dof_dampings().cpu().numpy()  # (N, dofs)
    masses = _view.get_masses().cpu().numpy()  # (N, bodies)
    jpos = _rb.data.joint_pos.cpu().numpy()  # (N, dofs) -- post-reset initial state
    jvel = _rb.data.joint_vel.cpu().numpy()  # (N, dofs)
    # per-episode pole-velocity SENSOR noise std (set by randomize_pole_vel_noise_std on the env)
    pole_noise = getattr(env.unwrapped, "pole_vel_noise_std", None)
    pole_noise = pole_noise.cpu().numpy().ravel() if pole_noise is not None else None

    def _fmt(prefix, params):
        return f"cfg {prefix}{tuple(round(float(x), 6) for x in params)}" if params else "cfg (none)"

    print("=" * 120)
    print(f"DR SPREAD CHECK  (DR ON, one reset, {num_envs} envs)   [SPREAD = randomizing ; FIXED = not varying]")
    print("=" * 120)
    print("-- initial state (reset_* events; ranges are OFFSETS added to the default joint value) --")
    _stat_line("cart_pos", jpos[:, _sidx], _fmt("offset", cfg_cart_pos))
    _stat_line("cart_vel", jvel[:, _sidx], _fmt("offset", cfg_cart_vel))
    _stat_line("pole_pos", jpos[:, _pidx], _fmt("offset", cfg_pole_pos))
    _stat_line("pole_vel", jvel[:, _pidx], _fmt("offset", cfg_pole_vel))
    print("-- slider joint (friction effort N ; armature kg) --")
    _stat_line("slider static", fp[:, _sidx, 0], _fmt("uniform", cfg_static))
    _stat_line("slider dynamic", fp[:, _sidx, 1], _fmt("gauss", cfg_dyn))
    _stat_line("slider viscous", fp[:, _sidx, 2], f"cfg fixed {float(fr_params.get('viscous_value', 0.0))}")
    _stat_line("slider armature", arm[:, _sidx], _fmt("gauss", cfg_arm))
    print("-- pole joint (friction effort ; actuator damping) --")
    _stat_line("pole static", fp[:, _pidx, 0], _fmt("gauss", cfg_pole_fric))
    _stat_line("pole dynamic", fp[:, _pidx, 1], _fmt("gauss", cfg_pole_fric))
    _stat_line("pole viscous", fp[:, _pidx, 2], _fmt("gauss", cfg_pole_fric))
    _stat_line("pole damping", damp[:, _pidx], _fmt("gauss", cfg_pole_damp))
    print("-- observation-noise DR (per-episode sensor noise std, rad/s) --")
    if pole_noise is not None:
        _stat_line("pole_vel noise", pole_noise, _fmt("gauss", cfg_noise))
    else:
        print("  pole_vel noise  | [!] env.pole_vel_noise_std not set -- noise DR event inactive/absent")
    print("-- body masses (kg) --")
    body_cols = {}
    for bn in ["cart_1", "shaft_1", "pendulum_1", "weight_1"]:
        ids, _ = _rb.find_bodies(bn)
        if not ids:
            continue
        col = masses[:, ids[0]]
        body_cols[bn] = col
        term = mass_terms.get(bn)
        cfg = (
            _fmt(term.params.get("distribution", "?"), term.params.get("mass_distribution_params"))
            if term
            else "cfg (none)"
        )
        _stat_line(f"mass {bn}", col, cfg)

    print("-- joint limits (from USD; NOT randomized -- catches the pole-lock trap DR checks miss) --")
    try:
        lims = _view.get_dof_limits().cpu().numpy()  # (N, dofs, 2): lower, upper
        slo, shi = float(lims[0, _sidx, 0]), float(lims[0, _sidx, 1])
        plo, phi = float(lims[0, _pidx, 0]), float(lims[0, _pidx, 1])
        print(f"  slider limit : [{slo:+.4f}, {shi:+.4f}]  (expect ~[-0.5, +0.5])")
        if abs(phi - plo) < 1e-3:
            print(f"  pole limit   : [{plo:+.4f}, {phi:+.4f}]  [!!] POLE LOCKED -> it CANNOT swing up.")
            print("               Revert cart_to_pole lowerLimit/upperLimit in the USDA to +/-1e6 before training!")
        else:
            print(f"  pole limit   : [{plo:+.4f}, {phi:+.4f}]  (wide -> pole free to swing) OK")
    except Exception as e:
        print(f"  <could not read joint limits: {e}>")
        print("  -> verify the USDA cart_to_pole lowerLimit/upperLimit are +/-1e6 (NOT 0) by hand.")

    print("-" * 120)
    static = fp[:, _sidx, 0]
    dynamic = fp[:, _sidx, 1]
    # PhysX constraint (our event clamps it): dynamic <= static, every env
    viol = int(np.sum(dynamic > static + 1e-6))
    print(
        f"  clamp dynamic<=static : {'OK (0 violations)' if viol == 0 else f'VIOLATED in {viol} envs'}   "
        f"max(dynamic-static)={float((dynamic - static).max()):+.5f}"
    )

    # auto-flag: a term configured to vary that came back FIXED means DR didn't reach the sim for it
    def _wants_spread(cfg, kind):
        if cfg is None:
            return False
        return abs(float(cfg[1]) - float(cfg[0])) > 0.0 if kind == "range" else float(cfg[1]) > 0.0

    auto = [
        ("cart_pos", jpos[:, _sidx], cfg_cart_pos, "range"),
        ("pole_pos", jpos[:, _pidx], cfg_pole_pos, "range"),
        ("slider static", fp[:, _sidx, 0], cfg_static, "range"),
        ("slider dynamic", fp[:, _sidx, 1], cfg_dyn, "std"),
        ("slider armature", arm[:, _sidx], cfg_arm, "std"),
        ("pole friction", fp[:, _pidx, 0], cfg_pole_fric, "std"),
        ("pole damping", damp[:, _pidx], cfg_pole_damp, "std"),
        ("pole_vel noise", pole_noise, cfg_noise, "std"),
    ]
    for bn in body_cols:
        term = mass_terms.get(bn)
        p = term.params.get("mass_distribution_params") if term else None
        auto.append((f"mass {bn}", body_cols[bn], p, "std"))
    if num_envs < 32:
        print(
            f"  [!] only {num_envs} env(s): the SPREAD check is MEANINGLESS -- one sample per term is "
            "std=0 by definition, so every row reads FIXED even when DR is working."
        )
        print(
            "      Re-run with many envs (e.g. --num_envs 4096) to see the real spread. "
            "(Values above being OFF-center & in-range already shows DR is firing.)"
        )
    else:
        fixed_flags = [
            nm
            for nm, col, cfg, kind in auto
            if col is not None and _wants_spread(cfg, kind) and float(np.max(col) - np.min(col)) < 1e-9
        ]
        if fixed_flags:
            print(f"  [!] configured to vary but came back FIXED: {fixed_flags}  (DR not reaching the sim for these)")
        else:
            print("  spread sanity: everything configured to vary is varying. ✓")

    # physical-range sanity: masses / armature / damping / friction must be strictly positive
    neg = []
    for nm, col in [(f"mass {b}", body_cols[b]) for b in body_cols] + [
        ("slider armature", arm[:, _sidx]),
        ("pole damping", damp[:, _pidx]),
        ("slider static", fp[:, _sidx, 0]),
        ("pole static", fp[:, _pidx, 0]),
    ]:
        if float(np.min(col)) <= 0.0:
            neg.append(f"{nm} (min={float(np.min(col)):.6g})")
    if neg:
        print(f"  [!] non-positive samples (std too large?): {neg}")
    else:
        print("  physical-range sanity: masses/armature/damping/friction all strictly positive. ✓")
    print("=" * 120)

    # per-env CSV for histograms / inspection
    csv_path = os.path.join(OUT_DIR, CSV_PATH)
    with open(csv_path, "w", newline="") as f:
        wtr = csv.writer(f)
        noise_col = ["pole_vel_noise"] if pole_noise is not None else []
        cols = (
            [
                "env",
                "cart_pos",
                "cart_vel",
                "pole_pos",
                "pole_vel",
                "slider_static",
                "slider_dynamic",
                "slider_viscous",
                "slider_armature",
                "pole_static",
                "pole_dynamic",
                "pole_viscous",
                "pole_damping",
            ]
            + noise_col
            + [f"mass_{b}" for b in body_cols]
        )
        wtr.writerow(cols)
        for i in range(num_envs):
            row = [
                i,
                f"{jpos[i, _sidx]:.6f}",
                f"{jvel[i, _sidx]:.6f}",
                f"{jpos[i, _pidx]:.6f}",
                f"{jvel[i, _pidx]:.6f}",
                f"{fp[i, _sidx, 0]:.6f}",
                f"{fp[i, _sidx, 1]:.6f}",
                f"{fp[i, _sidx, 2]:.6f}",
                f"{arm[i, _sidx]:.6f}",
                f"{fp[i, _pidx, 0]:.8f}",
                f"{fp[i, _pidx, 1]:.8f}",
                f"{fp[i, _pidx, 2]:.8f}",
                f"{damp[i, _pidx]:.8f}",
            ]
            if pole_noise is not None:
                row.append(f"{pole_noise[i]:.5f}")
            row += [f"{body_cols[b][i]:.6f}" for b in body_cols]
            wtr.writerow(row)
    print(f"  per-env CSV -> {csv_path}")
    print("=" * 120)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
