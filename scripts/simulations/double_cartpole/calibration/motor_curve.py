# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Constant-force cart response on the double pendulum (sim counterpart of the rig fixed-torque test):
drive the cart at each --forces level with both poles hanging free, and log cart velocity vs time in
play's env. The slope (initial accel a0) and its scaling with force give the effective mass + dynamic
(Coulomb) friction to match against the rig. Tune ARMATURE (mass) and DYNAMIC (Coulomb) until the sim
a0 matches the rig (rig: a0~4.1 @5N, ~12.3 @10N -> m_eff~0.61 kg, Coulomb~2.5 N).

USDA setup (same as the breakaway test): slider_to_cart FREE, both revolute joints FREE; the probe
seeds inner=pi / outer=0 (hanging) and lets them rotate. The rig pushed -X (force -5/-10 N) from
+0.26 m, so the default drive is +0.26 -> -0.25. Writes one <F>n/001.csv per force with the rig's
columns -> copy the <F>n folders into analysis/double_pole/calibration/motor_curve/sim/.
Run:  python motor_curve.py --task Swingup-DoubleCartpole-v0 --num_envs 1 --headless --forces 5,10
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Constant-force cart probe (sim counterpart of rig fixed-torque).")
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
parser.add_argument("--forces", type=str, default="5,10", help="Comma-separated force magnitudes [N] to sweep.")
parser.add_argument("--start_pos", type=float, default=0.26, help="Cart start position [m] (mirrors rig +0.26).")
parser.add_argument(
    "--stop_pos", type=float, default=-0.25, help="Stop when the cart reaches this [m] (mirrors rig -0.25)."
)
parser.add_argument("--max_steps", type=int, default=250, help="Safety cap on steps per force.")
parser.add_argument(
    "--out_dir", type=str, default=None, help="Output folder (default: double_cartpole/output/motor_curve/)."
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
# PROBE KNOBS  <-- tune ARMATURE + DYNAMIC to match the rig a0
# ============================================================================
INNER_HANG = math.pi  # cart_to_ipole seed [rad]; inner straight down (0 = upright). Poles left free.
OUTER_HANG = 0.0  # ipole_to_opole seed [rad]; outer aligned below the inner (0 = hanging)
SLIDER_STATIC = 3.24  # slider static friction [N] (calibrated breakaway); barely matters above breakaway
DYNAMIC = 2.5  # slider dynamic (Coulomb) friction [N] -- TUNE (rig estimate ~2.5)
ARMATURE = 0.30  # slider armature (adds to effective mass) -- TUNE to match the rig a0 / m_eff
WEIGHT_MASS = 0.05  # restore real weight_1 mass [kg] (env DR zeroes it for the free-swing test)
RIG_A0 = {5: 4.1, 10: 12.3}  # rig reference a0 [m/s^2] (m_eff~0.61 kg, Coulomb~2.5 N)


def _out_dir(name):
    """scripts/simulations/double_cartpole/output/<name>/ (created), anchored to the repo, not cwd."""
    d = os.path.dirname(os.path.abspath(__file__))
    while os.path.basename(d) != "double_cartpole" and os.path.dirname(d) != d:
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
    """Freeze every randomized physics term to a deterministic value: masses at their USD/DR centers
    (weight restored to its real mass), slider static/dynamic/armature to the probe knobs. Poles stay
    present & free so their hanging inertia loads the cart like the rig. Widen the cart bound so
    env.step() never resets during the drive."""
    ev = env_cfg.events
    for term_name, key in [
        ("randomize_inner_pole_friction", "friction_distribution_params"),
        ("randomize_outer_pole_friction", "friction_distribution_params"),
        ("randomize_inner_pole_damping", "damping_distribution_params"),
        ("randomize_outer_pole_damping", "damping_distribution_params"),
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
    ev.randomize_weight_mass.params["mass_distribution_params"] = (WEIGHT_MASS, 0.0)
    ev.randomize_slider_armature.params["armature_distribution_params"] = (ARMATURE, 0.0)
    try:
        fr = ev.randomize_slider_friction.params
        fr["static_range"] = (SLIDER_STATIC, SLIDER_STATIC)
        fr["dynamic_params"] = (DYNAMIC, 0.0)
    except Exception as e:  # noqa: BLE001
        print(f"[freeze_dr] skip randomize_slider_friction: {e}")
    try:
        ev.reset_cart_position = None
    except Exception as e:  # noqa: BLE001
        print(f"[freeze_dr] reset_cart_position: {e}")
    try:
        env_cfg.terminations.cart_out_of_bounds.params["bounds"] = (-5.0, 5.0)
    except Exception as e:  # noqa: BLE001
        print(f"[freeze_dr] could not widen cart bounds: {e}")
    env_cfg.episode_length_s = 1.0e6
    return ev


def _print_effective(rb, view):
    """Read back what PhysX ACTUALLY holds for env 0 (not the cfg/knobs) -- catches set-vs-used mismatches."""
    sid = rb.find_joints("slider_to_cart")[0][0]
    print("  --- EFFECTIVE values read back from PhysX (env 0) ---")
    for meth in ("get_dof_friction_coefficients", "get_dof_friction_properties"):
        if hasattr(view, meth):
            try:
                print(f"    {meth}[slider] : {getattr(view, meth)()[0, sid].tolist()}  (may lie; behavior is truth)")
            except Exception as e:  # noqa: BLE001
                print(f"    {meth} failed: {e}")
    try:
        print(f"    slider armature       : {view.get_dof_armatures()[0, sid].item():.5f}")
    except Exception as e:  # noqa: BLE001
        print(f"    armature read failed: {e}")
    try:
        masses = view.get_masses()[0]
        coms = view.get_coms()[0]  # (num_bodies, 7): pos[0:3] + quat[3:7], body-local frame
        iner = view.get_inertias()[0].reshape(-1, 9)
        for b in ("cart_1", "ishaft_1", "ipole_1", "oshaft_1", "opole_1", "weight_1"):
            bi = rb.find_bodies(b)[0][0]
            d = iner[bi]
            c = coms[bi]
            print(
                f"    mass[{b:9s}]={masses[bi].item():.5f}  CoM(x,y,z)="
                f"({c[0].item():.5f},{c[1].item():.5f},{c[2].item():.5f})  I(xx,yy,zz)="
                f"{d[0].item():.3e},{d[4].item():.3e},{d[8].item():.3e}"
            )
    except Exception as e:  # noqa: BLE001
        print(f"    mass/CoM/inertia read failed: {e}")
    try:
        for name, a in rb.actuators.items():
            parts = []
            for attr in ("saturation_effort", "velocity_limit", "effort_limit", "armature", "stiffness", "damping"):
                if hasattr(a, attr):
                    val = getattr(a, attr)
                    try:
                        val = round(float(val.flatten()[0]), 4)
                    except Exception:  # noqa: BLE001
                        contextlib.suppress(Exception)
                    parts.append(f"{attr}={val}")
            print(f"    actuator[{name}]: {', '.join(parts)}")
    except Exception as e:  # noqa: BLE001
        print(f"    actuator read failed: {e}")
    try:
        import omni.usd
        from pxr import PhysxSchema  # noqa: F401

        stage = omni.usd.get_context().get_stage()
        for prim in stage.Traverse():
            if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                pos = prim.GetAttribute("physxArticulation:solverPositionIterationCount").Get()
                vel = prim.GetAttribute("physxArticulation:solverVelocityIterationCount").Get()
                print(f"    solver iters APPLIED to {prim.GetName()}: position={pos}, velocity={vel}")
                break
    except Exception as e:  # noqa: BLE001
        print(f"    solver-iteration read failed: {e}")


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
    action_scale = float(getattr(env_cfg.actions.joint_effort, "scale", 40.0))
    direction = -1.0 if args_cli.stop_pos < args_cli.start_pos else 1.0  # rig pushed -X
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
    _view = _rb.root_physx_view
    _sidx = _rb.find_joints("slider_to_cart")[0][0]
    _iidx = _rb.find_joints("cart_to_ipole")[0][0]
    _oidx = _rb.find_joints("ipole_to_opole")[0][0]
    dt = getattr(env.unwrapped, "step_dt", env.unwrapped.physics_dt * env_cfg.decimation)
    e0 = torch.tensor([0], device=dev)

    print("=" * 78)
    print("MOTOR-CURVE (constant-force cart response, double pendulum, poles hanging free)")
    print(f"  num_envs={num_envs}  device={dev}  step_dt={dt:.4f}s  action_scale={action_scale} N")
    print(f"  armature={ARMATURE}  dynamic={DYNAMIC} N  static={SLIDER_STATIC} N  weight={WEIGHT_MASS} kg")
    print(
        f"  forces [N] = {forces} x dir {direction:+.0f}   drive {args_cli.start_pos:+.2f} -> "
        f"{args_cli.stop_pos:+.2f} m  (max {args_cli.max_steps} steps)"
    )
    _print_effective(_rb, _view)
    print("=" * 78)

    def reset_cart():
        """Cart at start_pos (at rest); both poles hanging and still."""
        with torch.inference_mode():
            jp = _rb.data.joint_pos[0:1].clone()
            jv = _rb.data.joint_vel[0:1].clone()
            jp[0, _sidx] = args_cli.start_pos
            jp[0, _iidx] = INNER_HANG
            jp[0, _oidx] = OUTER_HANG
            jv[0, _sidx] = 0.0
            jv[0, _iidx] = 0.0
            jv[0, _oidx] = 0.0
            _rb.write_joint_state_to_sim(jp, jv, env_ids=e0)
            _rb.update(env.unwrapped.physics_dt)

    out_dir = os.path.abspath(args_cli.out_dir) if args_cli.out_dir else _out_dir("motor_curve")
    os.makedirs(out_dir, exist_ok=True)
    summary = []  # (force, a0, v_max)
    for F in forces:
        reset_cart()
        act = torch.zeros((num_envs, 1), dtype=torch.float32, device=dev)
        act[0, 0] = direction * F / action_scale
        frows = []
        speeds = []
        vmax = 0.0
        for k in range(args_cli.max_steps):
            p = float(_rb.data.joint_pos[0, _sidx])
            v = float(_rb.data.joint_vel[0, _sidx])
            ae = float(_rb.data.applied_torque[0, _sidx])  # effort PhysX ACTUALLY applied to the slider
            speeds.append(abs(v))
            vmax = max(vmax, abs(v))
            frows.append(
                (k, k * dt, p, v, float(_rb.data.joint_pos[0, _iidx]), float(_rb.data.joint_pos[0, _oidx]), ae)
            )
            if (direction < 0 and p <= args_cli.stop_pos) or (direction > 0 and p >= args_cli.stop_pos):
                break
            with torch.inference_mode():
                _, _, _, dones, _ = env.step(act)
            if bool(dones[0]):
                print(f"  [WARN] env reset during F={F} at step {k}.")
                break
        a0 = (speeds[2] - speeds[1]) / dt if len(speeds) >= 3 else float("nan")
        summary.append((F, a0, vmax))
        aes = [abs(r[6]) for r in frows[1:]]  # skip step 0 (pre-step stale value)
        if aes:
            print(
                f"  F={F:.0f}N delivered slider effort: cmd={F:.1f}  mean={sum(aes) / len(aes):.3f}"
                f"  min={min(aes):.3f}  max={max(aes):.3f}  (sag below cmd => motor derating)"
            )
        fdir = os.path.join(out_dir, f"{F:.0f}n")
        os.makedirs(fdir, exist_ok=True)
        with open(os.path.join(fdir, "001.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "force_n",
                    "step",
                    "time_s",
                    "cart_pos_m",
                    "cart_vel_mps",
                    "inner_angle_rad",
                    "outer_angle_rad",
                    "applied_effort_n",
                ]
            )
            for k, t, p, v, ia, oa, ae in frows:
                w.writerow(
                    [
                        f"{direction * F:.2f}",
                        k,
                        f"{t:.6f}",
                        f"{p:.6f}",
                        f"{v:.6f}",
                        f"{ia:.6f}",
                        f"{oa:.6f}",
                        f"{ae:.6f}",
                    ]
                )

    print("=" * 78)
    print(f"  {'F (N)':>6} | {'sim a0':>8} | {'rig a0':>8} | {'sim/rig':>8} | {'v_max':>7}")
    Fs, a0s = [], []
    for F, a0, vmax in summary:
        rig = RIG_A0.get(int(F), float("nan"))
        print(f"  {F:>6.0f} | {a0:>8.2f} | {rig:>8.2f} | {a0 / rig if rig else float('nan'):>8.2f} | {vmax:>7.2f}")
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
        print(f"  SIM effective mass = {m_eff:.3f} kg   Coulomb ~ {coulomb:.2f} N   (rig ~0.61 kg / ~2.5 N)")
        print("  raise ARMATURE if sim m_eff too low (a0 too high); raise DYNAMIC if sim Coulomb too low.")
    print(f"  wrote {len(forces)} files -> {out_dir}{os.sep}<F>n{os.sep}001.csv")
    print("  copy the <F>n folders into analysis/double_pole/calibration/motor_curve/sim/ and re-run the plot.")
    print("=" * 78)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
