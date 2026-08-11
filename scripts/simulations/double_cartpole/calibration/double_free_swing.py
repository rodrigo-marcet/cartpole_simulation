# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coupled free-swing sim2real check: both poles swinging together with the CART LOCKED.

Teacher-forces each row of a trimmed rig free-swing log into the env, steps once with zero action,
and compares the sim's true next state to the rig's next. With no force and a locked cart the only
things setting th1dd and th2dd are the mass matrix, gravity and joint friction -- so this isolates
the two-link COUPLING that the policy replay could only see through a failing controller.

Headline output is the JOINT SPLIT block. th2 is relative, so th1+th2 is the absolute outer
orientation; if the sim places the outer link correctly but divides the motion wrongly between the
joints, the two residuals anti-correlate and their sum collapses. That is an inertia-ratio /
coupling signature rather than a per-pivot friction one. The split is then binned by |th2|: the
coupling term goes as h4*cos(th2), so an error that tracks cos(th2) indicts h4 while a flat one
indicts the diagonal inertias (h3 vs h6).

Input is the trimmed CSV from analysis/double_pole/calibration/double_free_swing/trim_release.py
(dt_s + the eight pole columns) -- no conversion step; cart position, velocity and force are all
zero by construction. Deliberately standalone: it duplicates the residual maths from ../replay.py
rather than sharing it, so nothing there changes.

Run:  python double_free_swing.py --task Swingup-DoubleCartpole-v0 --num_envs 1 --headless \
          --logs inputs/free_swing.csv
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Coupled free-swing sim2real check (cart locked).")
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
parser.add_argument("--logs", type=str, default=None, help="Trimmed rig CSV (default: inputs/free_swing.csv).")

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
from isaaclab_tasks.utils.hydra import hydra_task_config

# ============================================================================
# PROBE KNOBS
# ============================================================================
CSV_ONESTEP = "onestep.csv"
CART_LOCK_N = 500.0  # slider friction effort used to pin the cart; pole reactions are a few N
WHIRL_RADPS = 20.0  # above this the rig's finite-diff estimator aliases -> its own log is wrong

DIM_NAMES = ["ipole_ang", "ipole_vel", "opole_ang", "opole_vel"]
DIM_UNITS = ["rad", "rad/s", "rad", "rad/s"]
ANGLE_DIMS = (0, 2)
VEL_DIMS = (1, 3)

# per-sample MEASUREMENT noise of the rig log; both poles share the single cart-pole's encoder
# and estimator, so they inherit its measured floor. NOT a one-step prediction floor.
MEAS_NOISE = {"ipole_ang": 1e-3, "ipole_vel": 0.57, "opole_ang": 1e-3, "opole_vel": 0.57}
# the teacher-forced velocity is stale by this fraction of a control period: the firmware's
# estimator is a one-sample backward difference, i.e. the mean velocity over the PAST interval.
LAG_FRAC = 0.5

CSV_KEYS = {
    "ipole_ang": "inner_pole_angle_rad",
    "ipole_vel": "inner_pole_vel_radps",
    "opole_ang": "outer_pole_angle_rad",
    "opole_vel": "outer_pole_vel_radps",
}


def _anchor(*parts):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


OUT_DIR = _anchor("output", "double_free_swing")
os.makedirs(OUT_DIR, exist_ok=True)

if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def _zero_std(term, key):
    p = term.params[key]
    term.params[key] = (p[0], 0.0)


def _freeze_dr(env_cfg):
    """Every randomized physics term to its configured center, the cart pinned, and the episode
    stretched so env.step() never auto-resets mid-replay."""
    ev = env_cfg.events
    for term_name, key in [
        ("randomize_cart_mass", "mass_distribution_params"),
        ("randomize_ishaft_mass", "mass_distribution_params"),
        ("randomize_ipole_mass", "mass_distribution_params"),
        ("randomize_oshaft_mass", "mass_distribution_params"),
        ("randomize_opole_mass", "mass_distribution_params"),
        ("randomize_weight_mass", "mass_distribution_params"),
        ("randomize_slider_armature", "armature_distribution_params"),
        ("randomize_inner_pole_friction", "friction_distribution_params"),
        ("randomize_outer_pole_friction", "friction_distribution_params"),
        ("randomize_inner_pole_damping", "damping_distribution_params"),
        ("randomize_outer_pole_damping", "damping_distribution_params"),
    ]:
        try:
            _zero_std(getattr(ev, term_name), key)
        except Exception as e:  # noqa: BLE001
            print(f"[freeze_dr] skip {term_name}.{key}: {e}")
    # LOCK THE CART: the rig was under ODrive position hold, so the sim's slider must not move
    # either. Isaac 5.x joint friction is an effort in N, so a large one is an immovable slider.
    try:
        fr = ev.randomize_slider_friction.params
        fr["static_range"] = (CART_LOCK_N, CART_LOCK_N)
        fr["dynamic_params"] = (CART_LOCK_N, 0.0)
        print(f"[cart-lock] slider friction effort set to {CART_LOCK_N} N")
    except Exception as e:  # noqa: BLE001
        print(f"[cart-lock] FAILED to pin the slider: {e}")
    try:
        env_cfg.terminations.cart_out_of_bounds.params["bounds"] = (-2.0, 2.0)
    except Exception as e:  # noqa: BLE001
        print(f"[freeze_dr] could not widen cart bounds: {e}")
    env_cfg.episode_length_s = 1.0e6


def _load_logs(path):
    """Trimmed free-swing CSV -> float arrays. Cart and force are zero by construction."""
    ap = os.path.abspath(path)
    if not os.path.isfile(ap):
        raise FileNotFoundError(
            f"free-swing CSV not found: {ap}\n"
            "Produce it with analysis/double_pole/calibration/double_free_swing/trim_release.py, "
            "then copy it here or pass --logs <path>."
        )
    with open(ap) as f:
        rows = list(csv.DictReader(line for line in f if line.strip()))
    if not rows:
        raise SystemExit(f"{ap}: no data rows")
    dt_col = "dt_s" if "dt_s" in rows[0] else ("time_s" if "time_s" in rows[0] else None)
    missing = [c for c in CSV_KEYS.values() if c not in rows[0]]
    if missing or dt_col is None:
        raise SystemExit(f"{ap}: missing columns {missing or ['dt_s']}")
    out = {k: np.array([float(r[c]) for r in rows]) for k, c in CSV_KEYS.items()}
    out["dt"] = np.array([float(r[dt_col]) for r in rows])
    out["path"] = ap
    return out


def _deriv(v, dt):
    a = np.zeros_like(v)
    a[:-1] = np.diff(v) / dt[:-1]
    if len(a) > 1:
        a[-1] = a[-2]
    return a


def _method_floor(name, accel_slice, dt):
    """Smallest one-step residual this test can resolve: the rig's noise enters twice, and the
    teacher-forced velocity is LAG_FRAC*dt stale so at |a| it injects |a|*LAG_FRAC*dt of error."""
    meas = math.sqrt(2.0) * MEAS_NOISE[name]
    lag = float(np.sqrt(np.mean((accel_slice * LAG_FRAC * dt) ** 2)))
    return math.sqrt(meas**2 + lag**2), meas, lag


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

    rig = _load_logs(args_cli.logs or _anchor("inputs", "free_swing.csv"))
    N = len(rig["dt"])
    dt_mean = float(np.mean(rig["dt"]))
    env_cfg.sim.dt = dt_mean / env_cfg.decimation
    print(
        f"[dt-match] rig mean dt = {dt_mean * 1000:.3f} ms -> sim.dt = {env_cfg.sim.dt * 1e6:.0f} us "
        f"x{env_cfg.decimation} = {env_cfg.decimation * env_cfg.sim.dt * 1000:.3f} ms/step"
    )

    _freeze_dr(env_cfg)
    env_cfg.log_dir = os.path.abspath(os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"]))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    env.reset()
    dev = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    _rb = env.unwrapped.scene["robot"]
    _sidx = _rb.find_joints("slider_to_cart")[0][0]
    _iidx = _rb.find_joints("cart_to_ipole")[0][0]
    _oidx = _rb.find_joints("ipole_to_opole")[0][0]
    dt_ctrl = getattr(env.unwrapped, "step_dt", env.unwrapped.physics_dt * env_cfg.decimation)
    e0 = torch.tensor([0], device=dev)
    zero_act = torch.zeros((num_envs, 1), dtype=torch.float32, device=dev)

    peak = np.maximum(np.abs(rig["ipole_vel"]), np.abs(rig["opole_vel"]))
    accel = {n: _deriv(rig[n], rig["dt"]) for n in ("ipole_vel", "opole_vel")}

    print("=" * 78)
    print("COUPLED FREE SWING (sim2real one-step residual, cart locked, zero action)")
    print(f"  logs: {rig['path']}  ({N} rows, {N * dt_ctrl:.2f} s @ {1 / dt_ctrl:.1f} Hz)")
    print(f"  device={dev}  step_dt={dt_ctrl:.5f} s")
    print(
        f"  rig |w1|max={np.abs(rig['ipole_vel']).max():.1f}  |w2|max={np.abs(rig['opole_vel']).max():.1f} rad/s"
        f"   ({int((peak >= WHIRL_RADPS).sum())}/{N} rows in the aliasing regime)"
    )
    print("=" * 78)

    def set_state(i):
        """Env 0 to rig row i, cart forced to the origin at rest. inference_mode required: step()
        marks the data buffers as inference tensors, so the write must run inside it too."""
        with torch.inference_mode():
            jp = _rb.data.joint_pos[0:1].clone()
            jv = _rb.data.joint_vel[0:1].clone()
            jp[0, _sidx] = 0.0
            jv[0, _sidx] = 0.0
            jp[0, _iidx] = float(rig["ipole_ang"][i])
            jp[0, _oidx] = float(rig["opole_ang"][i])
            jv[0, _iidx] = float(rig["ipole_vel"][i])
            jv[0, _oidx] = float(rig["opole_vel"][i])
            _rb.write_joint_state_to_sim(jp, jv, env_ids=e0)
            _rb.update(env.unwrapped.physics_dt)

    def step_zero():
        with torch.inference_mode():
            _, _, _, dones, _ = env.step(zero_act)
        return (
            float(_rb.data.joint_pos[0, _iidx]),
            float(_rb.data.joint_vel[0, _iidx]),
            float(_rb.data.joint_pos[0, _oidx]),
            float(_rb.data.joint_vel[0, _oidx]),
            float(_rb.data.joint_pos[0, _sidx]),
            bool(dones[0]),
        )

    # ---- one-step residual ----
    res, applied, cart_creep = [], [], []
    for i in range(N - 1):
        set_state(i)
        *sim_next, cart_x, done = step_zero()
        rig_next = tuple(float(rig[n][i + 1]) for n in DIM_NAMES)
        res.append(
            [
                float(_wrap(sim_next[k] - rig_next[k])) if k in ANGLE_DIMS else sim_next[k] - rig_next[k]
                for k in range(4)
            ]
        )
        shown = tuple(float(_wrap(sim_next[k])) if k in ANGLE_DIMS else sim_next[k] for k in range(4))
        applied.append((i + 1, shown, rig_next))
        cart_creep.append(abs(cart_x))
        if done:
            print(f"  [WARN] env reset during transition {i}.")
    res = np.array(res)
    print(
        f"CART LOCK CHECK: max |x| reached inside a step = {max(cart_creep) * 1000:.3f} mm "
        f"({'ok' if max(cart_creep) < 1e-3 else 'LOOSE -- raise CART_LOCK_N'})"
    )

    # ---- per-dim bias vs its own standard error ----
    mean, std = np.mean(res, axis=0), np.std(res, axis=0)
    sem = std / math.sqrt(max(len(res), 1))
    print("-" * 78)
    print("ONE-STEP RESIDUAL (sim_next - rig_next)   [bias vs its own standard error]")
    for k, name in enumerate(DIM_NAMES):
        t = abs(mean[k]) / sem[k] if sem[k] > 0 else 0.0
        print(
            f"  {name:>9} [{DIM_UNITS[k]:>5}] : bias={mean[k]:+.5f}  std={std[k]:.5f}  "
            f"sem={sem[k]:.5f}  |bias|/sem={t:5.2f} -> {'SYSTEMATIC' if t > 2 else 'consistent with zero'}"
        )
    print("  Samples along one trajectory are correlated, so the true n is below the row count and")
    print("  these ratios are optimistic. The correlation below is the evidence that survives that.")

    pk = peak[: N - 1]
    usable = pk < WHIRL_RADPS
    sl = slice(0, N - 1)
    if not usable.any():
        print("\n  no usable rows below the aliasing threshold -- nothing further to report.")
        env.close()
        return

    # ---- how much of the residual the method itself explains ----
    print("-" * 78)
    print(f"WHICH LINK (usable regime, n={int(usable.sum())}). Velocity dims only -- the angle dims are")
    print("  those same errors integrated over one step, so they carry no new information:")
    print(f"  {'dim':<11}{'rmse':>9}{'meas':>9}{'lag':>9}{'floor':>9}{'excess':>8}")
    for k in VEL_DIMS:
        name = DIM_NAMES[k]
        rr = float(np.sqrt(np.mean(res[usable, k] ** 2)))
        floor, meas, lag = _method_floor(name, accel[name][sl][usable], dt_ctrl)
        print(f"  {name:<11}{rr:>9.4f}{meas:>9.4f}{lag:>9.4f}{floor:>9.4f}{rr / floor:>8.2f}")
    print("  excess = rmse / method floor. ~1 means the test cannot see anything at that joint.")

    # ---- the headline: is the error a SPLIT between joints, or a genuine absolute error? ----
    print("-" * 78)
    print("JOINT SPLIT vs ABSOLUTE ERROR  (th2 is relative, so th1+th2 is the absolute outer):")
    for lbl, ka, kb, unit in (("angle", 0, 2, "rad"), ("vel", 1, 3, "rad/s")):
        a, b = res[usable, ka], res[usable, kb]
        s = a + b
        r = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")
        ra, rb, rs = (float(np.sqrt(np.mean(x**2))) for x in (a, b, s))
        print(
            f"  {lbl:>5} [{unit:>5}]: rmse inner={ra:.4f}  outer={rb:.4f}  SUM={rs:.4f}  "
            f"corr={r:+.3f}  collapse={1 - rs / max(ra, rb):.0%}"
        )
    print("  The two poles use SEPARATE encoders, so their noise is independent and cannot correlate.")
    print("  Strongly negative corr with the sum collapsing => the absolute pose is right and only")
    print("  the split is wrong: an inertia-ratio / coupling error, not a per-pivot friction one.")

    # ---- which term: the coupling goes as h4*cos(th2), the diagonals do not ----
    print("-" * 78)
    print("WHICH TERM  (split error binned by |th2|; coupling m12 = h6 + h4*cos(th2)):")
    th2 = np.abs(_wrap(rig["opole_ang"][sl]))[usable]
    split = res[usable, 1] - res[usable, 3]  # antisymmetric part: how lopsided the split is
    print(f"  {'|th2| bin':<18}{'n':>5}{'mean split':>13}{'rms split':>12}{'cos(th2)':>10}")
    edges = [0, 30, 60, 90, 120, 150, 180]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (th2 >= math.radians(lo)) & (th2 < math.radians(hi))
        if m.sum() >= 3:
            print(
                f"  [{lo:>3},{hi:>3}) deg      {int(m.sum()):>5}{split[m].mean():>13.4f}"
                f"{float(np.sqrt(np.mean(split[m] ** 2))):>12.4f}{math.cos(math.radians((lo + hi) / 2)):>10.3f}"
            )
    c = np.cos(th2)
    if len(c) > 2 and np.std(c) > 1e-6:
        slope, intercept = np.polyfit(c, split, 1)
        rc = float(np.corrcoef(c, split)[0, 1])
        print(f"  fit  split = {intercept:+.4f} {slope:+.4f}*cos(th2)   corr(cos th2, split) = {rc:+.3f}")
        print("  tracks cos(th2) (large slope, |corr| high)  -> h4 = m2*L1*l2, the coupling term")
        print("  flat (slope ~0, big intercept)              -> the diagonal ratio h3 vs h6")

    # ---- CSV dump ----
    with open(os.path.join(OUT_DIR, CSV_ONESTEP), "w", newline="") as f:
        w = csv.writer(f)
        cols = ["step", "t_s", "peak_pole_vel_start", "usable", "abs_th2_rad"]
        for name in DIM_NAMES:
            cols += [f"rig_{name}", f"sim_{name}", f"resid_{name}"]
        cols += ["resid_abs_opole_ang", "resid_abs_opole_vel", "split_vel"]
        w.writerow(cols)
        t = np.concatenate([[0.0], np.cumsum(rig["dt"])[:-1]])
        for j, (idx, sim_next, rig_next) in enumerate(applied):
            row = [
                idx,
                round(float(t[idx]), 5),
                round(float(pk[j]), 4),
                int(pk[j] < WHIRL_RADPS),
                round(float(abs(_wrap(rig["opole_ang"][j]))), 5),
            ]
            for k in range(4):
                row += [rig_next[k], sim_next[k], res[j, k]]
            row += [res[j, 0] + res[j, 2], res[j, 1] + res[j, 3], res[j, 1] - res[j, 3]]
            w.writerow(row)

    print("=" * 78)
    print(f"  one-step CSV -> {os.path.join(OUT_DIR, CSV_ONESTEP)}")
    print("  Read the JOINT SPLIT block first, then WHICH TERM. With the cart locked and no action")
    print("  there is nothing else left to blame: it is the mass matrix or the joint friction.")
    print("=" * 78)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
