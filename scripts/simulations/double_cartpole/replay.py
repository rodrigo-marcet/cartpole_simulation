# noqa: C901, E501
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Sim2real one-step residual for the DOUBLE cart-pole: teacher-force each rig-log row into the env,
step once with the logged force, and compare the sim's TRUE next state to the rig's next state.
A per-dim bias that survives averaging = a real dynamics gap.

Six state dims (cart + both revolutes). Angles: 0 = upright, outer is RELATIVE to the inner link.
Force is always flipped: the rig logs the policy's negated cart state but the sensor-frame force
(double_pendulum.cpp:108,129 vs :371), so flipping puts everything in the policy frame.

Writes onestep.csv and rollout.csv under double_cartpole/output/replay/.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sim2real one-step residual probe, double cart-pole.")
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
parser.add_argument("--logs", type=str, default=None, help="Rig CSV from logs_to_csv.py (default: inputs/logs.csv).")

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
CSV_ONESTEP = "onestep.csv"
CSV_ROLLOUT = "rollout.csv"

# rows 0-1 are finite-diff startup artifacts: the firmware's velocity estimator needs one prior
# call, and a stale encoder read between the first two gives an exact 0.0 (see logs_to_csv.py)
SKIP_FIRST = 2
WHIRL_RADPS = 20.0  # above this the firmware's finite-diff estimator aliases -> log is unreliable

DIM_NAMES = ["cart_pos", "cart_vel", "ipole_ang", "ipole_vel", "opole_ang", "opole_vel"]
DIM_UNITS = ["m", "m/s", "rad", "rad/s", "rad", "rad/s"]
ANGLE_DIMS = (2, 4)
VEL_DIMS = (1, 3, 5)  # the only dims carrying independent information (see WHICH LINK below)

# per-sample MEASUREMENT noise of the rig log. Both poles use the same encoder + finite-diff
# estimator as the single cart-pole's pole, so they inherit its measured floor. This is NOT a
# one-step prediction floor -- see _method_floor().
MEAS_NOISE = {
    "cart_pos": 5e-6,
    "cart_vel": 0.011,
    "ipole_ang": 1e-3,
    "ipole_vel": 0.57,
    "opole_ang": 1e-3,
    "opole_vel": 0.57,
}
M_EFF_KG = 0.77  # 0.47 kg translating + 0.30 slider armature; converts a cart dv into a force

# Fraction of the control period by which each teacher-forced velocity is STALE. The poles come
# from the firmware's one-sample backward difference at the control rate, which reports the mean
# velocity over the PAST interval -> half a sample. The cart comes from the ODrive's 1 kHz PLL,
# roughly an order of magnitude tighter; 0.05 is an estimate, not a measurement.
LAG_FRAC = {"cart_vel": 0.05, "ipole_vel": 0.5, "opole_vel": 0.5}


def _anchor(*parts):
    """<repo>/scripts/simulations/double_cartpole/<parts...>, anchored to this file, not cwd."""
    d = os.path.dirname(os.path.abspath(__file__))
    while os.path.basename(d) != "double_cartpole" and os.path.dirname(d) != d:
        d = os.path.dirname(d)
    p = os.path.join(d, *parts)
    return p


OUT_DIR = _anchor("output", "replay")
os.makedirs(OUT_DIR, exist_ok=True)


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
    """Load logs.csv -> dict of float arrays (6 state dims + force)."""
    ap = os.path.abspath(path)
    if not os.path.isfile(ap):
        raise FileNotFoundError(f"logs CSV not found: {ap}\nRun inputs/logs_to_csv.py first, or pass --logs <path>.")
    keys = {
        "cart_pos": "cart_pos_m",
        "cart_vel": "cart_vel_mps",
        "ipole_ang": "inner_angle_rad",
        "ipole_vel": "inner_vel_radps",
        "opole_ang": "outer_angle_rad",
        "opole_vel": "outer_vel_radps",
        "force": "force_n",
    }
    out = {k: [] for k in keys}
    dt = []
    with open(ap) as fh:
        for row in csv.DictReader(fh):
            for k, col in keys.items():
                out[k].append(float(row[col]))
            if row.get("dt_s"):
                dt.append(float(row["dt_s"]))
    res = {k: np.array(v) for k, v in out.items()}
    res["path"] = ap
    res["dt"] = np.array(dt) if dt else None
    res["dt_mean"] = float(np.mean(dt)) if dt else None
    return res


def _deriv(v, dt):
    """Forward difference of a rig channel, same length as v (last entry repeated)."""
    a = np.zeros_like(v)
    a[:-1] = np.diff(v) / dt[:-1]
    if len(a) > 1:
        a[-1] = a[-2]
    return a


def _method_floor(name, accel_slice, dt):
    """Smallest one-step residual this test can resolve, in that dim's units.

    Two irreducible terms, added in quadrature:
      measurement -- the rig's noise enters twice (once teacher-forced in, once in the comparison)
      lag         -- the teacher-forced velocity is stale by LAG_FRAC[name] * dt, so at an
                     acceleration |a| it injects |a| * LAG_FRAC * dt of error
    A residual near this floor says nothing about the sim; only the excess over it does.
    """
    meas = math.sqrt(2.0) * MEAS_NOISE[name]
    lag = float(np.sqrt(np.mean((accel_slice * LAG_FRAC[name] * dt) ** 2)))
    return math.sqrt(meas**2 + lag**2), meas, lag


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):  # noqa: C901
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    logs_path = args_cli.logs or _anchor("inputs", "logs.csv")
    rig = _load_logs(logs_path)
    N = len(rig["force"])

    # ALWAYS flip force_n from the sensor frame into the policy frame the cart state is logged in.
    rig["force"] = -rig["force"]

    # # FRAME TEST -- delete these 3 lines to revert.
    # for _k in ("ipole_ang", "ipole_vel", "opole_ang", "opole_vel"):
    #     rig[_k] = -rig[_k]

    action_scale = float(getattr(env_cfg.actions.joint_effort, "scale", 40.0))

    if rig.get("dt_mean"):
        env_cfg.sim.dt = rig["dt_mean"] / env_cfg.decimation
        print(
            f"[dt-match] rig mean dt = {rig['dt_mean'] * 1000:.2f} ms -> sim.dt = {env_cfg.sim.dt * 1e6:.0f} us "
            f"x{env_cfg.decimation} = {env_cfg.decimation * env_cfg.sim.dt * 1000:.2f} ms/step"
        )

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
    _sidx = _rb.find_joints("slider_to_cart")[0][0]
    _iidx = _rb.find_joints("cart_to_ipole")[0][0]
    _oidx = _rb.find_joints("ipole_to_opole")[0][0]
    dt_ctrl = getattr(env.unwrapped, "step_dt", env.unwrapped.physics_dt * env_cfg.decimation)
    e0 = torch.tensor([0], device=dev)

    peak = np.maximum(np.abs(rig["ipole_vel"]), np.abs(rig["opole_vel"]))
    dt_arr = rig["dt"] if rig["dt"] is not None else np.full(N, dt_ctrl)
    accel = {n: _deriv(rig[n], dt_arr) for n in ("cart_vel", "ipole_vel", "opole_vel")}
    print("=" * 78)
    print("REPLAY (sim2real one-step residual, double cart-pole)")
    print(f"  logs: {rig['path']}  ({N} rows, {N * dt_ctrl:.2f} s @ {1 / dt_ctrl:.0f} Hz)")
    print(f"  num_envs={num_envs}  device={dev}  step_dt={dt_ctrl:.4f}s  action_scale={action_scale} N")
    print(
        f"  force sign: FLIPPED (sensor -> policy frame; standard).  "
        f"rig |cart_v|max={np.max(np.abs(rig['cart_vel'])):.2f} m/s"
    )
    print(
        f"  rig |w1|max={np.max(np.abs(rig['ipole_vel'])):.1f}  |w2|max={np.max(np.abs(rig['opole_vel'])):.1f} rad/s"
        f"   ({int((peak >= WHIRL_RADPS).sum())}/{N} rows in the aliasing regime)"
    )
    print(f"  skipping rows 0..{SKIP_FIRST - 1} (startup artifacts); {N - 1 - SKIP_FIRST} transitions used")
    print("=" * 78)

    def set_state(i):
        """Hard-set env 0 to rig row i. inference_mode required: env.step() marks the data buffers
        as inference tensors, so the in-place joint-state write must run inside it too."""
        with torch.inference_mode():
            jp = _rb.data.joint_pos[0:1].clone()
            jv = _rb.data.joint_vel[0:1].clone()
            jp[0, _sidx] = float(rig["cart_pos"][i])
            jp[0, _iidx] = float(rig["ipole_ang"][i])
            jp[0, _oidx] = float(rig["opole_ang"][i])
            jv[0, _sidx] = float(rig["cart_vel"][i])
            jv[0, _iidx] = float(rig["ipole_vel"][i])
            jv[0, _oidx] = float(rig["opole_vel"][i])
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
            float(_rb.data.joint_pos[0, _iidx]),
            float(_rb.data.joint_vel[0, _iidx]),
            float(_rb.data.joint_pos[0, _oidx]),
            float(_rb.data.joint_vel[0, _oidx]),
            bool(dones[0]),
        )

    def rig_state(i):
        return tuple(float(rig[n][i]) for n in DIM_NAMES)

    # ---- one-step residual: drive i -> i+1 with force[i], compare sim next-state to rig next ----
    res, applied = [], []
    for i in range(SKIP_FIRST, N - 1):
        set_state(i)
        fi = rig["force"][i]
        *sim_next, done = step_force(fi)
        rig_next = rig_state(i + 1)
        res.append(
            [
                float(_wrap(sim_next[k] - rig_next[k])) if k in ANGLE_DIMS else sim_next[k] - rig_next[k]
                for k in range(6)
            ]
        )
        sim_show = tuple(float(_wrap(sim_next[k])) if k in ANGLE_DIMS else sim_next[k] for k in range(6))
        applied.append((i + 1, sim_show, rig_next, float(fi), float(rig["cart_vel"][i])))
        if done:
            print(f"  [WARN] env reset during transition {i} (widen bounds / episode_length).")
    res = np.array(res)

    # ---- per-dim bias: is it systematic? compare to the standard error, not to a sample floor ----
    mean, std = np.mean(res, axis=0), np.std(res, axis=0)
    sem = std / math.sqrt(max(len(res), 1))
    print("ONE-STEP RESIDUAL (sim_next - rig_next)   [bias vs its own standard error]")
    for k, name in enumerate(DIM_NAMES):
        t = abs(mean[k]) / sem[k] if sem[k] > 0 else 0.0
        verdict = "SYSTEMATIC" if t > 2.0 else "consistent with zero"
        print(
            f"  {name:>9} [{DIM_UNITS[k]:>5}] : bias={mean[k]:+.5f}  std={std[k]:.5f}  "
            f"sem={sem[k]:.5f}  |bias|/sem={t:5.2f} -> {verdict}"
        )
    print("  A bias under ~2 sem is noise no matter how it compares to the sensor resolution.")

    # ---- regime split: above WHIRL_RADPS the rig's own velocity estimate aliases ----
    print("-" * 78)
    print(f"BY REGIME (peak pole speed at transition start; >={WHIRL_RADPS:.0f} rad/s the RIG log itself aliases):")
    pk = peak[SKIP_FIRST : N - 1]
    for label, m in [
        (f"usable   peak|w|<{WHIRL_RADPS:.0f}", pk < WHIRL_RADPS),
        (f"aliased  peak|w|>={WHIRL_RADPS:.0f}", pk >= WHIRL_RADPS),
    ]:
        if m.any():
            mm = np.mean(res[m], axis=0)
            rr = np.sqrt(np.mean(res[m] ** 2, axis=0))
            print(f"  {label} : n={int(m.sum()):>3}")
            print(
                f"      cart_vel  bias={mm[1]:+.3f} rmse={rr[1]:.3f}   "
                f"ipole_vel bias={mm[3]:+.2f} rmse={rr[3]:.2f}   opole_vel bias={mm[5]:+.2f} rmse={rr[5]:.2f}"
            )

    # ---- which link carries the gap, measured against what the METHOD can actually resolve ----
    usable = pk < WHIRL_RADPS
    sl = slice(SKIP_FIRST, N - 1)
    if usable.any():
        print("-" * 78)
        print("WHICH LINK (usable regime). Only the velocity dims carry independent information:")
        print(f"  {'dim':<11}{'rmse':>9}{'meas':>9}{'lag':>9}{'floor':>9}{'excess':>8}")
        for k in VEL_DIMS:
            name = DIM_NAMES[k]
            rr = float(np.sqrt(np.mean(res[usable, k] ** 2)))
            floor, meas, lag = _method_floor(name, accel[name][sl][usable], dt_ctrl)
            print(f"  {name:<11}{rr:>9.4f}{meas:>9.4f}{lag:>9.4f}{floor:>9.4f}{rr / floor:>8.2f}")
        print("  excess = rmse / method floor. ~1 means this test cannot see anything at that joint;")
        print("  >>1 is real model error. More samples shrink the BIAS bars above, never this floor.")
        print("  dependent dims (angle == that link's velocity integrated -> no new information):")
        for ka, kv in ((0, 1), (2, 3), (4, 5)):
            ra = float(np.sqrt(np.mean(res[usable, ka] ** 2)))
            rv = float(np.sqrt(np.mean(res[usable, kv] ** 2)))
            ratio = ra / (rv * dt_ctrl) if rv > 0 else float("nan")
            print(f"    {DIM_NAMES[ka]:<10} rmse={ra:.5f} = {ratio:.2f} x ({DIM_NAMES[kv]} rmse * dt)")

        # th2 is RELATIVE, so th1+th2 is the ABSOLUTE outer orientation. If the sim puts the outer
        # link in the right place but divides it wrongly between the joints, the two residuals
        # anti-correlate and their sum collapses -> inertia ratio / h4 coupling, not a single pivot.
        print("-" * 78)
        print("JOINT SPLIT vs ABSOLUTE ERROR  (th2 is relative, so th1+th2 is the absolute outer):")
        for lbl, ka, kb, unit in (("angle", 2, 4, "rad"), ("vel", 3, 5, "rad/s")):
            a, b = res[usable, ka], res[usable, kb]
            s = a + b
            r = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")
            ra, rb, rs = (float(np.sqrt(np.mean(x**2))) for x in (a, b, s))
            print(
                f"  {lbl:>5} [{unit:>5}]: rmse inner={ra:.4f}  outer={rb:.4f}  SUM={rs:.4f}  "
                f"corr={r:+.3f}  collapse={1 - rs / max(ra, rb):.0%}"
            )
        print("  corr strongly NEGATIVE with the sum collapsing => the absolute pose is right and only")
        print("  the split between joints is wrong: suspect the inertia RATIO / the h4 coupling term.")
        print("  corr ~0 with no collapse => the two joints are independently wrong.")

    # ---- cart_vel residual binned by |cart_vel| -> exposes DCMotor velocity_limit derating ----
    print("-" * 78)
    print("CART_VEL residual binned by |cart_vel| (exposes DCMotor velocity_limit=2.65 derating):")
    cv_at = np.abs(rig["cart_vel"][SKIP_FIRST : N - 1])
    cv_res = res[:, 1]
    for a, b in zip([0.0, 0.5, 1.0, 1.5, 2.0], [0.5, 1.0, 1.5, 2.0, 10.0]):
        m = (cv_at >= a) & (cv_at < b)
        if m.any():
            print(
                f"|v| in [{a:.1f},{b:.1f}) m/s : n={int(m.sum()):>3}  mean cart_vel resid = {cv_res[m].mean():+.4f} m/s"
            )
    print("  (residual trending NEGATIVE as |v| rises => sim under-drives => raise velocity_limit.)")

    # ---- drive vs brake: derating only bites while accelerating (F*v>0) ----
    print("-" * 78)
    print("DRIVE vs BRAKE (sign of force*velocity; isolates the DCMotor velocity_limit derating):")
    f_app = rig["force"][SKIP_FIRST : N - 1]
    v0 = rig["cart_vel"][SKIP_FIRST : N - 1]
    push = np.sign(f_app) * res[:, 1]
    speed = np.abs(v0)
    brk = (f_app * v0) < 0.0
    if brk.any():
        print(
            f"  BRAKING      (F*v<0, derating N/A)  : n={int(brk.sum()):>3}  "
            f"mean push err = {push[brk].mean():+.4f} m/s   (expect ~0 if the motor curve is the issue)"
        )
    print("  ACCELERATING (F*v>0, derating bites) -- by cart speed:")
    for a, b in [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 10.0)]:
        m = (~brk) & (speed >= a) & (speed < b)
        if m.any():
            print(f"    |v| in [{a:.1f},{b:.1f}) m/s : n={int(m.sum()):>3}  mean push err = {push[m].mean():+.4f} m/s")
    print("  Fingerprint: BRAKING ~0 while ACCELERATING grows with speed => the motor curve.")
    print("  NEGATIVE = sim under-drives (velocity_limit too low), POSITIVE = over-drives (too high).")
    print("  BRAKING the SAME sign and size as ACCELERATING => not the motor curve at all, it is a")
    print("  constant offset -> see the friction/lag split below.")

    # ---- a constant offset is friction; one that scales with acceleration is estimator lag ----
    print("-" * 78)
    print("CART: FRICTION or ESTIMATOR LAG?  (push err vs |cart acceleration|)")
    a_cart = np.abs(accel["cart_vel"][SKIP_FIRST : N - 1])
    for lo, hi in [(0.0, 5.0), (5.0, 15.0), (15.0, 30.0), (30.0, 60.0), (60.0, float("inf"))]:
        m = (a_cart >= lo) & (a_cart < hi)
        if m.any():
            hi_s = "inf" if hi == float("inf") else f"{hi:.0f}"
            print(
                f"  |a| in [{lo:>3.0f},{hi_s:>3}) m/s^2 : n={int(m.sum()):>3}  "
                f"mean push err = {push[m].mean():+.4f} m/s"
            )
    if len(a_cart) > 2:
        c1, c0 = np.polyfit(a_cart, push, 1)
        print(f"  fit  push = c0 + c1*|a|   ->   c0 = {c0:+.4f} m/s    c1 = {c1 * 1000:+.3f} ms")
        print(f"    c0 -> {M_EFF_KG * c0 / dt_ctrl:+.2f} N of missing opposing force   [FRICTION, actionable]")
        print(f"    c1 -> an effective velocity lag of {c1 * 1000:+.2f} ms      [ESTIMATOR LAG, not a sim bug]")
        print(f"    for scale: half a control sample = {dt_ctrl * 500:.1f} ms; measured breakaway = 3.24 N.")
    print("  Flat bins with a large c0 => Coulomb friction is under-modelled (dynamic_params /")
    print("  viscous_value were calibrated on the SINGLE; the double has 1.42x the moving mass).")

    # ---- one-step CSV dump ----
    with open(os.path.join(OUT_DIR, CSV_ONESTEP), "w", newline="") as f:
        w = csv.writer(f)
        cols = ["step", "t_s", "force_n", "cart_vel_start", "cart_accel", "peak_pole_vel_start", "usable"]
        for name in DIM_NAMES:
            cols += [f"rig_{name}", f"sim_{name}", f"resid_{name}"]
        # th1+th2 = ABSOLUTE outer; plot these against resid_ipole_* to see the split vs pose error
        cols += ["resid_abs_opole_ang", "resid_abs_opole_vel"]
        w.writerow(cols)
        for j, (idx, sim_next, rig_next, force_n, cart_vel_start) in enumerate(applied):
            row = [
                idx,
                round(idx * dt_ctrl, 5),
                round(force_n, 5),
                round(cart_vel_start, 5),
                round(float(accel["cart_vel"][SKIP_FIRST + j]), 4),
                round(float(pk[j]), 4),
                int(pk[j] < WHIRL_RADPS),
            ]
            for k in range(6):
                row += [rig_next[k], sim_next[k], res[j, k]]
            row += [res[j, 2] + res[j, 4], res[j, 3] + res[j, 5]]
            w.writerow(row)

    # ---- open-loop divergence (seed once, replay forces free) ----
    print("=" * 78)
    print(f"OPEN-LOOP DIVERGENCE  (seed at row {SKIP_FIRST}, replay forces free)")
    set_state(SKIP_FIRST)
    roll = []
    for i in range(SKIP_FIRST, N - 1):
        *sim_next, done = step_force(rig["force"][i])
        sim_show = tuple(float(_wrap(sim_next[k])) if k in ANGLE_DIMS else sim_next[k] for k in range(6))
        roll.append((i + 1, rig_state(i + 1), sim_show))
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
            for k in range(6):
                row += [rt[k], st[k]]
            w.writerow(row)

    for h in [1, 5, 10, 20, 40]:
        if h <= len(roll):
            idx, rt, st = roll[h - 1]
            err = [float(_wrap(st[k] - rt[k])) if k in ANGLE_DIMS else st[k] - rt[k] for k in range(6)]
            print(
                f"  after {h:>2} steps ({h * dt_ctrl * 1000:>4.0f} ms): "
                f"d_cart={err[0]:+.4f} m / {err[1]:+.4f} m/s  |  "
                f"d_ipole={err[2]:+.4f} rad / {err[3]:+.2f} rad/s  |  "
                f"d_opole={err[4]:+.4f} rad / {err[5]:+.2f} rad/s"
            )

    print("=" * 78)
    print(f"  one-step CSV -> {os.path.join(OUT_DIR, CSV_ONESTEP)}")
    print(f"  rollout  CSV -> {os.path.join(OUT_DIR, CSV_ROLLOUT)}")
    print("  Reading it, in order: (1) any bias over 2 sem, (2) any velocity dim whose EXCESS over the")
    print("  method floor is >>1, (3) the friction/lag split on the cart. Filter to usable==1 first --")
    print("  above the aliasing threshold the rig's own velocity estimate is wrong, so a residual there")
    print("  says nothing about the sim. The open-loop block measures chaos, not fidelity; ignore it")
    print("  unless the trajectory stays slow.")
    print("=" * 78)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
