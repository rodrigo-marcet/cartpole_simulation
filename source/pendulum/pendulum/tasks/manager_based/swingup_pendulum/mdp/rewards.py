# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l1(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value using L1 norm."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    return torch.sum(torch.abs(joint_pos - target), dim=1)


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def joint_pos_target_cos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward upright pole position via cosine. Returns +1 when upright, -1 when hanging."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    return torch.sum(torch.cos(joint_pos), dim=1)


def joint_effort_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize large effort commands (action magnitude) using L2."""
    return torch.sum(env.action_manager.action**2, dim=1)


def action_rate_l2(env) -> torch.Tensor:
    """Penalize rapid changes between consecutive actions."""
    return torch.sum(
        (env.action_manager.action - env.action_manager.prev_action) ** 2,
        dim=1,
    )


def swingup_reward(
    env: ManagerBasedRLEnv,
    pole_cfg: SceneEntityCfg,
    cart_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """DeepMind-style multiplicative swingup reward."""
    asset: Articulation = env.scene[pole_cfg.name]

    # pole angle and angular velocity
    pole_pos = asset.data.joint_pos[:, pole_cfg.joint_ids[0]]  # θ (rad)
    pole_vel = asset.data.joint_vel[:, pole_cfg.joint_ids[0]]  # θ̇ (rad/s)

    # cart position
    cart_asset: Articulation = env.scene[cart_cfg.name]
    cart_pos = cart_asset.data.joint_pos[:, cart_cfg.joint_ids[0]]

    # raw action (after tanh, so already [-1, 1])
    action = env.action_manager.action[:, 0]

    # upright: cos(θ) mapped from [-1, 1] → [0, 1]
    upright = (torch.cos(pole_pos) + 1.0) / 2.0

    # centered: soft penalty on cart position, margin=2
    centered = torch.exp(-0.5 * (cart_pos / 2.0) ** 2)
    centered = (1.0 + centered) / 2.0  # [0.5, 1]

    # small_velocity: soft penalty on pole angular velocity, margin=5
    small_vel = torch.exp(-0.5 * (pole_vel / 5.0) ** 2)
    small_vel = (1.0 + small_vel) / 2.0  # [0.5, 1]

    # small_control: soft penalty on action magnitude, margin=1
    small_ctrl = torch.exp(-0.5 * (action / 1.0) ** 2)
    small_ctrl = (4.0 + small_ctrl) / 5.0  # [0.8, 1]

    return upright * centered * small_vel * small_ctrl  # [0, 1]


THETA_DOT_MAX = 2.625


def swingup_reward_v2(
    env: ManagerBasedRLEnv,
    pole_cfg: SceneEntityCfg,
    cart_cfg: SceneEntityCfg,
    theta_dot_max: float = THETA_DOT_MAX,  # catch ceiling, rad/s  (the placeholder)
    pole_mass: float = 0.020,
    weight_mass: float = 0.030,
    pole_length: float = 0.15,
    g: float = 9.81,
) -> torch.Tensor:
    """Energy-shaping swing-up reward: pump freely when low, CAP the energy as
    the pole rises, which forces it to arrive at the top slow enough to catch.

    Three multiplicative factors, each in [0, 1]:
      • upright     – climb gradient: rewards getting the pole up (fast learning,
                      same role as in your current reward).
      • energy_cap  – the "pump out" term: =1 while total energy ≤ E_top, and
                      decays once you OVER-pump. Because potential energy grows
                      as you rise, this tolerates high speed only when the pole
                      is low (where you need it to pump) and tightens to ~0 speed
                      at the very top. It is a position-aware speed limit.
      • center_gate – only requires the cart near center NEAR THE TOP, so cart
                      excursions during pumping are not punished.

    Control effort is intentionally NOT penalized here — your separate
    `effort_penalty` and `action_rate` reward terms already do that.
    """
    asset: Articulation = env.scene[pole_cfg.name]
    pole_pos = asset.data.joint_pos[:, pole_cfg.joint_ids[0]]  # θ : 0 = upright, π = down
    pole_vel = asset.data.joint_vel[:, pole_cfg.joint_ids[0]]  # θ̇ (rad/s)

    cart_asset: Articulation = env.scene[cart_cfg.name]
    cart_pos = cart_asset.data.joint_pos[:, cart_cfg.joint_ids[0]]  # x (m)

    # ── system constants (scalars) ────────────────────────────────────────
    total_m = pole_mass + weight_mass
    I = total_m * pole_length**2
    E_top = total_m * g * pole_length  # target energy: upright, at rest

    # ── (1) climb gradient ────────────────────────────────────────────────
    upright = (torch.cos(pole_pos) + 1.0) / 2.0  # 1 upright, 0 hanging

    # ── (2) one-sided energy cap (the "pump out" term) ────────────────────
    # E_potential = m g l cos θ  →  +E_top at top (θ=0), −E_top at bottom (θ=π)
    E_current = 0.5 * I * pole_vel**2 + total_m * g * pole_length * torch.cos(pole_pos)
    e_excess = torch.clamp((E_current - E_top) / E_top, min=0.0)  # 0 if ≤ target, >0 if over

    # sharpness c chosen so energy_cap = 0.5 exactly at the catch ceiling:
    # at the top, e_excess = 0.5 * I * theta_dot_max^2 / E_top
    e_ceiling = 0.5 * I * theta_dot_max**2 / E_top
    c_energy = math.log(2.0) / (e_ceiling**2)  # larger theta_dot_max → gentler cap
    energy_cap = torch.exp(-c_energy * e_excess**2)

    # ── (3) centering, gated to matter only near the top ──────────────────
    CATCH_X = 0.15  # m: keep the catch off the ±0.3 track ends
    centered = torch.exp(-0.5 * (cart_pos / CATCH_X) ** 2)
    b = torch.clamp((upright - 0.8) / 0.15, 0.0, 1.0)  # 0 far from top → 1 within ~25°
    center_gate = (1.0 - b) + b * centered  # =1 during pumping, tightens near the top

    return upright * energy_cap * center_gate  # [0, 1]


def swingup_reward_energy(
    env: ManagerBasedRLEnv,
    pole_cfg: SceneEntityCfg,
    cart_cfg: SceneEntityCfg,
    pole_mass: float = 0.020,
    weight_mass: float = 0.030,
    pole_length: float = 0.15,
    g: float = 9.81,
) -> torch.Tensor:
    asset: Articulation = env.scene[pole_cfg.name]
    pole_pos = asset.data.joint_pos[:, pole_cfg.joint_ids[0]]  # 0=upright, π=down
    pole_vel = asset.data.joint_vel[:, pole_cfg.joint_ids[0]]
    cart_asset: Articulation = env.scene[cart_cfg.name]
    cart_pos = cart_asset.data.joint_pos[:, cart_cfg.joint_ids[0]]

    total_m = pole_mass + weight_mass
    I = total_m * pole_length**2

    # Energy: potential is m*g*l*cos(θ)  →  +mgl at top, -mgl at bottom
    E_kinetic = 0.5 * I * pole_vel**2
    E_potential = total_m * g * pole_length * torch.cos(pole_pos)
    E_current = E_kinetic + E_potential

    # Target: potential at upright (θ=0) with zero velocity
    E_target = total_m * g * pole_length  # = mgl * cos(0) = mgl

    # Normalized energy error — negative when below target, positive when overshooting
    energy_error = (E_current - E_target) / E_target
    energy_reward = torch.exp(-4.0 * energy_error**2)

    # ── Upright and balance terms ─────────────────────────────────────────────
    upright = (torch.cos(pole_pos) + 1.0) / 2.0  # 1 at θ=0, 0 at θ=π  ✓

    small_vel = (1.0 + torch.exp(-0.5 * (pole_vel / 3.0) ** 2)) / 2.0
    cart_center = torch.exp(-0.5 * (cart_pos / 0.12) ** 2)
    centered = (1.0 + cart_center) / 2.0
    balance_reward = upright * small_vel * centered

    # ── Smooth blend: energy shaping → balance ────────────────────────────────
    # blend=0 → pure energy reward;  blend=1 → pure balance reward
    blend = torch.clamp((upright - 0.7) / 0.15, 0.0, 1.0)
    return (1.0 - blend) * energy_reward * cart_center + blend * balance_reward


CATCH_OMEGA = 3.5  # rad/s : max pole speed the balance net can catch
CATCH_ANGLE_DEG = 20.0  # deg   : angle half-width of the handoff basin


def swingup_reward_gated(
    env: ManagerBasedRLEnv,
    pole_cfg: SceneEntityCfg,
    cart_cfg: SceneEntityCfg,
    catch_omega: float = CATCH_OMEGA,
    catch_angle_deg: float = CATCH_ANGLE_DEG,
    deliver_bonus: float = 3.0,  # how much the basin "jackpot" outweighs the climb shaping
) -> torch.Tensor:
    """Swing-up reward for a GATED architecture. Its ONLY job is to DELIVER the
    pole into the catch basin of your existing balance network — it does NOT try
    to balance, so it must NOT throttle corrective velocity (that was the v2 bug).

    reward = climb  +  deliver_bonus * in_basin
    • climb    : dense; pulls the pole up and keeps the cart off the walls.
                    Velocity is NOT penalized here, so pumping is free.
    • in_basin : ~1 only when (near upright) AND (speed <= catch ceiling),
                    i.e. exactly a state your balance net can take over from.
    """
    asset = env.scene[pole_cfg.name]
    pole_pos = asset.data.joint_pos[:, pole_cfg.joint_ids[0]]  # θ : 0 = upright, π = down
    pole_vel = asset.data.joint_vel[:, pole_cfg.joint_ids[0]]  # θ̇ (rad/s)
    cart_pos = env.scene[cart_cfg.name].data.joint_pos[:, cart_cfg.joint_ids[0]]  # x (m)

    # ── dense climb shaping (gets it up; pumping/corrective speed unpenalized) ──
    upright = (torch.cos(pole_pos) + 1.0) / 2.0  # 1 up, 0 down
    centered = torch.exp(-0.5 * (cart_pos / 0.15) ** 2)  # leave room for the handoff
    climb = upright * centered  # [0, 1]

    # ── deliverable-basin bonus (the actual objective) ──
    theta_basin = math.radians(catch_angle_deg)
    ang = wrap_to_pi(pole_pos)  # measure angle from upright
    angle_ok = torch.exp(-0.5 * (ang / theta_basin) ** 2)  # ~1 inside the angle basin
    speed_ok = torch.exp(-0.5 * (pole_vel / catch_omega) ** 2)  # ~1 under the catch ceiling
    in_basin = angle_ok * speed_ok  # [0, 1]

    return climb + deliver_bonus * in_basin  # [0, ~4]


def swingup_reward_energy_pump(
    env: ManagerBasedRLEnv,
    pole_cfg: SceneEntityCfg,
    cart_cfg: SceneEntityCfg,
    pole_mass: float = 0.020,
    weight_mass: float = 0.030,
    pole_length: float = 0.15,
    g: float = 9.81,
    energy_excess: float = 0.05,  # aim a hair ABOVE E_top so it reliably crests (≈2.6 rad/s arrival)
    k_energy: float = 4.0,  # sharpness of the energy target (higher = punishes overshoot harder)
    arrive_omega: float = 3.0,  # rad/s : arrival-speed tolerance for the bonus (keep < your PID catch ceiling)
    arrive_angle_deg: float = 15.0,
    arrive_bonus: float = 2.0,  # weight of the "at the top, slow, centered" jackpot
    handoff_x: float = 0.05,  # m : how tightly to center the cart AT the handoff
) -> torch.Tensor:
    """Swing-up-ONLY reward for a gated NN -> PID system.

    Objective: pump the pole's total mechanical energy to (just above) the
    upright value E_top — i.e. onto the homoclinic orbit — so it arrives at the
    top with ~zero angular velocity, with the cart near center, using minimal
    travel. Balancing is the PID's job, so there is NOTHING here that suppresses
    or holds the top; this is exactly what energy shaping is designed to do.

    reward = r_energy  +  arrive_bonus * arrive
    • r_energy : DENSE everywhere. Pumps energy up to the target AND penalizes
                    OVER-pumping (which is what makes the pole spin). Not gated by
                    cart position, so pumping inside the ±0.1 m window is free.
    • arrive   : the handoff jackpot — at the top, slow, and centered.
    """
    asset = env.scene[pole_cfg.name]
    pole_pos = asset.data.joint_pos[:, pole_cfg.joint_ids[0]]  # θ : 0 = up, π = down
    pole_vel = asset.data.joint_vel[:, pole_cfg.joint_ids[0]]  # θ̇ (rad/s)
    cart_pos = env.scene[cart_cfg.name].data.joint_pos[:, cart_cfg.joint_ids[0]]  # x (m)

    total_m = pole_mass + weight_mass
    I = total_m * pole_length**2
    E_top = total_m * g * pole_length

    # total pole energy ; mgl·cosθ = E_top·cosθ  (+E_top at top, −E_top at bottom)
    E = 0.5 * I * pole_vel**2 + E_top * torch.cos(pole_pos)
    E_target = E_top * (1.0 + energy_excess)
    e = (E - E_target) / E_top  # ≈0 ⇒ on the orbit that reaches the top and (nearly) stops

    # (1) energy targeting — dense pumping gradient + anti-overshoot (no spinning)
    r_energy = torch.exp(-k_energy * e**2)

    # (2) arrival jackpot — at the top, slow, centered for the PID to grab
    ang = wrap_to_pi(pole_pos)
    at_top = torch.exp(-0.5 * (ang / math.radians(arrive_angle_deg)) ** 2)
    slow = torch.exp(-0.5 * (pole_vel / arrive_omega) ** 2)
    centered = torch.exp(-0.5 * (cart_pos / handoff_x) ** 2)
    arrive = at_top * slow * centered

    return r_energy + arrive_bonus * arrive
