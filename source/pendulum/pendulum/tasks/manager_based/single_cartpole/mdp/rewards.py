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
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
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
    asset: Articulation = env.scene[pole_cfg.name]
    pole_pos = asset.data.joint_pos[:, pole_cfg.joint_ids[0]]
    pole_vel = asset.data.joint_vel[:, pole_cfg.joint_ids[0]]

    cart_asset: Articulation = env.scene[cart_cfg.name]
    cart_pos = cart_asset.data.joint_pos[:, cart_cfg.joint_ids[0]]

    total_m = pole_mass + weight_mass
    I = total_m * pole_length**2
    E_top = total_m * g * pole_length

    upright = (torch.cos(pole_pos) + 1.0) / 2.0

    E_current = 0.5 * I * pole_vel**2 + total_m * g * pole_length * torch.cos(pole_pos)
    e_excess = torch.clamp((E_current - E_top) / E_top, min=0.0)

    e_ceiling = 0.5 * I * theta_dot_max**2 / E_top
    c_energy = math.log(2.0) / (e_ceiling**2)
    energy_cap = torch.exp(-c_energy * e_excess**2)

    CATCH_X = 0.15
    centered = torch.exp(-0.5 * (cart_pos / CATCH_X) ** 2)
    b = torch.clamp((upright - 0.8) / 0.15, 0.0, 1.0)
    center_gate = (1.0 - b) + b * centered

    return upright * energy_cap * center_gate


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

    E_kinetic = 0.5 * I * pole_vel**2
    E_potential = total_m * g * pole_length * torch.cos(pole_pos)
    E_current = E_kinetic + E_potential

    E_target = total_m * g * pole_length  # = mgl * cos(0) = mgl

    energy_error = (E_current - E_target) / E_target
    energy_reward = torch.exp(-4.0 * energy_error**2)

    upright = (torch.cos(pole_pos) + 1.0) / 2.0

    small_vel = (1.0 + torch.exp(-0.5 * (pole_vel / 3.0) ** 2)) / 2.0
    cart_center = torch.exp(-0.5 * (cart_pos / 0.12) ** 2)
    centered = (1.0 + cart_center) / 2.0
    balance_reward = upright * small_vel * centered

    blend = torch.clamp((upright - 0.7) / 0.15, 0.0, 1.0)
    return (1.0 - blend) * energy_reward * cart_center + blend * balance_reward


CATCH_OMEGA = 3.5
CATCH_ANGLE_DEG = 20.0


def swingup_reward_gated(
    env: ManagerBasedRLEnv,
    pole_cfg: SceneEntityCfg,
    cart_cfg: SceneEntityCfg,
    catch_omega: float = CATCH_OMEGA,
    catch_angle_deg: float = CATCH_ANGLE_DEG,
    deliver_bonus: float = 3.0,
) -> torch.Tensor:
    asset = env.scene[pole_cfg.name]
    pole_pos = asset.data.joint_pos[:, pole_cfg.joint_ids[0]]
    pole_vel = asset.data.joint_vel[:, pole_cfg.joint_ids[0]]
    cart_pos = env.scene[cart_cfg.name].data.joint_pos[:, cart_cfg.joint_ids[0]]

    upright = (torch.cos(pole_pos) + 1.0) / 2.0
    centered = torch.exp(-0.5 * (cart_pos / 0.15) ** 2)
    climb = upright * centered

    theta_basin = math.radians(catch_angle_deg)
    ang = wrap_to_pi(pole_pos)
    angle_ok = torch.exp(-0.5 * (ang / theta_basin) ** 2)
    speed_ok = torch.exp(-0.5 * (pole_vel / catch_omega) ** 2)
    in_basin = angle_ok * speed_ok

    return climb + deliver_bonus * in_basin  # [0, ~4]


def swingup_reward_energy_pump(
    env: ManagerBasedRLEnv,
    pole_cfg: SceneEntityCfg,
    cart_cfg: SceneEntityCfg,
    pole_mass: float = 0.020,
    weight_mass: float = 0.030,
    pole_length: float = 0.15,
    g: float = 9.81,
    energy_excess: float = 0.05,
    k_energy: float = 4.0,
    arrive_omega: float = 3.0,
    arrive_angle_deg: float = 15.0,
    arrive_bonus: float = 2.0,
    handoff_x: float = 0.05,
) -> torch.Tensor:
    asset = env.scene[pole_cfg.name]
    pole_pos = asset.data.joint_pos[:, pole_cfg.joint_ids[0]]
    pole_vel = asset.data.joint_vel[:, pole_cfg.joint_ids[0]]
    cart_pos = env.scene[cart_cfg.name].data.joint_pos[:, cart_cfg.joint_ids[0]]

    total_m = pole_mass + weight_mass
    I = total_m * pole_length**2
    E_top = total_m * g * pole_length

    E = 0.5 * I * pole_vel**2 + E_top * torch.cos(pole_pos)
    E_target = E_top * (1.0 + energy_excess)
    e = (E - E_target) / E_top

    r_energy = torch.exp(-k_energy * e**2)

    ang = wrap_to_pi(pole_pos)
    at_top = torch.exp(-0.5 * (ang / math.radians(arrive_angle_deg)) ** 2)
    slow = torch.exp(-0.5 * (pole_vel / arrive_omega) ** 2)
    centered = torch.exp(-0.5 * (cart_pos / handoff_x) ** 2)
    arrive = at_top * slow * centered

    return r_energy + arrive_bonus * arrive


OMEGA_N = 8.84


def swingup_reward_faithful(
    env: ManagerBasedRLEnv,
    pole_cfg: SceneEntityCfg,
    cart_cfg: SceneEntityCfg,
    omega_n: float = OMEGA_N,
    energy_excess: float = 0.05,
    k_energy: float = 4.0,
    arrive_omega: float = 3.0,
    arrive_angle_deg: float = 15.0,
    arrive_bonus: float = 2.0,
    handoff_x: float = 0.05,
) -> torch.Tensor:
    asset = env.scene[pole_cfg.name]
    pole_pos = asset.data.joint_pos[:, pole_cfg.joint_ids[0]]
    pole_vel = asset.data.joint_vel[:, pole_cfg.joint_ids[0]]
    cart_pos = env.scene[cart_cfg.name].data.joint_pos[:, cart_cfg.joint_ids[0]]

    e_norm = 0.5 * (pole_vel / omega_n) ** 2 + torch.cos(pole_pos)
    e = e_norm - (1.0 + energy_excess)
    r_energy = torch.exp(-k_energy * e**2)

    ang = wrap_to_pi(pole_pos)
    at_top = torch.exp(-0.5 * (ang / math.radians(arrive_angle_deg)) ** 2)
    slow = torch.exp(-0.5 * (pole_vel / arrive_omega) ** 2)
    centered = torch.exp(-0.5 * (cart_pos / handoff_x) ** 2)
    arrive = at_top * slow * centered

    return r_energy + arrive_bonus * arrive


def swingup_reward_unified(
    env: ManagerBasedRLEnv,
    pole_cfg: SceneEntityCfg,
    cart_cfg: SceneEntityCfg,
    omega_n: float = 8.9,
    energy_excess: float = 0.0,
    k_energy: float = 4.0,
    balance_bonus: float = 2.5,
    upright_angle_deg: float = 25.0,
    pole_vel_margin: float = 2.5,
    cart_vel_margin: float = 0.6,
    cart_pos_margin: float = 0.15,
) -> torch.Tensor:
    asset: Articulation = env.scene[pole_cfg.name]
    pole_pos = asset.data.joint_pos[:, pole_cfg.joint_ids[0]]
    pole_vel = asset.data.joint_vel[:, pole_cfg.joint_ids[0]]
    cart_asset: Articulation = env.scene[cart_cfg.name]
    cart_pos = cart_asset.data.joint_pos[:, cart_cfg.joint_ids[0]]
    cart_vel = cart_asset.data.joint_vel[:, cart_cfg.joint_ids[0]]

    # (1) energy shaping to swing up: peaks on the upright homoclinic orbit (dimensionless energy)
    e_norm = 0.5 * (pole_vel / omega_n) ** 2 + torch.cos(pole_pos)
    r_energy = torch.exp(-k_energy * (e_norm - (1.0 + energy_excess)) ** 2)

    # (2) smooth, damped balance basin: upright AND slow pole AND slow cart AND centered
    ang = wrap_to_pi(pole_pos)
    at_top = torch.exp(-0.5 * (ang / math.radians(upright_angle_deg)) ** 2)
    pole_still = torch.exp(-0.5 * (pole_vel / pole_vel_margin) ** 2)
    cart_still = torch.exp(-0.5 * (cart_vel / cart_vel_margin) ** 2)
    centered = torch.exp(-0.5 * (cart_pos / cart_pos_margin) ** 2)
    r_balance = at_top * pole_still * cart_still * centered

    return r_energy + balance_bonus * r_balance


def swingup_reward_quadratic(
    env: ManagerBasedRLEnv,
    pole_cfg: SceneEntityCfg,
    cart_cfg: SceneEntityCfg,
    w_angle: float = 0.5,
    w_cart: float = 0.3,
    w_effort: float = 0.2,
    x_max: float = 0.35,
) -> torch.Tensor:
    asset: Articulation = env.scene[pole_cfg.name]
    pole_pos = asset.data.joint_pos[:, pole_cfg.joint_ids[0]]
    cart_pos = env.scene[cart_cfg.name].data.joint_pos[:, cart_cfg.joint_ids[0]]
    u = env.action_manager.action[:, 0]

    theta = wrap_to_pi(pole_pos)  # 0 upright, +/-pi hanging
    cost = w_angle * (theta / math.pi) ** 2 + w_cart * (cart_pos / x_max) ** 2 + w_effort * u**2
    return 1.0 - cost
