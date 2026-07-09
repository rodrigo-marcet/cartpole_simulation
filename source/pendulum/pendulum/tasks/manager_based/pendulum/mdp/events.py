from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

__all__ = [
    "randomize_slider_friction_effort",
    "randomize_pole_vel_noise_std",
    "reset_joints_by_offset_unclamped",
    "reset_pole_position_curriculum",
]


def randomize_slider_friction_effort(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    static_range: tuple[float, float] = (1.7, 2.0),  # was (1.5, 2.0)
    dynamic_params: tuple[float, float] = (1.7, 0.17),  # was (1.18, 0.12); scales with effective mass
    viscous_value: float = 0.0,
) -> None:
    """Per-env reset DR for the slider joint's 3-component friction (effort model, N).

    Isaac Sim 5.x models joint friction as an *effort* (force in Newtons) with three
    INDEPENDENT channels stored per DOF as ``(static, dynamic, viscous)``:
      * static  = breakaway effort that must be exceeded to start moving,
      * dynamic = Coulomb effort opposing motion once sliding (constant),
      * viscous = effort proportional to joint velocity.

    The built-in ``mdp.randomize_joint_parameters`` takes a SINGLE
    ``friction_distribution_params`` and samples static AND dynamic from that same
    tuple, so it cannot represent stiction (our rig has static ~1.8 N > dynamic ~1.18 N).
    This term samples the channels independently and writes them straight to the PhysX
    articulation view via ``set_dof_friction_properties`` -- the exact API path proven
    to reach play's GPU sim in ``scripts/skrl/play_solution.py``.

    Calibrated from bench data (see analysis/FRICTION_DR_REPORT.md):
      static  ~ uniform(1.7, 2.0) N   (breakaway torque 0.015-0.020 N*m / r=0.01 m)
      dynamic ~ gaussian(1.7, 0.17) N (coast-down decel 3.57 m/s^2 * EFFECTIVE mass ~0.48 kg)
      viscous = 0                      (coast is linear, R^2=0.995)

    NOTE: the Coulomb EFFORT scales with EFFECTIVE mass, which includes the drivetrain
    armature (~0.15 kg motor rotor + pulley + belt, set on cart_actuator in
    configurations.py) -- NOT just the 0.33 kg body mass. Values computed against the
    0.33 kg body alone (old: dynamic 1.18 N, static (1.5,2.0)) made the cart accelerate
    ~4-5x too fast under drive, because the coast test only fixes the ratio F_dyn/m = 3.57.

    Args:
        asset_cfg: the robot, with ``joint_names=["slider_to_cart"]``.
        static_range: (low, high) for the uniform static/breakaway effort [N].
        dynamic_params: (mean, std) for the gaussian dynamic/Coulomb effort [N].
        viscous_value: constant viscous coefficient (0 keeps coast linear).
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve the slider DOF index on this articulation (explicit, resolve-timing safe)
    slider_ids, _ = asset.find_joints(asset_cfg.joint_names)
    slider_idx = slider_ids[0]

    n = len(env_ids)
    device = asset.device

    # --- sample the three channels per resetting env ---
    static = torch.empty(n, device=device).uniform_(static_range[0], static_range[1])
    dynamic = torch.normal(mean=dynamic_params[0], std=dynamic_params[1], size=(n,), device=device)
    dynamic = dynamic.clamp_min(0.0)
    # PhysX requires dynamic <= static (sliding <= breakaway). Enforce per sample,
    # else the low tail of `static` can dip under the high tail of `dynamic`.
    dynamic = torch.minimum(dynamic, static)
    viscous = torch.full((n,), float(viscous_value), device=device)

    # --- write via the Isaac Lab wrapper (channels: static, dynamic, viscous) ---
    # IMPORTANT: use write_joint_friction_coefficient_to_sim, NOT a raw
    # root_physx_view.set_dof_friction_properties. The wrapper also updates
    # asset.data.joint_{,dynamic_,viscous_}friction_coeff. That matters because a LATER reset
    # event that calls randomize_joint_parameters (here: randomize_pole_friction) rewrites the
    # WHOLE friction-props array from those data buffers -- a raw physx-view write would be
    # silently CLOBBERED back to the buffer's stale value (the USDA's legacy jointFriction),
    # so the slider would run ~frictionless / at the wrong value. Verified via play_friction.py.
    asset.write_joint_friction_coefficient_to_sim(
        static.unsqueeze(-1),
        joint_dynamic_friction_coeff=dynamic.unsqueeze(-1),
        joint_viscous_friction_coeff=viscous.unsqueeze(-1),
        joint_ids=[slider_idx],
        env_ids=env_ids,
    )


def randomize_pole_vel_noise_std(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mean: float = 0.4,
    std: float = 0.122,
    clip: tuple[float, float] = (0.05, 0.8),
) -> None:
    """Per-episode DR for the pole angular-velocity SENSOR noise std (rad/s).

    Samples a per-env noise std ~ N(mean, std) at reset and stores it on the env as
    ``env.pole_vel_noise_std`` (shape ``(num_envs, 1)``). ``mdp.pole_angular_vel_noisy``
    reads that buffer and applies the (within-episode constant) Gaussian noise level to the
    pole velocity observation, so different episodes see different sensor-noise magnitudes.

    mean=0.4, std=0.122 -> ~90% of episodes in [0.2, 0.6] (5th-95th pct = mean +/- 1.645*std),
    matching the rig's measured/behavioural pole-velocity noise. ``clip`` guards the tails so
    the std stays strictly positive.

    Unlike the joint-property DR terms this writes NO physics -- it only updates the buffer the
    observation function reads. It also needs no ``asset_cfg``.
    """
    device = env.device
    if getattr(env, "pole_vel_noise_std", None) is None:
        env.pole_vel_noise_std = torch.full((env.num_envs, 1), float(mean), device=device)
    n = len(env_ids)
    samp = torch.normal(mean=mean, std=std, size=(n, 1), device=device).clamp_(clip[0], clip[1])
    env.pole_vel_noise_std[env_ids] = samp


def reset_joints_by_offset_unclamped(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
) -> None:
    """Reset joints to default +/- a uniform offset, writing velocity WITHOUT the velocity clamp.

    Isaac Lab's built-in ``mdp.reset_joints_by_offset`` clamps the sampled joint velocity to the
    joint's velocity limit. For the PASSIVE pole (ImplicitActuator with no ``velocity_limit``)
    that limit is 0, so the configured ``velocity_range`` was silently zeroed -- the pole always
    started at rest even though DR asked for +/-1 rad/s. Seen in play_dr_check: ``pole_vel`` came
    back FIXED at 0 while ``pole_pos`` spread. This variant writes the sampled position AND
    velocity straight to sim (no soft clamp); PhysX still enforces its own hard joint limits,
    which for the pole are far above +/-1 rad/s (the free swing reaches ~18 rad/s).

    Position is NOT clamped here either -- intended for the pole, whose position limits are wide
    (+/-1e6). Do not reuse on a tightly position-limited joint without adding a position clamp.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ids, _ = asset.find_joints(asset_cfg.joint_names)
    pos = asset.data.default_joint_pos[env_ids][:, ids].clone()
    vel = asset.data.default_joint_vel[env_ids][:, ids].clone()
    pos += torch.empty_like(pos).uniform_(position_range[0], position_range[1])
    vel += torch.empty_like(vel).uniform_(velocity_range[0], velocity_range[1])
    asset.write_joint_state_to_sim(pos, vel, joint_ids=ids, env_ids=env_ids)


def reset_pole_position_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    initial_range: float = 0.2,
    final_range: float = math.pi,
    curriculum_steps: int = 50000,
) -> None:
    """Balance-task helper (currently UNUSED): widen the pole reset range over training.

    Kept from the original balance task. Not wired into pendulum_env_cfg.py (the balance reset
    uses a fixed near-upright range). Left here so any external script that imports it keeps
    working; delete if you're sure nothing references it.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    progress = min(env.common_step_counter / curriculum_steps, 1.0)
    current_range = initial_range + (final_range - initial_range) * progress

    if asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids

    joint_pos = asset.data.default_joint_pos[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_vel = asset.data.default_joint_vel[iter_env_ids, asset_cfg.joint_ids].clone()

    joint_pos += torch.zeros_like(joint_pos).uniform_(-current_range, current_range)

    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
