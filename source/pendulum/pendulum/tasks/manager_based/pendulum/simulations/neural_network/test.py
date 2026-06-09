# ruff: noqa: E501
import argparse
import random

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import sys

import torch
import torch.nn as nn

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

sys.path.insert(0, r"D:/omniverse/pendulum/source/pendulum")

from pendulum.tasks.manager_based.pendulum import mdp

POLICY_PATH = r"D:/omniverse/pendulum/logs/eridani/weights/skrl_pendulum_v0_061/skrl_pendulum_v0_061/skrl/pendulum_direct/2026-06-08_18-44-46_ppo_torch/checkpoints/best_agent.pt"
DURATION = 10.0
DECIMATION = 10


def _tick_noise(resolution: float, max_ticks: int = 3) -> float:
    ticks = torch.arange(-max_ticks, max_ticks + 1).float()
    weights = (max_ticks + 1 - ticks.abs()).float()
    idx = torch.multinomial(weights, num_samples=1).item()
    return ticks[idx].item() * resolution


def get_obs(robot, slider_idx, pole_idx) -> torch.Tensor:
    cart_pos = robot.data.joint_pos[0, slider_idx].item()
    cart_vel = robot.data.joint_vel[0, slider_idx].item()
    pole_angle = robot.data.joint_pos[0, pole_idx].item()
    pole_vel = robot.data.joint_vel[0, pole_idx].item()

    res_pole = (2 * math.pi) / 4096
    pole_angle_q = round(pole_angle / res_pole) * res_pole
    pole_angle_noisy = pole_angle_q + _tick_noise(res_pole)

    res_cart_vel = 0.8 / 16384
    res_pole_vel = (2 * math.pi) / 4096

    return torch.tensor(
        [
            [
                cart_pos,
                cart_vel + _tick_noise(res_cart_vel),
                math.sin(pole_angle_noisy),
                math.cos(pole_angle_noisy),
                pole_vel + _tick_noise(res_pole_vel),
            ]
        ],
        dtype=torch.float32,
    )


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1 / 1000))
    sim.set_camera_view([0.0, -5.0, 3.0], [0.0, 0.0, 2.0])

    sim_utils.DomeLightCfg(intensity=500.0, color=(0.9, 0.9, 0.9)).func(
        "/World/DomeLight", sim_utils.DomeLightCfg(intensity=500.0)
    )
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())

    robot_cfg = mdp.FUSION_CARTPOLE_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)

    sim.reset()
    robot.update(sim.get_physics_dt())

    slider_idx, _ = robot.find_joints("slider_to_cart")
    pole_idx, _ = robot.find_joints("cart_to_pole")
    slider_idx = slider_idx[0]
    pole_idx = pole_idx[0]

    # --- load policy ---
    model = nn.Sequential(nn.Linear(5, 32), nn.ELU(), nn.Linear(32, 32), nn.ELU(), nn.Linear(32, 1), nn.Tanh())

    checkpoint = torch.load(POLICY_PATH, map_location="cpu")
    policy_state = checkpoint["policy"]

    model.load_state_dict(
        {
            "0.weight": policy_state["net_container.0.weight"],
            "0.bias": policy_state["net_container.0.bias"],
            "2.weight": policy_state["net_container.2.weight"],
            "2.bias": policy_state["net_container.2.bias"],
            "4.weight": policy_state["policy_layer.weight"],
            "4.bias": policy_state["policy_layer.bias"],
        }
    )
    model.eval()

    # --- load scaler ---
    # if this key errors, print checkpoint.keys() to find the right one
    scaler_state = checkpoint["state_preprocessor"]
    running_mean = scaler_state["running_mean"].float()
    running_var = scaler_state["running_variance"].float()

    # --- initial state ---
    root_state = robot.data.default_root_state.clone()
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()

    # randomize cart
    joint_pos[0, slider_idx] += random.uniform(-0.2, 0.2)
    joint_vel[0, slider_idx] += random.uniform(-0.2, 0.2)

    # randomize pole
    # joint_pos[0, pole_idx] += random.uniform(-0.25 * math.pi, 0.25 * math.pi)
    # joint_vel[0, pole_idx] += random.uniform(-0.25 * math.pi, 0.25 * math.pi)
    joint_pos[0, pole_idx] += random.uniform(-0.1 * math.pi, 0.1 * math.pi)
    joint_vel[0, pole_idx] += random.uniform(-0.1 * math.pi, 0.1 * math.pi)

    robot.write_root_state_to_sim(root_state)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.update(sim.get_physics_dt())

    effort = torch.zeros_like(robot.data.joint_effort_target)
    dt = sim.get_physics_dt()
    steps = int(DURATION / dt)

    for i in range(steps):
        if i % DECIMATION == 0:
            obs = get_obs(robot, slider_idx, pole_idx)
            obs_scaled = (obs - running_mean) / (running_var.sqrt() + 1e-8)
            with torch.no_grad():
                action = model(obs_scaled)
            force = action[0, 0].item() * 40.0
            torque = force * 0.01  # F * r

            cart_pos = obs[0, 0].item()
            cart_vel = obs[0, 1].item()
            pole_sin = obs[0, 2].item()
            pole_cos = obs[0, 3].item()
            pole_vel = obs[0, 4].item()
            pole_angle = math.atan2(pole_sin, pole_cos)

            print(
                f"t={i * dt:6.3f} | "
                f"cart_pos={cart_pos:+.4f}m  "
                f"cart_vel={cart_vel:+.4f}m/s  "
                f"pole={math.degrees(pole_angle):+7.3f}deg  "
                f"sin={pole_sin:+.4f}  "
                f"cos={pole_cos:+.4f}  "
                f"pole_vel={pole_vel:+.4f}rad/s  "
                f"action={action[0, 0].item():+7.4f}N  "
                f"force={force:+7.3f}N  "
                f"torque={torque:+.5f}Nm"
            )

            effort[0, slider_idx] = force
            robot.set_joint_effort_target(effort)
            robot.write_data_to_sim()

        sim.step()
        robot.update(dt)

    simulation_app.close()


if __name__ == "__main__":
    main()
