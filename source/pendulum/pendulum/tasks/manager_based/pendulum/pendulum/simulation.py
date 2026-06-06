# test_step_response.py
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv
import math
import sys

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

sys.path.insert(0, r"D:/omniverse/pendulum/source/pendulum")

from pendulum.tasks.manager_based.pendulum import mdp


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1 / 1000))
    sim.set_camera_view([0.0, -5.0, 3.0], [0.0, 0.0, 2.0])

    # add light and ground
    sim_utils.DomeLightCfg(intensity=500.0, color=(0.9, 0.9, 0.9)).func(
        "/World/DomeLight", sim_utils.DomeLightCfg(intensity=500.0)
    )
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())

    # spawn robot
    robot_cfg = mdp.FUSION_CARTPOLE_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)

    sim.reset()
    robot.update(sim.get_physics_dt())

    # find joint indices
    pole_idx, _ = robot.find_joints("cart_to_pole")

    # set initial velocity, zero position
    root_state = robot.data.default_root_state.clone()
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()

    joint_vel[0, pole_idx[0]] = 0.067465  # m/s
    joint_pos[0, pole_idx[0]] = 0.288913  # m/s

    robot.write_root_state_to_sim(root_state)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.update(sim.get_physics_dt())

    # STEPS = 121910
    STEPS = 121910
    LOG_EVERY = 10
    dt = sim.get_physics_dt()

    angle = robot.data.joint_pos[0, pole_idx[0]].item()
    vel = robot.data.joint_vel[0, pole_idx[0]].item()
    log = []
    i = 0
    # for i in range(steps):
    # while(vel > 0.005):
    # while True:
    while i < STEPS:
        sim.step()
        robot.update(dt)

        angle = robot.data.joint_pos[0, pole_idx[0]].item()
        vel = robot.data.joint_vel[0, pole_idx[0]].item()

        if i % LOG_EVERY == 0:
            t = i * dt
            pole_cos = math.cos(angle)
            pole_sin = math.sin(angle)
            log.append([t, pole_cos, pole_sin, angle, vel])

        i += 1

    # write CSV
    with open("step_response_sim.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dt", "cos", "sin", "angle", "angular_vel"])
        writer.writerows(log)

    print("Done. Saved to step_response_sim.csv")
    simulation_app.close()


if __name__ == "__main__":
    main()
