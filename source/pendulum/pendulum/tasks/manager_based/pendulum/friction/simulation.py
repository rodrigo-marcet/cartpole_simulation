# test_step_response.py
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv
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
    slider_idx, _ = robot.find_joints("slider_to_cart")

    # set initial velocity, zero position
    root_state = robot.data.default_root_state.clone()
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()

    # joint_vel[0, slider_idx[0]] = 1.1  # m/s
    # joint_pos[0, slider_idx[0]] = -.4036  # m/s
    joint_vel[0, slider_idx[0]] = 2.1  # m/s
    joint_pos[0, slider_idx[0]] = -0.176  # m/s

    robot.write_root_state_to_sim(root_state)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.update(sim.get_physics_dt())

    LOG_EVERY = 10
    dt = sim.get_physics_dt()

    pos = robot.data.joint_pos[0, slider_idx[0]].item()
    vel = robot.data.joint_vel[0, slider_idx[0]].item()
    log = []
    i = 0
    # for i in range(steps):
    while vel > 0.05:
        t = i * dt

        sim.step()
        robot.update(dt)

        if i % LOG_EVERY == 0:
            pos = robot.data.joint_pos[0, slider_idx[0]].item()
            vel = robot.data.joint_vel[0, slider_idx[0]].item()
            log.append([t + 0.01, pos, vel])

        i += 1

    # write CSV
    with open("step_response_sim.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "cart_pos", "cart_vel"])
        writer.writerows(log)

    print("Done. Saved to step_response_sim.csv")
    simulation_app.close()


if __name__ == "__main__":
    main()
