# test_step_response.py

import argparse
import csv
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

sys.path.insert(0, r"D:/omniverse/pendulum/source/pendulum")

from pendulum.tasks.manager_based.swingup_pendulum import mdp

# --------------------------------------------------------------------------
# Pendulum energy parameters
# --------------------------------------------------------------------------

m = 0.03  # kg
L = 0.125  # m
g = 9.81  # m/s²

I = m * L**2
mgL = m * g * L


def compute_energy(angle: float, angular_vel: float) -> float:
    """
    Angle convention: 0 = upright (inverted pendulum)

    PE = mgL * (1 + cos(theta))
        -> 0 at bottom
        -> 2*mgL at top

    KE = 0.5 * I * omega^2
    """
    pe = mgL * (1.0 + math.cos(angle))
    ke = 0.5 * I * angular_vel**2
    return ke + pe


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1 / 1000))
    sim.set_camera_view([0.0, -5.0, 3.0], [0.0, 0.0, 2.0])

    # ----------------------------------------------------------------------
    # Scene
    # ----------------------------------------------------------------------

    sim_utils.DomeLightCfg(
        intensity=500.0,
        color=(0.9, 0.9, 0.9),
    ).func(
        "/World/DomeLight",
        sim_utils.DomeLightCfg(intensity=500.0),
    )

    sim_utils.GroundPlaneCfg().func(
        "/World/Ground",
        sim_utils.GroundPlaneCfg(),
    )

    # ----------------------------------------------------------------------
    # Robot
    # ----------------------------------------------------------------------

    robot_cfg = mdp.FUSION_CARTPOLE_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)

    sim.reset()
    robot.update(sim.get_physics_dt())

    # Joint indices
    pole_idx, _ = robot.find_joints("cart_to_pole")

    # ----------------------------------------------------------------------
    # Initial state
    # ----------------------------------------------------------------------

    root_state = robot.data.default_root_state.clone()
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()

    joint_vel[0, pole_idx[0]] = 0.067465
    joint_pos[0, pole_idx[0]] = 0.288913

    robot.write_root_state_to_sim(root_state)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    robot.update(sim.get_physics_dt())

    # ----------------------------------------------------------------------
    # Simulation settings
    # ----------------------------------------------------------------------

    LOG_EVERY = 10
    ENERGY_THRESHOLD = 1e-4  # Joules

    dt = sim.get_physics_dt()

    print(f"Energy threshold: {ENERGY_THRESHOLD:.6e} J")
    print(f"Maximum pendulum energy: {2 * mgL:.6e} J")

    # ----------------------------------------------------------------------
    # Logging
    # ----------------------------------------------------------------------
    log = []
    i = 0
    while True:
        sim.step()
        robot.update(dt)

        angle = robot.data.joint_pos[0, pole_idx[0]].item()
        vel = robot.data.joint_vel[0, pole_idx[0]].item()

        energy = compute_energy(angle, vel)

        if i % LOG_EVERY == 0:
            t = i * dt

            log.append(
                [
                    t,
                    math.cos(angle),
                    math.sin(angle),
                    angle,
                    vel,
                    energy,
                ]
            )

        if energy < ENERGY_THRESHOLD:
            print(f"Stopping at t={i * dt:.3f}s (energy={energy:.6e} J)")
            break
        i += 1
    # ----------------------------------------------------------------------
    # Save CSV
    # ----------------------------------------------------------------------
    with open("step_response_sim.csv", "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "dt",
                "cos",
                "sin",
                "angle",
                "angular_vel",
                "energy",
            ]
        )

        writer.writerows(log)

    print(f"Done. Saved {len(log)} samples to step_response_sim.csv")

    simulation_app.close()


if __name__ == "__main__":
    main()
