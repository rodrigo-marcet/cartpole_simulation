# System Identification & Pre-Deployment Tests

> **WARNING:** Both the conversion scripts and this document were created using AI. They do work, but just to keep you informed.

This folder holds two kinds of scripts: **system identification** tools that help you measure real physical parameters from the sim, and **neural network sanity checks** you run before flashing anything to the ESP32.

The idea is simple: before you trust the policy on real hardware, you make sure (a) your sim parameters actually match reality, and (b) the network produces the exact same output in Python as it does in C++.


## System ID Scripts

These all follow the same pattern: spawn the robot in Isaac Lab, apply a controlled input or initial condition, let physics run and dump a CSV with all of the needed info.

### `friction.py`: Cart friction

Gives the cart an initial velocity and lets it coast to a stop with zero applied force. Logs `[time, cart_pos, cart_vel]` to `step_response_sim.csv`.

**What to do with the CSV:** plot `cart_vel` vs `time`. Coulomb-dominated friction gives you a straight line to zero (constant deceleration), viscous friction gives you an exponential decay.

### `torque.py`: Breakaway / static friction

Ramps up applied torque linearly over time (`0.005 * elapsed Nm`) until the cart starts moving (`vel > 0.5 m/s`). Logs `[time, torque, vel]`.

**What to do with the CSV:** find the first timestep where velocity becomes non-negligible. The torque at that point is your static friction threshold. Convert to force: `F_static = torque / pulley_radius` (pulley radius = 0.01 m).

This is the number that should sit inside your domain randomization range for Coulomb friction. If reality breaks away at 0.017 Nm and sim breaks away at 0.013 Nm, that 30% gap needs to be covered by randomization.

### `pendulum.py`: Pole damping / inertia

Sets the pole to a known initial angle and angular velocity, then lets it swing freely for `STEPS` timesteps. Logs `[time, cos, sin, angle, angular_vel]`.

**What to do with the CSV:** fit an exponentially-decaying sinusoid to the angle trace. The decay envelope gives you the damping coefficient `b`; the oscillation frequency gives you `sqrt(g / L)`, which you can use to sanity-check your pole length `L` against the URDF.

Starting conditions (`joint_pos`, `joint_vel`) are hardcoded near the top of `main()`, change them to match whatever initial condition you used on the real pendulum.

---

## Neural Network Scripts (`neural_network/`)

### `test.py`: Full policy rollout in sim

Runs the trained policy inside Isaac Lab for `DURATION` seconds. Loads the checkpoint, applies observation normalization, steps the sim at 1 kHz, and runs inference every `DECIMATION` steps (so 100 Hz, matching the ESP32). Prints a live trace of cart position, pole angle, raw action, force, and torque each control step.

Use this to sanity-check a new checkpoint before touching the real system. If it can't hold the pole up in sim, don't bother deploying it.

Update `POLICY_PATH` at the top to point at the checkpoint you want to test.

### `single_runthrough.py`: Diff Python vs ESP32

No Isaac Lab needed, runs anywhere. Paste a real observation frame logged from the ESP32 (`cart_pos`, `cart_vel`, `cos`, `sin`, `pole_vel`) and the force the ESP32 actually output. The script runs the same observation through Python inference and prints the difference.

If the difference is larger than a few mN, something is wrong, usually a mismatch in:
- observation order (the network expects `[cart_pos, cart_vel, sin, cos, pole_vel]`, not cos/sin swapped)
- normalization (`running_mean` / `running_var` not matching the checkpoint)
- scale factor (`* 40.0` missing or wrong on the ESP32 side)

---

## Common Gotchas

**Wrong observation order.** The network was trained with `[cart_pos, cart_vel, sin, cos, pole_vel]`. The real AS5600 gives you the angle directly; compute `sin`/`cos` from it before passing to the network. `test.py` has the correct order hardcoded, match it exactly on the ESP32.

**Scale factor not applied.** The network output is in `[-1, 1]` (Tanh). Multiply by `scale` to get Newtons, then divide by `pulley_radius` to get Nm for ODrive. If you only send the raw Tanh output, forces are 40x too weak.

**Stale checkpoint path.** `POLICY_PATH` is hardcoded in every script. When you swap to a new checkpoint, update all of them, not just the one you're actively using.

**CSV overwrite.** All three sysid scripts write to the same filename (`step_response_sim.csv` in the working directory). Rename or move the file before running the next experiment, or they'll silently overwrite each other.
