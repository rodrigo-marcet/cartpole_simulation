# Self-Balancing Double and Single Pendulum Isaac Lab Environment

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

<p align="center">
  <img src="docs/sim.gif" width="600" alt="[Project] simulation screenshot">
</p>

## Overview

Isaac Lab extension and training environment for self-balancing single and double inverted pendulums (cartpoles). It defines the tasks, MDP (rewards, observations, terminations, events), and robot assets used to train swing-up and balancing policies with reinforcement learning, plus the scripts to calibrate the sim against the real rig and export trained policies for embedded deployment. It is the training-side counterpart to the [firmware repo](https://github.com/rodrigo-marcet/cartpole), which runs the exported policies on the physical hardware.

📺 Watch here: https://youtu.be/dMohMW29gSM

## How It Works

Each pendulum variant (single and double cartpole) is a manager-based Isaac Lab task, registered as a Gym environment and trained with `skrl` (PPO). Reset distributions, reward shaping, and domain randomization are tuned to close the sim-to-real gap. Once a policy converges, the conversion pipeline exports it from PyTorch to ONNX to TFLite to a C header, ready to be dropped into the firmware.

## Related

* Firmware / embedded deployment: [pendulum firmware repo](FIRMWARE_REPO_URL)

## Getting Started

* Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

```bash
git clone https://github.com/rodrigo-marcet/cartpole_simulation.git
cd cartpole_simulation

# use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab isn't in a venv/conda env
python -m pip install -e source/pendulum

# train the double cartpole swing-up task
python scripts/skrl/train.py --task=Swingup-DoubleCartpole-v0
```

See [`source/pendulum`](source/pendulum) for task and MDP details, and [`scripts/conversion`](scripts/conversion) for exporting a trained policy to the firmware.

## License

Distributed under the [MIT](LICENSE) License. See [`LICENSE`](LICENSE) for details.
