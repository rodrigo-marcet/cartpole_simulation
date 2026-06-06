import argparse

import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Example:
#
# python .\scripts\conversion\pt_to_onnx.py ^
#   -i C:/path/to/best_agent.pt ^
#   -o C:/path/to/output/policy.onnx
#
# Quotes are only needed if the path contains spaces.
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Convert PyTorch policy to ONNX")

parser.add_argument(
    "-i",
    "--input",
    required=True,
    help="Path to input .pt checkpoint",
)

parser.add_argument(
    "-o",
    "--output",
    required=True,
    help="Path to output .onnx file",
)

args = parser.parse_args()

checkpoint_path = args.input
output_path = args.output

obs_size = 5
action_size = 1

# Reconstruct network
model = nn.Sequential(
    nn.Linear(obs_size, 32), nn.ELU(), nn.Linear(32, 32), nn.ELU(), nn.Linear(32, action_size), nn.Tanh()
)

# Load checkpoint
checkpoint = torch.load(checkpoint_path)

print(checkpoint.keys())
print(checkpoint["policy"].keys())

policy_state = checkpoint["policy"]

# Remap weights
stripped = {
    "0.weight": policy_state["net_container.0.weight"],
    "0.bias": policy_state["net_container.0.bias"],
    "2.weight": policy_state["net_container.2.weight"],
    "2.bias": policy_state["net_container.2.bias"],
    "4.weight": policy_state["policy_layer.weight"],
    "4.bias": policy_state["policy_layer.bias"],
}

model.load_state_dict(stripped)
model.eval()

dummy_input = torch.randn(1, obs_size)

# Export ONNX
torch.onnx.export(
    model,
    dummy_input,
    output_path,
    input_names=["obs"],
    output_names=["action"],
    opset_version=11,
)

print(f"ONNX model exported to: {output_path}")
