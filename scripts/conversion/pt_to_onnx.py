import argparse

import torch
from architectures import build_model, remap_state_dict

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

parser.add_argument(
    "--type",
    dest="net_type",
    required=True,
    choices=["single", "double"],
    help="Network architecture: single (5 obs) or double (8 obs)",
)

args = parser.parse_args()

checkpoint_path = args.input
output_path = args.output

# Reconstruct network for the selected architecture
model = build_model(args.net_type)

# Load checkpoint
checkpoint = torch.load(checkpoint_path)

print(checkpoint.keys())
print(checkpoint["policy"].keys())

policy_state = checkpoint["policy"]

# Remap skrl weights onto the Sequential and load
model.load_state_dict(remap_state_dict(model, policy_state))
model.eval()

dummy_input = torch.randn(1, model[0].in_features)

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
