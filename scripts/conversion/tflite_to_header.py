import argparse

import torch

parser = argparse.ArgumentParser(description="Convert a TFLite model into a C header file")
parser.add_argument("-i", "--input", required=True, help="Path to input .tflite file")
parser.add_argument("-o", "--output", required=True, help="Path to output .h file")
parser.add_argument("-c", "--checkpoint", required=True, help="Path to .pt checkpoint file")
args = parser.parse_args()

# Load scaler from checkpoint
checkpoint = torch.load(args.checkpoint, map_location="cpu")
scaler_state = checkpoint["state_preprocessor"]
running_mean = scaler_state["running_mean"].float().tolist()
running_var = scaler_state["running_variance"].float().tolist()


def float_array_to_c(name, values):
    inner = ", ".join(f"{v:.8f}f" for v in values)
    return f"constexpr float {name}[] = {{{inner}}};\n"


# Load TFLite model
with open(args.input, "rb") as f:
    data = f.read()

with open(args.output, "w") as f:
    f.write("#pragma once\n\n")

    # Scaler arrays
    f.write("// Input normalisation (running mean / variance from SKRL preprocessor)\n")
    f.write(float_array_to_c("MODEL_INPUT_MEAN", running_mean))
    f.write(float_array_to_c("MODEL_INPUT_VAR", running_var))
    f.write(f"constexpr int MODEL_INPUT_SIZE = {len(running_mean)};\n\n")

    # Model weights
    f.write("inline constexpr unsigned char policy_model[] = {\n    ")
    for i, byte in enumerate(data):
        f.write(f"0x{byte:02x}")
        if i != len(data) - 1:
            f.write(", ")
        if (i + 1) % 12 == 0 and i != len(data) - 1:
            f.write("\n    ")
    f.write("\n};\n")
    f.write(f"inline constexpr unsigned int policy_model_len = {len(data)};\n")

print(f"Header generated: {args.output}")
print(f"Model size: {len(data)} bytes")
print(f"Scaler input size: {len(running_mean)}")
