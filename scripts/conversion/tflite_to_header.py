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
    return f"float {name}[] = {{{inner}}};\n"


# Load TFLite model
with open(args.input, "rb") as f:
    data = f.read()

with open(args.output, "w") as f:
    # Guard
    f.write("#ifndef POLICY_MODEL_H\n#define POLICY_MODEL_H\n\n")

    # Scaler arrays
    f.write("// Input normalisation (running mean / variance from SKRL preprocessor)\n")
    f.write(float_array_to_c("input_mean", running_mean))
    f.write(float_array_to_c("input_var", running_var))
    f.write(f"const int input_size = {len(running_mean)};\n\n")

    # Model weights
    f.write("unsigned char policy_model[] = {\n    ")
    for i, byte in enumerate(data):
        f.write(f"0x{byte:02x}")
        if i != len(data) - 1:
            f.write(", ")
        if (i + 1) % 12 == 0 and i != len(data) - 1:
            f.write("\n    ")
    f.write("\n};\n")
    f.write(f"unsigned int policy_model_len = {len(data)};\n\n")

    f.write("#endif // POLICY_MODEL_H\n")

print(f"Header generated: {args.output}")
print(f"Model size: {len(data)} bytes")
print(f"Scaler input size: {len(running_mean)}")
