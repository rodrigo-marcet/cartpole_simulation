"""Convert a TFLite model into a C header for the ESP32.

Emits the model bytes plus the input-normalisation constants the firmware expects. tflite.cpp
computes (obs - MEAN) / (sqrtf(VAR) + 1e-8f), so a fixed per-channel scale k is encoded as
MEAN = 0, VAR = (1/k)^2 -- no firmware change needed.

Two sources for the normalisation, picked automatically:
  --obs-scales   the fixed scales (OBS_SCALE in the env cfg). Floats OR fractions, so
                 "2.5,0.4,1,1,1/15,1,1,1/15" stays exact and 1/15 yields VAR = 225 rather than
                 224.78 from a rounded 0.0667. Use this for anything trained with
                 state_preprocessor: null.
  (omitted)      falls back to the checkpoint's skrl RunningStandardScaler -- the original
                 behaviour, still correct for single-cartpole policies.

The width is cross-checked against the checkpoint's first layer (and the .tflite input tensor when
TensorFlow is importable), so a wrong-length list or a mismatched .pt/.tflite pair fails loudly
instead of producing a plausible but wrong header.
"""

import argparse
import os
import sys
from fractions import Fraction

import torch

parser = argparse.ArgumentParser(description="Convert a TFLite model into a C header file")
parser.add_argument("-i", "--input", required=True, help="Path to input .tflite file")
parser.add_argument("-o", "--output", required=True, help="Path to output .h file")
parser.add_argument("-c", "--checkpoint", required=True, help="Path to .pt checkpoint file")
parser.add_argument(
    "--obs-scales",
    default=None,
    help="Fixed per-channel observation scales, comma separated. Accepts floats and fractions, "
    'e.g. "2.5,0.4,1,1,1/15,1,1,1/15". Omit to read the skrl RunningStandardScaler from the '
    "checkpoint instead.",
)
args = parser.parse_args()


def parse_scale(tok: str) -> float:
    """One scale: plain float or an exact 'a/b' fraction."""
    tok = tok.strip()
    if not tok:
        raise ValueError("empty scale entry")
    return float(Fraction(tok)) if "/" in tok else float(tok)


def first_layer_in_features(policy_state: dict) -> int | None:
    """Observation width from the checkpoint's first hidden layer (skrl: net_container.<i>)."""
    keys = [k for k in policy_state if k.startswith("net_container.") and k.endswith(".weight")]
    if not keys:
        return None
    first = min(keys, key=lambda k: int(k.split(".")[1]))
    return int(policy_state[first].shape[1])


def tflite_in_features(path: str) -> int | None:
    """Input width straight from the .tflite, when TensorFlow is available. Best effort."""
    try:
        import tensorflow as tf
    except ImportError:
        return None
    try:
        interp = tf.lite.Interpreter(model_path=path)
        interp.allocate_tensors()
        return int(interp.get_input_details()[0]["shape"][-1])
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] could not read the .tflite input shape: {e}")
        return None


def float_array_to_c(name, values):
    inner = ", ".join(f"{v:.8f}f" for v in values)
    return f"constexpr float {name}[] = {{{inner}}};\n"


checkpoint = torch.load(args.checkpoint, map_location="cpu")
n_ckpt = first_layer_in_features(checkpoint.get("policy", {}))

# ---- normalisation: fixed scales, or the checkpoint's running scaler --------------------------
if args.obs_scales:
    raw = [t for t in args.obs_scales.split(",")]
    try:
        scales = [parse_scale(t) for t in raw]
    except (ValueError, ZeroDivisionError) as e:
        sys.exit(f"ERROR: could not parse --obs-scales {args.obs_scales!r}: {e}")
    bad = [t for t, s in zip(raw, scales) if s <= 0.0]
    if bad:
        sys.exit(f"ERROR: scales must be > 0 (the header stores 1/scale^2); got {bad}")
    running_mean = [0.0] * len(scales)
    running_var = [1.0 / (s * s) for s in scales]
    source = "fixed --obs-scales"
    shown = ", ".join(t.strip() for t in raw)
else:
    if "state_preprocessor" not in checkpoint:
        sys.exit(
            "ERROR: this checkpoint has no 'state_preprocessor' (trained with "
            "state_preprocessor: null), so there is nothing to read.\n"
            "       Pass the fixed scales explicitly, e.g.\n"
            '         --obs-scales "2.5,0.4,1,1,1/15,1,1,1/15"'
        )
    scaler_state = checkpoint["state_preprocessor"]
    running_mean = scaler_state["running_mean"].float().flatten().tolist()
    running_var = scaler_state["running_variance"].float().flatten().tolist()
    scales = None
    source = "skrl RunningStandardScaler (from the checkpoint)"
    shown = "n/a"

# ---- width checks ----------------------------------------------------------------------------
n_norm = len(running_mean)
if n_ckpt is not None and n_norm != n_ckpt:
    sys.exit(
        f"ERROR: normalisation has {n_norm} channels but the checkpoint's first layer takes "
        f"{n_ckpt} inputs.\n       Check the order/length of --obs-scales against OBS_SCALE."
    )
n_tfl = tflite_in_features(args.input)
if n_tfl is not None and n_tfl != n_norm:
    sys.exit(
        f"ERROR: the .tflite takes {n_tfl} inputs but the normalisation has {n_norm} channels.\n"
        f"       Are the .pt and the .tflite from the same policy?"
    )

with open(args.input, "rb") as f:
    data = f.read()

with open(args.output, "w") as f:
    f.write("#pragma once\n\n")
    f.write("// Generated by scripts/conversion/tflite_to_header.py -- do not edit by hand.\n")
    f.write(f"//   normalisation source : {source}\n")
    if scales is not None:
        f.write(f"//   scales (as given)    : {shown}\n")
        f.write(f"//   scales (evaluated)   : {', '.join(f'{s:.8g}' for s in scales)}\n")
    f.write(f"//   checkpoint           : {os.path.abspath(args.checkpoint)}\n")
    f.write(f"//   tflite               : {os.path.abspath(args.input)} ({len(data)} bytes)\n")
    f.write(f"//   observation width    : {n_norm}")
    f.write(f" (checkpoint {n_ckpt}, tflite {n_tfl if n_tfl is not None else 'unchecked'})\n")
    f.write("// tflite.cpp computes (obs - MEAN)/(sqrtf(VAR)+1e-8f), so VAR = (1/scale)^2\n")
    f.write("// reproduces obs*scale. NOTE it also clamps the result to +/-5, which the training\n")
    f.write("// env does NOT -- unreachable for live states under these scales, but not identical.\n\n")
    f.write(float_array_to_c("MODEL_INPUT_MEAN", running_mean))
    f.write(float_array_to_c("MODEL_INPUT_VAR", running_var))
    f.write(f"constexpr int MODEL_INPUT_SIZE = {n_norm};\n\n")

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
print(f"  normalisation : {source}")
if scales is not None:
    print(f"  scales        : {shown}")
    print(f"  -> MODEL_INPUT_VAR = {', '.join(f'{v:.6g}' for v in running_var)}")
print(f"  obs width     : {n_norm} (checkpoint {n_ckpt}, tflite {n_tfl if n_tfl is not None else 'unchecked'})")
print(f"  model size    : {len(data)} bytes")
