"""
test_conversion.py — compare PyTorch (.pt) vs TFLite outputs end-to-end.

Usage:
    python test_conversion.py -i path/to/best_agent.pt -c path/to/model.tflite
"""

import argparse
import sys

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn

OBS_SIZE = 5

# ── architecture ──────────────────────────────────────────────────────────────


def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(OBS_SIZE, 32),
        nn.ELU(),
        nn.Linear(32, 32),
        nn.ELU(),
        nn.Linear(32, 1),
    )


# ── loaders ───────────────────────────────────────────────────────────────────


def load_pytorch(pt_path: str) -> nn.Module:
    checkpoint = torch.load(pt_path, map_location="cpu")
    policy_state = checkpoint["policy"]
    stripped = {
        "0.weight": policy_state["net_container.0.weight"],
        "0.bias": policy_state["net_container.0.bias"],
        "2.weight": policy_state["net_container.2.weight"],
        "2.bias": policy_state["net_container.2.bias"],
        "4.weight": policy_state["policy_layer.weight"],
        "4.bias": policy_state["policy_layer.bias"],
    }
    model = build_model()
    model.load_state_dict(stripped)
    model.eval()
    return model


def run_pytorch(model: nn.Module, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.tensor(x)).numpy()


def run_tflite(tflite_path: str, x: np.ndarray) -> np.ndarray:
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp_detail = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]

    # cast to whatever dtype the tflite model expects (float32 or int8)
    x_cast = x.astype(inp_detail["dtype"])
    interp.set_tensor(inp_detail["index"], x_cast)
    interp.invoke()
    return interp.get_tensor(out_detail["index"])


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Compare PyTorch vs TFLite model outputs.")
    parser.add_argument("-i", "--input", required=True, help="Path to .pt checkpoint")
    parser.add_argument("-c", "--conversion", required=True, help="Path to .tflite model")
    parser.add_argument(
        "-n", "--num-samples", type=int, default=1000, help="Number of random test inputs (default: 1000)"
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"\nLoading PyTorch checkpoint : {args.input}")
    try:
        pt_model = load_pytorch(args.input)
    except Exception as e:
        print(f"  ERROR loading .pt: {e}", file=sys.stderr)
        sys.exit(1)
    print("  OK")

    print(f"Loading TFLite model       : {args.conversion}")
    try:
        # dry-run to catch path / format errors early
        interp_check = tf.lite.Interpreter(model_path=args.conversion)
        interp_check.allocate_tensors()
    except Exception as e:
        print(f"  ERROR loading .tflite: {e}", file=sys.stderr)
        sys.exit(1)
    print("  OK\n")

    # ── single hand-picked input ──────────────────────────────────────────────
    single = np.array([[0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)  # pole upright
    pt_single = run_pytorch(pt_model, single)
    tfl_single = run_tflite(args.conversion, single)
    print("── Sanity check (pole upright: sin=0, cos=1, dθ=0, x=0, ẋ=0) ──")
    print(f"  PyTorch : {pt_single.flat[0]:+.6f}")
    print(f"  TFLite  : {tfl_single.flat[0]:+.6f}")
    print(f"  Diff    : {abs(pt_single.flat[0] - tfl_single.flat[0]):.2e}\n")

    # ── bulk random test ──────────────────────────────────────────────────────
    print(f"── Random test ({args.num_samples} samples, seed={args.seed}) ──")
    inputs = rng.standard_normal((args.num_samples, 1, OBS_SIZE)).astype(np.float32)

    pt_outs = np.array([run_pytorch(pt_model, x).flat[0] for x in inputs])
    tfl_outs = np.array([run_tflite(args.conversion, x).flat[0] for x in inputs])

    diffs = np.abs(pt_outs - tfl_outs)
    print(f"  Max diff    : {diffs.max():.4e}")
    print(f"  Mean diff   : {diffs.mean():.4e}")
    print(f"  Median diff : {np.median(diffs):.4e}")
    print(f"  Std diff    : {diffs.std():.4e}")

    thresholds = [1e-3, 1e-4, 1e-5]
    for t in thresholds:
        pct = (diffs < t).mean() * 100
        print(f"  Samples < {t:.0e} : {pct:.1f}%")

    worst_idx = diffs.argmax()
    print(f"\n  Worst-case input : {inputs[worst_idx].squeeze()}")
    print(f"  PyTorch output   : {pt_outs[worst_idx]:+.6f}")
    print(f"  TFLite  output   : {tfl_outs[worst_idx]:+.6f}")

    # ── pass/fail ─────────────────────────────────────────────────────────────
    PASS_THRESHOLD = 1e-3
    if diffs.max() < PASS_THRESHOLD:
        print(f"\n  ✓ PASS — max diff {diffs.max():.2e} < {PASS_THRESHOLD:.0e}")
    else:
        print(f"\n  ✗ FAIL — max diff {diffs.max():.2e} ≥ {PASS_THRESHOLD:.0e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
