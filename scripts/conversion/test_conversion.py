import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

obs_size = 5

# --- rebuild pytorch model ---
model = nn.Sequential(
    nn.Linear(obs_size, 32),
    nn.ELU(),
    nn.Linear(32, 32),
    nn.ELU(),
    nn.Linear(32, 1),
)

checkpoint = torch.load(
    "./outputs/eridani/weights/skrl_pendulum_v0_047/skrl_pendulum_v0_047/skrl/pendulum_direct/2026-05-27_20-37-36_ppo_torch/checkpoints/best_agent.pt",
    map_location="cpu",
)
policy_state = checkpoint["policy"]
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

# --- same random input for both ---
dummy = np.random.randn(1, obs_size).astype(np.float32)

# pytorch forward
with torch.no_grad():
    pt_out = model(torch.tensor(dummy)).numpy()

# onnx forward
sess = ort.InferenceSession("./scripts/conversion/outputs/onnx/friction.onnx")
onnx_out = sess.run(None, {"obs": dummy})[0]

print(f"PyTorch : {pt_out}")
print(f"ONNX    : {onnx_out}")
print(f"Max diff: {np.abs(pt_out - onnx_out).max():.2e}")
