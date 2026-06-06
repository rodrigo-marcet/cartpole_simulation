import torch
import torch.nn as nn

obs_size = 5

model = nn.Sequential(
    nn.Linear(obs_size, 256),
    nn.ELU(),
    nn.Linear(256, 128),
    nn.ELU(),
    nn.Linear(128, 1),
)

checkpoint = torch.load(
    "D:/omniverse/pendulum/logs/skrl/pendulum_direct/2026-05-09_15-00-52_ppo_torch/checkpoints/best_agent.pt"
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

dummy = torch.tensor([[0.1, 0.05, 0.0, 0.0, 0.1]], dtype=torch.float32)

with torch.no_grad():
    out = model(dummy)

print(f"PyTorch output: {out.item():.6f}")
