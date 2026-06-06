import torch
import torch.nn as nn

POLICY_PATH = r"D:/omniverse/pendulum/logs/skrl/pendulum_direct/gaussian_pole/checkpoints/best_agent.pt"

# --- load model ---
model = nn.Sequential(
    nn.Linear(5, 32),
    nn.ELU(),
    nn.Linear(32, 32),
    nn.ELU(),
    nn.Linear(32, 1),
    nn.Tanh(),
)

checkpoint = torch.load(POLICY_PATH, map_location="cpu")
policy_state = checkpoint["policy"]

model.load_state_dict(
    {
        "0.weight": policy_state["net_container.0.weight"],
        "0.bias": policy_state["net_container.0.bias"],
        "2.weight": policy_state["net_container.2.weight"],
        "2.bias": policy_state["net_container.2.bias"],
        "4.weight": policy_state["policy_layer.weight"],
        "4.bias": policy_state["policy_layer.bias"],
    }
)
model.eval()

scaler_state = checkpoint["state_preprocessor"]
running_mean = scaler_state["running_mean"].float()
running_var = scaler_state["running_variance"].float()

# --- input from real system ---
cart_pos = -0.000019
cart_vel = -0.000479
cos_angle = 0.99983
sin_angle = -0.01856
pole_vel = -0.39072
force_real = 18.642195
torque_real = 0.186422


obs_raw = torch.tensor([[cart_pos, cart_vel, sin_angle, cos_angle, pole_vel]], dtype=torch.float32)
obs_scaled = (obs_raw - running_mean) / (running_var.sqrt() + 1e-8)

with torch.no_grad():
    action = model(obs_scaled)

force_unscaled = action[0, 0].item()
force = force_unscaled * 40.0

print(f"obs raw:      {obs_raw.numpy()}")
print(f"obs scaled:   {obs_scaled.numpy()}")
print(f"network out:  {force_unscaled:.6f}")
print(f"force:        {force:.6f} N")
print(f"torque:       {force * 0.01:.6f} Nm")
print()
print(f"real system got: force = {force_real:.6f} N")
print(f"difference:      {abs((-force) - force_real):.6f} N")


print("running_mean:", running_mean.tolist())
print("running_var: ", running_var.tolist())
