import torch
import torch.nn as nn

# Point this at your trained DOUBLE-cartpole checkpoint (skrl experiment dir = "double_pendulum_direct").
POLICY_PATH = r"D:\omniverse\pendulum\logs\skrl\double_pendulum_direct\swingup_noise_v1\checkpoints\best_agent.pt"

# Must match the double deployment: action->force scale and pulley radius (force -> torque).
# The double env authors JointEffortActionCfg scale=40.0 (the single's deploy used 30.0) -- set to your rig.
ACTION_SCALE = 40.0
PULLEY_RADIUS = 0.01

# --- load model (double: 8 obs -> 32 -> 32 -> 1) ---
model = nn.Sequential(
    nn.Linear(8, 128),
    nn.ELU(),
    nn.Linear(128, 128),
    nn.ELU(),
    nn.Linear(128, 1),
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

# --- input from real system (paste ONE captured sample) ---
# obs order = double ObservationsCfg:
# cart_pos, cart_vel,
# ipole_sin, ipole_cos, ipole_vel,
# opole_sin, opole_cos, opole_vel

cart_pos = 0.001679
cart_vel = 0.001917
ipole_sin = 0.697480
ipole_cos = -0.716604
ipole_vel = 0.300041
opole_sin = -0.934639
opole_cos = 0.355598
opole_vel = 2.400126
force_real = -23.839039  # force the real system reported for this obs (N)
torque_real = -0.238390  # torque the real system reported for this obs (Nm)

obs_raw = torch.tensor(
    [[cart_pos, cart_vel, ipole_sin, ipole_cos, ipole_vel, opole_sin, opole_cos, opole_vel]],
    dtype=torch.float32,
)
obs_scaled = ((obs_raw - running_mean) / (running_var.sqrt() + 1e-8)).clamp(-5.0, 5.0)

print("CLIP_THRESHOLD:", 5.0)  # the value YOU hardcoded in .clip(-5, 5)
print("EPSILON       :", 1e-8)
print("MEAN:", running_mean.flatten().tolist())
print("VAR :", running_var.flatten().tolist())

pre_clip = (obs_raw - running_mean) / (running_var.sqrt() + 1e-8)
print("SR SCALED (pre-clip) :", pre_clip.flatten().tolist())
print("SR SCALED (post-clip):", obs_scaled.flatten().tolist())  # what your net actually receives

print("policy keys:", list(policy_state.keys()))
print(model)

with torch.no_grad():
    action = model(obs_scaled)

force_unscaled = action[0, 0].item()
force = force_unscaled * ACTION_SCALE

print(f"obs raw:      {obs_raw.numpy()}")
print(f"obs scaled:   {obs_scaled.numpy()}")
print(f"network out:  {force_unscaled:.6f}")
print(f"force:        {force:.6f} N")
print(f"torque:       {force * PULLEY_RADIUS:.6f} Nm")
print()
print(f"real system got: force = {force_real:.6f} N")
print(f"difference:      {abs((-force) - force_real):.6f} N")


print("running_mean:", running_mean.tolist())
print("running_var: ", running_var.tolist())
