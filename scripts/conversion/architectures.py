"""Policy network architectures for the sim -> embedded conversion pipeline.

One builder per rig. `single` and `double` are structurally identical today (they differ
only in observation size), but they are kept as separate, fully-spelled-out definitions so
either can change independently without touching the conversion scripts.
"""

import torch.nn as nn


def build_model(net_type: str) -> nn.Module:
    """Untrained policy network for the given rig type."""
    if net_type == "single":
        return nn.Sequential(
            nn.Linear(5, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )
    if net_type == "double":
        return nn.Sequential(
            nn.Linear(8, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )
    raise ValueError(f"unknown --type {net_type!r} (expected 'single' or 'double')")


def remap_state_dict(model: nn.Module, policy_state: dict) -> dict:
    """Map an skrl policy checkpoint onto `model`'s Linear layers. skrl stores the hidden stack
    under net_container.<i> and the output under policy_layer; we mirror that structure, so the
    remap follows the architecture (any width / layer count) instead of hardcoding indices."""
    linears = [i for i, m in enumerate(model) if isinstance(m, nn.Linear)]
    stripped = {}
    for i in linears[:-1]:
        stripped[f"{i}.weight"] = policy_state[f"net_container.{i}.weight"]
        stripped[f"{i}.bias"] = policy_state[f"net_container.{i}.bias"]
    out = linears[-1]
    stripped[f"{out}.weight"] = policy_state["policy_layer.weight"]
    stripped[f"{out}.bias"] = policy_state["policy_layer.bias"]
    return stripped
