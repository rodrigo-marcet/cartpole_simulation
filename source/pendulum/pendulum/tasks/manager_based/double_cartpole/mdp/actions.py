from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointEffortActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointEffortAction
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = ["LaggedJointEffortAction", "LaggedJointEffortActionCfg"]


class LaggedJointEffortAction(JointEffortAction):
    """JointEffortAction with a one-tap first-order lag on the delivered effort."""

    cfg: LaggedJointEffortActionCfg

    def __init__(self, cfg: LaggedJointEffortActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._prev_cmd = torch.zeros_like(self.processed_actions)

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        blended = self.cfg.alpha * self._processed_actions + (1.0 - self.cfg.alpha) * self._prev_cmd
        self._prev_cmd[:] = self._processed_actions  # store the UNBLENDED command
        self._processed_actions[:] = blended

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._prev_cmd.zero_()
        else:
            self._prev_cmd[env_ids] = 0.0


@configclass
class LaggedJointEffortActionCfg(JointEffortActionCfg):
    """See LaggedJointEffortAction. alpha = 1.0 disables the lag."""

    class_type: type = LaggedJointEffortAction

    alpha: float = 0.414
    """Weight on the NEW command. Mean delay = (1 - alpha) * dt; 0.414 -> 6.0 ms at 97.55 Hz.

    NOTE the convention: this is the weight on the new command, not on the history.
    """
