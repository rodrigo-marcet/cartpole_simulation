import torch

from isaaclab.envs.mdp.actions import JointEffortAction, JointEffortActionCfg
from isaaclab.utils import configclass


class SmoothedJointEffortAction(JointEffortAction):
    cfg: "SmoothedJointEffortActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._smoothed_action = None
        self._alpha = cfg.alpha

    def process_actions(self, actions: torch.Tensor):
        if self._smoothed_action is None:
            self._smoothed_action = torch.zeros_like(actions)

        self._smoothed_action = self._alpha * self._smoothed_action + (1.0 - self._alpha) * actions
        super().process_actions(self._smoothed_action)

    def reset(self, env_ids=None):
        if self._smoothed_action is not None:
            if env_ids is None:
                self._smoothed_action.zero_()
            else:
                self._smoothed_action[env_ids] = 0.0
        super().reset(env_ids)


@configclass
class SmoothedJointEffortActionCfg(JointEffortActionCfg):
    class_type: type = SmoothedJointEffortAction
    alpha: float = 0.6
