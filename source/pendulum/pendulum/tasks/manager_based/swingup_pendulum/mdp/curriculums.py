from collections.abc import Sequence

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase


class NarrowCartBoundsCurriculum(ManagerTermBase):
    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._term_cfg = env.termination_manager.get_term_cfg(cfg.params["term_name"])

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        start_bound: float,
        end_bound: float,
        num_steps: int,
    ) -> float:
        progress = min(env.common_step_counter / num_steps, 1.0)
        current_bound = start_bound + (end_bound - start_bound) * progress
        self._term_cfg.params["bounds"] = (-current_bound, current_bound)
        env.termination_manager.set_term_cfg(term_name, self._term_cfg)
        return current_bound
