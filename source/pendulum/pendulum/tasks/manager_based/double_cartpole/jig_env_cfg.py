import os

from isaaclab.utils import configclass

from ..single_cartpole.balancing_env_cfg import BalancingEnvCfg

_JIG_USD = os.path.join(os.path.dirname(__file__), "mdp/assets/single_cartpole/cartpole/cartpole.usda")


@configclass
class JigEnvCfg(BalancingEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.robot.spawn.usd_path = _JIG_USD
        self.terminations.pole_out_of_bounds = None
        self.terminations.cart_out_of_bounds.params["bounds"] = (-1.0, 1.0)
        self.episode_length_s = 120.0
        self.events.randomize_cart_mass = None
        self.events.randomize_shaft_mass = None
        self.events.randomize_pole_mass = None
        self.events.randomize_weight_mass = None
