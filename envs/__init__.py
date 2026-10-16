"""
Environment factory and registration helpers.
"""

import gymnasium as gym
from typing import Optional, Callable
from config.settings import EnvironmentConfig, DomainRandomizationConfig, RewardConfig, env_cfg, dr_cfg, reward_cfg
from envs.rocket_wrapper import RocketEnvWrapper


def make_rocket_env(
    env_config: Optional[EnvironmentConfig] = None,
    dr_config: Optional[DomainRandomizationConfig] = None,
    reward_config: Optional[RewardConfig] = None,
    curriculum_level: float = 1.0,
    render_mode: Optional[str] = None
) -> Callable[[], gym.Env]:
    """
    Factory creating a callable returning a wrapped rocket environment.
    Compatible with Stable-Baselines3 SubprocVecEnv and DummyVecEnv.
    """
    cfg_e = env_config or env_cfg
    cfg_dr = dr_config or dr_cfg
    cfg_r = reward_config or reward_cfg

    def _init() -> gym.Env:
        base_env = gym.make(
            cfg_e.env_id,
            continuous=True,
            gravity=-10.0,
            enable_wind=cfg_dr.enable_wind,
            wind_power=10.0,
            turbulence_power=1.0,
            render_mode=render_mode
        )
        return RocketEnvWrapper(
            env=base_env,
            dr_cfg=cfg_dr,
            reward_cfg=cfg_r,
            curriculum_level=curriculum_level
        )

    return _init
