"""
Tesla-Style Sim-to-Real Rocket Environment Wrapper.
Applies Domain Randomization (wind, engine thrust degradation, sensor noise, initial tilt)
and binds the Anti-Reward-Hacking engine.
"""

import gymnasium as gym
import numpy as np
from typing import Optional, Dict, Any, Tuple
from config.settings import DomainRandomizationConfig, RewardConfig
from envs.reward_engine import AntiHackingRewardEngine


class RocketEnvWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        dr_cfg: DomainRandomizationConfig,
        reward_cfg: RewardConfig,
        curriculum_level: float = 1.0  # 0.0 (calm beginner) to 1.0 (full extreme conditions)
    ):
        super().__init__(env)
        self.dr_cfg = dr_cfg
        self.reward_cfg = reward_cfg
        self.curriculum_level = float(curriculum_level)
        self.reward_engine = AntiHackingRewardEngine(reward_cfg)
        
        # Episode-specific randomized parameters
        self.current_thrust_scale = 1.0
        self.current_wind_power = 0.0
        self.current_turbulence = 0.0

    def set_curriculum_level(self, level: float):
        """Adjust environmental difficulty (0.0 = calm/easy, 1.0 = full storm/dr)."""
        self.curriculum_level = np.clip(level, 0.0, 1.0)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)

        # Domain Randomization: Randomize physics parameters scaled by curriculum level
        if self.dr_cfg.enable_wind and hasattr(self.env.unwrapped, "wind_power"):
            w_min, w_max = self.dr_cfg.wind_power_range
            t_min, t_max = self.dr_cfg.turbulence_power_range
            
            # Scale wind with curriculum
            self.current_wind_power = np.random.uniform(w_min, w_max) * self.curriculum_level
            self.current_turbulence = np.random.uniform(t_min, t_max) * self.curriculum_level
            
            self.env.unwrapped.wind_power = self.current_wind_power
            self.env.unwrapped.turbulence_power = self.current_turbulence

        # Engine thrust dispersion (simulates actuator degradation / fuel pressure variance)
        th_min, th_max = self.dr_cfg.thrust_scale_range
        # Blend from nominal (1.0) to full dispersion based on curriculum
        low = 1.0 - (1.0 - th_min) * self.curriculum_level
        high = 1.0 + (th_max - 1.0) * self.curriculum_level
        self.current_thrust_scale = float(np.random.uniform(low, high))

        # Initial attitude and velocity perturbation
        if hasattr(self.env.unwrapped, "lander") and self.env.unwrapped.lander is not None:
            lander = self.env.unwrapped.lander
            
            # Random initial tilt (up to +/- 20 degrees)
            tilt_range = self.dr_cfg.initial_tilt_range
            delta_tilt = np.random.uniform(tilt_range[0], tilt_range[1]) * self.curriculum_level
            lander.angle += delta_tilt

            # Random initial horizontal velocity
            vel_x_range = self.dr_cfg.initial_vel_x_range
            delta_vx = np.random.uniform(vel_x_range[0], vel_x_range[1]) * self.curriculum_level
            lander.linearVelocity.x += delta_vx

        # Refresh ground truth state after body perturbation
        if hasattr(self.env.unwrapped, "lander"):
            # Update obs from internal Box2D state if perturbed
            x = (lander.position.x - 10.0) / 10.0
            y = (lander.position.y - 1.4) / 10.0
            vx = lander.linearVelocity.x * (0.05)
            vy = lander.linearVelocity.y * (0.05)
            angle = lander.angle
            angular_vel = lander.angularVelocity * 0.05
            left_leg = obs[6]
            right_leg = obs[7]
            obs = np.array([x, y, vx, vy, angle, angular_vel, left_leg, right_leg], dtype=np.float32)

        # Reset reward engine with true ground-truth state
        self.reward_engine.reset(obs)

        # Apply sensor observation noise
        noisy_obs = self._apply_sensor_noise(obs)
        
        info["true_obs"] = obs
        info["thrust_scale"] = self.current_thrust_scale
        info["wind_power"] = self.current_wind_power
        info["curriculum_level"] = self.curriculum_level

        return noisy_obs, info

    def _apply_sensor_noise(self, obs: np.ndarray) -> np.ndarray:
        """Injects IMU gyro and altimeter radar noise onto continuous observations."""
        if self.dr_cfg.obs_noise_std <= 0:
            return obs.copy()
        
        noise = np.random.normal(0, self.dr_cfg.obs_noise_std * self.curriculum_level, size=obs.shape).astype(np.float32)
        noisy = obs.copy() + noise
        
        # Keep leg ground-contact binary signals clean
        noisy[6] = obs[6]
        noisy[7] = obs[7]
        return noisy

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Apply physical thrust scaling (motor efficiency variation)
        applied_action = action.copy()
        applied_action[0] = float(np.clip(applied_action[0] * self.current_thrust_scale, -1.0, 1.0))
        applied_action[1] = float(np.clip(applied_action[1] * self.current_thrust_scale, -1.0, 1.0))

        # Advance physics step
        obs, default_reward, terminated, truncated, info = super().step(applied_action)

        # Compute anti-hacking shaped reward
        shaped_reward, reward_metrics = self.reward_engine.compute_reward(
            obs=obs,
            action=action,
            terminated=terminated,
            truncated=truncated,
            info=info
        )

        # Merge telemetry metrics
        info.update(reward_metrics)
        info["true_obs"] = obs
        info["applied_action"] = applied_action
        info["thrust_scale"] = self.current_thrust_scale

        # Apply sensor noise to state observation fed to neural network
        noisy_obs = self._apply_sensor_noise(obs)

        return noisy_obs, shaped_reward, terminated, truncated, info
