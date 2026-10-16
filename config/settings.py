"""
Project Configuration and Hyperparameters.
Contains environment settings, domain randomization bounds, anti-hacking reward parameters,
and PPO model architecture configurations.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class EnvironmentConfig:
    env_id: str = "LunarLanderContinuous-v3"
    render_mode: str = "rgb_array"
    max_episode_steps: int = 1000
    seed: int = 42


@dataclass
class DomainRandomizationConfig:
    """
    Tesla-style Sim-to-Real Domain Randomization parameters.
    Randomizes physics and sensory inputs across episodes so policy generalizes.
    """
    # Wind turbulence
    enable_wind: bool = True
    wind_power_range: Tuple[float, float] = (-8.0, 8.0)
    turbulence_power_range: Tuple[float, float] = (0.0, 1.5)
    
    # Engine thrust variability (simulates motor wear, voltage drop, fuel pressure fluctuation)
    thrust_scale_range: Tuple[float, float] = (0.85, 1.15)
    
    # Sensor noise (radar altimeter & IMU gyro sensor noise)
    obs_noise_std: float = 0.015
    
    # Initial state perturbations
    initial_tilt_range: Tuple[float, float] = (-0.35, 0.35)  # radians (~ +/- 20 degrees)
    initial_vel_x_range: Tuple[float, float] = (-0.5, 0.5)
    initial_vel_y_range: Tuple[float, float] = (-0.2, 0.2)


@dataclass
class RewardConfig:
    """
    Anti-Reward-Hacking shaped reward parameters.
    Prevents hovering exploit, kamikaze exploit, and spin-to-win exploits.
    """
    # Potential-based distance and velocity weights
    w_distance: float = 100.0
    w_velocity: float = 100.0
    w_tilt: float = 100.0
    w_angular_vel: float = 20.0
    
    # Fuel expenditure penalty (destroys the "hover forever" loop)
    fuel_penalty_main: float = 0.03
    fuel_penalty_side: float = 0.01
    
    # Per-step time pressure penalty
    step_penalty: float = 0.05
    
    # Safe leg contact bonus
    leg_contact_bonus: float = 15.0
    
    # Terminal outcomes
    safe_landing_bonus: float = 150.0
    crash_penalty: float = -150.0
    
    # Strict safety invariants for touchdown qualification
    max_safe_vx: float = 0.35
    max_safe_vy: float = 0.35
    max_safe_tilt: float = 0.18  # ~ 10 degrees


@dataclass
class TrainingConfig:
    total_timesteps: int = 600_000
    num_envs: int = 8  # Parallel SubprocVecEnv workers for CPU
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    log_dir: str = "logs"
    model_dir: str = "models"
    eval_freq: int = 10_000
    eval_episodes: int = 15


env_cfg = EnvironmentConfig()
dr_cfg = DomainRandomizationConfig()
reward_cfg = RewardConfig()
train_cfg = TrainingConfig()
