"""
Evaluation Benchmark Suite.
Tests trained rocket landing policies against diverse Sim-to-Real stress conditions:
calm weather, crosswinds, engine degradation, and high-tilt emergency recoveries.
"""

import os
import sys

# Ensure Python's base install directory is registered for Windows DLL resolution (finds python312.dll)
if hasattr(os, "add_dll_directory") and hasattr(sys, "base_exec_prefix") and os.path.isdir(sys.base_exec_prefix):
    try:
        os.add_dll_directory(sys.base_exec_prefix)
    except Exception:
        pass

import argparse
import numpy as np
from stable_baselines3 import PPO

from config.settings import env_cfg, dr_cfg, reward_cfg, DomainRandomizationConfig
from envs.rocket_wrapper import RocketEnvWrapper
import gymnasium as gym


def run_benchmark_scenario(
    model: PPO,
    scenario_name: str,
    dr_config: DomainRandomizationConfig,
    n_episodes: int = 20,
    curriculum_level: float = 1.0
) -> dict:
    base_env = gym.make(
        env_cfg.env_id,
        continuous=True,
        gravity=-10.0,
        enable_wind=dr_config.enable_wind,
        wind_power=10.0,
        turbulence_power=1.0,
        render_mode=None
    )
    env = RocketEnvWrapper(
        env=base_env,
        dr_cfg=dr_config,
        reward_cfg=reward_cfg,
        curriculum_level=curriculum_level
    )

    successes = 0
    crashes = 0
    fuel_burns = []
    final_speeds = []
    final_tilts = []
    rewards = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=1000 + ep)
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        rewards.append(ep_reward)
        fuel_burns.append(info.get("total_fuel_burned", 0.0))
        final_speeds.append(info.get("speed", 0.0))
        final_tilts.append(info.get("tilt_deg", 0.0))

        if info.get("is_safe_landing", 0) > 0:
            successes += 1
        elif info.get("is_crash", 0) > 0:
            crashes += 1

    env.close()

    return {
        "scenario": scenario_name,
        "episodes": n_episodes,
        "success_rate": (successes / n_episodes) * 100.0,
        "crash_rate": (crashes / n_episodes) * 100.0,
        "mean_reward": float(np.mean(rewards)),
        "mean_fuel": float(np.mean(fuel_burns)),
        "mean_touchdown_speed": float(np.mean(final_speeds)),
        "mean_touchdown_tilt": float(np.mean(final_tilts)),
    }


def evaluate(model_path: str = "models/best_model.zip", n_episodes_per_scenario: int = 20):
    if not os.path.exists(model_path):
        fallback = "models/rocket_ppo_final.zip"
        if os.path.exists(fallback):
            model_path = fallback
        else:
            raise FileNotFoundError(f"Model file not found at {model_path} or {fallback}. Train a model first!")

    print(f"\nLoading policy weights from: {model_path}")
    model = PPO.load(model_path)

    # Define Tesla-style Stress Test Scenarios
    scenarios = [
        (
            "1. Nominal Calm (Zero Wind / Gentle Tilt)",
            DomainRandomizationConfig(
                enable_wind=False,
                thrust_scale_range=(1.0, 1.0),
                obs_noise_std=0.0,
                initial_tilt_range=(-0.05, 0.05),
                initial_vel_x_range=(-0.1, 0.1)
            ),
            0.1
        ),
        (
            "2. Crosswind Turbulence (Turbulent Storm)",
            DomainRandomizationConfig(
                enable_wind=True,
                wind_power_range=(-10.0, 10.0),
                turbulence_power_range=(0.5, 1.8),
                thrust_scale_range=(0.95, 1.05),
                obs_noise_std=0.01
            ),
            1.0
        ),
        (
            "3. Emergency High-Tilt Recovery (+/- 25 deg)",
            DomainRandomizationConfig(
                enable_wind=False,
                thrust_scale_range=(1.0, 1.0),
                obs_noise_std=0.01,
                initial_tilt_range=(-0.45, 0.45),
                initial_vel_x_range=(-0.4, 0.4)
            ),
            1.0
        ),
        (
            "4. Actuator Degradation (80%-90% Thrust)",
            DomainRandomizationConfig(
                enable_wind=True,
                wind_power_range=(-4.0, 4.0),
                thrust_scale_range=(0.80, 0.90),
                obs_noise_std=0.015
            ),
            1.0
        ),
    ]

    print("\n" + "=" * 90)
    print(" TESLA-STYLE SIM-TO-REAL FLIGHT VERIFICATION BENCHMARK ")
    print("=" * 90)

    results = []
    for name, cfg, curr in scenarios:
        print(f"Running scenario: {name} ({n_episodes_per_scenario} episodes)...")
        res = run_benchmark_scenario(model, name, cfg, n_episodes_per_scenario, curr)
        results.append(res)

    print("\n" + "-" * 90)
    print(f"{'Scenario':<42} | {'Success':<8} | {'Crash':<7} | {'Mean Fuel':<10} | {'Touchdown Vel':<14}")
    print("-" * 90)
    for r in results:
        print(
            f"{r['scenario']:<42} | {r['success_rate']:>6.1f} % | {r['crash_rate']:>5.1f} % | "
            f"{r['mean_fuel']:>10.3f} | {r['mean_touchdown_speed']:>10.3f} m/s"
        )
    print("-" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trained Rocket Landing Policy")
    parser.add_argument("--model", type=str, default="models/best_model.zip", help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per scenario")
    args = parser.parse_args()

    evaluate(model_path=args.model, n_episodes_per_scenario=args.episodes)
