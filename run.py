"""
Master CLI Entry Point for Autonomous Rocket Landing RL.
Supports:
- train: Multi-worker PPO reinforcement learning with curriculum.
- demo: Live interactive 1200x720 SpaceX avionics HUD window at 50 FPS.
- record: Headless high-def MP4 video export for LinkedIn / showcase.
- benchmark: 4-scenario Sim-to-Real stress benchmark.
- test: Anti-reward-hacking unit tests.
"""

import sys
import os

# Ensure Python's base install directory is registered for Windows DLL resolution (finds python312.dll)
if hasattr(os, "add_dll_directory") and hasattr(sys, "base_exec_prefix") and os.path.isdir(sys.base_exec_prefix):
    try:
        os.add_dll_directory(sys.base_exec_prefix)
    except Exception:
        pass

import time
import argparse
import numpy as np
import pygame

from config.settings import train_cfg, env_cfg, dr_cfg, reward_cfg, DomainRandomizationConfig
from envs import make_rocket_env
from telemetry import AvionicsHUD


def run_demo(
    model_path: str = "models/best_model.zip",
    n_episodes: int = 5,
    curriculum_level: float = 1.0,
    fps: int = 40
):
    """
    Renders the live simulation with full Tesla-style avionics HUD in a Pygame window.
    """
    from stable_baselines3 import PPO

    if not os.path.exists(model_path):
        fallback = "models/rocket_ppo_final.zip"
        if os.path.exists(fallback):
            model_path = fallback
        else:
            print(f"\n[WARNING] Checkpoint '{model_path}' not found.")
            print("Running in DEMO MODE with random baseline policy. Train a model to see intelligent landing!")
            model = None
    else:
        print(f"\nLoading trained model: {model_path}")
        model = PPO.load(model_path)

    # Initialize Environment with rgb_array render mode
    env_fn = make_rocket_env(render_mode="rgb_array", curriculum_level=curriculum_level)
    env = env_fn()

    hud = AvionicsHUD(width=1200, height=720)
    screen = pygame.display.set_mode((1200, 720))
    pygame.display.set_caption("SpaceX Autonomous Rocket Landing: Live Mission Control HUD")
    clock = pygame.time.Clock()

    print("\n" + "=" * 65)
    print(" LIVE SPACEX AUTONOMOUS RECOVERY AVIONICS HUD ")
    print(" Press ESC or Close Window to exit ")
    print("=" * 65)

    for episode in range(1, n_episodes + 1):
        obs, info = env.reset(seed=200 + episode)
        hud.reset()
        done = False
        step_idx = 0
        ep_reward = 0.0

        print(f"\n--- Episode {episode}/{n_episodes} (Wind: {info.get('wind_power', 0.0):+.1f}N, Motor Eff: {info.get('thrust_scale', 1.0)*100:.0f}%) ---")

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    env.close()
                    pygame.quit()
                    return

            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step_idx += 1
            done = terminated or truncated

            sim_rgb = env.render()
            if sim_rgb is not None:
                composite_frame = hud.render_frame(
                    sim_rgb_array=sim_rgb,
                    obs=obs,
                    action=action,
                    info=info,
                    step_idx=step_idx,
                    total_reward=ep_reward,
                    env_unwrapped=env.unwrapped
                )
                
                surf = pygame.surfarray.make_surface(np.transpose(composite_frame, (1, 0, 2)))
                screen.blit(surf, (0, 0))
                pygame.display.flip()

            clock.tick(fps)

        status = "SUCCESSFUL TOUCHDOWN" if info.get("is_safe_landing", 0) > 0 else "CRASH / ABORT"
        print(f"Episode {episode} Result: {status} | Score: {ep_reward:.1f} | Fuel Burned: {info.get('total_fuel_burned', 0.0):.3f}")
        time.sleep(1.2)

    env.close()
    pygame.quit()
    print("\nDemo finished.")


def record_video(
    model_path: str = "models/best_model.zip",
    output_path: str = "rocket_landing_linkedin.mp4",
    n_episodes: int = 3,
    curriculum_level: float = 1.0,
    fps: int = 50
):
    """
    Renders and exports a high-definition 50 FPS MP4 video ready for LinkedIn!
    """
    from stable_baselines3 import PPO
    import imageio

    if not os.path.exists(model_path):
        fallback = "models/rocket_ppo_final.zip"
        if os.path.exists(fallback):
            model_path = fallback
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}. Train a model first!")

    print(f"\nLoading trained model: {model_path}")
    model = PPO.load(model_path)

    env_fn = make_rocket_env(render_mode="rgb_array", curriculum_level=curriculum_level)
    env = env_fn()

    hud = AvionicsHUD(width=1200, height=720)
    frames = []

    print(f"\nRecording high-definition video ({n_episodes} episodes) to: {output_path}")
    print("Simulating landing sequence at 50 FPS...")

    # Seeds selected for diverse wind gusts, approach angles, and smooth hoverslam landings
    diverse_seeds = [202, 105, 303, 404, 505, 606, 707, 808]

    for ep in range(n_episodes):
        seed = diverse_seeds[ep % len(diverse_seeds)]
        obs, info = env.reset(seed=seed)
        hud.reset()
        done = False
        step_idx = 0
        ep_reward = 0.0

        print(f"  -> Recording Episode {ep + 1}/{n_episodes} (Seed: {seed})...")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step_idx += 1
            done = terminated or truncated

            sim_rgb = env.render()
            composite_frame = hud.render_frame(
                sim_rgb_array=sim_rgb,
                obs=obs,
                action=action,
                info=info,
                step_idx=step_idx,
                total_reward=ep_reward,
                env_unwrapped=env.unwrapped
            )
            frames.append(composite_frame)

        status = "SUCCESSFUL TOUCHDOWN" if info.get("is_safe_landing", 0) > 0 else "CRASH / ABORT"
        print(f"     Episode {ep + 1} finished: {status} | Score: {ep_reward:.1f}")

        # Add a pause at touchdown so viewers can appreciate the landing
        hold_frames = 35 if ep < n_episodes - 1 else 50
        for _ in range(hold_frames):
            frames.append(frames[-1])

    env.close()

    print(f"\nCaptured {len(frames)} total frames ({len(frames)/fps:.1f} seconds). Encoding MP4 video...")
    writer = imageio.get_writer(output_path, fps=fps, quality=9)
    for f in frames:
        writer.append_data(f)
    writer.close()

    print("\n" + "=" * 65)
    print(f" VIDEO SAVED SUCCESSFULLY: {output_path} ")
    print(f" Duration: {len(frames)/fps:.1f}s | Resolution: 1200x720 | FPS: {fps}")
    print(" Ready to upload directly to LinkedIn!")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Tesla-Style Autonomous Rocket Landing RL System")
    parser.add_argument(
        "--mode",
        choices=["train", "demo", "record", "benchmark", "test"],
        default="demo",
        help="Execution mode"
    )
    parser.add_argument("--timesteps", type=int, default=train_cfg.total_timesteps, help="Timesteps for training")
    parser.add_argument("--envs", type=int, default=train_cfg.num_envs, help="Parallel environments")
    parser.add_argument("--model", type=str, default="models/best_model.zip", help="Model checkpoint path")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes for demo, evaluation, or recording")
    parser.add_argument("--difficulty", type=float, default=1.0, help="Curriculum difficulty (0.0 to 1.0)")
    parser.add_argument("--output", type=str, default="rocket_landing_linkedin.mp4", help="Output MP4 file path for recording")
    args = parser.parse_args()

    if args.mode == "test":
        from tests.test_anti_hacking import (
            test_hover_exploit_prevention,
            test_suicide_crash_exploit_prevention,
            test_spin_exploit_prevention
        )
        print("\nRunning anti-hacking verification suite...")
        test_hover_exploit_prevention()
        test_suicide_crash_exploit_prevention()
        test_spin_exploit_prevention()
        print("\nAll exploit prevention tests passed!")

    elif args.mode == "train":
        from train import train
        train(total_timesteps=args.timesteps, num_envs=args.envs)

    elif args.mode == "demo":
        run_demo(model_path=args.model, n_episodes=args.episodes, curriculum_level=args.difficulty)

    elif args.mode == "record":
        record_video(
            model_path=args.model,
            output_path=args.output,
            n_episodes=args.episodes,
            curriculum_level=args.difficulty
        )

    elif args.mode == "benchmark":
        from evaluate import evaluate
        evaluate(model_path=args.model, n_episodes_per_scenario=args.episodes)


if __name__ == "__main__":
    main()
