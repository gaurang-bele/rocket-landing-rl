"""
Multi-worker PPO Training Pipeline with Curriculum Learning & TensorBoard Logging.
Trains the rocket landing policy across vectorized environments.
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
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from config.settings import train_cfg, env_cfg, dr_cfg, reward_cfg
from envs import make_rocket_env


class CurriculumCallback(BaseCallback):
    """
    Progressively increases environment difficulty (wind, turbulence, tilt perturbations)
    as the agent reaches training milestones.
    """
    def __init__(self, total_timesteps: int, verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.current_level = 0.2

    def _on_step(self) -> bool:
        progress = self.num_timesteps / float(self.total_timesteps)

        # 3-Stage Curriculum
        if progress < 0.2:
            new_level = 0.2  # Stage 1: Gentle drops, minimal wind
        elif progress < 0.5:
            new_level = 0.6  # Stage 2: Moderate crosswinds and lateral drift
        else:
            new_level = 1.0  # Stage 3: Full Tesla-grade stress test

        if new_level != self.current_level:
            self.current_level = new_level
            if self.verbose > 0:
                print(f"\n[CURRICULUM UPGRADE] Step {self.num_timesteps:,}: Difficulty Level -> {self.current_level * 100:.0f}%")
            self.training_env.env_method("set_curriculum_level", self.current_level)

        return True


def train(
    total_timesteps: int = train_cfg.total_timesteps,
    num_envs: int = train_cfg.num_envs,
    save_dir: str = train_cfg.model_dir,
    log_dir: str = train_cfg.log_dir
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 70)
    print(" TESLA-STYLE AUTONOMOUS ROCKET LANDING: PPO TRAINING PIPELINE ")
    print("=" * 70)
    print(f"Total Timesteps  : {total_timesteps:,}")
    print(f"Parallel Workers : {num_envs}")
    print(f"Log Directory    : {log_dir}")
    print(f"Model Directory  : {save_dir}")
    print("=" * 70)

    # 1. Create Vectorized Training Environments
    env_fns = [make_rocket_env(curriculum_level=0.2) for _ in range(num_envs)]
    if num_envs > 1:
        train_env = SubprocVecEnv(env_fns)
    else:
        train_env = DummyVecEnv(env_fns)

    # 2. Evaluation Environment
    eval_env_fn = make_rocket_env(curriculum_level=1.0)
    eval_env = Monitor(eval_env_fn())

    # 3. Callbacks
    curriculum_cb = CurriculumCallback(total_timesteps=total_timesteps)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=train_cfg.eval_freq // num_envs,
        n_eval_episodes=train_cfg.eval_episodes,
        deterministic=True,
        render=False
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000 // num_envs,
        save_path=save_dir,
        name_prefix="rocket_ppo_step"
    )

    # 4. Neural Network Policy Architecture
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=train_cfg.learning_rate,
        n_steps=train_cfg.n_steps,
        batch_size=train_cfg.batch_size,
        n_epochs=train_cfg.n_epochs,
        gamma=train_cfg.gamma,
        gae_lambda=train_cfg.gae_lambda,
        clip_range=train_cfg.clip_range,
        ent_coef=train_cfg.ent_coef,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        verbose=1,
    )

    # Check if tqdm and rich are available for the progress bar
    try:
        import tqdm  # noqa: F401
        import rich  # noqa: F401
        use_progress_bar = True
    except ImportError:
        use_progress_bar = False

    print("\nStarting reinforcement learning loop...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[curriculum_cb, eval_cb, checkpoint_cb],
            progress_bar=use_progress_bar
        )
    finally:
        train_env.close()
        eval_env.close()

    # Save final model
    final_path = os.path.join(save_dir, "rocket_ppo_final.zip")
    model.save(final_path)
    print(f"\nTraining completed! Final model saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Rocket Landing RL Agent")
    parser.add_argument("--timesteps", type=int, default=train_cfg.total_timesteps, help="Total training timesteps")
    parser.add_argument("--envs", type=int, default=train_cfg.num_envs, help="Number of parallel environments")
    args = parser.parse_args()

    train(total_timesteps=args.timesteps, num_envs=args.envs)
