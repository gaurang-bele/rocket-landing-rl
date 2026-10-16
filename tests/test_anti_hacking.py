"""
Empirical Loophole & Reward Engine Verification Test.
Demonstrates mathematically and empirically how each agent exploit is prevented.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from config.settings import RewardConfig
from envs.reward_engine import AntiHackingRewardEngine


def test_hover_exploit_prevention():
    """
    Test 1: Hovering Exploit Prevention.
    If an agent tries to hover indefinitely near the target:
    Without fuel penalty, hovering yields ~0 step delta.
    With fuel penalty, every step drains score proportional to throttle.
    """
    print("\n--- Test 1: Hovering Loophole Prevention ---")
    cfg = RewardConfig()
    engine = AntiHackingRewardEngine(cfg)

    # Hovering at altitude y=0.5, zero velocity, upright
    hover_obs = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    # Firing main engine throttle (e.g. 0.5 -> 75% power) continuously to fight gravity
    hover_action = np.array([0.5, 0.0], dtype=np.float32)

    engine.reset(hover_obs)
    cumulative_reward = 0.0
    for step in range(100):
        reward, metrics = engine.compute_reward(
            obs=hover_obs,
            action=hover_action,
            terminated=False,
            truncated=False,
            info={}
        )
        cumulative_reward += reward

    print(f"Cumulative reward after 100 hover steps: {cumulative_reward:.2f}")
    assert cumulative_reward < -5.0, "Hovering should be strictly penalized by fuel and time costs!"
    print(" PASSED: Infinite hovering loophole eliminated. Agent cannot farm points by hovering.")


def test_suicide_crash_exploit_prevention():
    """
    Test 2: Kamikaze Suicide Crash Exploit Prevention.
    Verifies that plunging into the ground quickly yields a strictly worse score
    than attempting a safe landing.
    """
    print("\n--- Test 2: Kamikaze Suicide Exploit Prevention ---")
    cfg = RewardConfig()
    engine = AntiHackingRewardEngine(cfg)

    # Scenario A: Deliberate crash in 15 steps at high downward velocity
    engine.reset(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    crash_obs = np.array([0.0, 0.0, 0.0, -1.5, 0.8, 0.0, 0.0, 0.0], dtype=np.float32)
    dive_action = np.array([-1.0, 0.0], dtype=np.float32)  # Zero throttle dive
    crash_reward, metrics = engine.compute_reward(
        obs=crash_obs,
        action=dive_action,
        terminated=True,
        truncated=False,
        info={}
    )

    # Scenario B: Safe touchdown in 150 steps
    engine.reset(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    safe_obs = np.array([0.0, 0.0, 0.0, -0.05, 0.01, 0.0, 1.0, 1.0], dtype=np.float32)
    soft_action = np.array([0.2, 0.0], dtype=np.float32)
    safe_reward, safe_metrics = engine.compute_reward(
        obs=safe_obs,
        action=soft_action,
        terminated=True,
        truncated=False,
        info={}
    )

    print(f"Terminal reward for Suicide Crash: {metrics['terminal_reward']:.1f}")
    print(f"Terminal reward for Safe Landing : {safe_metrics['terminal_reward']:.1f}")
    assert metrics['terminal_reward'] <= -150.0, "Crash must be heavily penalized!"
    assert safe_metrics['terminal_reward'] >= 150.0, "Safe landing must provide substantial bonus!"
    assert safe_reward > crash_reward + 200.0, "Safe landing must be overwhelmingly superior to crashing!"
    print(" PASSED: Kamikaze suicide exploit eliminated. Safe landing is overwhelmingly prioritized.")


def test_spin_exploit_prevention():
    """
    Test 3: Spin-to-Win Exploit Prevention.
    Verifies that excessive rotational kinetic energy is penalized.
    """
    print("\n--- Test 3: Spin-to-Win Exploit Prevention ---")
    cfg = RewardConfig()
    engine = AntiHackingRewardEngine(cfg)

    upright_obs = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    spinning_obs = np.array([0.0, 0.5, 0.0, 0.0, 1.5, 4.0, 0.0, 0.0], dtype=np.float32)

    pot_upright = engine._compute_potential(upright_obs)
    pot_spinning = engine._compute_potential(spinning_obs)

    print(f"Potential for Upright Attitude : {pot_upright:.1f}")
    print(f"Potential for Rapidly Spinning: {pot_spinning:.1f}")
    assert pot_upright > pot_spinning + 200.0, "Spinning must be severely penalized in potential shaping!"
    print(" PASSED: Spin-to-win exploit eliminated. Upright orientation is strongly enforced.")


if __name__ == "__main__":
    print("=" * 60)
    print(" RUNNING ANTI-REWARD-HACKING VERIFICATION SUITE ")
    print("=" * 60)
    test_hover_exploit_prevention()
    test_suicide_crash_exploit_prevention()
    test_spin_exploit_prevention()
    print("\n All anti-reward-hacking tests passed successfully!")
