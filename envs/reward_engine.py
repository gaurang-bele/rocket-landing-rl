"""
Anti-Reward-Hacking Reward Engine.
Calculates potential-based rewards, fuel penalties, and calibrated terminal payouts
to prevent hovering, kamikaze crashing, and rotational exploits.
"""

import numpy as np
from typing import Tuple, Dict, Any
from config.settings import RewardConfig


class AntiHackingRewardEngine:
    def __init__(self, config: RewardConfig):
        self.cfg = config
        self.prev_potential = None
        self.total_fuel_burned = 0.0
        self.steps_count = 0

    def reset(self, initial_obs: np.ndarray):
        """Reset internal potential tracking at episode start."""
        self.total_fuel_burned = 0.0
        self.steps_count = 0
        self.prev_potential = self._compute_potential(initial_obs)

    def _compute_potential(self, obs: np.ndarray) -> float:
        """
        Potential function Phi(s) measuring how desirable a state is.
        Target is x=0, y=0, vx=0, vy=0, tilt=0, omega=0.
        """
        x = obs[0]
        y = obs[1]
        vx = obs[2]
        vy = obs[3]
        tilt = obs[4]
        omega = obs[5]
        left_leg = obs[6]
        right_leg = obs[7]

        dist = np.sqrt(x**2 + y**2)
        vel = np.sqrt(vx**2 + vy**2)
        tilt_abs = np.abs(tilt)
        omega_abs = np.abs(omega)

        potential = (
            - self.cfg.w_distance * dist
            - self.cfg.w_velocity * vel
            - self.cfg.w_tilt * tilt_abs
            - self.cfg.w_angular_vel * omega_abs
            + self.cfg.leg_contact_bonus * (left_leg + right_leg)
        )
        return float(potential)

    def compute_reward(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Computes the shaped reward and component breakdowns.
        Returns total_reward and a metrics dictionary.
        """
        self.steps_count += 1
        x, y, vx, vy, tilt, omega, left_leg, right_leg = obs

        # 1. Potential-based progress reward: Delta Phi = Phi(s') - Phi(s)
        current_potential = self._compute_potential(obs)
        if self.prev_potential is None:
            self.prev_potential = current_potential
        shaping_reward = current_potential - self.prev_potential
        self.prev_potential = current_potential

        # 2. Fuel Penalty (Action cost prevents hovering indefinitely)
        # In continuous action space: action[0] is main throttle [-1, 1], action[1] is side thruster [-1, 1]
        main_throttle = np.clip((action[0] + 1.0) / 2.0 if action[0] > 0 else 0.0, 0.0, 1.0)
        side_throttle = np.abs(action[1]) if np.abs(action[1]) > 0.5 else 0.0
        
        fuel_used = (
            self.cfg.fuel_penalty_main * main_throttle +
            self.cfg.fuel_penalty_side * side_throttle
        )
        self.total_fuel_burned += fuel_used

        # 3. Small per-step time pressure
        step_penalty = self.cfg.step_penalty

        # 4. Terminal Evaluation (Crash vs Safe Touchdown)
        terminal_reward = 0.0
        is_safe_landing = False
        is_crash = False

        if terminated:
            # Check if lander is upright and touched down softly
            both_legs_down = bool(left_leg and right_leg)
            is_upright = abs(tilt) <= self.cfg.max_safe_tilt
            is_slow_vertical = abs(vy) <= self.cfg.max_safe_vy
            is_slow_horizontal = abs(vx) <= self.cfg.max_safe_vx
            
            # Distance from center landing pad
            on_pad = abs(x) <= 0.4

            if both_legs_down and is_upright and is_slow_vertical and is_slow_horizontal and on_pad:
                is_safe_landing = True
                terminal_reward = self.cfg.safe_landing_bonus
            else:
                # Any premature termination that is not a qualified safe landing is penalized as crash
                is_crash = True
                terminal_reward = self.cfg.crash_penalty

        # Total combined reward
        total_reward = (
            shaping_reward
            - fuel_used
            - step_penalty
            + terminal_reward
        )

        metrics = {
            "shaping_reward": float(shaping_reward),
            "fuel_used": float(fuel_used),
            "total_fuel_burned": float(self.total_fuel_burned),
            "terminal_reward": float(terminal_reward),
            "is_safe_landing": float(is_safe_landing),
            "is_crash": float(is_crash),
            "speed": float(np.sqrt(vx**2 + vy**2)),
            "tilt_deg": float(np.degrees(abs(tilt))),
        }

        return float(total_reward), metrics
