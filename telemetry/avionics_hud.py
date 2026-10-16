"""
Avionics Telemetry HUD & Real-time Flight Instrument Visualizer.
Renders high-fidelity telemetry gauges, artificial horizon, descent velocity indicators,
fuel mass meters, and trajectory breadcrumbs using Pygame.
"""

import math
import numpy as np
import pygame
from typing import Dict, Any, List, Tuple, Optional


class AvionicsHUD:
    def __init__(self, width: int = 1000, height: int = 700):
        pygame.init()
        pygame.font.init()
        
        self.width = width
        self.height = height
        self.sim_width = 700
        self.sim_height = 700
        self.hud_width = width - self.sim_width  # 300px sidebar
        
        # Color Palette (Dark Sci-Fi Cockpit / Tesla Telemetry)
        self.BG_COLOR = (12, 16, 24)
        self.PANEL_BG = (18, 24, 38)
        self.BORDER_COLOR = (35, 48, 72)
        self.TEXT_COLOR = (220, 230, 242)
        self.CYAN = (0, 220, 255)
        self.GREEN = (0, 255, 136)
        self.YELLOW = (255, 204, 0)
        self.RED = (255, 51, 85)
        self.DARK_GRAY = (40, 50, 70)
        
        # Fonts
        self.font_title = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_main = pygame.font.SysFont("Consolas", 12)
        self.font_large = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_badge = pygame.font.SysFont("Consolas", 13, bold=True)
        
        from telemetry.rocket_renderer import RocketRenderer
        self.rocket_renderer = RocketRenderer(viewport_w=self.sim_width, viewport_h=self.sim_height)

        # Trajectory history cache: list of (x_screen, y_screen)
        self.trajectory_points: List[Tuple[int, int]] = []
        self.max_points = 200

    def reset(self):
        """Reset flight trajectory history and particle effects."""
        self.trajectory_points.clear()
        self.rocket_renderer.exhaust_particles.clear()
        self.rocket_renderer.rcs_particles.clear()

    def render_frame(
        self,
        sim_rgb_array: Optional[np.ndarray],
        obs: np.ndarray,
        action: np.ndarray,
        info: Dict[str, Any],
        step_idx: int,
        total_reward: float,
        env_unwrapped=None
    ) -> np.ndarray:
        """
        Renders the SpaceX Falcon 9 simulation viewport and composites the Avionics HUD sidebar.
        Returns (H, W, 3) uint8 array.
        """
        surface = pygame.Surface((self.width, self.height))
        surface.fill(self.BG_COLOR)

        # 1. Render Simulation Viewport (left 700x700)
        if env_unwrapped is not None and hasattr(env_unwrapped, "lander") and env_unwrapped.lander is not None:
            # Record noiseless true physical position for smooth trajectory
            px, py = self.rocket_renderer.world_to_screen(
                env_unwrapped.lander.position.x,
                env_unwrapped.lander.position.y
            )
            self.trajectory_points.append((px, py))
            if len(self.trajectory_points) > self.max_points:
                self.trajectory_points.pop(0)

            # Render photorealistic SpaceX Falcon 9 rocket and drone ship
            cinematic_surf = self.rocket_renderer.render_cinematic_world(
                env_unwrapped=env_unwrapped,
                action=action,
                info=info,
                trajectory_points=self.trajectory_points
            )
            surface.blit(cinematic_surf, (0, 0))
        elif sim_rgb_array is not None:
            sim_surf = pygame.surfarray.make_surface(np.transpose(sim_rgb_array, (1, 0, 2)))
            scaled_sim = pygame.transform.smoothscale(sim_surf, (self.sim_width, self.sim_height))
            surface.blit(scaled_sim, (0, 0))

        # 2. Render HUD Sidebar (Right 300px)
        sidebar_rect = pygame.Rect(self.sim_width, 0, self.hud_width, self.height)
        pygame.draw.rect(surface, self.PANEL_BG, sidebar_rect)
        pygame.draw.line(surface, self.BORDER_COLOR, (self.sim_width, 0), (self.sim_width, self.height), 2)

        y_offset = 15
        x_start = self.sim_width + 15

        # Header Title
        title_surf = self.font_title.render("AUTONOMOUS RECOVERY AVIONICS", True, self.CYAN)
        surface.blit(title_surf, (x_start, y_offset))
        y_offset += 24
        sub_surf = self.font_main.render("TESLA-STYLE RL FLIGHT CONTROLLER", True, (140, 160, 190))
        surface.blit(sub_surf, (x_start, y_offset))
        y_offset += 28

        pygame.draw.line(surface, self.BORDER_COLOR, (x_start, y_offset), (self.width - 15, y_offset), 1)
        y_offset += 15

        # Flight Status Badge
        vx, vy = obs[2], obs[3]
        tilt_rad = obs[4]
        tilt_deg = math.degrees(tilt_rad)
        speed = math.sqrt(vx**2 + vy**2)
        left_leg, right_leg = bool(obs[6]), bool(obs[7])

        status_text, status_color = self._determine_status(speed, tilt_deg, left_leg, right_leg, info)
        badge_surf = self.font_badge.render(f"STATUS: {status_text}", True, status_color)
        badge_bg = pygame.Rect(x_start, y_offset, self.hud_width - 30, 26)
        pygame.draw.rect(surface, (28, 38, 56), badge_bg, border_radius=4)
        pygame.draw.rect(surface, status_color, badge_bg, width=1, border_radius=4)
        surface.blit(badge_surf, (x_start + 10, y_offset + 5))
        y_offset += 40

        # Altimeter & Descent Velocity
        y_offset = self._draw_flight_dynamics(surface, x_start, y_offset, obs, speed, tilt_deg)

        # Artificial Horizon / Pitch Indicator
        y_offset = self._draw_artificial_horizon(surface, x_start, y_offset, tilt_rad)

        # Propulsion & Actuator Telemetry
        y_offset = self._draw_propulsion_telemetry(surface, x_start, y_offset, action, info)

        # Sim-to-Real Domain Randomization Diagnostics
        y_offset = self._draw_sim_to_real_diagnostics(surface, x_start, y_offset, info)

        # Performance & Reward Score
        self._draw_score_panel(surface, x_start, y_offset, total_reward, step_idx)

        # Convert back to RGB numpy array
        output_array = pygame.surfarray.array3d(surface)
        return np.transpose(output_array, (1, 0, 2))

    def _determine_status(
        self,
        speed: float,
        tilt_deg: float,
        left_leg: bool,
        right_leg: bool,
        info: Dict[str, Any]
    ) -> Tuple[str, Tuple[int, int, int]]:
        if info.get("is_safe_landing", 0) > 0 or (left_leg and right_leg and speed < 0.2 and abs(tilt_deg) < 8):
            return "TOUCHDOWN SAFE", self.GREEN
        if info.get("is_crash", 0) > 0:
            return "TERMINAL CRASH", self.RED
        if abs(tilt_deg) > 15:
            return "EXCESSIVE TILT", self.YELLOW
        if speed > 0.6:
            return "HIGH VELOCITY", self.YELLOW
        return "NOMINAL GLIDE", self.CYAN

    def _draw_flight_dynamics(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        obs: np.ndarray,
        speed: float,
        tilt_deg: float
    ) -> int:
        header = self.font_title.render("PRIMARY FLIGHT DATA", True, self.TEXT_COLOR)
        surface.blit(header, (x, y))
        y += 20

        alt_m = max(0.0, float(obs[1]) * 100.0)
        vx_ms = float(obs[2]) * 20.0
        vy_ms = float(obs[3]) * 20.0

        items = [
            ("ALTITUDE (AGL)", f"{alt_m:6.1f} m", self.TEXT_COLOR),
            ("VERTICAL SPEED", f"{vy_ms:+6.2f} m/s", self.GREEN if abs(vy_ms) < 4.0 else self.YELLOW),
            ("HORIZONTAL SPD", f"{vx_ms:+6.2f} m/s", self.GREEN if abs(vx_ms) < 4.0 else self.YELLOW),
            ("PITCH ATTITUDE", f"{tilt_deg:+6.1f}°", self.GREEN if abs(tilt_deg) < 10.0 else self.YELLOW),
            ("LEGS LOCKED", f"L:{int(obs[6])} R:{int(obs[7])}", self.CYAN if (obs[6] and obs[7]) else self.TEXT_COLOR),
        ]

        for label, val, color in items:
            lbl_s = self.font_main.render(label, True, (150, 165, 185))
            val_s = self.font_main.render(val, True, color)
            surface.blit(lbl_s, (x, y))
            surface.blit(val_s, (x + 160, y))
            y += 18

        y += 8
        return y

    def _draw_artificial_horizon(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        tilt_rad: float
    ) -> int:
        center_x = x + (self.hud_width - 30) // 2
        center_y = y + 36
        radius = 32

        # Draw circle background
        pygame.draw.circle(surface, (24, 32, 48), (center_x, center_y), radius)
        pygame.draw.circle(surface, self.BORDER_COLOR, (center_x, center_y), radius, width=2)

        # Draw pitch line rotated by tilt
        line_len = 26
        dx = line_len * math.cos(tilt_rad)
        dy = line_len * math.sin(tilt_rad)
        p1 = (int(center_x - dx), int(center_y - dy))
        p2 = (int(center_x + dx), int(center_y + dy))
        pygame.draw.line(surface, self.CYAN, p1, p2, width=3)

        # Center reticle
        pygame.draw.circle(surface, self.YELLOW, (center_x, center_y), 3)

        y += 80
        return y

    def _draw_propulsion_telemetry(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        action: np.ndarray,
        info: Dict[str, Any]
    ) -> int:
        header = self.font_title.render("PROPULSION ACTUATORS", True, self.TEXT_COLOR)
        surface.blit(header, (x, y))
        y += 20

        # Main throttle (0 to 100%)
        main_throttle = np.clip((action[0] + 1.0) / 2.0 if action[0] > 0 else 0.0, 0.0, 1.0)
        side_gimbal = float(action[1])

        # Main Engine Progress Bar
        bar_w = 170
        bar_h = 10
        pygame.draw.rect(surface, (30, 40, 56), (x + 90, y + 2, bar_w, bar_h))
        pygame.draw.rect(surface, self.CYAN, (x + 90, y + 2, int(bar_w * main_throttle), bar_h))
        lbl_main = self.font_main.render(f"MAIN ({int(main_throttle*100)}%)", True, (150, 165, 185))
        surface.blit(lbl_main, (x, y))
        y += 18

        # Attitude Thrusters Bar (-100% to +100%)
        pygame.draw.rect(surface, (30, 40, 56), (x + 90, y + 2, bar_w, bar_h))
        center_bar_x = x + 90 + bar_w // 2
        thruster_len = int((bar_w // 2) * side_gimbal)
        if thruster_len > 0:
            pygame.draw.rect(surface, self.YELLOW, (center_bar_x, y + 2, thruster_len, bar_h))
        else:
            pygame.draw.rect(surface, self.YELLOW, (center_bar_x + thruster_len, y + 2, abs(thruster_len), bar_h))
        pygame.draw.line(surface, self.TEXT_COLOR, (center_bar_x, y), (center_bar_x, y + bar_h + 3), 1)

        lbl_side = self.font_main.render("RCS THRUST", True, (150, 165, 185))
        surface.blit(lbl_side, (x, y))
        y += 26

        return y

    def _draw_sim_to_real_diagnostics(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        info: Dict[str, Any]
    ) -> int:
        header = self.font_title.render("SIM-TO-REAL TELEMETRY", True, self.TEXT_COLOR)
        surface.blit(header, (x, y))
        y += 20

        wind_pwr = info.get("wind_power", 0.0)
        thrust_eff = info.get("thrust_scale", 1.0) * 100.0
        curriculum = info.get("curriculum_level", 1.0) * 100.0

        items = [
            ("WIND POWER", f"{wind_pwr:+5.1f} N", self.CYAN if abs(wind_pwr) < 4.0 else self.YELLOW),
            ("MOTOR EFFICIENCY", f"{thrust_eff:5.1f} %", self.TEXT_COLOR),
            ("CURRICULUM LEVEL", f"{curriculum:5.0f} %", self.GREEN),
        ]

        for label, val, color in items:
            lbl_s = self.font_main.render(label, True, (150, 165, 185))
            val_s = self.font_main.render(val, True, color)
            surface.blit(lbl_s, (x, y))
            surface.blit(val_s, (x + 160, y))
            y += 18

        y += 10
        return y

    def _draw_score_panel(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        total_reward: float,
        step_idx: int
    ):
        pygame.draw.line(surface, self.BORDER_COLOR, (x, y), (self.width - 15, y), 1)
        y += 12

        rew_color = self.GREEN if total_reward >= 0 else self.RED
        r_text = self.font_large.render(f"SCORE: {total_reward:+6.1f}", True, rew_color)
        surface.blit(r_text, (x, y))

        step_text = self.font_main.render(f"SIM STEPS: {step_idx}", True, (150, 165, 185))
        surface.blit(step_text, (x, y + 26))
