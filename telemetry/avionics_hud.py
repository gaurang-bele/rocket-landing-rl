"""
LinkedIn-Grade Mission Control Avionics HUD & Flight Visualizer.
Features:
- 16:9 Widescreen 1200x720 Layout with Dark Glassmorphism Telemetry Cards.
- Mission Elapsed Clock (T+00:XX.X).
- Live Phase-Plane Hoverslam Trajectory Corridor (Altitude vs Velocity).
- Artificial Horizon Attitude Indicator with Pitch/Bank Angle.
- Propulsion & RCS Actuator Gauges.
- Sim-to-Real Domain Randomization Telemetry (Wind Vector, Motor Wear).
- Real-time Reward Scoring & Fuel Budget Gauge.
"""

import math
import numpy as np
import pygame
from typing import Dict, Any, List, Tuple, Optional


class AvionicsHUD:
    def __init__(self, width: int = 1200, height: int = 720):
        pygame.init()
        pygame.font.init()
        
        self.width = width
        self.height = height
        self.sim_width = 840
        self.sim_height = 720
        self.hud_width = width - self.sim_width  # 360px sidebar
        
        # Color Palette (SpaceX Webcast & Tesla Cybertruck Dark Glass)
        self.BG_COLOR = (8, 12, 18)
        self.PANEL_BG = (14, 20, 32)
        self.CARD_BG = (20, 28, 44)
        self.BORDER_COLOR = (36, 52, 78)
        self.TEXT_COLOR = (230, 240, 252)
        self.CYAN = (0, 220, 255)
        self.GREEN = (0, 255, 136)
        self.YELLOW = (255, 204, 0)
        self.RED = (255, 51, 85)
        self.DARK_GRAY = (40, 52, 72)
        
        # Typography
        self.font_header = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_sub = pygame.font.SysFont("Consolas", 11)
        self.font_body = pygame.font.SysFont("Consolas", 12)
        self.font_data = pygame.font.SysFont("Consolas", 13, bold=True)
        self.font_score = pygame.font.SysFont("Consolas", 22, bold=True)
        self.font_clock = pygame.font.SysFont("Consolas", 15, bold=True)
        
        from telemetry.rocket_renderer import RocketRenderer
        self.rocket_renderer = RocketRenderer(viewport_w=self.sim_width, viewport_h=self.sim_height)

        # Trajectory history cache: list of (x_screen, y_screen)
        self.trajectory_points: List[Tuple[int, int]] = []
        self.max_points = 250

        # Phase-plane history: list of (alt, vy)
        self.phase_plane_history: List[Tuple[float, float]] = []

    def reset(self):
        """Reset flight telemetry history."""
        self.trajectory_points.clear()
        self.phase_plane_history.clear()
        self.rocket_renderer.exhaust_particles.clear()
        self.rocket_renderer.rcs_particles.clear()
        self.rocket_renderer.dust_particles.clear()
        self.rocket_renderer.confetti_particles.clear()
        self.rocket_renderer.touchdown_celebrated = False

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
        surface = pygame.Surface((self.width, self.height))
        surface.fill(self.BG_COLOR)

        # 1. Render Cinematic Simulation Viewport (Left 840x720)
        if env_unwrapped is not None and hasattr(env_unwrapped, "lander") and env_unwrapped.lander is not None:
            px, py = self.rocket_renderer.world_to_screen(
                env_unwrapped.lander.position.x,
                env_unwrapped.lander.position.y
            )
            self.trajectory_points.append((px, py))
            if len(self.trajectory_points) > self.max_points:
                self.trajectory_points.pop(0)

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

        # 2. Render Mission Control HUD Sidebar (Right 360x720)
        sidebar_rect = pygame.Rect(self.sim_width, 0, self.hud_width, self.height)
        pygame.draw.rect(surface, self.PANEL_BG, sidebar_rect)
        pygame.draw.line(surface, self.BORDER_COLOR, (self.sim_width, 0), (self.sim_width, self.height), 2)

        x_start = self.sim_width + 16
        y_cursor = 16

        # --- Section 1: SpaceX Broadcast Header & Clock ---
        y_cursor = self._draw_mission_header(surface, x_start, y_cursor, step_idx)

        # --- Section 2: Mission Flight Status Badge ---
        y_cursor = self._draw_status_badge(surface, x_start, y_cursor, obs, info)

        # --- Section 3: Primary Flight Telemetry ---
        y_cursor = self._draw_primary_telemetry(surface, x_start, y_cursor, obs)

        # --- Section 4: Artificial Horizon Attitude Compass ---
        y_cursor = self._draw_attitude_indicator(surface, x_start, y_cursor, obs[4])

        # --- Section 5: Phase-Plane Hoverslam Corridor Graph ---
        y_cursor = self._draw_phase_plane_corridor(surface, x_start, y_cursor, obs[1], obs[3])

        # --- Section 6: Propulsion Actuators (Main Engine & RCS Thrusters) ---
        y_cursor = self._draw_propulsion_telemetry(surface, x_start, y_cursor, action, info)

        # --- Section 7: Sim-to-Real Domain Randomization Diagnostics ---
        y_cursor = self._draw_sim_to_real_diagnostics(surface, x_start, y_cursor, info)

        # --- Section 8: Live Neural Net Reward Score Card ---
        self._draw_score_card(surface, x_start, y_cursor, total_reward, step_idx)

        # Convert to numpy array (H, W, 3)
        output_array = pygame.surfarray.array3d(surface)
        return np.transpose(output_array, (1, 0, 2))

    def _draw_mission_header(self, surface: pygame.Surface, x: int, y: int, step_idx: int) -> int:
        # Title
        t1 = self.font_header.render("SPACEX AUTONOMOUS RECOVERY", True, self.CYAN)
        surface.blit(t1, (x, y))
        y += 20

        # Subtitle
        t2 = self.font_sub.render("DEEP RL FLIGHT CONTROLLER // PPO", True, (150, 170, 195))
        surface.blit(t2, (x, y))

        # Mission Elapsed Timer (T+00:XX.X assuming 50Hz)
        sim_sec = step_idx * 0.02
        clock_txt = f"T+{int(sim_sec // 60):02d}:{sim_sec % 60:04.1f}"
        clock_surf = self.font_clock.render(clock_txt, True, self.YELLOW)
        surface.blit(clock_surf, (self.width - 16 - clock_surf.get_width(), y - 2))

        y += 26
        pygame.draw.line(surface, self.BORDER_COLOR, (x, y), (self.width - 16, y), 1)
        y += 12
        return y

    def _draw_status_badge(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        obs: np.ndarray,
        info: Dict[str, Any]
    ) -> int:
        speed = math.sqrt(obs[2]**2 + obs[3]**2)
        tilt_deg = math.degrees(abs(obs[4]))
        left_leg, right_leg = bool(obs[6]), bool(obs[7])

        if info.get("is_safe_landing", 0) > 0 or (left_leg and right_leg and speed < 0.2 and tilt_deg < 8):
            status = "TOUCHDOWN CONFIRMED"
            col = self.GREEN
        elif info.get("is_crash", 0) > 0:
            status = "IMPACT / FLIGHT ABORT"
            col = self.RED
        elif tilt_deg > 14:
            status = "CRITICAL TILT ATTITUDE"
            col = self.YELLOW
        elif speed > 0.6:
            status = "HIGH VELOCITY GLIDE"
            col = self.CYAN
        else:
            status = "TERMINAL HOVERSLAM"
            col = self.CYAN

        card_w = self.hud_width - 32
        card_h = 28
        badge_rect = pygame.Rect(x, y, card_w, card_h)
        pygame.draw.rect(surface, (20, 30, 48), badge_rect, border_radius=5)
        pygame.draw.rect(surface, col, badge_rect, width=1, border_radius=5)

        # Pulse indicator circle
        pygame.draw.circle(surface, col, (x + 14, y + card_h // 2), 4)

        text_s = self.font_data.render(status, True, col)
        surface.blit(text_s, (x + 26, y + 6))

        y += card_h + 14
        return y

    def _draw_primary_telemetry(self, surface: pygame.Surface, x: int, y: int, obs: np.ndarray) -> int:
        hdr = self.font_header.render("PRIMARY FLIGHT DYNAMICS", True, self.TEXT_COLOR)
        surface.blit(hdr, (x, y))
        y += 20

        alt_m = max(0.0, float(obs[1]) * 100.0)
        vy_ms = float(obs[3]) * 20.0
        vx_ms = float(obs[2]) * 20.0
        tilt_deg = math.degrees(obs[4])

        rows = [
            ("ALTITUDE (AGL)", f"{alt_m:6.1f} m", self.TEXT_COLOR),
            ("VERTICAL VELOCITY", f"{vy_ms:+6.2f} m/s", self.GREEN if abs(vy_ms) < 4.0 else self.YELLOW),
            ("HORIZONTAL VELOCITY", f"{vx_ms:+6.2f} m/s", self.GREEN if abs(vx_ms) < 3.0 else self.YELLOW),
            ("PITCH ATTITUDE", f"{tilt_deg:+6.1f}°", self.GREEN if abs(tilt_deg) < 10.0 else self.YELLOW),
            ("LEGS CONTACT", f"L:{'LOCKED' if obs[6] else 'OPEN'} R:{'LOCKED' if obs[7] else 'OPEN'}", self.CYAN if (obs[6] and obs[7]) else (150, 165, 185)),
        ]

        for label, val, color in rows:
            lbl_s = self.font_body.render(label, True, (150, 165, 185))
            val_s = self.font_data.render(val, True, color)
            surface.blit(lbl_s, (x, y))
            surface.blit(val_s, (self.width - 16 - val_s.get_width(), y))
            y += 18

        y += 8
        return y

    def _draw_attitude_indicator(self, surface: pygame.Surface, x: int, y: int, tilt_rad: float) -> int:
        center_x = x + (self.hud_width - 32) // 2
        center_y = y + 36
        radius = 32

        # Artificial Horizon Instrument Dial
        pygame.draw.circle(surface, (22, 30, 46), (center_x, center_y), radius)
        pygame.draw.circle(surface, self.BORDER_COLOR, (center_x, center_y), radius, width=2)

        # Pitch Reticle Line
        line_len = 26
        dx = line_len * math.cos(tilt_rad)
        dy = line_len * math.sin(tilt_rad)
        pygame.draw.line(surface, self.CYAN, (int(center_x - dx), int(center_y - dy)), (int(center_x + dx), int(center_y + dy)), width=3)
        pygame.draw.circle(surface, self.YELLOW, (center_x, center_y), 3)

        # Degree Labels
        deg_txt = f"{math.degrees(tilt_rad):+4.1f}°"
        deg_s = self.font_sub.render(deg_txt, True, self.CYAN)
        surface.blit(deg_s, (center_x - deg_s.get_width() // 2, center_y + radius + 4))

        y += 86
        return y

    def _draw_phase_plane_corridor(self, surface: pygame.Surface, x: int, y: int, alt_norm: float, vy_norm: float) -> int:
        """
        Draws a live Phase-Plane corridor graph: Altitude vs Vertical Speed.
        Shows the optimal hoverslam envelope curve and the rocket's position inside it.
        """
        hdr = self.font_header.render("HOVERSLAM ENVELOPE (PHASE-PLANE)", True, self.TEXT_COLOR)
        surface.blit(hdr, (x, y))
        y += 20

        graph_w = self.hud_width - 32
        graph_h = 70
        graph_rect = pygame.Rect(x, y, graph_w, graph_h)
        pygame.draw.rect(surface, (18, 25, 38), graph_rect, border_radius=4)
        pygame.draw.rect(surface, self.BORDER_COLOR, graph_rect, width=1, border_radius=4)

        # Axes: X is Altitude (0 to 120m), Y is Vertical Speed (-20 to +5 m/s)
        # Safe hoverslam boundary curve: v_safe = -sqrt(2 * g * y)
        curve_pts = []
        for gx in range(0, graph_w, 4):
            y_ratio = gx / float(graph_w)
            # Theoretical suicide burn safe limit
            vy_limit = -math.sqrt(max(0.0, y_ratio)) * 0.8
            gy = int(graph_h * 0.7 - (vy_limit * (graph_h * 0.4)))
            curve_pts.append((x + gx, y + np.clip(gy, 4, graph_h - 4)))

        if len(curve_pts) > 1:
            pygame.draw.lines(surface, (50, 90, 70), False, curve_pts, width=1)

        # Current live state point
        alt_x = int(x + np.clip(alt_norm, 0.0, 1.2) * (graph_w / 1.2))
        vel_y = int(y + graph_h * 0.5 - (vy_norm * 40.0))
        vel_y = np.clip(vel_y, y + 4, y + graph_h - 4)

        self.phase_plane_history.append((alt_x, vel_y))
        if len(self.phase_plane_history) > 40:
            self.phase_plane_history.pop(0)

        # Draw mini trail
        if len(self.phase_plane_history) > 1:
            pygame.draw.lines(surface, (0, 180, 220), False, self.phase_plane_history, width=1)

        pygame.draw.circle(surface, self.GREEN, (alt_x, vel_y), 4)

        y += graph_h + 14
        return y

    def _draw_propulsion_telemetry(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        action: np.ndarray,
        info: Dict[str, Any]
    ) -> int:
        hdr = self.font_header.render("PROPULSION ACTUATORS", True, self.TEXT_COLOR)
        surface.blit(hdr, (x, y))
        y += 20

        main_throttle = np.clip((action[0] + 1.0) / 2.0 if action[0] > 0 else 0.0, 0.0, 1.0)
        side_gimbal = float(action[1])

        bar_w = self.hud_width - 140
        bar_h = 10

        # Main Throttle
        lbl_main = self.font_body.render(f"MAIN ({int(main_throttle*100)}%)", True, (150, 165, 185))
        surface.blit(lbl_main, (x, y))
        pygame.draw.rect(surface, (25, 35, 52), (self.width - 16 - bar_w, y + 2, bar_w, bar_h))
        pygame.draw.rect(surface, self.CYAN, (self.width - 16 - bar_w, y + 2, int(bar_w * main_throttle), bar_h))
        y += 18

        # RCS Attitude Thruster (-100% to +100%)
        lbl_side = self.font_body.render("RCS ATTITUDE", True, (150, 165, 185))
        surface.blit(lbl_side, (x, y))
        bar_x = self.width - 16 - bar_w
        pygame.draw.rect(surface, (25, 35, 52), (bar_x, y + 2, bar_w, bar_h))
        center_bar_x = bar_x + bar_w // 2
        thruster_len = int((bar_w // 2) * side_gimbal)
        if thruster_len > 0:
            pygame.draw.rect(surface, self.YELLOW, (center_bar_x, y + 2, thruster_len, bar_h))
        else:
            pygame.draw.rect(surface, self.YELLOW, (center_bar_x + thruster_len, y + 2, abs(thruster_len), bar_h))
        pygame.draw.line(surface, self.TEXT_COLOR, (center_bar_x, y), (center_bar_x, y + bar_h + 3), 1)
        y += 24

        return y

    def _draw_sim_to_real_diagnostics(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        info: Dict[str, Any]
    ) -> int:
        hdr = self.font_header.render("SIM-TO-REAL TELEMETRY", True, self.TEXT_COLOR)
        surface.blit(hdr, (x, y))
        y += 18

        wind_pwr = info.get("wind_power", 0.0)
        thrust_eff = info.get("thrust_scale", 1.0) * 100.0
        curriculum = info.get("curriculum_level", 1.0) * 100.0

        items = [
            ("TURBULENT CROSSWIND", f"{wind_pwr:+5.1f} N", self.CYAN if abs(wind_pwr) < 4.0 else self.YELLOW),
            ("MOTOR EFFICIENCY", f"{thrust_eff:5.1f} %", self.TEXT_COLOR),
            ("DOMAIN RANDOMIZATION", f"{curriculum:5.0f} %", self.GREEN),
        ]

        for label, val, color in items:
            lbl_s = self.font_body.render(label, True, (150, 165, 185))
            val_s = self.font_data.render(val, True, color)
            surface.blit(lbl_s, (x, y))
            surface.blit(val_s, (self.width - 16 - val_s.get_width(), y))
            y += 16

        y += 8
        return y

    def _draw_score_card(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        total_reward: float,
        step_idx: int
    ):
        pygame.draw.line(surface, self.BORDER_COLOR, (x, y), (self.width - 16, y), 1)
        y += 10

        col = self.GREEN if total_reward >= 0 else self.RED
        r_text = self.font_score.render(f"SCORE: {total_reward:+6.1f}", True, col)
        surface.blit(r_text, (x, y))

        fps_text = self.font_sub.render(f"STEPS: {step_idx} | INFERENCE: 50Hz", True, (150, 165, 185))
        surface.blit(fps_text, (x, y + 28))
