"""
Photorealistic & Cinematic SpaceX-Style Rocket Renderer.
Replaces the default polygonal lander with a sleek Falcon 9 / Starship booster,
animated supersonic thrust plumes with Mach diamonds, RCS cold-gas thruster jets,
SpaceX autonomous drone ship landing pad, and twinkling starfield.
"""

import math
import random
import numpy as np
import pygame
from pygame import gfxdraw
from typing import List, Tuple, Dict, Any, Optional


class Particle:
    def __init__(self, x: float, y: float, vx: float, vy: float, ttl: float, color: Tuple[int, int, int], size: float):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.ttl = ttl
        self.max_ttl = ttl
        self.color = color
        self.size = size

    def update(self) -> bool:
        self.x += self.vx
        self.y += self.vy
        self.ttl -= 0.05
        return self.ttl > 0


class RocketRenderer:
    def __init__(self, viewport_w: int = 700, viewport_h: int = 700):
        self.w = viewport_w
        self.h = viewport_h
        
        # Scale factors from Box2D world (20m wide x 13.33m high) to viewport
        # LunarLander VIEWPORT_W=600, VIEWPORT_H=400, SCALE=30
        self.scale_x = self.w / 20.0
        self.scale_y = self.h / 14.0
        
        # Pre-generate stars for a deep space aesthetic
        random.seed(42)
        self.stars = [
            (random.randint(0, self.w), random.randint(0, int(self.h * 0.75)), random.choice([1, 2]), random.randint(150, 255))
            for _ in range(120)
        ]
        
        # Particle systems
        self.exhaust_particles: List[Particle] = []
        self.rcs_particles: List[Particle] = []
        self.frame_count = 0

    def world_to_screen(self, x_b2: float, y_b2: float) -> Tuple[int, int]:
        """Convert Box2D coordinates to screen pixel coordinates."""
        px = int(x_b2 * self.scale_x)
        py = int(self.h - (y_b2 * self.scale_y))
        return px, py

    def render_cinematic_world(
        self,
        env_unwrapped,
        action: np.ndarray,
        info: Dict[str, Any],
        trajectory_points: List[Tuple[int, int]]
    ) -> pygame.Surface:
        self.frame_count += 1
        surf = pygame.Surface((self.w, self.h))

        # 1. Deep Space Sky with Gradient and Stars
        self._draw_sky(surf)

        # 2. Lunar Terrain Surface
        self._draw_terrain(surf, env_unwrapped)

        # 3. SpaceX Drone Ship Landing Pad
        self._draw_drone_ship(surf, env_unwrapped)

        # 4. Smooth Trajectory Ribbon
        self._draw_trajectory(surf, trajectory_points)

        # 5. Particle Systems (Exhaust smoke / RCS gas)
        self._update_and_draw_particles(surf)

        # 6. Render SpaceX Falcon 9 / Starship Rocket
        self._draw_rocket_booster(surf, env_unwrapped, action, info)

        return surf

    def _draw_sky(self, surf: pygame.Surface):
        """Deep celestial sky with gradient and twinkling stars."""
        # Vertical gradient: deep cosmos black at top to dark slate twilight near horizon
        top_color = (6, 8, 14)
        bottom_color = (18, 26, 42)
        for y in range(0, self.h, 4):
            ratio = y / self.h
            r = int(top_color[0] + (bottom_color[0] - top_color[0]) * ratio)
            g = int(top_color[1] + (bottom_color[1] - top_color[1]) * ratio)
            b = int(top_color[2] + (bottom_color[2] - top_color[2]) * ratio)
            pygame.draw.rect(surf, (r, g, b), (0, y, self.w, 4))

        # Draw stars
        for sx, sy, s_rad, s_alpha in self.stars:
            twinkle = int(s_alpha * (0.8 + 0.2 * math.sin(self.frame_count * 0.05 + sx)))
            color = (twinkle, twinkle, min(255, twinkle + 20))
            if s_rad == 1:
                surf.set_at((sx, sy), color)
            else:
                pygame.draw.circle(surf, color, (sx, sy), s_rad)

    def _draw_terrain(self, surf: pygame.Surface, env_unwrapped):
        """Draws realistic lunar landscape with shaded terrain polygons."""
        if not hasattr(env_unwrapped, "moon") or env_unwrapped.moon is None:
            return

        moon_body = env_unwrapped.moon
        if not moon_body.fixtures:
            return

        surface_pts = []
        for f in moon_body.fixtures:
            trans = f.body.transform
            v0 = trans * f.shape.vertices[0]
            surface_pts.append(self.world_to_screen(v0.x, v0.y))
        
        last_f = moon_body.fixtures[-1]
        v_last = last_f.body.transform * last_f.shape.vertices[1]
        surface_pts.append(self.world_to_screen(v_last.x, v_last.y))

        if len(surface_pts) >= 2:
            # Closed terrain polygon down to bottom corners
            terrain_poly = surface_pts + [(self.w, self.h), (0, self.h)]
            # Lunar surface regolith gradient
            pygame.draw.polygon(surf, (215, 224, 235), terrain_poly)
            # Bright lunar crest rim line
            for i in range(len(surface_pts) - 1):
                pygame.draw.line(surf, (255, 255, 255), surface_pts[i], surface_pts[i + 1], width=2)

    def _draw_drone_ship(self, surf: pygame.Surface, env_unwrapped):
        """Draws a SpaceX Autonomous Spaceport Drone Ship (ASDS) landing barge."""
        if not hasattr(env_unwrapped, "helipad_x1"):
            return

        x1_b2 = env_unwrapped.helipad_x1
        x2_b2 = env_unwrapped.helipad_x2
        y_b2 = env_unwrapped.helipad_y

        p1 = self.world_to_screen(x1_b2, y_b2)
        p2 = self.world_to_screen(x2_b2, y_b2)

        deck_x = p1[0] - 12
        deck_y = p1[1] - 4
        deck_w = (p2[0] - p1[0]) + 24
        deck_h = 20

        # 1. Barge Hull (Dark Industrial Steel with Bevel)
        pygame.draw.rect(surf, (22, 28, 38), (deck_x, deck_y, deck_w, deck_h), border_radius=4)
        pygame.draw.rect(surf, (50, 65, 88), (deck_x, deck_y, deck_w, deck_h), width=2, border_radius=4)

        # 2. Yellow & Black Hazard Stripes on Outer Edges
        stripe_w = 20
        pygame.draw.rect(surf, (255, 204, 0), (deck_x, deck_y, stripe_w, deck_h))
        pygame.draw.rect(surf, (255, 204, 0), (deck_x + deck_w - stripe_w, deck_y, stripe_w, deck_h))
        for s in range(0, stripe_w, 6):
            pygame.draw.line(surf, (20, 24, 32), (deck_x + s, deck_y), (deck_x + s + 4, deck_y + deck_h), 2)
            pygame.draw.line(surf, (20, 24, 32), (deck_x + deck_w - stripe_w + s, deck_y), (deck_x + deck_w - stripe_w + s + 4, deck_y + deck_h), 2)

        # 3. Center Landing Bullseye & Target "X"
        center_x = (p1[0] + p2[0]) // 2
        center_y = deck_y + deck_h // 2
        r_x = int(deck_w * 0.20)
        r_y = int(deck_h * 0.38)
        
        # Perspective Ellipse for 3D Deck look
        pygame.draw.ellipse(surf, (255, 204, 0), (center_x - r_x, center_y - r_y, r_x * 2, r_y * 2), width=2)
        
        # Center Target "X"
        pygame.draw.line(surf, (255, 255, 255), (center_x - 7, center_y - 4), (center_x + 7, center_y + 4), width=2)
        pygame.draw.line(surf, (255, 255, 255), (center_x - 7, center_y + 4), (center_x + 7, center_y - 4), width=2)

        # 4. Telemetry Radio Towers / Antenna Beacons
        tower_h = 24
        # Left Tower
        pygame.draw.line(surf, (160, 175, 195), (deck_x + 4, deck_y), (deck_x + 4, deck_y - tower_h), width=2)
        pygame.draw.circle(surf, (255, 45, 45), (deck_x + 4, deck_y - tower_h), 3)  # Red beacon
        # Right Tower
        pygame.draw.line(surf, (160, 175, 195), (deck_x + deck_w - 4, deck_y), (deck_x + deck_w - 4, deck_y - tower_h), width=2)
        pygame.draw.circle(surf, (45, 255, 100), (deck_x + deck_w - 4, deck_y - tower_h), 3)  # Green beacon

        # 5. Flame Ground Reflection Glow
        if hasattr(env_unwrapped, "lander") and env_unwrapped.lander is not None:
            l_pos = env_unwrapped.lander.position
            l_px, l_py = self.world_to_screen(l_pos.x, l_pos.y)
            dist_to_deck = deck_y - l_py
            if 0 < dist_to_deck < 180:
                glow_alpha = int(max(0, 120 * (1.0 - dist_to_deck / 180.0)))
                glow_radius = int(50 + (180 - dist_to_deck) * 0.4)
                glow_surf = pygame.Surface((glow_radius * 2, glow_radius), pygame.SRCALPHA)
                pygame.draw.ellipse(glow_surf, (255, 140, 20, glow_alpha), (0, 0, glow_radius * 2, glow_radius))
                surf.blit(glow_surf, (center_x - glow_radius, deck_y - glow_radius // 2))

    def _draw_trajectory(self, surf: pygame.Surface, trajectory_points: List[Tuple[int, int]]):
        """Draws smooth anti-aliased fading flight trajectory ribbon."""
        if len(trajectory_points) < 2:
            return

        for i in range(len(trajectory_points) - 1):
            alpha_ratio = i / float(len(trajectory_points))
            # Smooth neon cyan glow
            r = int(0 * alpha_ratio)
            g = int(220 * alpha_ratio)
            b = int(255 * alpha_ratio)
            p1 = trajectory_points[i]
            p2 = trajectory_points[i + 1]
            pygame.draw.line(surf, (r, g, b), p1, p2, width=2)

    def _update_and_draw_particles(self, surf: pygame.Surface):
        """Update and draw exhaust and RCS gas particle mist."""
        # Exhaust Particles
        surviving_exhaust = []
        for p in self.exhaust_particles:
            if p.update():
                surviving_exhaust.append(p)
                ratio = p.ttl / p.max_ttl
                curr_size = max(1.0, p.size * (1.0 + (1.0 - ratio) * 1.5))
                c = (int(p.color[0] * ratio), int(p.color[1] * ratio * 0.8), int(p.color[2] * ratio * 0.5))
                pygame.draw.circle(surf, c, (int(p.x), int(p.y)), int(curr_size))
        self.exhaust_particles = surviving_exhaust

        # RCS Particles
        surviving_rcs = []
        for p in self.rcs_particles:
            if p.update():
                surviving_rcs.append(p)
                ratio = p.ttl / p.max_ttl
                c = (int(200 * ratio), int(240 * ratio), int(255 * ratio))
                pygame.draw.circle(surf, c, (int(p.x), int(p.y)), max(1, int(p.size * ratio)))
        self.rcs_particles = surviving_rcs

    def _draw_rocket_booster(
        self,
        surf: pygame.Surface,
        env_unwrapped,
        action: np.ndarray,
        info: Dict[str, Any]
    ):
        """
        Draws the SpaceX Falcon 9 / Starship rocket with:
        - Sleek cylindrical booster fuselage
        - Black aerodynamic interstage & payload aerocap
        - Titanium grid fins rotating with RCS control
        - Carbon fiber deployable landing legs
        - Supersonic Mach diamond rocket exhaust flame
        """
        if not hasattr(env_unwrapped, "lander") or env_unwrapped.lander is None:
            return

        lander = env_unwrapped.lander
        pos = lander.position
        angle = lander.angle  # In radians, counter-clockwise

        center_px, center_py = self.world_to_screen(pos.x, pos.y)

        # Sleek aerospace proportions: tall booster (66px) x slender core (14px)
        rocket_h = 66
        rocket_w = 14
        half_w = rocket_w / 2.0
        half_h = rocket_h / 2.0

        # Cosine and sine for rotation
        cos_a = math.cos(-angle)
        sin_a = math.sin(-angle)

        def rotate_pt(lx: float, ly: float) -> Tuple[int, int]:
            rx = lx * cos_a - ly * sin_a
            ry = lx * sin_a + ly * cos_a
            return int(center_px + rx), int(center_py + ry)

        # -------------------------------------------------------------
        # A. MAIN RETRO ENGINE PLUME (FALCON 9 SUPERSONIC HOVERSLAM FLAME)
        # -------------------------------------------------------------
        main_throttle = np.clip((action[0] + 1.0) / 2.0 if action[0] > 0 else 0.0, 0.0, 1.0)
        if main_throttle > 0:
            flame_len = 35 + int(main_throttle * 45) + random.randint(-3, 3)
            flame_w = 12 + int(main_throttle * 6)

            # Plume base starts at bottom nozzle
            nozzle_pt = rotate_pt(0, half_h + 3)
            flame_tip = rotate_pt(0, half_h + flame_len)
            flame_left = rotate_pt(-flame_w / 2, half_h + 6)
            flame_right = rotate_pt(flame_w / 2, half_h + 6)

            # 1. Outer Deep Orange/Red Flame
            outer_flame_pts = [nozzle_pt, flame_left, flame_tip, flame_right]
            pygame.draw.polygon(surf, (255, 69, 0), outer_flame_pts)

            # 2. Middle Yellow Core
            mid_len = int(flame_len * 0.7)
            mid_w = int(flame_w * 0.6)
            mid_tip = rotate_pt(0, half_h + mid_len)
            mid_left = rotate_pt(-mid_w / 2, half_h + 4)
            mid_right = rotate_pt(mid_w / 2, half_h + 4)
            pygame.draw.polygon(surf, (255, 215, 0), [nozzle_pt, mid_left, mid_tip, mid_right])

            # 3. Inner White/Cyan Mach Diamond Core
            core_len = int(flame_len * 0.4)
            core_w = int(flame_w * 0.3)
            core_tip = rotate_pt(0, half_h + core_len)
            core_left = rotate_pt(-core_w / 2, half_h + 3)
            core_right = rotate_pt(core_w / 2, half_h + 3)
            pygame.draw.polygon(surf, (255, 255, 255), [nozzle_pt, core_left, core_tip, core_right])

            # 4. Mach Shock Diamonds (Diamond shapes inside the flame)
            for d_idx in range(1, 3):
                d_dist = half_h + (flame_len * 0.28 * d_idx)
                d_pt = rotate_pt(0, d_dist)
                d_sz = 3
                d_pts = [
                    rotate_pt(0, d_dist - d_sz),
                    rotate_pt(d_sz, d_dist),
                    rotate_pt(0, d_dist + d_sz),
                    rotate_pt(-d_sz, d_dist)
                ]
                pygame.draw.polygon(surf, (230, 245, 255), d_pts)

            # Spawn exhaust smoke particles
            if len(self.exhaust_particles) < 80:
                p_vx = -sin_a * (3.0 + random.uniform(0, 2)) + random.uniform(-0.5, 0.5)
                p_vy = cos_a * (3.0 + random.uniform(0, 2)) + random.uniform(-0.5, 0.5)
                self.exhaust_particles.append(
                    Particle(
                        x=flame_tip[0],
                        y=flame_tip[1],
                        vx=p_vx,
                        vy=p_vy,
                        ttl=random.uniform(0.6, 1.2),
                        color=(255, 120, 20),
                        size=random.uniform(4.0, 7.0)
                    )
                )

        # -------------------------------------------------------------
        # B. COLD-GAS RCS ATTITUDE THRUSTERS
        # -------------------------------------------------------------
        side_throttle = float(action[1])
        rcs_y = -half_h + 12  # Located near top of booster

        # Fire Right Thruster (pushes rocket left)
        if side_throttle < -0.3:
            burst_len = int(abs(side_throttle) * 20)
            rcs_origin = rotate_pt(half_w, rcs_y)
            rcs_tip = rotate_pt(half_w + burst_len, rcs_y)
            pygame.draw.line(surf, (200, 240, 255), rcs_origin, rcs_tip, width=3)
            if len(self.rcs_particles) < 40:
                self.rcs_particles.append(
                    Particle(rcs_tip[0], rcs_tip[1], cos_a * 2.0, sin_a * 2.0, 0.4, (220, 245, 255), 2.5)
                )

        # Fire Left Thruster (pushes rocket right)
        elif side_throttle > 0.3:
            burst_len = int(side_throttle * 20)
            rcs_origin = rotate_pt(-half_w, rcs_y)
            rcs_tip = rotate_pt(-half_w - burst_len, rcs_y)
            pygame.draw.line(surf, (200, 240, 255), rcs_origin, rcs_tip, width=3)
            if len(self.rcs_particles) < 40:
                self.rcs_particles.append(
                    Particle(rcs_tip[0], rcs_tip[1], -cos_a * 2.0, -sin_a * 2.0, 0.4, (220, 245, 255), 2.5)
                )

        # -------------------------------------------------------------
        # C. ROCKET BOOSTER FUSELAGE (FALCON 9 WHITE & BLACK LIVERY)
        # -------------------------------------------------------------
        # Main cylindrical fuselage
        p_tl = rotate_pt(-half_w, -half_h + 8)
        p_tr = rotate_pt(half_w, -half_h + 8)
        p_br = rotate_pt(half_w, half_h)
        p_bl = rotate_pt(-half_w, half_h)

        body_poly = [p_tl, p_tr, p_br, p_bl]
        # Sleek Falcon pure white fuselage
        pygame.draw.polygon(surf, (245, 248, 252), body_poly)
        gfxdraw.aapolygon(surf, body_poly, (200, 210, 225))

        # Carbon fiber interstage band (Black band near upper section)
        p_band_tl = rotate_pt(-half_w, -half_h + 14)
        p_band_tr = rotate_pt(half_w, -half_h + 14)
        p_band_br = rotate_pt(half_w, -half_h + 20)
        p_band_bl = rotate_pt(-half_w, -half_h + 20)
        pygame.draw.polygon(surf, (35, 40, 48), [p_band_tl, p_band_tr, p_band_br, p_band_bl])

        # Conical Fairing / Aerocap (Top Nose)
        nose_tip = rotate_pt(0, -half_h - 6)
        nose_l = rotate_pt(-half_w + 1, -half_h + 8)
        nose_r = rotate_pt(half_w - 1, -half_h + 8)
        nose_poly = [nose_tip, nose_r, nose_l]
        pygame.draw.polygon(surf, (45, 52, 65), nose_poly)
        gfxdraw.aapolygon(surf, nose_poly, (70, 80, 100))

        # Engine Nozzle (Bell housing at base)
        nozzle_l = rotate_pt(-5, half_h)
        nozzle_r = rotate_pt(5, half_h)
        nozzle_bl = rotate_pt(-4, half_h + 5)
        nozzle_br = rotate_pt(4, half_h + 5)
        pygame.draw.polygon(surf, (30, 32, 38), [nozzle_l, nozzle_r, nozzle_br, nozzle_bl])

        # Longitudinal Booster Shading (Gives 3D cylindrical specular highlight)
        line_top = rotate_pt(-half_w + 3, -half_h + 8)
        line_bottom = rotate_pt(-half_w + 3, half_h)
        pygame.draw.line(surf, (255, 255, 255), line_top, line_bottom, width=2)

        # -------------------------------------------------------------
        # D. TITANIUM GRID FINS (DEPLOYED & ACTIVELY STEERING)
        # -------------------------------------------------------------
        # Grid fins rotate slightly with RCS action for active steering appearance
        fin_steer = float(action[1]) * 0.2
        fin_y = -half_h + 10
        
        # Left Grid Fin
        fin_l_base = rotate_pt(-half_w, fin_y)
        fin_l_tip1 = rotate_pt(-half_w - 9, fin_y - 2 + int(fin_steer * 8))
        fin_l_tip2 = rotate_pt(-half_w - 9, fin_y + 4 + int(fin_steer * 8))
        fin_l_base2 = rotate_pt(-half_w, fin_y + 4)
        pygame.draw.polygon(surf, (25, 30, 38), [fin_l_base, fin_l_tip1, fin_l_tip2, fin_l_base2])
        pygame.draw.polygon(surf, (80, 95, 120), [fin_l_base, fin_l_tip1, fin_l_tip2, fin_l_base2], width=1)

        # Right Grid Fin
        fin_r_base = rotate_pt(half_w, fin_y)
        fin_r_tip1 = rotate_pt(half_w + 9, fin_y - 2 - int(fin_steer * 8))
        fin_r_tip2 = rotate_pt(half_w + 9, fin_y + 4 - int(fin_steer * 8))
        fin_r_base2 = rotate_pt(half_w, fin_y + 4)
        pygame.draw.polygon(surf, (25, 30, 38), [fin_r_base, fin_r_tip1, fin_r_tip2, fin_r_base2])
        pygame.draw.polygon(surf, (80, 95, 120), [fin_r_base, fin_r_tip1, fin_r_tip2, fin_r_base2], width=1)

        # -------------------------------------------------------------
        # E. DEPLOYABLE CARBON FIBER LANDING LEGS
        # -------------------------------------------------------------
        # Legs connect near base (half_h - 4) and extend diagonally outward with footpads
        leg_base_y = half_h - 4
        leg_foot_y = half_h + 10
        leg_span = 22

        # Left Leg Assembly
        leg_l_hinge = rotate_pt(-half_w + 2, leg_base_y)
        leg_l_foot = rotate_pt(-half_w - leg_span, leg_foot_y)
        leg_l_strut = rotate_pt(-half_w + 2, leg_base_y + 8)
        # Main carbon leg strut
        pygame.draw.line(surf, (30, 35, 45), leg_l_hinge, leg_l_foot, width=3)
        # Hydraulic piston brace
        pygame.draw.line(surf, (150, 160, 175), leg_l_strut, leg_l_foot, width=2)
        # Foot pad
        foot_l_pts = [
            (leg_l_foot[0] - 4, leg_l_foot[1]),
            (leg_l_foot[0] + 4, leg_l_foot[1])
        ]
        pygame.draw.line(surf, (20, 20, 20), foot_l_pts[0], foot_l_pts[1], width=3)

        # Right Leg Assembly
        leg_r_hinge = rotate_pt(half_w - 2, leg_base_y)
        leg_r_foot = rotate_pt(half_w + leg_span, leg_foot_y)
        leg_r_strut = rotate_pt(half_w - 2, leg_base_y + 8)
        # Main carbon leg strut
        pygame.draw.line(surf, (30, 35, 45), leg_r_hinge, leg_r_foot, width=3)
        # Hydraulic piston brace
        pygame.draw.line(surf, (150, 160, 175), leg_r_strut, leg_r_foot, width=2)
        # Foot pad
        foot_r_pts = [
            (leg_r_foot[0] - 4, leg_r_foot[1]),
            (leg_r_foot[0] + 4, leg_r_foot[1])
        ]
        pygame.draw.line(surf, (20, 20, 20), foot_r_pts[0], foot_r_pts[1], width=3)

        # Navigation Strobe (flashes green/red)
        if (self.frame_count // 10) % 2 == 0:
            strobe_pt = rotate_pt(0, -half_h + 2)
            pygame.draw.circle(surf, (0, 255, 128), strobe_pt, 2)
