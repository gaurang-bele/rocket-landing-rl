"""
Cinematic SpaceX Falcon 9 Rocket & Drone Ship Renderer.
Features:
- Sleek Falcon 9 First Stage Booster with USA flag decal, SpaceX livery, and grid fins.
- Supersonic retro-propulsion flame with Mach shock diamonds and volumetric bloom.
- Dynamic cold-gas RCS attitude bursts.
- SpaceX Autonomous Spaceport Drone Ship (ASDS - "A Shortfall of Gravitas") with ocean ripples,
  diagonal caution stripes, perspective bullseye, and beacon radar towers.
- Ground blast dust billows and flame reflection glow upon landing approach.
- Twinkling cosmic starfield and atmospheric twilight horizon.
"""

import math
import random
import numpy as np
import pygame
from pygame import gfxdraw
from typing import List, Tuple, Dict, Any, Optional


class Particle:
    def __init__(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        ttl: float,
        color: Tuple[int, int, int],
        size: float,
        alpha: int = 255,
        fade_rate: float = 0.04
    ):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.ttl = ttl
        self.max_ttl = ttl
        self.color = color
        self.size = size
        self.alpha = alpha
        self.fade_rate = fade_rate

    def update(self) -> bool:
        self.x += self.vx
        self.y += self.vy
        self.ttl -= self.fade_rate
        return self.ttl > 0


class RocketRenderer:
    def __init__(self, viewport_w: int = 840, viewport_h: int = 720):
        self.w = viewport_w
        self.h = viewport_h
        
        # Scaling Box2D world (20m x 14m) to viewport
        self.scale_x = self.w / 20.0
        self.scale_y = self.h / 14.0
        
        # Starfield (multi-spectral stars: white, pale blue, warm amber)
        random.seed(1337)
        self.stars = []
        for _ in range(160):
            sx = random.randint(0, self.w)
            sy = random.randint(0, int(self.h * 0.72))
            s_rad = random.choice([1, 1, 2])
            s_type = random.random()
            if s_type < 0.6:
                base_color = (235, 245, 255)  # White-blue
            elif s_type < 0.85:
                base_color = (180, 215, 255)  # Deep cyan
            else:
                base_color = (255, 225, 180)  # Amber
            self.stars.append((sx, sy, s_rad, base_color, random.uniform(0.5, 1.0)))

        # Particle Systems
        self.exhaust_particles: List[Particle] = []
        self.rcs_particles: List[Particle] = []
        self.dust_particles: List[Particle] = []
        self.confetti_particles: List[Particle] = []
        
        self.frame_count = 0
        self.touchdown_celebrated = False

    def world_to_screen(self, x_b2: float, y_b2: float) -> Tuple[int, int]:
        """Convert Box2D world meters to screen pixels."""
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

        # 1. Deep Space Cosmic Sky & Atmospheric Horizon
        self._draw_sky_and_atmosphere(surf)

        # 2. Ocean Surface / Coastline Terrain
        self._draw_terrain_and_ocean(surf, env_unwrapped)

        # 3. SpaceX Drone Ship Landing Barge ("A Shortfall of Gravitas")
        self._draw_drone_ship(surf, env_unwrapped)

        # 4. Smooth Trajectory Ribbon
        self._draw_trajectory(surf, trajectory_points)

        # 5. Particle Systems (Exhaust smoke, ground dust, RCS bursts)
        self._update_and_draw_particles(surf)

        # 6. Render SpaceX Falcon 9 Booster
        self._draw_falcon9_booster(surf, env_unwrapped, action, info)

        # 7. Touchdown Celebration Banner (if landing confirmed)
        if info.get("is_safe_landing", 0) > 0:
            self._trigger_touchdown_fx(surf, env_unwrapped)

        return surf

    def _draw_sky_and_atmosphere(self, surf: pygame.Surface):
        """Cosmic twilight sky with atmospheric gradient and twinkling stars."""
        # Sky gradient: dark void at top down to twilight indigo
        top_c = (4, 6, 12)
        mid_c = (14, 20, 36)
        bot_c = (22, 34, 58)
        
        for y in range(0, self.h, 4):
            ratio = y / float(self.h)
            if ratio < 0.6:
                t = ratio / 0.6
                r = int(top_c[0] + (mid_c[0] - top_c[0]) * t)
                g = int(top_c[1] + (mid_c[1] - top_c[1]) * t)
                b = int(top_c[2] + (mid_c[2] - top_c[2]) * t)
            else:
                t = (ratio - 0.6) / 0.4
                r = int(mid_c[0] + (bot_c[0] - mid_c[0]) * t)
                g = int(mid_c[1] + (bot_c[1] - mid_c[1]) * t)
                b = int(mid_c[2] + (bot_c[2] - mid_c[2]) * t)
            pygame.draw.rect(surf, (r, g, b), (0, y, self.w, 4))

        # Twinkling Stars
        for sx, sy, s_rad, s_color, s_speed in self.stars:
            twinkle = 0.75 + 0.25 * math.sin(self.frame_count * 0.06 * s_speed + sx)
            c = (
                int(s_color[0] * twinkle),
                int(s_color[1] * twinkle),
                int(s_color[2] * twinkle)
            )
            if s_rad == 1:
                surf.set_at((sx, sy), c)
            else:
                pygame.draw.circle(surf, c, (sx, sy), s_rad)

    def _draw_terrain_and_ocean(self, surf: pygame.Surface, env_unwrapped):
        """Draws realistic oceanic coastline terrain beneath the sky."""
        if not hasattr(env_unwrapped, "moon") or env_unwrapped.moon is None or not env_unwrapped.moon.fixtures:
            return

        moon_body = env_unwrapped.moon
        surface_pts = []
        for f in moon_body.fixtures:
            v0 = f.body.transform * f.shape.vertices[0]
            surface_pts.append(self.world_to_screen(v0.x, v0.y))
        
        last_f = moon_body.fixtures[-1]
        v_last = last_f.body.transform * last_f.shape.vertices[1]
        surface_pts.append(self.world_to_screen(v_last.x, v_last.y))

        if len(surface_pts) >= 2:
            # Solid oceanic bedrock / lunar regolith
            terrain_poly = surface_pts + [(self.w, self.h), (0, self.h)]
            pygame.draw.polygon(surf, (20, 28, 42), terrain_poly)
            
            # Subtle ocean water gradient layer
            water_surf = pygame.Surface((self.w, 40), pygame.SRCALPHA)
            for y_off in range(0, 40, 2):
                alpha = int(90 * (1.0 - y_off / 40.0))
                pygame.draw.line(water_surf, (0, 180, 230, alpha), (0, y_off), (self.w, y_off), 2)
            surf.blit(water_surf, (0, surface_pts[len(surface_pts)//2][1] - 8))

            # Crest highlight rim
            for i in range(len(surface_pts) - 1):
                pygame.draw.line(surf, (70, 95, 130), surface_pts[i], surface_pts[i + 1], width=2)

    def _draw_drone_ship(self, surf: pygame.Surface, env_unwrapped):
        """Draws SpaceX Autonomous Spaceport Drone Ship (ASDS - 'A Shortfall of Gravitas')."""
        if not hasattr(env_unwrapped, "helipad_x1"):
            return

        p1 = self.world_to_screen(env_unwrapped.helipad_x1, env_unwrapped.helipad_y)
        p2 = self.world_to_screen(env_unwrapped.helipad_x2, env_unwrapped.helipad_y)

        deck_x = p1[0] - 16
        deck_y = p1[1] - 6
        deck_w = (p2[0] - p1[0]) + 32
        deck_h = 24

        # 1. Water Wake / Ocean Foam around the Barge
        for wave_i in range(3):
            wave_y = deck_y + deck_h + 3 + wave_i * 4
            wave_shift = math.sin(self.frame_count * 0.08 + wave_i) * 6
            pygame.draw.line(surf, (40, 90, 130), (deck_x - 12 + wave_shift, wave_y), (deck_x + deck_w + 12 + wave_shift, wave_y), 2)

        # 2. Barge Hull (Heavy Steel Structure)
        pygame.draw.rect(surf, (18, 22, 30), (deck_x, deck_y, deck_w, deck_h), border_radius=4)
        pygame.draw.rect(surf, (45, 60, 85), (deck_x, deck_y, deck_w, deck_h), width=2, border_radius=4)

        # 3. Yellow/Black Industrial Caution Hazard Stripes
        stripe_w = 24
        # Left stripes
        pygame.draw.rect(surf, (255, 204, 0), (deck_x, deck_y, stripe_w, deck_h))
        for s in range(0, stripe_w, 6):
            pygame.draw.line(surf, (20, 24, 30), (deck_x + s, deck_y), (deck_x + s + 5, deck_y + deck_h), 2)
        # Right stripes
        pygame.draw.rect(surf, (255, 204, 0), (deck_x + deck_w - stripe_w, deck_y, stripe_w, deck_h))
        for s in range(0, stripe_w, 6):
            pygame.draw.line(surf, (20, 24, 30), (deck_x + deck_w - stripe_w + s, deck_y), (deck_x + deck_w - stripe_w + s + 5, deck_y + deck_h), 2)

        # 4. Landing Target Bullseye & Center "X"
        center_x = (p1[0] + p2[0]) // 2
        center_y = deck_y + deck_h // 2
        r_x = int(deck_w * 0.22)
        r_y = int(deck_h * 0.40)

        # Outer Landing Ring (Perspective 3D Ellipse)
        pygame.draw.ellipse(surf, (255, 204, 0), (center_x - r_x, center_y - r_y, r_x * 2, r_y * 2), width=2)
        # Inner Circle
        pygame.draw.ellipse(surf, (255, 220, 80), (center_x - r_x // 2, center_y - r_y // 2, r_x, r_y), width=1)

        # Center Landing "X"
        x_sz_x = 9
        x_sz_y = 5
        pygame.draw.line(surf, (255, 255, 255), (center_x - x_sz_x, center_y - x_sz_y), (center_x + x_sz_x, center_y + x_sz_y), width=2)
        pygame.draw.line(surf, (255, 255, 255), (center_x - x_sz_x, center_y + x_sz_y), (center_x + x_sz_x, center_y - x_sz_y), width=2)

        # 5. Telemetry Radio Beacon Masts
        mast_h = 28
        # Port mast (Red Beacon)
        pygame.draw.line(surf, (140, 160, 180), (deck_x + 5, deck_y), (deck_x + 5, deck_y - mast_h), width=2)
        pygame.draw.circle(surf, (255, 40, 40), (deck_x + 5, deck_y - mast_h), 3)
        # Starboard mast (Green Beacon)
        pygame.draw.line(surf, (140, 160, 180), (deck_x + deck_w - 5, deck_y), (deck_x + deck_w - 5, deck_y - mast_h), width=2)
        pygame.draw.circle(surf, (40, 255, 100), (deck_x + deck_w - 5, deck_y - mast_h), 3)

        # 6. Deck Stencil Label: "ASDS: A SHORTFALL OF GRAVITAS"
        font_stencil = pygame.font.SysFont("Consolas", 10, bold=True)
        lbl = font_stencil.render("A SHORTFALL OF GRAVITAS", True, (120, 140, 165))
        surf.blit(lbl, (center_x - lbl.get_width() // 2, deck_y + deck_h - 11))

        # 7. Dynamic Ground Flame Reflection Glow
        if hasattr(env_unwrapped, "lander") and env_unwrapped.lander is not None:
            l_pos = env_unwrapped.lander.position
            l_px, l_py = self.world_to_screen(l_pos.x, l_pos.y)
            dist_to_deck = deck_y - l_py
            if 0 < dist_to_deck < 220:
                glow_intensity = max(0.0, 1.0 - (dist_to_deck / 220.0))
                glow_rad_x = int(60 + glow_intensity * 70)
                glow_rad_y = int(15 + glow_intensity * 25)
                glow_alpha = int(glow_intensity * 140)

                glow_surf = pygame.Surface((glow_rad_x * 2, glow_rad_y * 2), pygame.SRCALPHA)
                pygame.draw.ellipse(
                    glow_surf,
                    (255, 120, 20, glow_alpha),
                    (0, 0, glow_rad_x * 2, glow_rad_y * 2)
                )
                surf.blit(glow_surf, (center_x - glow_rad_x, deck_y - glow_rad_y // 2))

    def _draw_trajectory(self, surf: pygame.Surface, trajectory_points: List[Tuple[int, int]]):
        """Draws sleek anti-aliased neon cyan trajectory ribbon."""
        if len(trajectory_points) < 2:
            return

        for i in range(len(trajectory_points) - 1):
            ratio = i / float(len(trajectory_points))
            # Neon cyan trail fading into deep space
            r = int(0 * ratio)
            g = int(220 * ratio)
            b = int(255 * ratio)
            p1 = trajectory_points[i]
            p2 = trajectory_points[i + 1]
            pygame.draw.line(surf, (r, g, b), p1, p2, width=2)

    def _update_and_draw_particles(self, surf: pygame.Surface):
        """Update and draw exhaust, RCS gas, dust billows, and celebration confetti."""
        # Exhaust Smoke Particles
        surv_ex = []
        for p in self.exhaust_particles:
            if p.update():
                surv_ex.append(p)
                ratio = p.ttl / p.max_ttl
                sz = max(1.0, p.size * (1.0 + (1.0 - ratio) * 1.8))
                col = (int(p.color[0] * ratio), int(p.color[1] * ratio * 0.7), int(p.color[2] * ratio * 0.4))
                pygame.draw.circle(surf, col, (int(p.x), int(p.y)), int(sz))
        self.exhaust_particles = surv_ex

        # RCS Gas Mist
        surv_rcs = []
        for p in self.rcs_particles:
            if p.update():
                surv_rcs.append(p)
                ratio = p.ttl / p.max_ttl
                col = (int(200 * ratio), int(240 * ratio), int(255 * ratio))
                pygame.draw.circle(surf, col, (int(p.x), int(p.y)), max(1, int(p.size * ratio)))
        self.rcs_particles = surv_rcs

        # Ground Dust Clouds
        surv_dust = []
        for p in self.dust_particles:
            if p.update():
                surv_dust.append(p)
                ratio = p.ttl / p.max_ttl
                sz = max(2.0, p.size * (1.0 + (1.0 - ratio) * 2.2))
                alpha = int(140 * ratio)
                dust_surf = pygame.Surface((int(sz * 2), int(sz * 2)), pygame.SRCALPHA)
                pygame.draw.circle(dust_surf, (180, 195, 215, alpha), (int(sz), int(sz)), int(sz))
                surf.blit(dust_surf, (int(p.x - sz), int(p.y - sz)))
        self.dust_particles = surv_dust

        # Confetti Celebration
        surv_conf = []
        for p in self.confetti_particles:
            if p.update():
                surv_conf.append(p)
                pygame.draw.circle(surf, p.color, (int(p.x), int(p.y)), int(p.size))
        self.confetti_particles = surv_conf

    def _draw_falcon9_booster(
        self,
        surf: pygame.Surface,
        env_unwrapped,
        action: np.ndarray,
        info: Dict[str, Any]
    ):
        """
        Renders the SpaceX Falcon 9 First Stage Booster:
        - Slender aerospace proportions (74px x 15px)
        - Pure white cylindrical fuselage with 3D specular highlight
        - Black interstage band and dark aerodynamic nosecone
        - Active titanium grid fins
        - Deployed carbon-fiber landing legs with suspension struts
        - Multi-stage supersonic flame plume with Mach shock diamonds
        - Ground dust cloud generator
        """
        if not hasattr(env_unwrapped, "lander") or env_unwrapped.lander is None:
            return

        lander = env_unwrapped.lander
        pos = lander.position
        angle = lander.angle  # Radians, counter-clockwise

        center_px, center_py = self.world_to_screen(pos.x, pos.y)

        # Proportions: 74px tall x 15px wide (authentic booster silhouette)
        rocket_h = 74
        rocket_w = 15
        half_w = rocket_w / 2.0
        half_h = rocket_h / 2.0

        cos_a = math.cos(-angle)
        sin_a = math.sin(-angle)

        def rotate_pt(lx: float, ly: float) -> Tuple[int, int]:
            rx = lx * cos_a - ly * sin_a
            ry = lx * sin_a + ly * cos_a
            return int(center_px + rx), int(center_py + ry)

        # -----------------------------------------------------------------
        # A. SUPERSONIC RETRO-PROPULSION FLAME (MACH SHOCK DIAMONDS)
        # -----------------------------------------------------------------
        main_throttle = np.clip((action[0] + 1.0) / 2.0 if action[0] > 0 else 0.0, 0.0, 1.0)
        if main_throttle > 0:
            flame_len = 45 + int(main_throttle * 55) + random.randint(-4, 4)
            flame_w = 14 + int(main_throttle * 8)

            nozzle_pt = rotate_pt(0, half_h + 3)
            flame_tip = rotate_pt(0, half_h + flame_len)
            flame_l = rotate_pt(-flame_w / 2, half_h + 8)
            flame_r = rotate_pt(flame_w / 2, half_h + 8)

            # Volumetric Flame Bloom (Glowing Radial Halo)
            halo_rad = int(flame_len * 0.7)
            halo_surf = pygame.Surface((halo_rad * 2, halo_rad * 2), pygame.SRCALPHA)
            pygame.draw.circle(halo_surf, (255, 120, 20, 45), (halo_rad, halo_rad), halo_rad)
            surf.blit(halo_surf, (nozzle_pt[0] - halo_rad, nozzle_pt[1] - halo_rad // 2))

            # Layer 1: Outer Turbulent Orange/Red Plume
            pygame.draw.polygon(surf, (255, 60, 0), [nozzle_pt, flame_l, flame_tip, flame_r])

            # Layer 2: Mid Supersonic Yellow Core
            mid_len = int(flame_len * 0.65)
            mid_w = int(flame_w * 0.6)
            mid_tip = rotate_pt(0, half_h + mid_len)
            mid_l = rotate_pt(-mid_w / 2, half_h + 5)
            mid_r = rotate_pt(mid_w / 2, half_h + 5)
            pygame.draw.polygon(surf, (255, 215, 0), [nozzle_pt, mid_l, mid_tip, mid_r])

            # Layer 3: Inner White-Hot Combustion Throat
            core_len = int(flame_len * 0.35)
            core_w = int(flame_w * 0.32)
            core_tip = rotate_pt(0, half_h + core_len)
            core_l = rotate_pt(-core_w / 2, half_h + 3)
            core_r = rotate_pt(core_w / 2, half_h + 3)
            pygame.draw.polygon(surf, (255, 255, 255), [nozzle_pt, core_l, core_tip, core_r])

            # Layer 4: Mach Shock Diamonds (Diamond diamonds along supersonic axis)
            for d_idx in range(1, 3):
                d_dist = half_h + (flame_len * 0.28 * d_idx)
                d_sz = 3
                d_pts = [
                    rotate_pt(0, d_dist - d_sz),
                    rotate_pt(d_sz, d_dist),
                    rotate_pt(0, d_dist + d_sz),
                    rotate_pt(-d_sz, d_dist)
                ]
                pygame.draw.polygon(surf, (220, 240, 255), d_pts)

            # Spawn Exhaust Smoke Particles
            if len(self.exhaust_particles) < 90:
                p_vx = -sin_a * (3.5 + random.uniform(0, 2)) + random.uniform(-0.6, 0.6)
                p_vy = cos_a * (3.5 + random.uniform(0, 2)) + random.uniform(-0.6, 0.6)
                self.exhaust_particles.append(
                    Particle(
                        x=flame_tip[0],
                        y=flame_tip[1],
                        vx=p_vx,
                        vy=p_vy,
                        ttl=random.uniform(0.7, 1.3),
                        color=(255, 140, 30),
                        size=random.uniform(5.0, 9.0)
                    )
                )

            # Ground Dust Billows (Kick up when approaching the deck < 70m)
            alt_agl = pos.y - 3.33  # Approx deck altitude
            if 0 < alt_agl < 3.0 and len(self.dust_particles) < 60:
                # Left billow
                self.dust_particles.append(
                    Particle(
                        x=flame_tip[0],
                        y=flame_tip[1],
                        vx=-random.uniform(2.5, 5.0),
                        vy=-random.uniform(0.2, 1.0),
                        ttl=random.uniform(0.6, 1.2),
                        color=(190, 205, 225),
                        size=random.uniform(6.0, 12.0)
                    )
                )
                # Right billow
                self.dust_particles.append(
                    Particle(
                        x=flame_tip[0],
                        y=flame_tip[1],
                        vx=random.uniform(2.5, 5.0),
                        vy=-random.uniform(0.2, 1.0),
                        ttl=random.uniform(0.6, 1.2),
                        color=(190, 205, 225),
                        size=random.uniform(6.0, 12.0)
                    )
                )

        # -----------------------------------------------------------------
        # B. COLD-GAS RCS ATTITUDE THRUSTERS
        # -----------------------------------------------------------------
        side_throttle = float(action[1])
        rcs_y = -half_h + 14

        # Right Thruster Blast (Pushes rocket left)
        if side_throttle < -0.25:
            burst_len = int(abs(side_throttle) * 24)
            rcs_orig = rotate_pt(half_w, rcs_y)
            rcs_tip = rotate_pt(half_w + burst_len, rcs_y)
            pygame.draw.line(surf, (220, 245, 255), rcs_orig, rcs_tip, width=3)
            if len(self.rcs_particles) < 50:
                self.rcs_particles.append(
                    Particle(rcs_tip[0], rcs_tip[1], cos_a * 2.5, sin_a * 2.5, 0.4, (220, 245, 255), 3.0)
                )

        # Left Thruster Blast (Pushes rocket right)
        elif side_throttle > 0.25:
            burst_len = int(side_throttle * 24)
            rcs_orig = rotate_pt(-half_w, rcs_y)
            rcs_tip = rotate_pt(-half_w - burst_len, rcs_y)
            pygame.draw.line(surf, (220, 245, 255), rcs_orig, rcs_tip, width=3)
            if len(self.rcs_particles) < 50:
                self.rcs_particles.append(
                    Particle(rcs_tip[0], rcs_tip[1], -cos_a * 2.5, -sin_a * 2.5, 0.4, (220, 245, 255), 3.0)
                )

        # -----------------------------------------------------------------
        # C. FALCON 9 CYLINDRICAL BOOSTER FUSELAGE
        # -----------------------------------------------------------------
        # Main Cylinder
        p_tl = rotate_pt(-half_w, -half_h + 10)
        p_tr = rotate_pt(half_w, -half_h + 10)
        p_br = rotate_pt(half_w, half_h)
        p_bl = rotate_pt(-half_w, half_h)
        body_poly = [p_tl, p_tr, p_br, p_bl]

        # Glossy aerospace white with anti-aliasing
        pygame.draw.polygon(surf, (248, 250, 254), body_poly)
        gfxdraw.aapolygon(surf, body_poly, (200, 212, 230))

        # Carbon-Fiber Black Interstage Band (Upper Stage Interface)
        b_tl = rotate_pt(-half_w, -half_h + 17)
        b_tr = rotate_pt(half_w, -half_h + 17)
        b_br = rotate_pt(half_w, -half_h + 24)
        b_bl = rotate_pt(-half_w, -half_h + 24)
        pygame.draw.polygon(surf, (30, 34, 42), [b_tl, b_tr, b_br, b_bl])

        # Conical Black Aerodynamic Nosecone / Fairing
        nose_tip = rotate_pt(0, -half_h - 7)
        nose_l = rotate_pt(-half_w + 1, -half_h + 10)
        nose_r = rotate_pt(half_w - 1, -half_h + 10)
        pygame.draw.polygon(surf, (40, 46, 58), [nose_tip, nose_r, nose_l])
        gfxdraw.aapolygon(surf, [nose_tip, nose_r, nose_l], (70, 80, 100))

        # Cylindrical 3D Specular Highlight (Reflection bar along booster length)
        spec_top = rotate_pt(-half_w + 3, -half_h + 10)
        spec_bot = rotate_pt(-half_w + 3, half_h)
        pygame.draw.line(surf, (255, 255, 255), spec_top, spec_bot, width=2)

        # Falcon Engine Bell (Superheated Inconel / Niobium Nozzle)
        nozzle_color = (255, 100, 30) if main_throttle > 0 else (45, 48, 56)
        noz_l = rotate_pt(-5, half_h)
        noz_r = rotate_pt(5, half_h)
        noz_br = rotate_pt(4, half_h + 6)
        noz_bl = rotate_pt(-4, half_h + 6)
        pygame.draw.polygon(surf, nozzle_color, [noz_l, noz_r, noz_br, noz_bl])

        # -----------------------------------------------------------------
        # D. TITANIUM GRID FINS (ACTIVE STEERING WITH RCS)
        # -----------------------------------------------------------------
        fin_y = -half_h + 12
        fin_steer = float(action[1]) * 0.25

        # Left Grid Fin (Waffle Grid pattern)
        fl_base1 = rotate_pt(-half_w, fin_y)
        fl_tip1 = rotate_pt(-half_w - 11, fin_y - 2 + int(fin_steer * 8))
        fl_tip2 = rotate_pt(-half_w - 11, fin_y + 5 + int(fin_steer * 8))
        fl_base2 = rotate_pt(-half_w, fin_y + 5)
        pygame.draw.polygon(surf, (24, 28, 36), [fl_base1, fl_tip1, fl_tip2, fl_base2])
        pygame.draw.polygon(surf, (90, 110, 140), [fl_base1, fl_tip1, fl_tip2, fl_base2], width=1)
        # Waffle grid cross-rib
        fl_mid1 = rotate_pt(-half_w - 5, fin_y)
        fl_mid2 = rotate_pt(-half_w - 5, fin_y + 4)
        pygame.draw.line(surf, (90, 110, 140), fl_mid1, fl_mid2, width=1)

        # Right Grid Fin
        fr_base1 = rotate_pt(half_w, fin_y)
        fr_tip1 = rotate_pt(half_w + 11, fin_y - 2 - int(fin_steer * 8))
        fr_tip2 = rotate_pt(half_w + 11, fin_y + 5 - int(fin_steer * 8))
        fr_base2 = rotate_pt(half_w, fin_y + 5)
        pygame.draw.polygon(surf, (24, 28, 36), [fr_base1, fr_tip1, fr_tip2, fr_base2])
        pygame.draw.polygon(surf, (90, 110, 140), [fr_base1, fr_tip1, fr_tip2, fr_base2], width=1)
        # Waffle grid cross-rib
        fr_mid1 = rotate_pt(half_w + 5, fin_y)
        fr_mid2 = rotate_pt(half_w + 5, fin_y + 4)
        pygame.draw.line(surf, (90, 110, 140), fr_mid1, fr_mid2, width=1)

        # -----------------------------------------------------------------
        # E. ARTICULATED CARBON-FIBER LANDING LEGS
        # -----------------------------------------------------------------
        leg_hinge_y = half_h - 6
        leg_span = 26
        foot_y = half_h + 12

        # Left Leg Assembly
        leg_l_h = rotate_pt(-half_w + 2, leg_hinge_y)
        leg_l_f = rotate_pt(-half_w - leg_span, foot_y)
        leg_l_piston = rotate_pt(-half_w + 2, leg_hinge_y + 10)
        # Main carbon strut
        pygame.draw.line(surf, (25, 30, 40), leg_l_h, leg_l_f, width=3)
        # Silver pneumatic piston
        pygame.draw.line(surf, (170, 185, 205), leg_l_piston, leg_l_f, width=2)
        # Foot pad
        foot_l_1 = rotate_pt(-half_w - leg_span - 5, foot_y)
        foot_l_2 = rotate_pt(-half_w - leg_span + 5, foot_y)
        pygame.draw.line(surf, (15, 18, 22), foot_l_1, foot_l_2, width=3)

        # Right Leg Assembly
        leg_r_h = rotate_pt(half_w - 2, leg_hinge_y)
        leg_r_f = rotate_pt(half_w + leg_span, foot_y)
        leg_r_piston = rotate_pt(half_w - 2, leg_hinge_y + 10)
        # Main carbon strut
        pygame.draw.line(surf, (25, 30, 40), leg_r_h, leg_r_f, width=3)
        # Silver pneumatic piston
        pygame.draw.line(surf, (170, 185, 205), leg_r_piston, leg_r_f, width=2)
        # Foot pad
        foot_r_1 = rotate_pt(half_w + leg_span - 5, foot_y)
        foot_r_2 = rotate_pt(half_w + leg_span + 5, foot_y)
        pygame.draw.line(surf, (15, 18, 22), foot_r_1, foot_r_2, width=3)

        # Flashing Green Navigation Strobe
        if (self.frame_count // 12) % 2 == 0:
            strobe_pt = rotate_pt(0, -half_h + 4)
            pygame.draw.circle(surf, (0, 255, 136), strobe_pt, 2)

    def _trigger_touchdown_fx(self, surf: pygame.Surface, env_unwrapped):
        """Spawns confetti and draws celebratory HUD banner upon landing."""
        if not self.touchdown_celebrated and hasattr(env_unwrapped, "lander"):
            self.touchdown_celebrated = True
            pos = env_unwrapped.lander.position
            px, py = self.world_to_screen(pos.x, pos.y)
            # Spawn celebration confetti
            for _ in range(50):
                self.confetti_particles.append(
                    Particle(
                        x=px + random.uniform(-40, 40),
                        y=py + random.uniform(-20, 20),
                        vx=random.uniform(-4.0, 4.0),
                        vy=random.uniform(-5.0, -1.0),
                        ttl=random.uniform(1.5, 3.0),
                        color=random.choice([
                            (0, 255, 136), (0, 220, 255), (255, 204, 0), (255, 255, 255)
                        ]),
                        size=random.uniform(2.5, 4.5),
                        fade_rate=0.015
                    )
                )

        # Draw Mission Success Celebration Banner
        banner_w = 420
        banner_h = 44
        banner_x = (self.w - banner_w) // 2
        banner_y = 30
        
        banner_surf = pygame.Surface((banner_w, banner_h), pygame.SRCALPHA)
        pygame.draw.rect(banner_surf, (0, 40, 25, 210), (0, 0, banner_w, banner_h), border_radius=6)
        pygame.draw.rect(banner_surf, (0, 255, 136), (0, 0, banner_w, banner_h), width=2, border_radius=6)
        surf.blit(banner_surf, (banner_x, banner_y))

        font_success = pygame.font.SysFont("Consolas", 16, bold=True)
        txt = font_success.render("MISSION SUCCESS: TOUCHDOWN CONFIRMED", True, (0, 255, 136))
        surf.blit(txt, (banner_x + (banner_w - txt.get_width()) // 2, banner_y + 12))
