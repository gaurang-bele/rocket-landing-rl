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

        # Top surface of the deck is EXACTLY at the physical ground line p1[1]
        deck_y = p1[1]
        deck_x = p1[0] - 18
        deck_w = (p2[0] - p1[0]) + 36
        deck_h = 30  # Heavy-duty ocean barge hull extending downwards

        # 1. Water Wake / Ocean Foam under the Barge Hull
        for wave_i in range(3):
            wave_y = deck_y + deck_h + 3 + wave_i * 4
            wave_shift = math.sin(self.frame_count * 0.08 + wave_i) * 6
            pygame.draw.line(surf, (35, 80, 120), (deck_x - 14 + wave_shift, wave_y), (deck_x + deck_w + 14 + wave_shift, wave_y), 2)

        # 2. Barge Hull (Heavy Steel Structure Extending Downward)
        pygame.draw.rect(surf, (16, 20, 28), (deck_x, deck_y, deck_w, deck_h), border_radius=4)
        pygame.draw.rect(surf, (45, 60, 85), (deck_x, deck_y, deck_w, deck_h), width=2, border_radius=4)

        # 3. Top Deck Landing Surface (Non-Skid Steel Plating)
        top_plate_h = 10
        pygame.draw.rect(surf, (32, 40, 56), (deck_x + 2, deck_y, deck_w - 4, top_plate_h))
        pygame.draw.line(surf, (70, 95, 130), (deck_x + 2, deck_y), (deck_x + deck_w - 2, deck_y), width=2)

        # 4. Yellow/Black Industrial Caution Hazard Stripes
        stripe_w = 26
        # Left hazard zone
        pygame.draw.rect(surf, (255, 204, 0), (deck_x, deck_y, stripe_w, deck_h))
        for s in range(0, stripe_w, 6):
            pygame.draw.line(surf, (20, 24, 30), (deck_x + s, deck_y), (deck_x + s + 5, deck_y + deck_h), 2)
        # Right hazard zone
        pygame.draw.rect(surf, (255, 204, 0), (deck_x + deck_w - stripe_w, deck_y, stripe_w, deck_h))
        for s in range(0, stripe_w, 6):
            pygame.draw.line(surf, (20, 24, 30), (deck_x + deck_w - stripe_w + s, deck_y), (deck_x + deck_w - stripe_w + s + 5, deck_y + deck_h), 2)

        # 5. Landing Target Bullseye & Center "X" on the Deck Surface
        center_x = (p1[0] + p2[0]) // 2
        bullseye_y = deck_y + 5
        r_x = int(deck_w * 0.22)
        r_y = 6

        # Perspective Ellipse resting flush on the deck
        pygame.draw.ellipse(surf, (255, 204, 0), (center_x - r_x, bullseye_y - r_y, r_x * 2, r_y * 2), width=2)
        pygame.draw.ellipse(surf, (255, 220, 80), (center_x - r_x // 2, bullseye_y - r_y // 2, r_x, r_y), width=1)

        # Center Landing "X"
        pygame.draw.line(surf, (255, 255, 255), (center_x - 7, bullseye_y - 3), (center_x + 7, bullseye_y + 3), width=2)
        pygame.draw.line(surf, (255, 255, 255), (center_x - 7, bullseye_y + 3), (center_x + 7, bullseye_y - 3), width=2)

        # 6. Telemetry Radio Beacon Masts
        mast_h = 28
        # Port mast (Red Beacon)
        pygame.draw.line(surf, (140, 160, 180), (deck_x + 6, deck_y), (deck_x + 6, deck_y - mast_h), width=2)
        pygame.draw.circle(surf, (255, 40, 40), (deck_x + 6, deck_y - mast_h), 3)
        # Starboard mast (Green Beacon)
        pygame.draw.line(surf, (140, 160, 180), (deck_x + deck_w - 6, deck_y), (deck_x + deck_w - 6, deck_y - mast_h), width=2)
        pygame.draw.circle(surf, (40, 255, 100), (deck_x + deck_w - 6, deck_y - mast_h), 3)

        # 7. Drone Ship Name Stenciled on Front Hull Face (Below the Deck)
        font_stencil = pygame.font.SysFont("Consolas", 10, bold=True)
        lbl = font_stencil.render("A SHORTFALL OF GRAVITAS", True, (130, 150, 175))
        surf.blit(lbl, (center_x - lbl.get_width() // 2, deck_y + 14))

        # 8. Ground Flame Reflection Glow (Flat on the Deck Floor)
        if hasattr(env_unwrapped, "lander") and env_unwrapped.lander is not None:
            l_pos = env_unwrapped.lander.position
            l_px, l_py = self.world_to_screen(l_pos.x, l_pos.y)
            dist_to_deck = deck_y - l_py
            if 0 < dist_to_deck < 200:
                glow_intensity = max(0.0, 1.0 - (dist_to_deck / 200.0))
                glow_rad_x = int(45 + glow_intensity * 55)
                glow_alpha = int(glow_intensity * 120)

                glow_surf = pygame.Surface((glow_rad_x * 2, 14), pygame.SRCALPHA)
                pygame.draw.ellipse(
                    glow_surf,
                    (255, 140, 20, glow_alpha),
                    (0, 0, glow_rad_x * 2, 14)
                )
                surf.blit(glow_surf, (center_x - glow_rad_x, deck_y - 5))

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

        # Aerospace proportions aligned to exact Box2D physical center of mass
        # At touchdown, physical ground is exactly +18.5px below center_py
        half_w = 7.5
        foot_y = 18.5  # Footpads sit 100% flush on the launchpad deck at touchdown
        nozzle_bot_y = 12.0  # Merlin engine nozzle sits 6.5px above deck
        body_bot_y = 7.0
        body_top_y = -56.0
        nose_tip_y = -68.0
        interstage_top = -48.0
        interstage_bot = -40.0
        fin_y = -44.0
        leg_hinge_y = 3.0
        leg_span = 26.0

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
            # Measure clearance to deck so flame compresses near ground
            deck_screen_y = self.world_to_screen(0, env_unwrapped.helipad_y)[1] if hasattr(env_unwrapped, "helipad_y") else 548
            dist_to_deck = max(0, deck_screen_y - center_py)
            
            raw_flame_len = 35 + int(main_throttle * 45) + random.randint(-3, 3)
            # Compress flame if close to ground
            flame_len = min(raw_flame_len, max(8, dist_to_deck - 6))
            flame_w = 13 + int(main_throttle * 6)

            nozzle_pt = rotate_pt(0, nozzle_bot_y)
            flame_tip = rotate_pt(0, nozzle_bot_y + flame_len)
            flame_l = rotate_pt(-flame_w / 2, nozzle_bot_y + min(8, flame_len * 0.3))
            flame_r = rotate_pt(flame_w / 2, nozzle_bot_y + min(8, flame_len * 0.3))

            # Volumetric Flame Bloom (Glowing Radial Halo around retro plume)
            halo_rad = int(flame_len * 0.7)
            if halo_rad > 4:
                halo_surf = pygame.Surface((halo_rad * 2, halo_rad * 2), pygame.SRCALPHA)
                pygame.draw.circle(halo_surf, (255, 120, 20, 40), (halo_rad, halo_rad), halo_rad)
                surf.blit(halo_surf, (nozzle_pt[0] - halo_rad, nozzle_pt[1] - halo_rad // 3))

            # Layer 1: Outer Turbulent Orange/Red Plume
            pygame.draw.polygon(surf, (255, 60, 0), [nozzle_pt, flame_l, flame_tip, flame_r])

            # Layer 2: Mid Supersonic Yellow Core
            mid_len = int(flame_len * 0.65)
            mid_w = int(flame_w * 0.58)
            mid_tip = rotate_pt(0, nozzle_bot_y + mid_len)
            mid_l = rotate_pt(-mid_w / 2, nozzle_bot_y + min(5, mid_len * 0.3))
            mid_r = rotate_pt(mid_w / 2, nozzle_bot_y + min(5, mid_len * 0.3))
            pygame.draw.polygon(surf, (255, 215, 0), [nozzle_pt, mid_l, mid_tip, mid_r])

            # Layer 3: Inner White-Hot Combustion Throat
            core_len = int(flame_len * 0.35)
            core_w = int(flame_w * 0.30)
            core_tip = rotate_pt(0, nozzle_bot_y + core_len)
            core_l = rotate_pt(-core_w / 2, nozzle_bot_y + 2)
            core_r = rotate_pt(core_w / 2, nozzle_bot_y + 2)
            pygame.draw.polygon(surf, (255, 255, 255), [nozzle_pt, core_l, core_tip, core_r])

            # Layer 4: Mach Shock Diamonds along supersonic axis
            if flame_len > 25:
                for d_idx in range(1, 3):
                    d_dist = nozzle_bot_y + (flame_len * 0.28 * d_idx)
                    d_sz = 3
                    d_pts = [
                        rotate_pt(0, d_dist - d_sz),
                        rotate_pt(d_sz, d_dist),
                        rotate_pt(0, d_dist + d_sz),
                        rotate_pt(-d_sz, d_dist)
                    ]
                    pygame.draw.polygon(surf, (220, 240, 255), d_pts)

            # Spawn Exhaust Smoke Particles
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
                        color=(255, 140, 30),
                        size=random.uniform(4.0, 8.0)
                    )
                )

            # Ground Dust Billows when retro plume impinges on deck
            alt_agl = pos.y - (env_unwrapped.helipad_y if hasattr(env_unwrapped, "helipad_y") else 3.33)
            if 0 < alt_agl < 2.5 and len(self.dust_particles) < 60:
                self.dust_particles.append(
                    Particle(
                        x=nozzle_pt[0],
                        y=deck_screen_y,
                        vx=-random.uniform(3.0, 6.0),
                        vy=-random.uniform(0.2, 0.8),
                        ttl=random.uniform(0.6, 1.2),
                        color=(190, 205, 225),
                        size=random.uniform(6.0, 12.0)
                    )
                )
                self.dust_particles.append(
                    Particle(
                        x=nozzle_pt[0],
                        y=deck_screen_y,
                        vx=random.uniform(3.0, 6.0),
                        vy=-random.uniform(0.2, 0.8),
                        ttl=random.uniform(0.6, 1.2),
                        color=(190, 205, 225),
                        size=random.uniform(6.0, 12.0)
                    )
                )

        # -----------------------------------------------------------------
        # B. COLD-GAS RCS ATTITUDE THRUSTERS
        # -----------------------------------------------------------------
        side_throttle = float(action[1])
        rcs_y = body_top_y + 12

        if side_throttle < -0.25:
            burst_len = int(abs(side_throttle) * 22)
            rcs_orig = rotate_pt(half_w, rcs_y)
            rcs_tip = rotate_pt(half_w + burst_len, rcs_y)
            pygame.draw.line(surf, (220, 245, 255), rcs_orig, rcs_tip, width=3)
            if len(self.rcs_particles) < 40:
                self.rcs_particles.append(
                    Particle(rcs_tip[0], rcs_tip[1], cos_a * 2.5, sin_a * 2.5, 0.4, (220, 245, 255), 3.0)
                )
        elif side_throttle > 0.25:
            burst_len = int(side_throttle * 22)
            rcs_orig = rotate_pt(-half_w, rcs_y)
            rcs_tip = rotate_pt(-half_w - burst_len, rcs_y)
            pygame.draw.line(surf, (220, 245, 255), rcs_orig, rcs_tip, width=3)
            if len(self.rcs_particles) < 40:
                self.rcs_particles.append(
                    Particle(rcs_tip[0], rcs_tip[1], -cos_a * 2.5, -sin_a * 2.5, 0.4, (220, 245, 255), 3.0)
                )

        # -----------------------------------------------------------------
        # C. FALCON 9 CYLINDRICAL BOOSTER FUSELAGE
        # -----------------------------------------------------------------
        p_tl = rotate_pt(-half_w, body_top_y)
        p_tr = rotate_pt(half_w, body_top_y)
        p_br = rotate_pt(half_w, body_bot_y)
        p_bl = rotate_pt(-half_w, body_bot_y)
        body_poly = [p_tl, p_tr, p_br, p_bl]

        # Glossy aerospace white with anti-aliasing
        pygame.draw.polygon(surf, (248, 250, 254), body_poly)
        gfxdraw.aapolygon(surf, body_poly, (200, 212, 230))

        # Carbon-Fiber Black Interstage Band (Upper Stage Interface)
        b_tl = rotate_pt(-half_w, interstage_top)
        b_tr = rotate_pt(half_w, interstage_top)
        b_br = rotate_pt(half_w, interstage_bot)
        b_bl = rotate_pt(-half_w, interstage_bot)
        pygame.draw.polygon(surf, (30, 34, 42), [b_tl, b_tr, b_br, b_bl])

        # Conical Black Aerodynamic Nosecone / Fairing
        nose_tip = rotate_pt(0, nose_tip_y)
        nose_l = rotate_pt(-half_w + 1, body_top_y)
        nose_r = rotate_pt(half_w - 1, body_top_y)
        pygame.draw.polygon(surf, (40, 46, 58), [nose_tip, nose_r, nose_l])
        gfxdraw.aapolygon(surf, [nose_tip, nose_r, nose_l], (70, 80, 100))

        # Cylindrical 3D Specular Highlight
        spec_top = rotate_pt(-half_w + 3, body_top_y)
        spec_bot = rotate_pt(-half_w + 3, body_bot_y)
        pygame.draw.line(surf, (255, 255, 255), spec_top, spec_bot, width=2)

        # Falcon Engine Bell (Superheated Inconel / Niobium Nozzle)
        nozzle_color = (255, 100, 30) if main_throttle > 0 else (45, 48, 56)
        noz_l = rotate_pt(-5, body_bot_y)
        noz_r = rotate_pt(5, body_bot_y)
        noz_br = rotate_pt(4, nozzle_bot_y)
        noz_bl = rotate_pt(-4, nozzle_bot_y)
        pygame.draw.polygon(surf, nozzle_color, [noz_l, noz_r, noz_br, noz_bl])

        # -----------------------------------------------------------------
        # D. TITANIUM GRID FINS (ACTIVE STEERING WITH RCS)
        # -----------------------------------------------------------------
        fin_steer = float(action[1]) * 0.25

        # Left Grid Fin
        fl_base1 = rotate_pt(-half_w, fin_y)
        fl_tip1 = rotate_pt(-half_w - 10, fin_y - 2 + int(fin_steer * 8))
        fl_tip2 = rotate_pt(-half_w - 10, fin_y + 5 + int(fin_steer * 8))
        fl_base2 = rotate_pt(-half_w, fin_y + 5)
        pygame.draw.polygon(surf, (24, 28, 36), [fl_base1, fl_tip1, fl_tip2, fl_base2])
        pygame.draw.polygon(surf, (90, 110, 140), [fl_base1, fl_tip1, fl_tip2, fl_base2], width=1)

        # Right Grid Fin
        fr_base1 = rotate_pt(half_w, fin_y)
        fr_tip1 = rotate_pt(half_w + 10, fin_y - 2 - int(fin_steer * 8))
        fr_tip2 = rotate_pt(half_w + 10, fin_y + 5 - int(fin_steer * 8))
        fr_base2 = rotate_pt(half_w, fin_y + 5)
        pygame.draw.polygon(surf, (24, 28, 36), [fr_base1, fr_tip1, fr_tip2, fr_base2])
        pygame.draw.polygon(surf, (90, 110, 140), [fr_base1, fr_tip1, fr_tip2, fr_base2], width=1)

        # -----------------------------------------------------------------
        # E. ARTICULATED CARBON-FIBER LANDING LEGS (REST FLUSH ON DECK)
        # -----------------------------------------------------------------
        # Left Leg Assembly
        leg_l_h = rotate_pt(-half_w + 2, leg_hinge_y)
        leg_l_f = rotate_pt(-half_w - leg_span, foot_y)
        leg_l_piston = rotate_pt(-half_w + 2, leg_hinge_y + 6)
        # Main carbon strut
        pygame.draw.line(surf, (25, 30, 40), leg_l_h, leg_l_f, width=3)
        # Silver pneumatic piston
        pygame.draw.line(surf, (170, 185, 205), leg_l_piston, leg_l_f, width=2)
        # Foot pad resting flat on launchpad deck
        foot_l_1 = rotate_pt(-half_w - leg_span - 5, foot_y)
        foot_l_2 = rotate_pt(-half_w - leg_span + 5, foot_y)
        pygame.draw.line(surf, (15, 18, 22), foot_l_1, foot_l_2, width=3)

        # Right Leg Assembly
        leg_r_h = rotate_pt(half_w - 2, leg_hinge_y)
        leg_r_f = rotate_pt(half_w + leg_span, foot_y)
        leg_r_piston = rotate_pt(half_w - 2, leg_hinge_y + 6)
        # Main carbon strut
        pygame.draw.line(surf, (25, 30, 40), leg_r_h, leg_r_f, width=3)
        # Silver pneumatic piston
        pygame.draw.line(surf, (170, 185, 205), leg_r_piston, leg_r_f, width=2)
        # Foot pad resting flat on launchpad deck
        foot_r_1 = rotate_pt(half_w + leg_span - 5, foot_y)
        foot_r_2 = rotate_pt(half_w + leg_span + 5, foot_y)
        pygame.draw.line(surf, (15, 18, 22), foot_r_1, foot_r_2, width=3)

        # Flashing Green Navigation Strobe
        if (self.frame_count // 12) % 2 == 0:
            strobe_pt = rotate_pt(0, body_top_y + 4)
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
