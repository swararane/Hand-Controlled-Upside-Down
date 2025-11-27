import numpy as np
import cv2
import math
import time
from .particles import ParticleEngine
from .particles import SporeEngine
from .lightning import draw_lightning
from .shaders import displacement_map, heat_distort, glow_effect
from utils.helpers import make_circle_mask, draw_glow, lerp
from .shaders import chromatic_aberration, crt_filter, color_grade_upside_down


class Portal:
    def __init__(self, size=(1280, 720)):
        self.width, self.height = size[0], size[1]
        self.center = (self.width // 2, self.height // 2)
        self.radius = min(self.width, self.height) // 6
        self.open_amount = 0.0
        self.state = 'closed'  # closed, opening, open, closing
        self.particles = ParticleEngine(max_particles=800)
        self.spores = SporeEngine((self.height, self.width), max_spores=400)
        self.last_t = time.time()
        self.rotation = 0.0
        self.twist = 0.0
        self.last_open_t = 0
        self.core_color = (30, 10, 180)  # will tint later

    def update(self, dt):
        # update particles and animation states
        self.particles.update(dt)
        if self.state == 'opening':
            self.open_amount = min(1.0, self.open_amount + dt * 1.2)
            if self.open_amount >= 1.0:
                self.state = 'open'
        elif self.state == 'closing':
            self.open_amount = max(0.0, self.open_amount - dt * 1.8)
            if self.open_amount <= 0.0:
                self.state = 'closed'
        # slow twist decay
        self.twist *= 0.92

    def open(self):
        self.state = 'opening'
        self.last_open_t = time.time()
        # burst particles
        for _ in range(60):
            self.particles.emit((self.center[0] + np.random.randint(-10, 10), self.center[1] + np.random.randint(-10, 10)), count=6, spread=60)
        # spawn spores more heavily on open
        for _ in range(100):
            self.spores.emit_spore((self.center[0] + np.random.randint(-int(self.radius/2), int(self.radius/2)), self.center[1] + np.random.randint(-int(self.radius/2), int(self.radius/2))))

    def close(self):
        self.state = 'closing'
        # shockwave: emit heavy particles
        for _ in range(80):
            self.particles.emit((self.center[0], self.center[1]), count=8, spread=200)
        # dissipate spores
        for _ in range(120):
            self.spores.emit_spore((self.center[0] + np.random.randint(-self.radius, self.radius), self.center[1] + np.random.randint(-self.radius, self.radius)))

    def set_pos(self, pos):
        self.center = (int(pos[0]), int(pos[1]))

    def apply_twist(self, amount):
        self.twist += amount

    def render(self, frame, upside_down=True):
        h, w = frame.shape[:2]
        now = int(time.time() * 1000)
        # If Upside Down visuals are disabled, return original camera frame (normal webcam)
        if not upside_down:
            return frame

        # create portal mask
        radius = int(self.radius * (0.55 + self.open_amount * 1.6))
        mask = make_circle_mask(frame.shape, self.center, radius)
        # start with desaturated/darker background for Upside Down mood
        bg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
        bg = cv2.convertScaleAbs(bg, alpha=0.75, beta=-20)
        # tint background slightly
        bg = color_grade_upside_down(bg)
        portal_img = np.zeros_like(frame)
        # radial gradient core
        Y, X = np.indices((h, w))
        dx = X - self.center[0]
        dy = Y - self.center[1]
        dist = np.sqrt(dx * dx + dy * dy)
        with_core = dist < radius
        t = (radius - dist) / (radius + 1e-6)
        t = np.clip(t, 0, 1)
        # create red Vecna-like core with noise
        noise = (np.random.rand(h, w) * 0.6 + 0.4) * t
        core = np.zeros_like(frame, dtype=np.float32)
        # red/purple center
        core[..., 2] = (np.clip(200 + 90 * noise, 0, 255)) * (t ** 2.0)
        core[..., 1] = (np.clip(18 + 12 * noise, 0, 255)) * (t ** 1.3)
        core[..., 0] = (np.clip(6 + 4 * noise, 0, 255)) * (t ** 1.0)
        portal_img = np.clip(core, 0, 255).astype(np.uint8)
        # inner moving ripples
        ripple = np.zeros_like(frame)
        ripple_radius = radius * (0.3 + 0.7 * self.open_amount)
        freq = 12.0
        theta = np.arctan2(dy, dx) + self.twist * 0.5
        ripple_field = np.sin(dist / max(1.0, ripple_radius / freq) + theta * 4.0 + now / 400.0)
        ripple_mask = (dist < ripple_radius * 1.1) & (dist > ripple_radius * 0.2)
        # ensure shapes align for broadcasting: make both (h,w,1)
        rf = (ripple_field * 0.5 + 0.5)[..., None]
        tt = (t ** 1.4)[..., None]
        ripple_strength = rf * tt
        ripple[..., 2] = (ripple_strength[..., 0] * 160).astype(np.uint8)
        portal_img = cv2.addWeighted(portal_img, 1.0, ripple.astype(np.uint8), 0.55 + 0.25 * self.open_amount, 0)
        # displacement/distortion
        # heavier displacement when open
        distorted = displacement_map(frame.copy(), self.center, int(radius * (1.0 + 0.9 * self.open_amount)), strength=26 * (0.3 + self.open_amount))
        # composite portal onto distorted background using graded bg
        mask_f = (mask.astype(np.float32) / 255.0)[..., None]
        comp = (distorted.astype(np.float32) * (1 - mask_f) + (portal_img.astype(np.float32) * mask_f)).astype(np.uint8)
        # blend with the moody background outside portal
        bg_mask = (1 - mask_f)
        comp = (comp.astype(np.float32) * (1 - mask_f) + bg.astype(np.float32) * bg_mask).astype(np.uint8)
        # heat distort on portal area
        comp = heat_distort(comp, self.center, radius, now, strength=12 * (0.3 + self.open_amount))
        # add lightning around rim and rim cracks
        lightning_layer = np.zeros_like(frame)
        if self.open_amount > 0.03:
            draw_lightning(lightning_layer, self.center, int(radius * (1.0 + 0.12 * np.random.rand())), intensity=1.0 + self.open_amount, color=(40, 20, 240))
        comp = cv2.addWeighted(comp, 1.0, lightning_layer, 0.9 * (0.6 + self.open_amount * 0.7), 0)
        # rim cracks overlay
        from .lightning import draw_rim_cracks
        cracks = draw_rim_cracks(frame, self.center, radius, intensity=1.0 * self.open_amount)
        cracks_col = cv2.cvtColor(cracks, cv2.COLOR_GRAY2BGR)
        comp = cv2.addWeighted(comp, 1.0, cracks_col, 0.5 * self.open_amount, 0)
        # particle render
        self.particles.render(comp)
        # ambient spores render
        self.spores.update(1.0 / 30.0)
        self.spores.render(comp)
        # glow
        glow_mask = (mask * (0.5 + 0.5 * self.open_amount)).astype(np.uint8)
        comp = glow_effect(comp, glow_mask, ksize=51, intensity=1.0 * (0.8 + self.open_amount), color=(40, 16, 220))
        # chromatic aberration and CRT tint for Upside Down feel
        comp = chromatic_aberration(comp, amount=6 * (0.4 + self.open_amount))
        comp = color_grade_upside_down(comp)
        comp = crt_filter(comp, scan_alpha=0.06)
        # vignette / CRT flicker
        flicker = (np.random.rand() * 0.06 + 0.97) * (0.95 + 0.05 * np.sin(now / 90.0))
        comp = np.clip(comp.astype(np.float32) * flicker, 0, 255).astype(np.uint8)
        return comp
