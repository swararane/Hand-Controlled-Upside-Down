import numpy as np
import random
import cv2


class Particle:
    def __init__(self, pos, vel, life, color):
        self.pos = np.array(pos, dtype=np.float32)
        self.vel = np.array(vel, dtype=np.float32)
        self.life = life
        self.max_life = life
        self.color = color

    def update(self, dt):
        self.pos += self.vel * dt
        self.life -= dt
        # slight gravity/downward drift
        self.vel[1] += 10.0 * dt

    def alive(self):
        return self.life > 0


class ParticleEngine:
    def __init__(self, max_particles=500):
        self.max_particles = max_particles
        self.particles = []

    def emit(self, pos, count=6, spread=30, color=(120, 10, 10)):
        for _ in range(count):
            if len(self.particles) >= self.max_particles:
                break
            ang = random.uniform(0, 2 * np.pi)
            speed = random.uniform(10, spread)
            vel = (np.cos(ang) * speed, np.sin(ang) * speed)
            life = random.uniform(0.8, 2.5)
            c = (random.randint(max(0, color[0]-30), min(255, color[0]+30)),
                 random.randint(max(0, color[1]-30), min(255, color[1]+30)),
                 random.randint(max(0, color[2]-30), min(255, color[2]+30)))
            self.particles.append(Particle(pos, vel, life, c))

    def update(self, dt):
        for p in self.particles:
            p.update(dt)
        self.particles = [p for p in self.particles if p.alive()]

    def render(self, img):
        for p in self.particles:
            alpha = max(0.0, min(1.0, p.life / p.max_life))
            x, y = int(p.pos[0]), int(p.pos[1])
            if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
                continue
            cv2.circle(img, (x, y), int(2 + 3 * (1 - alpha)), p.color, -1, lineType=cv2.LINE_AA)


class SporeParticle:
    def __init__(self, pos, vel, life, color=(120, 30, 30), size=2):
        self.pos = np.array(pos, dtype=np.float32)
        self.vel = np.array(vel, dtype=np.float32)
        self.life = life
        self.max_life = life
        self.color = color
        self.size = size

    def update(self, dt):
        # slower floating movement
        self.pos += self.vel * dt
        self.life -= dt
        # gentle drift
        self.vel += np.array((np.random.randn() * 2.0, np.random.randn() * 2.0)) * dt

    def alive(self):
        return self.life > 0


class SporeEngine:
    def __init__(self, bounds, max_spores=600):
        self.bounds = bounds
        self.max_spores = max_spores
        self.spores = []

    def emit_spore(self, pos=None):
        h, w = self.bounds
        if pos is None:
            pos = (np.random.randint(0, w), np.random.randint(0, h))
        vel = (np.random.uniform(-8, 8), np.random.uniform(-12, 6))
        life = np.random.uniform(4.0, 14.0)
        color = (np.random.randint(80, 200), np.random.randint(10, 40), np.random.randint(30, 160))
        self.spores.append(SporeParticle(pos, vel, life, color=color, size=np.random.randint(1, 4)))
        if len(self.spores) > self.max_spores:
            self.spores = self.spores[-self.max_spores:]

    def update(self, dt):
        for s in self.spores:
            s.update(dt)
        self.spores = [s for s in self.spores if s.alive()]

    def render(self, img):
        for s in self.spores:
            a = max(0.0, min(1.0, s.life / s.max_life))
            x, y = int(s.pos[0]), int(s.pos[1])
            if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
                continue
            col = tuple(int(c * a) for c in s.color)
            cv2.circle(img, (x, y), s.size, col, -1, lineType=cv2.LINE_AA)
