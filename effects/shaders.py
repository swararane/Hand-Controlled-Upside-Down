import numpy as np
import cv2
import math


def displacement_map(img, center, radius, strength=15.0, seed=0):
    # create a simple radial displacement using sin+noise
    h, w = img.shape[:2]
    Y, X = np.indices((h, w), dtype=np.float32)
    cx, cy = center
    dx = X - cx
    dy = Y - cy
    dist = np.sqrt(dx * dx + dy * dy)
    mask = dist <= radius
    # normalized radius
    nr = (radius - dist) / (radius + 1e-6)
    nr = np.clip(nr, 0, 1)
    # noise
    rng = np.random.RandomState(seed)
    noise = rng.randn(h, w).astype(np.float32)
    # displacement factor
    disp = np.sin(dist / 8.0 + noise * 3.0) * (nr ** 1.2) * strength
    # compute offsets
    ox = (dx / (dist + 1e-6)) * disp
    oy = (dy / (dist + 1e-6)) * disp
    map_x = (X + ox).astype(np.float32)
    map_y = (Y + oy).astype(np.float32)
    out = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out


def heat_distort(img, center, radius, time_ms, strength=8.0):
    # small FFT-like jitter using sin waves
    h, w = img.shape[:2]
    Y, X = np.indices((h, w), dtype=np.float32)
    cx, cy = center
    dx = X - cx
    dy = Y - cy
    dist = np.sqrt(dx * dx + dy * dy)
    mask = dist <= radius
    nr = (radius - dist) / (radius + 1e-6)
    nr = np.clip(nr, 0, 1)
    t = time_ms / 1000.0
    ox = np.sin((Y + t * 120.0) / 10.0) * (nr ** 1.5) * (strength * 0.6)
    oy = np.cos((X + t * 90.0) / 12.0) * (nr ** 1.5) * (strength * 0.6)
    map_x = (X + ox).astype(np.float32)
    map_y = (Y + oy).astype(np.float32)
    out = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out


def glow_effect(img, mask, ksize=31, intensity=1.0, color=(0, 0, 255)):
    if mask is None:
        return img
    blur = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    blur = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)
    colored = np.zeros_like(img, dtype=np.uint8)
    b, g, r = color
    colored[:, :, 0] = (blur * (b / 255.0)).astype(np.uint8)
    colored[:, :, 1] = (blur * (g / 255.0)).astype(np.uint8)
    colored[:, :, 2] = (blur * (r / 255.0)).astype(np.uint8)
    out = cv2.addWeighted(img, 1.0, colored, intensity, 0)
    return out


def chromatic_aberration(img, amount=6):
    # split channels and offset slightly to produce chromatic aberration
    h, w = img.shape[:2]
    b, g, r = cv2.split(img)
    M = np.float32([[1, 0, amount * 0.2], [0, 1, amount * -0.2]])
    r_off = cv2.warpAffine(r, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    M2 = np.float32([[1, 0, -amount * 0.3], [0, 1, amount * 0.3]])
    b_off = cv2.warpAffine(b, M2, (w, h), borderMode=cv2.BORDER_REFLECT)
    out = cv2.merge([b_off, g, r_off])
    return out


def crt_filter(img, scan_alpha=0.05, curvature=0.0008):
    # simple scanlines + slight vignette / curvature
    h, w = img.shape[:2]
    out = img.copy().astype(np.float32)
    # scanlines
    y = np.arange(h).reshape(h, 1)
    scan = (0.5 + 0.5 * np.sin(y / 2.5))
    scan = (1.0 - scan * scan_alpha)[:, None]
    out = out * scan
    # slight vignette
    X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    radius = np.sqrt(X * X + Y * Y)
    vign = 1.0 - (radius ** 2) * 0.6
    vign = np.clip(vign, 0.45, 1.0)
    out = out * vign[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def color_grade_upside_down(img):
    # shift midtones to purple/red, crush blacks
    lut = np.arange(256, dtype=np.uint8)
    # slightly raise reds, reduce greens, boost blues for eerie tone
    b, g, r = cv2.split(img)
    r = cv2.add(r, 30)
    g = cv2.subtract(g, 10)
    b = cv2.add(b, 10)
    merged = cv2.merge([b, g, r])
    # increase contrast
    p = 1.15
    merged = cv2.convertScaleAbs(merged, alpha=p, beta=-10)
    # apply slight color desaturation, then overlay tint
    gray = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    tinted = cv2.addWeighted(gray3, 0.25, merged, 0.75, 0)
    # add a red/purple wash
    wash = np.full_like(tinted, (18, 10, 60))
    out = cv2.addWeighted(tinted, 0.85, wash, 0.15, 0)
    return out
