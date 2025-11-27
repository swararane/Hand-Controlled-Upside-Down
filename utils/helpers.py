import math
import time
import cv2
import numpy as np


def millis():
    return int(time.time() * 1000)


def clamp(v, a, b):
    return max(a, min(b, v))


def lerp(a, b, t):
    return a + (b - a) * t


def map_range(v, a, b, c, d):
    if a == b:
        return c
    return c + (d - c) * ((v - a) / (b - a))


def draw_glow(img, mask, color=(0, 0, 255), intensity=1.0, ksize=51):
    # mask should be single channel 0..255
    if mask is None:
        return img
    glow = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    glow_norm = cv2.normalize(glow, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    b, g, r = color
    overlay = np.zeros_like(img, dtype=np.float32) / 255.0
    overlay[..., 0] = (b / 255.0) * glow_norm * intensity
    overlay[..., 1] = (g / 255.0) * glow_norm * intensity
    overlay[..., 2] = (r / 255.0) * glow_norm * intensity
    out = img.astype(np.float32) / 255.0
    out = np.clip(out + overlay, 0, 1.0)
    return (out * 255).astype(np.uint8)


def make_circle_mask(shape, center, radius):
    mask = np.zeros((shape[0], shape[1]), dtype=np.uint8)
    cv2.circle(mask, (int(center[0]), int(center[1])), int(radius), 255, -1)
    return mask


def angle_between(v1, v2):
    a1 = math.atan2(v1[1], v1[0])
    a2 = math.atan2(v2[1], v2[0])
    return a2 - a1


def rotate_point(pt, center, angle):
    s = math.sin(angle)
    c = math.cos(angle)
    x = pt[0] - center[0]
    y = pt[1] - center[1]
    xr = x * c - y * s
    yr = x * s + y * c
    return (xr + center[0], yr + center[1])
