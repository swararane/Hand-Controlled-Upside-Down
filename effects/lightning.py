import numpy as np
import cv2
import random


def draw_lightning(img, center, radius, intensity=1.0, segments=6, color=(0, 0, 255)):
    h, w = img.shape[:2]
    cx, cy = int(center[0]), int(center[1])
    for i in range(int(2 + intensity * 4)):
        angle = random.random() * 2 * np.pi
        start = (int(cx + np.cos(angle) * radius * 0.8), int(cy + np.sin(angle) * radius * 0.8))
        end = (int(cx + np.cos(angle) * radius * (1.15 + random.uniform(-0.1, 0.4))),
               int(cy + np.sin(angle) * radius * (1.15 + random.uniform(-0.1, 0.4))))
        points = [start]
        for s in range(segments):
            t = s / float(segments)
            nx = int(lerp(start[0], end[0], t) + random.randint(-int(radius * 0.15), int(radius * 0.15)))
            ny = int(lerp(start[1], end[1], t) + random.randint(-int(radius * 0.15), int(radius * 0.15)))
            points.append((nx, ny))
        points.append(end)
        thickness = int(1 + intensity * 2)
        for j in range(len(points) - 1):
            cv2.line(img, points[j], points[j + 1], color, thickness, lineType=cv2.LINE_AA)
    # subtle glow
    glow = cv2.GaussianBlur(img, (7, 7), 0)
    img[:] = cv2.addWeighted(img, 0.6, glow, 0.8, 0)
    return img


def lerp(a, b, t):
    return int(a + (b - a) * t)


def draw_rim_cracks(mask_shape, center, radius, intensity=1.0):
    # returns an image with white crack lines on black background
    # mask_shape can be either a (h,w) tuple or an image array
    if hasattr(mask_shape, 'shape'):
        h, w = mask_shape.shape[:2]
    else:
        h, w = int(mask_shape[0]), int(mask_shape[1])
    cx, cy = int(center[0]), int(center[1])
    out = np.zeros((int(h), int(w)), dtype=np.uint8)
    for i in range(int(6 + intensity * 12)):
        angle = random.random() * 2 * np.pi
        length = int(radius * (1.0 + random.uniform(0.05, 0.35)))
        start = (int(cx + np.cos(angle) * radius), int(cy + np.sin(angle) * radius))
        end = (int(cx + np.cos(angle) * (radius + length)), int(cy + np.sin(angle) * (radius + length)))
        pts = [start]
        segs = int(6 + random.randint(0, 6))
        for s in range(segs):
            t = s / float(segs)
            nx = int(lerp(start[0], end[0], t) + random.randint(-int(radius * 0.08), int(radius * 0.08)))
            ny = int(lerp(start[1], end[1], t) + random.randint(-int(radius * 0.08), int(radius * 0.08)))
            pts.append((nx, ny))
        pts.append(end)
        for j in range(len(pts) - 1):
            cv2.line(out, pts[j], pts[j + 1], 255, int(1 + intensity * 2), lineType=cv2.LINE_AA)
    out = cv2.GaussianBlur(out, (5, 5), 0)
    return out
