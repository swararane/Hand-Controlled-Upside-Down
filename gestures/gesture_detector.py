import math
import numpy as np
from collections import deque


class GestureState:
    def __init__(self):
        self.pinch = [False, False]
        self.pinch_pos = [None, None]
        self.pinch_history = [deque(maxlen=8), deque(maxlen=8)]
        self.two_hand_distance = 0.0
        self.two_hand_scale = 1.0
        self.rotation = 0.0
        self.pushing = False
        self.push_cooldown = 0


class GestureDetector:
    def __init__(self, pinch_thresh=0.06, push_thresh=0.06):
        # thresholds in normalized coordinates
        self.pinch_thresh = pinch_thresh
        self.push_thresh = push_thresh
        self.state = GestureState()

    @staticmethod
    def _norm_point(landmark, frame_shape):
        h, w = frame_shape[:2]
        return (landmark[0] / w, landmark[1] / h, landmark[2])

    @staticmethod
    def _dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def update(self, hands, frame_shape):
        # hands is list of dicts with 'label' and 'landmarks' (x, y, z in px and z normalized)
        s = self.state
        # reset
        s.two_hand_distance = 0.0
        s.rotation = 0.0
        s.pushing = False
        # compute per-hand gestures
        for i in range(2):
            s.pinch[i] = False
        if len(hands) == 0:
            return s
        # map hands to left/right consistently
        hands_map = {}
        for h in hands:
            hands_map[h['label'].lower()] = h
        ordered = []
        if 'left' in hands_map and 'right' in hands_map:
            ordered = [hands_map['left'], hands_map['right']]
        else:
            # single hand: use index 0 as primary
            ordered = hands
        # detect pinch for up to two hands
        for idx, hand in enumerate(ordered[:2]):
            lm = hand['landmarks']
            frame = frame_shape
            # thumb tip is index 4, index tip 8
            thumb = self._norm_point(lm[4], frame)
            index = self._norm_point(lm[8], frame)
            dist = self._dist(thumb, index)
            if dist < self.pinch_thresh:
                s.pinch[idx] = True
                # store pinch pos as midpoint
                px = (lm[4][0] + lm[8][0]) / 2.0
                py = (lm[4][1] + lm[8][1]) / 2.0
                s.pinch_pos[idx] = (px, py)
                s.pinch_history[idx].append((px, py))
            else:
                s.pinch_pos[idx] = None
        # two-hand distance & scale
        if len(ordered) >= 2:
            # we'll use wrist (0) or index tip (8) distance
            a = ordered[0]['landmarks'][0]
            b = ordered[1]['landmarks'][0]
            h, w = frame_shape[:2]
            an = (a[0] / w, a[1] / h)
            bn = (b[0] / w, b[1] / h)
            s.two_hand_distance = self._dist(an, bn)
        # rotation: measure angle of vector from wrist->index for primary hand
        p = ordered[0]
        v1 = (p['landmarks'][8][0] - p['landmarks'][0][0], p['landmarks'][8][1] - p['landmarks'][0][1])
        # second vector: wrist->middle_mcp (or use previous frame?) we'll use wrist->palm base (landmark 9)
        v2 = (p['landmarks'][9][0] - p['landmarks'][0][0], p['landmarks'][9][1] - p['landmarks'][0][1])
        # compute angle
        ang1 = math.atan2(v1[1], v1[0])
        ang2 = math.atan2(v2[1], v2[0])
        s.rotation = ang1 - ang2
        # palm push detection: use z of wrist (landmark 0) or average of palm landmarks
        # Mediapipe z is negative toward camera; detect sudden change forward (more negative)
        if len(ordered) > 0:
            zvals = [p['landmarks'][i][2] for i in [0, 1, 5, 9]]
            zavg = sum(zvals) / len(zvals)
            # store history in deque
            if not hasattr(self, 'z_history'):
                self.z_history = deque(maxlen=6)
            self.z_history.append(zavg)
            if len(self.z_history) >= 4:
                # compute delta
                dz = self.z_history[-2] - self.z_history[-1]
                if dz > self.push_thresh and (not getattr(self, 'last_push_time', 0) or (millis() - getattr(self, 'last_push_time', 0) > 600)):
                    s.pushing = True
                    self.last_push_time = millis()
        return s


def millis():
    import time
    return int(time.time() * 1000)
