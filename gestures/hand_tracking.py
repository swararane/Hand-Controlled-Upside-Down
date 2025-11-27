import cv2
import mediapipe as mp
import numpy as np


class HandTracker:
    def __init__(self, max_hands=2, detection_conf=0.6, track_conf=0.5, static_image_mode=False):
        self.max_hands = max_hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=static_image_mode,
                                         max_num_hands=max_hands,
                                         min_detection_confidence=detection_conf,
                                         min_tracking_confidence=track_conf)
        self.mp_draw = mp.solutions.drawing_utils

    def process(self, frame, draw=False):
        # frame: BGR
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        hands_out = []
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                lm = []
                for l in hand_landmarks.landmark:
                    lm.append((int(l.x * w), int(l.y * h), l.z))
                label = handedness.classification[0].label
                hands_out.append({
                    'label': label,
                    'landmarks': lm,
                    'landmark_list': hand_landmarks,
                })
                if draw:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return hands_out

    def close(self):
        self.hands.close()
