import cv2
import time
import numpy as np
from gestures.hand_tracking import HandTracker
from gestures.gesture_detector import GestureDetector
import cv2
import time
import math
import numpy as np
from gestures.hand_tracking import HandTracker
from gestures.gesture_detector import GestureDetector
from effects.portal import Portal
from utils.helpers import map_range


def draw_status_overlay(img, gstate, portal, hands):
    h, w = img.shape[:2]
    y = 40
    fh = 32
    # FPS and portal
    cv2.putText(img, f'Portal: {portal.state} {portal.open_amount:.2f}', (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
    y += fh
    # Two-hand distance
    cv2.putText(img, f'TwoHandDist: {gstate.two_hand_distance:.2f}', (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 180, 200), 2)
    y += fh
    # Rotation
    rot_deg = math.degrees(gstate.rotation)
    cv2.putText(img, f'Rotation: {rot_deg:.1f} deg', (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 220, 200), 2)
    y += fh
    # Push
    push_txt = 'PUSH DETECTED' if gstate.pushing else 'push: -'
    cv2.putText(img, push_txt, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 230, 250) if gstate.pushing else (180, 180, 180), 2)
    y += fh
    # Pinch info
    for i in range(2):
        status = 'Pinch' if gstate.pinch[i] else 'NoPinch'
        xpos = int(20 + i * 220)
        cv2.putText(img, f'Hand{i+1}: {status}', (xpos, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 120, 140) if gstate.pinch[i] else (150, 150, 150), 2)
    # draw pinch positions
    for i in range(2):
        if gstate.pinch_pos[i] is not None:
            px, py = int(gstate.pinch_pos[i][0]), int(gstate.pinch_pos[i][1])
            cv2.circle(img, (px, py), 12, (0, 255, 255), 2, lineType=cv2.LINE_AA)
    # draw portal center
    cv2.circle(img, (int(portal.center[0]), int(portal.center[1])), max(6, int(8 + portal.open_amount * 10)), (180, 40, 220), 2, lineType=cv2.LINE_AA)


def draw_mode_hint(img, upside_down_mode, demo_mode):
    h, w = img.shape[:2]
    txt = 'Mode: UpsideDown' if upside_down_mode else 'Mode: Normal'
    demo = 'Demo: ON' if demo_mode else 'Demo: OFF'
    cv2.putText(img, txt, (w - 380, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 200, 220), 2)
    cv2.putText(img, demo, (w - 380, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # use lower resolution for better real-time performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    tracker = HandTracker(max_hands=2, detection_conf=0.6, track_conf=0.5)
    detector = GestureDetector(pinch_thresh=0.06, push_thresh=0.02)
    ret, frame = cap.read()
    if not ret:
        print('Cannot open camera')
        return
    h, w = frame.shape[:2]
    portal = Portal(size=(w, h))
    last_time = time.time()
    dragging = False
    drag_offset = (0, 0)

    upside_down_mode = True
    demo_mode = False
    demo_timer = 0.0
    try:
        # auto mode: switch to Normal when no hands for a timeout, UpsideDown when hands present
        last_hand_seen = 0.0
        hand_absence_timeout = 1.5
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            now = time.time()
            dt = now - last_time
            last_time = now
            hands = tracker.process(frame, draw=False)
            gstate = detector.update(hands, frame.shape)
            # update last seen hands time
            if len(hands) > 0:
                last_hand_seen = now
                # if hands reappear, switch into UpsideDown mode automatically
                if not upside_down_mode:
                    upside_down_mode = True
            else:
                # no hands detected for a while -> return to normal mode and close portal
                if (now - last_hand_seen) > hand_absence_timeout:
                    if upside_down_mode:
                        upside_down_mode = False
                        portal.close()
            # gesture actions
            # Two-hand stretch -> open portal when distance exceeds threshold
            if gstate.two_hand_distance > 0.25 and portal.state in ['closed', 'closing']:
                portal.open()
            # pinch + drag (primary hand)
            if len(hands) > 0:
                if gstate.pinch[0] and gstate.pinch_pos[0] is not None:
                    px, py = gstate.pinch_pos[0]
                    if not dragging:
                        dragging = True
                        drag_offset = (portal.center[0] - px, portal.center[1] - py)
                    portal.set_pos((px + drag_offset[0], py + drag_offset[1]))
                else:
                    dragging = False
            # rotate hand -> twist
            if abs(gstate.rotation) > 0.05:
                portal.apply_twist(gstate.rotation * 0.8)
            # palm push -> close
            if gstate.pushing:
                portal.close()
            # update portal
            portal.update(dt)
            # render scene (portal render still respects portal state)
            # pass upside_down_mode to portal if later we want different params (currently portal global visual toggles are via main)
            out = portal.render(frame, upside_down=upside_down_mode)
            # overlay HUD
            draw_status_overlay(out, gstate, portal, hands)
            draw_mode_hint(out, upside_down_mode, demo_mode)
            # fps
            fps = int(1.0 / max(1e-6, dt))
            cv2.putText(out, f'FPS: {fps}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
            cv2.imshow('Open the Gate to the Upside Down', out)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('u'):
                upside_down_mode = not upside_down_mode
                # toggle stronger effects by flipping portal params
                if upside_down_mode:
                    portal.radius = int(min(w, h) // 5)
                else:
                    portal.radius = int(min(w, h) // 6)
            elif key == ord('d'):
                demo_mode = not demo_mode
            # demo automation: open/close every few seconds
            if demo_mode:
                demo_timer += dt
                if demo_timer > 3.0:
                    demo_timer = 0.0
                    if portal.state in ['closed', 'closing']:
                        portal.open()
                    else:
                        portal.close()
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()
