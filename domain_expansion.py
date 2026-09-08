"""
DOMAIN EXPANSION — Gesture-Activated Webcam Effect
====================================================
Feel like Satoru Gojo. Bring your hands together in the "sealing" pose
(both hands raised in front of you, fingers spread and interlaced,
palms toward the camera) and hold it — the screen will collapse into
a Domain Expansion.

HOW IT WORKS
------------
1. Opens your front/webcam camera.
2. Uses MediaPipe Hands to track both of your hands in real time.
3. Shows an on-screen tutorial telling you exactly what pose to make.
4. When both hands are detected close together with fingers spread
   (open palms, wrists near each other) and you HOLD that pose for
   about a second, it triggers a full-screen "Domain Expansion"
   animation: vignette darkening, an expanding sorcery-circle mandala,
   radiating energy rings, and your domain's name stamped on screen.
5. After a few seconds it fades back and cools down before you can
   trigger it again.

RUN
---
    pip install -r requirements.txt
    python domain_expansion.py

Press 'q' at any time to quit. Press 'n' to cycle preset domain names.
"""

import time
import math
import collections

import cv2
import numpy as np
import mediapipe as mp

# --------------------------------------------------------------------------
# CONFIG — tweak these to taste
# --------------------------------------------------------------------------
CAM_INDEX = 0                 # which camera to open (0 = default/front cam)
HOLD_SECONDS = 1.0            # how long you must hold the pose to trigger
DOMAIN_DURATION = 4.5         # how long the domain expansion effect lasts
COOLDOWN_SECONDS = 2.5        # cooldown before you can trigger it again
HANDS_TOGETHER_MAX_DIST = 0.28  # max normalized distance between wrists to
                                 # count as "hands brought together"
DOMAIN_NAMES = [
    "UNLIMITED VOID",
    "MALEVOLENT SHRINE",
    "INFINITE VOID",
    "CHIMERA SHADOW GARDEN",
]

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


# --------------------------------------------------------------------------
# Gesture detection helpers
# --------------------------------------------------------------------------
def fingers_extended(hand_landmarks):
    """Return how many of the 5 fingers are extended (roughly) for a hand.
    Uses the classic 'tip further from wrist than pip joint' heuristic,
    which is orientation-tolerant enough for a webcam pose check.
    """
    lm = hand_landmarks.landmark
    wrist = lm[mp_hands.HandLandmark.WRIST]

    def dist(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    tips = [
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
    ]
    pips = [
        mp_hands.HandLandmark.THUMB_IP,
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.PINKY_PIP,
    ]

    count = 0
    for tip_id, pip_id in zip(tips, pips):
        if dist(lm[tip_id], wrist) > dist(lm[pip_id], wrist):
            count += 1
    return count


def hand_center(hand_landmarks):
    lm = hand_landmarks.landmark
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def check_domain_pose(multi_hand_landmarks):
    """Returns True if the current frame's hands match the 'sealing' pose:
    both hands present, both mostly open (fingers spread), and brought
    close together in front of the body.
    """
    if not multi_hand_landmarks or len(multi_hand_landmarks) < 2:
        return False

    h1, h2 = multi_hand_landmarks[0], multi_hand_landmarks[1]

    if fingers_extended(h1) < 4 or fingers_extended(h2) < 4:
        return False

    c1 = hand_center(h1)
    c2 = hand_center(h2)
    dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])

    return dist < HANDS_TOGETHER_MAX_DIST


# --------------------------------------------------------------------------
# Visual effect: the Domain Expansion itself
# --------------------------------------------------------------------------
def draw_mandala(frame, center, radius, thickness, color, rotation):
    """Draws a rotating sorcery-circle mandala at `center` with given radius."""
    cx, cy = center
    # outer ring
    cv2.circle(frame, (cx, cy), int(radius), color, thickness, cv2.LINE_AA)
    # inner ring
    cv2.circle(frame, (cx, cy), int(radius * 0.7), color, max(1, thickness - 1), cv2.LINE_AA)
    # radiating spokes + small glyphs
    spokes = 12
    for i in range(spokes):
        angle = rotation + (2 * math.pi * i / spokes)
        x1 = int(cx + radius * 0.7 * math.cos(angle))
        y1 = int(cy + radius * 0.7 * math.sin(angle))
        x2 = int(cx + radius * 1.0 * math.cos(angle))
        y2 = int(cy + radius * 1.0 * math.sin(angle))
        cv2.line(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        gx = int(cx + radius * 0.85 * math.cos(angle))
        gy = int(cy + radius * 0.85 * math.sin(angle))
        cv2.circle(frame, (gx, gy), 3, color, -1, cv2.LINE_AA)
    # inscribed polygon
    poly_pts = []
    sides = 6
    for i in range(sides):
        angle = -rotation * 1.3 + (2 * math.pi * i / sides)
        poly_pts.append((
            int(cx + radius * 0.55 * math.cos(angle)),
            int(cy + radius * 0.55 * math.sin(angle)),
        ))
    cv2.polylines(frame, [np.array(poly_pts)], True, color, thickness, cv2.LINE_AA)


def render_domain_expansion(frame, elapsed, duration, domain_name):
    """Overlays the full domain-expansion effect onto `frame` in place.
    `elapsed` is seconds since the effect started (used to animate it).
    """
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    progress = min(elapsed / duration, 1.0)

    # 1) Vignette / void darkening — ramps in fast, holds, fades at the end
    if progress < 0.15:
        dark_amount = progress / 0.15
    elif progress > 0.85:
        dark_amount = (1.0 - progress) / 0.15
    else:
        dark_amount = 1.0
    dark_amount = max(0.0, min(1.0, dark_amount)) * 0.75

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (5, 0, 10), -1)
    cv2.addWeighted(overlay, dark_amount, frame, 1 - dark_amount, 0, frame)

    # 2) Expanding / pulsing mandala rings (purple/blue "cursed energy" look)
    rotation = elapsed * 1.8
    base_radius = min(w, h) * 0.15 + progress * min(w, h) * 0.55
    pulse = 1.0 + 0.05 * math.sin(elapsed * 6)
    color = (255, 60, 180)  # BGR: vivid magenta/purple energy
    draw_mandala(frame, (cx, cy), base_radius * pulse, 2, color, rotation)
    draw_mandala(frame, (cx, cy), base_radius * pulse * 0.55, 1, (255, 150, 220), -rotation * 1.4)

    # 3) Extra thin energy rings radiating outward continuously
    for k in range(3):
        r = ((elapsed * 220) + k * 140) % (max(w, h) * 0.75)
        alpha = max(0.0, 1.0 - r / (max(w, h) * 0.75))
        if alpha <= 0:
            continue
        ring_color = tuple(int(c * alpha) for c in (255, 90, 200))
        cv2.circle(frame, (cx, cy), int(r), ring_color, 2, cv2.LINE_AA)

    # 4) Domain name text, glowing / punch-in
    text = domain_name
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.3 + 0.05 * math.sin(elapsed * 5)
    thickness = 3
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    tx, ty = cx - tw // 2, int(h * 0.15)
    # glow (drawn a few times slightly offset/blurred-look via thicker under-stroke)
    cv2.putText(frame, text, (tx, ty), font, scale, (40, 0, 60), thickness + 6, cv2.LINE_AA)
    cv2.putText(frame, text, (tx, ty), font, scale, (255, 120, 255), thickness, cv2.LINE_AA)

    subtitle = "DOMAIN EXPANSION"
    (sw, sh), _ = cv2.getTextSize(subtitle, font, 0.7, 2)
    cv2.putText(frame, subtitle, (cx - sw // 2, ty + 35), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


# --------------------------------------------------------------------------
# Main loop / state machine
# --------------------------------------------------------------------------
class State:
    IDLE = "idle"
    HOLDING = "holding"
    ACTIVE = "active"
    COOLDOWN = "cooldown"


def draw_tutorial(frame, state, hold_progress):
    h, w = frame.shape[:2]
    box_h = 92
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    if state == State.IDLE:
        line1 = "Raise BOTH hands, fingers spread, palms toward camera."
        line2 = "Bring your hands together in front of your chest to seal the domain."
    elif state == State.HOLDING:
        line1 = "Hold the pose steady..."
        line2 = f"Charging cursed energy: {int(hold_progress * 100)}%"
    elif state == State.ACTIVE:
        line1 = "DOMAIN EXPANSION ACTIVE"
        line2 = "Guaranteed hit."
    else:
        line1 = "Cooling down..."
        line2 = "Recover your cursed energy before casting again."

    cv2.putText(frame, line1, (16, 34), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, line2, (16, 66), font, 0.6, (200, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "[q] quit   [n] change domain name", (w - 320, box_h - 10),
                font, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    if state == State.HOLDING:
        bar_w = int((w - 32) * hold_progress)
        cv2.rectangle(frame, (16, box_h - 6), (16 + bar_w, box_h - 2), (0, 255, 180), -1)

    return frame


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("ERROR: could not open the camera. Check CAM_INDEX / permissions.")
        return

    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.6,
    )

    state = State.IDLE
    hold_start = None
    active_start = None
    cooldown_start = None
    domain_idx = 0
    pose_history = collections.deque(maxlen=5)  # smooths flicker in detection

    print("Domain Expansion system running. Press 'q' in the video window to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera frame not received — stopping.")
            break

        frame = cv2.flip(frame, 1)  # mirror, feels natural for a front camera
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        pose_detected = False
        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hl, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
            pose_detected = check_domain_pose(results.multi_hand_landmarks)

        pose_history.append(pose_detected)
        pose_stable = sum(pose_history) >= 3  # majority of recent frames

        now = time.time()
        hold_progress = 0.0

        if state == State.IDLE:
            if pose_stable:
                state = State.HOLDING
                hold_start = now
        elif state == State.HOLDING:
            if not pose_stable:
                state = State.IDLE
                hold_start = None
            else:
                hold_progress = min((now - hold_start) / HOLD_SECONDS, 1.0)
                if hold_progress >= 1.0:
                    state = State.ACTIVE
                    active_start = now
        elif state == State.ACTIVE:
            elapsed = now - active_start
            frame = render_domain_expansion(frame, elapsed, DOMAIN_DURATION, DOMAIN_NAMES[domain_idx])
            if elapsed >= DOMAIN_DURATION:
                state = State.COOLDOWN
                cooldown_start = now
        elif state == State.COOLDOWN:
            if now - cooldown_start >= COOLDOWN_SECONDS:
                state = State.IDLE
                pose_history.clear()

        frame = draw_tutorial(frame, state, hold_progress)

        cv2.imshow("Domain Expansion", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            domain_idx = (domain_idx + 1) % len(DOMAIN_NAMES)

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
