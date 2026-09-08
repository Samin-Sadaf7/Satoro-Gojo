"""
DOMAIN EXPANSION — Gesture-Activated Webcam Effect

Rewritten for current MediaPipe releases.

WHY THIS VERSION EXISTS
------------------------
Google removed the old "Solutions" API (mp.solutions.hands,
mp.solutions.drawing_utils, mp.solutions.drawing_styles) from recent
MediaPipe Python packages. That API only exists in old wheels like
0.10.32 and won't import on current MediaPipe. This script uses the
API that replaced it: the "Tasks" API (mp.tasks.vision.HandLandmarker).

The Tasks API needs a small model file (hand_landmarker.task). This
script downloads it automatically the first time you run it and
caches it next to the script, so you don't have to find it yourself.

Gesture:
    1. Raise both hands.
    2. Keep your fingers extended.
    3. Bring the hands close together.
    4. Hold the pose for HOLD_SECONDS.
    5. DOMAIN EXPANSION activates.

Controls:
    q -> quit
    n -> next domain name

Setup (macOS):
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install mediapipe opencv-python numpy

    First run will trigger a macOS camera-permission prompt. If you
    don't see one, or the camera fails to open, go to:
    System Settings -> Privacy & Security -> Camera
    and enable access for Terminal (or whatever app/IDE runs Python).
"""

import datetime
import math
import os
import sys
import time
import urllib.request
import collections

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions


# ============================================================================
# CONFIGURATION
# ============================================================================

CAM_INDEX = 0

# Camera resolution
CAM_WIDTH = 1280
CAM_HEIGHT = 720

# Gesture settings
HOLD_SECONDS = 1.0

# How long the Domain Expansion animation lasts
DOMAIN_DURATION = 4.5

# Time before another activation is possible
COOLDOWN_SECONDS = 2.5

# Maximum normalized distance between the centers of both hands
HANDS_TOGETHER_MAX_DIST = 0.28

# MediaPipe Tasks confidence values
MIN_HAND_DETECTION_CONFIDENCE = 0.65
MIN_HAND_PRESENCE_CONFIDENCE = 0.60
MIN_TRACKING_CONFIDENCE = 0.60

# Number of frames used to stabilize the gesture
POSE_HISTORY_LENGTH = 7

# Number of positive frames required for a stable pose
POSE_HISTORY_REQUIRED = 4

# ------------------------------------------------------------------------
# Session recording
# ------------------------------------------------------------------------

# If True, the whole session (everything shown on screen, overlays
# included) is saved to a video file when the program exits.
RECORD_VIDEO = True

# Where recordings are saved. Defaults to a "recordings" folder next
# to this script.
VIDEO_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "recordings",
)

# Used if the camera doesn't report a usable FPS value.
VIDEO_FPS_FALLBACK = 30.0

# Where the hand landmark model is cached locally.
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")

# Official Google-hosted model file. Public, no auth needed.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


# ============================================================================
# DOMAIN NAMES
# ============================================================================

DOMAIN_NAMES = [
    "UNLIMITED VOID",
    "MALEVOLENT SHRINE",
    "INFINITE VOID",
    "CHIMERA SHADOW GARDEN",
]


# ============================================================================
# MEDIAPIPE LANDMARK INDICES
# (Unchanged: the Tasks API uses the same 21-point hand topology as the
#  old Solutions API.)
# ============================================================================

WRIST = 0

THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


FINGER_TIPS = [
    THUMB_TIP,
    INDEX_TIP,
    MIDDLE_TIP,
    RING_TIP,
    PINKY_TIP,
]

# Standard 21-point hand skeleton edges. The old mp_hands.HAND_CONNECTIONS
# constant came from the now-removed solutions module, so it's redefined
# here for drawing.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm
]


# ============================================================================
# MODEL DOWNLOAD
# ============================================================================

def ensure_model_downloaded():
    """
    Make sure hand_landmarker.task exists locally, downloading it
    from Google's public model store if needed.
    """

    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        return True

    print(f"[INFO] Hand landmark model not found at:\n       {MODEL_PATH}")
    print(f"[INFO] Downloading it from:\n       {MODEL_URL}")

    try:
        tmp_path = MODEL_PATH + ".part"

        def _report(block_num, block_size, total_size):
            if total_size <= 0:
                return
            downloaded = block_num * block_size
            pct = min(100, downloaded * 100 // total_size)
            print(f"\r[INFO] Downloading model... {pct}%", end="", flush=True)

        urllib.request.urlretrieve(MODEL_URL, tmp_path, reporthook=_report)
        print()
        os.replace(tmp_path, MODEL_PATH)
        print("[OK] Model downloaded successfully.")
        return True

    except Exception as exc:
        print("\n[ERROR] Could not download the hand landmark model.")
        print(f"        {type(exc).__name__}: {exc}")
        print("\n[FIX] Download it manually and save it as:")
        print(f"        {MODEL_PATH}")
        print(f"      URL: {MODEL_URL}")
        return False


# ============================================================================
# MEDIAPIPE HANDLANDMARKER SETUP (Tasks API)
# ============================================================================

def create_hand_landmarker():
    """
    Create the MediaPipe Tasks HandLandmarker in VIDEO mode.

    VIDEO mode is used (instead of LIVE_STREAM) because it's
    synchronous: landmarker.detect_for_video() blocks and returns a
    result immediately, which fits a simple frame-by-frame webcam
    loop without needing an async callback.
    """

    options = mp_vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    return mp_vision.HandLandmarker.create_from_options(options)


# ============================================================================
# GEOMETRY HELPERS
# ============================================================================

def distance_2d(a, b):
    """
    Euclidean distance between two normalized MediaPipe landmarks.
    Works with the Tasks API's NormalizedLandmark objects, which
    still expose .x / .y / .z directly.
    """

    return math.hypot(
        a.x - b.x,
        a.y - b.y,
    )


def hand_center(landmarks):
    """
    Calculate the center of a detected hand.

    `landmarks` is a plain list of 21 NormalizedLandmark objects
    (Tasks API result.hand_landmarks[i]) — there is no wrapping
    `.landmark` attribute like the old Solutions API had.
    """

    if not landmarks:
        return 0.0, 0.0

    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]

    return (
        sum(xs) / len(xs),
        sum(ys) / len(ys),
    )


# ============================================================================
# FINGER DETECTION
# ============================================================================

def fingers_extended(landmarks):
    """
    Estimate how many fingers are extended.

    `landmarks` is a plain list of 21 landmarks (see hand_center above).

    Returns:
        integer from 0 to 5
    """

    lm = landmarks

    count = 0

    # ------------------------------------------------------------------------
    # Index, middle, ring and pinky
    # ------------------------------------------------------------------------

    finger_pairs = [
        (INDEX_TIP, INDEX_PIP),
        (MIDDLE_TIP, MIDDLE_PIP),
        (RING_TIP, RING_PIP),
        (PINKY_TIP, PINKY_PIP),
    ]

    wrist = lm[WRIST]

    for tip_id, pip_id in finger_pairs:

        tip_distance = distance_2d(
            lm[tip_id],
            wrist,
        )

        pip_distance = distance_2d(
            lm[pip_id],
            wrist,
        )

        if tip_distance > pip_distance * 1.10:
            count += 1

    # ------------------------------------------------------------------------
    # Thumb
    # ------------------------------------------------------------------------

    thumb_tip_distance = distance_2d(
        lm[THUMB_TIP],
        wrist,
    )

    thumb_mcp_distance = distance_2d(
        lm[THUMB_MCP],
        wrist,
    )

    if thumb_tip_distance > thumb_mcp_distance * 1.15:
        count += 1

    return count


# ============================================================================
# GESTURE DETECTION
# ============================================================================

def check_domain_pose(hand_landmarks_list):
    """
    Detect the Domain Expansion sealing pose.

    `hand_landmarks_list` is result.hand_landmarks from the Tasks API:
    a list where each element is itself a list of 21 landmarks for
    one detected hand.

    Conditions:
        - At least two hands must be detected.
        - Both hands should have mostly extended fingers.
        - Both hands should be close together.
    """

    if not hand_landmarks_list:
        return False

    if len(hand_landmarks_list) < 2:
        return False

    # Use the first two detected hands.
    h1 = hand_landmarks_list[0]
    h2 = hand_landmarks_list[1]

    # ------------------------------------------------------------------------
    # Both hands should be open.
    # ------------------------------------------------------------------------

    fingers_1 = fingers_extended(h1)
    fingers_2 = fingers_extended(h2)

    if fingers_1 < 4 or fingers_2 < 4:
        return False

    # ------------------------------------------------------------------------
    # Find hand centers.
    # ------------------------------------------------------------------------

    c1 = hand_center(h1)
    c2 = hand_center(h2)

    # ------------------------------------------------------------------------
    # Distance between hands in normalized coordinates.
    # ------------------------------------------------------------------------

    center_distance = math.hypot(
        c1[0] - c2[0],
        c1[1] - c2[1],
    )

    return center_distance < HANDS_TOGETHER_MAX_DIST


# ============================================================================
# DRAWING HAND LANDMARKS
# (Replaces mp_draw.draw_landmarks, which came from the removed
#  solutions.drawing_utils module.)
# ============================================================================

def draw_hand_landmarks(frame, landmarks):
    """
    Draw hand landmarks + skeleton connections directly with OpenCV,
    since the Tasks API ships no built-in drawing helper.

    `landmarks` is a plain list of 21 normalized landmarks for one hand.
    """

    h, w = frame.shape[:2]

    points = [
        (int(lm.x * w), int(lm.y * h))
        for lm in landmarks
    ]

    # Skeleton lines
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start_idx],
            points[end_idx],
            (120, 220, 120),
            2,
            cv2.LINE_AA,
        )

    # Joints
    for i, (x, y) in enumerate(points):
        color = (0, 200, 255) if i in FINGER_TIPS else (255, 255, 255)
        cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)


# ============================================================================
# DOMAIN EXPANSION VISUAL EFFECT
# ============================================================================

def draw_mandala(
    frame,
    center,
    radius,
    thickness,
    color,
    rotation,
):
    """
    Draw a rotating sorcery-circle mandala.
    """

    cx, cy = center

    radius = max(1, int(radius))

    # ------------------------------------------------------------------------
    # Outer ring
    # ------------------------------------------------------------------------

    cv2.circle(
        frame,
        (cx, cy),
        radius,
        color,
        thickness,
        cv2.LINE_AA,
    )

    # ------------------------------------------------------------------------
    # Inner ring
    # ------------------------------------------------------------------------

    cv2.circle(
        frame,
        (cx, cy),
        max(1, int(radius * 0.7)),
        color,
        max(1, thickness - 1),
        cv2.LINE_AA,
    )

    # ------------------------------------------------------------------------
    # Radiating spokes
    # ------------------------------------------------------------------------

    spokes = 12

    for i in range(spokes):

        angle = (
            rotation
            + (2 * math.pi * i / spokes)
        )

        x1 = int(
            cx
            + radius * 0.7 * math.cos(angle)
        )

        y1 = int(
            cy
            + radius * 0.7 * math.sin(angle)
        )

        x2 = int(
            cx
            + radius * math.cos(angle)
        )

        y2 = int(
            cy
            + radius * math.sin(angle)
        )

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            1,
            cv2.LINE_AA,
        )

        gx = int(
            cx
            + radius * 0.85 * math.cos(angle)
        )

        gy = int(
            cy
            + radius * 0.85 * math.sin(angle)
        )

        cv2.circle(
            frame,
            (gx, gy),
            3,
            color,
            -1,
            cv2.LINE_AA,
        )

    # ------------------------------------------------------------------------
    # Hexagonal inner polygon
    # ------------------------------------------------------------------------

    poly_pts = []

    sides = 6

    for i in range(sides):

        angle = (
            -rotation * 1.3
            + (2 * math.pi * i / sides)
        )

        poly_pts.append(
            (
                int(
                    cx
                    + radius * 0.55 * math.cos(angle)
                ),
                int(
                    cy
                    + radius * 0.55 * math.sin(angle)
                ),
            )
        )

    cv2.polylines(
        frame,
        [np.array(poly_pts, dtype=np.int32)],
        True,
        color,
        thickness,
        cv2.LINE_AA,
    )


# ============================================================================
# DOMAIN EXPANSION RENDERING
# ============================================================================

def render_domain_expansion(
    frame,
    elapsed,
    duration,
    domain_name,
):
    """
    Render the full Domain Expansion effect.
    """

    h, w = frame.shape[:2]

    cx = w // 2
    cy = h // 2

    progress = min(
        elapsed / duration,
        1.0,
    )

    # ------------------------------------------------------------------------
    # Vignette / darkness
    # ------------------------------------------------------------------------

    if progress < 0.15:

        dark_amount = (
            progress / 0.15
        )

    elif progress > 0.85:

        dark_amount = (
            (1.0 - progress) / 0.15
        )

    else:

        dark_amount = 1.0

    dark_amount = (
        max(0.0, min(1.0, dark_amount))
        * 0.75
    )

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (w, h),
        (5, 0, 10),
        -1,
    )

    cv2.addWeighted(
        overlay,
        dark_amount,
        frame,
        1.0 - dark_amount,
        0,
        frame,
    )

    # ------------------------------------------------------------------------
    # Main mandala
    # ------------------------------------------------------------------------

    rotation = elapsed * 1.8

    base_radius = (
        min(w, h) * 0.15
        + progress * min(w, h) * 0.55
    )

    pulse = (
        1.0
        + 0.05 * math.sin(elapsed * 6)
    )

    color = (255, 60, 180)

    draw_mandala(
        frame,
        (cx, cy),
        base_radius * pulse,
        2,
        color,
        rotation,
    )

    draw_mandala(
        frame,
        (cx, cy),
        base_radius * pulse * 0.55,
        1,
        (255, 150, 220),
        -rotation * 1.4,
    )

    # ------------------------------------------------------------------------
    # Energy rings
    # ------------------------------------------------------------------------

    max_radius = max(w, h) * 0.75

    for k in range(3):

        radius = (
            elapsed * 220
            + k * 140
        ) % max_radius

        alpha = max(
            0.0,
            1.0 - radius / max_radius,
        )

        if alpha <= 0:
            continue

        ring_color = tuple(
            int(c * alpha)
            for c in (255, 90, 200)
        )

        cv2.circle(
            frame,
            (cx, cy),
            int(radius),
            ring_color,
            2,
            cv2.LINE_AA,
        )

    # ------------------------------------------------------------------------
    # Domain name
    # ------------------------------------------------------------------------

    font = cv2.FONT_HERSHEY_SIMPLEX

    text = domain_name

    scale = (
        1.3
        + 0.05 * math.sin(elapsed * 5)
    )

    thickness = 3

    (tw, th), _ = cv2.getTextSize(
        text,
        font,
        scale,
        thickness,
    )

    tx = cx - tw // 2
    ty = int(h * 0.15)

    # Glow
    cv2.putText(
        frame,
        text,
        (tx, ty),
        font,
        scale,
        (40, 0, 60),
        thickness + 6,
        cv2.LINE_AA,
    )

    # Main text
    cv2.putText(
        frame,
        text,
        (tx, ty),
        font,
        scale,
        (255, 120, 255),
        thickness,
        cv2.LINE_AA,
    )

    # ------------------------------------------------------------------------
    # Subtitle
    # ------------------------------------------------------------------------

    subtitle = "DOMAIN EXPANSION"

    (sw, sh), _ = cv2.getTextSize(
        subtitle,
        font,
        0.7,
        2,
    )

    cv2.putText(
        frame,
        subtitle,
        (cx - sw // 2, ty + 35),
        font,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return frame


# ============================================================================
# STATE MACHINE
# ============================================================================

class State:

    IDLE = "idle"

    HOLDING = "holding"

    ACTIVE = "active"

    COOLDOWN = "cooldown"


# ============================================================================
# TUTORIAL UI
# ============================================================================

def draw_tutorial(
    frame,
    state,
    hold_progress,
    current_domain,
    detected_hands,
):
    """
    Draw instructions and camera information.
    """

    h, w = frame.shape[:2]

    box_h = 125

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (w, box_h),
        (20, 20, 20),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.55,
        frame,
        0.45,
        0,
        frame,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    # ------------------------------------------------------------------------
    # Main instruction
    # ------------------------------------------------------------------------

    if state == State.IDLE:

        line1 = (
            "Raise BOTH hands, fingers spread, "
            "palms toward camera."
        )

        line2 = (
            "Bring your hands together "
            "to seal the domain."
        )

    elif state == State.HOLDING:

        line1 = "Hold the pose steady..."

        line2 = (
            f"Charging cursed energy: "
            f"{int(hold_progress * 100)}%"
        )

    elif state == State.ACTIVE:

        line1 = "DOMAIN EXPANSION ACTIVE"

        line2 = "Guaranteed hit."

    else:

        line1 = "Cooling down..."

        line2 = (
            "Recover your cursed energy "
            "before casting again."
        )

    # ------------------------------------------------------------------------
    # Draw main UI
    # ------------------------------------------------------------------------

    cv2.putText(
        frame,
        line1,
        (16, 32),
        font,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        line2,
        (16, 62),
        font,
        0.55,
        (200, 200, 255),
        1,
        cv2.LINE_AA,
    )

    # ------------------------------------------------------------------------
    # Current domain
    # ------------------------------------------------------------------------

    domain_text = f"Domain: {current_domain}"

    cv2.putText(
        frame,
        domain_text,
        (16, 90),
        font,
        0.45,
        (180, 180, 255),
        1,
        cv2.LINE_AA,
    )

    # ------------------------------------------------------------------------
    # Camera / detection information
    # ------------------------------------------------------------------------

    camera_text = f"Hands detected: {detected_hands}"

    cv2.putText(
        frame,
        camera_text,
        (16, 112),
        font,
        0.45,
        (180, 255, 180),
        1,
        cv2.LINE_AA,
    )

    controls = "[q] quit   [n] next domain"

    (cw, ch), _ = cv2.getTextSize(
        controls,
        font,
        0.45,
        1,
    )

    cv2.putText(
        frame,
        controls,
        (w - cw - 16, 90),
        font,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    # ------------------------------------------------------------------------
    # Charging bar
    # ------------------------------------------------------------------------

    if state == State.HOLDING:

        bar_x1 = 16
        bar_y1 = box_h - 6

        bar_x2 = w - 16

        bar_width = int(
            (bar_x2 - bar_x1)
            * hold_progress
        )

        cv2.rectangle(
            frame,
            (bar_x1, bar_y1),
            (bar_x1 + bar_width, box_h - 2),
            (0, 255, 180),
            -1,
        )

    return frame


# ============================================================================
# CAMERA
# ============================================================================

def open_camera():
    """
    Open the Mac webcam.

    On macOS, explicitly requesting the AVFoundation backend avoids
    the slow auto-probing OpenCV otherwise does across backends, and
    sidesteps some "OpenCV: not authorized to capture video" warnings
    when permissions haven't been granted yet.
    """

    if sys.platform == "darwin":
        cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(CAM_INDEX)

    if not cap.isOpened():
        return None

    # Request desired resolution.
    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAM_WIDTH,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAM_HEIGHT,
    )

    return cap


# ============================================================================
# VIDEO RECORDING
# ============================================================================

def create_video_writer(cap, frame_width, frame_height):
    """
    Create a cv2.VideoWriter that saves the session to an .mp4 file.

    Returns (writer, output_path) or (None, None) if recording is
    disabled or the writer could not be created (e.g. no working
    codec available). The caller should treat a None writer as
    "recording unavailable" and keep running the camera loop anyway.
    """

    if not RECORD_VIDEO:
        return None, None

    try:
        os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
    except Exception as exc:
        print(f"[WARNING] Could not create recordings folder: {exc}")
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 1 or fps > 240:
        fps = VIDEO_FPS_FALLBACK

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"domain_expansion_{timestamp}.mp4"
    output_path = os.path.join(VIDEO_OUTPUT_DIR, filename)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (frame_width, frame_height),
    )

    if not writer.isOpened():
        print(
            "[WARNING] Could not open a video writer for recording. "
            "Continuing without saving this session."
        )
        return None, None

    print(f"[OK] Recording this session to:\n     {output_path}")

    return writer, output_path


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 70)
    print("DOMAIN EXPANSION")
    print("MediaPipe Tasks API - HandLandmarker")
    print("=" * 70)

    # ------------------------------------------------------------------------
    # Report MediaPipe version
    # ------------------------------------------------------------------------

    try:
        version = mp.__version__
        print(f"[INFO] MediaPipe version: {version}")
    except Exception:
        pass

    # ------------------------------------------------------------------------
    # Make sure the hand landmark model is available
    # ------------------------------------------------------------------------

    if not ensure_model_downloaded():
        return

    # ------------------------------------------------------------------------
    # Open camera
    # ------------------------------------------------------------------------

    cap = open_camera()

    if cap is None:

        print(
            "\n[ERROR] Could not open the camera."
        )

        print(
            "Check CAM_INDEX and macOS camera permissions."
        )

        print(
            "On macOS: System Settings -> Privacy & Security -> Camera, "
            "then enable access for Terminal / VS Code / your Python app."
        )

        return

    # ------------------------------------------------------------------------
    # Create HandLandmarker
    # ------------------------------------------------------------------------

    landmarker = None

    try:

        landmarker = create_hand_landmarker()

    except Exception as exc:

        print(
            "\n[ERROR] Could not initialize "
            "MediaPipe HandLandmarker."
        )

        print(exc)

        cap.release()

        return

    # ------------------------------------------------------------------------
    # Create OpenCV window
    # ------------------------------------------------------------------------

    window_name = "Domain Expansion - Camera"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        window_name,
        CAM_WIDTH,
        CAM_HEIGHT,
    )

    # ------------------------------------------------------------------------
    # State variables
    # ------------------------------------------------------------------------

    state = State.IDLE

    hold_start = None

    active_start = None

    cooldown_start = None

    domain_idx = 0

    pose_history = collections.deque(
        maxlen=POSE_HISTORY_LENGTH
    )

    # VIDEO running mode needs strictly increasing timestamps per frame.
    start_time = time.time()
    last_timestamp_ms = -1

    # Video writer is created lazily once we know the actual frame
    # size the camera is delivering (it can differ slightly from the
    # requested CAM_WIDTH/CAM_HEIGHT).
    video_writer = None
    video_output_path = None

    print(
        "\n[OK] MediaPipe HandLandmarker initialized."
    )

    print(
        "[OK] Camera initialized."
    )

    print(
        "[OK] Domain Expansion system running."
    )

    print(
        "[INFO] Raise both hands and bring them together."
    )

    print(
        "[INFO] Press 'q' to quit."
    )

    print(
        "[INFO] Press 'n' to change domain."
    )

    print()

    # ------------------------------------------------------------------------
    # Main camera loop
    # ------------------------------------------------------------------------

    try:

        while True:

            # ================================================================
            # Capture frame
            # ================================================================

            ok, frame = cap.read()

            if not ok:

                print(
                    "[ERROR] Camera frame "
                    "could not be read."
                )

                break

            # ================================================================
            # Mirror the webcam
            # ================================================================

            frame = cv2.flip(
                frame,
                1,
            )

            # ================================================================
            # Start recording (first frame only, once we know the
            # actual frame size the camera is delivering)
            # ================================================================

            if RECORD_VIDEO and video_writer is None:

                video_writer, video_output_path = create_video_writer(
                    cap,
                    frame.shape[1],
                    frame.shape[0],
                )

            # ================================================================
            # Convert BGR -> RGB and wrap as a mediapipe.Image
            # ================================================================

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            # ================================================================
            # MediaPipe processing (Tasks API, VIDEO mode)
            # ================================================================

            timestamp_ms = int((time.time() - start_time) * 1000)

            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1

            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            # ================================================================
            # Extract landmarks
            # ================================================================

            hand_landmarks_list = (
                result.hand_landmarks
                if result and result.hand_landmarks
                else []
            )

            detected_hands = len(
                hand_landmarks_list
            )

            pose_detected = False

            # ================================================================
            # Draw detected hands
            # ================================================================

            if hand_landmarks_list:

                for landmarks in hand_landmarks_list:

                    draw_hand_landmarks(
                        frame,
                        landmarks,
                    )

                # ============================================================
                # Check gesture
                # ============================================================

                pose_detected = check_domain_pose(
                    hand_landmarks_list
                )

            # ================================================================
            # Stabilize gesture detection
            # ================================================================

            pose_history.append(
                pose_detected
            )

            pose_stable = (
                sum(pose_history)
                >= POSE_HISTORY_REQUIRED
            )

            # ================================================================
            # State machine
            # ================================================================

            now = time.time()

            hold_progress = 0.0

            # ----------------------------------------------------------------
            # IDLE
            # ----------------------------------------------------------------

            if state == State.IDLE:

                if pose_stable:

                    state = State.HOLDING

                    hold_start = now

            # ----------------------------------------------------------------
            # HOLDING
            # ----------------------------------------------------------------

            elif state == State.HOLDING:

                if not pose_stable:

                    state = State.IDLE

                    hold_start = None

                else:

                    if hold_start is None:

                        hold_start = now

                    hold_progress = min(
                        (
                            now - hold_start
                        ) / HOLD_SECONDS,
                        1.0,
                    )

                    if hold_progress >= 1.0:

                        state = State.ACTIVE

                        active_start = now

            # ----------------------------------------------------------------
            # ACTIVE
            # ----------------------------------------------------------------

            elif state == State.ACTIVE:

                if active_start is None:

                    active_start = now

                elapsed = (
                    now - active_start
                )

                frame = render_domain_expansion(
                    frame,
                    elapsed,
                    DOMAIN_DURATION,
                    DOMAIN_NAMES[domain_idx],
                )

                if elapsed >= DOMAIN_DURATION:

                    state = State.COOLDOWN

                    cooldown_start = now

            # ----------------------------------------------------------------
            # COOLDOWN
            # ----------------------------------------------------------------

            elif state == State.COOLDOWN:

                if cooldown_start is None:

                    cooldown_start = now

                if (
                    now - cooldown_start
                    >= COOLDOWN_SECONDS
                ):

                    state = State.IDLE

                    cooldown_start = None

                    active_start = None

                    hold_start = None

                    pose_history.clear()

            # ================================================================
            # Draw tutorial UI
            # ================================================================

            frame = draw_tutorial(
                frame,
                state,
                hold_progress,
                DOMAIN_NAMES[domain_idx],
                detected_hands,
            )

            # ================================================================
            # Write this frame to the recording (includes all overlays,
            # exactly what's shown on screen)
            # ================================================================

            if video_writer is not None:
                video_writer.write(frame)

            # ================================================================
            # Show camera
            # ================================================================

            cv2.imshow(
                window_name,
                frame,
            )

            # ================================================================
            # Keyboard controls
            # ================================================================

            key = cv2.waitKey(1) & 0xFF

            # Quit
            if key == ord("q"):

                break

            # Next domain
            elif key == ord("n"):

                domain_idx = (
                    domain_idx + 1
                ) % len(DOMAIN_NAMES)

                print(
                    "Domain changed to:",
                    DOMAIN_NAMES[domain_idx],
                )

    except KeyboardInterrupt:

        print(
            "\n[INFO] Interrupted by user."
        )

    except Exception as exc:

        print(
            "\n[ERROR] Runtime error:"
        )

        print(
            type(exc).__name__,
            ":",
            exc,
        )

    finally:

        print(
            "\n[INFO] Shutting down..."
        )

        # Close MediaPipe
        try:

            if landmarker is not None:
                landmarker.close()

        except Exception:
            pass

        # Release camera
        cap.release()

        # Finalize the recording
        if video_writer is not None:
            video_writer.release()

            if video_output_path and os.path.exists(video_output_path):
                size_mb = os.path.getsize(video_output_path) / (1024 * 1024)
                print(
                    f"[OK] Session saved: {video_output_path} "
                    f"({size_mb:.1f} MB)"
                )

        # Close OpenCV windows
        cv2.destroyAllWindows()

        print(
            "[OK] Domain Expansion system stopped."
        )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()