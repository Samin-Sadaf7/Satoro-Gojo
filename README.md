# Domain Expansion — Gesture-Activated Webcam Effect

Feel like Satoru Gojo. This script opens your webcam, tracks your hands with
MediaPipe, and casts a full-screen "Domain Expansion" effect when you strike
the right pose and hold it.

## Setup

```bash
pip install -r requirements.txt
python domain_expansion.py
```

Requires Python 3.9+ and a working webcam. On first run your OS may ask for
camera permission — allow it.

## The gesture

Raise both hands in front of your chest, **fingers spread open, palms facing
the camera**, and bring your hands together (like sealing a technique). Hold
it steady for about a second — a progress bar at the top fills as your
"cursed energy" charges — and the domain triggers automatically.

You don't need finger-perfect interlacing (that's genuinely hard for a 2D
webcam to detect reliably); the detector looks for **both hands open, and
close together**, which reads as the same pose on camera and feels natural
to perform.

## What happens when it triggers

- The screen darkens into a void-like vignette
- A rotating sorcery-circle mandala expands from the center
- Pulsing energy rings radiate outward
- Your domain's name is stamped on screen (default: "UNLIMITED VOID")
- After ~4.5 seconds it fades out and cools down for a couple seconds
  before you can cast again

## Customizing

Open `domain_expansion.py` and edit the `CONFIG` section near the top:

- `HOLD_SECONDS` — how long you must hold the pose before it triggers
- `DOMAIN_DURATION` — how long the effect plays
- `COOLDOWN_SECONDS` — cooldown before re-casting
- `HANDS_TOGETHER_MAX_DIST` — how close your hands must be (lower = stricter)
- `DOMAIN_NAMES` — list of names to cycle through with the `n` key while
  the window is focused (add your own, e.g. "INFINITE CASTLE")

## Controls

- `q` — quit
- `n` — cycle to the next domain name

## Notes

- Works with any camera OpenCV can open — change `CAM_INDEX` at the top of
  the script if you have multiple cameras and the wrong one opens.
- Detection is intentionally forgiving (majority vote over the last few
  frames) so it doesn't flicker in and out — but if it's triggering too
  easily or not enough, tweak `HANDS_TOGETHER_MAX_DIST` and the finger-count
  threshold in `check_domain_pose`.
