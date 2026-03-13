#!/usr/bin/env python3
"""
rpi_capture.py — Run this ON the Raspberry Pi.

Flow:
  1. Wait for button press → start exercise session
  2. Capture frames for CAPTURE_DURATION seconds, POST each to /coach/analyze
  3. Each response contains a coach_message from Claude — speak the last one
  4. Repeat
"""

import io
import time
import base64
import requests
import subprocess

from gpiozero import Button

# ── Config ─────────────────────────────────────────────────────────────────────
BACKEND_URL      = "https://YOUR-APP.onrender.com"   # ← your friend's deployed URL
EXERCISE         = "squat"       # squat | pushup | lunge | bicep_curl | plank
SESSION_ID       = "rpi-session-1"
MODE             = "beginner"    # beginner | pro
TOTAL_SETS       = 3
REPS_PER_SET     = 10
CAPTURE_DURATION = 10            # seconds to capture after button press
CAPTURE_FPS      = 1             # frames per second to send
BUTTON_PIN       = 17            # GPIO pin the button is wired to

# ── Camera import ──────────────────────────────────────────────────────────────
try:
    from picamera2 import Picamera2
    USE_PICAMERA2 = True
except ImportError:
    import picamera
    USE_PICAMERA2 = False


# ── Functions ──────────────────────────────────────────────────────────────────

def capture_image(cam) -> str:
    """Capture a single JPEG frame and return it as a base64 string."""
    buf = io.BytesIO()
    if USE_PICAMERA2:
        cam.capture_file(buf, format="jpeg")
    else:
        cam.capture(buf, format="jpeg", use_video_port=True)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def send_to_cloud(image_b64: str) -> str | None:
    """
    POST a frame to /coach/analyze.
    Returns the coach_message string from Claude, or None on failure.
    """
    try:
        resp = requests.post(
            f"{BACKEND_URL}/coach/analyze",
            json={
                "image_b64":  image_b64,
                "exercise":   EXERCISE,
                "session_id": SESSION_ID,
                "mode":       MODE,
                "plan": {
                    "total_sets":   TOTAL_SETS,
                    "reps_per_set": REPS_PER_SET,
                },
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        coach_message = data.get("coach_message", "")
        analysis      = data.get("analysis", {})
        plan_state    = data.get("plan_state", {})

        print(f"[RPi] reps={analysis.get('rep_count')}  state={analysis.get('state')}  "
              f"set={plan_state.get('current_set')}/{plan_state.get('total_sets')}  "
              f"coach='{coach_message}'")

        return coach_message or None

    except requests.exceptions.ConnectionError:
        print("[RPi] Could not reach backend — skipping frame")
    except Exception as e:
        print(f"[RPi] send_to_cloud error: {e}")
    return None


def speak_feedback(text: str):
    """Speak text aloud through the USB speaker using espeak."""
    print(f"[RPi] Speaking: {text}")
    try:
        subprocess.run(["espeak", "-s", "150", "-v", "en", text], check=True)
    except FileNotFoundError:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[RPi] TTS error: {e}")


# ── Camera context helper ──────────────────────────────────────────────────────

class Camera:
    def __init__(self):
        self.cam = None

    def __enter__(self):
        if USE_PICAMERA2:
            self.cam = Picamera2()
            config = self.cam.create_still_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            self.cam.configure(config)
            self.cam.start()
            time.sleep(1)
        else:
            self.cam = picamera.PiCamera(resolution=(640, 480))
            self.cam.framerate = CAPTURE_FPS
            time.sleep(2)
        return self.cam

    def __exit__(self, *_):
        if USE_PICAMERA2:
            self.cam.stop()
        else:
            self.cam.close()
