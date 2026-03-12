#!/usr/bin/env python3
"""
rpi_capture.py — Run this ON the Raspberry Pi.

Flow:
  1. Wait for button press → start exercise session
  2. Capture frames for CAPTURE_DURATION seconds, POST each to cloud backend
  3. After session ends, fetch latest feedback from backend
  4. Speak feedback through USB speaker
  5. Repeat
"""

import io
import time
import base64
import requests
import subprocess

from gpiozero import Button

# ── Config ─────────────────────────────────────────────────────────────────────
BACKEND_URL      = "https://YOUR-APP.onrender.com"   # ← paste your friend's cloud URL here
EXERCISE         = "squat"
SESSION_ID       = "rpi-session-1"
MODE             = "normal"
CAPTURE_DURATION = 10        # seconds to capture after button press
CAPTURE_FPS      = 1         # frames per second to send (1 is plenty for pose estimation)
BUTTON_PIN       = 17        # GPIO pin the button is wired to

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


def send_to_cloud(image_b64: str):
    """POST a frame to the cloud backend for pose analysis."""
    try:
        resp = requests.post(
            f"{BACKEND_URL}/analyze",
            json={
                "image_b64":  image_b64,
                "exercise":   EXERCISE,
                "session_id": SESSION_ID,
                "mode":       MODE,
            },
            timeout=10,
        )
        resp.raise_for_status()
        print(f"[RPi] Frame sent OK — {resp.json()}")
    except requests.exceptions.ConnectionError:
        print("[RPi] Could not reach backend — skipping frame")
    except Exception as e:
        print(f"[RPi] send_to_cloud error: {e}")


def get_feedback() -> str:
    """Fetch the latest feedback text from the backend."""
    try:
        resp = requests.get(f"{BACKEND_URL}/feedback", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # Try common field names — adjust if your friend uses a different key
        return (
            data.get("coach_message")
            or data.get("feedback")
            or data.get("message")
            or str(data)
        )
    except Exception as e:
        print(f"[RPi] get_feedback error: {e}")
        return "Could not retrieve feedback."


def speak_feedback(text: str):
    """Speak text aloud through the USB speaker using espeak."""
    print(f"[RPi] Speaking: {text}")
    try:
        subprocess.run(
            ["espeak", "-s", "150", "-v", "en", text],
            check=True,
        )
    except FileNotFoundError:
        # Fallback: use pyttsx3 if espeak isn't installed
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[RPi] TTS error: {e}")


# ── Camera context helper ──────────────────────────────────────────────────────

class Camera:
    """Simple context manager that works for both picamera2 and picamera."""
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
            time.sleep(1)  # warm-up
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


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    button = Button(BUTTON_PIN, pull_up=True)
    interval = 1.0 / CAPTURE_FPS

    print("[RPi] Ready. Press the button to start an exercise session.")
    speak_feedback("Ready. Press the button to begin.")

    with Camera() as cam:
        while True:
            # ── Wait for button press ──────────────────────────────────────
            button.wait_for_press()
            print("[RPi] Button pressed — starting session!")
            speak_feedback("Starting. Go!")

            # ── Capture + send frames for CAPTURE_DURATION seconds ────────
            session_start = time.time()
            while time.time() - session_start < CAPTURE_DURATION:
                t0 = time.time()

                image_b64 = capture_image(cam)
                send_to_cloud(image_b64)

                elapsed = time.time() - t0
                time.sleep(max(0, interval - elapsed))

            print("[RPi] Session complete — fetching feedback…")
            speak_feedback("Session complete. Getting your feedback.")

            # ── Fetch and speak feedback ───────────────────────────────────
            # Small delay to let the backend finish processing the last frame
            time.sleep(1.5)
            feedback = get_feedback()
            speak_feedback(feedback)

            print("[RPi] Ready for next session.")
            time.sleep(1)


if __name__ == "__main__":
    main()