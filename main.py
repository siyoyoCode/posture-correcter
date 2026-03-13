#!/usr/bin/env python3

import time
from gpiozero import Button
from rpi_capture import capture_image, send_to_cloud, speak_feedback, Camera, CAPTURE_DURATION, CAPTURE_FPS, BUTTON_PIN

# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    button   = Button(BUTTON_PIN, pull_up=True)
    interval = 1.0 / CAPTURE_FPS

    print("[RPi] Ready. Press the button to start an exercise session.")
    speak_feedback("Ready. Press the button to begin.")

    with Camera() as cam:
        while True:
            # ── Wait for button press ──────────────────────────────────────
            button.wait_for_press()
            print("[RPi] Button pressed — starting session!")
            speak_feedback("Starting. Go!")

            # ── Capture + send frames, collect last coach message ──────────
            last_message = None
            session_start = time.time()

            while time.time() - session_start < CAPTURE_DURATION:
                t0 = time.time()

                image_b64 = capture_image(cam)
                message   = send_to_cloud(image_b64)
                if message:
                    last_message = message   # keep the most recent coaching cue

                elapsed = time.time() - t0
                time.sleep(max(0, interval - elapsed))

            # ── Speak the last coach message from Claude ───────────────────
            print("[RPi] Session complete.")
            feedback = last_message or "Session complete. Good work!"
            speak_feedback(feedback)

            print("[RPi] Ready for next session.")
            time.sleep(1)


if __name__ == "__main__":
    main()