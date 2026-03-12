#!/usr/bin/env python3
"""
rpi_capture.py — Run this ON the Raspberry Pi.
Captures frames from the camera and streams them to the laptop server.
"""

import io
import time
import base64
import requests
import argparse

# ── Camera import (picamera2 for modern RPi OS, fallback to picamera) ─────────
try:
    from picamera2 import Picamera2
    USE_PICAMERA2 = True
except ImportError:
    import picamera
    USE_PICAMERA2 = False


# ──────────────────────────────────────────────────────────────────────────────
def capture_loop(server_url: str, fps: float, width: int, height: int):
    """Continuously capture frames and POST them to the laptop server."""

    print(f"[RPi] Connecting to server at {server_url}")
    print(f"[RPi] Resolution: {width}x{height}  |  Target FPS: {fps}")

    interval = 1.0 / fps

    if USE_PICAMERA2:
        cam = Picamera2()
        config = cam.create_still_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        cam.configure(config)
        cam.start()
        time.sleep(1)  # warm-up

        print("[RPi] Camera ready (picamera2). Starting stream…")
        try:
            while True:
                t0 = time.time()

                buf = io.BytesIO()
                cam.capture_file(buf, format="jpeg")
                frame_b64 = base64.b64encode(buf.getvalue()).decode()

                _send_frame(server_url, frame_b64)

                elapsed = time.time() - t0
                time.sleep(max(0, interval - elapsed))

        except KeyboardInterrupt:
            print("\n[RPi] Stopped by user.")
        finally:
            cam.stop()

    else:
        with picamera.PiCamera(resolution=(width, height)) as cam:
            cam.framerate = fps
            time.sleep(2)  # warm-up

            print("[RPi] Camera ready (picamera). Starting stream…")
            buf = io.BytesIO()
            try:
                for _ in cam.capture_continuous(buf, format="jpeg", use_video_port=True):
                    t0 = time.time()

                    buf.seek(0)
                    frame_b64 = base64.b64encode(buf.read()).decode()
                    _send_frame(server_url, frame_b64)

                    buf.seek(0)
                    buf.truncate()

                    elapsed = time.time() - t0
                    time.sleep(max(0, interval - elapsed))

            except KeyboardInterrupt:
                print("\n[RPi] Stopped by user.")


def _send_frame(server_url: str, frame_b64: str):
    try:
        resp = requests.post(
            f"{server_url}/frame",
            json={"image": frame_b64},
            timeout=5,
        )
        if resp.status_code == 200:
            data = resp.json()
            print(f"[RPi] LLM ▶ {data.get('response', '(no response)')}")
        else:
            print(f"[RPi] Server error {resp.status_code}")
    except requests.exceptions.ConnectionError:
        print("[RPi] Could not reach server — retrying next frame…")
    except Exception as e:
        print(f"[RPi] Error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RPi camera → LLM streamer")
    parser.add_argument("--server", default="http://192.168.1.100:5000",
                        help="Laptop server URL (default: http://192.168.1.100:5000)")
    parser.add_argument("--fps", type=float, default=0.5,
                        help="Frames per second to send (default: 0.5 = 1 frame every 2 s)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    capture_loop(args.server, args.fps, args.width, args.height)
