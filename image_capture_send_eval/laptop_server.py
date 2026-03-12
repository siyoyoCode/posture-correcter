#!/usr/bin/env python3
"""
laptop_server.py — Run this ON YOUR LAPTOP.
Receives JPEG frames from the RPi and forwards them to the pose-estimation backend.
"""

import base64
import io
import time

import requests
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND_URL  = "http://localhost:8000/analyze"   # your friend's server
EXERCISE     = "squat"                           # change if needed
SESSION_ID   = "rpi-session-1"                   # any unique string
MODE         = "normal"                          # adjust if the backend supports other modes

# ──────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

state = {
    "frame_counter": 0,
    "latest_frame_b64": None,
    "latest_response": None,
    "latest_response_frame": 0,
}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "frames_received": state["frame_counter"]})


@app.route("/snapshot", methods=["GET"])
def snapshot():
    """Return the latest JPEG frame so the dashboard can display it."""
    if not state["latest_frame_b64"]:
        return "No frame yet", 204
    jpeg_bytes = base64.b64decode(state["latest_frame_b64"])
    return send_file(io.BytesIO(jpeg_bytes), mimetype="image/jpeg")


@app.route("/latest_response", methods=["GET"])
def latest_response():
    return jsonify({
        "response": state["latest_response"],
        "frame": state["latest_response_frame"],
    })


@app.route("/frame", methods=["POST"])
def receive_frame():
    data = request.get_json(force=True)
    if not data or "image" not in data:
        return jsonify({"error": "missing 'image' field"}), 400

    frame_b64 = data["image"]
    state["frame_counter"] += 1
    state["latest_frame_b64"] = frame_b64
    fc = state["frame_counter"]

    print(f"[Server] Frame #{fc} received ({len(frame_b64) // 1024} KB) — forwarding to backend…")

    try:
        resp = requests.post(
            BACKEND_URL,
            json={
                "image_b64": frame_b64,
                "exercise":  EXERCISE,
                "session_id": SESSION_ID,
                "mode":      MODE,
            },
            timeout=10,
        )
        resp.raise_for_status()
        result = resp.json()
        state["latest_response"] = result
        state["latest_response_frame"] = fc
        print(f"[Server] Backend response: {result}")
        return jsonify({"response": result, "frame": fc})

    except requests.exceptions.ConnectionError:
        msg = f"Could not reach backend at {BACKEND_URL}"
        print(f"[Server] ERROR: {msg}")
        return jsonify({"error": msg}), 502
    except requests.exceptions.HTTPError as e:
        msg = f"Backend returned {e.response.status_code}: {e.response.text}"
        print(f"[Server] ERROR: {msg}")
        return jsonify({"error": msg}), 502
    except Exception as e:
        print(f"[Server] ERROR: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 55)
    print("  RPi Camera -> Pose Estimation Forwarder")
    print(f"  Backend: {BACKEND_URL}")
    print(f"  Exercise: {EXERCISE}  |  Session: {SESSION_ID}")
    print("  Listening on http://0.0.0.0:5000")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=False)