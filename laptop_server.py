#!/usr/bin/env python3
"""
laptop_server.py — Run this ON YOUR LAPTOP.
Receives JPEG frames from the RPi, forwards them to Claude, returns the response.
Also serves /snapshot and /latest_response for the live dashboard.
"""

import base64
import io
import os
import time

import anthropic
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

# ── Config ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")  # or paste key here
MODEL = "claude-opus-4-6"
MAX_TOKENS = 512

# Tweak this system prompt for your hackathon use-case!
SYSTEM_PROMPT = """You are a real-time visual assistant analysing a live camera feed from a 
Raspberry Pi. For each frame you receive, give a concise (1-3 sentence) description of what 
you see, highlighting anything notable, unusual, or that has changed. Be direct and factual."""

# Minimum seconds between LLM calls (avoid hammering the API)
MIN_LLM_INTERVAL = 2.0

# ──────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

state = {
    "frame_counter": 0,
    "last_llm_call": 0.0,
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

    print(f"[Server] Frame #{fc} received ({len(frame_b64) // 1024} KB)")

    now = time.time()
    if now - state["last_llm_call"] < MIN_LLM_INTERVAL:
        return jsonify({"response": "(rate-limited)", "frame": fc})

    state["last_llm_call"] = now

    try:
        response_text = call_claude(frame_b64)
        state["latest_response"] = response_text
        state["latest_response_frame"] = fc
        print(f"[Server] Claude >> {response_text}")
        return jsonify({"response": response_text, "frame": fc})
    except Exception as e:
        print(f"[Server] Claude error: {e}")
        return jsonify({"error": str(e)}), 500


def call_claude(frame_b64: str) -> str:
    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": frame_b64,
                    },
                },
                {"type": "text", "text": "What do you see in this camera frame? Be concise."},
            ],
        }],
    )
    return message.content[0].text


if __name__ == "__main__":
    if not ANTHROPIC_API_KEY:
        print("WARNING: ANTHROPIC_API_KEY not set. Export it or paste it into this file.")
    print("=" * 55)
    print("  RPi Camera -> Claude Vision Server")
    print("  Listening on http://0.0.0.0:5000")
    print("  Open dashboard.html in your browser")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=False)
