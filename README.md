# RPi Camera → Claude Vision — Hackathon Starter

```
[Raspberry Pi]  →  (HTTP POST /frame)  →  [Laptop Server]  →  [Claude API]
     rpi_capture.py                         laptop_server.py
                                                  ↕
                                           dashboard.html  (browser)
```

---

## Files

| File | Where to run |
|------|-------------|
| `laptop_server.py` | Your laptop |
| `rpi_capture.py` | Raspberry Pi |
| `dashboard.html` | Browser (open file directly) |

---

## 1 — Laptop setup

```bash
pip install flask flask-cors anthropic

export ANTHROPIC_API_KEY="sk-ant-..."   # Linux/macOS
# set ANTHROPIC_API_KEY=sk-ant-...      # Windows

python laptop_server.py
```

The server listens on **port 5000** on all interfaces.

---

## 2 — Find your laptop's IP

```bash
# macOS / Linux
ipconfig getifaddr en0   # Wi-Fi
# or
hostname -I

# Windows
ipconfig
```

You'll get something like `192.168.1.42`.

---

## 3 — Raspberry Pi setup

Copy `rpi_capture.py` to the RPi (USB, scp, or shared folder):

```bash
scp rpi_capture.py pi@raspberrypi.local:~/
```

Install deps on the RPi:

```bash
sudo apt install -y python3-picamera2   # modern OS (Bullseye+)
pip install requests
```

Run the capture script, pointing at your laptop:

```bash
python rpi_capture.py --server http://192.168.1.42:5000 --fps 0.5
```

**Flags**

| Flag | Default | Description |
|------|---------|-------------|
| `--server` | `http://192.168.1.100:5000` | Laptop IP |
| `--fps` | `0.5` | Frames/sec sent (0.5 = 1 every 2 s) |
| `--width` | `640` | Capture width px |
| `--height` | `480` | Capture height px |

---

## 4 — Open the dashboard

Open `dashboard.html` in your browser, set the server URL to
`http://localhost:5000` (or your laptop's IP if viewing from another device),
then click **▶ Start**.

---

## Customising the LLM behaviour

Edit the `SYSTEM_PROMPT` at the top of `laptop_server.py`:

```python
SYSTEM_PROMPT = """You are a quality-control inspector on a factory line.
Flag any items that appear damaged, misaligned, or out of place."""
```

You can also tune:
- `MIN_LLM_INTERVAL` — seconds between API calls (default `2.0`)
- `MAX_TOKENS` — max response length
- `MODEL` — Claude model to use

---

## Troubleshooting

**`ConnectionError` on the RPi**  
→ Make sure both devices are on the same Wi-Fi. Check the IP address.  
→ Temporarily disable your laptop's firewall for port 5000.

**`No module named picamera2`**  
→ Run `sudo apt install -y python3-picamera2` on the RPi.

**Camera not found**  
→ Run `libcamera-hello` on the RPi to verify the camera is detected.  
→ In `raspi-config` → Interface Options → Legacy Camera — try toggling.
