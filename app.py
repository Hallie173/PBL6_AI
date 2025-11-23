#!/usr/bin/env python3
"""
Flask AI Server - PURE LOGIC MODE (Rewritten)
Behavior:
 - Per-frame aggregation: append exactly one label per processed frame ("FIRE" / "FALL" / "NONE")
 - Sliding window: WINDOW_SAMPLES frames (6) -> representing 3s (0.5s sample interval)
 - Trigger only when buffer is full (== 6) AND count >= 4 (4/6)
 - Prevent multi-box -> multi-append issue by aggregating boxes per frame
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import numpy as np
import cv2
import torch
import os
import threading
import time
from collections import deque
import gc
import sys
import traceback
import requests
import json
import base64

# ----------------------------
# 1. CONFIG
# ----------------------------
MODEL_PATH = "Data/best.pt"
TARGET_FPS = 30
INFER_SIZE = 640
CONF_THRES = 0.25
SKIP_IF_STALE_MS = 2000
FRAME_QUEUE_MAXLEN = 2

# Sliding window config
SAMPLE_INTERVAL_S = 0.5
WINDOW_S = 3.0
WINDOW_SAMPLES = int((WINDOW_S / SAMPLE_INTERVAL_S) + 0.5)  # expected 6
REQUIRED_MATCH = 4  # require 4 out of 6

# Endpoint of Node.js (JSON-only)
NODE_CREATE_ALERT_URL = "http://localhost:8080/api/receive"

RATE_LIMIT_MS = 30 * 1000  # 30 sec cooldown per source after trigger

LOG_DETECTION = True
LOG_ALERT_FLOW = True

# sanity
if WINDOW_SAMPLES <= 0:
    print("FATAL: WINDOW_SAMPLES invalid")
    sys.exit(1)

# check model exists
if not os.path.exists(MODEL_PATH):
    print(f"FATAL: Model not found at {MODEL_PATH}")
    sys.exit(1)

# ----------------------------
# 2. APP & GLOBALS
# ----------------------------
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

state_lock = threading.Lock()
frame_queue = deque(maxlen=FRAME_QUEUE_MAXLEN)
last_detection_result = {"frame_width": 0, "frame_height": 0, "detections": []}
worker_running = True
alert_worker_running = True

# state per source: buffer (deque of labels), last_trigger_at (ms)
states = {}
alert_task_queue = deque()

# ----------------------------
# 3. UTIL
# ----------------------------
def normalize_label(raw_label: str) -> str:
    if raw_label is None: 
        return "UNKNOWN"
    return " ".join(str(raw_label).upper().replace("-", " ").replace("_", " ").split())

# ----------------------------
# 4. ALERT SENDER
# ----------------------------
def send_alert_to_node_task(task):
    try:
        if LOG_ALERT_FLOW:
            print(f"[ALERT SENDER] Sending '{task['alert_type']}' signal for UserID: {task.get('userID')}")
        payload = {
            "userID": task.get("userID", 1),
            "alert_type": task.get("alert_type"),
            "content": task.get("content", ""),
            "source": task.get("source"),
            "timestamp": int(time.time() * 1000)
        }
        resp = requests.post(NODE_CREATE_ALERT_URL, json=payload, timeout=5)
        if LOG_ALERT_FLOW:
            if resp.status_code < 400:
                print(f"[ALERT SENDER] Success. Node responded: {resp.status_code}")
            else:
                print(f"[ALERT SENDER] Failed. Node responded: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"[ALERT SENDER] Error: {e}")

def alert_sender_loop():
    while alert_worker_running:
        if not alert_task_queue:
            time.sleep(0.1)
            continue
        try:
            task = alert_task_queue.popleft()
            if task:
                send_alert_to_node_task(task)
        except Exception:
            pass

t_alert = threading.Thread(target=alert_sender_loop, daemon=True)
t_alert.start()

# ----------------------------
# 5. PREPROCESS HELPERS
# ----------------------------
def letterbox(image, new_size=INFER_SIZE, color=(114,114,114)):
    h0, w0 = image.shape[:2]
    new_w, new_h = (new_size, new_size) if isinstance(new_size, int) else new_size
    ratio = min(new_w / w0, new_h / h0)
    new_unpad_w, new_unpad_h = int(round(w0 * ratio)), int(round(h0 * ratio))
    dw, dh = (new_w - new_unpad_w) / 2, (new_h - new_unpad_h) / 2
    resized = cv2.resize(image, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color), ratio, (left, top)

def scale_coords_back(xyxy, ratio, pad, orig_shape):
    left, top = pad
    h0, w0 = orig_shape[:2]
    x1 = max(0, min(w0, int(round((xyxy[0] - left) / ratio))))
    y1 = max(0, min(h0, int(round((xyxy[1] - top) / ratio))))
    x2 = max(0, min(w0, int(round((xyxy[2] - left) / ratio))))
    y2 = max(0, min(h0, int(round((xyxy[3] - top) / ratio))))
    return [x1, y1, x2, y2]

# ----------------------------
# 6. LOAD MODEL
# ----------------------------
print(f"Loading YOLO model: {MODEL_PATH} ...")
try:
    model = YOLO(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded on: {device}")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# ----------------------------
# 7. WORKER LOOP (REWRITTEN)
# ----------------------------
def worker_loop():
    global last_detection_result, worker_running, states
    target_period = 1.0 / TARGET_FPS if TARGET_FPS > 0 else 0.03
    ALERT_LABELS = {"FIRE", "FALL"}

    while worker_running:
        start_t = time.time()

        # Pop latest frame, drop old
        frame_tuple = None
        with state_lock:
            if frame_queue:
                frame_tuple = frame_queue.pop()
                frame_queue.clear()

        if not frame_tuple:
            time.sleep(0.01)
            continue

        frame, frame_ts, user_id = frame_tuple

        # Skip stale
        if (time.time() - frame_ts) * 1000.0 > SKIP_IF_STALE_MS:
            continue

        try:
            h0, w0 = frame.shape[:2]
            resized, ratio, pad = letterbox(frame, new_size=INFER_SIZE)
            with torch.no_grad():
                results = model(resized, conf=CONF_THRES, verbose=False)

            # Build detection list (for FE) and compute per-frame aggregated label
            detections = []
            frame_has_fire = False
            frame_has_fall = False

            for r in results:
                boxes = getattr(r, "boxes", None)
                if boxes is None:
                    continue

                for box in boxes:
                    xy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], "cpu") else box.xyxy[0]
                    x1, y1, x2, y2 = map(float, xy)

                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    label = normalize_label(r.names.get(cls_id, ""))

                    scaled_box = scale_coords_back([x1, y1, x2, y2], ratio, pad, (h0, w0))

                    detections.append({
                        "box": scaled_box,
                        "confidence": round(conf, 3),
                        "label": label
                    })

                    # aggregate per-frame
                    if label == "FIRE":
                        frame_has_fire = True
                    elif label == "FALL":
                        frame_has_fall = True

            # Decide single frame label: prioritize FIRE over FALL (you can change priority)
            if frame_has_fire:
                frame_label = "FIRE"
            elif frame_has_fall:
                frame_label = "FALL"
            else:
                frame_label = "NONE"

            # Update sliding window buffer (one append per frame)
            src = "camera_main"
            ts_ms = int(time.time() * 1000)

            with state_lock:
                if src not in states:
                    states[src] = {"buffer": deque(maxlen=WINDOW_SAMPLES), "last_trigger_at": 0}
                st = states[src]
                # append single label per frame (FIRE/FALL/NONE)
                st["buffer"].append({"label": frame_label, "ts": ts_ms})

                # only evaluate when cooldown passed
                buffer_len = len(st["buffer"])
                if buffer_len == WINDOW_SAMPLES:
                    labels_list = [x["label"] for x in st["buffer"]]
                    fire_count = labels_list.count("FIRE")
                    fall_count = labels_list.count("FALL")

                    trigger_type = None
                    if fire_count >= REQUIRED_MATCH:
                        trigger_type = "FIRE"
                    elif fall_count >= REQUIRED_MATCH:
                        trigger_type = "FALL"

                    # Only send alert if cooldown passed
                    if trigger_type:
                        if ts_ms - st["last_trigger_at"] > RATE_LIMIT_MS:
                            # Allowed → send alert
                            st["last_trigger_at"] = ts_ms
                            content_data = {
                                "trigger_logic": f"{trigger_type} 4/6 frames in 3s",
                                "stats": {"fire": fire_count, "fall": fall_count, "total": buffer_len},
                                "timestamp": ts_ms
                            }
                            task = {
                                "userID": user_id,
                                "alert_type": trigger_type,
                                "content": json.dumps(content_data),
                                "source": src
                            }
                            alert_task_queue.append(task)
                            if LOG_ALERT_FLOW:
                                print(f"⚠️ TRIGGERED ({trigger_type}) after cooldown")
                        else:
                            if LOG_ALERT_FLOW:
                                print(f"⏳ Cooldown active ({trigger_type}). {RATE_LIMIT_MS - (ts_ms - st['last_trigger_at'])}ms remaining.")

            # publish last_detection_result for FE
            with state_lock:
                last_detection_result = {
                    "frame_width": w0,
                    "frame_height": h0,
                    "detections": detections
                }

            if LOG_DETECTION and any(d['label'] in ALERT_LABELS for d in detections):
                alert_dets = [d for d in detections if d['label'] in ALERT_LABELS]
                print(f"🔥 Detect Frame: {len(alert_dets)} alert objects; frame_label={frame_label}")

            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"Worker Error: {e}")
            traceback.print_exc()

        # maintain target fps
        elapsed = time.time() - start_t
        time.sleep(max(0.005, target_period - elapsed))

t_worker = threading.Thread(target=worker_loop, daemon=True)
t_worker.start()
print("✅ Background AI Worker started (Pure Logic Mode)")

# ----------------------------
# 8. ROUTES
# ----------------------------
@app.route("/")
def index():
    return "Flask AI Server (Pure Logic) Running."

@app.route("/api/detect_frame", methods=["POST"])
def detect_frame_route():
    global last_detection_result
    try:
        data = request.get_json(force=True, silent=True)
        if not data or "image" not in data:
            return jsonify({"error": "No image data"}), 400

        user_id = int(data.get("userID", 1))

        raw_img = data["image"]
        if "," in raw_img:
            raw_img = raw_img.split(",")[1]

        np_arr = np.frombuffer(base64.b64decode(raw_img), dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Image decode failed"}), 400

        with state_lock:
            frame_queue.append((frame, time.time(), user_id))
            resp = dict(last_detection_result)

        return jsonify(resp)

    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "mode": "pure_logic"})

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    except KeyboardInterrupt:
        worker_running = False
        alert_worker_running = False
        print("Server shutting down...")
