#!/usr/bin/env python3
"""
Flask AI detection + sliding-window alert producer.

- Keeps optimized worker for YOLO inference (as before)
- Maintains per-source sliding-window buffers
- When trigger condition met, enqueues alert task (non-blocking)
- Alert sender worker calls Node endpoint to create alert and upload snapshots
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import base64
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

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "Data/best.pt"
TARGET_FPS = 30
INFER_SIZE = 640
CONF_THRES = 0.25
SKIP_IF_STALE_MS = 2000
FRAME_QUEUE_MAXLEN = 2

# Sliding-window / trigger config
SAMPLE_INTERVAL_S = 0.5        # approximate sampling rate (FE -> Flask) in seconds
WINDOW_S = 3.0                 # window length in seconds for decision
WINDOW_SAMPLES = int((WINDOW_S / SAMPLE_INTERVAL_S) + 0.5)  # e.g., 6
FIRE_SMOKE_RATIO = 0.66        # fraction in window to trigger fire/smoke
FALL_CONSECUTIVE = WINDOW_SAMPLES  # require consecutive FALL for full WINDOW_S

# Alert/send config (Node endpoints)
NODE_CREATE_ALERT_URL = "http://localhost:8080/api/alerts/receive"
NODE_UPLOAD_SNAPSHOT_URL_TEMPLATE = "http://localhost:8080/api/alerts/{id}/snapshot"

# How many snapshots to collect (including initial) and interval between them
SNAPSHOTS_TO_COLLECT = 10
SNAPSHOT_INTERVAL_S = 1.0

# Rate-limit triggers per source (ms)
RATE_LIMIT_MS = 30 * 1000

# Logging flags
LOG_DETECTION = True
LOG_ALERT_FLOW = True

# -------------------------
if not os.path.exists(MODEL_PATH):
    print("FATAL: Model not found at", MODEL_PATH)
    sys.exit(1)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Shared state for inference worker
state_lock = threading.Lock()
frame_queue = deque(maxlen=FRAME_QUEUE_MAXLEN)
last_detection_result = {
    "frame_width": 0,
    "frame_height": 0,
    "detections": []
}
worker_running = True

# Per-source sliding-window state
# states[source] = {
#   buffer: deque(maxlen=WINDOW_SAMPLES) of { label, confidence, ts, snapshot_b64 },
#   last_trigger_at: timestamp_ms,
#   collecting_snapshots: bool,
#   pending_snapshots: list of base64 strings to be uploaded,
# }
states = {}

# Tasks queue: each task is dict with alert_type, content, snapshots(list base64), source, etc.
alert_task_queue = deque()
alert_worker_running = True

def normalize_label(raw_label: str) -> str:
    if raw_label is None:
        return "UNKNOWN"
    s = str(raw_label).upper().strip()
    s = s.replace("-", " ").replace("_", " ")
    s = " ".join(s.split())
    return s

# helper: send create alert to Node and upload snapshots
def send_alert_to_node_task(task):
    """
    task: {
      alert_type: 'fire'|'smoke'|'fall',
      content: str,
      source: str,
      snapshots: [base64,...]  # optional
    }
    """
    try:
        if LOG_ALERT_FLOW:
            print("[ALERT TASK] create alert:", task['alert_type'], "from", task.get('source'))

        # create alert at Node
        payload = {
            "userID": task.get("userID", 1),
            "alert_type": task["alert_type"],
            "content": task.get("content", ""),
            # do not include snapshots here (we will upload separately)
        }

        resp = None
        try:
            resp = requests.post(NODE_CREATE_ALERT_URL, json=payload, timeout=5)
        except Exception as e:
            if LOG_ALERT_FLOW:
                print("[ALERT TASK] failed to call Node create:", e)
            return

        if resp is None or resp.status_code >= 400:
            if LOG_ALERT_FLOW:
                print("[ALERT TASK] Node create failed:", resp.status_code if resp else None, resp.text if resp else "")
            return

        try:
            j = resp.json()
        except Exception:
            j = {}

        # Expect Node to return { alertID: <id> } or alert object containing alertID
        alertID = j.get("alertID") or j.get("id") or j.get("insertId") or j.get("alertId")
        if not alertID:
            # maybe the Node returns created record; try common patterns
            alertID = j.get("data", {}).get("alertID") if isinstance(j.get("data"), dict) else None

        if not alertID:
            if LOG_ALERT_FLOW:
                print("[ALERT TASK] Node didn't return alertID; response:", j)
            return

        if LOG_ALERT_FLOW:
            print("[ALERT TASK] alert created id=", alertID)

        # upload snapshots sequentially (non-blocking relative to inference)
        snapshots = task.get("snapshots") or []
        # limit to SNAPSHOTS_TO_COLLECT or available
        snapshots = snapshots[:SNAPSHOTS_TO_COLLECT]
        upload_url = NODE_UPLOAD_SNAPSHOT_URL_TEMPLATE.format(id=alertID)
        for idx, s_b64 in enumerate(snapshots):
            try:
                # send {"snapshot": "<base64>"} - Node handler should save
                r = requests.post(upload_url, json={"snapshot": s_b64}, timeout=5)
                if LOG_ALERT_FLOW:
                    print(f"[ALERT TASK] upload snapshot {idx+1}/{len(snapshots)} -> {r.status_code}")
            except Exception as e:
                if LOG_ALERT_FLOW:
                    print("[ALERT TASK] snapshot upload error:", e)
            # wait between uploads
            time.sleep(SNAPSHOT_INTERVAL_S)
    except Exception as e:
        print("[ALERT TASK] unexpected error:", e)
        traceback.print_exc()

def alert_sender_loop():
    while alert_worker_running:
        if not alert_task_queue:
            time.sleep(0.1)
            continue
        task = None
        try:
            task = alert_task_queue.popleft()
        except Exception:
            task = None
        if task:
            send_alert_to_node_task(task)

# start alert sender thread
alert_thread = threading.Thread(target=alert_sender_loop, daemon=True)
alert_thread.start()

# ----- Letterbox + helpers (copied from your optimized implementation) -----
def letterbox(image, new_size=INFER_SIZE, color=(114,114,114)):
    h0, w0 = image.shape[:2]
    if isinstance(new_size, int):
        new_w = new_h = new_size
    else:
        new_w, new_h = new_size
    ratio = min(new_w / w0, new_h / h0)
    new_unpad_w = int(round(w0 * ratio))
    new_unpad_h = int(round(h0 * ratio))
    dw = new_w - new_unpad_w
    dh = new_h - new_unpad_h
    dw /= 2
    dh /= 2
    resized = cv2.resize(image, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    resized = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return resized, ratio, (left, top)

def scale_coords_back(xyxy, ratio, pad, orig_shape):
    left, top = pad
    x1 = (xyxy[0] - left) / ratio
    y1 = (xyxy[1] - top) / ratio
    x2 = (xyxy[2] - left) / ratio
    y2 = (xyxy[3] - top) / ratio
    h0, w0 = orig_shape[:2]
    x1 = max(0, min(w0, int(round(x1))))
    x2 = max(0, min(w0, int(round(x2))))
    y1 = max(0, min(h0, int(round(y1))))
    y2 = max(0, min(h0, int(round(y2))))
    return [x1, y1, x2, y2]

# ----- Load model once -----
print("Loading model...", MODEL_PATH)
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print("Error loading model:", e)
    traceback.print_exc()
    raise

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
try:
    model.to(device)
    if device == "cuda":
        # best-effort half precision
        model.model.half() if hasattr(model, "model") else None
except Exception:
    pass

# warm-up
try:
    dummy = np.zeros((INFER_SIZE, INFER_SIZE, 3), dtype=np.uint8)
    with torch.no_grad():
        _ = model(dummy, conf=CONF_THRES, verbose=False)
    print("Model warm-up done")
except Exception:
    pass

# ----- Worker thread for inference -----
def worker_loop():
    global last_detection_result, worker_running, states
    target_period = 1.0 / TARGET_FPS if TARGET_FPS > 0 else 0.03

    ALERT_LABELS = {"FIRE", "SMOKE", "FALL"}

    while worker_running:
        start_t = time.time()
        frame_tuple = None
        with state_lock:
            if len(frame_queue) == 0:
                frame_tuple = None
            else:
                frame_tuple = frame_queue.pop()
                frame_queue.clear()
        if frame_tuple is None:
            time.sleep(min(0.01, target_period))
            continue

        frame, frame_ts = frame_tuple
        if (time.time() - frame_ts) * 1000.0 > SKIP_IF_STALE_MS:
            continue

        try:
            h0, w0 = frame.shape[:2]
            resized, ratio, pad = letterbox(frame, new_size=INFER_SIZE)
            with torch.no_grad():
                results = model(resized, conf=CONF_THRES, verbose=False)

            detections = []
            # parse results robustly
            for r in results:
                names = getattr(r, "names", {}) or {}
                boxes = getattr(r, "boxes", None)
                if boxes is None:
                    continue
                for box in boxes:
                    try:
                        raw_xy = box.xyxy[0] if hasattr(box.xyxy, "__len__") else box.xyxy
                        xy = raw_xy.cpu().numpy() if hasattr(raw_xy, "cpu") else np.array(raw_xy)
                        x1, y1, x2, y2 = [float(v) for v in xy.tolist()]
                    except Exception:
                        try:
                            xy2 = np.array(box.xyxy).astype(float).flatten()
                            x1, y1, x2, y2 = xy2.tolist()[:4]
                        except Exception:
                            continue

                    try:
                        conf = float(box.conf[0].cpu().numpy() if hasattr(box.conf[0], "cpu") else box.conf[0])
                    except Exception:
                        try:
                            conf = float(box.conf)
                        except Exception:
                            conf = 0.0

                    try:
                        cls_val = int(box.cls[0].cpu().numpy() if hasattr(box.cls[0], "cpu") else box.cls[0])
                    except Exception:
                        try:
                            cls_val = int(box.cls)
                        except Exception:
                            cls_val = -1

                    scaled_box = scale_coords_back([x1, y1, x2, y2], ratio, pad, (h0, w0))
                    label = normalize_label(names.get(cls_val, f"class_{cls_val}"))
                    detections.append({
                        "box": scaled_box,
                        "confidence": round(conf, 3),
                        "label": label
                    })

                    # Add event to sliding window state for this source
                    if label in ALERT_LABELS:
                        # encode snapshot (JPEG base64)
                        try:
                            ret, jpeg = cv2.imencode(".jpg", frame)
                            if ret:
                                snap_b64 = base64.b64encode(jpeg.tobytes()).decode()
                            else:
                                snap_b64 = None
                        except Exception as e:
                            snap_b64 = None
                            if LOG_DETECTION:
                                print("snapshot encode error:", e)

                        src = "camera1"
                        ts_ms = int(time.time() * 1000)
                        with state_lock:
                            st = states.get(src)
                            if st is None:
                                st = {
                                    "buffer": deque(maxlen=WINDOW_SAMPLES),
                                    "last_trigger_at": 0,
                                    "collecting": False,
                                }
                                states[src] = st
                            # push to buffer
                            st["buffer"].append({
                                "label": label,
                                "confidence": round(conf, 3),
                                "ts": ts_ms,
                                "snapshot": snap_b64
                            })
                            # evaluate rules immediately (non-blocking)
                            # check rate-limit
                            now_ms = ts_ms
                            if now_ms - st["last_trigger_at"] < RATE_LIMIT_MS:
                                # rate limited; skip evaluation
                                pass
                            else:
                                # Evaluate FIRE/SMOKE ratio in window
                                labels_list = list(item["label"] for item in st["buffer"])
                                fire_count = sum(1 for L in labels_list if L == "FIRE")
                                smoke_count = sum(1 for L in labels_list if L == "SMOKE")
                                min_needed = int((len(st["buffer"]) * FIRE_SMOKE_RATIO) + 0.9999)
                                # If buffer not full, we still evaluate based on current length
                                triggered = False
                                trigger_type = None

                                if len(st["buffer"]) >= 1 and (fire_count >= min_needed or smoke_count >= min_needed) and len(st["buffer"]) >= WINDOW_SAMPLES//2:
                                    # require at least some minimal samples (protect small bursts)
                                    trigger_type = "fire" if fire_count >= min_needed else "smoke"
                                    triggered = True

                                # For FALL require consecutive FALLs equal to FALL_CONSECUTIVE
                                if not triggered:
                                    # check last N entries
                                    last_k = list(st["buffer"])[-FALL_CONSECUTIVE:]
                                    if len(last_k) >= FALL_CONSECUTIVE and all(x["label"] == "FALL" for x in last_k) and len(last_k) >= WINDOW_SAMPLES//2:
                                        trigger_type = "fall"
                                        triggered = True

                                if triggered:
                                    # mark last_trigger_at
                                    st["last_trigger_at"] = now_ms
                                    st["collecting"] = True
                                    # collect snapshots: include entire buffer snapshots (most recent first) and reserve slots to collect future ones
                                    snaps = [item["snapshot"] for item in st["buffer"] if item.get("snapshot")]
                                    # pad with None as placeholders for future snapshots
                                    # we'll attempt to collect additional snapshots in background and append to this list
                                    # create alert task
                                    content = {
                                        "source": src,
                                        "buffer_len": len(st["buffer"]),
                                        "counts": {"fire": fire_count, "smoke": smoke_count},
                                        "ts": now_ms,
                                    }
                                    task = {
                                        "alert_type": trigger_type,
                                        "content": str(content),
                                        "source": src,
                                        "snapshots": snaps.copy(),  # initial
                                        "userID": 1
                                    }
                                    # enqueue alert task (non-blocking)
                                    alert_task_queue.append(task)
                                    if LOG_ALERT_FLOW:
                                        print("[ALERT] triggered:", trigger_type, "from", src, "buffer_len", len(st["buffer"]))
            # update last result
            with state_lock:
                last_detection_result = {
                    "frame_width": w0,
                    "frame_height": h0,
                    "detections": detections
                }

            if LOG_DETECTION and any(d['label'] in ["FALL","FIRE","SMOKE"] for d in detections):
                print("🔥 DETECT:", detections)

            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            gc.collect()

        except Exception as e:
            print("Worker inference error:", e)
            traceback.print_exc()
            time.sleep(0.01)

        elapsed = time.time() - start_t
        to_sleep = target_period - elapsed
        if to_sleep > 0:
            time.sleep(to_sleep)
        else:
            time.sleep(0.005)

# start worker
worker_thread = threading.Thread(target=worker_loop, daemon=True)
worker_thread.start()
print("Background worker started.")

# ----- Flask routes ----- #
@app.route("/")
def index():
    return "AI Detection Server (with sliding-window alert producer) running."

@app.route("/api/detect_frame", methods=["POST"])
def detect_frame_route():
    global last_detection_result
    # quick log: ensure POST arrives
    if LOG_DETECTION:
        print("🔥 Received POST /api/detect_frame")
    data = request.get_json(force=True, silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "Missing image field"}), 400
    raw = data["image"]
    try:
        if isinstance(raw, str) and raw.startswith("data:"):
            raw = raw.split(",",1)[1]
        img_bytes = base64.b64decode(raw)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Could not decode image"}), 400
        with state_lock:
            frame_queue.append((frame, time.time()))
        with state_lock:
            resp = dict(last_detection_result)
        return jsonify(resp)
    except Exception as e:
        print("Route error:", e)
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {e}"}), 500

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status":"ok", "device": device})

def run_server():
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        global worker_running, alert_worker_running
        worker_running = False
        alert_worker_running = False
        if worker_thread.is_alive():
            worker_thread.join(timeout=2)
        print("Exited.")

if __name__ == "__main__":
    run_server()
