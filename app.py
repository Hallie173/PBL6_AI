#!/usr/bin/env python3
"""
Flask AI Server - PURE LOGIC MODE (FIXED)
Fixes:
 - Fix lỗi trigger ngay lập tức do buffer cũ còn lưu.
 - Tách biệt trạng thái theo UserID.
 - Reset buffer sau khi trigger alert.
 - Hiển thị log chi tiết buffer để debug.
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
CONF_THRES = 0.45  # Tăng ngưỡng tin cậy lên chút để giảm nhiễu
SKIP_IF_STALE_MS = 2000
FRAME_QUEUE_MAXLEN = 5

# Sliding window config
# FE gửi 500ms/frame -> Window 6 frames = 3 giây dữ liệu
WINDOW_SAMPLES = 6 
REQUIRED_MATCH = 4  # Cần 4/6 frame là FIRE/FALL mới báo động

# Endpoint of Node.js
RATE_LIMIT_MS = 30 * 1000  # 30 sec cooldown

LOG_DETECTION = True
LOG_BUFFER_STATE = True # Bật cái này để xem logic đếm

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

# states = { "user_1": { "buffer": deque(...), "last_trigger_at": 0 } }
states = {}
alert_task_queue = deque()

# ----------------------------
# 3. UTIL
# ----------------------------
def normalize_label(raw_label: str) -> str:
    if raw_label is None: return "NONE"
    return " ".join(str(raw_label).upper().replace("-", " ").replace("_", " ").split())

# ----------------------------
# 5. PREPROCESS
# ----------------------------
def letterbox(image, new_size=INFER_SIZE, color=(114,114,114)):
    h0, w0 = image.shape[:2]
    ratio = min(new_size / w0, new_size / h0)
    new_unpad = (int(round(w0 * ratio)), int(round(h0 * ratio)))
    dw, dh = (new_size - new_unpad[0]) / 2, (new_size - new_unpad[1]) / 2
    resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
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
# 6. MODEL LOAD
# ----------------------------
print(f"Loading YOLO model...")
model = YOLO(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model ready on {device}. CONF_THRES={CONF_THRES}")

# ----------------------------
# 7. WORKER LOOP (LOGIC CORE)
# ----------------------------
def worker_loop():
    global last_detection_result
    ALERT_LABELS = {"FIRE", "FALL"}

    while worker_running:
        frame_tuple = None
        with state_lock:
            if frame_queue:
                frame_tuple = frame_queue.pop() # Lấy frame mới nhất
                frame_queue.clear() # Xóa frame cũ để tránh lag

        if not frame_tuple:
            time.sleep(0.01)
            continue

        frame, frame_ts, user_id = frame_tuple

        # Bỏ qua frame quá cũ (> 2 giây)
        if (time.time() - frame_ts) * 1000.0 > SKIP_IF_STALE_MS:
            continue

        try:
            h0, w0 = frame.shape[:2]
            resized, ratio, pad = letterbox(frame, new_size=INFER_SIZE)
            
            with torch.no_grad():
                results = model(resized, conf=CONF_THRES, verbose=False)

            detections = []
            frame_label = "NONE" # Mặc định là không có gì
            has_fire = False
            has_fall = False

            confirmed_alert = None

            for r in results:
                boxes = getattr(r, "boxes", None)
                if boxes is None: continue

                for box in boxes:
                    xy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].item())
                    cls_id = int(box.cls[0].item())
                    label = normalize_label(r.names.get(cls_id, ""))
                    
                    # Scale box về kích thước gốc
                    final_box = scale_coords_back(xy, ratio, pad, (h0, w0))
                    
                    detections.append({
                        "box": final_box,
                        "confidence": round(conf, 3),
                        "label": label
                    })

                    if label == "FIRE": has_fire = True
                    if label == "FALL": has_fall = True

            # Xác định nhãn duy nhất cho frame này (Ưu tiên FIRE > FALL)
            if has_fire: frame_label = "FIRE"
            elif has_fall: frame_label = "FALL"

            # --- LOGIC SLIDING WINDOW ---
            ts_ms = int(time.time() * 1000)
            
            # TẠO KHÓA RIÊNG CHO TỪNG USER ĐỂ KHÔNG BỊ TRỘN DỮ LIỆU
            state_key = f"user_{user_id}"

            with state_lock:
                # Khởi tạo state nếu user mới
                if state_key not in states:
                    states[state_key] = {
                        "buffer": deque(maxlen=WINDOW_SAMPLES), 
                        "last_trigger_at": 0
                    }
                
                st = states[state_key]
                st["buffer"].append(frame_label) # Thêm nhãn frame vào đuôi

                # LOG DEBUG QUAN TRỌNG: Xem buffer đang chứa gì
                if LOG_BUFFER_STATE:
                    print(f"[User {user_id}] Buffer: {list(st['buffer'])} | Label: {frame_label}")

                # Chỉ kiểm tra khi buffer ĐÃ ĐẦY (đủ 6 frame)
                if len(st["buffer"]) == WINDOW_SAMPLES:
                    fire_count = st["buffer"].count("FIRE")
                    fall_count = st["buffer"].count("FALL")
                    
                    trigger_type = None
                    if fire_count >= REQUIRED_MATCH: trigger_type = "FIRE"
                    elif fall_count >= REQUIRED_MATCH: trigger_type = "FALL"

                    

                    if trigger_type:
                        # Kiểm tra cooldown
                        time_since_last = ts_ms - st["last_trigger_at"]
                        if time_since_last > RATE_LIMIT_MS:
                            # TRIGGER ALERT
                            st["last_trigger_at"] = ts_ms
                            
                            content = {
                                "logic": f"{trigger_type} detected in {fire_count if trigger_type=='FIRE' else fall_count}/6 frames",
                                "stats": {"fire": fire_count, "fall": fall_count}
                            }
                            
                            print(f"⚠️ TRIGGERED ALERT: {trigger_type}")

                            # QUAN TRỌNG: XÓA BUFFER SAU KHI TRIGGER
                            # Để tránh việc frame tiếp theo lại kích hoạt alert ngay lập tức
                            
                            confirmed_alert = trigger_type
                            
                            st["buffer"].clear()
                            print(f"[User {user_id}] Buffer cleared after alert.")

                        else:
                            print(f"⏳ Cooldown active for {user_id}. Wait {int((RATE_LIMIT_MS - time_since_last)/1000)}s")

            # Update kết quả cho API trả về
            with state_lock:
                last_detection_result = {
                    "frame_width": w0,
                    "frame_height": h0,
                    "detections": detections,
                    "alert_trigger": confirmed_alert
                }

            if device == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Worker Error: {e}")
            traceback.print_exc()

t_worker = threading.Thread(target=worker_loop, daemon=True)
t_worker.start()
print("✅ AI Worker Started (Fixed Logic)")

# ----------------------------
# 8. ROUTES
# ----------------------------
@app.route("/api/detect_frame", methods=["POST"])
def detect_frame_route():
    global last_detection_result
    try:
        data = request.get_json(force=True, silent=True)
        if not data or "image" not in data:
            return jsonify({"error": "No image data"}), 400

        user_id = int(data.get("userID", 1))
        raw_img = data["image"].split(",")[1] if "," in data["image"] else data["image"]
        
        np_arr = np.frombuffer(base64.b64decode(raw_img), dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None: return jsonify({"error": "Decode fail"}), 400

        with state_lock:
            frame_queue.append((frame, time.time(), user_id))
            resp = dict(last_detection_result)

        return jsonify(resp)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)