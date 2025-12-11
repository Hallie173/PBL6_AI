"""
Flask AI Server - FIXED NMS VERSION
Backend: TensorFlow/Keras
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import os
import threading
import time
from collections import deque
import sys
import traceback
import base64
import tensorflow as tf
from tensorflow import keras

# ----------------------------
# 1. CONFIG
# ----------------------------
MODEL_PATH = "Data/best_yolo_resnet.keras" # Đảm bảo đường dẫn đúng

INFER_SIZE = 320
CLASS_THRESHOLDS = {
    0: 0.15,  # Fire: Cần nhạy một chút (0.1)
    1: 0.2,  # Fall: Cần chắc chắn mới báo (0.6) để tránh nhầm với nằm ngủ
    2: 0.1   # Not-fall: Chỉ cần hơi giống là được (0.1)
}
DEFAULT_THRES = 0.1
IOU_THRES = 0.3   # Mức chuẩn: Box chồng nhau >30% sẽ bị xóa
FRAME_QUEUE_MAXLEN = 5

WINDOW_SAMPLES = 6
REQUIRED_MATCH = 4
RATE_LIMIT_MS = 30 * 1000

CLASS_NAMES = {0: "FIRE", 1: "FALL", 2: "NOT FALL"}

LOG_BUFFER_STATE = True

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
states = {}

# ----------------------------
# 3. MODEL LOAD
# ----------------------------
print(f"Loading Keras model from {MODEL_PATH}...")
try:
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully!")
    # Warmup
    dummy_input = np.zeros((1, INFER_SIZE, INFER_SIZE, 3), dtype=np.float32)
    model.predict(dummy_input, verbose=0)
    print("✅ Model warmup complete.")
except Exception as e:
    print(f"FATAL ERROR loading model: {e}")
    sys.exit(1)

# ----------------------------
# 4. UTILS & DECODER (UPDATED WITH TF NMS)
# ----------------------------

def letterbox_resize(image, target_size=INFER_SIZE, color=(128,128,128)):
    h0, w0 = image.shape[:2]
    scale = min(target_size/w0, target_size/h0)
    nw, nh = int(w0*scale), int(h0*scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = np.full((target_size, target_size, 3), color, dtype=np.uint8)
    dx, dy = (target_size - nw)//2, (target_size - nh)//2
    padded[dy:dy+nh, dx:dx+nw] = resized
    return padded, scale, (dx, dy)

def decode_predictions(preds, orig_shape, scale_params, iou_thres=IOU_THRES):
    """
    Giải mã output với ngưỡng Confidence riêng cho từng Class
    """
    output = preds[0] # (20, 20, 8)
    grid_h, grid_w = output.shape[0], output.shape[1]
    
    # Flatten
    output_flat = output.reshape(-1, output.shape[-1])
    
    # Tách thành phần
    objectness = output_flat[:, 0]
    box_coords = output_flat[:, 1:5]
    class_probs = output_flat[:, 5:]
    
    # --- 1. TÍNH TOÁN SCORE VÀ CLASS ---
    class_ids = np.argmax(class_probs, axis=1)
    class_max_scores = np.max(class_probs, axis=1)
    final_scores = objectness * class_max_scores
    
    # --- 2. LỌC THEO NGƯỠNG RIÊNG (DYNAMIC THRESHOLD) ---
    # Tạo mảng ngưỡng tương ứng với từng box dựa trên class_id của nó
    # List comprehension này sẽ tạo ra một mảng ngưỡng có độ dài bằng số lượng box (400)
    dynamic_thresholds = np.array([CLASS_THRESHOLDS.get(cid, DEFAULT_THRES) for cid in class_ids])
    
    # So sánh Score của từng box với ngưỡng của chính nó
    mask = final_scores > dynamic_thresholds
    
    if not np.any(mask):
        return []
    
    # Lấy dữ liệu đã lọc
    sel_boxes_norm = box_coords[mask]
    sel_scores = final_scores[mask]
    sel_classes = class_ids[mask]
    
    # Lấy lại index grid
    indices = np.where(mask)[0]
    grid_rows = indices // grid_w
    grid_cols = indices % grid_w
    
    # --- 3. DECODE TỌA ĐỘ ---
    tf_boxes = []
    decoded_boxes_info = [] 
    
    for i in range(len(sel_boxes_norm)):
        bx, by, bw, bh = sel_boxes_norm[i]
        c, r = grid_cols[i], grid_rows[i]
        
        # Grid Offset -> Normalized Global
        cx = (bx + c) / grid_w
        cy = (by + r) / grid_h
        w = bw
        h = bh 
        
        # Convert Center -> Corner [y1, x1, y2, x2]
        y1 = cy - h/2
        x1 = cx - w/2
        y2 = cy + h/2
        x2 = cx + w/2
        
        tf_boxes.append([y1, x1, y2, x2])
        decoded_boxes_info.append({
            "cx": cx, "cy": cy, "w": w, "h": h,
            "score": float(sel_scores[i]),
            "class_id": int(sel_classes[i])
        })
        
    if not tf_boxes: return []
        
    tf_boxes = np.array(tf_boxes, dtype=np.float32)
    tf_scores = np.array([d["score"] for d in decoded_boxes_info], dtype=np.float32)
    
    # --- 4. CHẠY NMS ---
    # Lưu ý: score_threshold ở đây để là 0.0 hoặc rất thấp, 
    # vì ta đã lọc thủ công bằng dynamic threshold ở bước 2 rồi.
    selected_indices = tf.image.non_max_suppression(
        boxes=tf_boxes,
        scores=tf_scores,
        max_output_size=20, 
        iou_threshold=iou_thres,
        score_threshold=0.0 # Quan trọng: Đừng lọc lại ở đây nữa
    )
    
    selected_indices = selected_indices.numpy()
    
    # --- 5. MAP VỀ ẢNH GỐC ---
    scale, dx, dy = scale_params
    final_detections = []
    
    for i in selected_indices:
        info = decoded_boxes_info[i]
        
        x_infer = info["cx"] * INFER_SIZE
        y_infer = info["cy"] * INFER_SIZE
        w_infer = info["w"] * INFER_SIZE
        h_infer = info["h"] * INFER_SIZE
        
        x1 = int((x_infer - w_infer/2 - dx) / scale)
        y1 = int((y_infer - h_infer/2 - dy) / scale)
        x2 = int((x_infer + w_infer/2 - dx) / scale)
        y2 = int((y_infer + h_infer/2 - dy) / scale)
        
        x1, y1 = max(0, x1), max(0, y1)
        
        label_text = CLASS_NAMES.get(info["class_id"], "UNKNOWN")
        
        final_detections.append({
            "box": [x1, y1, x2, y2],
            "confidence": round(info["score"], 2),
            "label": label_text
        })
        
    return final_detections

# ----------------------------
# 5. WORKER LOOP
# ----------------------------
def worker_loop():
    global last_detection_result
    print("✅ AI Worker Started with TF-NMS")

    while worker_running:
        frame_tuple = None
        with state_lock:
            if frame_queue:
                frame_tuple = frame_queue.pop()
                frame_queue.clear() # Lấy frame mới nhất, bỏ qua cũ

        if not frame_tuple:
            time.sleep(0.01)
            continue

        frame, frame_ts, user_id = frame_tuple

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h0, w0 = frame.shape[:2]

            # Preprocess
            img_padded, scale, (dx, dy) = letterbox_resize(frame_rgb, target_size=INFER_SIZE)
            input_tensor = img_padded.astype(np.float32) / 255.0
            input_tensor = np.expand_dims(input_tensor, axis=0)

            # Inference
            preds = model.predict(input_tensor, verbose=0)

            # Post-process (with corrected NMS)
            detections = decode_predictions(preds, (h0, w0), (scale, dx, dy))

            # --- LOGIC ALERT TRIGGER ---
            frame_label = "NONE"
            has_fire = any(d['label'] == "FIRE" for d in detections)
            has_fall = any(d['label'] == "FALL" for d in detections)

            if has_fire: frame_label = "FIRE"
            elif has_fall: frame_label = "FALL"

            confirmed_alert = None
            state_key = f"user_{user_id}"
            
            with state_lock:
                if state_key not in states:
                    states[state_key] = {"buffer": deque(maxlen=WINDOW_SAMPLES), "last_trigger_at": 0}
                
                st = states[state_key]
                st["buffer"].append(frame_label)
                
                if LOG_BUFFER_STATE:
                    print(f"User {user_id} Buf: {[x[0] for x in st['buffer']]} -> Detect: {len(detections)}")

                if len(st["buffer"]) == WINDOW_SAMPLES:
                    if st["buffer"].count("FIRE") >= REQUIRED_MATCH:
                        confirmed_alert = "FIRE"
                    elif st["buffer"].count("FALL") >= REQUIRED_MATCH:
                        confirmed_alert = "FALL"
                        
                    if confirmed_alert:
                        now = int(time.time() * 1000)
                        if now - st["last_trigger_at"] > RATE_LIMIT_MS:
                            st["last_trigger_at"] = now
                            print(f"🚨 ALERT TRIGGERED: {confirmed_alert}")
                            st["buffer"].clear()
                        else:
                            confirmed_alert = None # Cooldown

            with state_lock:
                last_detection_result = {
                    "frame_width": w0,
                    "frame_height": h0,
                    "detections": detections,
                    "alert_trigger": confirmed_alert
                }

        except Exception as e:
            print(f"Worker Error: {e}")
            traceback.print_exc()

# Start Worker
t = threading.Thread(target=worker_loop, daemon=True)
t.start()

# ----------------------------
# 6. ROUTES
# ----------------------------
@app.route("/api/detect_frame", methods=["POST"])
def detect_frame_route():
    try:
        data = request.get_json(force=True, silent=True)
        if not data or "image" not in data:
            return jsonify({"error": "No image"}), 400

        user_id = data.get("userID", 1)
        raw_img = data["image"].split(",")[1] if "," in data["image"] else data["image"]
        np_arr = np.frombuffer(base64.b64decode(raw_img), dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None: return jsonify({"error": "Decode fail"}), 400

        with state_lock:
            frame_queue.append((frame, time.time(), user_id))
            # Trả về kết quả của frame trước đó (gần nhất) để UI mượt hơn
            resp = dict(last_detection_result)

        return jsonify(resp)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)