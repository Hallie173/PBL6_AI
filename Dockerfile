FROM python:3.9-slim

# 1. Cài thư viện hệ thống tối thiểu
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Tạo requirements.txt trực tiếp (nếu bạn lười tạo file riêng)
# Lưu ý: opencv-python-headless rất quan trọng
RUN echo "Flask\nflask-cors\ngunicorn\nnumpy\ntensorflow\nopencv-python-headless" > requirements.txt

# 3. Cài thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy code và Model
COPY . .

# 5. Quan trọng: Tạo thư mục Data nếu chưa có (đề phòng lỗi)
# Bạn phải chắc chắn file 'Data/best_yolo_resnet.keras' nằm cùng thư mục với Dockerfile khi build
RUN if [ ! -f "Data/best_yolo_resnet.keras" ]; then echo "WARNING: Model not found during build!"; fi

# 6. Chạy với tham số tối ưu cho code dùng Global Variables
# Timeout tăng lên 120s phòng khi model load lâu
CMD ["gunicorn", "--workers", "1", "--threads", "4", "--timeout", "120", "--bind", "0.0.0.0:5000", "app1:app"]