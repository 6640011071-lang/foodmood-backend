# ---- base image (Python 3.10) ----
FROM python:3.10-slim

WORKDIR /app

# ระบบพื้นฐาน
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# จัดการ pip
RUN python -m pip install --upgrade pip

# ติดตั้ง requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกซอร์ส
COPY . .

# ให้ Tensorflow ใช้ CPU (ไม่มี CUDA) และบังคับ detector เบาๆ
ENV TF_CPP_MIN_LOG_LEVEL=2 \
    DETECTOR_BACKEND=opencv \
    MIN_FACE_CONF=0.8 \
    MIN_TOP_PROB=0.6

# Render/Railway จะส่ง $PORT มา -> bind 0.0.0.0:$PORT
CMD ["bash", "-lc", "gunicorn -w 2 -t 120 -b 0.0.0.0:${PORT:-8080} app:app"]
