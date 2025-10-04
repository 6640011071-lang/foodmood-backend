FROM python:3.10-bullseye

WORKDIR /app

# System dependencies ที่จำเป็นต่อ OpenCV / PIL
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# ปรับพฤติกรรม pip ให้เสถียรขึ้น
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=180 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    DETECTOR_BACKEND=opencv \
    MIN_FACE_CONF=0.8 \
    MIN_TOP_PROB=0.6 \
    DEEPFACE_HOME=/app/.deepface

RUN python -m pip install --upgrade pip

# ติดตั้งไลบรารี (เวอร์ชันจะอิงจาก requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# คัดลอกโค้ดแอป
COPY . .

# โฟลเดอร์ cache ให้ DeepFace
RUN mkdir -p /app/.deepface/weights

# Render จะกำหนด $PORT มาเอง
# ลด worker เป็น 1 เพื่อลด RAM ตอนบูต
CMD ["bash", "-lc", "gunicorn -w 1 -t 180 -b 0.0.0.0:${PORT:-8080} app:app"]
