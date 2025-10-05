FROM python:3.10-bullseye

WORKDIR /app

# ---------- System dependencies ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# ---------- Environment variables ----------
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=180 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    DETECTOR_BACKEND=opencv \
    MIN_FACE_CONF=0.8 \
    MIN_TOP_PROB=0.6 \
    DEEPFACE_HOME=/app/.deepface

# ---------- Install Python dependencies ----------
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --prefer-binary -r requirements.txt

# ---------- Copy app ----------
COPY . .

# ---------- Create cache folder ----------
RUN mkdir -p /app/.deepface/weights

# ---------- Gunicorn startup ----------
# เพิ่ม timeout = 300 เพื่อรอ DeepFace โหลดโมเดลตอนแรก
# Render จะส่งตัวแปร PORT มาอัตโนมัติ
CMD ["bash", "-lc", "gunicorn -w 1 -t 300 -b 0.0.0.0:${PORT:-8080} app:app"]
