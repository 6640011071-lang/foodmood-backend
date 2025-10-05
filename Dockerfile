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
    OMP_NUM_THREADS=1 \
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

# ---------- PREWARM (download model weights at build time) ----------
# ทำให้ DeepFace โหลดโมเดลอารมณ์ไว้ก่อน เพื่อตัดเวลาหน่วงตอนรันจริง
RUN python - <<'PY'
from deepface import DeepFace
import numpy as np, os
dummy = np.zeros((224,224,3), dtype='uint8')
DeepFace.analyze(
    img_path=dummy,
    actions=['emotion'],
    enforce_detection=False,
    detector_backend=os.getenv('DETECTOR_BACKEND','opencv')
)
print(">>> DeepFace emotion model prewarmed & cached")
PY

# ---------- Gunicorn startup ----------
# timeout ยาวหน่อย เผื่อช่วงแรก TF ใช้เวลา
CMD ["bash", "-lc", "gunicorn -w 1 -t 300 -b 0.0.0.0:${PORT:-8080} app:app"]
