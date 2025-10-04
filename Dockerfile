FROM python:3.10-slim

WORKDIR /app

# ===== System deps (แค่ที่จำเป็น) =====
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# ===== pip & network tuning =====
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    DETECTOR_BACKEND=opencv \
    MIN_FACE_CONF=0.8 \
    MIN_TOP_PROB=0.6 \
    DEEPFACE_HOME=/app/.deepface

RUN python -m pip install --upgrade pip

# ===== install pinned deps =====
# ใช้ requirements.txt ของเรา แต่กันกรณี Render เคยแคชเวอร์ชันแปลกๆ
COPY requirements.txt .
RUN pip install --force-reinstall --upgrade -r requirements.txt && \
    # บังคับเวอร์ชันสำคัญซ้ำอีกรอบ เผื่อ dependency ลาก TF เพี้ยน
    pip install --force-reinstall --upgrade \
        numpy==1.24.3 \
        pillow==10.2.0 \
        opencv-python-headless==4.9.0.80 \
        pandas==1.5.3 \
        tensorflow==2.12.1 \
        deepface==0.0.93 \
        retinaface==0.0.14 \
        mtcnn==0.1.0 \
        tf-keras==2.16.0

# ===== copy source =====
COPY . .

# สร้างโฟลเดอร์ cache โมเดล (สิทธิ์เขียน)
RUN mkdir -p /app/.deepface/weights

# ===== run gunicorn (Render จะส่ง $PORT มา) =====
CMD ["bash", "-lc", "gunicorn -w 2 -t 120 -b 0.0.0.0:${PORT:-8080} app:app"]
