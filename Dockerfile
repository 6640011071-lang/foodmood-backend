FROM python:3.10-bullseye

WORKDIR /app

# System deps ที่ต้องใช้กับ OpenCV / Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# ตั้งค่า ENV ให้เสถียรขึ้น และชี้ DEEPFACE_HOME ให้ถูก (กัน .deepface ซ้อน .deepface)
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=180 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    OMP_NUM_THREADS=1 \
    TF_NUM_INTRAOP_THREADS=1 \
    TF_NUM_INTEROP_THREADS=1 \
    DETECTOR_BACKEND=opencv \
    MIN_FACE_CONF=0.8 \
    MIN_TOP_PROB=0.6 \
    DEEPFACE_HOME=/app

# อัปเดต pip
RUN python -m pip install --upgrade pip

# ติดตั้งไลบรารีตาม requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# คัดลอกโค้ดเข้าอิมเมจ
COPY . .

# โฟลเดอร์ cache สำหรับ DeepFace (จะได้ /app/.deepface/weights พอดี)
RUN mkdir -p /app/.deepface/weights

# (ทางเลือก แนะนำ) อุ่นโมเดลตอน Build เพื่อตัด cold start ครั้งแรกบน Render
# ถ้าบิลด์นานหรือเน็ตช้าแล้ว fail ให้คอมเมนต์บล็อกนี้ไว้ได้
RUN python - << 'PY'
from deepface import DeepFace
import numpy as np
print("Warming up DeepFace emotion model...")
_ = DeepFace.analyze(
    img_path=np.zeros((224,224,3), dtype='uint8'),
    actions=["emotion"],
    enforce_detection=False,
    detector_backend="opencv",
)
print("Warmup finished.")
PY

# รันด้วย gunicorn; Render จะส่ง $PORT มาให้ ใช้ worker เดียวลด RAM และเพิ่ม timeout 300s
CMD ["bash", "-lc", "gunicorn -w 1 -t 300 -b 0.0.0.0:${PORT:-8080} app:app"]
