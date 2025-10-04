# app.py
from flask import Flask, request, jsonify
from deepface import DeepFace
from PIL import Image
import numpy as np
import base64
from io import BytesIO
import cv2
import os
import re
import logging

try:
    from flask_cors import CORS
except Exception:
    CORS = None

# -------------------- Flask setup --------------------
app = Flask(__name__)

# อนุญาต CORS (ถ้าอยู่หลังบ้านคนละโดเมนกับแอป)
if CORS:
    CORS(app, resources={r"/*": {"origins": "*"}})

# เปิด log ให้เห็นสาเหตุ 400 ชัดเจน
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("foodmood")

# -------------------- Config --------------------
DETECTOR_BACKEND = os.getenv("DETECTOR_BACKEND", "opencv").lower()  # opencv|retinaface|mtcnn
MIN_FACE_CONF = float(os.getenv("MIN_FACE_CONF", 0.80))
MIN_TOP_PROB  = float(os.getenv("MIN_TOP_PROB", 0.60))

# จำกัดขนาด payload (ป้องกันรูปมหึมา): 10MB
# หมายเหตุ: ถ้ารูปใหญ่กว่านี้ จะได้ 413 จาก werkzeug (ดีต่อเสถียรภาพ)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# -------------------- Helpers --------------------
_dataurl_re = re.compile(r"^data:image\/[a-zA-Z0-9.+-]+;base64,")

def _strip_data_url(s: str) -> str:
    """รองรับทั้ง data URL และ base64 ดิบ"""
    if _dataurl_re.match(s):
        return s.split(",", 1)[1]
    return s

def load_image(image_str: str, max_side: int = 1080) -> np.ndarray:
    """
    แปลง base64 -> RGB ndarray และลดขนาดให้ด้านยาวสุดไม่เกิน max_side
    พยายามใช้ Pillow ก่อน, ถ้าไม่สำเร็จ fallback เป็น OpenCV
    """
    if not image_str or not isinstance(image_str, str):
        raise ValueError("image is empty or invalid type")

    # แก้เคสที่ base64 ผ่าน URL-encoding (+ กลายเป็น space)
    image_str = _strip_data_url(image_str).replace(" ", "+").strip()

    try:
        raw = base64.b64decode(image_str, validate=True)
    except Exception as e:
        raise ValueError(f"invalid base64: {e}")

    # Pillow route
    try:
        img = Image.open(BytesIO(raw))
        img = img.convert("RGB")
        w, h = img.size
        if w == 0 or h == 0:
            raise ValueError("invalid image size")
        scale = min(1.0, float(max_side) / float(max(w, h)))
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        return np.array(img)
    except Exception as pil_err:
        # Fallback: OpenCV
        try:
            nparr = np.frombuffer(raw, np.uint8)
            bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("cv2.imdecode failed")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            if w == 0 or h == 0:
                raise ValueError("invalid image size")
            scale = min(1.0, float(max_side) / float(max(w, h)))
            if scale < 1.0:
                rgb = cv2.resize(
                    rgb, (max(1, int(w * scale)), max(1, int(h * scale)))
                )
            return rgb
        except Exception as cv_err:
            raise ValueError(f"cannot decode image (Pillow:{pil_err}) (OpenCV:{cv_err})")

def to_float(x):
    try:
        return float(x)
    except Exception:
        return x

def probs_to_serializable(probs):
    if not isinstance(probs, dict):
        return {}
    return {str(k): to_float(v) for k, v in probs.items()}

# -------------------- Routes --------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True}), 200

@app.route("/warmup", methods=["GET"])
def warmup():
    """
    อุ่น DeepFace/TF ครั้งแรก เพื่อลด cold start
    """
    try:
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        _ = DeepFace.analyze(
            img_path=dummy,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend=DETECTOR_BACKEND,
        )
        return jsonify({"warmed": True}), 200
    except Exception as e:
        logger.exception("Warmup failed")
        return jsonify({"warmed": False, "error": str(e)}), 500

@app.route("/detect_emotion", methods=["POST"])
def detect_emotion():
    """
    รับ JSON: {"image": "<base64 หรือ data url>"}
    คืนค่า: mood, probs, face_confidence, top_prob
    - ถ้าไม่มั่นใจพอ (หน้าไม่ชัด/ความเชื่อมั่นต่ำ) จะคืน 422 พร้อมข้อความไทย
    """
    try:
        try:
            data = request.get_json(force=True, silent=False)
        except Exception as e:
            logger.exception("Invalid JSON body")
            return jsonify({"error": f"invalid json: {e}"}), 400

        if not data or "image" not in data:
            return jsonify({"error": "Missing 'image' field"}), 400

        try:
            img = load_image(data["image"])
        except Exception as e:
            logger.info(f"load_image error: {e}")
            return jsonify({"error": f"image decode error: {e}"}), 400

        try:
            res = DeepFace.analyze(
                img_path=img,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend=DETECTOR_BACKEND,
            )
        except Exception as e:
            # DeepFace / TF โยนข้อผิดพลาด (เช่น รูปผิดรูปแบบ)
            logger.exception("DeepFace.analyze failed")
            return jsonify({"error": f"analyze failed: {e}"}), 400

        # DeepFace บางเวอร์ชันคืน list, บางเวอร์ชันคืน dict
        item = res[0] if isinstance(res, list) and len(res) > 0 else res
        probs = item.get("emotion", {}) or {}
        dominant = item.get("dominant_emotion")
        face_conf = to_float(item.get("face_confidence", 0.0))
        top_prob = to_float(max(probs.values())) if probs else 0.0

        detected = (float(face_conf or 0.0) >= MIN_FACE_CONF) and (float(top_prob or 0.0) >= MIN_TOP_PROB)

        if not detected:
            return jsonify({
                "detected": False,
                "error": "ไม่พบใบหน้าชัดเจน กรุณาหันหน้าให้กล้องหรือถ่ายใหม่",
                "face_confidence": face_conf,
                "top_prob": top_prob
            }), 422

        return jsonify({
            "detected": True,
            "mood": dominant,  # angry, disgust, fear, happy, sad, surprise, neutral
            "probs": probs_to_serializable(probs),
            "face_confidence": face_conf,
            "top_prob": top_prob
        }), 200

    except Exception as e:
        # กันเผื่อข้อผิดพลาดอื่น ๆ ที่หลุดมา
        logger.exception("Unhandled error in /detect_emotion")
        return jsonify({"error": str(e)}), 400

# -------------------- Local run --------------------
if __name__ == "__main__":
    # รันโลคัล
    app.run(host="0.0.0.0", port=5000, debug=True)
