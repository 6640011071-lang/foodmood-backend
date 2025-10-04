# app.py
from flask import Flask, request, jsonify
from deepface import DeepFace
from PIL import Image
import numpy as np
import base64
from io import BytesIO
import cv2
import os

try:
    from flask_cors import CORS
except Exception:
    CORS = None

app = Flask(__name__)
if CORS:
    CORS(app)

# ใช้ detector_backend ที่เบาและเร็ว
DETECTOR_BACKEND = os.getenv("DETECTOR_BACKEND", "opencv")  # opencv|retinaface|mtcnn

def load_image(image_str: str, max_side: int = 1080):
    # รองรับ data URL และ base64 ตรง
    if isinstance(image_str, str) and image_str.startswith("data:"):
        image_str = image_str.split(",", 1)[1]
    raw = base64.b64decode(image_str)

    # พยายามอ่านด้วย Pillow ก่อน
    try:
        img = Image.open(BytesIO(raw)).convert("RGB")
        w, h = img.size
        scale = min(1.0, float(max_side) / float(max(w, h)))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)))
        return np.array(img)
    except Exception:
        # fallback OpenCV
        nparr = np.frombuffer(raw, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Invalid image data")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = min(1.0, float(max_side) / float(max(w, h)))
        if scale < 1.0:
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))
        return rgb

def to_float(x):
    try:
        return float(x)
    except Exception:
        return x

def probs_to_serializable(probs):
    if not isinstance(probs, dict):
        return {}
    return {str(k): to_float(v) for k, v in probs.items()}

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True}), 200

@app.route("/warmup", methods=["GET"])
def warmup():
    # อุ่นเครื่องแบบไม่พึ่ง EMOTION_MODEL (ให้ DeepFace โหลดเอง)
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
        return jsonify({"warmed": False, "error": str(e)}), 500

@app.route("/detect_emotion", methods=["POST"])
def detect_emotion():
    try:
        data = request.get_json(force=True)
        if not data or "image" not in data:
            return jsonify({"error": "Missing 'image' field"}), 400

        img = load_image(data["image"])

        # เรียก DeepFace แบบตรงๆ ให้มันจัดการโหลดโมเดล/แคชเอง
        res = DeepFace.analyze(
            img_path=img,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend=DETECTOR_BACKEND,
        )

        item = res[0] if isinstance(res, list) and len(res) > 0 else res
        probs = item.get("emotion", {}) or {}
        dominant = item.get("dominant_emotion")
        face_conf = to_float(item.get("face_confidence", 0.0))
        top_prob = to_float(max(probs.values())) if probs else 0.0

        MIN_FACE_CONF = float(os.getenv("MIN_FACE_CONF", 0.80))
        MIN_TOP_PROB  = float(os.getenv("MIN_TOP_PROB", 0.60))
        detected = (face_conf >= MIN_FACE_CONF) and (top_prob >= MIN_TOP_PROB)

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
        print("Error analyzing image:", e)
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # รันโลคัล
    app.run(host="0.0.0.0", port=5000, debug=True)
