from flask import Flask, render_template, request
from deepface import DeepFace
import cv2
import os
import uuid
from PIL import ImageFont, ImageDraw, Image
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = "static"
FONT_PATH = "fonts/NotoSansCJKtc-Regular.otf"  # 中文字體路徑
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        img_file = request.files.get("image")
        if img_file:
            img_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.jpg")
            img_file.save(img_path)

            try:
                results = DeepFace.analyze(img_path=img_path, actions=["emotion"], enforce_detection=True)
                img = cv2.imread(img_path)

                def draw_face(face_data):
                    region = face_data["region"]
                    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
                    emotion = face_data["dominant_emotion"]
                    score = face_data["emotion"][emotion]

                    # 英文 → 中文情緒對照
                    emotion_translations = {
                        "happy": "開心",
                        "sad": "難過",
                        "angry": "生氣",
                        "fear": "害怕",
                        "fearful": "害怕",
                        "disgust": "厭惡",
                        "surprise": "驚訝",
                        "surprised": "驚訝",
                        "neutral": "中立"
                    }
                    emotion_zh = emotion_translations.get(emotion, emotion)
                    label = f"{emotion_zh} ({score:.1f}%)"

                    # 畫框
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # 用 PIL 畫中文字
                    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(img_pil)

                    # 載入中文字體
                    font = ImageFont.truetype(FONT_PATH, 24)
                    draw.text((x, y - 30), label, font=font, fill=(255, 255, 255))

                    # 回存成 OpenCV 格式
                    img_result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                    return img_result

                if isinstance(results, list):
                    for face in results:
                        img = draw_face(face)
                else:
                    img = draw_face(results)

                result_filename = f"{uuid.uuid4().hex}_result.jpg"
                result_path = os.path.join(UPLOAD_FOLDER, result_filename)
                cv2.imwrite(result_path, img)

                return render_template("index.html", result_img=result_filename)

            except Exception as e:
                return render_template("index.html", error=str(e))

    return render_template("index.html")

if __name__ == "__main__":
    # 取得 Render 提供的端口，如果沒有就用 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
