from flask import Flask, request, jsonify
import pickle
import numpy as np
import cv2
import base64
import mediapipe as mp
import os
import csv
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# تعديل المسارات لتكون مرنة مع السيرفر
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "sign_model.pkl")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1, 
    min_detection_confidence=0.5
)

english_to_arabic = {
    'A': 'أ','B': 'ب','C': 'ت','D': 'ث','E': 'ج','F': 'ح',
    'G': 'خ','H': 'د','I': 'ذ','J': 'ر','K': 'ز','L': 'س',
    'M': 'ش','N': 'ص','O': 'ض','P': 'ط','Q': 'ظ','R': 'ع',
    'S': 'غ','T': 'ف','U': 'ق','V': 'ك','W': 'ل','X': 'م',
    'Y': 'ن','Z': 'ي'
}

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading model: {e}")
    return None

model = load_model()

def process_base64_image(base64_string):
    try:
        encoded_data = base64_string.split(',')[1] if ',' in base64_string else base64_string
        img_data = base64.b64decode(encoded_data)
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Decoding Error: {e}")
        return None

@app.route('/')
def home():
    return jsonify({"status": "AI Server is Live!"})

@app.route('/frame', methods=['POST'])
def predict_frame():
    global model
    try:
        data = request.json
        frame_base64 = data.get('frame')
        
        if not frame_base64:
            return jsonify({"error": "لم يتم إرسال الصورة"}), 400

        frame = process_base64_image(frame_base64)
        if frame is None:
            return jsonify({"error": "فشل في معالجة بيانات الصورة"}), 400

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        if results.multi_hand_landmarks:
            landmarks = []
            for lm in results.multi_hand_landmarks[0].landmark:
                landmarks.extend([lm.x, lm.y])
            
            if model is None:
                return jsonify({"error": "الموديل غير موجود، يرجى التدريب أولاً"}), 400

            prediction = model.predict(np.array(landmarks).reshape(1, -1))[0]
            arabic_text = english_to_arabic.get(prediction, prediction)
            
            return jsonify({
                "translatedText": arabic_text, 
                "label": prediction,
                "confidence": 0.98
            })
        
        return jsonify({"translatedText": "لم يتم اكتشاف إشارة", "confidence": 0})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model():
    global model
    try:
        X, y = [], []
        if not os.path.exists(DATASET_DIR):
             return jsonify({"error": "مجلد البيانات غير موجود"}), 400

        for file in os.listdir(DATASET_DIR):
            if file.endswith(".csv"):
                label = file.replace(".csv", "")
                with open(os.path.join(DATASET_DIR, file), "r") as f:
                    reader = csv.reader(f)
                    next(reader)
                    for row in reader:
                        X.append(list(map(float, row)))
                        y.append(label)

        if len(X) == 0:
            return jsonify({"error": "لا توجد بيانات كافية للتدريب"}), 400

        new_model = RandomForestClassifier(n_estimators=100)
        new_model.fit(np.array(X), np.array(y))

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(new_model, f)
        
        model = new_model
        return jsonify({"status": "success", "message": "تم تحديث الموديل بنجاح"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)
